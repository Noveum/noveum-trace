"""End-to-end tests for Nova Guard usage ingest against the LIVE backend.

These prove the whole loop the mocked unit tests can only approximate: the SDK
builds an event, POSTs it to the real ``/policies/usage`` endpoint, and the real
server accepts it. Both outcomes are covered — a call that ran (ALLOWED, with
cost) and a call the Guard stopped (BLOCKED, with blockedBy/policyId).

Requires real credentials in .env:
    NOVEUM_API_KEY, NOVEUM_PROJECT   — always
    OPENAI_API_KEY                   — only for the ALLOWED test (real LLM call)

Run explicitly:

    pytest tests/integration/guard/test_usage_ingest_real.py -m integration -v -s

SIDE EFFECTS — these write to your real project:
  * The ALLOWED test makes one small real OpenAI call and reports its cost, so
    project spend increases by a fraction of a cent.
  * BLOCKED events are not metered, but they DO write the audit log and trigger
    the owner "limit hit" email (backend-throttled to one per org+project+limit
    type per hour).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional

import httpx
import pytest

# Load .env so locally-exported keys are picked up automatically.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv optional
    pass

import noveum_trace
from noveum_trace.core.config import DEFAULT_ENDPOINT
from noveum_trace.guard.api_client_http import HttpGuardAPIClient
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.policies.cost_cap import CostCapPolicy
from noveum_trace.guard.policies.rate_limit import RateLimitPolicy
from noveum_trace.guard.poller import PolicyPoller
from noveum_trace.guard.transport.adapters.openai_adapter import OpenAIAdapter
from noveum_trace.guard.types import EnforcementMode, ParsedRequest, PolicyContext

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:  # provider SDK optional
    openai = None  # type: ignore[assignment]
    OPENAI_AVAILABLE = False

NOVEUM_API_KEY = os.environ.get("NOVEUM_API_KEY")
NOVEUM_PROJECT = os.environ.get("NOVEUM_PROJECT")
NOVEUM_ENDPOINT = os.environ.get("NOVEUM_ENDPOINT", DEFAULT_ENDPOINT)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("NOVEUM_GUARD_TEST_MODEL", "gpt-4o-mini")

# Upper bound on real calls used to trip a rate limit, so a misconfigured
# window can never turn this test into an unbounded spend loop.
_MAX_RATE_ATTEMPTS = 12


def _is_valid_key(key: Optional[str]) -> bool:
    placeholders = {"", "your-api-key-here", "test-key", "sk-test", "sk-fake", "sk-..."}
    return bool(key) and key not in placeholders and len(key) > 10


def _guard_configured() -> bool:
    return _is_valid_key(NOVEUM_API_KEY) and bool(NOVEUM_PROJECT)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _guard_configured(),
        reason="NOVEUM_API_KEY / NOVEUM_PROJECT not set — see .env.example",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def usage_posts(monkeypatch):
    """Record every real POST to /policies/usage without changing behaviour.

    Wraps httpx.Client.post so the request still goes out over the wire; we only
    observe it. That keeps the test genuinely end-to-end while letting it assert
    on the status code and body the live backend returned.
    """
    recorded: list[dict[str, Any]] = []
    original = httpx.Client.post

    def spy(self, url, *args, **kwargs):
        resp = original(self, url, *args, **kwargs)
        if "/policies/usage" in str(url):
            try:
                body = resp.json()
            except Exception:
                body = {"_raw": resp.text[:300]}
            recorded.append(
                {
                    "url": str(url),
                    "events": kwargs.get("json"),
                    "status": resp.status_code,
                    "body": body,
                }
            )
        return resp

    monkeypatch.setattr(httpx.Client, "post", spy)
    return recorded


def _ctx() -> PolicyContext:
    return PolicyContext(
        project_id=NOVEUM_PROJECT or "",
        organization_id=None,
        environment="usage-ingest-e2e",
        trace_id=None,
        span_id=None,
        call_id=str(uuid.uuid4()),
    )


def _api() -> HttpGuardAPIClient:
    # flush_interval is long so nothing fires until the test calls close(),
    # which drains synchronously and makes the assertions deterministic.
    return HttpGuardAPIClient(
        api_key=NOVEUM_API_KEY or "",
        base_url=NOVEUM_ENDPOINT,
        flush_interval=3600,
    )


def _stack(api: HttpGuardAPIClient, max_usd: float, policy_id: Optional[str] = None):
    engine = PolicyEngine(api_client=api)
    engine.attach(
        CostCapPolicy(
            max_usd=max_usd,
            mode=EnforcementMode.strict,  # degrades to shared: HTTP has no reserve
            project_id=NOVEUM_PROJECT,
            policy_id=policy_id,
        )
    )
    engine.poll_all()  # real GET /policies/state — seeds spend from the backend
    return engine, _ctx()


def _rate_stack(
    api: HttpGuardAPIClient, max_requests: int, policy_id: Optional[str] = None
):
    engine = PolicyEngine(api_client=api)
    engine.attach(
        RateLimitPolicy(
            windows=[{"period": "1m", "maxRequests": max_requests}],
            project_id=NOVEUM_PROJECT,
            policy_id=policy_id,
        )
    )
    engine.poll_all()  # real GET /policies/state — seeds live rate counters
    return engine, _ctx()


def _probe_request(model: str, max_tokens: int) -> ParsedRequest:
    """A parsed request used only to price a call before making it."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one word: done."}],
            "max_tokens": max_tokens,
        }
    ).encode()
    request = httpx.Request(
        "POST", "https://api.openai.com/v1/chat/completions", content=payload
    )
    return OpenAIAdapter().parse_request(request)


def _events(recorded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for post in recorded for e in (post["events"] or [])]


def _dump(label: str, recorded: list[dict[str, Any]]) -> None:
    """Print the exact wire payload and the backend's reply."""
    for post in recorded:
        print(f"\n[{label}] POST {post['url']}")
        print(f"[{label}] -> HTTP {post['status']} {post['body']}")
        for event in post["events"] or []:
            print(f"[{label}]    {json.dumps(event, sort_keys=True)}")


# ---------------------------------------------------------------------------
# Policy sync — the real policyId reaches the SDK
# ---------------------------------------------------------------------------


class TestPolicySyncCarriesPolicyId:
    def test_effective_policies_include_policy_id(self):
        """A BLOCKED event can only name its policy if the id survives mapping."""
        api = _api()
        policies = api.fetch_remote_policies(NOVEUM_PROJECT or "")

        assert policies, "no enabled policies on this project — configure one first"
        for p in policies:
            assert p["type"] in ("cost_cap", "rate_limit")
            assert p.get("policy_id"), f"policy_id missing from mapped policy: {p}"
        print(f"\n[effective] {policies}")


# ---------------------------------------------------------------------------
# BLOCKED — the call never runs, the event still reaches the backend
# ---------------------------------------------------------------------------


class TestBlockedEventIngest:
    def test_blocked_call_posts_blocked_event(self, usage_posts):
        """Exhausted cap → no provider call, one BLOCKED event accepted (202).

        The cap is derived from the project's REAL polled spend, so the block is
        triggered by live state rather than a hand-set counter.
        """
        api = _api()
        real_spend = api.get_state(NOVEUM_PROJECT or "")["spend"]
        # Half of real spend is guaranteed to be already exceeded; when spend is
        # 0 a cap of 0 still blocks, since any estimate is > 0.
        cap = real_spend * 0.5
        engine, ctx = _stack(api, max_usd=cap, policy_id="e2e-blocked-probe")

        # A deliberately fake provider key: if the block leaks, OpenAI rejects it
        # and the test fails loudly instead of silently spending money.
        client = openai.OpenAI(
            api_key="sk-must-never-be-used",
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )

        with pytest.raises(openai.PermissionDeniedError):
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Reply with one word: done."}],
                max_tokens=16,
            )

        api.close()  # drains the queue → real POST /policies/usage

        assert usage_posts, "no POST reached /policies/usage"
        post = usage_posts[0]
        _dump("cost-cap", usage_posts)

        assert post["status"] == 202, f"backend rejected the event: {post}"
        assert post["url"].endswith(f"/v1/projects/{NOVEUM_PROJECT}/policies/usage")

        blocked = [e for e in _events(usage_posts) if e.get("outcome") == "BLOCKED"]
        assert len(blocked) == 1
        event = blocked[0]
        assert event["blockedBy"] == "COST_CAP"
        assert event["policyId"] == "e2e-blocked-probe"
        assert event["costUsd"] == 0.0  # the call never ran
        assert event["model"] == MODEL
        assert event["eventId"]
        assert "Cost cap" in event["reason"]

        # The server counts the blocks it took in this push.
        assert post["body"].get("success") is True
        assert post["body"].get("blocked") == 1

    def test_blocked_call_is_not_metered(self, usage_posts):
        """Spend must not move: a blocked call never reached the provider."""
        api = _api()
        before = api.get_state(NOVEUM_PROJECT or "")["spend"]

        engine, ctx = _stack(api, max_usd=before * 0.5, policy_id="e2e-not-metered")
        client = openai.OpenAI(
            api_key="sk-must-never-be-used",
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )
        with pytest.raises(openai.PermissionDeniedError):
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
            )
        api.close()

        after = api.get_state(NOVEUM_PROJECT or "")["spend"]
        print(f"\n[not-metered] spend before={before} after={after}")
        assert after == pytest.approx(before, abs=1e-9)


# ---------------------------------------------------------------------------
# ALLOWED — a real LLM call, reported with its real cost
# ---------------------------------------------------------------------------


class TestAllowedEventIngest:
    @pytest.mark.skipif(
        not (OPENAI_AVAILABLE and _is_valid_key(OPENAI_API_KEY)),
        reason="OPENAI_API_KEY not set/valid or openai not installed",
    )
    def test_successful_call_posts_allowed_event(self, usage_posts):
        api = _api()
        engine, ctx = _stack(api, max_usd=100.0)  # generous: must not block

        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with one word: done."}],
            max_tokens=16,
        )
        assert resp.choices[0].message.content  # a real answer came back

        api.close()

        assert usage_posts, "no POST reached /policies/usage"
        post = usage_posts[0]
        _dump("cost-allowed", usage_posts)

        assert post["status"] == 202, f"backend rejected the event: {post}"

        allowed = [e for e in _events(usage_posts) if e.get("outcome") is None]
        assert len(allowed) == 1
        event = allowed[0]
        # costUsd is required on the allowed path — omitting it is a 400.
        assert event["costUsd"] > 0.0
        assert event["inputTokens"] > 0
        assert event["outputTokens"] > 0
        # The allowed path reports the model the provider RESOLVED to
        # (e.g. "gpt-4o-mini-2024-07-18"), not the alias that was requested.
        assert event["model"].startswith(MODEL)
        assert event["requestCount"] == 1
        assert event["eventId"]

        assert post["body"].get("success") is True
        assert post["body"].get("accepted") == 1
        assert post["body"].get("blocked") == 0


# ---------------------------------------------------------------------------
# RATE_LIMIT — the second limit type, tripped naturally by a real call
# ---------------------------------------------------------------------------


class TestRateLimitBlockedEventIngest:
    @pytest.mark.skipif(
        not (OPENAI_AVAILABLE and _is_valid_key(OPENAI_API_KEY)),
        reason="OPENAI_API_KEY not set/valid or openai not installed",
    )
    def test_rate_limit_trips_after_a_real_call(self, usage_posts):
        """One real call is admitted, the next is blocked by RATE_LIMIT.

        The threshold is set one above the project's live ``requests_1m`` count,
        so the first call fits and the second trips the limit — the limit is
        reached by an actual call, not by pre-loading a counter. Both events ride
        one POST, so the backend reply shows the ALLOWED and BLOCKED split.
        """
        api = _api()
        live_requests = api.get_state(NOVEUM_PROJECT or "")["rate"].get(
            "requests_1m", 0
        )
        engine, ctx = _rate_stack(
            api,
            max_requests=live_requests + 1,
            policy_id="e2e-rate-limit-probe",
        )

        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )
        messages = [{"role": "user", "content": "Reply with one word: done."}]

        # Call 1 — under the limit, actually reaches OpenAI.
        first = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=16
        )
        assert first.choices[0].message.content

        # Call 2 — the limit is now reached, so the Guard stops it.
        with pytest.raises(openai.PermissionDeniedError):
            client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=16
            )

        api.close()
        _dump("rate-limit", usage_posts)

        assert usage_posts, "no POST reached /policies/usage"
        for post in usage_posts:
            assert post["status"] == 202, f"backend rejected the event: {post}"

        events = _events(usage_posts)
        allowed = [e for e in events if e.get("outcome") is None]
        blocked = [e for e in events if e.get("outcome") == "BLOCKED"]

        assert len(allowed) == 1, f"expected one ALLOWED event, got {events}"
        assert allowed[0]["costUsd"] > 0.0
        assert allowed[0]["inputTokens"] > 0

        assert len(blocked) == 1, f"expected one BLOCKED event, got {events}"
        assert blocked[0]["blockedBy"] == "RATE_LIMIT"
        assert blocked[0]["policyId"] == "e2e-rate-limit-probe"
        assert blocked[0]["costUsd"] == 0.0
        assert "Rate limit reached" in blocked[0]["reason"]

        # Both outcomes in one push; only the block is counted as blocked.
        assert sum(p["body"].get("blocked", 0) for p in usage_posts) == 1
        assert sum(p["body"].get("accepted", 0) for p in usage_posts) == 2


# ---------------------------------------------------------------------------
# Real backend policies — blocks carry the project's actual policyId
# ---------------------------------------------------------------------------


def _real_stack(api: HttpGuardAPIClient, keep: type):
    """Load this project's real policies from the backend, keep one type.

    Only the policy under test is kept attached, so which limit trips is
    deterministic; both are still instantiated from the live definitions, with
    their real policyIds.
    """
    engine = PolicyEngine(api_client=api)
    poller = PolicyPoller(engine, project_id=NOVEUM_PROJECT, context=_ctx())
    poller.force_refresh()  # real GET /policies/effective → attach + poll

    kept = [p for p in engine.policies if isinstance(p, keep)]
    if not kept:
        pytest.skip(f"no enabled {keep.__name__} on project {NOVEUM_PROJECT!r}")
    for policy in engine.policies:
        if not isinstance(policy, keep):
            engine.detach(policy.name)
    return engine, _ctx(), kept[0]


class TestRealBackendPolicies:
    @pytest.mark.skipif(
        not (OPENAI_AVAILABLE and _is_valid_key(OPENAI_API_KEY)),
        reason="OPENAI_API_KEY not set/valid or openai not installed",
    )
    def test_real_cost_cap_blocks_with_its_own_policy_id(self, usage_posts):
        """The project's configured COST_CAP blocks a request whose worst-case
        cost would breach the remaining headroom."""
        api = _api()
        engine, ctx, policy = _real_stack(api, CostCapPolicy)
        real_policy_id = policy._policy_id
        assert real_policy_id, "backend policy arrived without a policyId"

        spend = api.get_state(NOVEUM_PROJECT or "")["spend"]
        headroom = policy.max_usd - spend
        print(
            f"\n[real-cost-cap] policy={policy.name!r} id={real_policy_id} "
            f"cap=${policy.max_usd} spend=${spend} headroom=${headroom:.6f}"
        )

        # Size the request so its worst-case reservation exceeds the headroom.
        max_tokens = 16000
        reserved = policy._estimate_reserved_usd(
            _probe_request(model=MODEL, max_tokens=max_tokens)
        )
        if reserved <= headroom:
            pytest.skip(
                f"cap has too much headroom to trip: reserved={reserved} "
                f"headroom={headroom}"
            )

        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )
        with pytest.raises(openai.PermissionDeniedError):
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Reply with one word: done."}],
                max_tokens=max_tokens,
            )

        api.close()
        _dump("real-cost-cap", usage_posts)

        blocked = [e for e in _events(usage_posts) if e.get("outcome") == "BLOCKED"]
        assert len(blocked) == 1
        assert blocked[0]["blockedBy"] == "COST_CAP"
        assert blocked[0]["policyId"] == real_policy_id
        assert all(p["status"] == 202 for p in usage_posts)

    @pytest.mark.skipif(
        not (OPENAI_AVAILABLE and _is_valid_key(OPENAI_API_KEY)),
        reason="OPENAI_API_KEY not set/valid or openai not installed",
    )
    def test_real_rate_limit_blocks_with_its_own_policy_id(self, usage_posts):
        """The project's configured RATE_LIMIT trips after enough real calls."""
        api = _api()
        engine, ctx, policy = _real_stack(api, RateLimitPolicy)
        real_policy_id = policy._policy_id
        assert real_policy_id, "backend policy arrived without a policyId"
        print(
            f"\n[real-rate-limit] policy={policy.name!r} id={real_policy_id} "
            f"windows={policy.windows}"
        )

        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )
        messages = [{"role": "user", "content": "Reply with one word: done."}]

        allowed_calls = 0
        for _ in range(_MAX_RATE_ATTEMPTS):
            try:
                client.chat.completions.create(
                    model=MODEL, messages=messages, max_tokens=16
                )
                allowed_calls += 1
            except openai.PermissionDeniedError:
                break
        else:
            pytest.fail(
                f"rate limit never tripped in {_MAX_RATE_ATTEMPTS} calls "
                f"(windows={policy.windows})"
            )

        api.close()
        _dump("real-rate-limit", usage_posts)
        print(f"[real-rate-limit] allowed {allowed_calls} call(s) before the block")

        blocked = [e for e in _events(usage_posts) if e.get("outcome") == "BLOCKED"]
        assert len(blocked) == 1
        assert blocked[0]["blockedBy"] == "RATE_LIMIT"
        assert blocked[0]["policyId"] == real_policy_id
        assert all(p["status"] == 202 for p in usage_posts)


# ---------------------------------------------------------------------------
# Malformed payloads — the 400 path the checklist calls out
# ---------------------------------------------------------------------------


class TestBackendRejectsMalformed:
    def test_blocked_without_blocked_by_is_rejected(self):
        """Confirms the 400 the SDK guards against is real, so the local
        validation in report_blocked() is protecting something."""
        url = (
            f"{NOVEUM_ENDPOINT.rstrip('/')}/v1/projects/{NOVEUM_PROJECT}/policies/usage"
        )
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {NOVEUM_API_KEY}"},
            json={
                "model": MODEL,
                "outcome": "BLOCKED",  # no blockedBy — malformed on purpose
                "eventId": str(uuid.uuid4()),
            },
            timeout=15.0,
            follow_redirects=True,
        )
        print(f"\n[malformed] HTTP {resp.status_code} {resp.text[:300]}")
        assert resp.status_code == 400
