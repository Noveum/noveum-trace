"""Unit tests for HttpGuardAPIClient — the server-authoritative Guard backend.

The backend is exercised through a fake ``httpx.Client`` injected via
monkeypatch, so these tests assert the exact HTTP contract (URLs, request
bodies, response mapping) without a network.
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from noveum_trace.guard import api_client_http
from noveum_trace.guard.api_client_http import HttpGuardAPIClient
from noveum_trace.guard.exceptions import GuardBackendUnavailable


def _call_id() -> str:
    return str(uuid.uuid4())


class _Resp:
    def __init__(self, status_code: int, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._json


class _FakeClient:
    """Records every get/post so tests can assert the HTTP contract."""

    calls: list[dict] = []
    # Queued POST responses, consumed in order; the last one repeats so a retry
    # loop keeps seeing the same status. Empty means "always 202".
    post_resps: list = []

    def __init__(self, *, get_resp=None, get_exc=None, post_exc=None):
        self._get_resp = get_resp
        self._get_exc = get_exc
        self._post_exc = post_exc

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url, headers=None, params=None):
        _FakeClient.calls.append(
            {"method": "GET", "url": url, "headers": headers, "params": params}
        )
        if self._get_exc is not None:
            raise self._get_exc
        return self._get_resp

    def post(self, url, headers=None, params=None, json=None):
        _FakeClient.calls.append(
            {
                "method": "POST",
                "url": url,
                "headers": headers,
                "params": params,
                "json": json,
            }
        )
        if self._post_exc is not None:
            raise self._post_exc
        queued = _FakeClient.post_resps
        if queued:
            return queued.pop(0) if len(queued) > 1 else queued[0]
        return _Resp(202, {"success": True})


def _patch(monkeypatch, post_resps=None, **kwargs) -> None:
    _FakeClient.calls = []
    _FakeClient.post_resps = list(post_resps or [])
    monkeypatch.setattr(httpx, "Client", lambda **kw: _FakeClient(**kwargs))


def _posts() -> list[dict]:
    return [c for c in _FakeClient.calls if c["method"] == "POST"]


# ---------------------------------------------------------------------------
# get_state — GET /policies/state
# ---------------------------------------------------------------------------


class TestGetState:
    def test_reads_the_requested_window(self, monkeypatch):
        _patch(
            monkeypatch,
            get_resp=_Resp(200, {"cost": {"30d_rolling": 1247.81, "1d_rolling": 12.0}}),
        )
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        assert api.get_state("proj", "30d_rolling") == {"spend": 1247.81, "rate": {}}

    def test_missing_window_reads_zero(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(200, {"cost": {}}))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        assert api.get_state("proj", "7d_rolling") == {"spend": 0.0, "rate": {}}

    def test_reads_rate_counters(self, monkeypatch):
        _patch(
            monkeypatch,
            get_resp=_Resp(
                200,
                {
                    "cost": {"30d_rolling": 1.0},
                    "rate": {
                        "requests_1m": 3,
                        "requests_1h": 3,
                        "requests_1d": 3,
                        "tokens_1m": 1900,
                        "tokens_1h": 1900,
                        "tokens_1d": 1900,
                    },
                },
            ),
        )
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        state = api.get_state("proj", "30d_rolling")
        assert state["rate"] == {
            "requests_1m": 3,
            "requests_1h": 3,
            "requests_1d": 3,
            "tokens_1m": 1900,
            "tokens_1h": 1900,
            "tokens_1d": 1900,
        }

    def test_hits_project_state_url(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(200, {"cost": {}}))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        api.get_state("proj-1")
        call = _FakeClient.calls[0]
        assert call["url"] == "https://api.noveum.ai/v1/projects/proj-1/policies/state"
        assert call["headers"]["Authorization"] == "Bearer k"

    def test_org_slug_sent_as_query_param(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(200, {"cost": {}}))
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", organization_slug="acme"
        )
        api.get_state("proj")
        assert _FakeClient.calls[0]["params"] == {"organizationSlug": "acme"}

    def test_non_200_raises(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(500))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        with pytest.raises(GuardBackendUnavailable):
            api.get_state("proj")

    def test_network_error_raises(self, monkeypatch):
        _patch(monkeypatch, get_exc=httpx.ConnectError("boom"))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        with pytest.raises(GuardBackendUnavailable):
            api.get_state("proj")

    def test_preserves_api_suffix_from_base_url(self, monkeypatch):
        """Guard endpoints live under /api on the production backend (verified
        against the live control plane); DEFAULT_ENDPOINT already includes it,
        so it must be kept, not stripped.
        """
        _patch(monkeypatch, get_resp=_Resp(200, {"cost": {}}))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai/api")
        api.get_state("proj")
        assert _FakeClient.calls[0]["url"].startswith(
            "https://api.noveum.ai/api/v1/projects/proj"
        )


# ---------------------------------------------------------------------------
# report_usage — batched POST /policies/usage
# ---------------------------------------------------------------------------


class TestReportUsage:
    def test_event_has_backend_shape(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        cid = _call_id()
        api.report_usage(cid, "proj", 0.0123, "gpt-4o", 800, 200)
        api.close()  # drains

        post = [c for c in _FakeClient.calls if c["method"] == "POST"][0]
        assert post["url"] == "https://api.noveum.ai/v1/projects/proj/policies/usage"
        event = post["json"][0]
        assert event["model"] == "gpt-4o"
        assert event["inputTokens"] == 800
        assert event["outputTokens"] == 200
        assert event["costUsd"] == 0.0123
        assert event["requestCount"] == 1
        assert event["eventId"] == cid  # idempotency key = call_id
        assert event["timestamp"].endswith("Z")

    def test_auto_flush_when_batch_full(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k",
            base_url="https://api.noveum.ai",
            flush_interval=3600,
            batch_max=2,
        )
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")
        assert [c for c in _FakeClient.calls if c["method"] == "POST"] == []
        # Filling the batch wakes the worker; the flush runs on its thread (never
        # inline on the caller). With flush_interval=3600 only the wake can fire
        # it, so a POST appearing means the batch-full signal worked.
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")
        deadline = time.time() + 5.0
        posts: list = []
        while time.time() < deadline:
            posts = [c for c in _FakeClient.calls if c["method"] == "POST"]
            if posts:
                break
            time.sleep(0.01)
        assert len(posts) == 1
        assert len(posts[0]["json"]) == 2
        api.close()

    def test_events_grouped_by_scope(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        api.report_usage(_call_id(), "proj-a", 1.0, "gpt-4o")
        api.report_usage(_call_id(), "proj-b", 1.0, "gpt-4o")
        api.close()
        posts = [c for c in _FakeClient.calls if c["method"] == "POST"]
        urls = {c["url"] for c in posts}
        assert urls == {
            "https://api.noveum.ai/v1/projects/proj-a/policies/usage",
            "https://api.noveum.ai/v1/projects/proj-b/policies/usage",
        }

    def test_negative_cost_is_dropped(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        api.report_usage(_call_id(), "proj", -1.0, "gpt-4o")
        api.close()
        assert [c for c in _FakeClient.calls if c["method"] == "POST"] == []

    def test_push_failure_is_swallowed(self, monkeypatch):
        _patch(monkeypatch, post_exc=httpx.ConnectError("down"))
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")
        api.close()  # must not raise


# ---------------------------------------------------------------------------
# report_blocked — BLOCKED events on the same POST /policies/usage
# ---------------------------------------------------------------------------


class TestReportBlocked:
    def test_event_has_backend_shape(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        cid = _call_id()
        api.report_blocked(
            cid,
            "proj",
            "gpt-4o",
            "COST_CAP",
            policy_id="pol_abc123",
            reason="30d cost cap exceeded",
        )
        api.close()

        post = _posts()[0]
        assert post["url"] == "https://api.noveum.ai/v1/projects/proj/policies/usage"
        event = post["json"][0]
        assert event["outcome"] == "BLOCKED"
        assert event["blockedBy"] == "COST_CAP"
        assert event["model"] == "gpt-4o"
        assert event["costUsd"] == 0.0  # the call never ran
        assert event["policyId"] == "pol_abc123"
        assert event["reason"] == "30d cost cap exceeded"
        assert event["eventId"] == cid
        assert event["timestamp"].endswith("Z")

    def test_optional_fields_omitted_when_absent(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        api.report_blocked(_call_id(), "proj", "gpt-4o", "RATE_LIMIT")
        api.close()

        event = _posts()[0]["json"][0]
        assert "policyId" not in event
        assert "reason" not in event
        assert event["blockedBy"] == "RATE_LIMIT"

    def test_reason_truncated_to_backend_limit(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        api.report_blocked(_call_id(), "proj", "gpt-4o", "COST_CAP", reason="x" * 900)
        api.close()
        assert len(_posts()[0]["json"][0]["reason"]) == 500

    def test_invalid_blocked_by_is_dropped(self, monkeypatch):
        """An unknown blockedBy is a guaranteed 400 — never let it poison a batch."""
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        api.report_blocked(_call_id(), "proj", "gpt-4o", "SOMETHING_ELSE")
        api.close()
        assert _posts() == []

    def test_blocked_and_allowed_share_one_batch(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        api.report_usage(_call_id(), "proj", 0.01, "gpt-4o", 10, 5)
        api.report_blocked(_call_id(), "proj", "gpt-4o", "COST_CAP")
        api.close()

        posts = _posts()
        assert len(posts) == 1
        outcomes = [e.get("outcome", "ALLOWED") for e in posts[0]["json"]]
        assert outcomes == ["ALLOWED", "BLOCKED"]


# ---------------------------------------------------------------------------
# usage push — HTTP status handling
# ---------------------------------------------------------------------------


class TestUsagePushStatusHandling:
    @staticmethod
    def _api(monkeypatch, post_resps):
        _patch(monkeypatch, post_resps=post_resps)
        monkeypatch.setattr(api_client_http, "_RETRY_BASE_DELAY", 0.0)
        return HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )

    def test_success_posts_once(self, monkeypatch):
        api = self._api(monkeypatch, [_Resp(202, {"success": True})])
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")
        api._flush_once()
        assert len(_posts()) == 1
        api.close()

    def test_400_is_not_retried(self, monkeypatch):
        """A malformed payload fails identically every time — resending it only
        burns requests."""
        api = self._api(monkeypatch, [_Resp(400, {"error": "bad"})])
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")
        api._flush_once()
        assert len(_posts()) == 1
        api.close()

    def test_500_is_retried_to_the_attempt_limit(self, monkeypatch):
        api = self._api(monkeypatch, [_Resp(500)])
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")
        api._flush_once()
        assert len(_posts()) == api_client_http._MAX_PUSH_ATTEMPTS
        api.close()

    def test_retry_reuses_the_same_event_id(self, monkeypatch):
        """Idempotency: the backend dedups on eventId, so a resend must carry
        the original one or the retry double-counts."""
        api = self._api(monkeypatch, [_Resp(503)])
        cid = _call_id()
        api.report_usage(cid, "proj", 1.0, "gpt-4o")
        api._flush_once()
        sent = [p["json"][0]["eventId"] for p in _posts()]
        assert sent == [cid] * api_client_http._MAX_PUSH_ATTEMPTS
        api.close()

    def test_429_is_retried(self, monkeypatch):
        api = self._api(monkeypatch, [_Resp(429, headers={"Retry-After": "0"})])
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")
        api._flush_once()
        assert len(_posts()) == api_client_http._MAX_PUSH_ATTEMPTS
        api.close()

    def test_recovers_when_a_retry_succeeds(self, monkeypatch):
        api = self._api(monkeypatch, [_Resp(500), _Resp(202, {"success": True})])
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")
        api._flush_once()
        assert len(_posts()) == 2  # stops as soon as one succeeds
        api.close()


class TestRetryDelay:
    def test_honors_retry_after_seconds(self):
        api = HttpGuardAPIClient(api_key="k")
        assert api._retry_delay(_Resp(429, headers={"Retry-After": "7"}), 1) == 7.0

    def test_caps_retry_after(self):
        """A hostile or mistaken header must not park the flush thread for hours."""
        api = HttpGuardAPIClient(api_key="k")
        delay = api._retry_delay(_Resp(429, headers={"Retry-After": "99999"}), 1)
        assert delay == api_client_http._MAX_RETRY_DELAY

    def test_http_date_falls_back_to_backoff(self):
        api = HttpGuardAPIClient(api_key="k")
        resp = _Resp(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
        assert api._retry_delay(resp, 2) == api_client_http._RETRY_BASE_DELAY * 2

    def test_backoff_doubles_without_a_header(self):
        api = HttpGuardAPIClient(api_key="k")
        base = api_client_http._RETRY_BASE_DELAY
        delays = [api._retry_delay(_Resp(500), n) for n in (1, 2, 3)]
        assert delays == [base, base * 2, base * 4]


# ---------------------------------------------------------------------------
# reserve / reconcile — unsupported by the HTTP backend
# ---------------------------------------------------------------------------


class TestReservationUnsupported:
    def test_supports_reservation_is_false(self):
        api = HttpGuardAPIClient(api_key="k")
        assert api.supports_reservation is False

    def test_reserve_raises(self):
        api = HttpGuardAPIClient(api_key="k")
        with pytest.raises(NotImplementedError):
            api.reserve(_call_id(), "proj", 1.0, 100.0)

    def test_reconcile_raises(self):
        api = HttpGuardAPIClient(api_key="k")
        with pytest.raises(NotImplementedError):
            api.reconcile(_call_id(), "proj", 1.0)


# ---------------------------------------------------------------------------
# fetch_remote_policies — GET /policies/effective + shape mapping
# ---------------------------------------------------------------------------


class TestFetchRemotePolicies:
    def test_maps_cost_cap_shape(self, monkeypatch):
        _patch(
            monkeypatch,
            get_resp=_Resp(
                200,
                [
                    {
                        "policyId": "p1",
                        "name": "Monthly budget",
                        "type": "COST_CAP",
                        "enabled": True,
                        "failClosed": True,
                        "config": {
                            "window": "30d_rolling",
                            "maxUsd": 1500,
                            "action": "BLOCK",
                        },
                    }
                ],
            ),
        )
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        result = api.fetch_remote_policies("proj")
        assert result == [
            {
                "type": "cost_cap",
                "name": "Monthly budget",
                "fail_closed": True,
                "policy_id": "p1",
                "max_usd": 1500,
                "window": "30d_rolling",
            }
        ]

    def test_maps_rate_limit_shape(self, monkeypatch):
        _patch(
            monkeypatch,
            get_resp=_Resp(
                200,
                [
                    {
                        "policyId": "p2",
                        "name": "Burst rate",
                        "type": "RATE_LIMIT",
                        "enabled": True,
                        "failClosed": True,
                        "config": {
                            "windows": [
                                {
                                    "period": "1m",
                                    "maxRequests": 100,
                                    "maxTokens": 200000,
                                    "action": "BLOCK",
                                }
                            ]
                        },
                    }
                ],
            ),
        )
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        result = api.fetch_remote_policies("proj")
        assert result == [
            {
                "type": "rate_limit",
                "name": "Burst rate",
                "fail_closed": True,
                "policy_id": "p2",
                "windows": [
                    {
                        "period": "1m",
                        "maxRequests": 100,
                        "maxTokens": 200000,
                        "action": "BLOCK",
                    }
                ],
            }
        ]

    def test_hits_effective_url(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(200, []))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        api.fetch_remote_policies("proj")
        assert (
            _FakeClient.calls[0]["url"]
            == "https://api.noveum.ai/v1/projects/proj/policies/effective"
        )

    def test_disabled_policy_skipped(self, monkeypatch):
        _patch(
            monkeypatch,
            get_resp=_Resp(
                200,
                [
                    {
                        "name": "off",
                        "type": "COST_CAP",
                        "enabled": False,
                        "config": {"maxUsd": 10, "window": "1d_rolling"},
                    }
                ],
            ),
        )
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        assert api.fetch_remote_policies("proj") == []

    def test_unknown_type_skipped(self, monkeypatch):
        _patch(
            monkeypatch,
            get_resp=_Resp(
                200,
                [{"name": "x", "type": "SOMETHING_NEW", "enabled": True, "config": {}}],
            ),
        )
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        assert api.fetch_remote_policies("proj") == []

    def test_non_200_raises(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(403))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        with pytest.raises(GuardBackendUnavailable):
            api.fetch_remote_policies("proj")
