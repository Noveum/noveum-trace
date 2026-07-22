"""Real-call ("real time") integration tests for RateLimitPolicy.

Mirrors test_real_calls.py's pattern (real Anthropic calls through the full
Guard transport + PolicyEngine stack, backed by the in-memory GuardAPIClient)
but exercises RateLimitPolicy's request-count and token-count enforcement
instead of CostCapPolicy's spend cap.

Every test skips unless a valid ``ANTHROPIC_API_KEY`` is present, so the suite
is a no-op until keys are configured. Run explicitly with:

    pytest tests/integration/guard/test_rate_limit_real_calls.py -m integration -v
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

import pytest

# Load .env so locally-exported provider keys are picked up automatically.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv optional
    pass

import noveum_trace
from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.policies.rate_limit import RateLimitPolicy
from noveum_trace.guard.types import PolicyContext

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:  # provider SDK optional
    anthropic = None  # type: ignore[assignment]
    ANTHROPIC_AVAILABLE = False

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = os.environ.get("NOVEUM_GUARD_TEST_MODEL", "claude-haiku-4-5-20251001")


def _is_valid_key(key: Optional[str]) -> bool:
    invalid = {"", "your-anthropic-api-key-here", "test-key", "sk-test", "sk-fake"}
    return bool(key) and key not in invalid and len(key) > 10


def _should_test_anthropic() -> bool:
    return ANTHROPIC_AVAILABLE and _is_valid_key(ANTHROPIC_API_KEY)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _should_test_anthropic(),
        reason="ANTHROPIC_API_KEY not set/valid or anthropic not installed",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(project_id: str) -> PolicyContext:
    return PolicyContext(
        project_id=project_id,
        organization_id=None,
        environment="real-test",
        trace_id=None,
        span_id=None,
        call_id=str(uuid.uuid4()),
    )


def _messages() -> list[dict]:
    return [{"role": "user", "content": "Reply with one word: done."}]


def _build_stack(
    project_id: str, windows: list[dict]
) -> tuple[PolicyEngine, PolicyContext, GuardAPIClient]:
    """A real-call Guard stack: fresh in-memory api + a single RateLimitPolicy."""
    api = GuardAPIClient()
    engine = PolicyEngine(api_client=api)
    engine.attach(RateLimitPolicy(windows=windows, project_id=project_id))
    return engine, _ctx(project_id), api


# ---------------------------------------------------------------------------
# Request-count enforcement
# ---------------------------------------------------------------------------


class TestRequestCountLimit:
    def test_calls_within_limit_are_allowed(self):
        project_id = "guard-real-rate-requests-ok"
        engine, ctx, api = _build_stack(
            project_id, windows=[{"period": "1m", "maxRequests": 2}]
        )
        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )

        for _ in range(2):
            resp = client.messages.create(
                model=MODEL, messages=_messages(), max_tokens=16
            )
            assert resp.content[0].text  # real response came back

        assert api.current_rate(project_id)["requests_1m"] == 2

    def test_call_over_request_limit_is_blocked(self):
        project_id = "guard-real-rate-requests-blocked"
        engine, ctx, api = _build_stack(
            project_id, windows=[{"period": "1m", "maxRequests": 2}]
        )
        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )

        # First two calls consume the request budget for this window.
        for _ in range(2):
            client.messages.create(model=MODEL, messages=_messages(), max_tokens=16)

        # Third call is blocked in pre() — never reaches the network.
        with pytest.raises(anthropic.PermissionDeniedError):
            client.messages.create(model=MODEL, messages=_messages(), max_tokens=16)

        # The blocked call must not have been counted.
        assert api.current_rate(project_id)["requests_1m"] == 2


# ---------------------------------------------------------------------------
# Token-count enforcement
# ---------------------------------------------------------------------------


class TestTokenCountLimit:
    def test_call_over_token_limit_is_blocked(self):
        project_id = "guard-real-rate-tokens-blocked"
        # One real call comfortably uses more than 10 combined input+output
        # tokens, so the second call must be blocked.
        engine, ctx, api = _build_stack(
            project_id, windows=[{"period": "1m", "maxTokens": 10}]
        )
        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )

        resp = client.messages.create(model=MODEL, messages=_messages(), max_tokens=16)
        assert resp.content[0].text
        assert api.current_rate(project_id)["tokens_1m"] > 10

        with pytest.raises(anthropic.PermissionDeniedError):
            client.messages.create(model=MODEL, messages=_messages(), max_tokens=16)


# ---------------------------------------------------------------------------
# Async transport
# ---------------------------------------------------------------------------


class TestAsyncRequestCountLimit:
    async def test_async_call_over_request_limit_is_blocked(self):
        project_id = "guard-real-rate-async-blocked"
        engine, ctx, api = _build_stack(
            project_id, windows=[{"period": "1m", "maxRequests": 1}]
        )
        client = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.async_http_client(engine, ctx),
        )
        try:
            resp = await client.messages.create(
                model=MODEL, messages=_messages(), max_tokens=16
            )
            assert resp.content[0].text

            with pytest.raises(anthropic.PermissionDeniedError):
                await client.messages.create(
                    model=MODEL, messages=_messages(), max_tokens=16
                )
        finally:
            await client.close()

        assert api.current_rate(project_id)["requests_1m"] == 1


# ---------------------------------------------------------------------------
# Multiple windows combined
# ---------------------------------------------------------------------------


class TestMultipleWindowsRealCalls:
    def test_tight_token_window_blocks_before_generous_request_window(self):
        """A generous request limit but a tight token limit — the second call
        should be blocked by the token dimension, not the request dimension.
        """
        project_id = "guard-real-rate-combined"
        engine, ctx, api = _build_stack(
            project_id,
            windows=[
                {"period": "1m", "maxRequests": 100},
                {"period": "1h", "maxTokens": 10},
            ],
        )
        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )

        client.messages.create(model=MODEL, messages=_messages(), max_tokens=16)

        with pytest.raises(anthropic.PermissionDeniedError):
            client.messages.create(model=MODEL, messages=_messages(), max_tokens=16)

        # Well under the request cap — proves it was the token window that blocked.
        assert api.current_rate(project_id)["requests_1m"] == 1
