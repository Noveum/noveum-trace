"""Unit tests for RateLimitPolicy — shared read-then-check model (no reserve)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.policies.rate_limit import RateLimitPolicy
from noveum_trace.guard.types import (
    ParsedRequest,
    ParsedResponse,
    PolicyContext,
    PolicyDeps,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ctx(project_id: str = "proj") -> PolicyContext:
    return PolicyContext(
        project_id=project_id,
        organization_id=None,
        environment="test",
        trace_id=None,
        span_id=None,
        call_id=str(uuid.uuid4()),
    )


def _req() -> ParsedRequest:
    return ParsedRequest(
        provider="openai",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        max_tokens=100,
        estimated_input_tokens=50,
        raw_body=b"{}",
    )


def _resp(
    input_tokens: int = 50, output_tokens: int = 100, cost_usd: float = 0.001
) -> ParsedResponse:
    return ParsedResponse(
        model="gpt-4o-mini",
        text="Hello",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def _policy(windows, **kwargs) -> RateLimitPolicy:
    return RateLimitPolicy(windows=windows, project_id="proj", **kwargs)


# ---------------------------------------------------------------------------
# pre() — request-count enforcement
# ---------------------------------------------------------------------------


class TestRequestCountEnforcement:
    def test_allow_under_limit(self):
        policy = _policy([{"period": "1m", "maxRequests": 100}])
        deps = PolicyDeps(api=GuardAPIClient())
        with policy._lock:
            policy.data_map["requests_1m"] = 3

        decision = policy.pre(_req(), _ctx(), deps)

        assert not decision.is_blocking

    def test_block_when_at_limit(self):
        """Guide's formula blocks when the polled count is already >= max."""
        policy = _policy([{"period": "1m", "maxRequests": 100}])
        deps = PolicyDeps(api=GuardAPIClient())
        with policy._lock:
            policy.data_map["requests_1m"] = 100

        decision = policy.pre(_req(), _ctx(), deps)

        assert decision.is_blocking
        assert "requests" in decision.reason

    def test_allow_one_below_limit(self):
        policy = _policy([{"period": "1m", "maxRequests": 100}])
        deps = PolicyDeps(api=GuardAPIClient())
        with policy._lock:
            policy.data_map["requests_1m"] = 99

        decision = policy.pre(_req(), _ctx(), deps)

        assert not decision.is_blocking


# ---------------------------------------------------------------------------
# pre() — token-count enforcement
# ---------------------------------------------------------------------------


class TestTokenCountEnforcement:
    def test_allow_under_token_limit(self):
        policy = _policy([{"period": "1m", "maxTokens": 200_000}])
        deps = PolicyDeps(api=GuardAPIClient())
        with policy._lock:
            policy.data_map["tokens_1m"] = 1_900

        decision = policy.pre(_req(), _ctx(), deps)

        assert not decision.is_blocking

    def test_block_when_at_token_limit(self):
        policy = _policy([{"period": "1m", "maxTokens": 200_000}])
        deps = PolicyDeps(api=GuardAPIClient())
        with policy._lock:
            policy.data_map["tokens_1m"] = 200_000

        decision = policy.pre(_req(), _ctx(), deps)

        assert decision.is_blocking
        assert "tokens" in decision.reason


# ---------------------------------------------------------------------------
# pre() — multiple windows / thresholds
# ---------------------------------------------------------------------------


class TestMultipleWindows:
    def test_blocks_if_any_window_exceeded(self):
        policy = _policy(
            [
                {"period": "1m", "maxRequests": 100},
                {"period": "1h", "maxRequests": 1000},
            ]
        )
        deps = PolicyDeps(api=GuardAPIClient())
        with policy._lock:
            policy.data_map["requests_1m"] = 5
            policy.data_map["requests_1h"] = 1000  # this one is exhausted

        decision = policy.pre(_req(), _ctx(), deps)

        assert decision.is_blocking

    def test_unconfigured_threshold_is_not_enforced(self):
        """maxRequests omitted for a window → that dimension never blocks."""
        policy = _policy([{"period": "1m", "maxTokens": 10}])
        deps = PolicyDeps(api=GuardAPIClient())
        with policy._lock:
            policy.data_map["requests_1m"] = 999_999  # would block if enforced

        decision = policy.pre(_req(), _ctx(), deps)

        assert not decision.is_blocking

    def test_no_windows_configured_always_allows(self):
        policy = _policy([])
        deps = PolicyDeps(api=GuardAPIClient())

        decision = policy.pre(_req(), _ctx(), deps)

        assert not decision.is_blocking


# ---------------------------------------------------------------------------
# post() — local counter update + usage push
# ---------------------------------------------------------------------------


class TestPost:
    def test_post_never_blocks(self):
        policy = _policy([{"period": "1m", "maxRequests": 1}])
        deps = PolicyDeps(api=GuardAPIClient())
        ctx = _ctx()
        pre_decision = policy.pre(_req(), ctx, deps)

        post_decision = policy.post(_resp(), ctx, pre_decision, deps)

        assert not post_decision.is_blocking

    def test_post_increments_all_periods_locally(self):
        policy = _policy([{"period": "1m", "maxRequests": 100}])
        deps = PolicyDeps(api=GuardAPIClient())
        ctx = _ctx()
        pre_decision = policy.pre(_req(), ctx, deps)

        policy.post(_resp(input_tokens=50, output_tokens=100), ctx, pre_decision, deps)

        assert policy.data_map["requests_1m"] == 1
        assert policy.data_map["requests_1h"] == 1
        assert policy.data_map["requests_1d"] == 1
        assert policy.data_map["tokens_1m"] == 150
        assert policy.data_map["tokens_1h"] == 150
        assert policy.data_map["tokens_1d"] == 150

    def test_post_pushes_usage_to_backend(self):
        api = MagicMock(spec=GuardAPIClient)
        policy = _policy([{"period": "1m", "maxRequests": 100}])
        deps = PolicyDeps(api=api)
        ctx = _ctx()
        pre_decision = policy.pre(_req(), ctx, deps)

        policy.post(
            _resp(input_tokens=50, output_tokens=100, cost_usd=0.02),
            ctx,
            pre_decision,
            deps,
        )

        api.report_usage.assert_called_once_with(
            ctx.call_id, "proj", 0.02, "gpt-4o-mini", 50, 100
        )

    def test_post_usage_push_failure_is_swallowed(self):
        api = MagicMock(spec=GuardAPIClient)
        api.report_usage.side_effect = RuntimeError("network down")
        policy = _policy([{"period": "1m", "maxRequests": 100}])
        deps = PolicyDeps(api=api)
        ctx = _ctx()
        pre_decision = policy.pre(_req(), ctx, deps)

        policy.post(_resp(), ctx, pre_decision, deps)  # must not raise


# ---------------------------------------------------------------------------
# poll() — refresh data_map from backend state
# ---------------------------------------------------------------------------


class TestPoll:
    def test_poll_loads_rate_counters(self):
        api = GuardAPIClient()
        api.report_usage(
            str(uuid.uuid4()),
            "proj",
            0.01,
            "gpt-4o-mini",
            input_tokens=10,
            output_tokens=20,
        )
        policy = _policy([{"period": "1m", "maxRequests": 100}])
        deps = PolicyDeps(api=api)

        policy.poll(deps)

        assert policy.data_map["requests_1m"] == 1
        assert policy.data_map["tokens_1m"] == 30

    def test_poll_without_scope_is_noop(self):
        policy = RateLimitPolicy(windows=[{"period": "1m", "maxRequests": 100}])
        deps = PolicyDeps(api=GuardAPIClient())

        policy.poll(deps)  # must not raise despite no project_id/organization_id

        assert policy.data_map == {}

    def test_poll_swallows_backend_errors(self):
        api = MagicMock(spec=GuardAPIClient)
        api.get_state.side_effect = RuntimeError("boom")
        policy = _policy([{"period": "1m", "maxRequests": 100}])
        deps = PolicyDeps(api=api)

        policy.poll(deps)  # must not raise


# ---------------------------------------------------------------------------
# update_params() — backend-pushed config changes
# ---------------------------------------------------------------------------


class TestUpdateParams:
    def test_update_windows(self):
        policy = _policy([{"period": "1m", "maxRequests": 100}])

        policy.update_params({"windows": [{"period": "1h", "maxRequests": 5000}]})

        assert policy.windows == [{"period": "1h", "maxRequests": 5000}]

    def test_update_fail_closed(self):
        policy = _policy([{"period": "1m", "maxRequests": 100}], fail_closed=True)

        policy.update_params({"fail_closed": False})

        assert policy.fail_closed is False

    def test_unknown_keys_ignored(self):
        policy = _policy([{"period": "1m", "maxRequests": 100}])

        policy.update_params({"unrelated": "value"})  # must not raise

        assert policy.windows == [{"period": "1m", "maxRequests": 100}]


# ---------------------------------------------------------------------------
# bind_context() / scoping precedence
# ---------------------------------------------------------------------------


class TestScoping:
    def test_bind_context_adopts_ambient_project(self):
        policy = RateLimitPolicy(windows=[])
        ctx = PolicyContext(
            project_id="ambient-proj",
            organization_id=None,
            environment="test",
            trace_id=None,
            span_id=None,
            call_id=str(uuid.uuid4()),
        )

        policy.bind_context(ctx)

        assert policy._stored_scope_id() == "ambient-proj"

    def test_explicit_project_id_takes_precedence_over_ambient(self):
        policy = RateLimitPolicy(windows=[], project_id="explicit-proj")
        ctx = PolicyContext(
            project_id="ambient-proj",
            organization_id=None,
            environment="test",
            trace_id=None,
            span_id=None,
            call_id=str(uuid.uuid4()),
        )

        policy.bind_context(ctx)

        assert policy._stored_scope_id() == "explicit-proj"

    def test_organization_id_takes_precedence_over_project(self):
        policy = RateLimitPolicy(windows=[], project_id="proj", organization_id="org")

        assert policy._scope_id(_ctx()) == "org"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registered_under_rate_limit_type():
    from noveum_trace.guard.poller import _POLICY_TYPE_REGISTRY

    assert _POLICY_TYPE_REGISTRY["rate_limit"] is RateLimitPolicy
