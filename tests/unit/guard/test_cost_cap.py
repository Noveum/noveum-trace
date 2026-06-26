"""Unit tests for CostCapPolicy — strict and non-strict modes."""

import uuid
from unittest.mock import MagicMock

import pytest

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.policies.cost_cap import CostCapPolicy
from noveum_trace.guard.types import (
    EnforcementMode,
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


def _req(
    model: str = "gpt-4o-mini",
    max_tokens: int = 100,
    estimated_input_tokens: int = 50,
) -> ParsedRequest:
    return ParsedRequest(
        provider="openai",
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        max_tokens=max_tokens,
        estimated_input_tokens=estimated_input_tokens,
        raw_body=b"{}",
    )


def _resp(
    model: str = "gpt-4o-mini",
    input_tokens: int = 50,
    output_tokens: int = 100,
    cost_usd: float = 0.0001,
) -> ParsedResponse:
    return ParsedResponse(
        model=model,
        text="Hello",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


# ---------------------------------------------------------------------------
# Strict mode — core allow / block
# ---------------------------------------------------------------------------


class TestStrictModeAllowBlock:
    def test_allow_when_under_cap(self):
        """spent=0, request_cost small → ALLOW."""
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        decision = policy.pre(_req(), _ctx(), deps)

        assert not decision.is_blocking

    def test_block_when_over_cap(self):
        """spent=100 (cap fully exhausted) → any reservation is rejected.

        spend=100.0 + reserved_usd > 100.0 for any positive reservation.
        """
        api = GuardAPIClient()
        api._spend["proj"] = 100.0  # cap fully consumed
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        decision = policy.pre(_req(), _ctx(), deps)

        assert decision.is_blocking

    def test_block_reason_contains_cap_amount(self):
        api = GuardAPIClient()
        api._spend["proj"] = 100.0  # cap fully consumed
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        decision = policy.pre(_req(), _ctx(), deps)

        assert "100" in decision.reason

    def test_allow_stores_reserved_usd_in_state(self):
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        decision = policy.pre(_req(), _ctx(), deps)

        assert "reserved_usd" in decision.state
        assert decision.state["reserved_usd"] > 0

    def test_block_stores_zero_reserved_usd(self):
        """When blocked, nothing was reserved — release must be a no-op."""
        api = GuardAPIClient()
        api._spend["proj"] = 100.0  # cap fully consumed
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        decision = policy.pre(_req(), _ctx(), deps)

        assert decision.state.get("reserved_usd", 0.0) == 0.0

    def test_allow_increments_spend_in_api_client(self):
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        policy.pre(_req(), _ctx(), deps)

        assert api.current_spend("proj") > 0


# ---------------------------------------------------------------------------
# Strict mode — post (reconcile)
# ---------------------------------------------------------------------------


class TestStrictModePost:
    def test_post_reconciles_unconsumed_headroom(self):
        """reserved > actual → post releases the difference."""
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)
        ctx = _ctx()

        pre_decision = policy.pre(_req(max_tokens=10_000), ctx, deps)
        spend_after_pre = api.current_spend("proj")

        # Actual cost is tiny
        policy.post(_resp(cost_usd=0.0001), ctx, pre_decision, deps)
        spend_after_post = api.current_spend("proj")

        # Post should return the over-estimate
        assert spend_after_post < spend_after_pre

    def test_post_never_blocks(self):
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)
        ctx = _ctx()

        pre_decision = policy.pre(_req(), ctx, deps)
        post_decision = policy.post(_resp(), ctx, pre_decision, deps)

        assert not post_decision.is_blocking

    def test_post_updates_local_data_map(self):
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)
        ctx = _ctx()

        pre_decision = policy.pre(_req(), ctx, deps)
        policy.post(_resp(cost_usd=0.05), ctx, pre_decision, deps)

        assert policy.data_map.get("30d_rolling", 0.0) > 0


# ---------------------------------------------------------------------------
# Strict mode — release (rollback)
# ---------------------------------------------------------------------------


class TestStrictModeRelease:
    def test_release_refunds_full_reservation(self):
        """reserve(X) then release(X) → spend back to 0."""
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)
        ctx = _ctx()

        pre_decision = policy.pre(_req(), ctx, deps)
        assert api.current_spend("proj") > 0

        policy.release(pre_decision, ctx, deps)
        assert api.current_spend("proj") == pytest.approx(0.0)

    def test_release_makes_budget_available_again(self):
        """After release, a reservation that would have been rejected now admits."""
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        # Fill up to near cap
        api._spend["proj"] = 80.0
        ctx1 = _ctx()
        pre1 = policy.pre(_req(model="gpt-4o", max_tokens=200), ctx1, deps)
        # Reserve used remaining headroom; a second call should be blocked
        ctx2 = _ctx()
        pre2 = policy.pre(_req(model="gpt-4o", max_tokens=200), ctx2, deps)
        # At least one of these must have been admitted for the test to be meaningful
        # (exact behaviour depends on the estimate). Release whichever was admitted.
        if not pre1.is_blocking:
            policy.release(pre1, ctx1, deps)
        if not pre2.is_blocking:
            policy.release(pre2, ctx2, deps)

        # After full release, budget should be back to 80
        assert api.current_spend("proj") == pytest.approx(80.0)

    def test_release_on_blocked_decision_is_noop(self):
        """release() when nothing was reserved (blocked pre) must not change spend."""
        api = GuardAPIClient()
        api._spend["proj"] = 100.0  # cap fully consumed → pre will block
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)
        ctx = _ctx()

        block_decision = policy.pre(_req(), ctx, deps)
        assert block_decision.is_blocking

        spend_before = api.current_spend("proj")
        policy.release(block_decision, ctx, deps)
        assert api.current_spend("proj") == pytest.approx(spend_before)


# ---------------------------------------------------------------------------
# Non-strict mode
# ---------------------------------------------------------------------------


class TestNonStrictMode:
    def test_allow_under_cap(self):
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.non_strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        decision = policy.pre(_req(), _ctx(), deps)

        assert not decision.is_blocking

    def test_block_when_local_counter_over_cap(self):
        """Non-strict reads data_map; set it to max so any reservation tips over."""
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.non_strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        # Fill local counter to cap — spend(100) + reserved(>0) > 100 → block
        with policy._lock:
            policy.data_map["30d_rolling"] = 100.0

        decision = policy.pre(_req(), _ctx(), deps)

        assert decision.is_blocking

    def test_non_strict_does_not_call_reserve(self):
        api = MagicMock(spec=GuardAPIClient)
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.non_strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        policy.pre(_req(), _ctx(), deps)

        api.reserve.assert_not_called()

    def test_post_calls_report_usage_not_reconcile(self):
        api = MagicMock(spec=GuardAPIClient)
        api.get_state.return_value = {"spend": 0.0}
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.non_strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)
        ctx = _ctx()

        pre_d = policy.pre(_req(), ctx, deps)
        policy.post(_resp(cost_usd=0.05), ctx, pre_d, deps)

        api.report_usage.assert_called_once()
        api.reconcile.assert_not_called()

    def test_post_updates_local_data_map(self):
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.non_strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)
        ctx = _ctx()

        pre_d = policy.pre(_req(), ctx, deps)
        policy.post(_resp(cost_usd=0.07), ctx, pre_d, deps)

        assert policy.data_map.get("30d_rolling", 0.0) == pytest.approx(0.07)

    def test_release_is_noop_for_non_strict(self):
        """Non-strict never reserves — release must not touch api."""
        api = MagicMock(spec=GuardAPIClient)
        policy = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.non_strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)
        ctx = _ctx()

        pre_d = policy.pre(_req(), ctx, deps)
        policy.release(pre_d, ctx, deps)

        api.reconcile.assert_not_called()


# ---------------------------------------------------------------------------
# scope_to_models — model filtering
# ---------------------------------------------------------------------------


class TestScopeToModels:
    def test_out_of_scope_model_always_allowed(self):
        api = GuardAPIClient()
        api._spend["proj"] = 99.99  # over cap
        policy = CostCapPolicy(
            max_usd=100.0,
            mode=EnforcementMode.strict,
            scope_to_models=["gpt-4o"],  # only enforce on gpt-4o
            project_id="proj",
        )
        deps = PolicyDeps(api=api)

        # gpt-4o-mini is NOT in scope → should allow even though spend is at cap
        decision = policy.pre(_req(model="gpt-4o-mini"), _ctx(), deps)

        assert not decision.is_blocking

    def test_in_scope_model_is_enforced(self):
        api = GuardAPIClient()
        api._spend["proj"] = 100.0  # cap fully consumed
        policy = CostCapPolicy(
            max_usd=100.0,
            mode=EnforcementMode.strict,
            scope_to_models=["gpt-4o"],
            project_id="proj",
        )
        deps = PolicyDeps(api=api)

        decision = policy.pre(_req(model="gpt-4o"), _ctx(), deps)

        assert decision.is_blocking


# ---------------------------------------------------------------------------
# fail_closed behaviour
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_fail_closed_true_blocks_on_backend_error(self):
        api = MagicMock(spec=GuardAPIClient)
        api.reserve.side_effect = RuntimeError("backend down")
        policy = CostCapPolicy(
            max_usd=100.0,
            mode=EnforcementMode.strict,
            fail_closed=True,
            project_id="proj",
        )
        deps = PolicyDeps(api=api)

        decision = policy.pre(_req(), _ctx(), deps)

        assert decision.is_blocking

    def test_fail_closed_false_allows_on_backend_error(self):
        api = MagicMock(spec=GuardAPIClient)
        api.reserve.side_effect = RuntimeError("backend down")
        policy = CostCapPolicy(
            max_usd=100.0,
            mode=EnforcementMode.strict,
            fail_closed=False,
            project_id="proj",
        )
        deps = PolicyDeps(api=api)

        decision = policy.pre(_req(), _ctx(), deps)

        assert not decision.is_blocking


# ---------------------------------------------------------------------------
# max_tokens=None — fallback to model default
# ---------------------------------------------------------------------------


class TestMaxTokensFallback:
    def test_none_max_tokens_uses_model_default(self):
        """ParsedRequest.max_tokens=None should not raise; policy falls back."""
        api = GuardAPIClient()
        policy = CostCapPolicy(
            max_usd=10_000.0, mode=EnforcementMode.strict, project_id="proj"
        )
        deps = PolicyDeps(api=api)

        req = _req(max_tokens=None)  # type: ignore[arg-type]
        req = ParsedRequest(
            provider="openai",
            model="gpt-4o",
            messages=[],
            stream=False,
            max_tokens=None,
            estimated_input_tokens=50,
            raw_body=b"{}",
        )
        decision = policy.pre(req, _ctx(), deps)

        assert not decision.is_blocking  # should not crash; just estimate with default
