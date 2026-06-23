"""
Multi-policy engine tests.

Key scenario (the one the user described):
  CostCapPolicy (priority 10) → pre() ALLOWS, reserves budget
  BlockingPolicy  (priority 20) → pre() BLOCKS
  ↓
  Engine rolls back CostCapPolicy via release()
  → no LLM call happens
  → reserved budget returned to zero
  → api_client.current_spend() == 0
"""

import uuid

import pytest

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.policies.base import AbstractPolicy
from noveum_trace.guard.policies.cost_cap import CostCapPolicy
from noveum_trace.guard.types import (
    EnforcementMode,
    ParsedRequest,
    ParsedResponse,
    Phase,
    PolicyContext,
)

# ---------------------------------------------------------------------------
# Helpers
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


def _req(model: str = "gpt-4o-mini") -> ParsedRequest:
    return ParsedRequest(
        provider="openai",
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        max_tokens=100,
        estimated_input_tokens=50,
        raw_body=b"{}",
    )


def _resp(cost_usd: float = 1e-06) -> ParsedResponse:
    """Default cost is intentionally tiny — actual is always ≤ reserved (worst-case estimate)."""
    return ParsedResponse(
        model="gpt-4o-mini",
        text="hi",
        input_tokens=50,
        output_tokens=100,
        cost_usd=cost_usd,
    )


class AlwaysAllowPolicy(AbstractPolicy):
    """Dummy policy that always allows. Tracks calls via side_effects list."""

    name = "always_allow"
    priority = 50

    def __init__(self) -> None:
        super().__init__()
        self.pre_calls: list = []
        self.post_calls: list = []
        self.release_calls: list = []

    def pre(self, parsed, ctx, deps) -> PolicyDecision:
        self.pre_calls.append(ctx.call_id)
        return PolicyDecision.allow(self.name, Phase.pre, state={"marker": "allowed"})

    def post(self, resp, ctx, decision, deps) -> PolicyDecision:
        self.post_calls.append(ctx.call_id)
        return PolicyDecision.allow(self.name, Phase.post)

    def release(self, decision, ctx, deps) -> None:
        self.release_calls.append(ctx.call_id)


class AlwaysBlockPolicy(AbstractPolicy):
    """Dummy policy that always blocks in pre()."""

    name = "always_block"
    priority = 90

    def __init__(self) -> None:
        super().__init__()
        self.pre_calls: list = []
        self.release_calls: list = []

    def pre(self, parsed, ctx, deps) -> PolicyDecision:
        self.pre_calls.append(ctx.call_id)
        return PolicyDecision.block(
            self.name, Phase.pre, reason="blocked by test policy"
        )

    def release(self, decision, ctx, deps) -> None:
        self.release_calls.append(ctx.call_id)


class BlockInPostPolicy(AbstractPolicy):
    """Always allows in pre, always blocks in post."""

    name = "block_in_post"
    priority = 80

    def __init__(self) -> None:
        super().__init__()
        self.post_calls: list = []

    def pre(self, parsed, ctx, deps) -> PolicyDecision:
        return PolicyDecision.allow(self.name, Phase.pre)

    def post(self, resp, ctx, decision, deps) -> PolicyDecision:
        self.post_calls.append(ctx.call_id)
        return PolicyDecision.block(self.name, Phase.post, reason="blocked post")


# ---------------------------------------------------------------------------
# Core scenario: CostCap allows → second policy blocks → budget released
# ---------------------------------------------------------------------------


class TestCostCapRollbackOnPreBlock:
    def test_budget_is_zero_after_second_policy_blocks(self):
        """
        CostCapPolicy (priority 10) reserves budget.
        AlwaysBlockPolicy (priority 90) blocks.
        Engine rolls back CostCap via release().
        Final spend must be 0 — no money consumed.
        """
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        cost_cap = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        cost_cap.priority = 10

        blocker = AlwaysBlockPolicy()

        engine.attach(cost_cap)
        engine.attach(blocker)

        block_decision, ran = engine.pre_call(_req(), _ctx())

        assert block_decision is not None
        assert block_decision.is_blocking
        assert api.current_spend("proj") == pytest.approx(0.0)

    def test_llm_would_not_be_called_on_pre_block(self):
        """Caller receives a block decision — it must not forward the request."""
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        cost_cap = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        cost_cap.priority = 10
        engine.attach(cost_cap)
        engine.attach(AlwaysBlockPolicy())

        block_decision, _ = engine.pre_call(_req(), _ctx())

        # Transport layer checks this before forwarding
        assert block_decision is not None
        assert block_decision.is_blocking

    def test_blocker_policy_is_released_on_rollback(self):
        """All policies in ran — including the blocker — get release() called.

        The blocker must be released so that any inflight reservation it made
        before throwing (causing _safe_invoke to synthesise a block decision with
        empty state) is cleaned up; otherwise the inflight entry leaks permanently.
        """
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        allow_spy = AlwaysAllowPolicy()
        allow_spy.priority = 10
        blocker = AlwaysBlockPolicy()
        blocker.priority = 20

        engine.attach(allow_spy)
        engine.attach(blocker)

        engine.pre_call(_req(), _ctx())

        assert len(allow_spy.release_calls) == 1  # rolled back
        assert len(blocker.release_calls) == 1  # blocker is also released

    def test_inflight_entry_cleared_after_rollback(self):
        """reserve() adds to _inflight; release() via reconcile() removes it."""
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        cost_cap = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        cost_cap.priority = 10
        engine.attach(cost_cap)
        engine.attach(AlwaysBlockPolicy())

        engine.pre_call(_req(), _ctx())

        assert api.inflight_count() == 0


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_lower_priority_runs_first(self):
        """Policy with priority=10 must fire before priority=50."""
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        order: list = []

        class First(AbstractPolicy):
            name = "first"
            priority = 10

            def pre(self, parsed, ctx, deps):
                order.append("first")
                return PolicyDecision.allow(self.name, Phase.pre)

        class Second(AbstractPolicy):
            name = "second"
            priority = 50

            def pre(self, parsed, ctx, deps):
                order.append("second")
                return PolicyDecision.allow(self.name, Phase.pre)

        # attach in reverse order to prove sorting works
        engine.attach(Second())
        engine.attach(First())

        engine.pre_call(_req(), _ctx())

        assert order == ["first", "second"]

    def test_block_at_first_policy_skips_remaining(self):
        """If priority=10 blocks, priority=50 never runs."""
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        early_blocker = AlwaysBlockPolicy()
        early_blocker.priority = 10
        late_spy = AlwaysAllowPolicy()
        late_spy.priority = 50

        engine.attach(early_blocker)
        engine.attach(late_spy)

        engine.pre_call(_req(), _ctx())

        assert len(late_spy.pre_calls) == 0


# ---------------------------------------------------------------------------
# Post — all hooks run before block surfaces
# ---------------------------------------------------------------------------


class TestPostAllRunBeforeBlock:
    def test_cost_cap_post_runs_even_when_another_policy_blocks_post(self):
        """
        CostCapPolicy reconciles in post() — it must run even if a later
        policy returns BLOCK in post().
        """
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        cost_cap = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        cost_cap.priority = 10

        post_blocker = BlockInPostPolicy()
        post_blocker.priority = 50

        engine.attach(cost_cap)
        engine.attach(post_blocker)

        ctx = _ctx()

        # pre — both allow
        block, ran = engine.pre_call(_req(), ctx)
        assert block is None
        spend_after_pre = api.current_spend("proj")
        assert spend_after_pre > 0  # CostCap reserved something

        # actual (1e-06) < reserved (~6.75e-05) — always true by construction:
        # reserved = worst-case (max_tokens output), actual ≤ max_tokens output
        post_block = engine.post_call(_resp(), ctx, ran)

        assert post_block is not None
        assert post_block.is_blocking
        # Spend after post should be lower than after pre (reconcile returned over-estimate)
        assert api.current_spend("proj") < spend_after_pre

    def test_all_post_hooks_called_even_after_first_block(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        spy1 = AlwaysAllowPolicy()
        spy1.priority = 10
        spy1.name = "spy1"

        post_blocker = BlockInPostPolicy()
        post_blocker.priority = 20

        spy2 = AlwaysAllowPolicy()
        spy2.priority = 30
        spy2.name = "spy2"

        engine.attach(spy1)
        engine.attach(post_blocker)
        engine.attach(spy2)

        ctx = _ctx()
        _, ran = engine.pre_call(_req(), ctx)
        engine.post_call(_resp(), ctx, ran)

        # Both spies' post() must have been called
        assert len(spy1.post_calls) == 1
        assert len(spy2.post_calls) == 1


# ---------------------------------------------------------------------------
# release_all — called when LLM errors after forwarding
# ---------------------------------------------------------------------------


class TestReleaseAll:
    def test_release_all_refunds_all_policies(self):
        """If the LLM call throws after pre, engine.release_all() un-reserves everything."""
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        cost_cap = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        spy = AlwaysAllowPolicy()
        spy.priority = 50

        engine.attach(cost_cap)
        engine.attach(spy)

        ctx = _ctx()
        _, ran = engine.pre_call(_req(), ctx)
        spend_before_error = api.current_spend("proj")
        assert spend_before_error > 0

        engine.release_all(ctx, ran)

        assert api.current_spend("proj") == pytest.approx(0.0)
        assert len(spy.release_calls) == 1

    def test_release_all_clears_inflight(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        cost_cap = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj"
        )
        engine.attach(cost_cap)

        ctx = _ctx()
        _, ran = engine.pre_call(_req(), ctx)
        assert api.inflight_count() == 1

        engine.release_all(ctx, ran)
        assert api.inflight_count() == 0


# ---------------------------------------------------------------------------
# Two CostCap policies on different projects — isolation
# ---------------------------------------------------------------------------


class TestMultipleCostCapPolicies:
    def test_two_cost_caps_on_different_projects_are_independent(self):
        """Blocking proj-a must not affect proj-b's budget."""
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        cap_a = CostCapPolicy(
            max_usd=0.0, mode=EnforcementMode.strict, project_id="proj-a"
        )
        cap_a.name = "cost_cap_a"
        cap_a.priority = 10

        cap_b = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="proj-b"
        )
        cap_b.name = "cost_cap_b"
        cap_b.priority = 20

        engine.attach(cap_a)
        engine.attach(cap_b)

        ctx_a = _ctx(project_id="proj-a")
        block, _ = engine.pre_call(_req(), ctx_a)

        # proj-a blocks (cap=0), proj-b untouched
        assert block is not None
        assert api.current_spend("proj-b") == 0.0
