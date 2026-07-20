"""CostCapPolicy against a non-reserving (HTTP-style) backend.

When the backend cannot reserve atomically (``supports_reservation = False``),
strict mode must degrade to the shared read-then-check model: never call
``reserve``/``reconcile``, block off polled shared spend, and push the full
actual usage in post().
"""

from __future__ import annotations

import uuid

from noveum_trace.guard.policies.cost_cap import CostCapPolicy
from noveum_trace.guard.types import (
    EnforcementMode,
    ParsedRequest,
    ParsedResponse,
    PolicyContext,
    PolicyDeps,
)


class _FakeSharedClient:
    """Server-authoritative client stub: reserve/reconcile are hard errors."""

    supports_reservation = False

    def __init__(self, spend: float = 0.0):
        self._spend = spend
        self.usage_calls: list[tuple] = []

    def get_state(self, project_id, window=None):
        return {"spend": self._spend}

    def report_usage(
        self, call_id, project_id, actual_usd, model, input_tokens=0, output_tokens=0
    ):
        self.usage_calls.append(
            (call_id, project_id, actual_usd, model, input_tokens, output_tokens)
        )

    def reserve(self, *a, **k):  # must never be called in shared mode
        raise AssertionError("reserve() called against a non-reserving backend")

    def reconcile(self, *a, **k):
        raise AssertionError("reconcile() called against a non-reserving backend")


def _ctx() -> PolicyContext:
    return PolicyContext(
        project_id="proj",
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
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        max_tokens=100,
        estimated_input_tokens=50,
        raw_body=b"{}",
    )


def _resp(cost_usd: float) -> ParsedResponse:
    return ParsedResponse(
        model="gpt-4o-mini",
        text="ok",
        input_tokens=50,
        output_tokens=100,
        cost_usd=cost_usd,
    )


def _policy() -> CostCapPolicy:
    # strict is requested but must degrade because the backend can't reserve.
    return CostCapPolicy(max_usd=100.0, mode=EnforcementMode.strict, project_id="proj")


def test_strict_degrades_and_allows_under_shared_spend():
    api = _FakeSharedClient(spend=0.0)
    deps = PolicyDeps(api=api)
    policy = _policy()
    policy.poll(deps)  # load shared spend into data_map
    decision = policy.pre(_req(), _ctx(), deps)
    assert not decision.is_blocking  # and reserve() was never called


def test_strict_degrades_and_blocks_over_shared_spend():
    api = _FakeSharedClient(spend=100.0)  # cap exhausted per shared counter
    deps = PolicyDeps(api=api)
    policy = _policy()
    policy.poll(deps)
    decision = policy.pre(_req(), _ctx(), deps)
    assert decision.is_blocking


def test_post_pushes_full_actual_usage_with_tokens():
    api = _FakeSharedClient(spend=0.0)
    deps = PolicyDeps(api=api)
    policy = _policy()
    policy.poll(deps)
    ctx = _ctx()
    decision = policy.pre(_req(), ctx, deps)
    policy.post(_resp(cost_usd=0.5), ctx, decision, deps)

    assert len(api.usage_calls) == 1
    call_id, project_id, actual_usd, model, in_tok, out_tok = api.usage_calls[0]
    assert actual_usd == 0.5  # full actual, not an over-estimate excess
    assert (in_tok, out_tok) == (50, 100)


def test_release_does_not_reconcile_when_backend_cannot_reserve():
    api = _FakeSharedClient(spend=0.0)
    deps = PolicyDeps(api=api)
    policy = _policy()
    policy.poll(deps)
    ctx = _ctx()
    decision = policy.pre(_req(), ctx, deps)
    policy.release(decision, ctx, deps)  # must not raise AssertionError
