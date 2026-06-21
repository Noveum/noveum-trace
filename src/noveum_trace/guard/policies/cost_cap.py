from __future__ import annotations

from typing import Optional

from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.policies.base import AbstractPolicy
from noveum_trace.guard.types import (
    EnforcementMode,
    ParsedRequest,
    ParsedResponse,
    Phase,
    PolicyContext,
    PolicyDeps,
)
from noveum_trace.utils.llm_utils import estimate_cost, get_model_info


class CostCapPolicy(AbstractPolicy):
    """Block calls that would exceed a per-project USD spend cap.

    strict mode: reserves worst-case cost atomically in GuardAPIClient before
    forwarding; reconciles after. Prevents any overshoot under concurrency.

    non_strict mode: checks a local counter; allows all calls that look under-cap
    at the moment of pre(). Reports actual cost in post(). May overshoot slightly
    under high concurrency — advisory only.
    """

    name = "cost_cap"
    poll_interval: float = 30.0

    def __init__(
        self,
        max_usd: float,
        window: str = "30d_rolling",
        mode: EnforcementMode = EnforcementMode.strict,
        fail_closed: bool = True,
        scope_to_models: Optional[list[str]] = None,
        project_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_usd = max_usd
        self.window = window
        self.mode = mode
        self.fail_closed = fail_closed
        self.scope_to_models = scope_to_models
        self._project_id = (
            project_id  # used by poll(); falls back to ctx.project_id via bind_context
        )

    def bind_context(self, ctx: PolicyContext) -> None:
        # Adopt the ambient project so the background poller can scope get_state()
        # when the policy was constructed without an explicit project_id.
        if not self._project_id:
            self._project_id = ctx.project_id

    def _estimate_reserved_usd(self, parsed: ParsedRequest) -> float:
        info = get_model_info(parsed.model)
        max_out = parsed.max_tokens or (info.max_output_tokens if info else 4096)
        return estimate_cost(parsed.model, parsed.estimated_input_tokens, max_out)[
            "total_cost"
        ]

    def pre(
        self, parsed: ParsedRequest, ctx: PolicyContext, deps: PolicyDeps
    ) -> PolicyDecision:
        if self.scope_to_models and parsed.model not in self.scope_to_models:
            return PolicyDecision.allow(self.name, Phase.pre)

        reserved_usd = self._estimate_reserved_usd(parsed)

        if self.mode is EnforcementMode.strict:
            try:
                result = deps.api.reserve(
                    call_id=ctx.call_id,
                    project_id=ctx.project_id,
                    reserved_usd=reserved_usd,
                    max_usd=self.max_usd,
                    window=self.window,
                )
                if not result.admitted:
                    return PolicyDecision.block(
                        self.name,
                        Phase.pre,
                        reason=f"Cost cap ${self.max_usd:.2f} reached (spend=${result.current_spend_usd:.4f})",
                        state={
                            "reserved_usd": 0.0
                        },  # nothing reserved; release is a no-op
                    )
                return PolicyDecision.allow(
                    self.name, Phase.pre, state={"reserved_usd": reserved_usd}
                )
            except Exception as exc:
                if self.fail_closed:
                    return PolicyDecision.block(
                        self.name, Phase.pre, reason=f"backend error: {exc}"
                    )
                return PolicyDecision.allow(self.name, Phase.pre)

        else:  # non_strict — local counter, advisory
            with self._lock:
                spend = self.data_map.get(self.window, 0.0)
            if spend + reserved_usd > self.max_usd:
                return PolicyDecision.block(
                    self.name,
                    Phase.pre,
                    reason=f"Cost cap ${self.max_usd:.2f} reached (advisory, spend=${spend:.4f})",
                    state={"reserved_usd": 0.0},
                )
            return PolicyDecision.allow(
                self.name, Phase.pre, state={"reserved_usd": reserved_usd}
            )

    def post(
        self,
        resp: ParsedResponse,
        ctx: PolicyContext,
        decision: PolicyDecision,
        deps: PolicyDeps,
    ) -> PolicyDecision:
        actual_usd = resp.cost_usd
        reserved_usd = decision.state.get("reserved_usd", 0.0)

        # Update local counter regardless of mode — keeps poll() sync coherent.
        # In non_strict this mirrors report_usage()'s backend increment; both
        # track the same actual and poll() re-syncs data_map from the backend,
        # so there is no double counting against the cap.
        with self._lock:
            self.data_map[self.window] = (
                self.data_map.get(self.window, 0.0) + actual_usd
            )

        try:
            if self.mode is EnforcementMode.strict:
                unconsumed = max(0.0, reserved_usd - actual_usd)
                deps.api.reconcile(ctx.call_id, ctx.project_id, unconsumed)
            else:
                deps.api.report_usage(
                    ctx.call_id, ctx.project_id, actual_usd, resp.model
                )
        except Exception:
            pass  # reporting failure is non-blocking; local counter already updated

        return PolicyDecision.allow(self.name, Phase.post)  # cost post never blocks

    def release(
        self, decision: PolicyDecision, ctx: PolicyContext, deps: PolicyDeps
    ) -> None:
        """Return reservation to the pool after a pre-block by another policy or call error."""
        reserved_usd = decision.state.get("reserved_usd", 0.0)
        if reserved_usd > 0 and self.mode is EnforcementMode.strict:
            try:
                deps.api.reconcile(ctx.call_id, ctx.project_id, reserved_usd)
            except Exception:
                pass  # TTL on backend eventually cleans up inflight entries

    def poll(self, deps: PolicyDeps) -> None:
        project_id = self._project_id or ""
        if not project_id:
            return
        try:
            state = deps.api.get_state(project_id)
            with self._lock:
                self.data_map[self.window] = state.get("spend", 0.0)
        except Exception:
            pass
