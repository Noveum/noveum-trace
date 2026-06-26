from __future__ import annotations

import logging
from typing import Any, Optional

from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.policies.base import AbstractPolicy
from noveum_trace.guard.poller import register_policy_type
from noveum_trace.guard.types import (
    EnforcementMode,
    ParsedRequest,
    ParsedResponse,
    Phase,
    PolicyContext,
    PolicyDeps,
)
from noveum_trace.utils.llm_utils import estimate_cost, get_model_info

_log = logging.getLogger(__name__)


class CostCapPolicy(AbstractPolicy):
    """Block calls that would exceed a per-project or per-org USD spend cap.

    strict mode: reserves worst-case cost atomically in GuardAPIClient before
    forwarding; reconciles after. Prevents any overshoot under concurrency.

    non_strict mode: checks a local counter; allows all calls that look under-cap
    at the moment of pre(). Reports actual cost in post(). May overshoot slightly
    under high concurrency — advisory only.

    Scoping precedence (highest → lowest):
      1. explicit ``organization_id`` constructor arg → org-level cap
      2. explicit ``project_id`` constructor arg → project-level cap
      3. ambient ``PolicyContext.organization_id`` (if set) → org-level cap
      4. ambient ``PolicyContext.project_id`` → project-level cap
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
        organization_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_usd = max_usd
        self.window = window
        # Normalize to enum member so callers passing a raw string (e.g. "strict")
        # still produce a proper EnforcementMode instance; == comparisons below are
        # safe for str-Enum, but is-checks are not when mode is a plain str.
        self.mode = (
            EnforcementMode(mode) if not isinstance(mode, EnforcementMode) else mode
        )
        self.fail_closed = fail_closed
        self.scope_to_models = scope_to_models
        self._project_id = project_id
        self._organization_id = organization_id

    def bind_context(self, ctx: PolicyContext) -> None:
        # Adopt the ambient org/project so the background poller can scope
        # get_state() when the policy was constructed without explicit IDs.
        # Org-level takes precedence over project-level when both are present.
        if not self._organization_id and not self._project_id:
            self._organization_id = ctx.organization_id
            self._project_id = ctx.project_id
        elif not self._organization_id and ctx.organization_id:
            # explicit project_id set but no org — leave project scoping as-is
            pass
        if not self._project_id:
            self._project_id = ctx.project_id

    def _scope_id(self, ctx: PolicyContext) -> str:
        """Return the scope key to use for API calls: org > project."""
        with self._lock:
            org = self._organization_id
            proj = self._project_id
        if org:
            return org
        if proj:
            return proj
        # fallback to per-call context (should always be set after bind_context)
        return ctx.organization_id or ctx.project_id

    def _stored_scope_id(self) -> str:
        """Scope key for poll(), which has no per-call ctx."""
        with self._lock:
            return self._organization_id or self._project_id or ""

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
        # Snapshot scope and mode at pre() time so post()/release() use the same
        # values even if update_params() fires concurrently between pre and post.
        scope_id = self._scope_id(ctx)
        with self._lock:
            mode = self.mode

        if mode == EnforcementMode.strict:
            try:
                result = deps.api.reserve(
                    call_id=ctx.call_id,
                    project_id=scope_id,
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
                            "reserved_usd": 0.0,
                            "scope_id": scope_id,
                            "mode": mode.value,
                        },  # nothing reserved; release is a no-op
                    )
                return PolicyDecision.allow(
                    self.name,
                    Phase.pre,
                    state={
                        "reserved_usd": reserved_usd,
                        "scope_id": scope_id,
                        "mode": mode.value,
                    },
                )
            except Exception as exc:
                if self.fail_closed:
                    return PolicyDecision.block(
                        self.name,
                        Phase.pre,
                        reason=f"backend error: {exc}",
                        state={
                            "reserved_usd": 0.0,
                            "scope_id": scope_id,
                            "mode": mode.value,
                        },
                    )
                return PolicyDecision.allow(
                    self.name,
                    Phase.pre,
                    state={
                        "reserved_usd": 0.0,
                        "scope_id": scope_id,
                        "mode": mode.value,
                    },
                )

        else:  # non_strict — local counter, advisory
            with self._lock:
                if self.window not in self.data_map:
                    _log.error(
                        "CostCapPolicy: no spend value for window %r — poll() has not"
                        " run yet or get_state() failed; treating spend as 0.0",
                        self.window,
                    )
                spend = self.data_map.get(self.window, 0.0)
            if spend + reserved_usd > self.max_usd:
                return PolicyDecision.block(
                    self.name,
                    Phase.pre,
                    reason=f"Cost cap ${self.max_usd:.2f} reached (advisory, spend=${spend:.4f})",
                    state={
                        "reserved_usd": 0.0,
                        "scope_id": scope_id,
                        "mode": mode.value,
                    },
                )
            return PolicyDecision.allow(
                self.name,
                Phase.pre,
                state={
                    "reserved_usd": reserved_usd,
                    "scope_id": scope_id,
                    "mode": mode.value,
                },
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
        # Use scope/mode snapshotted at pre() time; fall back to current values for
        # decisions created before this fix (e.g. state dict without scope_id).
        scope_id: str = decision.state.get("scope_id") or self._scope_id(ctx)
        mode = (
            EnforcementMode(decision.state["mode"])
            if "mode" in decision.state
            else self.mode
        )

        # Update local counter regardless of mode — keeps poll() sync coherent.
        # In non_strict this mirrors report_usage()'s backend increment; both
        # track the same actual and poll() re-syncs data_map from the backend,
        # so there is no double counting against the cap.
        with self._lock:
            self.data_map[self.window] = (
                self.data_map.get(self.window, 0.0) + actual_usd
            )

        try:
            if mode == EnforcementMode.strict:
                if actual_usd <= reserved_usd:
                    # Return the unused headroom so the budget stays accurate.
                    deps.api.reconcile(ctx.call_id, scope_id, reserved_usd - actual_usd)
                else:
                    # Tokenizer underestimated; clear the inflight entry then charge
                    # the excess so spend never understates reality.
                    deps.api.reconcile(ctx.call_id, scope_id, 0.0)
                    deps.api.report_usage(
                        ctx.call_id,
                        scope_id,
                        actual_usd - reserved_usd,
                        resp.model,
                    )
            else:
                deps.api.report_usage(ctx.call_id, scope_id, actual_usd, resp.model)
        except Exception:
            pass  # reporting failure is non-blocking; local counter already updated

        return PolicyDecision.allow(self.name, Phase.post)  # cost post never blocks

    def release(
        self, decision: PolicyDecision, ctx: PolicyContext, deps: PolicyDeps
    ) -> None:
        """Return reservation to the pool after a pre-block by another policy or call error."""
        scope_id: str = decision.state.get("scope_id") or self._scope_id(ctx)
        mode = (
            EnforcementMode(decision.state["mode"])
            if "mode" in decision.state
            else self.mode
        )
        if mode == EnforcementMode.strict:
            reserved_usd = decision.state.get("reserved_usd", 0.0)
            try:
                # Always reconcile in strict mode — even with reserved_usd=0 this
                # pops any inflight entry that reserve() created before an exception
                # caused _safe_invoke to return a block decision with empty state.
                deps.api.reconcile(ctx.call_id, scope_id, reserved_usd)
            except Exception:
                pass  # TTL on backend eventually cleans up inflight entries

    def update_params(self, config: dict[str, Any]) -> None:
        """Apply parameter updates received from the backend.

        The backend may push a new cap or window at any time.  All field writes
        are guarded by ``self._lock`` so concurrent ``pre()`` calls always see a
        consistent snapshot.

        Recognised keys (all optional):
            ``max_usd``         — new spend cap in USD (must be non-negative)
            ``window``          — billing window string (e.g. ``"30d_rolling"``)
            ``mode``            — ``"strict"`` or ``"non_strict"``
            ``fail_closed``     — bool; whether to block on unexpected exception
            ``organization_id`` — switch or set org-level scoping
            ``project_id``      — switch or set project-level scoping
        """
        with self._lock:
            if "max_usd" in config:
                new_cap = float(config["max_usd"])
                if new_cap >= 0:
                    self.max_usd = new_cap
            if "window" in config:
                self.window = str(config["window"])
            if "mode" in config:
                try:
                    self.mode = EnforcementMode(config["mode"])
                except ValueError:
                    pass  # ignore unknown mode strings from backend
            if "fail_closed" in config:
                self.fail_closed = bool(config["fail_closed"])
            if "organization_id" in config:
                self._organization_id = config["organization_id"] or None
            if "project_id" in config:
                self._project_id = config["project_id"] or None

    def poll(self, deps: PolicyDeps) -> None:
        scope_id = self._stored_scope_id()
        if not scope_id:
            return
        try:
            state = deps.api.get_state(scope_id)
            with self._lock:
                self.data_map[self.window] = state.get("spend", 0.0)
        except Exception:
            pass


# Register so the backend poller can instantiate this policy by type name.
register_policy_type("cost_cap", CostCapPolicy)
