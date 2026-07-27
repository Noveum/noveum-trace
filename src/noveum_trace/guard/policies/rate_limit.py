from __future__ import annotations

import logging
from typing import Any, Optional

from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.policies.base import AbstractPolicy
from noveum_trace.guard.poller import register_policy_type
from noveum_trace.guard.types import (
    ParsedRequest,
    ParsedResponse,
    Phase,
    PolicyContext,
    PolicyDeps,
)

_log = logging.getLogger(__name__)

# Backend rate-limit periods (packages/api/.../guardrails/schemas.ts).
_PERIODS = ("1m", "1h", "1d")


class RateLimitPolicy(AbstractPolicy):
    """Block calls once a per-window request or token count is reached.

    Unlike CostCapPolicy there is no strict/reserve mode here: the backend has
    no atomic reserve endpoint for request or token counts (same gap
    documented on ``HttpGuardAPIClient`` for cost), so this policy always
    follows the shared read-then-check model — pre() compares against counts
    last read from GET /policies/state (refreshed every ``poll_interval``
    seconds), and post() pushes the call's actual usage via
    ``deps.api.report_usage`` so the backend's next snapshot reflects it.
    Overshoot within one process is bounded by the poll interval, matching
    CostCapPolicy's shared/non-strict tradeoff.

    ``windows`` mirrors the backend RATE_LIMIT config shape: a list of
    ``{"period": "1m"|"1h"|"1d", "maxRequests": int|None, "maxTokens": int|None}``.
    Either threshold may be omitted/None to leave that dimension unenforced.
    """

    name = "rate_limit"
    blocked_by = "RATE_LIMIT"
    poll_interval: float = 30.0

    def __init__(
        self,
        windows: Optional[list[dict[str, Any]]] = None,
        fail_closed: bool = True,
        project_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        policy_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.windows = list(windows) if windows else []
        self.fail_closed = fail_closed
        self._project_id = project_id
        self._organization_id = organization_id
        self._policy_id = policy_id

    def bind_context(self, ctx: PolicyContext) -> None:
        # Adopt the ambient org/project so the background poller can scope
        # get_state() when the policy was constructed without explicit IDs.
        if not self._organization_id and not self._project_id:
            self._organization_id = ctx.organization_id
            self._project_id = ctx.project_id
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
        return ctx.organization_id or ctx.project_id

    def _stored_scope_id(self) -> str:
        """Scope key for poll(), which has no per-call ctx."""
        with self._lock:
            return self._organization_id or self._project_id or ""

    def pre(
        self, parsed: ParsedRequest, ctx: PolicyContext, deps: PolicyDeps
    ) -> PolicyDecision:
        scope_id = self._scope_id(ctx)
        with self._lock:
            windows = list(self.windows)
            counts = dict(self.data_map)
            policy_id = self._policy_id
        block_state = {
            "scope_id": scope_id,
            "blocked_by": self.blocked_by,
            "policy_id": policy_id,
        }

        for w in windows:
            period = w.get("period")
            if period not in _PERIODS:
                continue
            max_requests = w.get("maxRequests")
            max_tokens = w.get("maxTokens")
            requests = counts.get(f"requests_{period}", 0)
            tokens = counts.get(f"tokens_{period}", 0)
            # ``is not None`` so a configured limit of 0 (block everything) is
            # enforced; only an omitted/None threshold is left unenforced.
            if max_requests is not None and requests >= max_requests:
                return PolicyDecision.block(
                    self.name,
                    Phase.pre,
                    reason=(
                        f"Rate limit reached: {requests}/{max_requests} "
                        f"requests per {period}"
                    ),
                    state=dict(block_state),
                )
            if max_tokens is not None and tokens >= max_tokens:
                return PolicyDecision.block(
                    self.name,
                    Phase.pre,
                    reason=(
                        f"Rate limit reached: {tokens}/{max_tokens} tokens per {period}"
                    ),
                    state=dict(block_state),
                )

        return PolicyDecision.allow(self.name, Phase.pre, state={"scope_id": scope_id})

    def post(
        self,
        resp: ParsedResponse,
        ctx: PolicyContext,
        decision: PolicyDecision,
        deps: PolicyDeps,
    ) -> PolicyDecision:
        scope_id: str = decision.state.get("scope_id") or self._scope_id(ctx)
        total_tokens = resp.input_tokens + resp.output_tokens

        # Update local counters immediately so concurrent calls in this same
        # process see this call's impact before the next poll() re-syncs from
        # the backend (mirrors CostCapPolicy's non-strict post()).
        with self._lock:
            for period in _PERIODS:
                self.data_map[f"requests_{period}"] = (
                    self.data_map.get(f"requests_{period}", 0) + 1
                )
                self.data_map[f"tokens_{period}"] = (
                    self.data_map.get(f"tokens_{period}", 0) + total_tokens
                )

        try:
            # Idempotent by call_id on the backend (and on the in-memory stub),
            # so this is safe even when a CostCapPolicy on the same call also
            # reports the identical usage.
            deps.api.report_usage(
                ctx.call_id,
                scope_id,
                resp.cost_usd,
                resp.model,
                resp.input_tokens,
                resp.output_tokens,
            )
        except Exception:
            pass  # reporting failure is non-blocking; local counters already updated

        return PolicyDecision.allow(self.name, Phase.post)  # rate post never blocks

    def update_params(self, config: dict[str, Any]) -> None:
        """Apply parameter updates received from the backend.

        Recognised keys (all optional):
            ``windows``         — list of {period, maxRequests, maxTokens}
            ``fail_closed``     — bool; whether to block on unexpected exception
            ``organization_id`` — switch or set org-level scoping
            ``project_id``      — switch or set project-level scoping
            ``policy_id``       — backend policy id, sent on BLOCKED events
        """
        with self._lock:
            if "windows" in config:
                self.windows = list(config["windows"] or [])
            if "fail_closed" in config:
                self.fail_closed = bool(config["fail_closed"])
            if "organization_id" in config:
                self._organization_id = config["organization_id"] or None
            if "project_id" in config:
                self._project_id = config["project_id"] or None
            if "policy_id" in config:
                self._policy_id = config["policy_id"] or None

    def poll(self, deps: PolicyDeps) -> None:
        scope_id = self._stored_scope_id()
        if not scope_id:
            return
        try:
            state = deps.api.get_state(scope_id)
            rate = state.get("rate") or {}
            with self._lock:
                self.data_map.update(rate)
        except Exception:
            pass


# Register so the backend poller can instantiate this policy by type name.
register_policy_type("rate_limit", RateLimitPolicy)
