from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.policies.base import AbstractPolicy
from noveum_trace.guard.types import (
    ParsedRequest,
    ParsedResponse,
    Phase,
    PolicyContext,
    PolicyDeps,
)

_log = logging.getLogger(__name__)


class PolicyEngine:
    """Thin orchestrator. All enforcement logic lives in policies.

    Invariants:
    - Pre-block → rollback every earlier policy via release().
    - Post → all post() hooks run before any block is surfaced.
    """

    def __init__(
        self,
        api_client: GuardAPIClient,
        *,
        fail_open_on_backend_unavailable: bool = False,
    ) -> None:
        self._api_client = api_client
        self._policies: list[AbstractPolicy] = []
        self._lock = threading.Lock()
        # Set by PolicyPoller on a backend fetch failure (vs. legitimately zero policies).
        self._backend_unavailable = threading.Event()
        # Fail closed by default when backend is unreachable; set True to keep
        # enforcing last-known policies instead (availability over strict enforcement).
        self._fail_open_on_backend_unavailable = fail_open_on_backend_unavailable

    # Policy registration

    def attach(self, policy: AbstractPolicy) -> None:
        with self._lock:
            self._policies = sorted([*self._policies, policy], key=lambda p: p.priority)

    def detach(self, policy_name: str) -> None:
        with self._lock:
            self._policies = [p for p in self._policies if p.name != policy_name]

    @property
    def policies(self) -> list[AbstractPolicy]:
        with self._lock:
            return list(self._policies)

    # Backend availability

    def set_backend_unavailable(self, unavailable: bool) -> None:
        """Toggle the control-plane-unreachable flag (set by PolicyPoller)."""
        if unavailable:
            # Log once per transition (not per call) so the operational impact
            # is loud without spamming the hot path.
            if not self._backend_unavailable.is_set():
                if self._fail_open_on_backend_unavailable:
                    _log.warning(
                        "NovaGuard control plane unreachable; failing OPEN — "
                        "continuing to enforce last-known policies "
                        "(guard_fail_open_on_backend_unavailable=True)."
                    )
                else:
                    _log.warning(
                        "NovaGuard control plane unreachable; failing CLOSED — "
                        "blocking all guarded calls until policy sync recovers."
                    )
            self._backend_unavailable.set()
        else:
            if self._backend_unavailable.is_set():
                _log.info(
                    "NovaGuard control plane reachable again; policy sync recovered."
                )
            self._backend_unavailable.clear()

    def is_backend_unavailable(self) -> bool:
        return self._backend_unavailable.is_set()

    def has_post_blocking_policies(self) -> bool:
        """True when any attached policy can return a blocking post() decision.

        Used by the transport to decide whether a streaming response must be
        buffered in full before any bytes reach the caller.
        """
        return any(p.can_block_post for p in self.policies)

    # Call lifecycle

    def pre_call(
        self,
        parsed: ParsedRequest,
        ctx: PolicyContext,
    ) -> tuple[Optional[PolicyDecision], list[tuple[AbstractPolicy, PolicyDecision]]]:
        """Run pre() on all policies in priority order.

        Returns (block_decision | None, ran_pairs).
        On block: rollback all previously ran policies; caller must NOT forward the request.
        """
        if (
            self._backend_unavailable.is_set()
            and not self._fail_open_on_backend_unavailable
        ):
            # Control plane unreachable and configured to fail closed: block
            # rather than silently keep enforcing stale (or absent) policy
            # state. No policies ran, so there is nothing to release. When
            # fail_open_on_backend_unavailable is set we skip this and fall
            # through to enforce the last-known attached policies instead.
            return (
                PolicyDecision.block(
                    "_guard_control_plane",
                    Phase.pre,
                    reason=(
                        "NovaGuard control plane unreachable; failing closed "
                        "until policy sync recovers"
                    ),
                ),
                [],
            )

        deps = PolicyDeps(api=self._api_client)
        ran: list[tuple[AbstractPolicy, PolicyDecision]] = []

        for policy in self.policies:
            decision = self._safe_invoke(policy, policy.pre, parsed, ctx, deps)
            ran.append((policy, decision))
            if decision.is_blocking:
                # Include the blocking policy itself: if it called reserve() before
                # throwing (causing _safe_invoke to synthesize a block), its inflight
                # entry must be cleaned up even though state may carry reserved_usd=0.
                for p, prev_d in ran:
                    self._safe_invoke(p, p.release, prev_d, ctx, deps)
                return decision, ran

        return None, ran

    def post_call(
        self,
        resp: ParsedResponse,
        ctx: PolicyContext,
        ran: list[tuple[AbstractPolicy, PolicyDecision]],
    ) -> Optional[PolicyDecision]:
        """Run post() on every policy that fired pre, then surface the first block.

        All post() calls execute even when one blocks — ensures reconcile/spend-update
        always runs regardless of a later policy's verdict.
        """
        deps = PolicyDeps(api=self._api_client)
        first_block: Optional[PolicyDecision] = None

        for policy, pre_decision in ran:
            d = self._safe_invoke(
                policy, policy.post, resp, ctx, pre_decision, deps, phase=Phase.post
            )
            if d.is_blocking and first_block is None:
                first_block = d

        return first_block

    def release_all(
        self,
        ctx: PolicyContext,
        ran: list[tuple[AbstractPolicy, PolicyDecision]],
    ) -> None:
        """Called when the LLM call errored after forwarding; un-reserves all policies."""
        deps = PolicyDeps(api=self._api_client)
        for policy, pre_decision in ran:
            self._safe_invoke(policy, policy.release, pre_decision, ctx, deps)

    def poll_all(self) -> None:
        """Trigger poll() on every policy (called by the poller thread)."""
        deps = PolicyDeps(api=self._api_client)
        for policy in self.policies:
            try:
                policy.poll(deps)
            except Exception:
                pass

    # Internal

    def _safe_invoke(
        self,
        policy: AbstractPolicy,
        fn: Callable[..., Any],
        *args: object,
        phase: Phase = Phase.pre,
    ) -> PolicyDecision:
        """Wrap every policy call. Unexpected exceptions respect fail_closed."""
        try:
            return fn(*args)
        except Exception as exc:
            if policy.fail_closed:
                return PolicyDecision.block(
                    policy.name, phase, reason=f"exception: {exc}"
                )
            return PolicyDecision.allow(policy.name, phase)
