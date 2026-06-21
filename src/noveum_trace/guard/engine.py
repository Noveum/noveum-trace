from __future__ import annotations

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


class PolicyEngine:
    """Thin orchestrator. All enforcement logic lives in policies.

    Invariants:
    - Pre-block → rollback every earlier policy via release().
    - Post → all post() hooks run before any block is surfaced.
    """

    def __init__(self, api_client: GuardAPIClient) -> None:
        self._api_client = api_client
        self._policies: list[AbstractPolicy] = []
        self._lock = threading.Lock()

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
        deps = PolicyDeps(api=self._api_client)
        ran: list[tuple[AbstractPolicy, PolicyDecision]] = []

        for policy in self.policies:
            decision = self._safe_invoke(policy, policy.pre, parsed, ctx, deps)
            ran.append((policy, decision))
            if decision.is_blocking:
                for p, prev_d in ran[:-1]:
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
