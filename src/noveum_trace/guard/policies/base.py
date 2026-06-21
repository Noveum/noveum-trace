from __future__ import annotations

import threading
from abc import ABC
from typing import Any, Optional

from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.types import (
    EnforcementMode,
    ParsedRequest,
    ParsedResponse,
    Phase,
    PolicyContext,
    PolicyDeps,
)


class AbstractPolicy(ABC):
    """Base class for all Guard policies.

    Subclasses implement pre/post/release/poll.  The engine calls them in
    priority order; lower priority value = runs first.

    Thread-safety: every policy owns a single _lock that guards data_map.
    Subclasses must acquire self._lock before reading/writing data_map.
    """

    name: str  # must be set as a class attribute
    mode: EnforcementMode = EnforcementMode.strict
    fail_closed: bool = True  # block on unexpected exception (safe default)
    priority: int = 100  # lower = runs first; ties broken by registration order
    poll_interval: Optional[float] = (
        None  # seconds; None means poller skips this policy
    )

    def __init__(self) -> None:
        self.data_map: dict[str, Any] = {}  # private mutable state; guard with _lock
        self._lock = threading.Lock()

    # Binding — called once at registration

    def bind_context(  # noqa: B027 - optional override hook, default no-op by design
        self, ctx: PolicyContext
    ) -> None:
        """Receive the ambient PolicyContext at registration time.

        The background poller has no per-call context, so a policy that needs
        project scope for poll() must capture it here. Called by the engine when
        the policy is attached. Default: no-op.
        """

    # Lifecycle — override as needed

    def pre(
        self,
        parsed: ParsedRequest,
        ctx: PolicyContext,
        deps: PolicyDeps,
    ) -> PolicyDecision:
        """Called before the provider request is forwarded. Block to abort."""
        return PolicyDecision.allow(self.name, Phase.pre)

    def post(
        self,
        resp: ParsedResponse,
        ctx: PolicyContext,
        decision: PolicyDecision,
        deps: PolicyDeps,
    ) -> PolicyDecision:
        """Called after a successful provider response.

        `decision` is the PolicyDecision returned by this policy's own pre().
        Its `state` dict carries reservation data (e.g. reserved_usd) from pre.
        """
        return PolicyDecision.allow(self.name, Phase.post)

    def release(  # noqa: B027 - optional override hook, default no-op by design
        self,
        decision: PolicyDecision,
        ctx: PolicyContext,
        deps: PolicyDeps,
    ) -> None:
        """Un-do pre() side-effects when the call will NOT proceed.

        Called when:
        - A later policy blocked in pre (engine rollback).
        - The forwarded call raised an exception (engine.release_all).

        Default: no-op. Policies that reserve resources must override.
        """

    def poll(self, deps: PolicyDeps) -> None:  # noqa: B027 - optional override hook
        """Background refresh of data_map from remote config.

        Called by the poller thread on poll_interval cadence.
        Default: no-op.
        """
