from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from noveum_trace.guard.engine import PolicyEngine
    from noveum_trace.guard.policies.base import AbstractPolicy
    from noveum_trace.guard.poller import PolicyPoller
    from noveum_trace.guard.types import PolicyContext

_engine: Optional[PolicyEngine] = None
_context: Optional[PolicyContext] = None
_poller: Optional[PolicyPoller] = None


def set_guard(
    engine: PolicyEngine,
    context: PolicyContext,
    poller: PolicyPoller,
) -> None:
    global _engine, _context, _poller
    _engine = engine
    _context = context
    _poller = poller


def get_engine() -> Optional[PolicyEngine]:
    return _engine


def get_context() -> Optional[PolicyContext]:
    return _context


def get_poller() -> Optional[PolicyPoller]:
    return _poller


def clear() -> None:
    global _engine, _context, _poller
    _engine = None
    _context = None
    _poller = None


def attach_policy(policy: AbstractPolicy) -> None:
    """Attach a policy to the running engine and immediately refresh its state."""
    engine = get_engine()
    if engine is None:
        raise RuntimeError(
            "Guard not initialized. Call noveum_trace.init() with guard_enabled=True first."
        )
    context = get_context()
    if context is not None:
        policy.bind_context(context)
    engine.attach(policy)
    poller = get_poller()
    if poller is not None:
        poller.force_refresh()


def detach_policy(policy_name: str) -> None:
    """Remove a policy from the running engine by name."""
    engine = get_engine()
    if engine is not None:
        engine.detach(policy_name)


def refresh() -> None:
    """Force an immediate poll of all attached policies."""
    poller = get_poller()
    if poller is not None:
        poller.force_refresh()
