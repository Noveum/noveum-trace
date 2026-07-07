from __future__ import annotations

from noveum_trace.utils.exceptions import NoveumTraceError

__all__ = ["NoveumGuardBlocked", "GuardBackendUnavailable"]


class NoveumGuardBlocked(NoveumTraceError):
    """Raised when a Guard policy blocks a call.

    Attributes:
        policy_name: Name of the policy that blocked the call.
        reason: Human-readable reason for the block.
        decision: The ``PolicyDecision`` instance that triggered the block.
    """

    def __init__(self, policy_name: str, reason: str, decision: object) -> None:
        self.policy_name = policy_name
        self.reason = reason
        self.decision = decision
        super().__init__(f"Blocked by {policy_name}: {reason}")


class GuardBackendUnavailable(NoveumTraceError):
    """Raised when the Guard control plane cannot be reached.

    Distinct from "not configured" (no API key — legitimate stub/test mode):
    this means an API key IS configured but the backend request failed
    (network error, non-2xx status, or an unparsable response).
    """
