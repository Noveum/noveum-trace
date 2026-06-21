from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from noveum_trace.guard.types import Action, BlockResponseMode, Phase


@dataclass(frozen=True)
class PolicyDecision:
    policy_name: str
    phase: Phase
    action: Action
    flagged: bool = False
    reason: str = ""
    block_response_mode: BlockResponseMode = BlockResponseMode.provider_error
    # Policies stash per-call state here (e.g. reserved_usd) so post/release
    # can act on the same values without shared mutable fields.
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        return self.action is Action.block

    @classmethod
    def allow(
        cls,
        policy_name: str,
        phase: Phase | str,
        *,
        state: Optional[dict[str, Any]] = None,
    ) -> PolicyDecision:
        return cls(
            policy_name=policy_name,
            phase=Phase(phase) if isinstance(phase, str) else phase,
            action=Action.allow,
            state=state or {},
        )

    @classmethod
    def block(
        cls,
        policy_name: str,
        phase: Phase | str,
        reason: str = "",
        *,
        state: Optional[dict[str, Any]] = None,
        mode: BlockResponseMode = BlockResponseMode.provider_error,
    ) -> PolicyDecision:
        return cls(
            policy_name=policy_name,
            phase=Phase(phase) if isinstance(phase, str) else phase,
            action=Action.block,
            reason=reason,
            block_response_mode=mode,
            state=state or {},
        )
