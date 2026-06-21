from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ReservationResult:
    admitted: bool
    current_spend_usd: float


class GuardAPIClient:
    """In-memory stub for the Noveum Guard backend.

    All state is per-process. Correct for single-process use and tests;
    multi-process deployments need the real HTTP backend (swap this file only).

    Thread-safety: a single Lock guards every mutation. The lock is held only
    for the minimal critical section so high-concurrency callers are not
    serialised longer than necessary.
    """

    def __init__(
        self, api_key: str = "", base_url: str = "https://api.noveum.ai"
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self._lock = threading.Lock()
        # project_id → accumulated spend (USD)
        self._spend: dict[str, float] = {}
        # call_id → amount currently reserved (USD); cleared on reconcile
        self._inflight: dict[str, float] = {}
        # project_id → arbitrary policy config dict (refreshed by poll)
        self._policy_configs: dict[str, dict[str, Any]] = {}

    # Core accounting

    def reserve(
        self,
        call_id: str,
        project_id: str,
        reserved_usd: float,
        max_usd: float,
        window: str = "30d_rolling",
    ) -> ReservationResult:
        """Atomic check-and-reserve.

        Admits the call only when current spend + reserved_usd ≤ max_usd.
        The comparison mirrors the Redis Lua atomic described in the design memo:
        no two threads can both see spend < cap and both increment past it.
        """
        with self._lock:
            spend = self._spend.get(project_id, 0.0)
            if spend + reserved_usd > max_usd:
                return ReservationResult(admitted=False, current_spend_usd=spend)
            new_spend = spend + reserved_usd
            self._spend[project_id] = new_spend
            self._inflight[call_id] = reserved_usd
            return ReservationResult(admitted=True, current_spend_usd=new_spend)

    def reconcile(
        self,
        call_id: str,
        project_id: str,
        unconsumed_usd: float,
    ) -> None:
        """Return unused headroom to the pool.

        Called by:
        - strict post(): unconsumed = reserved - actual (releases over-estimate)
        - strict release(): unconsumed = full reserved amount (call never happened)
        """
        with self._lock:
            current = self._spend.get(project_id, 0.0)
            self._spend[project_id] = max(0.0, current - unconsumed_usd)
            self._inflight.pop(call_id, None)

    def report_usage(
        self,
        call_id: str,
        project_id: str,
        actual_usd: float,
        model: str,
    ) -> None:
        """Record actual cost (non-strict post only).

        Non-strict never calls reserve(), so there is nothing to reconcile —
        we simply add the actual spend.
        """
        with self._lock:
            self._spend[project_id] = self._spend.get(project_id, 0.0) + actual_usd

    # Policy config / polling

    def get_state(self, project_id: str) -> dict[str, Any]:
        """Spend snapshot for poll(). Returns a copy to avoid lock-holding in caller."""
        with self._lock:
            return {"spend": self._spend.get(project_id, 0.0)}

    def get_policy_config(self, project_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._policy_configs.get(project_id, {}))

    def set_policy_config(self, project_id: str, config: dict[str, Any]) -> None:
        """Test helper / future backend push. Not part of the HTTP stub seam."""
        with self._lock:
            self._policy_configs[project_id] = dict(config)

    # Inspection (tests + debug)

    def current_spend(self, project_id: str) -> float:
        with self._lock:
            return self._spend.get(project_id, 0.0)

    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)

    def reset(self) -> None:
        """Wipe all state. Tests only."""
        with self._lock:
            self._spend.clear()
            self._inflight.clear()
            self._policy_configs.clear()
