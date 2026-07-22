from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

from noveum_trace.guard.exceptions import GuardBackendUnavailable


@dataclass
class ReservationResult:
    admitted: bool
    current_spend_usd: float


class GuardAPIClient:
    """In-memory backend for the Noveum Guard, and base class for HttpGuardAPIClient.

    All state is per-process. Correct for single-process use and tests, and is
    the only backend that supports atomic reserve/reconcile — multi-process
    deployments should use ``HttpGuardAPIClient`` (guard/api_client_http.py),
    which subclasses this and overrides the network-backed methods.

    Thread-safety: a single Lock guards every mutation. The lock is held only
    for the minimal critical section so high-concurrency callers are not
    serialised longer than necessary.
    """

    # In-memory reservations are atomic within this process, so strict-mode
    # reserve/reconcile are supported. The HTTP client sets this False.
    supports_reservation: bool = True

    def __init__(
        self, api_key: str = "", base_url: str = "https://api.noveum.ai"
    ) -> None:
        self.api_key = api_key
        # Guard endpoints live under the same /api prefix as the tracing API
        # (e.g. https://api.noveum.ai/api/v1/projects/{id}/policies/effective).
        # DEFAULT_ENDPOINT already includes /api, so just trim trailing slashes.
        self.base_url = base_url.rstrip("/")
        self._lock = threading.Lock()
        # project_id → accumulated spend (USD)
        self._spend: dict[str, float] = {}
        # call_id → amount currently reserved (USD); cleared on reconcile
        self._inflight: dict[str, float] = {}
        # project_id → arbitrary policy config dict (refreshed by poll)
        self._policy_configs: dict[str, dict[str, Any]] = {}
        # project_id → {"requests_1m": int, "tokens_1m": int, ...} for RateLimitPolicy
        self._rate: dict[str, dict[str, int]] = {}
        # call_ids already folded into _spend/_rate via report_usage — guards
        # against double counting when more than one policy (e.g. CostCapPolicy
        # in shared mode + RateLimitPolicy) reports the same call's usage.
        self._reported_event_ids: set[str] = set()

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
        if reserved_usd < 0:
            raise ValueError(f"reserved_usd must be non-negative, got {reserved_usd}")
        if max_usd < 0:
            raise ValueError(f"max_usd must be non-negative, got {max_usd}")
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
        if unconsumed_usd < 0:
            raise ValueError(
                f"unconsumed_usd must be non-negative, got {unconsumed_usd}"
            )
        with self._lock:
            # Clamp to what was actually reserved so an over-refund cannot
            # push spend below the amount owed by other in-flight calls.
            inflight = self._inflight.get(call_id, 0.0)
            clamped = min(unconsumed_usd, inflight)
            current = self._spend.get(project_id, 0.0)
            self._spend[project_id] = max(0.0, current - clamped)
            self._inflight.pop(call_id, None)

    def report_usage(
        self,
        call_id: str,
        project_id: str,
        actual_usd: float,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record actual cost and rate-limit counters for a completed call.

        Non-strict cost accounting never calls reserve(), so there is nothing
        to reconcile — we simply add the actual spend. Also bumps the
        request/token counters RateLimitPolicy reads back via get_state().
        Deduped by ``call_id`` so CostCapPolicy (shared mode) and
        RateLimitPolicy reporting the same call don't double count.
        """
        if actual_usd < 0:
            raise ValueError(f"actual_usd must be non-negative, got {actual_usd}")
        with self._lock:
            if call_id in self._reported_event_ids:
                return
            self._reported_event_ids.add(call_id)
            self._spend[project_id] = self._spend.get(project_id, 0.0) + actual_usd
            rate = self._rate.setdefault(project_id, {})
            for period in ("1m", "1h", "1d"):
                rate[f"requests_{period}"] = rate.get(f"requests_{period}", 0) + 1
                rate[f"tokens_{period}"] = (
                    rate.get(f"tokens_{period}", 0) + input_tokens + output_tokens
                )

    # Policy config / polling

    def get_state(
        self, project_id: str, window: Optional[str] = None
    ) -> dict[str, Any]:
        """Spend + rate snapshot for poll(). Returns a copy to avoid lock-holding
        in the caller.

        ``window`` is accepted for interface parity with the HTTP client (which
        selects a per-window cost counter); the in-memory stub tracks one
        bucket. ``rate`` mirrors the backend's requests_*/tokens_* shape.
        """
        with self._lock:
            return {
                "spend": self._spend.get(project_id, 0.0),
                "rate": dict(self._rate.get(project_id, {})),
            }

    def get_policy_config(self, project_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._policy_configs.get(project_id, {}))

    def set_policy_config(self, project_id: str, config: dict[str, Any]) -> None:
        """Test helper / future backend push. Not part of the HTTP stub seam."""
        with self._lock:
            self._policy_configs[project_id] = dict(config)

    def fetch_remote_policies(self, project_id: str) -> list[dict[str, Any]]:
        """Fetch policy definitions from the Noveum backend.

        Makes a real HTTP GET to ``{base_url}/v1/projects/{project_id}/guard/policies``
        and returns the ``policies`` list from the response JSON.  Each item is a
        dict with at least a ``"type"`` key (e.g. ``"cost_cap"``) plus the policy's
        own parameters (e.g. ``max_usd``, ``window``).

        Returns an empty list when no API key is configured (stub / test mode)
        or when the backend responds successfully but the project has zero
        configured policies — both legitimate, silent cases.

        Raises:
            GuardBackendUnavailable: an API key IS configured but the backend
                request failed (network error, non-2xx status, or the response
                could not be parsed as JSON). The caller (``PolicyPoller``) is
                responsible for deciding how to react — this method does not
                silently degrade to "no policies" for a real failure.
        """
        if not self.api_key:
            # Running in stub / in-memory mode — return locally stored configs
            # so test helpers that call set_policy_config() still work.
            with self._lock:
                stored = self._policy_configs.get(project_id, {})
            if stored:
                # Wrap in list format matching the backend wire format
                return [dict(stored)]
            return []

        try:
            import httpx  # only imported when actually needed

            url = f"{self.base_url.rstrip('/')}/v1/projects/{project_id}/guard/policies"
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            if resp.status_code == 200:
                data = resp.json()
                policies: list[dict[str, Any]] = data.get("policies", [])
                return policies
            raise GuardBackendUnavailable(
                f"fetch_remote_policies: backend returned "
                f"{resp.status_code} for project {project_id!r}"
            )
        except GuardBackendUnavailable:
            raise
        except Exception as exc:  # network error, JSON decode error, etc.
            raise GuardBackendUnavailable(
                f"fetch_remote_policies: request failed for project "
                f"{project_id!r} — {exc}"
            ) from exc

    # Inspection (tests + debug)

    def current_spend(self, project_id: str) -> float:
        with self._lock:
            return self._spend.get(project_id, 0.0)

    def current_rate(self, project_id: str) -> dict[str, int]:
        with self._lock:
            return dict(self._rate.get(project_id, {}))

    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)

    def reset(self) -> None:
        """Wipe all state. Tests only."""
        with self._lock:
            self._spend.clear()
            self._inflight.clear()
            self._policy_configs.clear()
            self._rate.clear()
            self._reported_event_ids.clear()
