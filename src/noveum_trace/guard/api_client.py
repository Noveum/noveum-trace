from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from noveum_trace.guard.exceptions import GuardBackendUnavailable

# In-memory rate windows (seconds). Mirrors the backend periods in
# guardrails/schemas.ts; the HTTP backend does its own windowing server-side.
_PERIODS = ("1m", "1h", "1d")
_WINDOW_SECONDS = {"1m": 60.0, "1h": 3600.0, "1d": 86400.0}
_MAX_WINDOW_SECONDS = 86400.0  # 1d — the longest window bounds retention

# The only values /policies/usage accepts for blockedBy; anything else is a 400.
_BLOCKED_BY_VALUES = ("COST_CAP", "RATE_LIMIT")


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
        # project_id → list of (monotonic_ts, tokens) per reported call, so the
        # windowed request/token counts RateLimitPolicy reads only include events
        # still inside their 1m/1h/1d window. Events past 1d are evicted.
        self._rate_events: dict[str, list[tuple[float, int]]] = {}
        # call_id → monotonic_ts of the report already folded into _spend/_rate —
        # guards against double counting when more than one policy (e.g.
        # CostCapPolicy in shared mode + RateLimitPolicy) reports the same call.
        # Timestamped so stale ids can be evicted rather than growing unbounded.
        self._reported_event_ids: dict[str, float] = {}
        # project_id → blocked events recorded for inspection. Never folded into
        # _spend/_rate: a blocked call never ran, so it is not metered.
        self._blocked_events: dict[str, list[dict[str, Any]]] = {}

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
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            if call_id in self._reported_event_ids:
                return
            self._reported_event_ids[call_id] = now
            self._spend[project_id] = self._spend.get(project_id, 0.0) + actual_usd
            self._rate_events.setdefault(project_id, []).append(
                (now, input_tokens + output_tokens)
            )

    def report_blocked(
        self,
        call_id: str,
        project_id: str,
        model: str,
        blocked_by: str,
        policy_id: Optional[str] = None,
        reason: str = "",
    ) -> None:
        """Record a call the Guard stopped before it reached the provider.

        Deliberately does not touch _spend/_rate — the call never ran, so it is
        excluded from the counters policies check. ``blocked_by`` is the backend
        limit type ("COST_CAP" or "RATE_LIMIT") that tripped.
        """
        if blocked_by not in _BLOCKED_BY_VALUES:
            return
        with self._lock:
            self._blocked_events.setdefault(project_id, []).append(
                {
                    "call_id": call_id,
                    "model": model,
                    "blocked_by": blocked_by,
                    "policy_id": policy_id,
                    "reason": reason,
                }
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
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            return {
                "spend": self._spend.get(project_id, 0.0),
                "rate": self._windowed_rate(project_id, now),
            }

    def _evict_expired(self, now: float) -> None:
        """Drop rate events and reported call_ids older than the largest window.

        Caller must hold ``self._lock``. Bounds memory and keeps the windowed
        counts accurate as events age out. Duplicate protection only needs to
        outlive the longest window — a genuine retry always lands far sooner.
        """
        cutoff = now - _MAX_WINDOW_SECONDS
        for pid in list(self._rate_events):
            kept = [e for e in self._rate_events[pid] if e[0] >= cutoff]
            if kept:
                self._rate_events[pid] = kept
            else:
                del self._rate_events[pid]
        for cid in [c for c, ts in self._reported_event_ids.items() if ts < cutoff]:
            del self._reported_event_ids[cid]

    def _windowed_rate(self, project_id: str, now: float) -> dict[str, int]:
        """Request/token counts per window, counting only events inside each.

        Caller must hold ``self._lock``. Returns ``{}`` when the project has no
        live events, matching get_state()'s zero-state for a fresh project.
        """
        events = self._rate_events.get(project_id)
        if not events:
            return {}
        result: dict[str, int] = {}
        for period in _PERIODS:
            cutoff = now - _WINDOW_SECONDS[period]
            requests = 0
            tokens = 0
            for ts, tok in events:
                if ts >= cutoff:
                    requests += 1
                    tokens += tok
            result[f"requests_{period}"] = requests
            result[f"tokens_{period}"] = tokens
        return result

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
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            return self._windowed_rate(project_id, now)

    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)

    def blocked_events(self, project_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._blocked_events.get(project_id, [])]

    def reset(self) -> None:
        """Wipe all state. Tests only."""
        with self._lock:
            self._spend.clear()
            self._inflight.clear()
            self._policy_configs.clear()
            self._rate_events.clear()
            self._reported_event_ids.clear()
            self._blocked_events.clear()
