from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from noveum_trace.guard.api_client import GuardAPIClient, ReservationResult
from noveum_trace.guard.exceptions import GuardBackendUnavailable

_log = logging.getLogger(__name__)

# Backend cost windows (packages/api/.../guardrails/schemas.ts).
_DEFAULT_WINDOW = "30d_rolling"

# Map backend PolicyType enum (uppercase) → SDK policy registry key (lowercase).
_TYPE_MAP = {"COST_CAP": "cost_cap", "RATE_LIMIT": "rate_limit"}


class HttpGuardAPIClient(GuardAPIClient):
    """Real HTTP client for the Noveum Guard backend.

    Spend is server-authoritative: every worker pushes each call's cost to the
    shared ``/policies/usage`` endpoint and reads accumulated spend back from
    ``/policies/state``. This is what makes a cost cap hold across processes —
    the in-memory ``GuardAPIClient`` only shares state within one process.

    There is no atomic reserve endpoint, so ``reserve``/``reconcile`` are not
    supported; ``supports_reservation`` is False and CostCapPolicy degrades to
    the shared read-then-check model (bounded overshoot ≈ one poll window).

    Usage pushes are batched on a background daemon thread so the hot path never
    blocks on the network. Retries are safe because the backend dedups on
    ``eventId`` (we send the per-call ``call_id``).
    """

    supports_reservation = False

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.noveum.ai",
        organization_slug: Optional[str] = None,
        flush_interval: float = 5.0,
        batch_max: int = 100,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        self._organization_slug = organization_slug
        self._flush_interval = flush_interval
        self._batch_max = batch_max
        self._timeout = timeout
        # (scope_id, event dict) pairs awaiting push, guarded by _queue_lock.
        self._queue: list[tuple[str, dict[str, Any]]] = []
        self._queue_lock = threading.Lock()
        self._stop = threading.Event()
        # Set to wake the worker for an early flush (batch full / shutdown) so the
        # caller never blocks on the HTTP request itself.
        self._wake = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None

    # HTTP helpers

    def _query(self) -> dict[str, str]:
        return (
            {"organizationSlug": self._organization_slug}
            if self._organization_slug
            else {}
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    # Spend read — GET /policies/state

    def get_state(
        self, project_id: str, window: Optional[str] = None
    ) -> dict[str, Any]:
        """Read shared spend for ``window`` from the backend live-state endpoint."""
        window = window or _DEFAULT_WINDOW
        url = f"{self.base_url}/v1/projects/{project_id}/policies/state"
        try:
            import httpx

            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                resp = client.get(url, headers=self._headers(), params=self._query())
            if resp.status_code == 200:
                cost = resp.json().get("cost", {}) or {}
                return {"spend": float(cost.get(window, 0.0) or 0.0)}
            raise GuardBackendUnavailable(
                f"get_state: backend returned {resp.status_code} "
                f"for project {project_id!r}"
            )
        except GuardBackendUnavailable:
            raise
        except Exception as exc:
            raise GuardBackendUnavailable(
                f"get_state: request failed for project {project_id!r} — {exc}"
            ) from exc

    # Spend write — enqueue for batched POST /policies/usage

    def report_usage(
        self,
        call_id: str,
        project_id: str,
        actual_usd: float,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Queue this call's actual usage for a batched push to the backend."""
        if actual_usd < 0:
            return
        event = {
            "model": model or "unknown",
            "inputTokens": max(0, int(input_tokens)),
            "outputTokens": max(0, int(output_tokens)),
            "costUsd": float(actual_usd),
            "requestCount": 1,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "eventId": call_id,  # idempotency key — backend dedups on retry
        }
        with self._queue_lock:
            self._queue.append((project_id, event))
            full = len(self._queue) >= self._batch_max
        self._ensure_thread()
        if full:
            # Wake the worker for a prompt flush; the POST runs on its thread so
            # the caller returns without waiting on the network.
            self._wake.set()

    # reserve/reconcile have no backend equivalent; CostCapPolicy must not call
    # them (it checks supports_reservation first). Guard against wiring bugs.

    def reserve(
        self,
        call_id: str,
        project_id: str,
        reserved_usd: float,
        max_usd: float,
        window: str = _DEFAULT_WINDOW,
    ) -> ReservationResult:
        raise NotImplementedError(
            "HttpGuardAPIClient does not support reserve(); the backend has no "
            "atomic reserve endpoint. Use the shared read-then-check path."
        )

    def reconcile(self, call_id: str, project_id: str, unconsumed_usd: float) -> None:
        raise NotImplementedError("HttpGuardAPIClient does not support reconcile().")

    # Policy definitions — GET /policies/effective (merged org + project set)

    def fetch_remote_policies(self, project_id: str) -> list[dict[str, Any]]:
        """Fetch the merged, enabled policy set and map it to the SDK's shape."""
        url = f"{self.base_url}/v1/projects/{project_id}/policies/effective"
        try:
            import httpx

            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                resp = client.get(url, headers=self._headers(), params=self._query())
            if resp.status_code == 200:
                raw = resp.json()
                policies = raw if isinstance(raw, list) else raw.get("policies", [])
                return [
                    mapped
                    for p in policies
                    if (mapped := _normalize_policy(p)) is not None
                ]
            raise GuardBackendUnavailable(
                f"fetch_remote_policies: backend returned {resp.status_code} "
                f"for project {project_id!r}"
            )
        except GuardBackendUnavailable:
            raise
        except Exception as exc:
            raise GuardBackendUnavailable(
                f"fetch_remote_policies: request failed for project "
                f"{project_id!r} — {exc}"
            ) from exc

    # Background flush

    def _ensure_thread(self) -> None:
        if self._flush_thread and self._flush_thread.is_alive():
            return
        with self._queue_lock:
            if self._flush_thread and self._flush_thread.is_alive():
                return
            self._stop.clear()
            self._flush_thread = threading.Thread(
                target=self._run, daemon=True, name="noveum-guard-usage-flush"
            )
            self._flush_thread.start()

    def _run(self) -> None:
        # Flush on the interval, or earlier when woken by a full batch / shutdown.
        while not self._stop.is_set():
            self._wake.wait(timeout=self._flush_interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self._flush_once()
            except Exception:  # never let the flush thread die
                pass

    def _flush_once(self) -> None:
        with self._queue_lock:
            if not self._queue:
                return
            batch = self._queue[: self._batch_max]
            del self._queue[: len(batch)]

        # Group events by scope so each POST targets one project.
        by_scope: dict[str, list[dict[str, Any]]] = {}
        for scope_id, event in batch:
            by_scope.setdefault(scope_id, []).append(event)

        for scope_id, events in by_scope.items():
            self._post_usage(scope_id, events)

    def _post_usage(self, scope_id: str, events: list[dict[str, Any]]) -> None:
        url = f"{self.base_url}/v1/projects/{scope_id}/policies/usage"
        try:
            import httpx

            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                client.post(
                    url,
                    headers=self._headers(),
                    params=self._query(),
                    json=events,
                )
        except Exception as exc:
            # Best-effort: a dropped batch under-counts spend slightly; it never
            # over-counts (idempotent) and the backend stays authoritative.
            _log.debug(
                "usage push failed for project %r (%d events) — %s",
                scope_id,
                len(events),
                exc,
            )

    def close(self) -> None:
        """Stop the flush thread and drain any queued usage events."""
        self._stop.set()
        self._wake.set()  # wake the worker so it exits without waiting the interval
        if self._flush_thread:
            self._flush_thread.join(timeout=self._flush_interval * 2)
        self._flush_once()


def _normalize_policy(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Map a backend policy row to the flat, snake_case dict the poller expects.

    Backend shape: ``{type: "COST_CAP", name, enabled, failClosed,
    config: {window, maxUsd, softUsd, action}}``. The poller wants a flat dict
    keyed by ``type`` (lowercase) plus constructor params (``max_usd`` …).
    Returns None for disabled or unmappable rows so they are skipped.
    """
    if not isinstance(raw, dict) or not raw.get("enabled", True):
        return None
    sdk_type = _TYPE_MAP.get(raw.get("type", ""))
    if sdk_type is None:
        return None
    config = raw.get("config") or {}
    mapped: dict[str, Any] = {
        "type": sdk_type,
        "name": raw.get("name", sdk_type),
        "fail_closed": raw.get("failClosed", True),
    }
    if "maxUsd" in config:
        mapped["max_usd"] = config["maxUsd"]
    if "window" in config:
        mapped["window"] = config["window"]
    return mapped
