"""Client-initiated NovaSynth calls: your dialler places the PSTN call and a
Noveum-hosted synthetic persona answers.

Inbound runs are dormant until claimed. Claiming a run reserves one of the
organisation's Noveum numbers and parks a persona in the LiveKit room, which
takes a while (~35s), and the call is rejected if it arrives before that is
done. So the loop is: claim the run, poll it until the platform reports
``ready``, dial, then poll until the run is finished.

Usage::

    from concurrent.futures import ThreadPoolExecutor
    from noveum_trace.novasynth import CallQueue

    q = CallQueue(run_ids)
    for call in q.iter_calls():
        my_dialer.place(to=call.dial_number, variables=call.profile)
        call.wait_until_finished()
    print(q.summary())

Runs are yielded one at a time. For concurrency, feed the same generator to a
pool -- ``ThreadPoolExecutor(3).map(handler, q.iter_calls())``. The platform
answers a claim with ``waiting`` while every number is in use, so a pool wider
than the number of provisioned numbers just idles.

Endpoints (base URL is ``config.endpoint``, which already ends in ``/api``):

- ``POST /v1/novasynth/inbound/runs/{runId}/claim`` -- reserve a number and
  start arming the run.
- ``GET /v1/novasynth/inbound/runs/{runId}`` -- poll one run.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit

from noveum_trace.core.config import get_config
from noveum_trace.utils.exceptions import ConfigurationError

__all__ = ["Call", "CallQueue", "READY", "TERMINAL"]

_log = logging.getLogger(__name__)

# Statuses are the platform's client-facing ones, stored verbatim:
#   waiting -> arming -> ready -> in_progress -> completed | failed | expired
# (``cancelled`` can arrive from any state). ``ready`` is the only status you
# may dial on; a call that arrives earlier is rejected.
READY = "ready"
TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})
# Not yet claimed, or every number is in use: claim (again) after a pause.
_WAITING = "waiting"
# Call is already up, so there is nothing for the client to do but wait.
_ACTIVE = frozenset({"in_progress"})

_POLL_SECONDS = 3.0
_ACTIVE_POLL_SECONDS = 10.0
_TIMEOUT = 10.0
# Statuses no amount of polling will get past: bad key, or wrong endpoint.
_PERMANENT = frozenset({401, 403, 404})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse the platform's ISO-8601 timestamps (JS ``toISOString()``, so a
    trailing ``Z``) on Python 3.9, which ``fromisoformat`` cannot do alone."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class Call:
    """One run the platform has said is safe to dial right now."""

    def __init__(self, queue: CallQueue, run_id: str, view: dict[str, Any]) -> None:
        self._queue = queue
        self.run_id = run_id
        # iter_calls() rejects a ready run without a usable number, so this is
        # always a real one by the time a caller sees it.
        self.dial_number: str = view["phoneNumber"]
        # The run's profile as the platform holds it. Today this is the whole
        # profile, and the persona sees the same dict.
        self.profile: dict[str, Any] = view.get("profile") or {}
        self.persona_name: str = view.get("personaName") or ""
        self.scenario_name: str = view.get("scenarioName") or ""
        self.dial_window_closes_at: Optional[datetime] = _parse_iso(
            view.get("dialWindowClosesAt")
        )

    @property
    def seconds_remaining(self) -> float:
        """Time left to get the call placed. Your dialler must beat this.

        Server deadline against local wall time, so clock skew moves it by a
        few seconds either way -- fine against a window of minutes. ``inf`` if
        the platform did not report a deadline.
        """
        if self.dial_window_closes_at is None:
            return float("inf")
        return max(0.0, (self.dial_window_closes_at - _utcnow()).total_seconds())

    def wait_until_finished(self) -> str:
        """Block until the run reaches a terminal status, and return it."""
        return self._queue._wait(self.run_id, TERMINAL)["status"]


class CallQueue:
    """Claims each run in a batch and hands it over once it is dialable.

    Args:
        run_ids: The run ids the batch-creation call returned to you.
        batch_run_id: Optional, for log lines only.
        api_key: Defaults to ``NOVEUM_API_KEY`` via the SDK config.
        base_url: Defaults to ``NOVEUM_ENDPOINT`` via the SDK config.
        organization_slug: Sent as ``?organizationSlug=`` when set.
    """

    def __init__(
        self,
        run_ids: Sequence[str],
        *,
        batch_run_id: str = "",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization_slug: Optional[str] = None,
    ) -> None:
        if api_key is None or base_url is None:
            config = get_config()
            api_key = api_key or config.api_key
            base_url = base_url or config.endpoint
        if not api_key:
            # No silent stub: a fake batch would show 30 imaginary calls
            # "succeeding" while the customer's dialler did nothing.
            raise ConfigurationError(
                "NovaSynth needs an API key — pass api_key= or set NOVEUM_API_KEY."
            )
        self.api_key = api_key
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ConfigurationError(
                f"NovaSynth needs an http(s) endpoint with a host, got {base_url!r}."
            )
        self.base_url = base_url.rstrip("/")
        self.batch_run_id = batch_run_id
        self.run_ids = list(run_ids)
        # Seeded with every run id up front, and never added to, so summary()
        # accounts for the whole batch even if a run is never reached.
        self.statuses: dict[str, str] = dict.fromkeys(self.run_ids, _WAITING)
        self._params = (
            {"organizationSlug": organization_slug} if organization_slug else {}
        )
        # One idempotency key per run, stable across claim retries, so a
        # resent claim is answered with the run it already armed.
        self._claim_keys: dict[str, str] = {}

    def iter_calls(self) -> Iterator[Call]:
        """Claim each run in turn and yield it once it is dialable."""
        for run_id in self.run_ids:
            view = self._arm(run_id)
            status = view.get("status")
            if status != READY:
                _log.warning(
                    "novasynth: run %s skipped — finished as %r before it was "
                    "handed out",
                    run_id,
                    status,
                )
                continue
            number = view.get("phoneNumber")
            if not isinstance(number, str) or not number:
                # Never hand a caller's dialler a number it cannot dial.
                _log.warning(
                    "novasynth: run %s is ready with no usable phoneNumber (%r) "
                    "— skipped",
                    run_id,
                    number,
                )
                continue
            yield Call(self, run_id, view)

    def summary(self) -> dict[str, int]:
        """Last-observed status of every run in the batch, counted."""
        return dict(Counter(self.statuses.values()))

    def _arm(self, run_id: str) -> dict[str, Any]:
        """Claim the run, then poll until it is ``ready`` or finished.

        ``waiting`` means the claim did not take -- every number is in use, or
        the run was reset to dormant -- so it is claimed again after the pause
        the platform asked for. Any other non-terminal status is polled.
        """
        while True:
            view = self._claim(run_id)
            while view.get("status") not in (READY, _WAITING) and (
                view.get("status") not in TERMINAL
            ):
                time.sleep(self._retry_after(view))
                view = self._get(run_id)
            if view.get("status") != _WAITING:
                return view
            _log.info(
                "novasynth: run %s not armed (%s) — retrying claim",
                run_id,
                view.get("reason") or "waiting",
            )
            time.sleep(self._retry_after(view))

    def _wait(self, run_id: str, until: frozenset[str]) -> dict[str, Any]:
        """Poll one run until its status is in ``until``; return the view."""
        while True:
            view = self._get(run_id)
            status = view.get("status")
            if status in until:
                return view
            time.sleep(
                _ACTIVE_POLL_SECONDS if status in _ACTIVE else self._retry_after(view),
            )

    @staticmethod
    def _retry_after(view: dict[str, Any]) -> float:
        retry_ms = view.get("retryAfterMs")
        if isinstance(retry_ms, (int, float)) and retry_ms > 0:
            return retry_ms / 1000.0
        return _POLL_SECONDS

    def _claim(self, run_id: str) -> dict[str, Any]:
        key = self._claim_keys.setdefault(run_id, str(uuid.uuid4()))
        return self._request(
            run_id, "POST", f"/runs/{run_id}/claim", {"idempotencyKey": key}
        )

    def _get(self, run_id: str) -> dict[str, Any]:
        return self._request(run_id, "GET", f"/runs/{run_id}")

    def _request(
        self,
        run_id: str,
        method: str,
        path: str,
        json: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """One request; records the run's status from the response.

        Not retried in-method: the next tick of the caller's loop is the retry,
        and an empty dict (no ``status``) is what a failed request returns.
        """
        url = f"{self.base_url}/v1/novasynth/inbound{path}"
        try:
            import httpx  # lazy, so the import stays monkeypatchable in tests

            with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
                if method == "POST":
                    resp = client.post(
                        url, headers=self._headers(), params=self._params, json=json
                    )
                else:
                    resp = client.get(url, headers=self._headers(), params=self._params)
        except Exception as exc:
            _log.warning("novasynth %s %s failed: %s", method, path, exc)
            return {}

        if resp.status_code in _PERMANENT:
            # Polling cannot fix any of these, and the caller's loop would
            # otherwise spin on them forever.
            hint = (
                "the run id is unknown, the route may not be deployed yet, or "
                "base_url is wrong"
                if resp.status_code == 404
                else "check api_key"
            )
            raise ConfigurationError(
                f"NovaSynth {method} {url} rejected with HTTP "
                f"{resp.status_code} — {hint}."
            )
        if resp.status_code == 400:
            # The platform refuses the run itself (not inbound, no endpoint):
            # no later request changes that.
            raise ConfigurationError(
                f"NovaSynth {method} {url} rejected with HTTP 400 — "
                f"{_error_message(resp)}"
            )
        if resp.status_code != 200:
            _log.warning("novasynth %s %s: HTTP %d", method, path, resp.status_code)
            return {}

        try:
            view = resp.json()
        except Exception as exc:
            _log.warning("novasynth %s %s: bad JSON: %s", method, path, exc)
            return {}
        if not isinstance(view, dict):
            return {}
        status = view.get("status")
        if isinstance(status, str) and status:
            if status == "expired" and self.statuses.get(run_id) != "expired":
                _log.warning("novasynth: run %s expired undialled", run_id)
            # An unrecognised status is stored as-is and is not terminal, so an
            # older SDK keeps polling a platform state it has never heard of.
            self.statuses[run_id] = status
        return view

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}


def _error_message(resp: Any) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("message"):
            return str(body["message"])
    except Exception:
        pass
    text = getattr(resp, "text", "")
    return str(text) if text else "no detail"
