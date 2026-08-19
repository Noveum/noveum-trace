"""Client-initiated NovaSynth calls: your dialler places the PSTN call and a
Noveum-hosted synthetic persona answers.

A session has to be parked in the LiveKit room before the call arrives or the
call is rejected, and parking takes ~35s. So the platform tells you when to
dial rather than the other way round: poll the batch, dial each run the moment
it reports ``ready_to_dial``, and report a dial that never connected so its
number is freed for the next run instead of idling until the window closes.

Usage::

    from concurrent.futures import ThreadPoolExecutor
    from noveum_trace.novasynth import CallQueue

    with CallQueue(run_ids) as q:            # `with`, so undialled runs release
        for call in q.iter_calls():
            try:
                sid = my_dialer.place(
                    to=call.dial_number, variables=call.agent_variables
                )
            except DialFailed as e:
                call.report_failed(reason=str(e), code="busy")
                continue
            call.wait_until_finished(provider_call_id=sid)
        print(q.summary())

Runs are yielded one at a time. For concurrency, feed the same generator to a
pool -- ``ThreadPoolExecutor(3).map(handler, q.iter_calls())``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from noveum_trace.core.config import get_config
from noveum_trace.utils.exceptions import ConfigurationError

__all__ = ["Call", "CallQueue", "READY", "TERMINAL"]

_log = logging.getLogger(__name__)

# The only status you may dial on. Dialling earlier is not matched to a run.
READY = "ready_to_dial"
TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})
# Call is already up, so there is nothing for the client to do but wait.
_ACTIVE = frozenset({"in_call", "evaluating"})

_POLL_SECONDS = 3.0
_ACTIVE_POLL_SECONDS = 10.0
_TIMEOUT = 10.0


class Call:
    """One run the platform has said is safe to dial right now."""

    def __init__(self, queue: CallQueue, row: dict[str, Any], ready_at: float) -> None:
        self._queue = queue
        self.run_id: str = row["runId"]
        self.dial_number: Optional[str] = row.get("dialNumber")
        self.agent_variables: dict[str, Any] = row.get("agentVariables") or {}
        self.persona: dict[str, Any] = row.get("persona") or {}
        self.scenario: dict[str, Any] = row.get("scenario") or {}
        # Deadline is the server's duration anchored to when we first saw the
        # run go ready, never a diff of its ISO timestamps against our clock.
        self._deadline = ready_at + float(row.get("dialWindowSeconds") or 0)
        self._event_id = str(uuid.uuid4())  # idempotency key, stable across retries
        self._provider_call_id: Optional[str] = None

    @property
    def seconds_remaining(self) -> float:
        """Time left to get the call placed. Your dialler must beat this."""
        return max(0.0, self._deadline - time.monotonic())

    def wait_until_finished(self, provider_call_id: Optional[str] = None) -> str:
        """Block until the run reaches a terminal status, and return it."""
        self._provider_call_id = provider_call_id
        self._queue._done.add(self.run_id)
        while True:
            self._queue._poll()
            status = self._queue.statuses.get(self.run_id, "")
            if status in TERMINAL:
                return status
            time.sleep(
                _ACTIVE_POLL_SECONDS if status in _ACTIVE else _POLL_SECONDS,
            )

    def report_failed(
        self,
        reason: str = "",
        code: str = "unknown",
        provider_call_id: Optional[str] = None,
    ) -> None:
        """Tear the session down and free the number now.

        Never raises: this is called from inside your own ``except`` block, and
        turning a dial failure into a second exception is hostile.
        """
        self._queue._report_failed(
            self, code, reason, provider_call_id or self._provider_call_id
        )


class CallQueue:
    """Bulk-polls a batch of runs and hands each one over when it is dialable.

    Args:
        run_ids: The run ids the batch-creation call returned to you.
        batch_run_id: Optional, for log lines only.
        api_key: Defaults to ``NOVEUM_API_KEY`` via the SDK config.
        base_url: Defaults to ``NOVEUM_ENDPOINT`` via the SDK config.
        organization_slug: Sent as ``?organizationSlug=`` when set.

    Use it as a context manager. On exit, any run handed out but never dialled
    is released immediately rather than holding its number until the window
    closes -- with only two or three provisioned numbers that is the difference
    between a batch finishing and stalling.
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
        self.base_url = base_url.rstrip("/")
        self.batch_run_id = batch_run_id
        self.run_ids = list(run_ids)
        # Seeded with every run id up front, and never added to, so summary()
        # accounts for the whole batch even if a run never appears in a poll.
        self.statuses: dict[str, str] = dict.fromkeys(self.run_ids, "queued")
        self._params = (
            {"organizationSlug": organization_slug} if organization_slug else {}
        )
        self._ready_at: dict[str, float] = {}
        self._calls: dict[str, Call] = {}  # handed to the caller
        self._done: set[str] = set()  # dialled or reported

    def __enter__(self) -> CallQueue:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def iter_calls(self) -> Iterator[Call]:
        """Yield each run as it becomes dialable, until the batch is finished."""
        while True:
            for row in self._poll():
                run_id = str(row.get("runId") or "")
                if row.get("status") != READY or run_id in self._calls:
                    continue
                ready_at = self._ready_at.get(run_id)
                if ready_at is None:
                    continue  # a row for a run outside this batch
                call = Call(self, row, ready_at)
                self._calls[run_id] = call
                if call.seconds_remaining <= 0:
                    # Went ready while we were busy with an earlier call.
                    self._skip(call, "dial window closed before it was handed out")
                    continue
                yield call
            # Interval comes from the state this poll just wrote, not the last.
            outstanding = [s for s in self.statuses.values() if s not in TERMINAL]
            if not outstanding:
                return
            time.sleep(
                _ACTIVE_POLL_SECONDS
                if all(s in _ACTIVE for s in outstanding)
                else _POLL_SECONDS
            )

    def summary(self) -> dict[str, int]:
        """Last-observed status of every run in the batch, counted."""
        return dict(Counter(self.statuses.values()))

    def close(self) -> None:
        """Release every run handed out but never dialled."""
        for run_id, call in self._calls.items():
            if run_id not in self._done:
                self._skip(call, "handed out but never dialled")

    def _skip(self, call: Call, why: str) -> None:
        _log.warning("novasynth: run %s skipped — %s", call.run_id, why)
        self._report_failed(call, "abandoned", why, None)

    def _poll(self) -> list[dict[str, Any]]:
        """One bulk GET for the whole batch.

        Not retried in-method: the next tick of the loop is the retry.
        """
        outstanding = [r for r in self.run_ids if self.statuses[r] not in TERMINAL]
        if not outstanding:
            return []
        rows: Any = []
        try:
            import httpx  # lazy, so the import stays monkeypatchable in tests

            with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = client.get(
                    f"{self.base_url}/v1/novasynth/runs/bulk",
                    headers=self._headers(),
                    params={**self._params, "run_ids": ",".join(outstanding)},
                )
            if resp.status_code == 200:
                rows = resp.json()
            else:
                _log.warning("novasynth poll: HTTP %d", resp.status_code)
        except Exception as exc:
            _log.warning("novasynth poll failed: %s", exc)
        if not isinstance(rows, list):
            return []
        now = time.monotonic()
        for row in rows:
            run_id = row.get("runId")
            status = row.get("status")
            if run_id not in self.statuses or not status:
                continue
            if status == READY:
                self._ready_at.setdefault(run_id, now)
            if status == "expired" and self.statuses[run_id] != "expired":
                _log.warning("novasynth: run %s expired unfired", run_id)
            # An unrecognised status is stored as-is and is not terminal, so an
            # older SDK keeps polling a platform state it has never heard of.
            self.statuses[run_id] = status
        return rows

    def _report_failed(
        self,
        call: Call,
        code: str,
        reason: str,
        provider_call_id: Optional[str],
    ) -> None:
        self._done.add(call.run_id)
        payload: dict[str, Any] = {
            "code": code,
            "reason": reason,
            "eventId": call._event_id,
        }
        if provider_call_id:
            payload["providerCallId"] = provider_call_id
        # ponytail: no retry — an undelivered report only costs the number until
        # its window closes. Add a retry loop if that measurably stalls batches.
        try:
            import httpx

            with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = client.post(
                    f"{self.base_url}/v1/novasynth/runs/{call.run_id}/dial-failed",
                    headers=self._headers(),
                    params=self._params,
                    json=payload,
                )
            # 409 only means the dial connected and this report raced it.
            if resp.status_code >= 400 and resp.status_code != 409:
                _log.error(
                    "novasynth dial-failed rejected for %s: HTTP %d",
                    call.run_id,
                    resp.status_code,
                )
        except Exception as exc:
            _log.warning(
                "novasynth dial-failed not delivered for %s: %s", call.run_id, exc
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}
