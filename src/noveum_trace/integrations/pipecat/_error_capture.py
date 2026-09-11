"""
Error and system-log capture mixin for NoveumTraceObserver.

Handles:
  - ErrorFrame / FatalErrorFrame → record span errors and events
  - SystemLogFrame              → record structured log events on turn span

Activated via ``NoveumPipecatTracer(capture_errors=True, capture_system_logs=False)``.

This mixin owns ``_handle_error``, superseding the minimal implementation that
previously lived in ``_turn_manager._TurnManagerMixin``.  It must appear before
``_TurnManagerMixin`` in ``NoveumTraceObserver``'s MRO so that Python's method
resolution picks this version.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from noveum_trace.core.span import SpanEvent, SpanStatus
from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase

logger = logging.getLogger(__name__)

# Only these log levels are recorded for SystemLogFrame; DEBUG/INFO are too noisy.
_LOG_LEVELS_TO_CAPTURE: frozenset[str] = frozenset({"warning", "error", "critical"})


class _ErrorCaptureMixin(_PipecatObserverMixinBase):
    """
    Handler methods for ``ErrorFrame``, ``FatalErrorFrame``, and ``SystemLogFrame``.

    State attributes used (declared in ``NoveumTraceObserver.__init__``):
        _capture_errors, _capture_system_logs,
        _active_llm_span, _active_tts_span, _current_turn_span, _trace
    """

    async def _handle_error(self, data: Any) -> None:
        """
        ``ErrorFrame`` / ``FatalErrorFrame``: record span errors and a trace event.

        When ``capture_errors=True`` (default):

        - Sets ``pipecat_span_status = "error"`` and ``pipecat_span_status_message``
          on every currently-active operation span (LLM, TTS) and the turn span.
        - **Also sets the native ``SpanStatus.ERROR``** on those spans and the trace,
          and bumps the trace ``error_count`` (D1). Without this the native
          ``status``/``error_count`` stay healthy on a Pipecat failure, so any
          dashboard/ETL query filtering on native status silently misses it.
        - Appends a ``pipecat.error`` ``SpanEvent`` with ``error.message`` and
          ``error.type`` (frame class name) to the turn span.
        - Always annotates the root trace with the error — visible in the dashboard
          even when no child span is open at the time the error fires.
        """
        if not self._capture_errors:
            return

        frame = data.frame
        error_msg = str(getattr(frame, "error", "Unknown error"))
        error_type = type(frame).__name__

        llm_operation = self._resolve_llm_operation(data, include_metrics_pending=True)
        if llm_operation is not None:
            llm_operation.error = {"message": error_msg}
            if llm_operation.phase == "metrics_pending":
                llm_operation.terminal_status = "error"
            span = llm_operation.span
            span.attributes["pipecat_span_status"] = "error"
            span.attributes["pipecat_span_status_message"] = error_msg
            self._mark_operation_span_error(span, error_msg)

        # Keep the existing TTS behavior: Pipecat does not always supply enough
        # service/context identity here to attribute a provider-level TTS error.
        if self._active_tts_span is not None:
            self._active_tts_span.attributes["pipecat_span_status"] = "error"
            self._active_tts_span.attributes["pipecat_span_status_message"] = error_msg
            self._mark_operation_span_error(self._active_tts_span, error_msg)

        if self._current_turn_span:
            self._current_turn_span.attributes["pipecat_span_status"] = "error"
            self._current_turn_span.attributes["pipecat_span_status_message"] = (
                error_msg
            )
            self._current_turn_span.set_status(SpanStatus.ERROR, error_msg)
            try:
                self._current_turn_span.events.append(
                    SpanEvent(
                        name="pipecat.error",
                        timestamp=datetime.now(timezone.utc),
                        attributes={
                            "error.message": error_msg,
                            "error.type": error_type,
                        },
                    )
                )
            except Exception:  # pylint: disable=broad-except
                pass

        if self._trace:
            self._trace.attributes["pipecat_span_status"] = "error"
            self._trace.attributes["pipecat_span_status_message"] = error_msg
            # D1: native trace status + error_count so native error filters work.
            # Pipecat emits several ErrorFrames per provider failure; dedupe by
            # message so error_count reflects failures, not frames.
            self._trace.set_status(SpanStatus.ERROR, error_msg)
            if error_msg not in self._native_error_messages:
                self._native_error_messages.add(error_msg)
                self._trace.error_count += 1
            try:
                self._trace.events.append(
                    SpanEvent(
                        name="pipecat.error",
                        timestamp=datetime.now(timezone.utc),
                        attributes={
                            "error.message": error_msg,
                            "error.type": error_type,
                        },
                    )
                )
            except Exception:  # pylint: disable=broad-except
                pass

        logger.debug("NoveumTraceObserver: recorded %s — %s", error_type, error_msg)

    @staticmethod
    def _mark_operation_span_error(span: Any, message: str) -> None:
        """Apply a late error to an active or logically-completed span."""
        if getattr(span, "is_finished", lambda: False)() is True:
            # Span.set_status intentionally ignores finished spans. A completed
            # operation retained for late correlation is still mutable until its
            # containing trace is exported.
            span.status = SpanStatus.ERROR
            span.status_message = message
            return
        span.set_status(SpanStatus.ERROR, message)

    async def _handle_system_log(self, data: Any) -> None:
        """
        ``SystemLogFrame``: record structured pipeline log messages as span events.

        Only records levels in ``_LOG_LEVELS_TO_CAPTURE`` (``warning``, ``error``,
        ``critical``).  DEBUG/INFO frames are dropped silently to avoid noise.

        Requires ``capture_system_logs=True`` (default ``False``).  The event is
        appended to the active turn span when one exists, falling back to the root
        trace so no log entry is lost mid-conversation.
        """
        if not self._capture_system_logs:
            return

        frame = data.frame
        level = str(getattr(frame, "level", "")).lower()
        message = str(getattr(frame, "message", getattr(frame, "text", "")))

        if not message or level not in _LOG_LEVELS_TO_CAPTURE:
            return

        target = self._current_turn_span or (self._trace if self._trace else None)
        if target is None:
            return

        try:
            target.events.append(
                SpanEvent(
                    name="pipecat.system_log",
                    timestamp=datetime.now(timezone.utc),
                    attributes={
                        "log.level": level,
                        "log.message": message,
                    },
                )
            )
        except Exception:  # pylint: disable=broad-except
            pass
