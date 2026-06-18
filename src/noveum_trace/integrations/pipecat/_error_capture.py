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

from noveum_trace.core.span import SpanEvent
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

        for span in filter(None, [self._active_llm_span, self._active_tts_span]):
            span.attributes["pipecat_span_status"] = "error"
            span.attributes["pipecat_span_status_message"] = error_msg

        if self._current_turn_span:
            self._current_turn_span.attributes["pipecat_span_status"] = "error"
            self._current_turn_span.attributes["pipecat_span_status_message"] = (
                error_msg
            )
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
