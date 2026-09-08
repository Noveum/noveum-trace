"""
Turn management mixin for NoveumTraceObserver.

Handles:
  - VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame
  - UserStartedSpeakingFrame / UserStoppedSpeakingFrame
  - BotStartedSpeakingFrame / BotStoppedSpeakingFrame
  - InterruptionFrame / ErrorFrame
  - TurnTrackingObserver integration (external turn mode)
  - UserBotLatencyObserver integration
  - Turn lifecycle helpers (_start_new_turn, _end_current_turn, etc.)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from noveum_trace.core.span import SpanEvent
from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
from noveum_trace.integrations.pipecat.pipecat_constants import (
    MAX_TEXT_BUFFER_LENGTH,
    SPAN_TURN,
)

logger = logging.getLogger(__name__)

# Accessed via the outer module's sentinel at import time
try:
    from pipecat.observers.base_observer import (
        BaseObserver as _PipecatBaseObserver,
    )  # noqa: F401

    _PIPECAT_AVAILABLE = True
except ImportError:
    _PIPECAT_AVAILABLE = False


class _TurnManagerMixin(_PipecatObserverMixinBase):
    """
    VAD/speaking frame handlers, external observer wiring, and turn lifecycle.

    Two turn-detection modes:

    **External** (preferred):
      Wire ``TurnTrackingObserver`` / ``UserBotLatencyObserver`` via
      :meth:`~NoveumTraceObserver.attach_to_task` or the constructor keyword args
      ``turn_tracking_observer`` / ``latency_observer``. Pipecat calls event
      handlers as ``handler(emitter, *args)``; the ``_emitter`` argument is
      captured but ignored.

    **Standalone** (fallback):
      Internal VAD frame tracking replicates ``TurnTrackingObserver`` logic:
      ``VADUserStartedSpeakingFrame`` starts/interrupts a turn; after the bot
      finishes speaking a timeout-based deferred task closes the turn.
    """

    # State attributes declared in NoveumTraceObserver.__init__:
    #   _trace, _current_turn_span, _current_turn_number, _turn_start_time,
    #   _active_llm_span, _active_tts_span,
    #   _llm_text_buffer, _tts_text_buffer, _tts_audio_buffer,
    #   _transcription_buffer, _metrics_accumulator,
    #   _turn_tracker, _latency_tracker,
    #   _using_external_turn_tracking,
    #   _is_bot_speaking, _bot_has_spoken_in_turn,
    #   _user_stopped_speaking_time, _turn_end_task,
    #   _turn_end_timeout_secs
    # Helpers: _create_child_span()

    # ---------------------------------------------------------------------- #
    # VAD frame handlers (standalone mode only)                              #
    # ---------------------------------------------------------------------- #

    async def _handle_vad_user_started(self, _data: Any) -> None:
        """
        ``VADUserStartedSpeakingFrame``: handle interruption or new turn.

        Ignored when external turn tracking is active.
        """
        if self._using_external_turn_tracking:
            return

        await self._cancel_turn_end_timer()

        if self._is_bot_speaking:
            await self._handle_interruption_internal(interrupted_by_user=True)
            await self._start_new_turn()
        elif (
            self._current_turn_span is not None
            and self._bot_has_spoken_in_turn
            and not self._is_bot_speaking
        ):
            await self._end_current_turn(was_interrupted=False)
            await self._start_new_turn()

    async def _handle_vad_user_stopped(self, _data: Any) -> None:
        """``VADUserStoppedSpeakingFrame``: record timestamp for latency calculation."""
        if self._using_external_turn_tracking:
            return

        self._user_stopped_speaking_time = asyncio.get_running_loop().time()

    # ---------------------------------------------------------------------- #
    # Speaking event handlers (standalone mode only)                         #
    # ---------------------------------------------------------------------- #

    async def _handle_user_started_speaking(self, _data: Any) -> None:
        """``UserStartedSpeakingFrame``: open a turn if none is active (standalone)."""
        if self._using_external_turn_tracking:
            return
        await self._cancel_turn_end_timer()
        if self._current_turn_span is None:
            await self._start_new_turn()

    async def _handle_user_stopped_speaking(self, _data: Any) -> None:
        """``UserStoppedSpeakingFrame``: record user speech duration (standalone)."""
        if self._using_external_turn_tracking:
            return

        if self._current_turn_span and self._turn_start_time is not None:
            elapsed = asyncio.get_running_loop().time() - self._turn_start_time
            self._current_turn_span.attributes["turn.user_speech_duration_seconds"] = (
                elapsed
            )

    async def _handle_bot_started_speaking(self, _data: Any) -> None:
        """
        ``BotStartedSpeakingFrame``: compute user→bot latency and set speaking flag.

        The latency attribute is written only in standalone mode; in external mode
        ``UserBotLatencyObserver`` provides the measured value via
        ``_handle_latency_measured``.
        """
        self._is_bot_speaking = True
        self._bot_has_spoken_in_turn = True

        if self._using_external_turn_tracking:
            return

        if (
            self._current_turn_span is not None
            and self._user_stopped_speaking_time is not None
        ):
            latency = (
                asyncio.get_running_loop().time() - self._user_stopped_speaking_time
            )

            self._current_turn_span.attributes["turn.user_bot_latency_seconds"] = (
                latency
            )

    async def _handle_bot_stopped_speaking(self, _data: Any) -> None:
        """``BotStoppedSpeakingFrame``: clear speaking flag and schedule deferred turn end."""
        self._is_bot_speaking = False

        if self._using_external_turn_tracking:
            return

        await self._cancel_turn_end_timer()
        if self._current_turn_span is not None:
            self._turn_end_task = asyncio.create_task(self._deferred_turn_end())

    # ---------------------------------------------------------------------- #
    # Error / interruption frame handlers                                    #
    # ---------------------------------------------------------------------- #

    async def _handle_interruption(self, _data: Any) -> None:
        """``InterruptionFrame``: cancel active spans and mark turn as interrupted."""
        await self._handle_interruption_internal(interrupted_by_user=True)

    # _handle_error is implemented by _ErrorCaptureMixin, which sits before
    # _TurnManagerMixin in NoveumTraceObserver's MRO.

    # ---------------------------------------------------------------------- #
    # Session / mute event handlers                                          #
    # ---------------------------------------------------------------------- #

    async def _handle_user_mute_started(self, _data: Any) -> None:
        """
        ``UserMuteStartedFrame``: user microphone muted by a mute strategy.

        Appends a ``user.muted`` ``SpanEvent`` to the current turn span. This
        explains gaps in transcription that would otherwise look like silence.
        """
        if self._current_turn_span:
            try:
                self._current_turn_span.events.append(
                    SpanEvent(
                        name="user.muted",
                        timestamp=datetime.now(timezone.utc),
                        attributes={},
                    )
                )
            except Exception:  # pylint: disable=broad-except
                pass

    async def _handle_user_mute_stopped(self, _data: Any) -> None:
        """
        ``UserMuteStoppedFrame``: user microphone unmuted.

        Appends a ``user.unmuted`` ``SpanEvent`` to the current turn span.
        """
        if self._current_turn_span:
            try:
                self._current_turn_span.events.append(
                    SpanEvent(
                        name="user.unmuted",
                        timestamp=datetime.now(timezone.utc),
                        attributes={},
                    )
                )
            except Exception:  # pylint: disable=broad-except
                pass

    async def _handle_client_connected(self, _data: Any) -> None:
        """
        ``ClientConnectedFrame``: a client (participant) connected to the transport.

        Appends a ``client.connected`` ``SpanEvent`` to the trace and flushes any
        buffered session metadata onto the root trace attributes.
        """
        if self._trace:
            try:
                self._trace.events.append(
                    SpanEvent(
                        name="client.connected",
                        timestamp=datetime.now(timezone.utc),
                        attributes={},
                    )
                )
            except Exception:  # pylint: disable=broad-except
                pass
        await self._flush_session_metadata()

    async def _handle_bot_connected(self, _data: Any) -> None:
        """
        ``BotConnectedFrame``: the bot successfully joined the SFU room.

        Appends a ``bot.connected`` ``SpanEvent`` to the trace and flushes any
        buffered session metadata.
        """
        if self._trace:
            try:
                self._trace.events.append(
                    SpanEvent(
                        name="bot.connected",
                        timestamp=datetime.now(timezone.utc),
                        attributes={},
                    )
                )
            except Exception:  # pylint: disable=broad-except
                pass
        await self._flush_session_metadata()

    async def _handle_stop_frame(self, _data: Any) -> None:
        """
        ``StopFrame``: stop the pipeline but keep processors running.

        Treated as a graceful finish (same as ``EndFrame``). Processors remain
        alive after this call, but tracing is complete.
        """
        await self._finish_conversation(cancelled=False)

    # ---------------------------------------------------------------------- #
    # External observer wiring                                               #
    # ---------------------------------------------------------------------- #

    def _attach_turn_tracker(self, turn_tracker: Any) -> None:
        """
        Subscribe to ``TurnTrackingObserver`` ``on_turn_started`` / ``on_turn_ended``.

        Pipecat's ``BaseObject._run_handler`` invokes handlers as
        ``handler(emitter, *args)``; the ``_emitter`` argument is captured but
        ignored. Idempotent — a second call with the same object is a no-op.
        """
        if not _PIPECAT_AVAILABLE or turn_tracker is None:
            return
        if self._turn_tracker is turn_tracker:
            return
        try:

            async def _on_turn_started(_emitter: Any, turn_number: int) -> None:
                await self._handle_turn_started(turn_number)

            async def _on_turn_ended(
                _emitter: Any,
                turn_number: int,
                duration: float,
                was_interrupted: bool,
            ) -> None:
                await self._handle_turn_ended(turn_number, duration, was_interrupted)

            for event_name, handler in (
                ("on_turn_started", _on_turn_started),
                ("on_turn_ended", _on_turn_ended),
            ):
                if hasattr(turn_tracker, "add_event_handler"):
                    turn_tracker.add_event_handler(event_name, handler)

            self._turn_tracker = turn_tracker

            self._using_external_turn_tracking = True
            logger.debug("Subscribed to TurnTrackingObserver events")
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Could not attach to TurnTrackingObserver: %s", e)

    def _attach_latency_tracker(self, latency_tracker: Any) -> None:
        """
        Subscribe to ``UserBotLatencyObserver.on_latency_measured``.

        Handler signature: ``handler(emitter, latency_seconds)`` — the emitter
        argument is captured but ignored. Idempotent.
        """
        if not _PIPECAT_AVAILABLE or latency_tracker is None:
            return

        if self._latency_tracker is latency_tracker:
            return
        try:

            async def _on_latency_measured(
                _emitter: Any, latency_seconds: float
            ) -> None:
                await self._handle_latency_measured(latency_seconds)

            if hasattr(latency_tracker, "add_event_handler"):
                latency_tracker.add_event_handler(
                    "on_latency_measured", _on_latency_measured
                )

            self._latency_tracker = latency_tracker
            logger.debug("Subscribed to UserBotLatencyObserver events")
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Could not attach to UserBotLatencyObserver: %s", e)

    # ---------------------------------------------------------------------- #
    # TurnTrackingObserver / UserBotLatencyObserver event callbacks          #
    # ---------------------------------------------------------------------- #

    async def _handle_turn_started(self, turn_number: int) -> None:
        """Called by ``TurnTrackingObserver`` when a new turn begins."""
        await self._start_new_turn(turn_number=turn_number)

    async def _handle_turn_ended(
        self,
        _turn_number: int,
        duration: float = 0.0,
        was_interrupted: bool = False,
    ) -> None:
        """Called by ``TurnTrackingObserver`` when a turn ends."""
        await self._end_current_turn(
            was_interrupted=was_interrupted, duration=duration if duration > 0 else None
        )

    async def _handle_latency_measured(self, latency_seconds: float) -> None:
        """Called by ``UserBotLatencyObserver`` with the measured user→bot latency."""
        if self._current_turn_span:
            self._current_turn_span.attributes["turn.user_bot_latency_seconds"] = (
                latency_seconds
            )

    # ---------------------------------------------------------------------- #
    # Turn lifecycle helpers                                                  #
    # ---------------------------------------------------------------------- #

    async def _start_new_turn(self, turn_number: Optional[int] = None) -> None:
        """Create a new ``pipecat.turn`` span under the conversation trace."""
        if self._trace is None:
            return

        if self._current_turn_span is not None:
            await self._end_current_turn(was_interrupted=False)

        if turn_number is not None:

            self._current_turn_number = turn_number
        else:
            self._current_turn_number += 1

        self._metrics_accumulator["turn_count"] = self._current_turn_number
        self._transcription_buffer.clear()
        self._bot_has_spoken_in_turn = False

        self._turn_start_time = asyncio.get_running_loop().time()

        self._current_turn_span = self._trace.create_span(
            name=SPAN_TURN,
            parent_span_id=None,
            attributes={"turn.number": self._current_turn_number},
        )

        # Flush any buffered EOU metrics onto the new turn span
        if self._pending_turn_eou_metrics:
            eou_attrs = {
                "turn.eou_is_complete": self._pending_turn_eou_metrics.get(
                    "turn_eou_is_complete"
                ),
                "turn.eou_confidence": self._pending_turn_eou_metrics.get(
                    "turn_eou_confidence"
                ),
                "turn.eou_processing_time_ms": self._pending_turn_eou_metrics.get(
                    "turn_eou_processing_time_ms"
                ),
                "turn.eou_inference_ms": self._pending_turn_eou_metrics.get(
                    "turn_eou_inference_ms"
                ),
                "turn.eou_server_total_ms": self._pending_turn_eou_metrics.get(
                    "turn_eou_server_total_ms"
                ),
            }
            # Only set non-None values
            for attr_name, val in eou_attrs.items():
                if val is not None:
                    self._current_turn_span.attributes[attr_name] = val
            logger.debug(
                "Flushed buffered EOU metrics to turn %s: %s",
                self._current_turn_number,
                [k for k, v in eou_attrs.items() if v is not None],
            )

            self._pending_turn_eou_metrics.clear()

        logger.debug("Started turn %s", self._current_turn_number)

    async def _end_current_turn(
        self,
        was_interrupted: bool = False,
        duration: Optional[float] = None,
    ) -> None:
        """Finish the active ``pipecat.turn`` span."""
        span = self._current_turn_span
        if span is None:
            return
        self._current_turn_span = None

        # The audio buffer and STT span are owned exclusively by the STT handlers
        # (_handle_vad_stt_start / _handle_vad_stt_stop / _handle_transcription).
        # A turn boundary must never cancel an in-flight STT span or clear the audio
        # buffer — the transcript may still be in-flight (network delay) or the user
        # may still be mid-utterance.  In both cases the STT handlers will close the
        # span and upload the audio when the transcript arrives, and
        # _handle_vad_stt_start will reset the buffer when the next utterance begins.
        if self._active_stt_span is not None:
            logger.debug(
                "STT span crosses turn boundary — preserved (always-buffer mode)"
            )

        # B7: OR-merge, never clobber. An interruption during the turn already set
        # this True (_handle_interruption_internal); an incoming was_interrupted=False
        # at turn end (the common external-turn-tracking value) must not overwrite it.
        span.attributes["turn.was_interrupted"] = bool(was_interrupted) or bool(
            span.attributes.get("turn.was_interrupted")
        )

        if duration is not None:
            span.attributes["turn.duration_seconds"] = duration
        elif self._turn_start_time is not None:
            span.attributes["turn.duration_seconds"] = (
                asyncio.get_running_loop().time() - self._turn_start_time
            )

        if self._transcription_buffer:

            user_input = " ".join(self._transcription_buffer)
            if len(user_input) > MAX_TEXT_BUFFER_LENGTH:
                user_input = user_input[:MAX_TEXT_BUFFER_LENGTH]
            span.attributes["turn.user_input"] = user_input

        if span.attributes.get("pipecat_span_status") != "error":
            span.attributes["pipecat_span_status"] = "ok"
        span.finish()

        logger.debug("Ended turn %s", self._current_turn_number)

    async def _handle_interruption_internal(
        self, interrupted_by_user: bool = True  # noqa: ARG002
    ) -> None:
        """Finalize partial LLM/TTS output and mark the turn interrupted."""
        for operation in list(self._llm_operations.active_operations):
            self._finalize_llm_operation(
                operation,
                complete=False,
                termination_reason="user_interruption",
                terminal_status="cancelled",
            )

        await self._finalize_tts_operation(
            complete=False,
            termination_reason="user_interruption",
            terminal_status="cancelled",
        )

        # STT span and audio buffer are intentionally left untouched here.
        # Interruptions are triggered by the user starting to speak, so there is
        # almost always an active STT span whose buffer is being filled right now.
        # Cancelling it would discard the audio before the transcript arrives.
        # _handle_vad_stt_start / _handle_transcription own the STT lifecycle.

        if self._current_turn_span:
            self._current_turn_span.attributes["turn.was_interrupted"] = True

        # Operation finalizers own the data-bearing buffers. These scalar fields
        # remain only for compatibility with direct-handler callers.
        self._llm_text_buffer.clear()
        self._pending_function_calls.clear()
        self._function_call_owner.clear()
        self._resolved_function_call_ids.clear()

    # ---------------------------------------------------------------------- #
    # Standalone turn-end timer                                              #
    # ---------------------------------------------------------------------- #

    async def _cancel_turn_end_timer(self) -> None:
        """Cancel a pending deferred turn-end ``asyncio.Task`` if one is running."""
        task = self._turn_end_task
        self._turn_end_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _deferred_turn_end(self) -> None:
        """Sleep for the configured timeout, then close the turn if still open."""
        try:

            await asyncio.sleep(self._turn_end_timeout_secs)

            if self._current_turn_span is not None and not self._is_bot_speaking:
                await self._end_current_turn(was_interrupted=False)
        except asyncio.CancelledError:
            pass
