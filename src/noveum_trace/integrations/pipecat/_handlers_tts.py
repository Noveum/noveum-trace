"""
TTS frame handler mixin for NoveumTraceObserver.

Handles:
  - TTSStartedFrame    — open pipecat.tts span
  - TTSTextFrame       — accumulate TTS input text
  - TTSAudioRawFrame   — buffer raw PCM for audio upload (opt-in)
  - TTSStoppedFrame    — finish span, optionally upload audio
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
from noveum_trace.integrations.pipecat._processor_registry import PROCESSOR_ROLE_TTS
from noveum_trace.integrations.pipecat.pipecat_constants import SPAN_TTS
from noveum_trace.integrations.pipecat.pipecat_utils import (
    extract_service_settings,
    upload_audio_frames,
)

logger = logging.getLogger(__name__)


class _TTSHandlersMixin(_PipecatObserverMixinBase):
    """Handler methods for TTS-related frames."""

    # State attributes declared in NoveumTraceObserver.__init__:
    #   _trace, _capture_text, _record_audio,
    #   _tts_text_buffer, _tts_audio_buffer, _tts_source_processor,
    #   _active_tts_span, _current_turn_span
    # Helpers: _create_child_span(), _get_client()

    async def _handle_tts_started(self, data: Any) -> None:
        """
        ``TTSStartedFrame``: open a ``pipecat.tts`` child span.

        Attributes set from the source processor's settings:
        ``tts.voice``, ``tts.model``.
        """
        if not self._trace:
            return

        if self._active_tts_span is not None:
            self._finalize_tts_operation(
                complete=False,
                termination_reason="overlapping_start",
                terminal_status="cancelled",
            )

        self._tts_text_buffer.clear()
        self._tts_audio_buffer.clear()
        # A new TTS span is opening — the stale backref is no longer valid.
        self._last_tts_span = None
        self._last_tts_source_processor = None

        attributes: dict[str, Any] = {}
        source = getattr(data, "source", None)
        # Pin the TTS source processor so _handle_tts_audio only buffers frames
        # from this processor.  Downstream resamplers / aggregators that re-emit
        # TTSAudioRawFrame with fresh IDs are silently ignored.
        self._tts_source_processor = source
        if source is not None:
            self._processor_registry.set_explicit_role(source, PROCESSOR_ROLE_TTS)
        context_id = getattr(data.frame, "context_id", None)
        self._tts_context_id = str(context_id) if context_id is not None else None
        frame_id = getattr(data.frame, "id", None)
        self._tts_start_frame_id = frame_id if isinstance(frame_id, int) else None
        logger.debug(
            "TTS started: pinned source processor %s",
            type(source).__name__ if source else None,
        )
        if source:
            settings = extract_service_settings(source)
            if settings.get("voice"):
                attributes["tts.voice"] = settings["voice"]
            if settings.get("model"):
                attributes["tts.model"] = settings["model"]
        if self._tts_context_id is not None:
            attributes["tts.context_id"] = self._tts_context_id

        self._active_tts_span = self._create_child_span(
            SPAN_TTS,
            parent_span=self._current_turn_span,
            attributes=attributes,
        )

    async def _handle_tts_text(self, data: Any) -> None:
        """
        ``TTSTextFrame``: accumulate TTS input text chunks.

        Written to ``tts.input_text`` when the TTS utterance ends.
        """
        if not self._capture_text:
            return
        source = getattr(data, "source", None)
        if (
            source is not None
            and self._tts_source_processor is not None
            and source is not self._tts_source_processor
        ):
            return
        context_id = getattr(data.frame, "context_id", None)
        if (
            context_id is not None
            and self._tts_context_id is not None
            and str(context_id) != self._tts_context_id
        ):
            return
        text_frame_id = getattr(data.frame, "id", None)
        if (
            isinstance(text_frame_id, int)
            and isinstance(self._tts_start_frame_id, int)
            and text_frame_id < self._tts_start_frame_id
        ):
            # A delayed text frame from the previous context-less generation must
            # not contaminate the current synthesis.
            return
        frame = data.frame
        text = getattr(frame, "text", None)
        if text:
            self._tts_text_buffer.append(str(text))

    async def _handle_tts_audio(self, _data: Any) -> None:
        """Buffer TTS PCM frames for per-span WAV upload (opt-in via ``record_audio``)."""
        if not self._record_audio:
            return
        source = getattr(_data, "source", None)
        pinned = self._tts_source_processor
        if pinned is not None and source is not pinned:
            logger.debug(
                "TTS audio: ignoring frame from non-TTS processor %s (expected %s)",
                type(source).__name__ if source else None,
                type(pinned).__name__,
            )
            return

        context_id = getattr(_data.frame, "context_id", None)
        if (
            context_id is not None
            and self._tts_context_id is not None
            and str(context_id) != self._tts_context_id
        ):
            return

        self._tts_audio_buffer.append(_data.frame)

    async def _handle_tts_stopped(self, data: Any) -> None:
        """
        ``TTSStoppedFrame``: finish the active ``pipecat.tts`` span.

        Attributes set:
          - ``tts.input_text`` — accumulated text (if ``capture_text=True``)
          - ``tts.audio_uuid`` — UUID of the uploaded WAV (if ``record_audio=True``)
        """
        if self._active_tts_span is None:
            return
        source = getattr(data, "source", None)
        if (
            source is not None
            and self._tts_source_processor is not None
            and source is not self._tts_source_processor
        ):
            return
        context_id = getattr(data.frame, "context_id", None)
        if (
            context_id is not None
            and self._tts_context_id is not None
            and str(context_id) != self._tts_context_id
        ):
            return
        stop_frame_id = getattr(data.frame, "id", None)
        if (
            isinstance(stop_frame_id, int)
            and isinstance(self._tts_start_frame_id, int)
            and stop_frame_id < self._tts_start_frame_id
        ):
            return
        self._finalize_tts_operation(
            complete=True,
            termination_reason="tts_stopped",
            terminal_status="ok",
        )

    def _finalize_tts_operation(
        self,
        *,
        complete: bool,
        termination_reason: str,
        terminal_status: str,
    ) -> Any:
        """Flush text/audio and finish the current TTS span at most once."""
        span = self._active_tts_span
        if span is None:
            return None

        source = self._tts_source_processor
        text = "".join(self._tts_text_buffer)
        audio_frames = list(self._tts_audio_buffer)
        self._active_tts_span = None
        self._tts_source_processor = None
        self._tts_context_id = None
        self._tts_start_frame_id = None
        self._tts_text_buffer.clear()
        self._tts_audio_buffer.clear()
        self._last_tts_span = span
        self._last_tts_source_processor = source

        existing_status = span.attributes.get("pipecat_span_status")
        if existing_status == "error":
            terminal_status = "error"
        if self._capture_text and text:
            span.attributes["tts.input_text"] = text
        span.attributes["tts.output.complete"] = complete
        span.attributes["tts.termination_reason"] = termination_reason

        if self._record_audio:
            span.attributes["tts.audio.complete"] = complete
            span.attributes["tts.audio.present"] = bool(audio_frames)
        if self._record_audio and audio_frames:
            audio_uuid = str(uuid.uuid4())
            try:
                upload_ok = upload_audio_frames(
                    audio_frames,
                    audio_uuid,
                    "tts",
                    span.trace_id,
                    span.span_id,
                    client=self._get_client(),
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to upload TTS audio %s: %s",
                    audio_uuid,
                    exc,
                    exc_info=True,
                )
                upload_ok = False
            span.attributes["tts.audio.upload_status"] = "ok" if upload_ok else "failed"
            if upload_ok:
                span.attributes["tts.audio_uuid"] = audio_uuid

        span.attributes["pipecat_span_status"] = terminal_status
        if hasattr(span, "set_status"):
            if terminal_status == "error":
                span.set_status(SpanStatus.ERROR)
            elif terminal_status == "cancelled":
                span.set_status(SpanStatus.OK)
        if getattr(span, "is_finished", lambda: False)() is not True:
            self._finish_managed_span(span)
        return span
