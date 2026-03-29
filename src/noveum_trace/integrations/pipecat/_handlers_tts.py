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

from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
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

        self._tts_text_buffer.clear()
        self._tts_audio_buffer.clear()
        # A new TTS span is opening — the stale backref is no longer valid.
        self._last_tts_span = None

        attributes: dict[str, Any] = {}
        source = getattr(data, "source", None)
        # Pin the TTS source processor so _handle_tts_audio only buffers frames
        # from this processor.  Downstream resamplers / aggregators that re-emit
        # TTSAudioRawFrame with fresh IDs are silently ignored.
        self._tts_source_processor = source
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

        self._tts_audio_buffer.append(_data.frame)

    async def _handle_tts_stopped(self, data: Any) -> None:
        """
        ``TTSStoppedFrame``: finish the active ``pipecat.tts`` span.

        Attributes set:
          - ``tts.input_text`` — accumulated text (if ``capture_text=True``)
          - ``tts.audio_uuid`` — UUID of the uploaded WAV (if ``record_audio=True``)
        """
        span = self._active_tts_span
        if not span:
            return
        self._active_tts_span = None
        self._tts_source_processor = None
        # Keep a backref so MetricsFrame data (TTS character counts, TTFB) arriving
        # after this span closes can still be attached to the right span.
        # Cleared when the next TTS span opens.
        self._last_tts_span = span

        if self._capture_text and self._tts_text_buffer:
            span.attributes["tts.input_text"] = "".join(self._tts_text_buffer)
        self._tts_text_buffer.clear()

        if self._record_audio and self._tts_audio_buffer:
            audio_uuid = str(uuid.uuid4())
            upload_audio_frames(
                self._tts_audio_buffer,
                audio_uuid,
                "tts",
                span.trace_id,
                span.span_id,
                client=self._get_client(),
            )
            span.attributes["tts.audio_uuid"] = audio_uuid
            self._tts_audio_buffer.clear()

        span.attributes["pipecat_span_status"] = "ok"
        span.finish()
