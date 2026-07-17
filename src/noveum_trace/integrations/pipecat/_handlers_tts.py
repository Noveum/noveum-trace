"""
TTS frame handler mixin for NoveumTraceObserver.

Handles:
  - TTSStartedFrame    — open pipecat.tts span
  - TTSTextFrame       — accumulate TTS input text
  - TTSAudioRawFrame   — buffer raw PCM for audio upload (opt-in)
  - TTSStoppedFrame    — finish span, optionally upload audio
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
from noveum_trace.integrations.pipecat.pipecat_constants import SPAN_TTS
from noveum_trace.integrations.pipecat.pipecat_utils import (
    calculate_audio_duration_ms,
    derive_provider,
    extract_service_settings,
    upload_audio_frames,
)

logger = logging.getLogger(__name__)


def _concatenate_tts_text(parts: list[tuple[str, bool]]) -> str:
    """Join TTS text chunks respecting per-frame ``includes_inter_frame_spaces``.

    Faithfully replicates pipecat's ``concatenate_aggregated_text``
    (``pipecat.utils.string``) so word/token-streamed frames — which omit
    inter-frame spaces (``includes_inter_frame_spaces=False``) — are reassembled
    with correct spacing instead of being mashed together (the de-spacing bug).
    Reimplemented here (rather than imported) to stay version-independent across
    pipecat releases. ``parts`` is a list of ``(text, includes_inter_frame_spaces)``.
    """
    result = ""
    last_ifs = False
    for text, ifs in parts:
        if not text:
            continue
        if not result:
            result += text
            last_ifs = ifs
            continue
        if ifs and last_ifs:
            result += text
        elif not ifs and not last_ifs:
            result += " " + text
        else:
            # transition between spaced/unspaced runs — add a space only if needed
            if not result[-1].isspace() and not text[0].isspace():
                result += " "
            result += text
        last_ifs = ifs
    return result.strip()


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

        Attributes set: ``tts.voice`` / ``tts.model`` (from the source processor's
        settings) and ``tts.started_at`` (the TTSStartedFrame arrival time, i.e. the
        span's own start).
        """
        if not self._trace:
            return

        self._tts_text_buffer.clear()
        self._tts_text_interim_buffer.clear()
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
            # D6: extract_service_settings already resolves language; the STT
            # handler copies it but this one used to drop it.
            if settings.get("language"):
                attributes["tts.language"] = settings["language"]
            provider = derive_provider(source, settings.get("model"))
            if provider:
                attributes["tts.provider"] = provider

        self._active_tts_span = self._create_child_span(
            SPAN_TTS,
            parent_span=self._current_turn_span,
            attributes=attributes,
        )
        # Explicit start timestamp from the TTSStartedFrame (== the span's start).
        if self._active_tts_span is not None:
            self._active_tts_span.attributes["tts.started_at"] = (
                self._active_tts_span.start_time.isoformat()
            )

    async def _handle_tts_text(self, data: Any) -> None:
        """
        ``TTSTextFrame``: accumulate TTS input text chunks.

        Pipecat emits two flavours of ``TTSTextFrame`` distinguished by
        ``aggregated_by``: sentence-level *final* text and interim word/token
        *streamed* text. The same speech can arrive as both, so concatenating
        them together double-counts. We bucket them separately (by
        ``aggregated_by``) and emit separate attributes — ``tts.input_text``
        (final) and ``tts.input_text_interim`` (interim) — keeping each frame's
        ``includes_inter_frame_spaces`` so spacing can be reconstructed.
        """
        if not self._capture_text:
            return
        frame = data.frame
        text = getattr(frame, "text", None)
        if not text:
            return
        part = (str(text), bool(getattr(frame, "includes_inter_frame_spaces", False)))
        aggregated_by = str(getattr(frame, "aggregated_by", "") or "").lower()
        if aggregated_by == "sentence":
            self._tts_text_buffer.append(part)
        else:
            self._tts_text_interim_buffer.append(part)

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

    def _flush_tts_text(self, span: Any) -> None:
        """Write buffered TTS text onto ``span`` and clear the buffers.

        Shared by ``_handle_tts_stopped`` and the finalizer force-close (D8) so
        partial text survives an abnormal close (error / hung provider) where no
        ``TTSStoppedFrame`` ever arrives. Each attribute reflects ONLY its own frame
        type (sentence-aggregated → ``tts.input_text``, word/token → interim), each
        spacing-reconstructed from ``includes_inter_frame_spaces``.
        """
        if span is None:
            return
        if self._capture_text:
            if self._tts_text_buffer:
                span.attributes["tts.input_text"] = _concatenate_tts_text(
                    self._tts_text_buffer
                )
            if self._tts_text_interim_buffer:
                span.attributes["tts.input_text_interim"] = _concatenate_tts_text(
                    self._tts_text_interim_buffer
                )
        self._tts_text_buffer.clear()
        self._tts_text_interim_buffer.clear()

    async def _handle_tts_stopped(self, data: Any) -> None:
        """
        ``TTSStoppedFrame``: finish the active ``pipecat.tts`` span.

        Attributes set:
          - ``tts.input_text`` — final sentence-aggregated text (only when
            sentence-level frames were emitted; absent otherwise)
          - ``tts.input_text_interim`` — interim word/token-streamed text (only
            when such frames were emitted; absent otherwise)
          - ``tts.audio_uuid`` — UUID of the uploaded WAV (if ``record_audio=True``)
          - ``tts.audio_duration_ms`` — length of the synthesized audio, summed
            from the buffered PCM frames (present only when ``record_audio=True``);
            a standalone attribute — it does NOT alter the span's wall-clock timing.
          - ``tts.stopped_at`` — the TTSStoppedFrame time (== the span's end).
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

        # Audio playback length, computed before the buffer may be cleared on a
        # successful upload. Recorded as its own attribute only — the span's
        # start_time / duration_ms remain true wall-clock (per TRACE_DESIGN §6).
        audio_ms = calculate_audio_duration_ms(self._tts_audio_buffer)

        try:
            self._flush_tts_text(span)

            if audio_ms and audio_ms > 0:
                span.attributes["tts.audio_duration_ms"] = audio_ms

            tts_status = "ok"
            if self._record_audio and self._tts_audio_buffer:
                client = self._get_client()
                audio_uuid = str(uuid.uuid4())
                upload_ok = False
                try:
                    # WAV encoding is CPU-bound and blocks the event loop; run it off
                    # the loop thread, matching _handlers_stt._handle_transcription.
                    upload_ok = await asyncio.to_thread(
                        upload_audio_frames,
                        self._tts_audio_buffer,
                        audio_uuid,
                        "tts",
                        span.trace_id,
                        span.span_id,
                        client,
                    )
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning(
                        "Failed to upload TTS audio %s: %s",
                        audio_uuid,
                        e,
                        exc_info=True,
                    )
                    upload_ok = False
                if upload_ok:
                    span.attributes["tts.audio_uuid"] = audio_uuid
                    self._tts_audio_buffer.clear()
                else:
                    tts_status = "upload_failed"

            span.attributes["pipecat_span_status"] = tts_status
        finally:
            span.finish()
            # Explicit stop timestamp from the TTSStoppedFrame (== the span's end).
            if span.end_time is not None:
                span.attributes["tts.stopped_at"] = span.end_time.isoformat()
