"""
STT frame handler mixin for NoveumTraceObserver.

Audio capture model (always-buffer):

  All ``UserAudioRawFrame`` / ``InputAudioRawFrame`` frames are accumulated
  continuously into ``_stt_audio_buffer`` whenever ``record_audio=True``.
  No VAD gating is applied to audio buffering.  On ``TranscriptionFrame``
  (final) the complete buffer — which contains everything the STT provider
  received for that utterance — is uploaded and then cleared.

  This works correctly for both STT modes:

  Streaming STT (Deepgram, AssemblyAI, …):
    Audio flows to the provider without interruption.  VAD events are hints
    only.  The buffer holds all audio from the previous transcript until the
    new one arrives, matching what the provider processed.

  Segment-based STT (Whisper, …):
    Pipecat's ``SegmentedSTTService`` collects audio internally between VAD
    start/stop and sends one batch to the provider.  The always-buffer
    captures that same window (plus a small amount of surrounding audio),
    which is harmless.

  Span lifecycle:
    VADUserStartedSpeakingFrame  → open pipecat.stt span (for timing)
    UserAudioRawFrame            → append to buffer continuously (all times)
    VADUserStoppedSpeakingFrame  → no-op for audio
    TranscriptionFrame (final)   → set attributes, upload buffer, close span

  Safety fallback (TranscriptionFrame with no active span):
    → create a point span exactly as before

Handles:
  - SpeechControlParamsFrame     — detect VAD presence (_vad_present flag)
  - UserAudioRawFrame / InputAudioRawFrame  — buffer raw PCM always
  - VAD start (via _handle_vad_stt_start)  — open span only
  - VAD stop  (via _handle_vad_stt_stop)   — no-op for audio
  - TranscriptionFrame           — close pipecat.stt span (final)
  - InterimTranscriptionFrame    — stt.interim_transcription event on active STT span
  - InputTextRawFrame            — typed user input → instant pipecat.stt point span
  - STTMetadataFrame             — stt.ttfs_p99_latency_ms on trace
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from noveum_trace.core.span import SpanEvent
from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
from noveum_trace.integrations.pipecat.pipecat_constants import SPAN_STT
from noveum_trace.integrations.pipecat.pipecat_utils import (
    extract_service_settings,
    extract_stt_confidence,
    upload_audio_frames,
)

logger = logging.getLogger(__name__)


class _STTHandlersMixin(_PipecatObserverMixinBase):
    """Handler methods for STT-related frames."""

    # These attributes are declared in NoveumTraceObserver.__init__:
    #   _trace, _capture_text, _record_audio, _transcription_buffer,
    #   _stt_audio_buffer, _stt_source_processor,
    #   _current_turn_span, _using_external_turn_tracking,
    #   _vad_present, _active_stt_span
    #   _vad_speech_start_time, _stt_interim_results, _stt_first_text_latency_recorded
    # and helpers: _get_client(), _create_child_span(), _start_new_turn()

    # ---------------------------------------------------------------------- #
    # VAD detection                                                           #
    # ---------------------------------------------------------------------- #

    async def _handle_speech_control_params(self, data: Any) -> None:
        """
        ``SpeechControlParamsFrame``: broadcast at pipeline start by VADController /
        base_input when a VAD analyzer is configured.

        Sets ``_vad_present = True`` when ``vad_params`` is non-null, which switches
        the STT span lifecycle to the VAD-gated path (span opens on VAD start).
        Audio is always buffered regardless of this flag.
        """
        frame = data.frame
        vad_params = getattr(frame, "vad_params", None)
        if vad_params is not None:
            self._vad_present = True
            logger.debug("VAD detected via SpeechControlParamsFrame")

    # ---------------------------------------------------------------------- #
    # VAD-side STT span open / close                                         #
    # ---------------------------------------------------------------------- #

    async def _handle_vad_stt_start(self, data: Any) -> None:
        """
        ``VADUserStartedSpeakingFrame`` (STT side): open a ``pipecat.stt`` span.

        Only acts when ``_vad_present`` is True.  Audio is already being buffered
        continuously; this call only manages the span for timing attribution.

        If a previous STT span is still open (no transcript arrived before this
        new utterance), it is cancelled first.

        Pipecat's ``_queued_broadcast_frame`` emits ``VADUserStartedSpeakingFrame``
        twice: once upstream (immediately) and once downstream.  The observer sees
        both because they carry different frame IDs.  We guard against the downstream
        duplicate by ignoring it when an active STT span already exists.
        """
        if not self._vad_present:
            return
        if not self._trace:
            return

        # Skip the downstream duplicate broadcast.
        try:
            from pipecat.processors.frame_processor import FrameDirection

            if (
                getattr(data, "direction", None) == FrameDirection.DOWNSTREAM
                and self._active_stt_span is not None
            ):
                logger.debug(
                    "Ignoring downstream VADUserStartedSpeakingFrame duplicate "
                    "(active STT span already exists)"
                )
                return
        except ImportError:
            pass

        # Close any orphaned span from a previous utterance that never got a transcript
        if self._active_stt_span is not None:
            logger.debug("Closing orphaned STT span before new VAD utterance")

            self._active_stt_span.attributes["pipecat_span_status"] = "cancelled"
            self._active_stt_span.finish()
            self._active_stt_span = None

        # Ensure a turn is open

        if self._current_turn_span is None and not self._using_external_turn_tracking:
            await self._start_new_turn()

        span = self._create_child_span(
            SPAN_STT,
            parent_span=self._current_turn_span,
        )
        if span:
            self._active_stt_span = span

            self._vad_speech_start_time = asyncio.get_running_loop().time()
            self._stt_interim_results.clear()

            self._stt_first_text_latency_recorded = False

        # Do NOT clear _stt_audio_buffer here — the always-buffer model keeps
        # pre-VAD audio in the buffer so the speech beginning is captured.
        logger.debug("STT span opened on VAD start (audio buffer preserved)")

    async def _handle_vad_stt_stop(self, data: Any) -> None:
        """
        ``VADUserStoppedSpeakingFrame`` (STT side): no-op for audio.

        Audio continues to be buffered after VAD stops — the STT provider
        (streaming or segment-based) may still be processing frames until it
        emits ``TranscriptionFrame``.  The buffer is flushed there.

        The direction-duplicate guard is kept so this handler stays a safe
        no-op if called multiple times.
        """
        if not self._vad_present:
            return

        try:
            from pipecat.processors.frame_processor import FrameDirection

            if getattr(data, "direction", None) == FrameDirection.DOWNSTREAM:
                logger.debug(
                    "Ignoring downstream VADUserStoppedSpeakingFrame duplicate"
                )
                return
        except ImportError:
            pass

        logger.debug(
            "VAD stopped — continuing to buffer audio until TranscriptionFrame"
        )

    # ---------------------------------------------------------------------- #
    # Audio buffering                                                         #
    # ---------------------------------------------------------------------- #

    async def _handle_user_audio(self, data: Any) -> None:
        """
        ``UserAudioRawFrame`` / ``InputAudioRawFrame``: buffer PCM continuously.

        Always appends to ``_stt_audio_buffer`` when ``record_audio=True``,
        regardless of VAD state or whether a span is open.  The buffer is
        flushed (uploaded and cleared) on ``TranscriptionFrame``.

        Source pinning: the first audio frame pins ``_stt_source_processor``.
        Subsequent frames from a different source (downstream re-emitters) are
        silently ignored to avoid duplicating audio bytes, mirroring the TTS fix.
        """
        if not self._record_audio:
            return

        source = getattr(data, "source", None)
        pinned = self._stt_source_processor

        if pinned is None:
            # Pin on the first audio frame we see
            self._stt_source_processor = source
            logger.debug(
                "STT audio: pinned source processor %s",
                type(source).__name__ if source else None,
            )
        elif source is not pinned:
            logger.debug(
                "STT audio: ignoring frame from non-STT processor %s (expected %s)",
                type(source).__name__ if source else None,
                type(pinned).__name__,
            )
            return

        self._stt_audio_buffer.append(data.frame)

    # ---------------------------------------------------------------------- #
    # Final transcript                                                        #
    # ---------------------------------------------------------------------- #

    async def _handle_transcription(self, data: Any) -> None:
        """
        ``TranscriptionFrame`` (final): close the active ``pipecat.stt`` span
        and upload all accumulated audio.

        The buffer contains everything recorded since the last transcript —
        for streaming STT this is all audio the provider received; for
        segment-based STT it includes the VAD segment plus a small amount of
        surrounding audio.

        If no span is active (safety fallback), a point span is created.

        Attributes set: ``stt.text``, ``stt.is_final``, ``stt.language`` (if
        present), ``stt.user_id`` (if present), ``stt.model`` (from processor
        settings), ``stt.confidence`` (from ``result`` when present),
        ``stt.vad_to_final_ms`` / ``stt.first_text_latency_ms`` (monotonic, when
        VAD-gated), ``stt.interim_results`` (JSON list of
        ``{"text","confidence"}``), ``stt.audio_uuid`` (if ``record_audio=True``).
        """
        if not self._trace:
            return

        frame = data.frame
        text = getattr(frame, "text", "") or ""
        language = getattr(frame, "language", None)
        user_id = getattr(frame, "user_id", None)

        if text and self._capture_text:
            self._transcription_buffer.append(text)

        # Ensure we have a turn open

        if self._current_turn_span is None and not self._using_external_turn_tracking:
            await self._start_new_turn()

        # Build attributes
        attributes: dict[str, Any] = {"stt.text": text, "stt.is_final": True}
        if language:
            attributes["stt.language"] = str(language)
        if user_id:
            attributes["stt.user_id"] = str(user_id)

        source = getattr(data, "source", None)
        if source:
            settings = extract_service_settings(source)
            if settings.get("model"):
                attributes["stt.model"] = settings["model"]

        # Reuse the long-lived span if one is open, otherwise create a point span
        span = self._active_stt_span
        if span is None:
            logger.debug(
                "No active STT span on TranscriptionFrame — creating point span (fallback)"
            )
            span = self._create_child_span(
                SPAN_STT,
                parent_span=self._current_turn_span,
                attributes=attributes,
            )
        else:
            for key, val in attributes.items():
                span.attributes[key] = val

        self._active_stt_span = None
        # Reset source pin so next utterance re-pins on its first audio frame
        self._stt_source_processor = None

        if span:
            loop = asyncio.get_running_loop()
            now = loop.time()
            raw_result = getattr(frame, "result", None)
            conf = extract_stt_confidence(raw_result)
            if conf is not None:
                span.attributes["stt.confidence"] = conf

            vad_start = self._vad_speech_start_time
            if vad_start is not None:
                span.attributes["stt.vad_to_final_ms"] = (now - vad_start) * 1000.0

            interim_pairs = self._stt_interim_results
            if interim_pairs:
                try:
                    span.attributes["stt.interim_results"] = json.dumps(
                        interim_pairs, ensure_ascii=False
                    )
                except Exception:  # pylint: disable=broad-except
                    span.attributes["stt.interim_results"] = json.dumps(
                        interim_pairs, default=str
                    )

            self._vad_speech_start_time = None
            self._stt_interim_results.clear()

            self._stt_first_text_latency_recorded = False

            if self._record_audio and self._stt_audio_buffer:
                audio_uuid = str(uuid.uuid4())
                upload_audio_frames(
                    self._stt_audio_buffer,
                    audio_uuid,
                    "stt",
                    span.trace_id,
                    span.span_id,
                    client=self._get_client(),
                )
                span.attributes["stt.audio_uuid"] = audio_uuid
            self._stt_audio_buffer.clear()

            span.attributes["pipecat_span_status"] = "ok"
            span.finish()

    # ---------------------------------------------------------------------- #
    # Interim transcription                                                   #
    # ---------------------------------------------------------------------- #

    async def _handle_interim_transcription(self, data: Any) -> None:
        """
        ``InterimTranscriptionFrame``: partial hypothesis while the user is still speaking.

        Appends ``{"text", "confidence"}`` to ``_stt_interim_results``, records
        ``stt.first_text_latency_ms`` on the STT span once (VAD start → first
        interim, monotonic clock), and attaches a ``stt.interim_transcription``
        ``SpanEvent`` with text and optional confidence.

        Only recorded when ``capture_text=True``.
        """
        if not self._capture_text:
            return
        frame = data.frame
        text = getattr(frame, "text", None)
        if not text:
            return

        raw_result = getattr(frame, "result", None)
        conf = extract_stt_confidence(raw_result)

        self._stt_interim_results.append({"text": str(text), "confidence": conf})

        target_span = self._active_stt_span or self._current_turn_span
        if target_span is None:
            return

        if (
            self._active_stt_span is not None
            and not self._stt_first_text_latency_recorded
            and self._vad_speech_start_time is not None
        ):
            elapsed = asyncio.get_running_loop().time() - self._vad_speech_start_time
            self._active_stt_span.attributes["stt.first_text_latency_ms"] = (
                elapsed * 1000.0
            )

            self._stt_first_text_latency_recorded = True

        try:
            ev_attrs: dict[str, Any] = {"text": str(text)}
            if conf is not None:
                ev_attrs["confidence"] = conf
            target_span.events.append(
                SpanEvent(
                    name="stt.interim_transcription",
                    timestamp=datetime.now(timezone.utc),
                    attributes=ev_attrs,
                )
            )
        except Exception:  # pylint: disable=broad-except
            pass

    # ---------------------------------------------------------------------- #
    # Typed text input (no audio phase)                                      #
    # ---------------------------------------------------------------------- #

    async def _handle_input_text(self, data: Any) -> None:
        """
        ``InputTextRawFrame``: typed user input (e.g. chat text injection).

        Creates an instant ``pipecat.stt`` point span with ``stt.input_type = "text"``.
        Does not interact with ``_active_stt_span`` or audio buffering state.
        """
        if not self._trace:
            return

        frame = data.frame
        text = getattr(frame, "text", "") or ""
        if not text:
            return

        if self._capture_text:
            self._transcription_buffer.append(text)

        if self._current_turn_span is None and not self._using_external_turn_tracking:
            await self._start_new_turn()

        attributes: dict[str, Any] = {
            "stt.text": text,
            "stt.is_final": True,
            "stt.input_type": "text",
        }

        span = self._create_child_span(
            SPAN_STT,
            parent_span=self._current_turn_span,
            attributes=attributes,
        )
        if span:
            span.attributes["pipecat_span_status"] = "ok"
            span.finish()

    # ---------------------------------------------------------------------- #
    # STT service metadata                                                   #
    # ---------------------------------------------------------------------- #

    async def _handle_stt_metadata(self, data: Any) -> None:
        """
        ``STTMetadataFrame``: broadcast by STT services at pipeline start (and on
        service switch / metadata refresh) with latency characteristics.

        Sets ``stt.ttfs_p99_latency_ms`` on the trace (P99 time from speech end to
        final transcript, converted from seconds to milliseconds). The value is the
        service's expected latency, not a per-utterance measurement.
        """
        if not self._trace:
            return
        frame = data.frame
        latency_secs = getattr(frame, "ttfs_p99_latency", None)
        if latency_secs is not None:
            try:
                self._trace.set_attributes(
                    {"stt.ttfs_p99_latency_ms": float(latency_secs) * 1000}
                )
            except Exception:  # pylint: disable=broad-except
                pass
