"""
TTS frame handler mixin for NoveumTraceObserver.

Handles:
  - LLMTextFrame at TTS input — open pipecat.tts span for text aggregation
  - AggregatedTextFrame — capture the context-bearing request boundary
  - TTSStartedFrame    — record synthesis/audio-start milestone (fallback opener)
  - TTSTextFrame       — accumulate spoken/progress text
  - TTSAudioRawFrame   — buffer raw PCM for audio upload (opt-in)
  - TTSStoppedFrame    — finish span, optionally upload audio
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
from noveum_trace.integrations.pipecat._processor_registry import PROCESSOR_ROLE_TTS
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
    #   _tts_request_text_buffer, _tts_text_buffer, _tts_audio_buffer,
    #   _tts_source_processor,
    #   _active_tts_span, _current_turn_span
    # Helpers: _create_child_span(), _get_client()

    @staticmethod
    def _tts_frame_context_id(data: Any) -> str | None:
        frame = getattr(data, "frame", None)
        context_id = getattr(frame, "context_id", None)
        return str(context_id) if context_id is not None else None

    def _tts_operation_matches(self, data: Any) -> bool:
        """Return whether ``data`` belongs to the currently observed TTS op."""
        # TODO: If a real provider fails because one TTS processor emits overlapping
        # context_id values, replace the scalar TTS operation state with records keyed
        # by (processor identity, context_id). Keep the current simpler lifecycle until
        # that behavior is reproduced in an actual trace.
        source = getattr(data, "source", None)
        context_id = self._tts_frame_context_id(data)
        source_matches = (
            source is None
            or self._tts_source_processor is None
            or source is self._tts_source_processor
        )
        context_matches = (
            context_id is None
            or self._tts_context_id is None
            or context_id == self._tts_context_id
        )
        return source_matches and context_matches

    @staticmethod
    def _set_tts_service_attributes(span: Any, source: Any) -> None:
        if span is None or source is None:
            return
        settings = extract_service_settings(source)
        for settings_key, attribute_key in (
            ("voice", "tts.voice"),
            ("model", "tts.model"),
            ("language", "tts.language"),
        ):
            value = settings.get(settings_key)
            if value is not None:
                span.attributes[attribute_key] = value
        provider = derive_provider(source, settings.get("model"))
        if provider:
            span.attributes["tts.provider"] = provider

    def _open_tts_operation(
        self,
        data: Any,
        *,
        request_boundary: bool,
        source_processor: Any = None,
        milestone: str | None = None,
    ) -> Any:
        """Open one TTS span at request time, with TTSStarted as fallback."""
        if not self._trace:
            return None

        self._tts_request_text_buffer.clear()
        self._tts_text_buffer.clear()
        self._tts_text_interim_buffer.clear()
        self._tts_audio_buffer.clear()
        self._last_tts_span = None
        self._last_tts_source_processor = None

        source = (
            source_processor
            if source_processor is not None
            else getattr(data, "source", None)
        )
        self._tts_source_processor = source
        if source is not None:
            self._processor_registry.set_explicit_role(source, PROCESSOR_ROLE_TTS)
        self._tts_context_id = self._tts_frame_context_id(data)
        frame_id = getattr(getattr(data, "frame", None), "id", None)
        self._tts_request_frame_id = (
            frame_id if request_boundary and isinstance(frame_id, int) else None
        )
        self._tts_start_frame_id = (
            frame_id if not request_boundary and isinstance(frame_id, int) else None
        )

        attributes: dict[str, Any] = {}
        if self._tts_context_id is not None:
            attributes["tts.context_id"] = self._tts_context_id
        span = self._create_child_span(
            SPAN_TTS,
            parent_span=self._current_turn_span,
            attributes=attributes,
        )
        self._active_tts_span = span
        self._set_tts_service_attributes(span, source)
        if span is not None:
            milestone_name = milestone or (
                "tts.requested_at" if request_boundary else "tts.started_at"
            )
            span.attributes[milestone_name] = span.start_time.isoformat()
        return span

    async def _handle_tts_input_text(self, data: Any, tts_processor: Any) -> None:
        """Open the TTS operation when its text-aggregation phase begins."""
        if not self._trace or tts_processor is None:
            return
        if bool(getattr(getattr(data, "frame", None), "skip_tts", False)):
            return
        if (
            self._active_tts_span is not None
            and self._tts_source_processor is not None
            and self._tts_source_processor is not tts_processor
        ):
            await self._finalize_tts_operation(
                complete=False,
                termination_reason="overlapping_request",
                terminal_status="cancelled",
            )
        if self._active_tts_span is None:
            self._open_tts_operation(
                data,
                request_boundary=True,
                source_processor=tts_processor,
                milestone="tts.aggregation_started_at",
            )

    async def _handle_tts_request(self, data: Any) -> None:
        """Open/reuse a TTS span at Pipecat's context-bearing request frame."""
        if not self._trace:
            return
        if self._active_tts_span is not None and not self._tts_operation_matches(data):
            await self._finalize_tts_operation(
                complete=False,
                termination_reason="overlapping_request",
                terminal_status="cancelled",
            )
        span = self._active_tts_span
        if span is None:
            span = self._open_tts_operation(data, request_boundary=True)
        if span is None:
            return

        source = getattr(data, "source", None)
        if self._tts_source_processor is None and source is not None:
            self._tts_source_processor = source
            self._processor_registry.set_explicit_role(source, PROCESSOR_ROLE_TTS)
            self._set_tts_service_attributes(span, source)
        context_id = self._tts_frame_context_id(data)
        if self._tts_context_id is None and context_id is not None:
            self._tts_context_id = context_id
            span.attributes["tts.context_id"] = context_id
        frame_id = getattr(getattr(data, "frame", None), "id", None)
        if self._tts_request_frame_id is None and isinstance(frame_id, int):
            self._tts_request_frame_id = frame_id
        span.attributes.setdefault(
            "tts.requested_at", datetime.now(timezone.utc).isoformat()
        )

        if self._capture_text:
            frame = data.frame
            text = getattr(frame, "text", None)
            if text:
                self._tts_request_text_buffer.append(
                    (
                        str(text),
                        bool(getattr(frame, "includes_inter_frame_spaces", False)),
                    )
                )

    async def _handle_tts_started(self, data: Any) -> None:
        """Record audio-context start; open a fallback span when needed."""
        if not self._trace:
            return
        if self._active_tts_span is not None and not self._tts_operation_matches(data):
            await self._finalize_tts_operation(
                complete=False,
                termination_reason="overlapping_start",
                terminal_status="cancelled",
            )
        span = self._active_tts_span
        if span is None:
            span = self._open_tts_operation(data, request_boundary=False)
        if span is None:
            return

        source = getattr(data, "source", None)
        if self._tts_source_processor is None and source is not None:
            self._tts_source_processor = source
            self._processor_registry.set_explicit_role(source, PROCESSOR_ROLE_TTS)
            self._set_tts_service_attributes(span, source)
        context_id = self._tts_frame_context_id(data)
        if self._tts_context_id is None and context_id is not None:
            self._tts_context_id = context_id
            span.attributes["tts.context_id"] = context_id
        frame_id = getattr(getattr(data, "frame", None), "id", None)
        if isinstance(frame_id, int):
            self._tts_start_frame_id = frame_id
        logger.debug(
            "TTS started: pinned source processor %s",
            type(source).__name__ if source else None,
        )
        span.attributes.setdefault(
            "tts.started_at", datetime.now(timezone.utc).isoformat()
        )

    async def _handle_tts_text(self, data: Any) -> None:
        """
        ``TTSTextFrame``: accumulate spoken/progress text chunks.

        Pipecat emits two flavours of ``TTSTextFrame`` distinguished by
        ``aggregated_by``: sentence-level *final* text and interim word/token
        *streamed* text. The same speech can arrive as both, so concatenating
        them together double-counts. We bucket them separately (by
        ``aggregated_by``), independently from request-side AggregatedTextFrame,
        and keep each frame's ``includes_inter_frame_spaces`` so spacing can be
        reconstructed.
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
        frame = getattr(data, "frame", None)
        context_id = getattr(frame, "context_id", None)
        if (
            context_id is not None
            and self._tts_context_id is not None
            and str(context_id) != self._tts_context_id
        ):
            return
        text_frame_id = getattr(data.frame, "id", None)
        operation_frame_id = self._tts_request_frame_id or self._tts_start_frame_id
        if (
            isinstance(text_frame_id, int)
            and isinstance(operation_frame_id, int)
            and text_frame_id < operation_frame_id
        ):
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

        context_id = getattr(_data.frame, "context_id", None)
        if (
            context_id is not None
            and self._tts_context_id is not None
            and str(context_id) != self._tts_context_id
        ):
            return

        self._tts_audio_buffer.append(_data.frame)

    def _flush_tts_text(self, span: Any) -> None:
        """Write buffered TTS text onto ``span`` and clear the buffers.

        Shared by ``_handle_tts_stopped`` and the finalizer force-close (D8) so
        partial text survives an abnormal close (error / hung provider) where no
        ``TTSStoppedFrame`` ever arrives. Each attribute reflects ONLY its own frame
        type (request aggregate → ``tts.input_text``, sentence playback text →
        ``tts.spoken_text``, word/token → interim), each spacing-reconstructed from
        ``includes_inter_frame_spaces``.
        """
        if span is None:
            return
        if self._capture_text:
            request_text = (
                _concatenate_tts_text(self._tts_request_text_buffer)
                if self._tts_request_text_buffer
                else ""
            )
            final_text = (
                _concatenate_tts_text(self._tts_text_buffer)
                if self._tts_text_buffer
                else ""
            )
            if request_text:
                span.attributes["tts.input_text"] = request_text
                span.attributes["tts.input_characters"] = len(request_text)
                # Backward-compatible canonical count: unlike Pipecat's usage
                # deltas, this is the complete text represented by this span.
                span.attributes["tts.characters"] = len(request_text)
                if final_text:
                    span.attributes["tts.spoken_text"] = final_text
            elif final_text:
                # Compatibility fallback for providers/versions that do not emit
                # the earlier AggregatedTextFrame request boundary.
                span.attributes["tts.input_text"] = final_text
                span.attributes.setdefault("tts.input_characters", len(final_text))
                span.attributes.setdefault("tts.characters", len(final_text))
            if self._tts_text_interim_buffer:
                span.attributes["tts.input_text_interim"] = _concatenate_tts_text(
                    self._tts_text_interim_buffer
                )
        self._tts_request_text_buffer.clear()
        self._tts_text_buffer.clear()
        self._tts_text_interim_buffer.clear()

    async def _handle_tts_stopped(self, data: Any) -> None:
        """
        ``TTSStoppedFrame``: finish the active ``pipecat.tts`` span.

        Attributes set:
          - ``tts.input_text`` — complete request-side aggregated text, with a
            sentence TTSTextFrame compatibility fallback
          - ``tts.input_text_interim`` — interim word/token-streamed text (only
            when such frames were emitted; absent otherwise)
          - ``tts.audio_uuid`` — UUID of the uploaded WAV (if ``record_audio=True``)
          - ``tts.audio_duration_ms`` — length of the synthesized audio, summed
            from the buffered PCM frames (present only when ``record_audio=True``);
            a standalone attribute — it does NOT alter the span's wall-clock timing.
          - ``tts.stopped_at`` — the TTSStoppedFrame time (== the span's end).
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
        frame = getattr(data, "frame", None)
        context_id = getattr(frame, "context_id", None)
        if (
            context_id is not None
            and self._tts_context_id is not None
            and str(context_id) != self._tts_context_id
        ):
            return
        stop_frame_id = getattr(frame, "id", None)
        operation_frame_id = self._tts_request_frame_id or self._tts_start_frame_id
        if (
            isinstance(stop_frame_id, int)
            and isinstance(operation_frame_id, int)
            and stop_frame_id < operation_frame_id
        ):
            return
        await self._finalize_tts_operation(
            complete=True,
            termination_reason="tts_stopped",
            terminal_status="ok",
        )

    async def _finalize_tts_operation(
        self,
        *,
        complete: bool,
        termination_reason: str,
        terminal_status: str,
    ) -> Any:
        """Flush and finish the active TTS operation exactly once."""
        span = self._active_tts_span
        if span is None:
            return None

        source = self._tts_source_processor
        frames = list(self._tts_audio_buffer)
        audio_ms = calculate_audio_duration_ms(frames)

        # Detach the operation before awaiting its sink/upload. Competing terminal
        # paths then see no active operation and cannot finish or upload it twice.
        self._active_tts_span = None
        self._tts_source_processor = None
        self._tts_context_id = None
        self._tts_request_frame_id = None
        self._tts_start_frame_id = None
        self._last_tts_span = span
        self._last_tts_source_processor = source

        self._flush_tts_text(span)

        if span.attributes.get("pipecat_span_status") == "error":
            terminal_status = "error"

        span.attributes["tts.output.complete"] = complete
        span.attributes["tts.termination_reason"] = termination_reason
        if audio_ms and audio_ms > 0:
            span.attributes["tts.audio_duration_ms"] = audio_ms

        if self._record_audio:
            span.attributes["tts.audio.complete"] = complete
            span.attributes["tts.audio.present"] = bool(frames)

        if self._record_audio and frames:
            audio_uuid = str(uuid.uuid4())
            upload_ok = False
            try:
                if self._audio_sink is not None:
                    upload_ok = await self._sink_segment_audio(
                        frames,
                        audio_uuid,
                        "tts",
                        span.trace_id,
                        span.span_id,
                    )
                else:
                    upload_ok = await asyncio.to_thread(
                        upload_audio_frames,
                        frames,
                        audio_uuid,
                        "tts",
                        span.trace_id,
                        span.span_id,
                        self._get_client(),
                    )
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to upload TTS audio %s: %s",
                    audio_uuid,
                    exc,
                    exc_info=True,
                )
            span.attributes["tts.audio.upload_status"] = "ok" if upload_ok else "failed"
            if upload_ok:
                span.attributes["tts.audio_uuid"] = audio_uuid
                self._tts_audio_buffer.clear()
            elif terminal_status == "ok":
                terminal_status = "upload_failed"
        else:
            self._tts_audio_buffer.clear()

        span.attributes["pipecat_span_status"] = terminal_status
        if hasattr(span, "set_status"):
            if terminal_status == "error":
                span.set_status(SpanStatus.ERROR)
            elif terminal_status == "cancelled":
                # The ingestion API accepts ok/error/unset only. Preserve the
                # richer cancellation outcome in pipecat_span_status.
                span.set_status(SpanStatus.OK)

        self._finish_managed_span(span)
        if span.end_time is not None:
            span.attributes["tts.stopped_at"] = span.end_time.isoformat()
        return span
