"""
NoveumTraceObserver for Pipecat pipelines.

Extends Pipecat's ``BaseObserver`` to map frame-based pipeline events into the
Noveum trace/span hierarchy.

Usage::

    from noveum_trace.integrations.pipecat import NoveumTraceObserver

    obs = NoveumTraceObserver()
    task = PipelineTask(pipeline, observers=[obs])
    await obs.attach_to_task(task)   # wire TurnTrackingObserver / UserBotLatencyObserver

``attach_to_task`` wires ``TurnTrackingObserver`` / ``UserBotLatencyObserver``
from the task so turn spans match Pipecat's turn boundaries.

Span hierarchy::

    Trace: pipecat.conversation
      Span: pipecat.turn  (one per conversation turn)
        Span: pipecat.stt
        Span: pipecat.llm
          attributes: llm.thoughts[], llm.thought_signatures[]  (model thinking blocks)
                      llm.function_calls[], llm.function_call_results[]  (tool calls)
        Span: pipecat.tts

Turn boundaries are detected in two modes:

- **External**: ``TurnTrackingObserver`` (and optionally ``UserBotLatencyObserver``)
  wired via :meth:`attach_to_task` or constructor args ``turn_tracking_observer`` /
  ``latency_observer``. Pipecat invokes event handlers as ``handler(emitter, *args)``.
- **Standalone**: internal VAD frame tracking replicates ``TurnTrackingObserver``
  logic (VAD started/stopped + BotStopped + timeout). (users should be using External mode only.
   In a further release standalone mode can even be removed)

Trace cleanup runs on ``EndFrame`` / ``CancelFrame`` via two complementary paths:

1. **Safety net (primary):** ``attach_to_task`` registers a ``on_pipeline_finished``
   event handler on the ``PipelineTask``. This fires inline in the main pipeline
   coroutine — before ``_cancel_tasks()`` kills the ``TaskObserver`` proxy tasks —
   so ``_finish_conversation`` is guaranteed to run regardless of proxy queue depth.

2. **Proxy path (secondary):** ``on_push_frame`` still handles ``EndFrame`` /
   ``CancelFrame`` as before. ``_finish_conversation`` is idempotent (no-op when
   ``_trace`` is already ``None``), so whichever path fires first wins and the other
   is a no-op.

Background: Pipecat's ``TaskObserver`` delivers ``FramePushed`` events to each
observer via a per-observer ``asyncio.Queue`` consumed by a dedicated asyncio task.
When ``task.cancel()`` races with a backlogged proxy queue, ``_cancel_tasks()`` can
kill the proxy task before it drains to the terminal-frame notification, causing the
trace to be silently lost. The ``on_pipeline_finished`` handler bypasses this queue.

Internal structure
------------------
Handler methods are split across focused mixin modules to keep each file small:

- :mod:`._handlers_stt`     — ``_STTHandlersMixin``
- :mod:`._handlers_llm`     — ``_LLMHandlersMixin``
- :mod:`._handlers_tts`     — ``_TTSHandlersMixin``
- :mod:`._handlers_metrics` — ``_MetricsHandlerMixin``
- :mod:`._turn_manager`     — ``_TurnManagerMixin``
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
import wave
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, Optional

from noveum_trace.integrations.pipecat._error_capture import _ErrorCaptureMixin
from noveum_trace.integrations.pipecat._handlers_llm import _LLMHandlersMixin
from noveum_trace.integrations.pipecat._handlers_metrics import _MetricsHandlerMixin
from noveum_trace.integrations.pipecat._handlers_stt import _STTHandlersMixin
from noveum_trace.integrations.pipecat._handlers_tts import _TTSHandlersMixin
from noveum_trace.integrations.pipecat._turn_manager import _TurnManagerMixin
from noveum_trace.integrations.pipecat.pipecat_constants import (
    DEFAULT_TURN_END_TIMEOUT_SECS,
    MAX_FRAME_DEDUP_HISTORY,
    MAX_STT_AUDIO_FRAMES,
    SPAN_CONVERSATION,
)

logger = logging.getLogger(__name__)

try:
    from pipecat.observers.base_observer import BaseObserver

    PIPECAT_AVAILABLE = True
except ImportError:
    PIPECAT_AVAILABLE = False

    class BaseObserver:  # type: ignore[no-redef]
        """Fallback stub when pipecat is not installed."""

        async def on_push_frame(self, data: Any) -> None:
            pass

        async def on_pipeline_started(self, pipeline: Any = None) -> None:
            pass

    logger.debug("Pipecat is not importable. Install it with: pip install pipecat-ai")


class NoveumTraceObserver(
    _STTHandlersMixin,
    _LLMHandlersMixin,
    _TTSHandlersMixin,
    _MetricsHandlerMixin,
    _ErrorCaptureMixin,  # must precede _TurnManagerMixin so _handle_error resolves here
    _TurnManagerMixin,
    BaseObserver,
):
    """
    Pipecat ``BaseObserver`` that maps pipeline frames to Noveum traces/spans.

    See module docstring for full span hierarchy and turn-detection mode details.
    """

    def __init__(
        self,
        trace_name_prefix: str = "pipecat",
        record_audio: bool = True,
        record_raw_input_audio: bool = True,
        capture_text: bool = True,
        capture_function_calls: bool = True,
        turn_end_timeout_secs: float = DEFAULT_TURN_END_TIMEOUT_SECS,
        turn_tracking_observer: Any = None,
        latency_observer: Any = None,
        auto_enable_metrics: bool = True,
        capture_errors: bool = True,
        capture_system_logs: bool = False,
        capture_session_metadata: bool = True,
        client: Any = None,
        deferred: bool = False,
        audio_sink: Optional[Callable[..., Awaitable[Any]]] = None,
        register_finish_safety_net: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the observer.

        Args:
            trace_name_prefix: Prefix for the conversation trace name.
            record_audio: When ``True``, buffers STT/TTS audio and uploads a WAV
                per span to ``/v1/audio``.
            record_raw_input_audio: When ``True`` (and ``record_audio=True``),
                also buffers pre-filter input audio via ``Noveum*Transport``
                wrappers and uploads as ``stt.raw_audio_uuid``.
            capture_text: When ``True``, accumulates LLM / TTS text buffers.
            capture_function_calls: When ``True``, creates ``function_call`` child
                spans.
            turn_end_timeout_secs: Seconds to wait after the bot stops speaking
                before closing a standalone-mode turn.
            turn_tracking_observer: Optional Pipecat ``TurnTrackingObserver`` to
                subscribe to for turn boundaries (or use :meth:`attach_to_task`).
            latency_observer: Optional ``UserBotLatencyObserver`` for
                ``on_latency_measured`` (or use :meth:`attach_to_task`).
            auto_enable_metrics: When ``True``, ``register_task_handlers``
                will automatically set ``enable_metrics=True`` and
                ``enable_usage_metrics=True`` on the task's ``PipelineParams`` if
                they are not already enabled, ensuring ``MetricsFrame`` emission.
            capture_errors: When ``True`` (default), ``ErrorFrame`` and
                ``FatalErrorFrame`` are recorded as span errors and trace events.
            capture_system_logs: When ``True``, ``SystemLogFrame`` entries
                at warning/error/critical level are appended as span events.
                Default ``False`` — opt-in because volume can be high.
            capture_session_metadata: When ``True`` (default), transport
                and runner metadata (room URL, transport type, idle timeout, etc.)
                are stamped onto the root conversation trace at connection time.
            client: Optional per-observer ``NoveumClient``. When set,
                :meth:`_get_client` resolves to it instead of the module-global
                client, so multiple observers in one process can each talk to a
                different Noveum project, and ``start_trace`` is called with
                ``set_as_current=False`` (per-call clients never write the
                shared current-trace context). When ``None`` (default), the
                global client from ``noveum_trace.init()`` is used as before.
            deferred: Declares that this observer's client uses a
                capture-only transport (``DeferredTransport``): the trace is
                built and finished normally but the export is retained in
                memory for :meth:`build_payload_snapshot` instead of being
                sent — the caller persists the snapshot and exports it later
                via ``NoveumClient.send_trace_dict``. The flag's enforcement
                job: deferred + ``record_audio=True`` REQUIRES ``audio_sink``
                (raises ``ValueError`` otherwise), because a capture-only
                transport has nowhere durable to put audio and spans must not
                be stamped with audio uuids that reference nothing.
            audio_sink: Optional async callable that receives each recorded
                audio segment instead of it being uploaded to Noveum. Called as
                ``await audio_sink(wav_bytes=..., audio_uuid=..., kind=...,
                trace_id=..., span_id=..., metadata=...)`` and must return a
                truthy value on success. ``kind`` is one of ``"stt"``,
                ``"stt_raw"``, ``"tts"``, ``"conversation"``. Span audio-uuid
                attributes are stamped exactly as in the upload path, so a sink
                that stores the bytes can upload them later under the same uuid.
            register_finish_safety_net: When ``True`` (default),
                :meth:`attach_to_task` registers the ``on_pipeline_finished``
                safety-net handler that guarantees trace teardown when
                Pipecat's proxy queue is cancelled. Hosts that call
                ``_finish_conversation`` themselves at a well-ordered point
                (e.g. dograh's ``on_call_finished``, which runs after the
                host's ``stop_recording()``) pass ``False`` so teardown cannot
                fire before their audio flush.

            Deferred-host recipe for dograh— a platform embedding this SDK (running
            many tenants' calls on shared workers, exporting out-of-band)
            typically sets all four together:

                client=NoveumClient(transport_instance=DeferredTransport())
                    Per-call injected client with a capture-only transport,
                    NOT the module-global client: shared workers run concurrent
                    calls, so a global client would race credentials and
                    cross-contaminate trace context. The trace is finished in
                    memory (``transport.captured_trace``) with zero network I/O.
                deferred=True
                    Arms the ``audio_sink``-required guard above.
                audio_sink=<sink>
                    Each segment is handed to the host's own store during the
                    call instead of uploaded to Noveum; the host exports the
                    parked bytes later under the same uuid.
                register_finish_safety_net=False
                    The host is the sole, well-ordered finisher (see above).

                The host then persists ``build_payload_snapshot()`` and later
                exports it with a REAL-credentialed client via
                ``NoveumClient.send_trace_dict`` / ``send_audio_sync``.
            **kwargs: Forwarded to Pipecat ``BaseObserver`` / ``BaseObject``
                (e.g. ``name=``).
        """
        super().__init__(**kwargs)

        # Per-observer client / deferred-export configuration (see docstring).
        self._injected_client = client
        self._deferred = deferred
        self._audio_sink = audio_sink
        self._register_finish_safety_net = register_finish_safety_net

        if deferred and record_audio and audio_sink is None:
            raise ValueError(
                "deferred=True with record_audio=True requires an audio_sink: "
                "a capture-only transport cannot store audio, and spans must "
                "not carry audio uuids that reference nothing. Pass an "
                "audio_sink or set record_audio=False."
            )

        self._trace_name_prefix = trace_name_prefix
        self._record_audio = record_audio
        self._record_raw_input_audio = record_raw_input_audio
        self._capture_text = capture_text
        self._capture_function_calls = capture_function_calls
        self._turn_end_timeout_secs = turn_end_timeout_secs

        self._auto_enable_metrics = auto_enable_metrics
        self._capture_errors = capture_errors
        self._capture_system_logs = capture_system_logs
        self._capture_session_metadata = capture_session_metadata

        # Distinct error messages already reflected in the native trace
        # ``error_count`` (D1) — dedupes the multiple ErrorFrames Pipecat emits for
        # a single provider failure so ``error_count`` counts failures, not frames.
        self._native_error_messages: set[str] = set()

        # ------------------------------------------------------------------ #
        # Conversation-level state                                            #
        # ------------------------------------------------------------------ #
        self._trace: Any = None

        # Turn state
        self._current_turn_span: Any = None
        self._current_turn_number: int = 0
        self._turn_start_time: Optional[float] = None
        # MetricsFrame with TurnMetricsData can arrive before _current_turn_span exists;
        # merge here and flush onto the next turn span in _start_new_turn.
        self._pending_turn_eou_metrics: dict[str, Any] = {}

        # Active operation spans
        self._active_llm_span: Any = None
        self._active_tts_span: Any = None
        # tool_call_id → call dict (kept for InProgress dedup + result enrichment)
        self._pending_function_calls: dict[str, dict[str, Any]] = {}
        # tool_call_id → the LLM span that REQUESTED the call, so a result/cancel
        # lands on the requesting span even after a follow-up span has opened.
        self._function_call_owner: dict[str, Any] = {}
        # tool_call_ids whose result/cancel has been written (dedup double-broadcast).
        self._resolved_function_call_ids: set[str] = set()

        # Backrefs to the most-recently-closed LLM/TTS span.
        #
        # Pipecat emits MetricsFrame AFTER LLMFullResponseEndFrame / TTSStoppedFrame,
        # so by the time token counts and character counts arrive the active span is
        # already None.  _last_llm_span is also used by _handle_function_call_start to
        # write function-call data to span 1 when the frame arrives after it has closed.
        # Both refs are set when the respective span closes and cleared when the next
        # span of the same type opens.
        self._last_llm_span: Any = None  # backref to most-recently-closed LLM span
        self._last_tts_span: Any = None  # backref to most-recently-closed TTS span

        # Text buffers
        self._llm_text_buffer: list[str] = []
        # TTS text buffers hold (text, includes_inter_frame_spaces) tuples so spacing
        # can be reconstructed; final = sentence-aggregated, interim = word/token.
        self._tts_text_buffer: list[tuple[str, bool]] = []
        self._tts_text_interim_buffer: list[tuple[str, bool]] = []
        self._transcription_buffer: list[str] = []

        # LLM context stash (filled by LLMContextFrame, flushed on LLMFullResponseStartFrame)
        self._pending_llm_context: dict[str, Any] = {}

        # LLM thought accumulation (flattened onto the LLM span as attribute lists)
        self._llm_thought_buffer: list[str] = []
        self._llm_thoughts_list: list[str] = []
        self._llm_thought_signatures_list: list[str] = []
        self._pending_thought_signatures: list[str] = []

        # Audio buffers (populated only when record_audio=True)
        self._stt_audio_buffer: list[Any] = []
        self._stt_raw_audio_buffer: list[Any] = []
        self._pipeline_has_stt: bool = False
        self._warned_no_stt_for_raw: bool = False
        self._tts_audio_buffer: list[Any] = []
        # Source processor that opened the current TTS span (set on TTSStartedFrame,
        # cleared on TTSStoppedFrame / interruption).  Audio frames are only buffered
        # when they come from this exact processor so that downstream resamplers /
        # aggregators that re-emit TTSAudioRawFrame with new IDs are ignored.
        self._tts_source_processor: Any = None

        # Set of tasks we have registered the on_pipeline_finished safety-net on.
        # Used as a dedup guard so repeated attach_to_task() calls on the same
        # task are no-ops, even if attach_to_task(task1), attach_to_task(task2),
        # attach_to_task(task1) is called in sequence.
        self._registered_pipeline_tasks: set[Any] = set()
        # Serializes conversation teardown. EndFrame/CancelFrame (on_push_frame)
        # and the on_pipeline_finished safety-net can fire concurrently on
        # Pipecat 1.x; without this lock both pass the ``_trace is None`` guard,
        # then interleave across the teardown's awaits and one nulls ``_trace``
        # out from under the other (AttributeError). The lock makes the whole
        # teardown run exactly once.
        #
        # Lazily created on first use (not here): on Python < 3.10 ``asyncio.Lock()``
        # binds to the current event loop at construction, but the observer is
        # usually built outside any loop (setup time) and run under a different
        # loop, which would raise "got Future attached to a different loop".
        self._finish_lock: Optional[asyncio.Lock] = None

        # Custom span processor (set by NoveumPipecatTracer.observe_pipeline when
        # capture_custom_spans=True; None otherwise).
        self._custom_span_processor: Any = None

        # Transport reference and session metadata buffer.
        # _store_transport() populates this dict at register_task_handlers time;
        # _flush_session_metadata() writes it to self._trace at first connection.
        self._transport: Any = None
        self._session_metadata: dict[str, Any] = {}

        # Full-conversation audio (populated via AudioBufferProcessor wired in attach_to_task)
        # Holds raw PCM bytes from on_audio_data until EndFrame flushes them.
        self._audio_buffer_processor: Any = None
        self._abp_is_recording: bool = (
            False  # track internally, don't probe private _recording
        )
        self._conversation_audio_chunks: list[bytes] = []
        self._conversation_audio_sample_rate: Optional[int] = None
        self._conversation_audio_num_channels: Optional[int] = None

        # STT span lifecycle state
        # set True on SpeechControlParamsFrame with vad_params
        self._vad_present: bool = False
        # long-lived STT span (open from VADUserStartedSpeaking → TranscriptionFrame)
        self._active_stt_span: Any = None
        # Source processor that first sent UserAudioRawFrame; used to filter
        # re-emitted frames from downstream processors (same logic as TTS).
        self._stt_source_processor: Any = None
        # Monotonic time at VADUserStartedSpeaking (STT path) for latency attrs
        self._vad_speech_start_time: Optional[float] = None
        # Pairs of interim text + confidence per utterance (JSON on final span)
        self._stt_interim_results: list[dict[str, Any]] = []
        self._stt_first_text_latency_recorded: bool = False

        # Conversation-wide metrics accumulator
        self._metrics_accumulator: dict[str, Any] = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "turn_count": 0,
        }

        # ------------------------------------------------------------------ #
        # Turn detection helpers                                              #
        # ------------------------------------------------------------------ #
        self._turn_tracker: Any = None
        self._latency_tracker: Any = None
        self._using_external_turn_tracking: bool = False

        # Standalone VAD state
        self._is_bot_speaking: bool = False
        self._bot_has_spoken_in_turn: bool = False
        self._user_stopped_speaking_time: Optional[float] = None
        self._turn_end_task: Optional[asyncio.Task[None]] = None

        # Frame deduplication (mirrors TurnTrackingObserver's own guard)
        self._processed_frame_ids: set[int] = set()
        self._frame_id_history: deque[int] = deque(maxlen=MAX_FRAME_DEDUP_HISTORY)

        # ------------------------------------------------------------------ #
        # Frame dispatch table                                                #
        # ------------------------------------------------------------------ #
        self._frame_handlers: dict[type, Any] = {}
        self._setup_dispatch_table()

        if turn_tracking_observer is not None:
            self.attach_turn_tracking_observer(turn_tracking_observer)
        if latency_observer is not None:
            self.attach_latency_observer(latency_observer)

    # ---------------------------------------------------------------------- #
    # Client accessor                                                         #
    # ---------------------------------------------------------------------- #

    def _get_client(self) -> Any:
        """
        Resolve the client for this observer: the injected per-observer client
        if one was passed to the constructor, else the module-global client
        (``None`` when the SDK is uninitialized).
        """
        if self._injected_client is not None:
            return self._injected_client
        try:
            from noveum_trace import get_client

            return get_client()
        except Exception:
            return None

    def build_payload_snapshot(self) -> Optional[dict[str, Any]]:
        """
        Serialize the conversation trace to a plain dict (``Trace.to_dict()``).

        Intended for deferred hosts: call after the conversation has finished
        to get a JSON-serializable snapshot for later export via
        ``NoveumClient.send_trace_dict``. The finished trace is read from the
        client transport's ``captured_trace`` (the ``DeferredTransport``
        contract); falls back to the in-progress trace if the conversation has
        not finished yet. Returns ``None`` if no trace was ever created.
        """
        transport = getattr(self._get_client(), "transport", None)
        trace = getattr(transport, "captured_trace", None) or self._trace
        if trace is None:
            return None
        try:
            return trace.to_dict()
        except Exception as e:
            logger.warning("Failed to snapshot pipecat trace: %s", e, exc_info=True)
            return None

    async def _sink_segment_audio(
        self,
        frames: list[Any],
        audio_uuid: str,
        kind: str,
        trace_id: str,
        span_id: str,
    ) -> bool:
        """
        Encode one buffered audio segment to WAV (off the event loop) and hand it
        to the configured ``audio_sink``. Returns True on sink success. Callers
        must only invoke this when ``self._audio_sink`` is set.
        """
        try:
            from noveum_trace.integrations.pipecat.pipecat_utils import (
                _frames_to_wav_bytes,
                calculate_audio_duration_ms,
            )

            # Shallow-copy the frame list: teardown (_finish_conversation_impl,
            # a different task) clears the instance buffers, and the encoder
            # thread must not iterate a list being emptied under it. Frame
            # objects are shared (immutable PCM bytes) — only membership is
            # isolated.
            frames = list(frames)
            wav_bytes = await asyncio.to_thread(_frames_to_wav_bytes, frames)
            if not wav_bytes:
                logger.debug(
                    "Empty WAV bytes for %s %s, skipping sink", kind, audio_uuid
                )
                return False
            metadata = {
                "duration_ms": calculate_audio_duration_ms(frames),
                "format": "wav",
                "type": kind,
            }
            return bool(
                await self._audio_sink(
                    wav_bytes=wav_bytes,
                    audio_uuid=audio_uuid,
                    kind=kind,
                    trace_id=trace_id,
                    span_id=span_id,
                    metadata=metadata,
                )
            )
        except Exception as e:
            logger.warning(
                "Audio sink failed for %s %s: %s", kind, audio_uuid, e, exc_info=True
            )
            return False

    def _bounded_append_stt_frame(self, buffer: list[Any], frame: Any) -> None:
        """Append an audio frame to an STT buffer, dropping oldest when over cap."""
        buffer.append(frame)
        overflow = len(buffer) - MAX_STT_AUDIO_FRAMES
        if overflow > 0:
            del buffer[:overflow]

    def capture_raw_input_audio(self, frame: Any) -> None:
        """
        Buffer a snapshot of pre-filter input audio (called from transport mixin).

        No-op unless ``record_audio``, ``record_raw_input_audio``, and an STT
        processor were detected in the pipeline at :meth:`attach_to_task`.
        """
        if not self._record_audio or not self._record_raw_input_audio:
            return
        if not self._pipeline_has_stt:
            return
        audio = getattr(frame, "audio", None)
        if not audio:
            return
        try:
            from pipecat.frames.frames import InputAudioRawFrame
        except ImportError:
            return
        snapshot = InputAudioRawFrame(
            audio=bytes(audio),
            sample_rate=getattr(frame, "sample_rate", None),
            num_channels=getattr(frame, "num_channels", None),
        )
        self._bounded_append_stt_frame(self._stt_raw_audio_buffer, snapshot)

    @staticmethod
    def _processor_is_stt(proc: Any) -> bool:
        for base in type(proc).__mro__:
            name = getattr(base, "__name__", "")
            if name == "STTService" or name.endswith("STTService"):
                return True
        return False

    def _detect_pipeline_has_stt(self, task: Any) -> None:
        """Set ``_pipeline_has_stt`` by scanning the task pipeline for STT services."""
        pipeline = getattr(task, "_pipeline", None) or getattr(task, "pipeline", None)
        if pipeline is None:
            self._pipeline_has_stt = False
            return
        self._pipeline_has_stt = any(
            self._processor_is_stt(proc)
            for proc in self._iter_nested_processors(pipeline)
        )
        if (
            self._record_raw_input_audio
            and not self._pipeline_has_stt
            and not self._warned_no_stt_for_raw
        ):
            self._warned_no_stt_for_raw = True
            logger.warning(
                "record_raw_input_audio=True but no STT service found in pipeline — "
                "raw input audio will not be buffered"
            )

    # ---------------------------------------------------------------------- #
    # External observer wiring (public API)                                  #
    # ---------------------------------------------------------------------- #

    async def attach_to_task(self, task: Any) -> None:
        """
        Subscribe to ``PipelineTask``-managed observers when present.

        Uses ``task.turn_tracking_observer`` and the task's internal
        ``_user_bot_latency_observer`` (when ``enable_tracing`` added latency
        tracking). Also auto-detects an ``AudioBufferProcessor`` in the pipeline
        to enable full-conversation stereo audio recording.

        Registers an ``on_pipeline_finished`` safety-net handler on the task so
        ``_finish_conversation`` is guaranteed to run even when Pipecat's
        ``TaskObserver`` proxy queue is cancelled before draining (the
        ``task.cancel()`` race condition).

        Call from async code after constructing ``PipelineTask`` and this observer,
        before ``runner.run(task)``, so conversation audio recording can start
        before the pipeline processes PCM.

        This delegates all synchronous wiring to :meth:`attach_to_task_sync` and
        then performs the one genuinely-async step — starting
        ``AudioBufferProcessor`` recording — so hosts running inside an event
        loop can attach everything with a single ``await``.

        Safe to call multiple times; repeated calls with the same observers are
        no-ops.
        """
        # attach_to_task_sync registers all observers, the safety net, and the
        # audio on_audio_data handler (setting self._audio_buffer_processor).
        self.attach_to_task_sync(task)

        # The only genuinely-async step: start AudioBufferProcessor recording.
        # _ensure_audio_buffer_recording is idempotent and a no-op when no ABP
        # was attached, so this safely covers the fresh/same/absent-proc cases.
        if self._record_audio:
            await self._ensure_audio_buffer_recording()

    def attach_to_task_sync(self, task: Any) -> None:
        """
        Synchronous variant of :meth:`attach_to_task` for hosts wiring the
        observer OUTSIDE an event loop.

        Registers everything that does not require ``await``: it resets the
        transport's captured trace, subscribes to the turn-tracking and latency
        observers, installs the ``on_pipeline_finished`` safety-net handler,
        detects whether the pipeline has an STT service, and — when
        ``record_audio=True`` — registers ``_on_conversation_audio`` on the
        ``AudioBufferProcessor``'s ``on_audio_data`` event.

        It does NOT start ``AudioBufferProcessor`` recording, because
        ``start_recording()`` is asynchronous. The HOST is responsible for
        calling ``AudioBufferProcessor.start_recording()`` before conversation
        audio flows, OR use the async :meth:`attach_to_task` which starts
        recording for you.

        Safe to call multiple times; repeated calls with the same observers are
        no-ops.
        """
        if task is None:
            return

        # Reuse guard (see also on_pipeline_started): attaching to a task marks
        # the start of a new conversation — drop any previous conversation's
        # capture so a conversation that dies before StartFrame cannot serve
        # the prior conversation's trace from build_payload_snapshot().
        transport = getattr(self._get_client(), "transport", None)
        if transport is not None and hasattr(transport, "captured_trace"):
            transport.captured_trace = None

        tto = getattr(task, "turn_tracking_observer", None)
        if tto is not None:
            self.attach_turn_tracking_observer(tto)
        lto = getattr(task, "_user_bot_latency_observer", None)
        if lto is not None:
            self.attach_latency_observer(lto)

        self._register_finish_safety_net_handler(task)

        self._detect_pipeline_has_stt(task)

        # Auto-detect AudioBufferProcessor and register the on_audio_data handler
        # (synchronous). Recording is started separately (async).
        if self._record_audio:
            self._attach_audio_buffer_handler_sync(task)

    def _register_finish_safety_net_handler(self, task: Any) -> None:
        """
        Register the ``on_pipeline_finished`` safety-net handler on ``task``.

        The safety net guarantees ``_finish_conversation`` runs even when
        Pipecat cancels the ``TaskObserver`` proxy queue before it drains.
        Synchronous: ``task.event_handler`` registration is not awaited.
        """
        # ---------------------------------------------------------------------- #
        # Safety net: on_pipeline_finished fires inline in the main pipeline      #
        # coroutine, before _cancel_tasks() kills the TaskObserver proxy tasks.   #
        # _finish_conversation is idempotent, so whichever path fires first wins. #
        # ---------------------------------------------------------------------- #
        if not self._register_finish_safety_net:
            # Host opted out (register_finish_safety_net=False): it calls
            # _finish_conversation itself at a point ordered AFTER its own
            # audio flush, and this handler firing earlier would finish the
            # trace with incomplete conversation audio on cancelled calls.
            logger.debug(
                "on_pipeline_finished safety net disabled by host — teardown "
                "relies on the frame path and the host's explicit finish"
            )
        elif hasattr(task, "event_handler"):
            # Deduplication guard: check our own set of already-registered tasks
            # so repeated attach_to_task() calls on the same task are no-ops,
            # regardless of interleaving with other tasks.
            if task in self._registered_pipeline_tasks:
                logger.debug(
                    "on_pipeline_finished safety-net already registered for this "
                    "observer+task pair — skipping duplicate registration"
                )
            else:
                try:
                    _CancelFrame: Any = None
                    try:
                        from pipecat.frames.frames import CancelFrame as _CF

                        _CancelFrame = _CF
                    except ImportError:
                        pass

                    observer_ref = self

                    @task.event_handler("on_pipeline_finished")
                    async def _on_pipeline_finished(task_ref: Any, frame: Any) -> None:
                        is_cancel = _CancelFrame is not None and isinstance(
                            frame, _CancelFrame
                        )
                        logger.debug(
                            "on_pipeline_finished fired (frame=%s, cancelled=%s) — "
                            "ensuring trace cleanup via safety net",
                            type(frame).__name__,
                            is_cancel,
                        )
                        await observer_ref._finish_conversation(cancelled=is_cancel)

                    self._registered_pipeline_tasks.add(task)
                    logger.debug(
                        "Registered on_pipeline_finished safety-net handler on task"
                    )
                except Exception as e:
                    logger.warning(
                        "Could not register on_pipeline_finished safety-net handler: %s",
                        e,
                    )
        else:
            logger.debug(
                "PipelineTask does not expose event_handler(); "
                "on_pipeline_finished safety net not registered — "
                "trace cleanup relies on on_push_frame CancelFrame/EndFrame path only"
            )

    def _iter_nested_processors(self, node: Any) -> Iterator[Any]:
        """Depth-first walk of Pipecat compound processors (Pipeline inside Pipeline)."""
        procs = (
            getattr(node, "processors", None)
            or getattr(node, "_processors", None)
            or []
        )
        for proc in procs:
            yield proc
            yield from self._iter_nested_processors(proc)

    def _attach_audio_buffer_handler_sync(self, task: Any) -> None:
        """
        Synchronous part of AudioBufferProcessor wiring.

        Walks the task's pipeline processors looking for an
        ``AudioBufferProcessor``. If a NEW one is found, registers
        ``_on_conversation_audio`` on its ``on_audio_data`` event, resets the
        conversation-audio buffers, and sets ``self._audio_buffer_processor`` so
        the full stereo conversation WAV is captured on session end.

        Does NOT start recording — ``start_recording()`` is asynchronous and is
        handled by :meth:`_attach_audio_buffer_from_pipeline` /
        :meth:`_ensure_audio_buffer_recording`. When the same ABP is already
        attached, this is a no-op (``self._audio_buffer_processor`` is left
        unchanged so the async caller can ensure recording).

        If ``record_audio=True`` but no ``AudioBufferProcessor`` is present, logs
        a warning so the user knows conversation-level audio won't be captured.
        """
        pipeline = getattr(task, "_pipeline", None) or getattr(task, "pipeline", None)
        if pipeline is None:
            if self._record_audio:
                logger.warning(
                    "attach_to_task: could not access pipeline processors — "
                    "full-conversation audio will not be recorded. "
                    "Add AudioBufferProcessor(num_channels=2) to your pipeline."
                )
            return

        # RTVI layouts: Task → RTVIProcessor → inner Pipeline; ABP is inside the inner one.
        found_proc: Any = None
        for proc in self._iter_nested_processors(pipeline):
            # Pipecat may wrap/subclass AudioBufferProcessor; accept subclasses too.
            if any(
                base.__name__ == "AudioBufferProcessor" for base in type(proc).__mro__
            ):
                found_proc = proc
                break

        if found_proc is None:
            if self._record_audio:
                logger.warning(
                    "No AudioBufferProcessor found in pipeline — full-conversation audio "
                    "will not be recorded. Add AudioBufferProcessor(num_channels=2) to "
                    "your pipeline and await attach_to_task() before runner.run()."
                )
            return

        # If the same ABP is still active for this observer, don't re-register the
        # handler. Leave self._audio_buffer_processor unchanged so the async
        # caller detects "same proc" and ensures recording instead.
        if found_proc is self._audio_buffer_processor:
            return

        # Swap ABP when a new PipelineTask (or changed pipeline) is attached.
        # Note: Pipecat may not support handler removal; _on_conversation_audio guards
        # against writing audio from stale processors.
        prev_proc = self._audio_buffer_processor
        self._audio_buffer_processor = found_proc
        self._conversation_audio_chunks = []
        self._conversation_audio_sample_rate = None
        self._conversation_audio_num_channels = None

        try:
            self._audio_buffer_processor.add_event_handler(
                "on_audio_data", self._on_conversation_audio
            )
            logger.info(
                "Noveum trace: full-conversation audio attached (nested pipeline OK)"
            )
        except Exception as e:
            logger.warning(
                "Failed to attach to AudioBufferProcessor: %s", e, exc_info=True
            )
            # If the new ABP couldn't be wired, prefer keeping the previous one.
            self._audio_buffer_processor = prev_proc
            return

    async def _attach_audio_buffer_from_pipeline(self, task: Any) -> None:
        """
        Walk the task's pipeline processors looking for an ``AudioBufferProcessor``.

        If found, register ``_on_conversation_audio`` on its ``on_audio_data``
        event so the full stereo conversation WAV is captured on session end and
        start recording (the one async step).

        If ``record_audio=True`` but no ``AudioBufferProcessor`` is present, logs
        a warning so the user knows conversation-level audio won't be captured.
        """
        prev_proc = self._audio_buffer_processor
        # Synchronous walk + handler registration (sets _audio_buffer_processor
        # when a new proc is found; leaves it unchanged for same/absent proc).
        self._attach_audio_buffer_handler_sync(task)

        proc = self._audio_buffer_processor
        if proc is None:
            return

        # Same ABP already attached (or none newly found): ensure recording.
        if proc is prev_proc:
            await self._ensure_audio_buffer_recording()
            return

        # Newly-attached proc: start recording directly. ABP drops
        # InputAudio/OutputAudio until start_recording(); observer on_push_frame
        # runs after process_frame, so start here before run().
        try:
            await proc.start_recording()
            self._abp_is_recording = True
            logger.debug(
                "Attached to AudioBufferProcessor; start_recording() completed"
            )
        except Exception as e:
            logger.warning(
                "Failed to start AudioBufferProcessor recording: %s",
                e,
                exc_info=True,
            )

    async def _ensure_audio_buffer_recording(self) -> None:
        """
        Ensure AudioBufferProcessor is recording.

        If ``record_audio=False`` we must never call ``start_recording`` even if
        callers manually attach an ``AudioBufferProcessor`` for testing/experiments.

        ``attach_to_task`` awaits ``start_recording()`` before ``runner.run``.  After
        ``stop_recording()`` (EndFrame), ``_abp_is_recording`` is False — call
        ``await start_recording()`` again.
        Skips if already recording to avoid ``start_recording()`` wiping buffers.
        """
        if not self._record_audio:
            return
        proc = self._audio_buffer_processor
        if proc is None:
            return

        # Prefer Pipecat's private `_recording` state when present: it is the
        # most reliable source of truth for whether `start_recording()` should
        # be called.
        if hasattr(proc, "_recording") and getattr(proc, "_recording", False):
            self._abp_is_recording = True
            return

        # Fall back to our internal tracking if Pipecat doesn't expose `_recording`.
        if not hasattr(proc, "_recording") and self._abp_is_recording:
            return
        try:
            await proc.start_recording()
            self._abp_is_recording = True
            logger.debug("AudioBufferProcessor recording restarted after session stop")
        except Exception as e:
            logger.warning(
                "Failed to start AudioBufferProcessor recording: %s", e, exc_info=True
            )

    async def _on_conversation_audio(
        self, processor: Any, audio: bytes, sample_rate: int, num_channels: int
    ) -> None:
        """
        ``AudioBufferProcessor.on_audio_data`` callback.

        Stores the raw PCM bytes in memory until ``_finish_conversation`` wraps
        them into a WAV and uploads them as the conversation-level recording.
        Chunks are concatenated when ``buffer_size > 0`` triggers multiple flushes.
        """
        # Ignore audio from stale processors after the observer swaps ABP between tasks.
        # Allow direct calls/tests when we haven't attached an ABP yet.
        if (
            self._audio_buffer_processor is not None
            and processor is not self._audio_buffer_processor
        ):
            return
        self._conversation_audio_chunks.append(audio)
        self._conversation_audio_sample_rate = sample_rate
        self._conversation_audio_num_channels = num_channels
        total = sum(len(chunk) for chunk in self._conversation_audio_chunks)
        logger.debug(
            "Conversation audio chunk: %d bytes (total %d), %d Hz, %d ch",
            len(audio),
            total,
            sample_rate,
            num_channels,
        )

    def attach_turn_tracking_observer(self, turn_tracker: Any) -> None:
        """Subscribe to ``TurnTrackingObserver`` turn events (external turn mode)."""
        self._attach_turn_tracker(turn_tracker)

    def attach_latency_observer(self, latency_tracker: Any) -> None:
        """Subscribe to ``UserBotLatencyObserver.on_latency_measured``."""
        self._attach_latency_tracker(latency_tracker)

    # ---------------------------------------------------------------------- #
    # Dispatch table setup                                                    #
    # ---------------------------------------------------------------------- #

    def _setup_dispatch_table(self) -> None:
        """Build the frame-type → handler mapping after Pipecat imports."""
        if not PIPECAT_AVAILABLE:
            return
        try:
            import pipecat.frames.frames as _ff

            def _reg(name: str, handler: Any) -> None:
                cls = getattr(_ff, name, None)
                if cls is not None:
                    self._frame_handlers[cls] = handler

            # ---------------------------------------------------------------- #
            # Core pipeline lifecycle                                           #
            # ---------------------------------------------------------------- #
            _reg("StartFrame", self._handle_start_frame)
            _reg("EndFrame", self._handle_end_frame)
            _reg("StopFrame", self._handle_stop_frame)
            _reg("CancelFrame", self._handle_cancel_frame)

            # ---------------------------------------------------------------- #
            # STT                                                               #
            # ---------------------------------------------------------------- #
            # SpeechControlParamsFrame carries vad_params at pipeline start;
            # used to detect whether a VAD processor is present.
            _reg("SpeechControlParamsFrame", self._handle_speech_control_params)
            _reg("TranscriptionFrame", self._handle_transcription)
            _reg("InterimTranscriptionFrame", self._handle_interim_transcription)
            _reg("InputTextRawFrame", self._handle_input_text)
            _reg("STTMetadataFrame", self._handle_stt_metadata)

            # ---------------------------------------------------------------- #
            # LLM — context stash (precedes LLMFullResponseStartFrame)         #
            # ---------------------------------------------------------------- #
            _reg("LLMContextFrame", self._handle_llm_context)
            # OpenAILLMContextFrame lives in openai_llm_context, not frames.py
            try:
                from pipecat.processors.aggregators.openai_llm_context import (
                    OpenAILLMContextFrame as _OAILLMFrame,
                )

                self._frame_handlers[_OAILLMFrame] = self._handle_llm_context
            except ImportError:
                pass

            # Legacy / explicit message + tool frames (Path B — no LLMContextFrame)
            _reg("LLMMessagesFrame", self._handle_llm_messages_replace)
            _reg("LLMMessagesUpdateFrame", self._handle_llm_messages_replace)
            _reg("LLMMessagesAppendFrame", self._handle_llm_messages_append)
            _reg("LLMSetToolsFrame", self._handle_llm_set_tools)
            _reg("LLMSetToolChoiceFrame", self._handle_llm_set_tool_choice)
            _reg("LLMContextSummaryRequestFrame", self._handle_llm_summary_request)

            # ---------------------------------------------------------------- #
            # LLM — response stream                                            #
            # ---------------------------------------------------------------- #
            _reg("LLMFullResponseStartFrame", self._handle_llm_response_start)
            _reg("LLMTextFrame", self._handle_llm_text)
            _reg("LLMFullResponseEndFrame", self._handle_llm_response_end)

            # Vision subclasses — route to same LLM handlers (fixes silent drop)
            _reg("VisionFullResponseStartFrame", self._handle_llm_response_start)
            _reg("VisionTextFrame", self._handle_llm_text)
            _reg("VisionFullResponseEndFrame", self._handle_llm_response_end)

            # ---------------------------------------------------------------- #
            # LLM — thought blocks (Anthropic extended thinking, o1-style)     #
            # ---------------------------------------------------------------- #
            _reg("LLMThoughtStartFrame", self._handle_llm_thought_start)
            _reg("LLMThoughtTextFrame", self._handle_llm_thought_text)
            _reg("LLMThoughtEndFrame", self._handle_llm_thought_end)

            # ---------------------------------------------------------------- #
            # Function calls                                                    #
            # ---------------------------------------------------------------- #
            _reg("FunctionCallsStartedFrame", self._handle_function_calls_started)
            _reg("FunctionCallInProgressFrame", self._handle_function_call_start)
            _reg("FunctionCallResultFrame", self._handle_function_call_result)
            _reg("FunctionCallCancelFrame", self._handle_function_call_cancel)

            # ---------------------------------------------------------------- #
            # Context summarization                                             #
            # ---------------------------------------------------------------- #
            _reg("LLMContextSummaryResultFrame", self._handle_llm_summary_result)

            # ---------------------------------------------------------------- #
            # TTS                                                               #
            # ---------------------------------------------------------------- #
            _reg("TTSStartedFrame", self._handle_tts_started)
            _reg("TTSTextFrame", self._handle_tts_text)
            _reg("TTSAudioRawFrame", self._handle_tts_audio)
            _reg("TTSStoppedFrame", self._handle_tts_stopped)

            # ---------------------------------------------------------------- #
            # Metrics                                                           #
            # ---------------------------------------------------------------- #
            _reg("MetricsFrame", self._handle_metrics)
            # LLMUsageMetricsFrame — standalone token-usage frame added in newer
            # Pipecat versions (may not exist in all releases).
            try:
                from pipecat.frames.frames import (  # noqa: PLC0415
                    LLMUsageMetricsFrame as _LLMUsageMetricsFrame,
                )

                self._frame_handlers[_LLMUsageMetricsFrame] = (
                    self._handle_llm_usage_metrics
                )
            except ImportError:
                pass

            # ---------------------------------------------------------------- #
            # VAD / speaking state                                              #
            # ---------------------------------------------------------------- #
            # Both the turn-manager and the STT span lifecycle need to react to
            # these frames, so we fan out to both handlers via thin wrappers.
            async def _vad_started_combined(data: Any) -> None:
                await self._handle_vad_user_started(data)
                await self._handle_vad_stt_start(data)

            async def _vad_stopped_combined(data: Any) -> None:
                await self._handle_vad_user_stopped(data)
                await self._handle_vad_stt_stop(data)

            _reg("VADUserStartedSpeakingFrame", _vad_started_combined)
            _reg("VADUserStoppedSpeakingFrame", _vad_stopped_combined)
            _reg("UserStartedSpeakingFrame", self._handle_user_started_speaking)
            _reg("UserStoppedSpeakingFrame", self._handle_user_stopped_speaking)
            _reg("BotStartedSpeakingFrame", self._handle_bot_started_speaking)
            _reg("BotStoppedSpeakingFrame", self._handle_bot_stopped_speaking)

            # ---------------------------------------------------------------- #
            # Mute events                                                       #
            # ---------------------------------------------------------------- #
            _reg("UserMuteStartedFrame", self._handle_user_mute_started)
            _reg("UserMuteStoppedFrame", self._handle_user_mute_stopped)

            # ---------------------------------------------------------------- #
            # Session / transport events                                        #
            # ---------------------------------------------------------------- #
            _reg("ClientConnectedFrame", self._handle_client_connected)
            _reg("BotConnectedFrame", self._handle_bot_connected)

            # ---------------------------------------------------------------- #
            # Errors / interruptions                                            #
            # ---------------------------------------------------------------- #
            _reg("InterruptionFrame", self._handle_interruption)
            # Subclass of InterruptionFrame — exact-type dispatch would miss it
            _reg("StartInterruptionFrame", self._handle_interruption)
            _reg("ErrorFrame", self._handle_error)
            # FatalErrorFrame is a subclass of ErrorFrame; register explicitly
            _reg("FatalErrorFrame", self._handle_error)
            # SystemLogFrame — structured pipeline log messages
            _reg("SystemLogFrame", self._handle_system_log)

            # User audio frame name varies across pipecat versions
            for frame_name in ("UserAudioRawFrame", "InputAudioRawFrame"):
                cls = getattr(_ff, frame_name, None)
                if cls and cls not in self._frame_handlers:
                    self._frame_handlers[cls] = self._handle_user_audio

        except ImportError as e:
            logger.warning("Failed to set up NoveumTraceObserver dispatch table: %s", e)

    # ---------------------------------------------------------------------- #
    # BaseObserver hooks                                                      #
    # ---------------------------------------------------------------------- #

    async def on_pipeline_started(self, *args: Any, **kwargs: Any) -> None:
        """Create the conversation-level trace when the pipeline starts."""
        if self._trace is not None:
            return
        try:
            client = self._get_client()
            if not client:
                logger.debug("No Noveum client available, skipping trace creation")
                return

            # Reuse guard: on a capture transport (DeferredTransport), drop the
            # previous conversation's capture so build_payload_snapshot() can
            # never serve call N-1's trace as call N's (e.g. call N dying
            # before its own finish).
            transport = getattr(client, "transport", None)
            if transport is not None and hasattr(transport, "captured_trace"):
                transport.captured_trace = None

            trace_name = f"{self._trace_name_prefix}.{SPAN_CONVERSATION.split('.')[-1]}"
            # Per-observer (injected) clients never write the global
            # current-trace context: in a many-calls-per-process host that
            # contextvar is shared state, and the observer builds spans
            # directly on the Trace object so it does not need it.
            self._trace = client.start_trace(
                name=trace_name,
                set_as_current=(self._injected_client is None),
            )
            logger.debug("Created pipecat conversation trace: %s", self._trace.trace_id)
        except Exception as e:
            logger.warning("Failed to create pipecat trace: %s", e, exc_info=True)

    async def on_push_frame(self, data: Any) -> None:
        """Route each ``FramePushed`` event to the appropriate handler."""
        try:
            frame = getattr(data, "frame", None)
            # If attach ran before AudioBufferProcessor existed, or recording was
            # stopped and a new session begins without re-attach.
            proc_ab = self._audio_buffer_processor
            if (
                self._record_audio
                and proc_ab is not None
                and frame is not None
                and not getattr(proc_ab, "_recording", False)
            ):
                fn = type(frame).__name__
                if fn in ("InputAudioRawFrame", "OutputAudioRawFrame"):
                    await self._ensure_audio_buffer_recording()

            # EndFrame/CancelFrame are observed once per hop. Finishing the trace on the
            # first hop runs _upload_full_conversation_audio before AudioBufferProcessor
            # has run stop_recording() → on_audio_data never populated in time. Wait until
            # the buffer processor pushes downstream (after flush).
            if frame is not None:
                lname = type(frame).__name__
                if lname in ("EndFrame", "CancelFrame"):
                    abp = self._audio_buffer_processor
                    if abp is not None and getattr(data, "source", None) is not abp:
                        return

            fid = getattr(frame, "id", None) if frame is not None else None
            if fid is not None and fid in self._processed_frame_ids:
                logger.debug(
                    "Dedup: dropping duplicate frame id=%s type=%s",
                    fid,
                    type(frame).__name__,
                )
                return

            if fid is not None:
                self._processed_frame_ids.add(fid)
                self._frame_id_history.append(fid)
                if len(self._processed_frame_ids) > len(self._frame_id_history):
                    # Window eviction: oldest IDs are being dropped while frames may
                    # still be in-flight through downstream processors.  If this fires
                    # frequently it means MAX_FRAME_DEDUP_HISTORY is too small and audio
                    # frames could be double-buffered, producing choppy WAV output.
                    logger.warning(
                        "Dedup window full (%d slots): evicting oldest frame IDs. "
                        "Consider increasing MAX_FRAME_DEDUP_HISTORY (currently %d).",
                        len(self._frame_id_history),
                        MAX_FRAME_DEDUP_HISTORY,
                    )
                    self._processed_frame_ids = set(self._frame_id_history)

            handler = self._frame_handlers.get(type(data.frame))
            if handler:
                await handler(data)
        except Exception as e:
            logger.warning("Error in on_push_frame: %s", e, exc_info=True)

    # ---------------------------------------------------------------------- #
    # Pipeline lifecycle frame handlers                                       #
    # ---------------------------------------------------------------------- #

    async def _handle_start_frame(self, data: Any) -> None:
        """
        ``StartFrame``: ensure the trace exists and record pipeline config.

        Uses :meth:`attach_to_task` (or constructor args) to wire
        ``TurnTrackingObserver`` / ``UserBotLatencyObserver`` — ``FramePushed.source``
        is a processor, not ``PipelineTask``, so observers cannot be discovered here.
        """
        await self._ensure_audio_buffer_recording()
        if self._trace is None:
            await self.on_pipeline_started()

        if self._trace:
            frame = data.frame
            attrs: dict[str, Any] = {}
            # B2: read the UNION of both pipecat lines' StartFrame fields. The
            # original bug was reading ONLY the pre-1.x names (which don't exist on
            # 1.x → pipeline.* always empty); the fix must ADD the 1.x fields, not
            # drop the legacy ones — ``getattr(..., None)`` is safe for names absent
            # on whichever version is running, so 0.0.x still captures
            # ``allow_interruptions`` and 1.x captures the new audio/enable fields.
            for attr in (
                # 0.0.x fields (absent on 1.x → skipped)
                "allow_interruptions",
                "sample_rate",
                "audio_sample_rate",
                # 1.x fields (absent on 0.0.x → skipped)
                "audio_in_sample_rate",
                "audio_out_sample_rate",
                "enable_metrics",
                "enable_usage_metrics",
                "enable_tracing",
                "report_only_initial_ttfb",
            ):
                val = getattr(frame, attr, None)
                if val is not None:
                    attrs[f"pipeline.{attr}"] = val
            if attrs:
                self._trace.set_attributes(attrs)

    async def _handle_end_frame(self, data: Any) -> None:
        """``EndFrame``: cleanly finish the conversation trace."""
        await self._finish_conversation()

    async def _handle_cancel_frame(self, data: Any) -> None:
        """``CancelFrame``: finish the conversation trace with cancelled status."""
        await self._finish_conversation(cancelled=True)

    # ---------------------------------------------------------------------- #
    # Span creation helper                                                    #
    # ---------------------------------------------------------------------- #

    def _create_child_span(
        self,
        name: str,
        parent_span: Any = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Create a span under the conversation trace.

        Uses ``trace.create_span()`` directly to avoid context-stack side effects.
        ``parent_span_id`` defaults to the current turn span.
        """
        if self._trace is None:
            return None
        try:
            parent_id = parent_span.span_id if parent_span is not None else None
            return self._trace.create_span(
                name=name,
                parent_span_id=parent_id,
                attributes=attributes or {},
            )
        except Exception as e:
            logger.warning("Failed to create span '%s': %s", name, e, exc_info=True)
            return None

    # ---------------------------------------------------------------------- #
    # LLMUsageMetricsFrame handler                                           #
    # ---------------------------------------------------------------------- #

    async def _handle_llm_usage_metrics(self, data: Any) -> None:
        """
        ``LLMUsageMetricsFrame``: standalone token-usage frame (newer Pipecat).

        Extracts token counts from the frame's ``tokens`` attribute (an
        ``LLMTokenUsage`` object) or directly from the frame, then writes the
        same set of attributes as ``_MetricsHandlerMixin._handle_metrics`` writes
        for ``LLMUsageMetricsData`` inside a ``MetricsFrame``.

        Idempotent: if both ``MetricsFrame`` and ``LLMUsageMetricsFrame`` are
        emitted for the same LLM call, the last write wins (same span target).
        """
        from noveum_trace.integrations.pipecat.pipecat_utils import (
            calculate_llm_cost,
            reasoning_is_extra_output,
        )

        llm_target = self._active_llm_span or self._last_llm_span
        if llm_target is None:
            return

        frame = data.frame
        # Shape 1 (preferred): frame.tokens is an LLMTokenUsage object.
        # Shape 2 (fallback): token counts are direct attributes on the frame.
        tokens_obj = getattr(frame, "tokens", None) or frame
        prompt = int(getattr(tokens_obj, "prompt_tokens", 0) or 0)
        completion = int(getattr(tokens_obj, "completion_tokens", 0) or 0)
        total = int(
            getattr(tokens_obj, "total_tokens", prompt + completion)
            or (prompt + completion)
        )

        if not (prompt or completion or total):
            return

        llm_target.attributes["llm.input_tokens"] = prompt
        llm_target.attributes["llm.output_tokens"] = completion
        llm_target.attributes["llm.total_tokens"] = total

        # Read from the source-of-truth field names first, falling back to the
        # span-attribute names, so both Pipecat naming conventions are covered.
        for dst, srcs in (
            ("cache_read_tokens", ("cache_read_input_tokens", "cache_read_tokens")),
            (
                "cache_creation_tokens",
                ("cache_creation_input_tokens", "cache_creation_tokens"),
            ),
            ("reasoning_tokens", ("reasoning_tokens",)),
        ):
            val = next(
                (
                    v
                    for v in (getattr(tokens_obj, s, None) for s in srcs)
                    if v is not None
                ),
                None,
            )
            if val is not None:
                llm_target.attributes[f"llm.{dst}"] = int(val)

        model = getattr(frame, "model", None) or llm_target.attributes.get(
            "llm.model", ""
        )
        if model:
            llm_target.attributes["llm.model"] = model
            # D9: price reasoning/thinking tokens at the output rate, adding them to
            # completion only for providers that report the two disjoint (Gemini) —
            # OpenAI-compatible providers already count reasoning inside
            # completion_tokens. Mirrors _MetricsHandlerMixin._handle_metrics.
            reasoning = int(getattr(tokens_obj, "reasoning_tokens", 0) or 0)
            billable_output = completion
            if reasoning_is_extra_output(
                processor=str(getattr(frame, "processor", "") or ""),
                model=str(model),
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
                reasoning_tokens=reasoning,
            ):
                billable_output += reasoning
            cost = calculate_llm_cost(model, prompt, billable_output)
            if cost:
                llm_target.attributes["llm.cost.input"] = cost["input"]
                llm_target.attributes["llm.cost.output"] = cost["output"]
                llm_target.attributes["llm.cost.total"] = cost["total"]
                llm_target.attributes["llm.cost.currency"] = cost["currency"]
                if reasoning:
                    rcost = calculate_llm_cost(model, 0, reasoning)
                    if rcost:
                        llm_target.attributes["llm.cost.reasoning"] = rcost["output"]
                self._metrics_accumulator["total_cost"] = (
                    self._metrics_accumulator["total_cost"] + cost["total"]
                )

        self._metrics_accumulator["total_input_tokens"] += prompt
        self._metrics_accumulator["total_output_tokens"] += completion

        logger.debug(
            "LLMUsageMetricsFrame: prompt=%d completion=%d model=%s",
            prompt,
            completion,
            model or "<unknown>",
        )

    # ---------------------------------------------------------------------- #
    # Session metadata helpers                                               #
    # ---------------------------------------------------------------------- #

    def _store_transport(
        self, transport: Any, runner_args: Optional[Any] = None
    ) -> None:
        """
        Save the transport reference and extract static session metadata.

        Called from ``NoveumPipecatTracer.register_task_handlers`` before
        ``attach_to_task``.  The trace may not exist yet at this point, so
        metadata is buffered in ``_session_metadata`` and flushed to the trace
        by ``_flush_session_metadata`` at first client/bot connection.

        Metadata extracted (all via ``getattr`` — degrades gracefully on any
        transport shape):

        From *transport*:
          - ``session.transport_type`` — transport class name
          - ``session.room_url`` — Daily / WebRTC room URL

        From *runner_args* (``pipecat.runner.types.RunnerArguments``):
          - ``session.room_url`` — authoritative room URL (fills gaps)
          - ``session.idle_timeout_secs`` — ``pipeline_idle_timeout_secs``
          - ``session.bot_name`` — ``bot_name`` if present
        """
        if not self._capture_session_metadata:
            return

        self._transport = transport
        meta: dict[str, Any] = {}

        try:
            meta["session.transport_type"] = type(transport).__name__
        except Exception:
            pass

        for attr in ("room_url", "room_name"):
            val = getattr(transport, attr, None)
            if val:
                meta["session.room_url"] = val
                break

        if runner_args is not None:
            try:
                room_url = getattr(runner_args, "room_url", None) or getattr(
                    runner_args, "room", None
                )
                if room_url:
                    meta["session.room_url"] = room_url

                idle = getattr(runner_args, "pipeline_idle_timeout_secs", None)
                if idle is not None:
                    meta["session.idle_timeout_secs"] = idle

                bot_name = getattr(runner_args, "bot_name", None)
                if bot_name:
                    meta["session.bot_name"] = bot_name
            except Exception:
                pass

        if meta:
            self._session_metadata.update(meta)
            logger.debug(
                "NoveumTraceObserver: buffered session metadata: %s",
                list(meta.keys()),
            )

    async def _flush_session_metadata(self) -> None:
        """
        Write buffered session metadata onto the root conversation trace.

        Called from ``_handle_client_connected`` and ``_handle_bot_connected``
        (whichever fires first) once ``self._trace`` is guaranteed to exist.
        Clears the buffer after writing so subsequent calls are no-ops.
        """
        if not self._capture_session_metadata:
            return
        if not self._session_metadata:
            return
        if self._trace is None:
            return
        try:
            self._trace.set_attributes(self._session_metadata)
            logger.debug(
                "NoveumTraceObserver: flushed session metadata to trace: %s",
                list(self._session_metadata.keys()),
            )
        except Exception as exc:
            logger.debug(
                "NoveumTraceObserver: failed to flush session metadata: %s", exc
            )
        finally:
            self._session_metadata.clear()

    # ---------------------------------------------------------------------- #
    # Conversation finish                                                     #
    # ---------------------------------------------------------------------- #

    async def _finish_conversation(self, cancelled: bool = False) -> None:
        """
        End all active spans, finish the trace, and flush the client.

        Safe to call multiple times and from concurrent callers: the
        ``_finish_lock`` serializes teardown and the inner ``_trace``
        ``None``-check makes the second caller a no-op. See ``_finish_lock``.
        """
        # Lazily bind the lock to the running loop (see __init__ for why).
        # The check-and-set is synchronous, so concurrent callers can't both
        # create one — whichever reaches here first wins atomically.
        if self._finish_lock is None:
            self._finish_lock = asyncio.Lock()
        async with self._finish_lock:
            await self._finish_conversation_impl(cancelled=cancelled)

    async def _finish_conversation_impl(self, cancelled: bool = False) -> None:
        """
        End all active spans, finish the trace, and flush the client.

        Must only be called while holding ``_finish_lock`` (via
        ``_finish_conversation``). The ``_trace`` ``None``-check below provides
        idempotency once the lock is held.
        """
        if self._trace is None:
            logger.debug(
                "_finish_conversation called but _trace is already None — idempotent no-op"
            )
            return

        logger.debug(
            "_finish_conversation starting (cancelled=%s, trace_id=%s)",
            cancelled,
            getattr(self._trace, "trace_id", "<unknown>"),
        )

        await self._cancel_turn_end_timer()

        # Drain any still-open custom spans before the trace closes.
        proc = self._custom_span_processor
        if proc is not None:
            import datetime as _dt

            _now = _dt.datetime.now(_dt.timezone.utc)
            for nov_span in list(proc._map.values()):
                try:
                    if self._trace is not None:
                        self._trace.finish_span(nov_span.span_id, end_time=_now)
                except Exception:
                    pass
            proc._map.clear()

        # Discard any partial thought accumulated so far
        self._llm_thought_buffer.clear()
        self._llm_thoughts_list.clear()
        self._llm_thought_signatures_list.clear()
        self._pending_thought_signatures.clear()

        # Close any in-flight STT span
        if self._active_stt_span and not self._active_stt_span.is_finished():
            self._active_stt_span.attributes["pipecat_span_status"] = (
                "cancelled" if cancelled else "ok"
            )
            self._active_stt_span.finish()
        self._active_stt_span = None
        self._stt_source_processor = None
        self._vad_speech_start_time = None
        self._stt_interim_results.clear()
        self._stt_first_text_latency_recorded = False
        self._stt_audio_buffer.clear()
        self._stt_raw_audio_buffer.clear()

        for span in filter(None, [self._active_llm_span, self._active_tts_span]):
            if not span.is_finished():
                # D8: on an abnormal close there is no TTSStoppedFrame, so flush any
                # buffered TTS text here before finishing (otherwise it is lost).
                if span is self._active_tts_span:
                    self._flush_tts_text(span)
                if span.attributes.get("pipecat_span_status") != "error":
                    span.attributes["pipecat_span_status"] = (
                        "cancelled" if cancelled else "ok"
                    )
                span.finish()
        self._active_llm_span = None
        self._active_tts_span = None
        self._last_llm_span = None
        self._last_tts_span = None

        self._pending_function_calls.clear()
        self._function_call_owner.clear()
        self._resolved_function_call_ids.clear()

        if self._pending_turn_eou_metrics:
            logger.debug(
                "Discarding pending turn EOU metrics at conversation end: %s",
                list(self._pending_turn_eou_metrics.keys()),
            )
        self._pending_turn_eou_metrics.clear()

        if self._current_turn_span is not None:
            await self._end_current_turn(was_interrupted=cancelled)

        await self._await_audio_buffer_pending_handlers()
        await self._upload_full_conversation_audio()

        # Annotate trace with conversation summary
        summary: dict[str, Any] = {}
        if self._metrics_accumulator.get("total_input_tokens"):
            summary["conversation.total_input_tokens"] = self._metrics_accumulator[
                "total_input_tokens"
            ]
        if self._metrics_accumulator.get("total_output_tokens"):
            summary["conversation.total_output_tokens"] = self._metrics_accumulator[
                "total_output_tokens"
            ]
        if self._metrics_accumulator.get("total_cost"):
            summary["conversation.total_cost"] = self._metrics_accumulator["total_cost"]
        if self._metrics_accumulator.get("turn_count"):
            summary["conversation.turn_count"] = self._metrics_accumulator["turn_count"]
        if self._transcription_buffer:
            summary["conversation.last_user_input"] = " ".join(
                self._transcription_buffer[-5:]
            )

        if summary:
            self._trace.set_attributes(summary)

        if self._trace.attributes.get("pipecat_span_status") != "error":
            self._trace.attributes["pipecat_span_status"] = (
                "cancelled" if cancelled else "ok"
            )

        trace = self._trace
        self._trace = None

        try:
            client = self._get_client()
            if client:
                client.finish_trace(trace)
                try:
                    client.flush()
                except Exception:
                    pass
            else:
                trace.finish()
        except Exception as e:
            logger.warning("Failed to finish pipecat trace: %s", e, exc_info=True)

        # Tear down / detach the custom-span processor so it does not leak on the
        # customer's global TracerProvider (OTEL has no public remove API): if we
        # own the provider, shut it down; otherwise best-effort unregister and
        # detach so a lingering processor holds no observer graph and no-ops.
        proc = self._custom_span_processor
        if proc is not None:
            try:
                from noveum_trace.integrations.pipecat.custom_spans import (
                    unregister_custom_span_processor,
                )

                unregister_custom_span_processor(proc)
            except Exception:
                logger.debug("custom-span processor teardown failed", exc_info=True)
            self._custom_span_processor = None

        logger.debug(
            "_finish_conversation completed — trace flushed (cancelled=%s)", cancelled
        )

        # Reset conversation-scoped caches for observer reuse
        self._metrics_accumulator = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "turn_count": 0,
        }
        self._transcription_buffer = []
        self._llm_text_buffer.clear()
        self._native_error_messages.clear()
        self._current_turn_number = 0
        self._processed_frame_ids.clear()
        self._frame_id_history.clear()
        self._pending_llm_context.clear()
        # Reset stored ABP so a new PipelineTask can attach the correct processor.
        self._audio_buffer_processor = None
        self._abp_is_recording = False
        self._conversation_audio_chunks = []
        self._conversation_audio_sample_rate = None
        self._conversation_audio_num_channels = None

        # Reset transport/metadata state for observer reuse.
        self._transport = None
        self._session_metadata.clear()

    async def _await_audio_buffer_pending_handlers(self) -> None:
        """
        Pipecat's ``FrameProcessor._call_event_handler`` schedules async handlers
        with ``asyncio.create_task`` (does not await). ``AudioBufferProcessor`` fires
        ``on_audio_data`` that way, so ``EndFrame`` can reach this observer before
        ``_on_conversation_audio`` has run. Drain pending handler tasks on the buffer
        processor before building the full-conversation WAV.
        """
        abp = self._audio_buffer_processor
        if abp is None:
            return
        tasks_set = getattr(abp, "_event_tasks", None)
        if tasks_set is None:
            logger.debug(
                "AudioBufferProcessor._event_tasks not found — cannot drain pending handlers. "
                "Conversation audio may be truncated if Pipecat changed its internal API."
            )
            return
        if not tasks_set:
            return
        pending = [t for (_name, t) in tasks_set]
        if pending:
            await asyncio.wait(pending)

    async def _upload_full_conversation_audio(self) -> None:
        """
        Build a WAV from the PCM delivered by ``AudioBufferProcessor`` and upload
        it as a dedicated ``pipecat.full_conversation`` span.

        Mirrors ``livekit_session._upload_full_conversation_audio`` so that the
        Noveum dashboard can treat both integrations identically.
        """
        if not self._record_audio:
            return
        if self._trace is None:
            return

        # Always create the span so the dashboard has a stable place to look,
        # even if we couldn't capture audio (e.g. ABP missing or recording stopped early).
        missing_reason: Optional[str] = None
        if not self._conversation_audio_chunks:
            missing_reason = (
                "no_conversation_audio_chunks_captured"
                if self._audio_buffer_processor is not None
                else "audio_buffer_processor_not_attached"
            )

        if missing_reason is not None:
            missing_attributes: dict[str, Any] = {
                "full_conversation.audio_source": "AudioBufferProcessor",
                "full_conversation.missing_reason": missing_reason,
                "pipecat_span_status_message": missing_reason,
            }
            try:
                span = self._trace.create_span(
                    name="pipecat.full_conversation",
                    attributes=missing_attributes,
                )
                if span is None:
                    logger.warning("Could not create pipecat.full_conversation span")
                    return
                span.attributes["pipecat_span_status"] = "error"
                self._trace.finish_span(span.span_id)
            except Exception as e:
                logger.warning(
                    "Failed to create pipecat.full_conversation missing span: %s",
                    e,
                    exc_info=True,
                )
            return

        pcm = b"".join(self._conversation_audio_chunks)
        self._conversation_audio_chunks = []  # release references early

        sr = self._conversation_audio_sample_rate or 16000
        ch = self._conversation_audio_num_channels or 2

        try:
            # Encode raw PCM → WAV in memory
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(ch)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(sr)
                wf.writeframes(pcm)
            wav_bytes = buf.getvalue()

            audio_uuid = str(uuid.uuid4())
            duration_ms = int(len(pcm) / (sr * ch * 2) * 1000)
            channels_label = "stereo" if ch == 2 else "mono"
            description = (
                "Full conversation - stereo recording (left=user, right=bot)"
                if ch == 2
                else "Full conversation - mono recording"
            )
            attributes: dict[str, Any] = {
                "full_conversation.audio_uuid": audio_uuid,
                "full_conversation.audio_format": "wav",
                "full_conversation.audio_channels": channels_label,
                "full_conversation.audio_source": "AudioBufferProcessor",
                "full_conversation.audio_description": description,
                "full_conversation.duration_ms": duration_ms,
                "full_conversation.sample_rate": sr,
            }
            if ch == 2:
                attributes["full_conversation.audio_channel_left"] = "user"
                attributes["full_conversation.audio_channel_right"] = "bot"

            span = self._trace.create_span(
                name="pipecat.full_conversation",
                attributes=attributes,
            )
            if span is None:
                logger.warning("Could not create pipecat.full_conversation span")
                return

            audio_metadata = {
                "duration_ms": duration_ms,
                "format": "wav",
                "type": "conversation",
                "num_channels": ch,
                "sample_rate": sr,
            }
            upload_ok = False
            try:
                if self._audio_sink is not None:
                    upload_ok = bool(
                        await self._audio_sink(
                            wav_bytes=wav_bytes,
                            audio_uuid=audio_uuid,
                            kind="conversation",
                            trace_id=span.trace_id,
                            span_id=span.span_id,
                            metadata=audio_metadata,
                        )
                    )
                    logger.debug(
                        "Sank full-conversation audio: %s (ok=%s)",
                        audio_uuid,
                        upload_ok,
                    )
                elif client := self._get_client():
                    client.export_audio(
                        audio_data=wav_bytes,
                        trace_id=span.trace_id,
                        span_id=span.span_id,
                        audio_uuid=audio_uuid,
                        metadata=audio_metadata,
                    )
                    upload_ok = True
                    logger.debug(
                        "Queued full-conversation audio upload: %s", audio_uuid
                    )
                else:
                    logger.warning(
                        "No client available — skipping conversation audio upload"
                    )
            except Exception as e:
                span.attributes["pipecat_span_status"] = "error"
                logger.warning(
                    "Failed to upload full-conversation audio %s: %s",
                    audio_uuid,
                    e,
                    exc_info=True,
                )
                raise
            finally:
                span.attributes["pipecat_span_status"] = "ok" if upload_ok else "error"
                self._trace.finish_span(span.span_id)

        except Exception as e:
            logger.warning(
                "Failed to build/upload full-conversation audio: %s", e, exc_info=True
            )
