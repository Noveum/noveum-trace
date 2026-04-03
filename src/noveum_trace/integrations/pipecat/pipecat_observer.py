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
from collections.abc import Iterator
from typing import Any, Optional

from noveum_trace.integrations.pipecat._handlers_llm import _LLMHandlersMixin
from noveum_trace.integrations.pipecat._handlers_metrics import _MetricsHandlerMixin
from noveum_trace.integrations.pipecat._handlers_stt import _STTHandlersMixin
from noveum_trace.integrations.pipecat._handlers_tts import _TTSHandlersMixin
from noveum_trace.integrations.pipecat._turn_manager import _TurnManagerMixin
from noveum_trace.integrations.pipecat.pipecat_constants import (
    DEFAULT_TURN_END_TIMEOUT_SECS,
    MAX_FRAME_DEDUP_HISTORY,
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
        capture_text: bool = True,
        capture_function_calls: bool = True,
        turn_end_timeout_secs: float = DEFAULT_TURN_END_TIMEOUT_SECS,
        turn_tracking_observer: Any = None,
        latency_observer: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the observer.

        Args:
            trace_name_prefix: Prefix for the conversation trace name.
            record_audio: When ``True``, buffers STT/TTS audio and uploads a WAV
                per span to ``/v1/audio``.
            capture_text: When ``True``, accumulates LLM / TTS text buffers.
            capture_function_calls: When ``True``, creates ``function_call`` child
                spans.
            turn_end_timeout_secs: Seconds to wait after the bot stops speaking
                before closing a standalone-mode turn.
            turn_tracking_observer: Optional Pipecat ``TurnTrackingObserver`` to
                subscribe to for turn boundaries (or use :meth:`attach_to_task`).
            latency_observer: Optional ``UserBotLatencyObserver`` for
                ``on_latency_measured`` (or use :meth:`attach_to_task`).
            **kwargs: Forwarded to Pipecat ``BaseObserver`` / ``BaseObject``
                (e.g. ``name=``).
        """
        super().__init__(**kwargs)

        self._trace_name_prefix = trace_name_prefix
        self._record_audio = record_audio
        self._capture_text = capture_text
        self._capture_function_calls = capture_function_calls
        self._turn_end_timeout_secs = turn_end_timeout_secs

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
        # tool_call_id → call dict
        self._pending_function_calls: dict[str, dict[str, Any]] = {}
        # completed/cancelled results
        self._function_call_results: list[dict[str, Any]] = []
        # tool_call_ids written directly to _last_llm_span (between span 1 close and
        # span 2 open); filtered out of span 2's llm.function_calls to avoid double-counting.
        self._pre_span_function_call_ids: set[str] = set()

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
        self._tts_text_buffer: list[str] = []
        self._transcription_buffer: list[str] = []

        # LLM context stash (filled by LLMContextFrame, flushed on LLMFullResponseStartFrame)
        self._pending_llm_context: dict[str, Any] = {}

        # LLM thought accumulation (flattened onto the LLM span as attribute lists)
        self._llm_thought_buffer: list[str] = []
        self._llm_thoughts_list: list[str] = []
        self._llm_thought_signatures_list: list[str] = []

        # Audio buffers (populated only when record_audio=True)
        self._stt_audio_buffer: list[Any] = []
        self._tts_audio_buffer: list[Any] = []
        # Source processor that opened the current TTS span (set on TTSStartedFrame,
        # cleared on TTSStoppedFrame / interruption).  Audio frames are only buffered
        # when they come from this exact processor so that downstream resamplers /
        # aggregators that re-emit TTSAudioRawFrame with new IDs are ignored.
        self._tts_source_processor: Any = None

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
        """Get the globally initialized Noveum client."""
        try:
            from noveum_trace import get_client

            return get_client()
        except Exception:
            return None

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

        Safe to call multiple times; repeated calls with the same observers are
        no-ops.
        """
        if task is None:
            return
        tto = getattr(task, "turn_tracking_observer", None)
        if tto is not None:
            self.attach_turn_tracking_observer(tto)
        lto = getattr(task, "_user_bot_latency_observer", None)
        if lto is not None:
            self.attach_latency_observer(lto)

        # ---------------------------------------------------------------------- #
        # Safety net: on_pipeline_finished fires inline in the main pipeline      #
        # coroutine, before _cancel_tasks() kills the TaskObserver proxy tasks.   #
        # _finish_conversation is idempotent, so whichever path fires first wins. #
        # ---------------------------------------------------------------------- #
        if hasattr(task, "event_handler"):
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

                logger.debug(
                    "Registered on_pipeline_finished safety-net handler on task"
                )
            except Exception as e:
                logger.warning(
                    "Could not register on_pipeline_finished safety-net handler: %s", e
                )
        else:
            logger.debug(
                "PipelineTask does not expose event_handler(); "
                "on_pipeline_finished safety net not registered — "
                "trace cleanup relies on on_push_frame CancelFrame/EndFrame path only"
            )

        # Auto-detect AudioBufferProcessor for full-conversation recording
        if self._record_audio:
            await self._attach_audio_buffer_from_pipeline(task)

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

    async def _attach_audio_buffer_from_pipeline(self, task: Any) -> None:
        """
        Walk the task's pipeline processors looking for an ``AudioBufferProcessor``.

        If found, register ``_on_conversation_audio`` on its ``on_audio_data``
        event so the full stereo conversation WAV is captured on session end.

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

        # If the same ABP is still active for this observer, don't re-register the handler.
        if found_proc is self._audio_buffer_processor:
            await self._ensure_audio_buffer_recording()
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
            # ABP drops InputAudio/OutputAudio until start_recording(); observer
            # on_push_frame runs after process_frame, so start here before run().
            try:
                await self._audio_buffer_processor.start_recording()
                self._abp_is_recording = True
            except Exception as e:
                logger.warning(
                    "Failed to start AudioBufferProcessor recording: %s",
                    e,
                    exc_info=True,
                )
            logger.info(
                "Noveum trace: full-conversation audio attached (nested pipeline OK)"
            )
            logger.debug(
                "Attached to AudioBufferProcessor; start_recording() completed"
            )
        except Exception as e:
            logger.warning(
                "Failed to attach to AudioBufferProcessor: %s", e, exc_info=True
            )
            # If the new ABP couldn't be wired, prefer keeping the previous one.
            self._audio_buffer_processor = prev_proc
            return
        return

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

            trace_name = f"{self._trace_name_prefix}.{SPAN_CONVERSATION.split('.')[-1]}"
            self._trace = client.start_trace(name=trace_name, set_as_current=True)
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
            for attr in ("allow_interruptions", "sample_rate", "audio_sample_rate"):
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
    # Conversation finish                                                     #
    # ---------------------------------------------------------------------- #

    async def _finish_conversation(self, cancelled: bool = False) -> None:
        """
        End all active spans, finish the trace, and flush the client.

        Safe to call multiple times (guarded by ``_trace`` ``None``-check).
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

        # Discard any partial thought accumulated so far
        self._llm_thought_buffer.clear()
        self._llm_thoughts_list.clear()
        self._llm_thought_signatures_list.clear()

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

        for span in filter(None, [self._active_llm_span, self._active_tts_span]):
            if not span.is_finished():
                span.attributes["pipecat_span_status"] = (
                    "cancelled" if cancelled else "ok"
                )
                span.finish()
        self._active_llm_span = None
        self._active_tts_span = None
        self._last_llm_span = None
        self._last_tts_span = None

        self._pending_function_calls.clear()
        self._function_call_results.clear()

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

            client = self._get_client()
            upload_ok = False
            try:
                if client:
                    client.export_audio(
                        audio_data=wav_bytes,
                        trace_id=span.trace_id,
                        span_id=span.span_id,
                        audio_uuid=audio_uuid,
                        metadata={
                            "duration_ms": duration_ms,
                            "format": "wav",
                            "type": "conversation",
                            "num_channels": ch,
                            "sample_rate": sr,
                        },
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
