"""
Typing helpers for Pipecat observer mixins.

``_PipecatObserverState`` holds instance attribute annotations (initialised in
``NoveumTraceObserver.__init__``). ``_PipecatObserverMethods`` is a ``Protocol``
documenting helper methods on ``NoveumTraceObserver`` (for typing call sites).
Mixins inherit only ``_PipecatObserverState`` via ``_PipecatObserverMixinBase`` so
cooperative ``super().__init__`` reaches Pipecat ``BaseObject`` (see mixin base).
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any, Optional, Protocol


class _PipecatObserverState:
    """Annotation-only: fields mirror ``NoveumTraceObserver.__init__``."""

    _trace_name_prefix: str
    # Per-observer client / deferred-export configuration
    _injected_client: Any
    _deferred: bool
    _audio_sink: Any
    _register_finish_safety_net: bool
    _record_audio: bool
    _record_raw_input_audio: bool
    _capture_text: bool
    _capture_function_calls: bool
    _turn_end_timeout_secs: float

    _trace: Any
    _current_turn_span: Any
    _current_turn_number: int
    _turn_start_time: Optional[float]
    _pending_turn_eou_metrics: dict[str, Any]

    _llm_operations: Any
    _processor_registry: Any
    _metric_fingerprints: dict[str, set[str]]

    _active_llm_span: Any
    _active_tts_span: Any
    _pending_function_calls: dict[str, dict[str, Any]]
    _function_call_owner: dict[str, Any]
    _resolved_function_call_ids: set[str]

    _last_llm_span: Any
    _last_tts_span: Any
    _last_tts_source_processor: Any

    _llm_text_buffer: list[str]
    _tts_request_text_buffer: list[tuple[str, bool]]
    _tts_text_buffer: list[tuple[str, bool]]
    _tts_text_interim_buffer: list[tuple[str, bool]]
    _transcription_buffer: list[str]

    _pending_llm_context: dict[str, Any]
    _global_llm_context_generation: int
    _global_llm_context_consumed: dict[int, int]

    _llm_thought_buffer: list[str]
    _llm_thoughts_list: list[str]
    _llm_thought_signatures_list: list[str]

    _stt_audio_buffer: list[Any]
    _stt_raw_audio_buffer: list[Any]
    _pipeline_has_stt: bool
    _warned_no_stt_for_raw: bool
    _tts_audio_buffer: list[Any]
    _tts_source_processor: Any
    _tts_context_id: Optional[str]
    _tts_request_frame_id: Optional[int]
    _tts_start_frame_id: Optional[int]
    _stt_metric_processor: Any
    _stt_start_frame_id: Optional[int]
    _last_stt_span: Any
    _last_stt_metric_processor: Any

    _audio_buffer_processor: Any
    _abp_is_recording: bool
    _conversation_audio_chunks: list[bytes]
    _conversation_audio_sample_rate: Optional[int]
    _conversation_audio_num_channels: Optional[int]

    _vad_present: bool
    _active_stt_span: Any
    _stt_source_processor: Any
    _vad_speech_start_time: Optional[float]
    _stt_interim_results: list[dict[str, Any]]
    _stt_first_text_latency_recorded: bool

    _metrics_accumulator: dict[str, Any]

    _capture_errors: bool
    _capture_system_logs: bool
    _capture_session_metadata: bool

    # Distinct error messages already counted in the native trace error_count (D1)
    _native_error_messages: set[str]

    # Transport metadata buffer
    _transport: Any
    _session_metadata: dict[str, Any]

    _turn_tracker: Any
    _latency_tracker: Any
    _using_external_turn_tracking: bool

    _is_bot_speaking: bool
    _bot_has_spoken_in_turn: bool
    _user_stopped_speaking_time: Optional[float]
    _turn_end_task: Optional[asyncio.Task[None]]

    _processed_frame_ids: set[int]
    _frame_id_history: deque[int]
    _processed_llm_input_routes: set[tuple[int, int]]
    _llm_input_route_history: deque[tuple[int, int]]
    _llm_input_frame_types: set[type]
    _processed_tts_input_routes: set[tuple[int, int]]
    _tts_input_route_history: deque[tuple[int, int]]

    _frame_handlers: dict[type, Any]


class _PipecatObserverMethods(Protocol):
    """Methods implemented on ``NoveumTraceObserver`` that mixins call."""

    def _create_child_span(
        self,
        name: str,
        parent_span: Any = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Any: ...

    def _get_client(self) -> Any: ...

    def _finish_managed_span(self, span: Any) -> None: ...

    async def _sink_segment_audio(
        self,
        frames: list[Any],
        audio_uuid: str,
        kind: str,
        trace_id: str,
        span_id: str,
    ) -> bool: ...

    async def _start_new_turn(self, turn_number: Optional[int] = None) -> None: ...

    async def _finish_conversation(self, cancelled: bool = False) -> None: ...

    def _resolve_llm_operation(
        self, data: Any, *, include_metrics_pending: bool = False
    ) -> Any: ...

    def _finalize_llm_operation(
        self,
        operation: Any,
        *,
        complete: bool,
        termination_reason: str,
        terminal_status: str,
    ) -> None: ...

    async def _finalize_tts_operation(
        self,
        *,
        complete: bool,
        termination_reason: str,
        terminal_status: str,
    ) -> Any: ...

    def _bounded_append_stt_frame(self, buffer: list[Any], frame: Any) -> None: ...

    async def _flush_session_metadata(self) -> None: ...


# Do not inherit _PipecatObserverMethods (Protocol) at runtime: it inserts
# typing.Protocol / Generic before BaseObserver in NoveumTraceObserver's MRO, so
# super().__init__ never reaches Pipecat BaseObject (no _name). For mypy, mirror
# the protocol methods only under TYPE_CHECKING so they are not on the live class.
class _PipecatObserverMixinBase(_PipecatObserverState):
    """Runtime: state attrs only. Type-check: same methods as ``_PipecatObserverMethods``."""

    if TYPE_CHECKING:

        def _create_child_span(
            self,
            name: str,
            parent_span: Any = None,
            attributes: Optional[dict[str, Any]] = None,
        ) -> Any: ...

        def _get_client(self) -> Any: ...

        def _finish_managed_span(self, span: Any) -> None: ...

        async def _sink_segment_audio(
            self,
            frames: list[Any],
            audio_uuid: str,
            kind: str,
            trace_id: str,
            span_id: str,
        ) -> bool: ...

        async def _start_new_turn(self, turn_number: Optional[int] = None) -> None: ...

        async def _finish_conversation(self, cancelled: bool = False) -> None: ...

        def _resolve_llm_operation(
            self, data: Any, *, include_metrics_pending: bool = False
        ) -> Any: ...

        def _finalize_llm_operation(
            self,
            operation: Any,
            *,
            complete: bool,
            termination_reason: str,
            terminal_status: str,
        ) -> None: ...

        async def _finalize_tts_operation(
            self,
            *,
            complete: bool,
            termination_reason: str,
            terminal_status: str,
        ) -> Any: ...

        def _bounded_append_stt_frame(self, buffer: list[Any], frame: Any) -> None: ...

        async def _flush_session_metadata(self) -> None: ...
