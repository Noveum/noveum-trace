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
    _record_audio: bool
    _capture_text: bool
    _capture_function_calls: bool
    _turn_end_timeout_secs: float

    _trace: Any
    _current_turn_span: Any
    _current_turn_number: int
    _turn_start_time: Optional[float]
    _pending_turn_eou_metrics: dict[str, Any]

    _active_llm_span: Any
    _active_tts_span: Any
    _pending_function_calls: dict[str, dict[str, Any]]
    _function_call_results: list[dict[str, Any]]
    _pre_span_function_call_ids: set[str]

    _last_llm_span: Any
    _last_tts_span: Any

    _llm_text_buffer: list[str]
    _tts_text_buffer: list[str]
    _transcription_buffer: list[str]

    _pending_llm_context: dict[str, Any]

    _llm_thought_buffer: list[str]
    _llm_thoughts_list: list[str]
    _llm_thought_signatures_list: list[str]

    _stt_audio_buffer: list[Any]
    _tts_audio_buffer: list[Any]
    _tts_source_processor: Any

    _audio_buffer_processor: Any
    _conversation_audio_data: Optional[bytes]
    _conversation_audio_sample_rate: Optional[int]
    _conversation_audio_num_channels: Optional[int]

    _vad_present: bool
    _active_stt_span: Any
    _stt_source_processor: Any
    _vad_speech_start_time: Optional[float]
    _stt_interim_results: list[dict[str, Any]]
    _stt_first_text_latency_recorded: bool

    _metrics_accumulator: dict[str, Any]

    _turn_tracker: Any
    _latency_tracker: Any
    _using_external_turn_tracking: bool

    _is_bot_speaking: bool
    _bot_has_spoken_in_turn: bool
    _user_stopped_speaking_time: Optional[float]
    _turn_end_task: Optional[asyncio.Task[None]]

    _processed_frame_ids: set[int]
    _frame_id_history: deque[int]

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

    async def _start_new_turn(self, turn_number: Optional[int] = None) -> None: ...

    async def _finish_conversation(self, cancelled: bool = False) -> None: ...


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

        async def _start_new_turn(self, turn_number: Optional[int] = None) -> None: ...

        async def _finish_conversation(self, cancelled: bool = False) -> None: ...
