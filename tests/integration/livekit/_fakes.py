"""
Shared fakes and helpers for LiveKit behavioral/regression tests.

Design notes
------------
* The *wrapper* objects under test (``LiveKitSTTWrapper`` etc.) must be REAL
  subclasses of the LiveKit base classes, so we drive them through their real
  ``super().__init__`` and assert ``isinstance(wrapper, BaseSTT)``.
* The *base providers* we hand to the wrappers are small concrete subclasses of
  the same LiveKit base classes (``FakeBaseSTT/TTS/LLM``). They are real
  ``EventEmitter``s, so ``base.emit(...)`` actually fires the wrapper's
  forwarding handlers — that is what lets us test event forwarding for real.
* Streams are driven by :class:`RecordingStream`, a single async-iterator that
  records ``push_frame``/``push_text`` and ``aclose``/``flush`` calls.

These helpers intentionally use REAL LiveKit event/data types (``SpeechEvent``,
``SynthesizedAudio``, ``ChatChunk`` …) so the tests catch LiveKit API drift,
while the assertions target the attribute *contract our integration emits*.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

pytest.importorskip("livekit.agents")

from livekit import rtc  # noqa: E402
from livekit.agents.llm import LLM as BaseLLM  # noqa: E402
from livekit.agents.llm import (
    ChatChunk,
    ChoiceDelta,
    CompletionUsage,
    FunctionToolCall,
)
from livekit.agents.stt import STT as BaseSTT  # noqa: E402
from livekit.agents.stt import (
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.tts import TTS as BaseTTS  # noqa: E402
from livekit.agents.tts import (
    SynthesizedAudio,
    TTSCapabilities,
)


# --------------------------------------------------------------------------- #
# Audio frame / event builders (real LiveKit objects)
# --------------------------------------------------------------------------- #
def make_frame(duration_s: float = 0.1, sample_rate: int = 16000) -> rtc.AudioFrame:
    """Build a real ``rtc.AudioFrame`` of the requested duration (mono, 16-bit)."""
    samples = int(sample_rate * duration_s)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


def make_speech_event(
    *,
    event_type: SpeechEventType = SpeechEventType.FINAL_TRANSCRIPT,
    text: str = "Hello world",
    confidence: float = 0.95,
    language: str = "en",
    request_id: str = "req_123",
) -> SpeechEvent:
    """Build a real ``SpeechEvent`` with one alternative."""
    return SpeechEvent(
        type=event_type,
        request_id=request_id,
        alternatives=[
            SpeechData(
                language=language,
                text=text,
                start_time=0.0,
                end_time=1.5,
                confidence=confidence,
            )
        ],
    )


def make_synth_audio(
    *,
    is_final: bool = True,
    request_id: str = "tts_req_1",
    segment_id: str = "seg_1",
    delta_text: str = "",
) -> SynthesizedAudio:
    """Build a real ``SynthesizedAudio`` chunk."""
    return SynthesizedAudio(
        frame=make_frame(0.1, sample_rate=24000),
        request_id=request_id,
        is_final=is_final,
        segment_id=segment_id,
        delta_text=delta_text,
    )


def make_chat_chunk(
    *,
    chunk_id: str = "chunk-1",
    content: Optional[str] = None,
    role: Optional[str] = None,
    tool_calls: Optional[list[FunctionToolCall]] = None,
    usage: Optional[CompletionUsage] = None,
) -> ChatChunk:
    """Build a real ``ChatChunk`` (delta omitted when nothing to put in it)."""
    delta = None
    if content is not None or role is not None or tool_calls:
        delta = ChoiceDelta(role=role, content=content, tool_calls=tool_calls or [])
    return ChatChunk(id=chunk_id, delta=delta, usage=usage)


# --------------------------------------------------------------------------- #
# Recording stream
# --------------------------------------------------------------------------- #
class RecordingStream:
    """A minimal async-iterator base stream that records interactions.

    Deliberately omits ``__aenter__``/``__aexit__`` so the wrapper's context
    manager falls back to ``aclose`` (a path the suite asserts).
    """

    def __init__(self, items: Optional[list[Any]] = None):
        self._items = list(items or [])
        self._i = 0
        self.pushed_frames: list[Any] = []
        self.pushed_text: list[str] = []
        self.closed = False
        self.flushed = False

    def __aiter__(self) -> RecordingStream:
        return self

    async def __anext__(self) -> Any:
        if self._i >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._i]
        self._i += 1
        return item

    def push_frame(self, frame: Any) -> None:
        self.pushed_frames.append(frame)

    def push_text(self, text: str) -> None:
        self.pushed_text.append(text)

    def flush(self) -> None:
        self.flushed = True

    async def aclose(self) -> None:
        self.closed = True


class ErrorStream(RecordingStream):
    """Async iterator that raises after yielding any seeded items."""

    def __init__(
        self, items: Optional[list[Any]] = None, error: Optional[Exception] = None
    ):
        super().__init__(items)
        self._error = error or RuntimeError("stream boom")

    async def __anext__(self) -> Any:
        if self._i < len(self._items):
            item = self._items[self._i]
            self._i += 1
            return item
        raise self._error


# --------------------------------------------------------------------------- #
# Concrete fake base providers (real EventEmitters)
# --------------------------------------------------------------------------- #
class FakeBaseSTT(BaseSTT):
    def __init__(
        self,
        *,
        recognize_event: Optional[SpeechEvent] = None,
        stream: Optional[RecordingStream] = None,
        model: str = "nova-2",
        provider: str = "deepgram",
        streaming: bool = True,
    ):
        super().__init__(
            capabilities=STTCapabilities(streaming=streaming, interim_results=True)
        )
        self._recognize_event = recognize_event
        self._stream_obj = stream if stream is not None else RecordingStream()
        self._model = model
        self._provider = provider
        self.aclose_called = False

    # model/provider/label are read-only @property on the base class; override.
    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def label(self) -> str:
        return f"{self._provider}.STT"

    async def _recognize_impl(self, buffer: Any, **kwargs: Any) -> SpeechEvent:
        return self._recognize_event

    def stream(self, **kwargs: Any) -> RecordingStream:
        return self._stream_obj

    async def aclose(self) -> None:
        self.aclose_called = True


class FakeBaseTTS(BaseTTS):
    def __init__(
        self,
        *,
        synth_stream: Optional[RecordingStream] = None,
        chunked_stream: Optional[RecordingStream] = None,
        model: str = "sonic",
        provider: str = "cartesia",
        sample_rate: int = 24000,
        num_channels: int = 1,
    ):
        super().__init__(
            capabilities=TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._synth_stream = (
            synth_stream if synth_stream is not None else RecordingStream()
        )
        self._chunked_stream = (
            chunked_stream if chunked_stream is not None else RecordingStream()
        )
        self._model = model
        self._provider = provider
        self.aclose_called = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def label(self) -> str:
        return f"{self._provider}.TTS"

    def synthesize(self, text: str, **kwargs: Any) -> RecordingStream:
        return self._chunked_stream

    def stream(self, **kwargs: Any) -> RecordingStream:
        return self._synth_stream

    async def aclose(self) -> None:
        self.aclose_called = True


class _Opts:
    """Stand-in for an OpenAI-style ``llm._opts`` sampling-parameter bag."""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeBaseLLM(BaseLLM):
    def __init__(
        self,
        *,
        chat_stream: Optional[RecordingStream] = None,
        model: str = "gpt-4o",
        provider: str = "openai",
        opts: Optional[_Opts] = None,
    ):
        super().__init__()
        self._chat_stream = (
            chat_stream if chat_stream is not None else RecordingStream()
        )
        self._model = model
        self._provider = provider
        self.last_chat_kwargs: dict[str, Any] = {}
        self.aclose_called = False
        if opts is not None:
            self._opts = opts

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def label(self) -> str:
        return f"{self._provider}.LLM"

    def chat(
        self, *, chat_ctx: Any, tools: Any = None, **kwargs: Any
    ) -> RecordingStream:
        self.last_chat_kwargs = {"chat_ctx": chat_ctx, "tools": tools, **kwargs}
        return self._chat_stream

    async def aclose(self) -> None:
        self.aclose_called = True


# --------------------------------------------------------------------------- #
# Span-capture helpers
# --------------------------------------------------------------------------- #
def spans_named(trace: Any, name: str) -> list[Any]:
    return [s for s in trace.spans if s.name == name]


def one_span(trace: Any, name: str) -> Any:
    matches = spans_named(trace, name)
    assert len(matches) == 1, (
        f"expected exactly one span named {name!r}, "
        f"got {[s.name for s in trace.spans]}"
    )
    return matches[0]
