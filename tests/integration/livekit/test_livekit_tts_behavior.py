"""
TTS wrapper behavioral / regression tests (Section C of LIVEKIT_TEST_PLAN.md).

Real client capture; asserts span names + attribute values + the
input_text -> delta_text -> "unknown" fallback chain.
"""

from __future__ import annotations

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents.tts import TTS as BaseTTS  # noqa: E402

from noveum_trace.core.span import SpanStatus  # noqa: E402
from noveum_trace.integrations.livekit import LiveKitTTSWrapper  # noqa: E402

from ._fakes import (  # noqa: E402
    FakeBaseTTS,
    RecordingStream,
    make_synth_audio,
    one_span,
    spans_named,
)


# --------------------------------------------------------------------------- #
# C1 — batch synthesize span
# --------------------------------------------------------------------------- #
async def test_synthesize_batch_creates_span_with_values(lk_trace, lk_client):
    """Guards: ``tts.synthesize`` span, batch mode, input_text + request/segment/
    sample_rate/num_channels, job prefix, exact audio duration, and that the
    synthesized audio is exported with the span's context."""
    audios = [make_synth_audio(is_final=True, request_id="r1", segment_id="seg1")]
    base = FakeBaseTTS(chunked_stream=RecordingStream(audios))
    wrapper = LiveKitTTSWrapper(
        tts=base, session_id="s1", job_context={"room_name": "room-1"}
    )

    stream = wrapper.synthesize("Hello TTS")
    out = [a async for a in stream]

    assert len(out) == 1
    span = one_span(lk_trace, "tts.synthesize")
    attrs = span.attributes
    assert attrs["tts.input_text"] == "Hello TTS"
    assert attrs["tts.mode"] == "batch"
    assert attrs["tts.request_id"] == "r1"
    assert attrs["tts.segment_id"] == "seg1"
    assert attrs["tts.sample_rate"] == 24000
    assert attrs["tts.num_channels"] == 1
    assert attrs["tts.audio_duration_ms"] == 100.0  # one 0.1s frame
    assert attrs["tts.provider"] == "cartesia"
    assert attrs["tts.model"] == "sonic"
    assert attrs["job.room_name"] == "room-1"  # bare key -> job. prefix
    assert span.status == SpanStatus.OK
    # synthesized audio exported and tagged with this span (the wrapper's purpose)
    _, kwargs = lk_client.transport.export_audio.call_args
    assert kwargs["trace_id"] == span.trace_id
    assert kwargs["span_id"] == span.span_id
    assert kwargs["metadata"]["type"] == "tts"
    assert len(kwargs["audio_data"]) > 0


async def test_chunked_aclose_creates_span_when_not_finalized(lk_trace):
    """Guards: a batch stream interrupted before the final chunk still produces a
    span on aclose (no lost synthesis)."""
    audios = [make_synth_audio(is_final=False)]  # never final
    base = FakeBaseTTS(chunked_stream=RecordingStream(audios))
    wrapper = LiveKitTTSWrapper(tts=base, session_id="s1")

    stream = wrapper.synthesize("partial")
    _ = [a async for a in stream]  # no final -> no span yet
    assert spans_named(lk_trace, "tts.synthesize") == []

    await stream.aclose()
    span = one_span(lk_trace, "tts.synthesize")
    assert span.attributes["tts.input_text"] == "partial"


# --------------------------------------------------------------------------- #
# C2 — streaming span, accumulated push_text
# --------------------------------------------------------------------------- #
async def test_stream_accumulates_pushed_text(lk_trace):
    """Guards: ``tts.stream`` span, streaming mode, push_text accumulation into
    input_text, and text forwarded to the base stream."""
    base_stream = RecordingStream([make_synth_audio(is_final=True)])
    base = FakeBaseTTS(synth_stream=base_stream)
    wrapper = LiveKitTTSWrapper(tts=base, session_id="s1")

    stream = wrapper.stream()
    stream.push_text("Hello ")
    stream.push_text("world")
    _ = [a async for a in stream]

    span = one_span(lk_trace, "tts.stream")
    assert span.attributes["tts.input_text"] == "Hello world"
    assert span.attributes["tts.mode"] == "streaming"
    assert base_stream.pushed_text == ["Hello ", "world"]


# --------------------------------------------------------------------------- #
# C3 — text fallback chain: input_text -> delta_text -> "unknown"
# --------------------------------------------------------------------------- #
async def test_stream_falls_back_to_delta_text(lk_trace):
    """Guards: when no push_text was called, ``delta_text`` is used as input_text."""
    base_stream = RecordingStream(
        [make_synth_audio(is_final=True, delta_text="from delta")]
    )
    base = FakeBaseTTS(synth_stream=base_stream)
    wrapper = LiveKitTTSWrapper(tts=base, session_id="s1")

    stream = wrapper.stream()
    _ = [a async for a in stream]  # no push_text

    span = one_span(lk_trace, "tts.stream")
    assert span.attributes["tts.input_text"] == "from delta"


async def test_stream_falls_back_to_unknown(lk_trace):
    """Guards: no push_text and no delta_text -> literal ``"unknown"`` (never empty)."""
    base_stream = RecordingStream([make_synth_audio(is_final=True, delta_text="")])
    base = FakeBaseTTS(synth_stream=base_stream)
    wrapper = LiveKitTTSWrapper(tts=base, session_id="s1")

    stream = wrapper.stream()
    _ = [a async for a in stream]

    span = one_span(lk_trace, "tts.stream")
    assert span.attributes["tts.input_text"] == "unknown"


async def test_stream_non_final_creates_no_span(lk_trace):
    """Guards: non-final synthesized chunks must not emit a span."""
    base_stream = RecordingStream([make_synth_audio(is_final=False)])
    base = FakeBaseTTS(synth_stream=base_stream)
    wrapper = LiveKitTTSWrapper(tts=base, session_id="s1")

    stream = wrapper.stream()
    stream.push_text("hi")
    _ = [a async for a in stream]

    assert spans_named(lk_trace, "tts.stream") == []


async def test_no_trace_creates_no_tts_span(lk_client):
    """Guards: with no active trace, synthesize() still yields audio but creates
    no span and exports nothing."""
    from noveum_trace.core.context import set_current_trace

    set_current_trace(None)
    base = FakeBaseTTS(
        chunked_stream=RecordingStream([make_synth_audio(is_final=True)])
    )
    wrapper = LiveKitTTSWrapper(tts=base, session_id="s1")

    out = [a async for a in wrapper.synthesize("hi")]
    assert len(out) == 1
    assert not lk_client.transport.export_audio.called


# --------------------------------------------------------------------------- #
# C6 — type compatibility + capability propagation
# --------------------------------------------------------------------------- #
def test_wrapper_is_instance_and_propagates_capabilities():
    """Guards: wrapper is a real BaseTTS and forwards sample_rate/num_channels to
    ``super().__init__`` (LiveKit reads these off the TTS for audio routing)."""
    # distinct non-default values so a hardcoded/dropped forward is detectable
    base = FakeBaseTTS(sample_rate=22050, num_channels=2)
    wrapper = LiveKitTTSWrapper(tts=base, session_id="s1")
    assert isinstance(wrapper, BaseTTS)
    assert wrapper.sample_rate == 22050
    assert wrapper.num_channels == 2


# --------------------------------------------------------------------------- #
# C7 — event forwarding + aclose
# --------------------------------------------------------------------------- #
def test_forwards_events_from_base():
    base = FakeBaseTTS()
    wrapper = LiveKitTTSWrapper(tts=base, session_id="s1")
    metrics, errors = [], []
    wrapper.on("metrics_collected", metrics.append)
    wrapper.on("error", errors.append)

    base.emit("metrics_collected", {"type": "tts_metrics"})
    base.emit("error", {"kind": "boom"})

    assert metrics == [{"type": "tts_metrics"}]
    assert errors == [{"kind": "boom"}]


async def test_aclose_closes_base_and_stops_forwarding():
    base = FakeBaseTTS()
    wrapper = LiveKitTTSWrapper(tts=base, session_id="s1")
    received = []
    wrapper.on("metrics_collected", received.append)

    await wrapper.aclose()
    assert base.aclose_called is True

    base.emit("metrics_collected", {"type": "tts_metrics"})
    assert received == []
