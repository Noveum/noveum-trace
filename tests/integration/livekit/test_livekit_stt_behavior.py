"""
STT wrapper behavioral / regression tests (Section B of LIVEKIT_TEST_PLAN.md).

Unlike the existing ``test_livekit.py`` STT tests (which mock the client and only
assert ``start_span``/``finish_span`` were *called*), these drive a REAL client
and assert the captured span's name, attribute *values*, status, and the
audio-upload wiring. Each test names the regression it guards.
"""

from __future__ import annotations

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents.stt import STT as BaseSTT  # noqa: E402
from livekit.agents.stt import SpeechEventType  # noqa: E402

from noveum_trace.core.context import set_current_trace  # noqa: E402
from noveum_trace.core.span import SpanStatus  # noqa: E402
from noveum_trace.integrations.livekit import LiveKitSTTWrapper  # noqa: E402

from ._fakes import (  # noqa: E402
    FakeBaseSTT,
    RecordingStream,
    make_frame,
    make_speech_event,
    one_span,
    spans_named,
)


# --------------------------------------------------------------------------- #
# B1 — batch recognition captures real attribute VALUES (not just "was called")
# --------------------------------------------------------------------------- #
async def test_recognize_impl_captures_transcript_and_metadata(lk_trace):
    """Guards: span name ``stt.recognize`` + transcript/confidence/language/mode
    values + job-context prefix normalization. Catches attribute swaps/drops the
    existing mock-only test cannot."""
    event = make_speech_event(text="Hello world", confidence=0.95, language="en")
    base = FakeBaseSTT(recognize_event=event)
    wrapper = LiveKitSTTWrapper(
        stt=base, session_id="s1", job_context={"job_id": "job_abc"}
    )

    buffer = [make_frame(0.5), make_frame(0.5)]  # 1.0s total -> 1000.0 ms
    result = await wrapper._recognize_impl(buffer)

    assert result is event  # original event passed through unchanged

    span = one_span(lk_trace, "stt.recognize")
    attrs = span.attributes
    assert attrs["stt.transcript"] == "Hello world"
    assert attrs["stt.confidence"] == 0.95
    assert attrs["stt.language"] == "en"
    assert attrs["stt.is_final"] is True
    assert attrs["stt.mode"] == "batch"
    assert attrs["stt.event_type"] == SpeechEventType.FINAL_TRANSCRIPT.value
    assert attrs["stt.request_id"] == "req_123"
    assert attrs["stt.provider"] == "deepgram"
    assert attrs["stt.model"] == "nova-2"
    assert attrs["stt.audio_duration_ms"] == 1000.0
    # job_context {"job_id": ...} normalizes to "job.id"
    assert attrs["job.id"] == "job_abc"
    assert "metadata" in attrs
    assert span.status == SpanStatus.OK


async def test_recognize_impl_uploads_audio_with_span_context(lk_trace, lk_client):
    """Guards: STT audio is exported and tagged with the *new span's*
    trace_id/span_id and ``type='stt'`` (mis-wiring would orphan the audio)."""
    base = FakeBaseSTT(recognize_event=make_speech_event())
    wrapper = LiveKitSTTWrapper(stt=base, session_id="s1")

    await wrapper._recognize_impl([make_frame(0.5)])

    span = one_span(lk_trace, "stt.recognize")
    export = lk_client.transport.export_audio
    assert export.called
    _, kwargs = export.call_args
    assert kwargs["trace_id"] == span.trace_id
    assert kwargs["span_id"] == span.span_id
    assert kwargs["metadata"]["type"] == "stt"


async def test_recognize_impl_without_trace_creates_no_span(lk_client):
    """Guards: no active trace -> no span, event still returned (no crash)."""
    set_current_trace(None)
    event = make_speech_event()
    wrapper = LiveKitSTTWrapper(stt=FakeBaseSTT(recognize_event=event), session_id="s1")

    result = await wrapper._recognize_impl([make_frame(0.5)])

    assert result is event
    # No trace -> no span and, crucially, no orphaned audio export.
    assert not lk_client.transport.export_audio.called


# --------------------------------------------------------------------------- #
# B3/B4 — streaming creates a span only on FINAL_TRANSCRIPT, and clears buffer
# --------------------------------------------------------------------------- #
async def test_stream_final_transcript_creates_span_and_clears_buffer(
    lk_trace, lk_client
):
    """Guards: ``stt.stream`` span on FINAL only, mode='streaming', frames
    buffered+forwarded (proven via duration, not the vacuous empty-buffer check),
    audio exported with the span's ids, and per-utterance buffer reset."""
    final = make_speech_event(
        text="streamed text", event_type=SpeechEventType.FINAL_TRANSCRIPT
    )
    base_stream = RecordingStream([final])
    base = FakeBaseSTT(stream=base_stream)
    wrapper = LiveKitSTTWrapper(stt=base, session_id="s1")

    stream = wrapper.stream()
    stream.push_frame(make_frame(0.5))
    stream.push_frame(make_frame(0.5))
    events = [ev async for ev in stream]

    assert events == [final]
    span = one_span(lk_trace, "stt.stream")
    assert span.attributes["stt.transcript"] == "streamed text"
    assert span.attributes["stt.mode"] == "streaming"
    # duration proves the two frames were actually buffered & used (0.5s + 0.5s).
    # If frame buffering regressed, this would be 0.0 even though the buffer
    # ends up empty either way.
    assert span.attributes["stt.audio_duration_ms"] == 1000.0
    # frames were forwarded to the underlying base stream
    assert len(base_stream.pushed_frames) == 2
    # audio exported tagged with this span's context and type="stt"
    _, kwargs = lk_client.transport.export_audio.call_args
    assert kwargs["trace_id"] == span.trace_id
    assert kwargs["span_id"] == span.span_id
    assert kwargs["metadata"]["type"] == "stt"
    # buffer cleared after emitting the final-transcript span
    assert stream._buffered_frames == []


async def test_stream_interim_transcript_creates_no_span(lk_trace):
    """Guards: interim transcripts must NOT create spans (only finals do)."""
    interim = make_speech_event(event_type=SpeechEventType.INTERIM_TRANSCRIPT)
    base = FakeBaseSTT(stream=RecordingStream([interim]))
    wrapper = LiveKitSTTWrapper(stt=base, session_id="s1")

    stream = wrapper.stream()
    stream.push_frame(make_frame(0.2))
    _ = [ev async for ev in stream]

    assert spans_named(lk_trace, "stt.stream") == []


# --------------------------------------------------------------------------- #
# B6 — type compatibility (the whole reason the wrapper subclasses BaseSTT)
# --------------------------------------------------------------------------- #
def test_wrapper_is_instance_of_base_stt():
    """Guards: wrapper must be a real BaseSTT subclass or LiveKit's
    agent_activity rejects it at runtime."""
    wrapper = LiveKitSTTWrapper(stt=FakeBaseSTT(), session_id="s1")
    assert isinstance(wrapper, BaseSTT)


# --------------------------------------------------------------------------- #
# B7/B8 — event forwarding and aclose unregistration (real EventEmitter)
# --------------------------------------------------------------------------- #
def test_forwards_metrics_and_error_events_from_base():
    """Guards: events emitted by the base STT reach listeners on the wrapper
    (agent_activity relies on this)."""
    base = FakeBaseSTT()
    wrapper = LiveKitSTTWrapper(stt=base, session_id="s1")

    metrics, errors = [], []
    wrapper.on("metrics_collected", metrics.append)
    wrapper.on("error", errors.append)

    base.emit("metrics_collected", {"type": "stt_metrics"})
    base.emit("error", {"kind": "boom"})

    assert metrics == [{"type": "stt_metrics"}]
    assert errors == [{"kind": "boom"}]


async def test_aclose_unregisters_forwarding_and_closes_base():
    """Guards: ``aclose`` stops forwarding (no leaked handlers) and closes base."""
    base = FakeBaseSTT()
    wrapper = LiveKitSTTWrapper(stt=base, session_id="s1")

    received = []
    wrapper.on("metrics_collected", received.append)

    await wrapper.aclose()
    assert base.aclose_called is True

    # After aclose the base no longer forwards through the wrapper.
    base.emit("metrics_collected", {"type": "stt_metrics"})
    assert received == []
