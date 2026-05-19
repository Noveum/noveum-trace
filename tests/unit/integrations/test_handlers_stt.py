"""Unit tests for Pipecat STT handler mixin (_handlers_stt)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def ff():
    pytest.importorskip("pipecat.frames.frames")
    from pipecat.frames import frames as _ff

    return _ff


def _obs_with_trace():
    from noveum_trace.core.trace import Trace
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(capture_text=True, record_audio=False)
    obs._trace = MagicMock(spec=Trace)
    return obs


@pytest.mark.asyncio
async def test_handle_speech_control_sets_vad_present(ff) -> None:
    obs = _obs_with_trace()
    frame = ff.SpeechControlParamsFrame()
    frame.vad_params = {"x": 1}
    data = MagicMock(frame=frame)
    await obs._handle_speech_control_params(data)
    assert obs._vad_present is True


@pytest.mark.asyncio
async def test_handle_transcription_fallback_point_span(ff) -> None:
    obs = _obs_with_trace()
    obs._active_stt_span = None
    obs._using_external_turn_tracking = True
    obs._current_turn_span = MagicMock()

    stt_span = MagicMock()
    stt_span.attributes = {}
    stt_span.trace_id = "t"
    stt_span.span_id = "s"
    stt_span.finish = MagicMock()
    stt_span.is_finished = MagicMock(return_value=False)

    def _cs(_name: str, parent_span=None, attributes=None) -> MagicMock:
        stt_span.attributes.update(attributes or {})
        return stt_span

    obs._create_child_span = MagicMock(side_effect=_cs)

    tf = ff.TranscriptionFrame(text="hello final", user_id="u1", timestamp="ts1")
    await obs._handle_transcription(MagicMock(frame=tf, source=None))

    obs._create_child_span.assert_called()
    assert stt_span.attributes.get("stt.text") == "hello final"
    stt_span.finish.assert_called_once()


@pytest.mark.asyncio
async def test_handle_input_text_creates_span(ff) -> None:
    obs = _obs_with_trace()
    obs._using_external_turn_tracking = True
    obs._current_turn_span = MagicMock()

    stt_span = MagicMock()
    stt_span.attributes = {}
    stt_span.finish = MagicMock()

    def _cs(_name: str, parent_span=None, attributes=None) -> MagicMock:
        stt_span.attributes.update(attributes or {})
        return stt_span

    obs._create_child_span = MagicMock(side_effect=_cs)

    itf = ff.InputTextRawFrame(text="typed")
    await obs._handle_input_text(MagicMock(frame=itf))

    assert "stt.input_type" in stt_span.attributes
    assert stt_span.attributes["stt.input_type"] == "text"


@pytest.mark.asyncio
async def test_handle_stt_metadata_sets_trace_attr(ff) -> None:
    if not hasattr(ff, "STTMetadataFrame"):
        pytest.skip("STTMetadataFrame not in this Pipecat version")
    obs = _obs_with_trace()
    meta = ff.STTMetadataFrame(service_name="test-stt", ttfs_p99_latency=0.5)
    await obs._handle_stt_metadata(MagicMock(frame=meta))
    obs._trace.set_attributes.assert_called()
    attrs = obs._trace.set_attributes.call_args[0][0]
    assert attrs["stt.ttfs_p99_latency_ms"] == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_handle_user_audio_pins_source(ff) -> None:
    obs = _obs_with_trace()
    obs._record_audio = True
    src = MagicMock()
    AudioCls = getattr(ff, "UserAudioRawFrame", None) or ff.InputAudioRawFrame
    frame = AudioCls(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
    await obs._handle_user_audio(MagicMock(frame=frame, source=src))
    assert obs._stt_source_processor is src
    assert len(obs._stt_audio_buffer) == 1


@pytest.mark.asyncio
async def test_handle_interim_transcription_appends_and_event(ff) -> None:
    obs = _obs_with_trace()
    obs._active_stt_span = MagicMock()
    obs._active_stt_span.attributes = {}
    obs._active_stt_span.events = []

    inf = ff.InterimTranscriptionFrame(text="partial", user_id="u1", timestamp="ts1")
    with patch.object(obs, "_vad_speech_start_time", 0.0):
        with patch("asyncio.get_running_loop") as mloop:
            mloop.return_value.time.return_value = 0.1
            await obs._handle_interim_transcription(MagicMock(frame=inf))

    assert obs._stt_interim_results


@pytest.mark.asyncio
async def test_handle_transcription_empty_raw_buffer_not_upload_failed(ff) -> None:
    """Stock transport (no raw tap): empty raw buffer must not fail the STT span."""
    obs = _obs_with_trace()
    obs._record_audio = True
    obs._record_raw_input_audio = True
    obs._using_external_turn_tracking = True
    obs._current_turn_span = MagicMock()

    stt_span = MagicMock()
    stt_span.attributes = {}
    stt_span.trace_id = "t"
    stt_span.span_id = "s"
    stt_span.finish = MagicMock()
    obs._active_stt_span = stt_span
    obs._stt_audio_buffer = [MagicMock()]
    obs._stt_raw_audio_buffer = []

    with patch(
        "noveum_trace.integrations.pipecat._handlers_stt.upload_audio_frames",
        return_value=True,
    ) as mock_upload:
        tf = ff.TranscriptionFrame(text="hi", user_id="u1", timestamp="ts1")
        await obs._handle_transcription(MagicMock(frame=tf, source=None))

    assert mock_upload.call_count == 1
    assert "stt.audio_uuid" in stt_span.attributes
    assert "stt.raw_audio_uuid" not in stt_span.attributes
    assert stt_span.attributes["pipecat_span_status"] == "ok"


@pytest.mark.asyncio
async def test_handle_transcription_dual_audio_upload(ff) -> None:
    obs = _obs_with_trace()
    obs._record_audio = True
    obs._record_raw_input_audio = True
    obs._using_external_turn_tracking = True
    obs._current_turn_span = MagicMock()

    stt_span = MagicMock()
    stt_span.attributes = {}
    stt_span.trace_id = "t"
    stt_span.span_id = "s"
    stt_span.finish = MagicMock()

    obs._active_stt_span = stt_span
    obs._stt_audio_buffer = [MagicMock()]
    obs._stt_raw_audio_buffer = [MagicMock()]

    with patch(
        "noveum_trace.integrations.pipecat._handlers_stt.upload_audio_frames",
        side_effect=[True, True],
    ) as mock_upload:
        tf = ff.TranscriptionFrame(text="hi", user_id="u1", timestamp="ts1")
        await obs._handle_transcription(MagicMock(frame=tf, source=None))

    assert mock_upload.call_count == 2
    assert "stt.audio_uuid" in stt_span.attributes
    assert "stt.raw_audio_uuid" in stt_span.attributes
    assert stt_span.attributes["pipecat_span_status"] == "ok"
    assert obs._stt_audio_buffer == []
    assert obs._stt_raw_audio_buffer == []


@pytest.mark.asyncio
async def test_handle_transcription_raw_opt_out(ff) -> None:
    obs = _obs_with_trace()
    obs._record_audio = True
    obs._record_raw_input_audio = False
    obs._using_external_turn_tracking = True
    obs._current_turn_span = MagicMock()

    stt_span = MagicMock()
    stt_span.attributes = {}
    stt_span.trace_id = "t"
    stt_span.span_id = "s"
    stt_span.finish = MagicMock()
    obs._active_stt_span = stt_span
    obs._stt_audio_buffer = [MagicMock()]
    obs._stt_raw_audio_buffer = [MagicMock()]

    with patch(
        "noveum_trace.integrations.pipecat._handlers_stt.upload_audio_frames",
        return_value=True,
    ) as mock_upload:
        tf = ff.TranscriptionFrame(text="hi", user_id="u1", timestamp="ts1")
        await obs._handle_transcription(MagicMock(frame=tf, source=None))

    assert mock_upload.call_count == 1
    assert "stt.audio_uuid" in stt_span.attributes
    assert "stt.raw_audio_uuid" not in stt_span.attributes


@pytest.mark.asyncio
async def test_handle_transcription_raw_upload_failed(ff) -> None:
    obs = _obs_with_trace()
    obs._record_audio = True
    obs._record_raw_input_audio = True
    obs._using_external_turn_tracking = True
    obs._current_turn_span = MagicMock()

    stt_span = MagicMock()
    stt_span.attributes = {}
    stt_span.trace_id = "t"
    stt_span.span_id = "s"
    stt_span.finish = MagicMock()
    obs._active_stt_span = stt_span
    obs._stt_audio_buffer = [MagicMock()]
    obs._stt_raw_audio_buffer = [MagicMock()]

    with patch(
        "noveum_trace.integrations.pipecat._handlers_stt.upload_audio_frames",
        side_effect=[True, False],
    ):
        tf = ff.TranscriptionFrame(text="hi", user_id="u1", timestamp="ts1")
        await obs._handle_transcription(MagicMock(frame=tf, source=None))

    assert "stt.audio_uuid" in stt_span.attributes
    assert "stt.raw_audio_uuid" not in stt_span.attributes
    assert stt_span.attributes["pipecat_span_status"] == "upload_failed"


@pytest.mark.asyncio
async def test_bounded_append_stt_frame_drops_oldest(ff) -> None:
    from noveum_trace.integrations.pipecat.pipecat_constants import MAX_STT_AUDIO_FRAMES
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    buf: list = []
    frame = ff.InputAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)
    for _ in range(MAX_STT_AUDIO_FRAMES + 5):
        obs._bounded_append_stt_frame(buf, frame)
    assert len(buf) == MAX_STT_AUDIO_FRAMES


@pytest.mark.asyncio
async def test_handle_vad_stt_start_opens_span_when_vad_present(ff) -> None:
    try:
        from pipecat.processors.frame_processor import FrameDirection
    except ImportError:
        pytest.skip("pipecat.processors.frame_processor not available")

    obs = _obs_with_trace()
    obs._vad_present = True
    obs._using_external_turn_tracking = True
    obs._current_turn_span = MagicMock()

    span = MagicMock()
    span.attributes = {}
    obs._create_child_span = MagicMock(return_value=span)

    data = MagicMock()
    data.direction = FrameDirection.UPSTREAM
    data.frame = ff.VADUserStartedSpeakingFrame()

    await obs._handle_vad_stt_start(data)
    assert obs._active_stt_span is span
