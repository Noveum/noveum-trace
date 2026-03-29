"""Unit tests for Pipecat TTS handler mixin (_handlers_tts)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def ff():
    pytest.importorskip("pipecat.frames.frames")
    from pipecat.frames import frames as _ff

    return _ff


def _obs():
    from noveum_trace.core.trace import Trace
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    o = NoveumTraceObserver(capture_text=True, record_audio=True)
    o._trace = MagicMock(spec=Trace)
    o._current_turn_span = MagicMock()
    return o


@pytest.mark.asyncio
async def test_tts_started_opens_span(ff) -> None:
    obs = _obs()
    tts_span = MagicMock()
    tts_span.attributes = {}
    tts_span.finish = MagicMock()
    tts_span.trace_id = "t"
    tts_span.span_id = "s"

    def _cs(_name: str, parent_span=None, attributes=None) -> MagicMock:
        tts_span.attributes.update(attributes or {})
        return tts_span

    obs._create_child_span = MagicMock(side_effect=_cs)
    src = MagicMock()
    src._settings = None

    await obs._handle_tts_started(MagicMock(frame=ff.TTSStartedFrame(), source=src))
    assert obs._active_tts_span is tts_span
    assert obs._tts_source_processor is src


@pytest.mark.asyncio
async def test_tts_text_and_stopped(ff) -> None:
    obs = _obs()
    tts_span = MagicMock()
    tts_span.attributes = {}
    tts_span.finish = MagicMock()
    tts_span.trace_id = "t"
    tts_span.span_id = "s"
    obs._active_tts_span = tts_span
    obs._tts_source_processor = MagicMock()

    await obs._handle_tts_text(
        MagicMock(frame=ff.TTSTextFrame(text="hi", aggregated_by="sentence"))
    )
    with patch(
        "noveum_trace.integrations.pipecat._handlers_tts.upload_audio_frames",
        return_value=True,
    ):
        await obs._handle_tts_stopped(MagicMock())

    assert tts_span.attributes.get("tts.input_text") == "hi"
    tts_span.finish.assert_called_once()


@pytest.mark.asyncio
async def test_tts_audio_ignores_wrong_source(ff) -> None:
    obs = _obs()
    pinned = MagicMock()
    obs._tts_source_processor = pinned
    other = MagicMock()
    frame = ff.TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
    await obs._handle_tts_audio(MagicMock(frame=frame, source=other))
    assert obs._tts_audio_buffer == []
