"""Unit tests for Pipecat TTS handler mixin (_handlers_tts)."""

from __future__ import annotations

from types import SimpleNamespace
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
    source = SimpleNamespace(name="tts")
    obs._tts_source_processor = source

    await obs._handle_tts_text(
        SimpleNamespace(
            frame=ff.TTSTextFrame(text="hi", aggregated_by="sentence"),
            source=source,
        )
    )
    with patch(
        "noveum_trace.integrations.pipecat._handlers_tts.upload_audio_frames",
        return_value=True,
    ):
        await obs._handle_tts_stopped(
            SimpleNamespace(frame=ff.TTSStoppedFrame(), source=source)
        )

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


@pytest.mark.asyncio
async def test_late_stop_for_old_context_does_not_finish_new_tts(ff) -> None:
    obs = _obs()
    source = SimpleNamespace(name="tts")
    span = MagicMock()
    span.attributes = {}
    span.is_finished.return_value = False
    obs._active_tts_span = span
    obs._tts_source_processor = source
    obs._tts_context_id = "new-context"

    await obs._handle_tts_stopped(
        SimpleNamespace(
            frame=ff.TTSStoppedFrame(context_id="old-context"), source=source
        )
    )

    assert obs._active_tts_span is span
    span.finish.assert_not_called()


@pytest.mark.asyncio
async def test_late_contextless_stop_uses_frame_generation(ff) -> None:
    obs = _obs()
    source = SimpleNamespace(name="tts")
    span = MagicMock()
    span.attributes = {}
    span.is_finished.return_value = False
    obs._active_tts_span = span
    obs._tts_source_processor = source
    obs._tts_context_id = None
    obs._tts_start_frame_id = 200
    old_stop = ff.TTSStoppedFrame()
    old_stop.id = 100

    await obs._handle_tts_stopped(SimpleNamespace(frame=old_stop, source=source))

    assert obs._active_tts_span is span
    span.finish.assert_not_called()
