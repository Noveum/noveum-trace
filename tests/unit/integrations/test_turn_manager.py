"""Unit tests for Pipecat turn manager mixin (_turn_manager)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _obs():
    pytest.importorskip("pipecat.observers.base_observer")
    from noveum_trace.core.trace import Trace
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    o = NoveumTraceObserver(turn_end_timeout_secs=0.01)
    o._trace = MagicMock(spec=Trace)
    return o


@pytest.mark.asyncio
async def test_start_new_turn_creates_span_and_increments() -> None:
    obs = _obs()
    turn = MagicMock()
    turn.attributes = {}
    turn.finish = MagicMock()
    obs._trace.create_span = MagicMock(return_value=turn)

    await obs._start_new_turn()

    assert obs._current_turn_span is turn
    assert obs._current_turn_number == 1
    obs._trace.create_span.assert_called_once()


@pytest.mark.asyncio
async def test_end_current_turn_sets_duration_and_user_input() -> None:
    obs = _obs()
    turn = MagicMock()
    turn.attributes = {}
    turn.finish = MagicMock()
    obs._current_turn_span = turn
    obs._turn_start_time = 0.0
    obs._transcription_buffer = ["hello", "world"]

    with patch("asyncio.get_running_loop") as mloop:
        mloop.return_value.time.return_value = 2.0
        await obs._end_current_turn(was_interrupted=False)

    assert turn.attributes.get("turn.duration_seconds") == pytest.approx(2.0)
    assert "turn.user_input" in turn.attributes
    turn.finish.assert_called_once()
    assert obs._current_turn_span is None


@pytest.mark.asyncio
async def test_handle_error_marks_spans() -> None:
    obs = _obs()
    obs._trace.attributes = {}
    llm = MagicMock()
    llm.attributes = {}
    llm.is_finished.return_value = False
    tts = MagicMock()
    tts.attributes = {}
    turn = MagicMock()
    turn.attributes = {}
    turn.events = []
    llm_source = SimpleNamespace(name="llm")
    tts_source = SimpleNamespace(name="tts")
    operation = obs._llm_operations.start(llm_source, span=llm, processor_name="llm")
    obs._active_llm_span = llm
    obs._active_tts_span = tts
    obs._tts_source_processor = tts_source
    obs._current_turn_span = turn

    err = MagicMock()
    err.error = "boom"
    data = SimpleNamespace(frame=err, source=llm_source)
    await obs._handle_error(data)

    assert llm.attributes.get("pipecat_span_status") == "error"
    assert operation.error == {"message": "boom"}
    assert tts.attributes.get("pipecat_span_status") is None
    assert obs._trace.attributes.get("pipecat_span_status") == "error"

    await obs._handle_error(SimpleNamespace(frame=err, source=tts_source))
    assert tts.attributes.get("pipecat_span_status") == "error"


@pytest.mark.asyncio
async def test_late_error_upgrades_finished_llm_and_native_trace_status() -> None:
    from noveum_trace.core.span import SpanStatus
    from noveum_trace.core.trace import Trace

    obs = _obs()
    trace = Trace("conversation")
    obs._trace = trace
    source = SimpleNamespace(name="llm")
    span = trace.create_span("pipecat.llm")
    operation = obs._llm_operations.start(source, span=span, processor_name="llm")
    obs._finalize_llm_operation(
        operation,
        complete=True,
        termination_reason="response_end",
        terminal_status="ok",
    )

    err = SimpleNamespace(error="late provider error")
    await obs._handle_error(SimpleNamespace(frame=err, source=source))

    assert span.status == SpanStatus.ERROR
    assert trace.status == SpanStatus.ERROR
    assert trace.error_count == 1


@pytest.mark.asyncio
async def test_interruption_preserves_partial_llm_and_tts() -> None:
    from noveum_trace.core.span import SpanStatus

    obs = _obs()
    llm = MagicMock()
    llm.attributes = {}
    llm.finish = MagicMock()
    llm.is_finished.return_value = False
    tts = MagicMock()
    tts.attributes = {}
    tts.finish = MagicMock()
    tts.is_finished.return_value = False
    tts.trace_id = "trace"
    tts.span_id = "tts"
    turn = MagicMock()
    turn.attributes = {}
    llm_source = SimpleNamespace(name="llm")
    operation = obs._llm_operations.start(llm_source, span=llm, processor_name="llm")
    operation.output_chunks.extend(["partial ", "answer"])
    operation.thought_chunks.append("unfinished thought")
    operation.requested_function_calls["tool-1"] = {
        "tool_call_id": "tool-1",
        "name": "lookup",
    }
    obs._active_llm_span = llm
    obs._active_tts_span = tts
    obs._tts_source_processor = SimpleNamespace(name="tts")
    obs._tts_text_buffer = ["partial speech"]
    obs._tts_audio_buffer = [SimpleNamespace(audio=b"\x00\x00")]
    obs._current_turn_span = turn

    with patch(
        "noveum_trace.integrations.pipecat._handlers_tts.upload_audio_frames",
        return_value=True,
    ) as upload:
        await obs._handle_interruption_internal(interrupted_by_user=True)
        await obs._handle_interruption_internal(interrupted_by_user=True)

    assert obs._active_llm_span is None
    assert obs._active_tts_span is None
    assert llm.attributes["llm.output"] == "partial answer"
    assert llm.attributes["llm.thoughts"] == ["unfinished thought"]
    assert llm.attributes["llm.output.complete"] is False
    assert llm.attributes["llm.termination_reason"] == "user_interruption"
    assert tts.attributes["tts.input_text"] == "partial speech"
    assert tts.attributes["tts.output.complete"] is False
    assert tts.attributes["tts.audio.complete"] is False
    llm.set_status.assert_called_once_with(SpanStatus.OK)
    tts.set_status.assert_called_once_with(SpanStatus.OK)
    llm.finish.assert_called_once()
    tts.finish.assert_called_once()
    upload.assert_called_once()


@pytest.mark.asyncio
async def test_interruption_does_not_overwrite_error_status() -> None:
    obs = _obs()
    source = SimpleNamespace(name="llm")
    span = MagicMock()
    span.attributes = {"pipecat_span_status": "error"}
    span.is_finished.return_value = False
    operation = obs._llm_operations.start(source, span=span, processor_name="llm")
    operation.error = {"message": "provider failed"}

    await obs._handle_interruption_internal(interrupted_by_user=True)

    assert span.attributes["pipecat_span_status"] == "error"


@pytest.mark.asyncio
async def test_deferred_turn_end_closes_turn() -> None:
    obs = _obs()
    turn = MagicMock()
    turn.attributes = {}
    turn.finish = MagicMock()
    obs._current_turn_span = turn
    obs._is_bot_speaking = False

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await obs._deferred_turn_end()

    turn.finish.assert_called_once()


@pytest.mark.asyncio
async def test_handle_stop_frame_finishes_conversation() -> None:
    obs = _obs()
    with patch.object(obs, "_finish_conversation", new_callable=AsyncMock) as fin:
        await obs._handle_stop_frame(MagicMock())
    fin.assert_called_once_with(cancelled=False)


@pytest.mark.asyncio
async def test_user_mute_events_append_to_turn() -> None:
    obs = _obs()
    turn = MagicMock()
    turn.events = []
    obs._current_turn_span = turn

    await obs._handle_user_mute_started(MagicMock())
    await obs._handle_user_mute_stopped(MagicMock())

    assert len(turn.events) == 2


@pytest.mark.asyncio
async def test_client_bot_connected_events_on_trace() -> None:
    obs = _obs()
    obs._trace.events = []

    await obs._handle_client_connected(MagicMock())
    await obs._handle_bot_connected(MagicMock())

    names = [e.name for e in obs._trace.events]
    assert "client.connected" in names
    assert "bot.connected" in names
