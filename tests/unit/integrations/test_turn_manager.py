"""Unit tests for Pipecat turn manager mixin (_turn_manager)."""

from __future__ import annotations

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
    tts = MagicMock()
    tts.attributes = {}
    turn = MagicMock()
    turn.attributes = {}
    turn.events = []
    obs._active_llm_span = llm
    obs._active_tts_span = tts
    obs._current_turn_span = turn

    err = MagicMock()
    err.error = "boom"
    data = MagicMock(frame=err)
    await obs._handle_error(data)

    assert llm.attributes.get("pipecat_span_status") == "error"
    assert tts.attributes.get("pipecat_span_status") == "error"
    assert obs._trace.attributes.get("pipecat_span_status") == "error"


@pytest.mark.asyncio
async def test_handle_interruption_internal_clears_llm_tts() -> None:
    obs = _obs()
    llm = MagicMock()
    llm.attributes = {}
    llm.finish = MagicMock()
    tts = MagicMock()
    tts.attributes = {}
    tts.finish = MagicMock()
    turn = MagicMock()
    turn.attributes = {}
    obs._active_llm_span = llm
    obs._active_tts_span = tts
    obs._current_turn_span = turn

    await obs._handle_interruption_internal(interrupted_by_user=True)

    assert obs._active_llm_span is None
    assert obs._active_tts_span is None
    llm.finish.assert_called_once()
    tts.finish.assert_called_once()


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
