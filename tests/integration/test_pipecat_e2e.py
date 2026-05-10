"""Integration-style tests for Pipecat observer (frame sequences, mocked client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_conversation_flow_start_llm_tts_end() -> None:
    pytest.importorskip("pipecat.frames.frames")
    from pipecat.frames import frames as ff

    from noveum_trace.core.trace import Trace
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(capture_text=True, record_audio=False)
    mock_trace = MagicMock(spec=Trace)
    mock_trace.attributes = {}
    mock_trace.events = []
    mock_trace.set_attributes = MagicMock()
    mock_trace.finish = MagicMock()
    mock_trace.finish_span = MagicMock()

    spans: list[MagicMock] = []

    def _mk_span(**kwargs: object) -> MagicMock:
        s = MagicMock()
        attrs = kwargs.get("attributes")
        s.attributes = dict(attrs) if isinstance(attrs, dict) else {}
        s.trace_id = "trace-1"
        s.span_id = f"span-{len(spans)}"
        s.finish = MagicMock()
        s.is_finished = MagicMock(return_value=False)
        spans.append(s)
        return s

    mock_trace.create_span = MagicMock(side_effect=_mk_span)

    client = MagicMock()
    client.start_trace = MagicMock(return_value=mock_trace)
    client.finish_trace = MagicMock()
    client.flush = MagicMock()

    turn = MagicMock()
    turn.attributes = {}
    turn.finish = MagicMock()
    turn.span_id = "turn-1"

    with patch.object(obs, "_get_client", return_value=client):
        await obs.on_pipeline_started()
        assert obs._trace is mock_trace

    obs._using_external_turn_tracking = True
    obs._current_turn_span = turn

    await obs._handle_llm_response_start(
        MagicMock(
            frame=ff.LLMFullResponseStartFrame(),
            source=MagicMock(_settings=None),
        )
    )
    await obs._handle_llm_text(MagicMock(frame=ff.LLMTextFrame(text="Answer")))
    await obs._handle_llm_response_end(MagicMock())

    await obs._handle_tts_started(
        MagicMock(
            frame=ff.TTSStartedFrame(),
            source=MagicMock(_settings=None),
        )
    )
    await obs._handle_tts_text(
        MagicMock(frame=ff.TTSTextFrame(text="Answer"))
    )
    await obs._handle_tts_stopped(MagicMock())

    with patch.object(obs, "_get_client", return_value=client):
        await obs._finish_conversation()

    client.finish_trace.assert_called()
    assert obs._trace is None
