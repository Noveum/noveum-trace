"""Unit tests for Pipecat LLM handler mixin (_handlers_llm)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def ff():
    pytest.importorskip("pipecat.frames.frames")
    from pipecat.frames import frames as _ff

    return _ff


def _obs():
    from noveum_trace.core.trace import Trace
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(capture_text=True, capture_function_calls=True)
    obs._trace = MagicMock(spec=Trace)
    obs._using_external_turn_tracking = True
    turn = MagicMock()
    turn.attributes = {}
    obs._current_turn_span = turn
    return obs


@pytest.mark.asyncio
async def test_llm_context_stash_and_flush(ff) -> None:
    obs = _obs()

    ctx = MagicMock()
    ctx.get_messages = MagicMock(return_value=[{"role": "user", "content": "hi"}])
    ctx.tools = None
    lcf = ff.LLMContextFrame(context=ctx)
    await obs._handle_llm_context(MagicMock(frame=lcf))
    assert "messages" in obs._pending_llm_context

    llm_span = MagicMock()
    llm_span.attributes = {}
    llm_span.finish = MagicMock()

    def _cs(_name: str, parent_span=None, attributes=None) -> MagicMock:
        llm_span.attributes.update(attributes or {})
        return llm_span

    obs._create_child_span = MagicMock(side_effect=_cs)

    src = MagicMock()
    src._settings = None
    await obs._handle_llm_response_start(
        MagicMock(frame=ff.LLMFullResponseStartFrame(), source=src)
    )

    assert obs._pending_llm_context == {}
    assert "llm.input" in llm_span.attributes
    obs._create_child_span.assert_called()


@pytest.mark.asyncio
async def test_llm_text_accumulates_and_end_writes_output(ff) -> None:
    obs = _obs()
    llm_span = MagicMock()
    llm_span.attributes = {}
    llm_span.finish = MagicMock()
    obs._active_llm_span = llm_span

    await obs._handle_llm_text(MagicMock(frame=ff.LLMTextFrame(text="hel")))
    await obs._handle_llm_text(MagicMock(frame=ff.LLMTextFrame(text="lo")))
    await obs._handle_llm_response_end(MagicMock())

    assert llm_span.attributes.get("llm.output") == "hello"
    llm_span.finish.assert_called_once()


@pytest.mark.asyncio
async def test_llm_thought_pipeline(ff) -> None:
    obs = _obs()
    llm_span = MagicMock()
    llm_span.attributes = {}
    llm_span.finish = MagicMock()
    obs._active_llm_span = llm_span

    await obs._handle_llm_thought_start(MagicMock())
    await obs._handle_llm_thought_text(
        MagicMock(frame=ff.LLMThoughtTextFrame(text="think"))
    )
    te = ff.LLMThoughtEndFrame()
    te.signature = "sig"
    await obs._handle_llm_thought_end(MagicMock(frame=te))
    await obs._handle_llm_response_end(MagicMock())

    assert "llm.thoughts" in llm_span.attributes


@pytest.mark.asyncio
async def test_function_call_start_result_cancel(ff) -> None:
    obs = _obs()
    llm_span = MagicMock()
    llm_span.attributes = {}
    llm_span.finish = MagicMock()
    obs._active_llm_span = llm_span

    prog = ff.FunctionCallInProgressFrame(
        function_name="fn", tool_call_id="t1", arguments="{}"
    )
    await obs._handle_function_call_start(MagicMock(frame=prog))
    res = ff.FunctionCallResultFrame(
        function_name="fn",
        tool_call_id="t1",
        arguments="{}",
        result="ok",
    )
    await obs._handle_function_call_result(MagicMock(frame=res))
    await obs._handle_llm_response_end(MagicMock())

    assert "llm.function_call_results" in llm_span.attributes

    obs2 = _obs()
    obs2._active_llm_span = MagicMock()
    obs2._active_llm_span.attributes = {}
    obs2._active_llm_span.finish = MagicMock()
    prog2 = ff.FunctionCallInProgressFrame(
        function_name="g", tool_call_id="t2", arguments="{}"
    )
    await obs2._handle_function_call_start(MagicMock(frame=prog2))
    can = ff.FunctionCallCancelFrame(function_name="g", tool_call_id="t2")
    await obs2._handle_function_call_cancel(MagicMock(frame=can))
    llm2 = obs2._active_llm_span
    await obs2._handle_llm_response_end(MagicMock())
    assert llm2 is not None
    results = llm2.attributes.get("llm.function_call_results", [])
    assert any(r.get("cancelled") for r in results)


@pytest.mark.asyncio
async def test_llm_summary_request_and_result(ff) -> None:
    if not hasattr(ff, "LLMContextSummaryRequestFrame") or not hasattr(
        ff, "LLMContextSummaryResultFrame"
    ):
        pytest.skip("Context summary frames not available")
    obs = _obs()
    req = ff.LLMContextSummaryRequestFrame(
        request_id="r1",
        context=MagicMock(),
        min_messages_to_keep=2,
        target_context_tokens=8000,
        summarization_prompt="sum",
        summarization_timeout=30.0,
    )
    await obs._handle_llm_summary_request(MagicMock(frame=req))
    assert obs._current_turn_span.attributes.get("llm.summary.request_id") == "r1"

    res = ff.LLMContextSummaryResultFrame(
        request_id="r1",
        summary="short",
        last_summarized_index=3,
        error=None,
    )
    await obs._handle_llm_summary_result(MagicMock(frame=res))
    assert obs._current_turn_span.attributes.get("llm.summary.text") == "short"


@pytest.mark.asyncio
async def test_pre_span_function_call_written_to_last_llm_span(ff) -> None:
    obs = _obs()
    last = MagicMock()
    last.attributes = {}
    obs._active_llm_span = None
    obs._last_llm_span = last

    prog = ff.FunctionCallInProgressFrame(
        function_name="late", tool_call_id="late1", arguments="{}"
    )
    await obs._handle_function_call_start(MagicMock(frame=prog))

    assert last.attributes.get("llm.function_calls")
    assert "late1" in obs._pre_span_function_call_ids
