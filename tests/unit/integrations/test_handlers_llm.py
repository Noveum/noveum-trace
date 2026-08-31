"""Unit tests for Pipecat LLM handler mixin (_handlers_llm)."""

from __future__ import annotations

from types import SimpleNamespace
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


async def _start_llm(obs, ff, source=None):
    source = source or SimpleNamespace(name="test-llm", _settings=None)
    span = MagicMock()
    span.attributes = {}
    span.finish = MagicMock()
    span.is_finished.return_value = False
    obs._create_child_span = MagicMock(return_value=span)
    await obs._handle_llm_response_start(
        SimpleNamespace(frame=ff.LLMFullResponseStartFrame(), source=source)
    )
    return source, span


@pytest.mark.asyncio
async def test_llm_context_stash_and_flush(ff) -> None:
    obs = _obs()

    ctx = MagicMock()
    ctx.get_messages = MagicMock(return_value=[{"role": "user", "content": "hi"}])
    ctx.tools = None
    lcf = ff.LLMContextFrame(context=ctx)
    await obs._handle_llm_context(SimpleNamespace(frame=lcf, destination=None))
    assert "messages" in obs._pending_llm_context

    llm_span = MagicMock()
    llm_span.attributes = {}
    llm_span.finish = MagicMock()

    def _cs(_name: str, parent_span=None, attributes=None) -> MagicMock:
        llm_span.attributes.update(attributes or {})
        return llm_span

    obs._create_child_span = MagicMock(side_effect=_cs)

    src = SimpleNamespace(name="test-llm", _settings=None)
    await obs._handle_llm_response_start(
        SimpleNamespace(frame=ff.LLMFullResponseStartFrame(), source=src)
    )

    # Source-less context is a broadcast default retained for other LLM
    # processors; this exact processor records that it consumed the generation.
    assert "messages" in obs._pending_llm_context
    assert "llm.input" in llm_span.attributes
    obs._create_child_span.assert_called()


@pytest.mark.asyncio
async def test_source_less_context_is_consumed_once_by_each_llm(ff) -> None:
    obs = _obs()
    messages = [{"role": "user", "content": "shared"}]
    await obs._handle_llm_messages_replace(
        SimpleNamespace(
            frame=ff.LLMMessagesUpdateFrame(messages=messages), destination=None
        )
    )

    first = SimpleNamespace(name="first", _settings=None)
    second = SimpleNamespace(name="second", _settings=None)
    spans = []

    def _create(_name, parent_span=None, attributes=None):
        span = MagicMock()
        span.attributes = dict(attributes or {})
        span.is_finished.return_value = False
        spans.append(span)
        return span

    obs._create_child_span = MagicMock(side_effect=_create)
    for source in (first, second):
        await obs._handle_llm_response_start(
            SimpleNamespace(frame=ff.LLMFullResponseStartFrame(), source=source)
        )
        await obs._handle_llm_response_end(SimpleNamespace(source=source))

    # No new broadcast occurred, so the next invocation from the first processor
    # must not inherit stale input.
    await obs._handle_llm_response_start(
        SimpleNamespace(frame=ff.LLMFullResponseStartFrame(), source=first)
    )

    assert "llm.input" in spans[0].attributes
    assert "llm.input" in spans[1].attributes
    assert "llm.input" not in spans[2].attributes


@pytest.mark.asyncio
async def test_llm_text_accumulates_and_end_writes_output(ff) -> None:
    obs = _obs()
    source, llm_span = await _start_llm(obs, ff)

    await obs._handle_llm_text(
        SimpleNamespace(frame=ff.LLMTextFrame(text="hel"), source=source)
    )
    await obs._handle_llm_text(
        SimpleNamespace(frame=ff.LLMTextFrame(text="lo"), source=source)
    )
    await obs._handle_llm_response_end(SimpleNamespace(source=source))

    assert llm_span.attributes.get("llm.output") == "hello"
    llm_span.finish.assert_called_once()


@pytest.mark.asyncio
async def test_llm_text_without_matching_operation_is_not_buffered(ff) -> None:
    obs = _obs()
    source = SimpleNamespace(name="orphan")
    await obs._handle_llm_text(
        SimpleNamespace(frame=ff.LLMTextFrame(text="leak?"), source=source)
    )
    assert obs._llm_text_buffer == []


@pytest.mark.asyncio
async def test_llm_thought_pipeline(ff) -> None:
    obs = _obs()
    source, llm_span = await _start_llm(obs, ff)

    await obs._handle_llm_thought_start(SimpleNamespace(source=source))
    await obs._handle_llm_thought_text(
        SimpleNamespace(frame=ff.LLMThoughtTextFrame(text="think"), source=source)
    )
    te = ff.LLMThoughtEndFrame()
    te.signature = "sig"
    await obs._handle_llm_thought_end(SimpleNamespace(frame=te, source=source))
    await obs._handle_llm_response_end(SimpleNamespace(source=source))

    assert "llm.thoughts" in llm_span.attributes


@pytest.mark.asyncio
async def test_function_call_start_result_cancel(ff) -> None:
    obs = _obs()
    source, llm_span = await _start_llm(obs, ff)

    prog = ff.FunctionCallInProgressFrame(
        function_name="fn", tool_call_id="t1", arguments="{}"
    )
    await obs._handle_function_call_start(SimpleNamespace(frame=prog, source=source))
    res = ff.FunctionCallResultFrame(
        function_name="fn",
        tool_call_id="t1",
        arguments="{}",
        result="ok",
    )
    await obs._handle_function_call_result(SimpleNamespace(frame=res, source=source))
    await obs._handle_llm_response_end(SimpleNamespace(source=source))

    assert "llm.function_call_results" in llm_span.attributes

    obs2 = _obs()
    source2, llm2 = await _start_llm(obs2, ff)
    prog2 = ff.FunctionCallInProgressFrame(
        function_name="g", tool_call_id="t2", arguments="{}"
    )
    await obs2._handle_function_call_start(SimpleNamespace(frame=prog2, source=source2))
    can = ff.FunctionCallCancelFrame(function_name="g", tool_call_id="t2")
    await obs2._handle_function_call_cancel(SimpleNamespace(frame=can, source=source2))
    await obs2._handle_llm_response_end(SimpleNamespace(source=source2))
    assert llm2 is not None
    results = llm2.attributes.get("llm.function_call_results", [])
    assert any(r.get("cancelled") for r in results)


@pytest.mark.asyncio
async def test_function_calls_started_preserves_request_order(ff) -> None:
    obs = _obs()
    source, span = await _start_llm(obs, ff)
    calls = [
        ff.FunctionCallFromLLM(
            function_name="first", tool_call_id="one", arguments={"a": 1}, context=None
        ),
        ff.FunctionCallFromLLM(
            function_name="second", tool_call_id="two", arguments={}, context=None
        ),
    ]

    await obs._handle_function_calls_started(
        SimpleNamespace(frame=ff.FunctionCallsStartedFrame(calls), source=source)
    )
    await obs._handle_llm_response_end(SimpleNamespace(source=source))

    captured = span.attributes["llm.function_calls"]
    assert [call["request_order"] for call in captured] == [0, 1]
    assert [call["name"] for call in captured] == ["first", "second"]


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
    source, last = await _start_llm(obs, ff)
    await obs._handle_llm_response_end(SimpleNamespace(source=source))

    prog = ff.FunctionCallInProgressFrame(
        function_name="late", tool_call_id="late1", arguments="{}"
    )
    await obs._handle_function_call_start(SimpleNamespace(frame=prog, source=source))

    assert last.attributes.get("llm.function_calls")


@pytest.mark.asyncio
async def test_parallel_same_provider_processors_do_not_mix_output(ff) -> None:
    obs = _obs()
    source_a = SimpleNamespace(name="OpenAILLMService", _settings=None)
    source_b = SimpleNamespace(name="OpenAILLMService", _settings=None)
    source_a, span_a = await _start_llm(obs, ff, source_a)
    source_b, span_b = await _start_llm(obs, ff, source_b)

    await obs._handle_llm_text(
        SimpleNamespace(frame=ff.LLMTextFrame(text="A"), source=source_a)
    )
    await obs._handle_llm_text(
        SimpleNamespace(frame=ff.LLMTextFrame(text="B"), source=source_b)
    )
    await obs._handle_llm_response_end(SimpleNamespace(source=source_b))
    await obs._handle_llm_response_end(SimpleNamespace(source=source_a))

    assert span_a.attributes["llm.output"] == "A"
    assert span_b.attributes["llm.output"] == "B"
    assert (
        span_a.attributes["llm.operation_id"] != span_b.attributes["llm.operation_id"]
    )


@pytest.mark.asyncio
async def test_parallel_processors_receive_only_their_destination_context(ff) -> None:
    obs = _obs()
    source_a = SimpleNamespace(name="same", _settings=None)
    source_b = SimpleNamespace(name="same", _settings=None)
    obs.register_processor_role(source_a, "llm")
    obs.register_processor_role(source_b, "llm")

    for source, text in ((source_a, "for-a"), (source_b, "for-b")):
        ctx = MagicMock()
        ctx.get_messages.return_value = [{"role": "user", "content": text}]
        ctx.tools = None
        await obs._handle_llm_context(
            SimpleNamespace(frame=ff.LLMContextFrame(context=ctx), destination=source)
        )

    _, span_a = await _start_llm(obs, ff, source_a)
    _, span_b = await _start_llm(obs, ff, source_b)

    assert "for-a" in span_a.attributes["llm.input"]
    assert "for-b" not in span_a.attributes["llm.input"]
    assert "for-b" in span_b.attributes["llm.input"]
    assert "for-a" not in span_b.attributes["llm.input"]


@pytest.mark.asyncio
async def test_llm_marker_is_sideband_and_ordered(ff) -> None:
    obs = _obs()
    source, span = await _start_llm(obs, ff)

    for marker, immediate in (("○", True), ("✓", False)):
        frame = SimpleNamespace(marker=marker, append_to_context_immediately=immediate)
        await obs._handle_llm_marker(SimpleNamespace(frame=frame, source=source))
    await obs._handle_llm_text(
        SimpleNamespace(frame=ff.LLMTextFrame(text="spoken"), source=source)
    )
    await obs._handle_llm_response_end(SimpleNamespace(source=source))

    assert span.attributes["llm.output"] == "spoken"
    assert span.attributes["llm.markers"] == [
        {"marker": "○", "append_to_context_immediately": True},
        {"marker": "✓", "append_to_context_immediately": False},
    ]


@pytest.mark.asyncio
async def test_unmatched_llm_marker_is_retained_on_turn(ff) -> None:
    obs = _obs()
    frame = SimpleNamespace(marker="◐", append_to_context_immediately=True)

    await obs._handle_llm_marker(
        SimpleNamespace(frame=frame, source=SimpleNamespace(name="unknown"))
    )

    assert obs._current_turn_span.attributes["llm.markers"] == [
        {"marker": "◐", "append_to_context_immediately": True}
    ]


@pytest.mark.asyncio
async def test_late_end_frame_cannot_finish_next_same_processor_call(ff) -> None:
    obs = _obs()
    source = SimpleNamespace(name="sequential")
    first = MagicMock(attributes={})
    second = MagicMock(attributes={})
    second.is_finished.return_value = False
    obs._llm_operations.start(
        source, span=first, processor_name="sequential", start_frame_id=100
    )
    obs._llm_operations.complete(source)
    obs._llm_operations.start(
        source, span=second, processor_name="sequential", start_frame_id=200
    )
    late_end = ff.LLMFullResponseEndFrame()
    late_end.id = 150

    await obs._handle_llm_response_end(SimpleNamespace(frame=late_end, source=source))

    second.finish.assert_not_called()
    assert obs._llm_operations.get_active(source) is not None
