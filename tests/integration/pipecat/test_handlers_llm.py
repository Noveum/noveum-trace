"""Unit tests for Pipecat LLM handler mixin (_handlers_llm)."""

from __future__ import annotations

import json
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
    await obs._handle_llm_context(MagicMock(frame=lcf))
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

    # Source-less context is a broadcast fallback retained so every LLM processor
    # can consume the same generation once.
    assert "messages" in obs._pending_llm_context
    assert "llm.input" in llm_span.attributes
    obs._create_child_span.assert_called()


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
    results = json.loads(llm2.attributes.get("llm.function_call_results", "[]"))
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
    # A function-call frame arriving after the requesting span has closed (active
    # span is None) is written immediately to the requesting span via the backref.
    obs = _obs()
    source, last = await _start_llm(obs, ff)
    await obs._handle_llm_response_end(SimpleNamespace(source=source))

    prog = ff.FunctionCallInProgressFrame(
        function_name="late", tool_call_id="late1", arguments="{}"
    )
    await obs._handle_function_call_start(SimpleNamespace(frame=prog, source=source))

    calls = json.loads(last.attributes.get("llm.function_calls", "[]"))
    assert any(c.get("tool_call_id") == "late1" for c in calls)


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
async def test_parallel_processors_receive_only_destination_context(ff) -> None:
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
async def test_downstream_google_signature_dispatches_from_llm_source(ff) -> None:
    from pipecat.processors.aggregators.llm_context import LLMSpecificMessage

    obs = _obs()
    source, span = await _start_llm(obs, ff)
    signature = LLMSpecificMessage(
        llm="google",
        message={"type": "thought_signature", "signature": "SIG-real"},
    )
    frame = ff.LLMMessagesAppendFrame(messages=[signature])

    await obs.on_push_frame(
        SimpleNamespace(
            frame=frame,
            source=source,
            destination=SimpleNamespace(name="assistant-context"),
        )
    )
    await obs._handle_llm_response_end(SimpleNamespace(source=source))

    assert span.attributes["llm.thought_signatures"] == ["SIG-real"]


@pytest.mark.asyncio
async def test_sourceless_signature_with_concurrent_operations_is_dropped(ff) -> None:
    """A signature that resolves to no single operation has no correlation key.

    It must be discarded rather than handed to whichever operation ends next.
    """
    from pipecat.processors.aggregators.llm_context import LLMSpecificMessage

    obs = _obs()
    source_a, span_a = await _start_llm(
        obs, ff, SimpleNamespace(name="llm-a", _settings=None)
    )
    source_b, span_b = await _start_llm(
        obs, ff, SimpleNamespace(name="llm-b", _settings=None)
    )
    signature = LLMSpecificMessage(
        llm="google", message={"type": "thought_signature", "signature": "SIG-x"}
    )
    await obs._handle_llm_messages_append(
        SimpleNamespace(
            frame=ff.LLMMessagesAppendFrame(messages=[signature]),
            source=None,
            destination=None,
        )
    )
    await obs._handle_llm_response_end(SimpleNamespace(source=source_a))
    await obs._handle_llm_response_end(SimpleNamespace(source=source_b))

    assert "llm.thought_signatures" not in span_a.attributes
    assert "llm.thought_signatures" not in span_b.attributes


@pytest.mark.asyncio
async def test_idless_in_progress_resolves_against_started_batch(ff) -> None:
    """ID-less FunctionCallInProgressFrame must reuse the batch's minted IDs.

    Regression: the in-progress path minted ``tool-<len(dict)+1>`` which, after a
    FunctionCallsStartedFrame had recorded N ID-less calls, produced a phantom
    N+1th entry instead of enriching ``tool-1``.
    """
    obs = _obs()
    source, _span = await _start_llm(obs, ff)
    started = ff.FunctionCallsStartedFrame(
        function_calls=[
            SimpleNamespace(function_name="a", tool_call_id="", arguments={}),
            SimpleNamespace(function_name="b", tool_call_id="", arguments={}),
        ]
    )
    await obs._handle_function_calls_started(
        SimpleNamespace(frame=started, source=source)
    )
    operation = obs._llm_operations.get_active(source)
    batch_ids = list(operation.requested_function_calls)
    assert batch_ids == [
        f"{operation.operation_id}:tool-1",
        f"{operation.operation_id}:tool-2",
    ]

    async def _in_progress(name: str, arguments: dict) -> None:
        await obs._handle_function_call_start(
            SimpleNamespace(
                frame=ff.FunctionCallInProgressFrame(
                    function_name=name, tool_call_id="", arguments=arguments
                ),
                source=source,
            )
        )

    await _in_progress("a", {"x": 1})
    # Positional resolution: no phantom third entry, tool-1 got enriched.
    assert list(operation.requested_function_calls) == batch_ids
    enriched = operation.requested_function_calls[batch_ids[0]]
    assert enriched["arguments"] == '{"x": 1}'

    await _in_progress("b", {})
    assert list(operation.requested_function_calls) == batch_ids

    # Batch exhausted: mint a fresh ID that cannot collide with the batch.
    await _in_progress("c", {})
    assert list(operation.requested_function_calls) == batch_ids + [
        f"{operation.operation_id}:tool-3"
    ]


@pytest.mark.asyncio
async def test_idless_result_and_cancel_are_recorded_uncorrelated(ff) -> None:
    """A result/cancel with no tool_call_id is kept, built from the frame alone."""
    obs = _obs()
    source, _span = await _start_llm(obs, ff)
    await obs._handle_function_call_result(
        SimpleNamespace(
            frame=ff.FunctionCallResultFrame(
                function_name="f", tool_call_id="", arguments={}, result={"ok": True}
            ),
            source=source,
        )
    )
    await obs._handle_function_call_cancel(
        SimpleNamespace(
            frame=ff.FunctionCallCancelFrame(function_name="g", tool_call_id=""),
            source=source,
        )
    )
    operation = obs._llm_operations.get_active(source)
    result, cancelled = operation.function_call_results
    assert result["tool_call_id"] == ""
    assert result["name"] == "f"
    assert json.loads(result["result"]) == {"ok": True}
    assert "cancelled" not in result
    assert cancelled == {"tool_call_id": "", "name": "g", "cancelled": True}
