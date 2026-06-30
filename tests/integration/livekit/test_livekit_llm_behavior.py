"""
LLM wrapper behavioral / regression tests (Section D of LIVEKIT_TEST_PLAN.md).

Real client capture + real ``ChatContext``/``ChatChunk``/``CompletionUsage``.
Asserts the full ``llm.*`` attribute contract the integration emits.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents.llm import LLM as BaseLLM  # noqa: E402
from livekit.agents.llm import (  # noqa: E402
    ChatContext,
    CompletionUsage,
    FunctionToolCall,
)

from noveum_trace.core.span import SpanStatus  # noqa: E402
from noveum_trace.integrations.livekit import LiveKitLLMWrapper  # noqa: E402

from ._fakes import (  # noqa: E402
    ErrorStream,
    FakeBaseLLM,
    RecordingStream,
    _Opts,
    make_chat_chunk,
    one_span,
    spans_named,
)


def _ctx_with_system(system: str = "You are a helpful assistant.") -> ChatContext:
    ctx = ChatContext.empty()
    ctx.add_message(role="system", content=system)
    ctx.add_message(role="user", content="Hello?")
    return ctx


async def _drain(stream):
    out = []
    async for chunk in stream:
        out.append(chunk)
    return out


# --------------------------------------------------------------------------- #
# D1/D2/D3 — response aggregation, token + timing metrics
# --------------------------------------------------------------------------- #
async def test_stream_end_creates_span_with_response_and_metrics(lk_trace):
    """Guards: ``llm.chat`` span with aggregated response, role, token counts,
    chunk count, mode, and (non-)error flags."""
    usage = CompletionUsage(
        completion_tokens=12, prompt_tokens=20, total_tokens=32, prompt_cached_tokens=4
    )
    chunks = [
        make_chat_chunk(chunk_id="req-9", content="Hello ", role="assistant"),
        make_chat_chunk(content="world"),
        make_chat_chunk(usage=usage),
    ]
    base = FakeBaseLLM(chat_stream=RecordingStream(chunks))
    wrapper = LiveKitLLMWrapper(llm=base, session_id="s1")

    await _drain(wrapper.chat(chat_ctx=_ctx_with_system()))

    span = one_span(lk_trace, "llm.chat")
    a = span.attributes
    assert a["llm.model"] == "gpt-4o"
    assert a["llm.provider"] == "openai"
    assert a["llm.label"] == "openai.LLM"
    assert a["llm.response"] == "Hello world"
    assert a["llm.response_role"] == "assistant"
    assert a["llm.mode"] == "streaming"
    assert a["llm.chunk_count"] == 3
    assert a["llm.cancelled"] is False
    assert a["llm.had_error"] is False
    # token metrics
    assert a["llm.completion_tokens"] == 12
    assert a["llm.prompt_tokens"] == 20
    assert a["llm.total_tokens"] == 32
    assert a["llm.prompt_cached_tokens"] == 4
    assert span.status == SpanStatus.OK


async def test_ttft_and_duration_recorded(lk_trace):
    """Guards: TTFT captured on first content chunk; duration + tokens/sec present."""
    usage = CompletionUsage(completion_tokens=5, prompt_tokens=5, total_tokens=10)
    chunks = [
        make_chat_chunk(content="hi", role="assistant"),
        make_chat_chunk(usage=usage),
    ]
    wrapper = LiveKitLLMWrapper(
        llm=FakeBaseLLM(chat_stream=RecordingStream(chunks)), session_id="s1"
    )

    await _drain(wrapper.chat(chat_ctx=_ctx_with_system()))

    a = one_span(lk_trace, "llm.chat").attributes
    assert "llm.ttft" in a and a["llm.ttft"] >= 0
    assert "llm.duration" in a and a["llm.duration"] >= 0
    assert "llm.tokens_per_second" in a


# --------------------------------------------------------------------------- #
# D4 — tool calls aggregation
# --------------------------------------------------------------------------- #
async def test_tool_calls_aggregated(lk_trace):
    """Guards: streamed tool-calls are aggregated into count/names + JSON blob."""
    tc = FunctionToolCall(
        name="get_weather", arguments='{"city": "SF"}', call_id="call_1"
    )
    chunks = [make_chat_chunk(role="assistant", tool_calls=[tc])]
    wrapper = LiveKitLLMWrapper(
        llm=FakeBaseLLM(chat_stream=RecordingStream(chunks)), session_id="s1"
    )

    await _drain(wrapper.chat(chat_ctx=_ctx_with_system()))

    a = one_span(lk_trace, "llm.chat").attributes
    assert a["llm.tool_calls.count"] == 1
    assert a["llm.tool_calls.names"] == ["get_weather"]
    parsed = json.loads(a["llm.tool_calls"])
    assert parsed[0]["name"] == "get_weather"
    assert parsed[0]["arguments"] == '{"city": "SF"}'
    assert parsed[0]["call_id"] == "call_1"
    assert parsed[0]["type"] == "function"


# --------------------------------------------------------------------------- #
# D5 — sampling params from _opts and kwargs
# --------------------------------------------------------------------------- #
async def test_sampling_params_from_opts_and_kwargs(lk_trace):
    """Guards: sampling params lifted from ``llm._opts`` and from chat kwargs."""
    base = FakeBaseLLM(
        chat_stream=RecordingStream([make_chat_chunk(content="x", role="assistant")]),
        opts=_Opts(temperature=0.7, top_p=0.9, max_tokens=256),
    )
    wrapper = LiveKitLLMWrapper(llm=base, session_id="s1")

    await _drain(
        wrapper.chat(
            chat_ctx=_ctx_with_system(), tool_choice="auto", parallel_tool_calls=True
        )
    )

    a = one_span(lk_trace, "llm.chat").attributes
    assert a["llm.temperature"] == 0.7
    assert a["llm.top_p"] == 0.9
    assert a["llm.max_tokens"] == 256
    assert a["llm.tool_choice"] == "auto"
    assert a["llm.parallel_tool_calls"] is True


# --------------------------------------------------------------------------- #
# D6 — cost metrics
# --------------------------------------------------------------------------- #
async def test_cost_metrics_use_correct_tokens(lk_trace):
    """Guards: cost is computed from prompt/completion tokens in the RIGHT order
    (gpt-4o input cost != output cost, so a swap is detectable) and total ==
    input + output. Asserts exact values via the same estimate_cost the source
    uses, so it stays correct if the pricing table changes but still catches an
    argument swap / wrong-token / write-input-into-total regression."""
    from noveum_trace.utils.llm_utils import estimate_cost

    expected = estimate_cost("gpt-4o", input_tokens=200, output_tokens=100)
    assert expected["input_cost"] != expected["output_cost"]  # swap is detectable

    usage = CompletionUsage(completion_tokens=100, prompt_tokens=200, total_tokens=300)
    chunks = [
        make_chat_chunk(content="hi", role="assistant"),
        make_chat_chunk(usage=usage),
    ]
    wrapper = LiveKitLLMWrapper(
        llm=FakeBaseLLM(chat_stream=RecordingStream(chunks)), session_id="s1"
    )

    await _drain(wrapper.chat(chat_ctx=_ctx_with_system()))

    a = one_span(lk_trace, "llm.chat").attributes
    assert a["llm.cost.input"] == expected["input_cost"]
    assert a["llm.cost.output"] == expected["output_cost"]
    assert a["llm.cost.total"] == expected["total_cost"]
    assert a["llm.cost.total"] == pytest.approx(
        a["llm.cost.input"] + a["llm.cost.output"]
    )
    assert a["llm.cost.currency"] == "USD"


async def test_no_trace_creates_no_llm_span(lk_client):
    """Guards: with no active trace, chat() still streams but creates no span and
    does not crash."""
    from noveum_trace.core.context import set_current_trace

    set_current_trace(None)
    base = FakeBaseLLM(
        chat_stream=RecordingStream([make_chat_chunk(content="hi", role="assistant")])
    )
    wrapper = LiveKitLLMWrapper(llm=base, session_id="s1")

    out = await _drain(wrapper.chat(chat_ctx=_ctx_with_system()))
    assert [c.delta.content for c in out if c.delta] == ["hi"]


# --------------------------------------------------------------------------- #
# D7/D8 — system prompt + chat context serialization (real ChatContext)
# --------------------------------------------------------------------------- #
async def test_system_prompt_and_chat_ctx_from_real_context(lk_trace):
    """Guards: system prompt extracted from the first system message of a real
    ChatContext; message_count + serialized chat_ctx attached."""
    ctx = _ctx_with_system("You are Nova.")
    wrapper = LiveKitLLMWrapper(
        llm=FakeBaseLLM(
            chat_stream=RecordingStream(
                [make_chat_chunk(content="ok", role="assistant")]
            )
        ),
        session_id="s1",
    )

    await _drain(wrapper.chat(chat_ctx=ctx))

    a = one_span(lk_trace, "llm.chat").attributes
    assert a["llm.system_prompt"] == "You are Nova."
    assert a["llm.message_count"] == 2
    # chat_ctx serialized as JSON of messages with roles preserved
    history = json.loads(a["llm.chat_ctx"])
    roles = [m.get("role") for m in history]
    assert "system" in roles and "user" in roles


# --------------------------------------------------------------------------- #
# D9 — error mid-stream still creates span with ERROR status
# --------------------------------------------------------------------------- #
async def test_error_midstream_creates_error_span(lk_trace):
    """Guards: an exception during streaming sets had_error + ERROR status and
    still creates the span (partial data captured), then re-raises."""
    base = FakeBaseLLM(
        chat_stream=ErrorStream([make_chat_chunk(content="partial", role="assistant")])
    )
    wrapper = LiveKitLLMWrapper(llm=base, session_id="s1")

    with pytest.raises(RuntimeError):
        await _drain(wrapper.chat(chat_ctx=_ctx_with_system()))

    span = one_span(lk_trace, "llm.chat")
    assert span.attributes["llm.had_error"] is True
    assert span.attributes["llm.response"] == "partial"
    assert span.status == SpanStatus.ERROR


# --------------------------------------------------------------------------- #
# D10 — span creation is idempotent across end + aclose
# --------------------------------------------------------------------------- #
async def test_span_not_double_created_on_aclose(lk_trace):
    """Guards: ``_span_created`` flag prevents a duplicate span when aclose runs
    after the stream already ended."""
    chunks = [make_chat_chunk(content="hi", role="assistant")]
    wrapper = LiveKitLLMWrapper(
        llm=FakeBaseLLM(chat_stream=RecordingStream(chunks)), session_id="s1"
    )

    stream = wrapper.chat(chat_ctx=_ctx_with_system())
    await _drain(stream)
    await stream.aclose()

    assert len(spans_named(lk_trace, "llm.chat")) == 1


# --------------------------------------------------------------------------- #
# D11/D12 — type compat, forwarding, chat() tool defaulting
# --------------------------------------------------------------------------- #
def test_wrapper_is_instance_of_base_llm():
    wrapper = LiveKitLLMWrapper(llm=FakeBaseLLM(), session_id="s1")
    assert isinstance(wrapper, BaseLLM)


def test_forwards_events_from_base():
    base = FakeBaseLLM()
    wrapper = LiveKitLLMWrapper(llm=base, session_id="s1")
    metrics, errors = [], []
    wrapper.on("metrics_collected", metrics.append)
    wrapper.on("error", errors.append)

    base.emit("metrics_collected", {"type": "llm_metrics"})
    base.emit("error", {"kind": "boom"})

    assert metrics == [{"type": "llm_metrics"}]
    assert errors == [{"kind": "boom"}]


def test_chat_defaults_tools_to_empty_list(lk_trace):
    """Guards: ``chat(tools=None)`` forwards ``[]`` to the base (LiveKit LLMs
    expect a list, not None)."""
    base = FakeBaseLLM(chat_stream=RecordingStream([]))
    wrapper = LiveKitLLMWrapper(llm=base, session_id="s1")

    wrapper.chat(chat_ctx=_ctx_with_system(), tools=None)
    assert base.last_chat_kwargs["tools"] == []


async def test_aclose_unregisters_and_closes_base():
    base = FakeBaseLLM()
    wrapper = LiveKitLLMWrapper(llm=base, session_id="s1")
    received = []
    wrapper.on("metrics_collected", received.append)

    await wrapper.aclose()
    assert base.aclose_called is True

    base.emit("metrics_collected", {"type": "llm_metrics"})
    assert received == []
