"""Behavioral tests for real captured LLM-span content in the LangChain handler.

These tests drive *real* ``langchain_core`` fakes (``FakeListLLM``,
``FakeListChatModel``, ``GenericFakeChatModel``) through ``.invoke`` with a
``NoveumTraceCallbackHandler`` built *inside* each test under the repo-wide
``client_with_mocked_transport`` fixture, then ``flush()`` and inspect the real
``Span.attributes`` / ``Span.status`` that landed on
``client.transport.export_trace``.  The two paths that no real fake can reach --
a realistic provider model name and the ``finish_reason``/``error`` wiring -- are
driven with direct ``on_*`` callbacks.

The existing LangChain suites patch ``get_client`` with a bare ``Mock`` and assert
on mock interactions, so they never see the *actual* exported attribute SHAPE
(list prompts vs JSON-string messages), the estimation-fallback trap, the
provider-usage passthrough, or the exact ``SpanStatus`` values.  That is the gap
these tests close.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

import noveum_trace

from ._helpers import (
    LANGCHAIN_AVAILABLE,
    attrs,
    find_span,
    span_status,
)

pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="LangChain not available"
)

if LANGCHAIN_AVAILABLE:
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_core.language_models.fake_chat_models import (
        FakeListChatModel,
        GenericFakeChatModel,
    )
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler


def test_completion_llm_span_real_attributes_and_estimation(
    client_with_mocked_transport,
):
    """The completion path exports a list of prompts and estimated token usage.

    ``FakeListLLM(responses=["hi"]).invoke("q")`` produces exactly one exported
    span named ``"llm.fake"`` with ``llm.operation == "completion"``,
    ``llm.input.prompts`` as a real LIST (``["q"]``), ``prompt_count == 1`` and --
    the TRAP -- positive estimated input/output/total tokens plus
    ``llm.cost.currency == "USD"`` even though the fake reports no provider usage.
    Status is ``"ok"``.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    result = FakeListLLM(responses=["hi"]).invoke("q", config={"callbacks": [handler]})
    assert result == "hi"

    noveum_trace.flush()

    span = find_span(client, name="llm.fake")
    a = attrs(span)
    assert a["llm.operation"] == "completion"
    # SHAPE: completion prompts are a real list, not a JSON string.
    assert isinstance(a["llm.input.prompts"], list)
    assert a["llm.input.prompts"] == ["q"]
    assert a["llm.input.prompt_count"] == 1

    # Estimation-fallback trap: the fake carries no usage, yet positive token
    # counts + USD cost are still produced. Values are tokenizer-dependent, so
    # assert positivity and the additive total invariant, not exact numbers.
    assert a["llm.input_tokens"] > 0
    assert a["llm.output_tokens"] > 0
    assert a["llm.total_tokens"] == a["llm.input_tokens"] + a["llm.output_tokens"]
    assert a["llm.cost.currency"] == "USD"
    assert a["llm.cost.total"] > 0

    assert span_status(span) == "ok"


def test_chat_model_span_messages_json_and_flags(client_with_mocked_transport):
    """The chat path exports messages as a JSON STRING and flags the system prompt.

    A ``SystemMessage`` + ``HumanMessage`` run yields ``llm.operation == "chat"``,
    ``llm.input.type == "chat"``, ``llm.input.messages`` a JSON STRING (parsing to
    a list), ``message_count == 2`` and ``has_system_prompt is True``.  A
    HumanMessage-only run gives ``has_system_prompt is False`` and
    ``message_count == 1``.  This pins the chat-vs-completion divergence (JSON
    messages vs list prompts) that the mock suite is blind to.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    FakeListChatModel(responses=["r"]).invoke(
        [SystemMessage("s"), HumanMessage("h")],
        config={"callbacks": [handler]},
    )
    FakeListChatModel(responses=["r"]).invoke(
        [HumanMessage("h")],
        config={"callbacks": [handler]},
    )

    noveum_trace.flush()

    # Both spans share the name llm.fake_chat_models; disambiguate by content.
    sys_span = find_span(
        client,
        name="llm.fake_chat_models",
        predicate=lambda s: attrs(s).get("llm.input.message_count") == 2,
    )
    a = attrs(sys_span)
    assert a["llm.operation"] == "chat"
    assert a["llm.input.type"] == "chat"
    # SHAPE: chat messages are a JSON string, not a list.
    assert isinstance(a["llm.input.messages"], str)
    parsed = json.loads(a["llm.input.messages"])
    assert isinstance(parsed, list)
    assert a["llm.input.message_count"] == 2
    assert a["llm.input.has_system_prompt"] is True
    assert span_status(sys_span) == "ok"

    human_span = find_span(
        client,
        name="llm.fake_chat_models",
        predicate=lambda s: attrs(s).get("llm.input.message_count") == 1,
    )
    h = attrs(human_span)
    assert h["llm.input.has_system_prompt"] is False
    assert h["llm.input.message_count"] == 1


def test_chat_model_token_cost_finish_reason_via_generic_fake(
    client_with_mocked_transport,
):
    """Canonical top-level usage_metadata + response_metadata finish_reason flow through.

    Regression test for two fixes:
    - Provider usage on the canonical top-level ``AIMessage.usage_metadata`` field
      is read (previously only ``response_metadata['usage_metadata']`` /
      ``generation_info`` were consulted), so the span shows the EXACT counts
      (10/5/15) rather than estimated values.
    - ``finish_reason`` is read from the generation message's ``response_metadata``
      (previously only ``response.llm_output`` was consulted, which the fake leaves
      ``{}``), so it resolves to ``"stop"``.

    See [[project-langchain-known-bugs]].
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    msg = AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
        response_metadata={"finish_reason": "stop"},
    )
    GenericFakeChatModel(messages=iter([msg])).invoke(
        [HumanMessage("hi")],
        config={"callbacks": [handler]},
    )

    noveum_trace.flush()

    span = find_span(client, name="llm.fake_chat_models")
    a = attrs(span)
    # Exact provider passthrough from top-level usage_metadata -- not estimation.
    assert a["llm.input_tokens"] == 10
    assert a["llm.output_tokens"] == 5
    assert a["llm.total_tokens"] == 15
    assert a["llm.cost.total"] > 0
    assert a["llm.cost.currency"] == "USD"
    # finish_reason sourced from the message's response_metadata.
    assert a["llm.output.finish_reason"] == "stop"


def test_chat_span_has_tool_calls_flag_true(client_with_mocked_transport):
    """``has_tool_calls`` reflects tool calls in the first input message batch.

    A run whose input batch contains an ``AIMessage(tool_calls=[...])`` sets
    ``llm.input.has_tool_calls is True`` on the captured span; a no-tool-call run
    leaves it ``False``.  This is a distinct boolean branch from
    ``has_system_prompt``.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}],
    )
    FakeListChatModel(responses=["r"]).invoke(
        [HumanMessage("hi"), tool_call_msg],
        config={"callbacks": [handler]},
    )
    FakeListChatModel(responses=["r"]).invoke(
        [HumanMessage("hi")],
        config={"callbacks": [handler]},
    )

    noveum_trace.flush()

    with_tools = find_span(
        client,
        name="llm.fake_chat_models",
        predicate=lambda s: attrs(s).get("llm.input.has_tool_calls") is True,
    )
    a = attrs(with_tools)
    assert a["llm.input.has_tool_calls"] is True
    assert a["llm.input.message_count"] == 2
    assert a["llm.operation"] == "chat"

    without_tools = find_span(
        client,
        name="llm.fake_chat_models",
        predicate=lambda s: attrs(s).get("llm.input.has_tool_calls") is False,
    )
    assert attrs(without_tools)["llm.input.has_tool_calls"] is False


def test_llm_error_real_span_status_and_message(client_with_mocked_transport):
    """``on_llm_error`` sets ``status == "error"`` and records the message.

    A direct ``on_chat_model_start`` (serialized ``kwargs.model`` ==
    ``gpt-4o-mini``) followed by ``on_llm_error(ValueError("boom"))`` yields a
    captured span with ``status.value == "error"``, ``llm.model ==
    "gpt-4o-mini"`` and the exact failure message ``"boom"`` on
    ``span.status_message``.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    run_id = uuid4()
    serialized = {
        "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
        "kwargs": {"model": "gpt-4o-mini"},
        "name": "ChatOpenAI",
    }
    handler.on_chat_model_start(serialized, [[HumanMessage("hi")]], run_id=run_id)
    handler.on_llm_error(ValueError("boom"), run_id=run_id)

    noveum_trace.flush()

    span = find_span(client, name="llm.gpt-4o-mini")
    assert span_status(span) == "error"
    assert attrs(span)["llm.model"] == "gpt-4o-mini"
    # The error message lives on status_message, not on the status enum value.
    assert span.status_message is not None
    assert "boom" in span.status_message


def test_realistic_model_name_and_provider_resolution(client_with_mocked_transport):
    """Span NAME uses the raw model string; ``llm.model`` is normalized.

    A direct ``on_chat_model_start`` with serialized id
    ``[..., "openai", "ChatOpenAI"]`` and ``kwargs.model == "gpt-4o-mini"``
    produces span name ``"llm.gpt-4o-mini"``, ``llm.model == "gpt-4o-mini"`` and
    ``llm.provider == "openai"``; closing via ``on_llm_end`` with
    ``llm_output={"finish_reason": "stop"}`` captures ``finish_reason == "stop"``.

    A ``models/``-prefixed Gemini model shows the NAME/attribute divergence: the
    span name keeps the raw ``"llm.models/gemini-2.0-flash"`` while ``llm.model``
    is normalized to ``"gemini-2.0-flash"`` and the provider resolves to
    ``"google"``.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    # OpenAI: clean model name drives both name and attribute.
    openai_run = uuid4()
    handler.on_chat_model_start(
        {
            "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
            "kwargs": {"model": "gpt-4o-mini"},
            "name": "ChatOpenAI",
        },
        [[HumanMessage("hi")]],
        run_id=openai_run,
    )
    openai_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="ok"))]],
        llm_output={"finish_reason": "stop"},
    )
    handler.on_llm_end(openai_result, run_id=openai_run)

    # Gemini: models/ prefix is stripped+normalized for the attribute only.
    gemini_run = uuid4()
    handler.on_chat_model_start(
        {
            "id": [
                "langchain",
                "chat_models",
                "google_genai",
                "ChatGoogleGenerativeAI",
            ],
            "kwargs": {"model": "models/gemini-2.0-flash"},
            "name": "ChatGoogleGenerativeAI",
        },
        [[HumanMessage("hi")]],
        run_id=gemini_run,
    )
    gemini_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="ok"))]],
        llm_output={},
    )
    handler.on_llm_end(gemini_result, run_id=gemini_run)

    noveum_trace.flush()

    openai_span = find_span(client, name="llm.gpt-4o-mini")
    oa = attrs(openai_span)
    assert oa["llm.model"] == "gpt-4o-mini"
    assert oa["llm.provider"] == "openai"
    # finish_reason DOES flow through when present on llm_output.
    assert oa["llm.output.finish_reason"] == "stop"
    assert span_status(openai_span) == "ok"

    # Name keeps the raw prefixed string ...
    gemini_span = find_span(client, name="llm.models/gemini-2.0-flash")
    ga = attrs(gemini_span)
    # ... while the attribute is normalized and the provider resolves.
    assert ga["llm.model"] == "gemini-2.0-flash"
    assert ga["llm.provider"] == "google"


def test_custom_name_overrides_span_name_real(client_with_mocked_transport):
    """``metadata.noveum.name`` renames the span; extra fields are serialized.

    A ``FakeListChatModel`` invoked with config ``metadata.noveum`` carrying a
    custom name ``"my_span"`` plus an extra field yields a span named
    ``"my_span"`` (not ``llm.fake_chat_models``), and a
    ``noveum.additional_attributes`` JSON attribute that parses to a dict
    containing the extra field.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    FakeListChatModel(responses=["r"]).invoke(
        [HumanMessage("h")],
        config={
            "callbacks": [handler],
            "metadata": {"noveum": {"name": "my_span", "extra": "v"}},
        },
    )

    noveum_trace.flush()

    span = find_span(client, name="my_span")
    a = attrs(span)
    assert span.name == "my_span"
    # The default operation-derived name must NOT have been used.
    assert span.name != "llm.fake_chat_models"

    raw = a["noveum.additional_attributes"]
    assert isinstance(raw, str)
    extra = json.loads(raw)
    assert extra.get("extra") == "v"
    # The custom name itself is consumed as the span name, not duplicated here.
    assert "name" not in extra
    assert span_status(span) == "ok"
