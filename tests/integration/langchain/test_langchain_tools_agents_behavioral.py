"""Behavioral tests for the LangChain integration's tool/agent/retriever paths.

These exercise the held-open-LLM-span tool-call correlation machinery, the
fallback-LLM error-append path, real ``@tool`` invocation, agent action/finish
attribute attachment, real ``BaseRetriever`` lifecycle, and ``langgraph``
routing custom events -- asserting on the *real* captured spans (or, for spans
that are intentionally held open and never exported, on the live handler state).

Unlike the older mock-interaction suites, the ID-correlation and
``llm.executed_tool_calls`` machinery is *not* patched out here, so these tests
pin its actual (and in one case buggy) behavior.
"""

from __future__ import annotations

import uuid

import pytest

import noveum_trace

from ._helpers import (  # noqa: F401
    LANGCHAIN_AVAILABLE,
    attrs,
    find_span,
    get_exported_spans,
    get_exported_traces,
    span_status,
)

pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="LangChain not available"
)

# Guard the optional-dependency import so the module imports cleanly (and the
# skip marker above can take effect) on installs without LangChain.
if LANGCHAIN_AVAILABLE:
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler


def _start_keepalive_chain(handler):
    """Start a non-LLM chain span and return its run_id.

    A tool-calling chat model is always nested under an executor in reality.
    Keeping a non-LLM span open prevents ``_finish_trace_if_needed`` from
    auto-finishing the trace the moment the held-open LLM span becomes the only
    remaining (stuck) run, which would otherwise wipe the tool-count tracking.
    """
    chain_rid = uuid.uuid4()
    handler.on_chain_start({"name": "AgentExecutor"}, {"input": "go"}, run_id=chain_rid)
    return chain_rid


def test_llm_tool_call_correlation_finishes_held_span(client_with_mocked_transport):
    """Held-open LLM span + tool_call_id correlation, driven end to end.

    Regression test for the run_id key-type fix. ``on_llm_end`` records the
    tool_call_id -> LLM mapping keyed by the LLM's UUID run_id (previously it was
    stored as ``str(run_id)``, which never matched the UUID-keyed ``runs`` /
    ``_llm_tool_counts`` dicts). With the fix, each ``on_tool_end`` whose
    ``ToolMessage`` carries a matching tool_call_id appends to the LLM span's
    ``llm.executed_tool_calls`` and increments the completed count; the second
    completion finishes (and pops) the held-open span. See [[project-langchain-known-bugs]].
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    chain_rid = _start_keepalive_chain(handler)
    llm_rid = uuid.uuid4()
    handler.on_chat_model_start(
        {"name": "ChatModel", "kwargs": {"model": "gpt-4o"}},
        [[]],
        run_id=llm_rid,
        parent_run_id=chain_rid,
    )

    # The span itself is stored under the UUID run_id (the string still misses).
    assert handler._get_run(llm_rid) is not None
    assert handler._get_run(str(llm_rid)) is None

    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "search", "args": {"q": "x"}, "id": "call_1", "type": "tool_call"},
            {"name": "lookup", "args": {"q": "y"}, "id": "call_2", "type": "tool_call"},
        ],
    )
    response = LLMResult(generations=[[ChatGeneration(message=ai)]])
    handler.on_llm_end(response, run_id=llm_rid, parent_run_id=chain_rid)

    # Span is held open (not exported yet), expected count == 2, completed == 0.
    assert handler._get_run(llm_rid) is not None
    assert handler._llm_tool_counts.get(llm_rid) == {"expected": 2, "completed": 0}
    # The id->llm mapping is recorded under the UUID run_id (so lookups match).
    assert handler._tool_call_id_to_llm.get("call_1") == llm_rid
    assert handler._tool_call_id_to_llm.get("call_2") == llm_rid

    # First tool completes -> appended + completed == 1, span still held open.
    t1 = uuid.uuid4()
    handler.on_tool_start({"name": "search"}, "x", run_id=t1, parent_run_id=llm_rid)
    handler.on_tool_end(
        ToolMessage(content="r1", tool_call_id="call_1"),
        run_id=t1,
        parent_run_id=llm_rid,
    )
    held = handler._get_run(llm_rid)
    assert held is not None
    assert handler._llm_tool_counts.get(llm_rid) == {"expected": 2, "completed": 1}
    executed_so_far = held.attributes.get("llm.executed_tool_calls")
    assert isinstance(executed_so_far, list) and len(executed_so_far) == 1

    # Second tool completes -> completed == 2 -> span finished and popped.
    t2 = uuid.uuid4()
    handler.on_tool_start({"name": "lookup"}, "y", run_id=t2, parent_run_id=llm_rid)
    handler.on_tool_end(
        ToolMessage(content="r2", tool_call_id="call_2"),
        run_id=t2,
        parent_run_id=llm_rid,
    )
    assert handler._get_run(llm_rid) is None  # finished + popped from runs
    assert llm_rid not in handler._llm_tool_counts
    # The span finished but the trace only exports once the chain ends.
    assert client.transport.export_trace.call_count == 0

    # Finishing the parent chain exports the trace; the LLM span carries both tools.
    handler.on_chain_end({"output": "done"}, run_id=chain_rid)
    noveum_trace.flush()

    llm_span = find_span(
        client, predicate=lambda s: attrs(s).get("llm.model") == "gpt-4o"
    )
    assert llm_span.name == "llm.gpt-4o"
    assert span_status(llm_span) == "ok"
    executed = attrs(llm_span).get("llm.executed_tool_calls")
    assert isinstance(executed, list) and len(executed) == 2
    assert {e["name"] for e in executed} == {"search", "lookup"}
    assert all(e["status"] == "ok" for e in executed)


def test_tool_error_appends_error_entry_to_fallback_llm(client_with_mocked_transport):
    """on_tool_error appends an error entry to the fallback LLM span.

    With ``parent_run_id`` == the LLM run (and no tool_call_id), ``on_tool_start``
    identifies the LLM as the fallback target; ``on_tool_error`` appends an entry
    to that span's ``llm.executed_tool_calls`` with status 'error' and an error
    type/message. The fallback path uses the UUID run_id directly, so unlike the
    tool_call_id path it actually attaches. The span is never exported here (no
    on_llm_end / chain end), so assert on the live held span.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    chain_rid = _start_keepalive_chain(handler)
    llm_rid = uuid.uuid4()
    handler.on_chat_model_start(
        {"name": "ChatModel", "kwargs": {"model": "gpt-4o"}},
        [[]],
        run_id=llm_rid,
        parent_run_id=chain_rid,
    )

    tool_rid = uuid.uuid4()
    handler.on_tool_start(
        {"name": "search"}, "x", run_id=tool_rid, parent_run_id=llm_rid
    )
    # The fallback LLM was identified during on_tool_start.
    pending = handler._pending_tool_calls.get(tool_rid)
    assert pending is not None
    assert pending.get("fallback_llm_run_id") == llm_rid

    handler.on_tool_error(RuntimeError("boom"), run_id=tool_rid, parent_run_id=llm_rid)

    llm_span = handler._get_run(llm_rid)
    assert llm_span is not None
    executed = llm_span.attributes.get("llm.executed_tool_calls")
    assert isinstance(executed, list)
    assert len(executed) == 1
    entry = executed[0]
    assert entry["name"] == "search"
    assert entry["status"] == "error"
    assert entry["error"] == {"type": "RuntimeError", "message": "boom"}

    # No count increment on the error-fallback path: the held span is not exported.
    assert client.transport.export_trace.call_count == 0


def test_real_tool_invoke_stores_pending_no_span(client_with_mocked_transport):
    """A real @tool .invoke() under the handler creates NO standalone span.

    Tools never create their own spans, and a raw ``@tool`` output (an int) carries
    no ToolMessage / tool_call_id, so the ID-correlation path is unreachable via a
    real ``@tool``. Pins that no tool span is ever exported.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    result = add.invoke({"a": 2, "b": 3}, config={"callbacks": [handler]})

    # @tool output is a raw int (no ToolMessage wrapper -> no id correlation).
    assert result == 5
    assert isinstance(result, int)

    noveum_trace.flush()

    spans = get_exported_spans(client)
    assert all(not s.name.startswith("tool") for s in spans)
    # No standalone tool span is ever produced for a bare @tool invocation.
    assert client.transport.export_trace.call_count == 0
    assert spans == []


def test_agent_action_and_finish_attach_to_llm_span_real(client_with_mocked_transport):
    """on_agent_action / on_agent_finish attach attributes to the active LLM span.

    No new span is created: both events write onto the SAME captured LLM span via
    its run_id, and on_agent_finish finishes it with status ok.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    from langchain_core.agents import AgentAction, AgentFinish

    chain_rid = _start_keepalive_chain(handler)
    llm_rid = uuid.uuid4()
    handler.on_chat_model_start(
        {"name": "ChatModel", "kwargs": {"model": "gpt-4o"}},
        [[]],
        run_id=llm_rid,
        parent_run_id=chain_rid,
    )
    llm_span_id = handler._get_run(llm_rid).span_id

    action = AgentAction(tool="search", tool_input={"q": "x"}, log="I should search")
    handler.on_agent_action(action, run_id=llm_rid)

    # Same span, no new one created.
    live = handler._get_run(llm_rid)
    assert live is not None and live.span_id == llm_span_id
    assert live.attributes.get("agent.output.action.tool") == "search"
    # tool_input is stored as str(action.tool_input).
    assert live.attributes.get("agent.output.action.tool_input") == "{'q': 'x'}"
    assert live.attributes.get("agent.output.action.log") == "I should search"

    finish = AgentFinish(return_values={"output": "done"}, log="Final Answer: done")
    handler.on_agent_finish(finish, run_id=llm_rid)
    # The span was popped (finished) by on_agent_finish.
    assert handler._get_run(llm_rid) is None

    handler.on_chain_end({"output": "done"}, run_id=chain_rid)
    noveum_trace.flush()

    llm_span = find_span(client, predicate=lambda s: s.span_id == llm_span_id)
    # Only the chain span and the one LLM span were exported -- no agent span.
    assert {s.name for s in get_exported_spans(client)} == {
        "chain.AgentExecutor",
        "llm.gpt-4o",
    }
    assert span_status(llm_span) == "ok"
    assert attrs(llm_span).get("agent.output.action.tool") == "search"
    assert attrs(llm_span).get("agent.output.finish.return_values") == {
        "output": "done"
    }
    assert attrs(llm_span).get("agent.output.finish.log") == "Final Answer: done"


def test_retriever_lifecycle_real_span_and_truncation(client_with_mocked_transport):
    """A real BaseRetriever produces a captured retrieval span with result/truncation.

    NOTE: documents current behavior. The langchain_core BaseRetriever passes a
    ``serialized`` that lacks a usable name, so ``get_operation_name`` falls back to
    ``"retriever_start.node"`` rather than the spec's hypothesized ``retrieval.<name>``.
    Asserted for N=2 (not truncated) and N=12 (truncated, sample capped at 10).
    """
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class CountingRetriever(BaseRetriever):
        n: int = 2

        def _get_relevant_documents(self, query, *, run_manager):
            return [Document(page_content=f"doc{i}") for i in range(self.n)]

    for n in (2, 12):
        client = client_with_mocked_transport
        client.transport.export_trace.reset_mock()
        handler = NoveumTraceCallbackHandler()

        CountingRetriever(n=n).invoke("find things", config={"callbacks": [handler]})
        noveum_trace.flush()

        span = find_span(
            client, predicate=lambda s: attrs(s).get("retrieval.type") == "search"
        )
        # Real BaseRetriever has no serialized name -> fallback span name.
        assert span.name == "retriever_start.node"
        assert span_status(span) == "ok"
        assert attrs(span).get("retrieval.query") == "find things"
        assert attrs(span).get("retrieval.result_count") == n
        sample = attrs(span).get("retrieval.sample_results")
        assert isinstance(sample, list)
        assert len(sample) <= 10
        assert attrs(span).get("retrieval.results_truncated") == (n > 10)


def test_retriever_error_real_span_status(client_with_mocked_transport):
    """A BaseRetriever whose _get_relevant_documents raises yields an error span.

    The captured retrieval span has status 'error', records the exception, and the
    status message carries the raised message.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    from langchain_core.retrievers import BaseRetriever

    class FailingRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            raise ValueError("retriever failed")

    with pytest.raises(ValueError, match="retriever failed"):
        FailingRetriever().invoke("q", config={"callbacks": [handler]})

    noveum_trace.flush()

    span = find_span(
        client, predicate=lambda s: attrs(s).get("retrieval.type") == "search"
    )
    assert span_status(span) == "error"
    assert span.status_message == "retriever failed"
    # An exception event was recorded on the span.
    event_names = [getattr(e, "name", e) for e in (span.events or [])]
    assert "exception" in event_names


def test_custom_event_routing_decision_real_span(client_with_mocked_transport):
    """dispatch_custom_event('langgraph.routing_decision', ...) yields a routing span.

    The routing span is finished immediately, named ``routing.<source>_to_<target>``,
    carries the routing attributes (with float confidence), and is parented under the
    dispatching lambda's span.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    from langchain_core.callbacks.manager import dispatch_custom_event
    from langchain_core.runnables import RunnableLambda

    def route(x):
        dispatch_custom_event(
            "langgraph.routing_decision",
            {"source_node": "a", "target_node": "b", "confidence": 0.9},
        )
        return x

    RunnableLambda(route).invoke(1, config={"callbacks": [handler]})
    noveum_trace.flush()

    routing = find_span(client, name="routing.a_to_b")
    assert span_status(routing) == "ok"
    assert attrs(routing).get("routing.type") == "conditional_edge"
    assert attrs(routing).get("routing.source_node") == "a"
    assert attrs(routing).get("routing.target_node") == "b"
    confidence = attrs(routing).get("routing.confidence")
    assert isinstance(confidence, float)
    assert confidence == 0.9

    # Parented under the lambda's chain span.
    chain = find_span(client, name="chain_start.node")
    assert routing.parent_span_id == chain.span_id


def test_custom_event_non_routing_name_noop_real(client_with_mocked_transport):
    """A non-routing custom event name creates no routing span.

    Only the dispatching lambda's chain span is captured -- the name gate prevents
    spurious spans for unrelated custom events.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    from langchain_core.callbacks.manager import dispatch_custom_event
    from langchain_core.runnables import RunnableLambda

    def emit(x):
        dispatch_custom_event("some.other.event", {"foo": "bar"})
        return x

    RunnableLambda(emit).invoke(1, config={"callbacks": [handler]})
    noveum_trace.flush()

    names = [s.name for s in get_exported_spans(client)]
    assert names == ["chain_start.node"]
    assert not any(n.startswith("routing.") for n in names)
