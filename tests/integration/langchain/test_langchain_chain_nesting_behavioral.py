"""Behavioral tests for real parent/child chain nesting in the LangChain handler.

These tests drive *real* ``RunnableLambda`` / ``RunnableSequence`` objects (and a
couple of direct callback invocations) and assert on the **captured** spans:
their names, ``parent_span_id``/``span_id`` linkage, single-trace grouping, and
``SpanStatus`` values.  This is the gap the existing mock-based parent/child and
trace-grouping tests leave open -- those stub ``_get_or_create_trace_context``
and only assert on mock interactions, so they never exercise the real
contextvar-driven nesting that LangChain callback propagation produces.
"""

from __future__ import annotations

import threading
from uuid import uuid4

import pytest

import noveum_trace

from ._helpers import (
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

if LANGCHAIN_AVAILABLE:
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_core.runnables import RunnableLambda

    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler


def test_runnable_sequence_real_parent_child_wiring(client_with_mocked_transport):
    """A RunnableSequence of two RunnableLambdas nests every step under one root.

    All spans land in ONE trace; the root (sequence) span has no parent, and
    each step span's ``parent_span_id`` points at the root span's ``span_id``.
    No orphan spans exist outside the root subtree.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    seq = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
    result = seq.invoke(5, config={"callbacks": [handler]})
    assert result == 12

    noveum_trace.flush()

    traces = get_exported_traces(client)
    assert len(traces) == 1, "all sequence steps must share a single trace"
    spans = traces[0].spans
    # Root sequence span + two RunnableLambda step spans.
    assert len(spans) == 3

    # NOTE: serialized is None for RunnableSequence/RunnableLambda, so the root
    # span is named "chain_start.node" (not a "sequence" name); selection is by
    # parent linkage, not by name.
    roots = [s for s in spans if s.parent_span_id is None]
    assert len(roots) == 1, "exactly one root span expected"
    root = roots[0]
    assert root.name == "chain_start.node"
    assert attrs(root)["name"] == "RunnableSequence"
    assert span_status(root) == "ok"

    children = [s for s in spans if s.parent_span_id is not None]
    assert len(children) == 2
    # Every non-root span must hang off the root -> no orphans, no cross-links.
    for child in children:
        assert child.parent_span_id == root.span_id
        assert attrs(child)["name"] == "RunnableLambda"
        assert span_status(child) == "ok"


def test_runnable_lambda_serialized_none_flat_inputs(client_with_mocked_transport):
    """A bare RunnableLambda yields a 'chain_start.node' span with FLAT inputs.

    Because serialized is None and the input is a raw (non-dict) value, the
    handler stores it under the flat ``chain.inputs`` key (NOT a dotted
    ``chain.inputs.<field>`` key), the output under ``chain.output.outputs``,
    and the status is ``ok``.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    result = RunnableLambda(lambda x: x + 1).invoke(5, config={"callbacks": [handler]})
    assert result == 6

    noveum_trace.flush()

    span = find_span(client, name="chain_start.node")
    a = attrs(span)
    # Flat key, raw int value preserved on the input side.
    assert "chain.inputs" in a
    assert a["chain.inputs"] == 5
    assert not any(
        k.startswith("chain.inputs.") for k in a
    ), "raw non-dict input must use the flat 'chain.inputs' key, not dotted keys"
    # Output is stringified by the handler (str(outputs)).
    assert a["chain.output.outputs"] == "6"
    assert a["chain.operation"] == "execution"
    assert span_status(span) == "ok"


def test_direct_chain_dict_inputs_dotted_keys_real(client_with_mocked_transport):
    """Direct on_chain_start/on_chain_end with dict I/O yields dotted-key attrs.

    A direct callback call with a serialized name and dict inputs produces a
    captured span named ``chain.MyChain`` with dotted ``chain.inputs.<key>`` and
    ``chain.output.<key>`` attributes, ``chain.operation == "execution"``, and
    status ``ok``.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    run_id = uuid4()
    handler.on_chain_start(
        {"name": "MyChain", "id": ["langchain", "schema", "MyChain"]},
        {"question": "q"},
        run_id=run_id,
    )
    handler.on_chain_end({"answer": "a"}, run_id=run_id)

    noveum_trace.flush()

    span = find_span(client, name="chain.MyChain")
    a = attrs(span)
    assert a["chain.name"] == "MyChain"
    assert a["chain.operation"] == "execution"
    # Dotted flattening for real dict inputs/outputs.
    assert a["chain.inputs.question"] == "q"
    assert a["chain.output.answer"] == "a"
    assert span_status(span) == "ok"


def test_chain_error_real_span_status(client_with_mocked_transport):
    """A raising RunnableLambda yields a captured chain span with error status.

    The span's status is exactly ``"error"``, the exception message is recorded
    on the span (both as an ``exception.message`` attribute and an exception
    event), and the trace still finishes and is exported despite the error.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    def boom(_x):
        raise ValueError("kaboom-msg")

    with pytest.raises(ValueError, match="kaboom-msg"):
        RunnableLambda(boom).invoke(1, config={"callbacks": [handler]})

    noveum_trace.flush()

    # The trace must still have been exported even though the chain errored.
    span = find_span(client, name="chain_start.node")
    assert span_status(span) == "error"
    a = attrs(span)
    assert a["exception.type"] == "ValueError"
    assert a["exception.message"] == "kaboom-msg"
    # The error is also recorded as an exception event.
    event_names = [getattr(ev, "name", ev) for ev in getattr(span, "events", [])]
    assert "exception" in event_names


def test_nested_chain_then_llm_single_trace(client_with_mocked_transport):
    """A RunnableLambda invoking an LLM nests the llm span under the chain span.

    Callbacks auto-propagate from the chain into the inner LLM call, so the
    chain span and the llm span share ONE trace and the llm span's
    ``parent_span_id`` points at the chain span's ``span_id``.
    """
    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    llm = FakeListLLM(responses=["hello world"])

    def body(_x):
        return llm.invoke("say hi")

    result = RunnableLambda(body).invoke("go", config={"callbacks": [handler]})
    assert result == "hello world"

    noveum_trace.flush()

    traces = get_exported_traces(client)
    assert len(traces) == 1, "chain and nested llm must share a single trace"
    spans = traces[0].spans

    chain_span = next(s for s in spans if s.name == "chain_start.node")
    llm_span = next(s for s in spans if s.name == "llm.fake")
    assert chain_span.parent_span_id is None
    # Cross-type (chain -> llm) nesting: llm hangs off the chain subtree.
    assert llm_span.parent_span_id == chain_span.span_id
    assert attrs(llm_span)["llm.model"] == "fake"
    assert attrs(llm_span)["llm.provider"] == "fake"
    assert span_status(chain_span) == "ok"
    assert span_status(llm_span) == "ok"


def test_concurrent_runnables_separate_root_traces_real(client_with_mocked_transport):
    """Two independent invokes from two threads produce two DISTINCT root traces.

    Each thread's RunnableLambda invoke is its own root (separate contextvar
    context per thread), so two distinct ``trace_id`` values are produced, each
    with a single self-consistent root span and no cross-trace parent leakage.
    """
    client = client_with_mocked_transport

    def run():
        handler = NoveumTraceCallbackHandler()
        RunnableLambda(lambda x: x + 1).invoke(1, config={"callbacks": [handler]})

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    noveum_trace.flush()

    traces = get_exported_traces(client)
    assert len(traces) == 2
    trace_ids = {t.trace_id for t in traces}
    assert len(trace_ids) == 2, "each independent invoke must be its own trace"

    # Each trace is self-consistent: one root span, no parent leaking elsewhere.
    for trace in traces:
        assert len(trace.spans) == 1
        root = trace.spans[0]
        assert root.parent_span_id is None
        assert root.name == "chain_start.node"
        assert span_status(root) == "ok"

    # No span anywhere points at a span_id outside its own trace.
    all_span_ids = {s.span_id for s in get_exported_spans(client)}
    for trace in traces:
        own_ids = {s.span_id for s in trace.spans}
        for s in trace.spans:
            if s.parent_span_id is not None:
                assert s.parent_span_id in own_ids
        assert own_ids <= all_span_ids
