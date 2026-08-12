"""
Unit tests for the LlamaIndex integration handlers.

The handlers subclass LlamaIndex's real (Pydantic) instrumentation base classes,
so these tests require the optional ``llamaindex`` extra and are skipped when it
is absent. The Noveum client/trace/span are faked, and the handler lifecycle is
driven directly (``new_span`` / ``handle`` / ``prepare_to_exit_span``).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "llama_index.core.instrumentation",
    reason="requires optional 'llamaindex' extra",
)

_src = Path(__file__).parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from noveum_trace.integrations.llamaindex import utils  # noqa: E402
from noveum_trace.integrations.llamaindex.handler import (  # noqa: E402
    NoveumLlamaIndexEventHandler,
    NoveumLlamaIndexSpanHandler,
    setup_llamaindex_tracing,
)

_BOUND_ARGS = inspect.signature(lambda: None).bind()
_RID = "RetrieverQueryEngine.query-11111111-1111-1111-1111-111111111111"
_CID = "OpenAI.chat-22222222-2222-2222-2222-222222222222"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_noveum_span(span_id: str, trace_id: str = "nt") -> MagicMock:
    span = MagicMock()
    span.span_id = span_id
    span.trace_id = trace_id
    span.attributes = {}
    span.set_attributes = MagicMock(
        side_effect=lambda attrs: span.attributes.update(attrs)
    )
    span.set_status = MagicMock()
    span.finish = MagicMock()
    span.is_finished = MagicMock(return_value=False)
    return span


def _make_client(trace_id: str = "nt") -> MagicMock:
    client = MagicMock()
    trace = MagicMock()
    trace.trace_id = trace_id
    counter = {"n": 0}

    def _create_span(name, parent_span_id=None, attributes=None, start_time=None):
        counter["n"] += 1
        span = _make_noveum_span(f"ns{counter['n']}", trace_id=trace_id)
        span.name = name
        span.parent_span_id = parent_span_id
        if attributes:
            span.attributes.update(attributes)
        return span

    trace.create_span = MagicMock(side_effect=_create_span)
    trace.finish = MagicMock()
    client._trace = trace
    client.start_trace = MagicMock(return_value=trace)
    client.finish_trace = MagicMock()
    client.flush = MagicMock()
    return client


def _event(class_name: str, **fields):
    obj = type(class_name, (), {})()
    for key, value in fields.items():
        setattr(obj, key, value)
    return obj


def _message(role: str, content: str):
    return type("ChatMessage", (), {"role": role, "content": content})()


def _open(sh: NoveumLlamaIndexSpanHandler, id_: str, parent: str | None = None):
    span = sh.new_span(id_=id_, bound_args=_BOUND_ARGS, parent_span_id=parent)
    if span is not None:
        sh.open_spans[id_] = span
    return span


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


class TestUtils:
    def test_operation_from_span_id_strips_uuid(self) -> None:
        assert utils.operation_from_span_id(_RID) == "RetrieverQueryEngine.query"

    def test_operation_from_span_id_no_uuid(self) -> None:
        assert utils.operation_from_span_id("plain-name") == "plain-name"

    def test_classify_operation(self) -> None:
        assert utils.classify_operation("VectorIndexRetriever.retrieve") == "retrieval"
        assert utils.classify_operation("OpenAI.chat") == "llm"
        assert utils.classify_operation("Mystery.thing") == "other"

    def test_token_usage_from_raw_usage(self) -> None:
        response = type("R", (), {})()
        response.raw = {"usage": {"prompt_tokens": 3, "completion_tokens": 4}}
        response.additional_kwargs = {}
        tokens = utils.extract_token_usage(response)
        assert tokens["input_tokens"] == 3
        assert tokens["output_tokens"] == 4
        assert tokens["total_tokens"] == 7


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_handlers_construct_with_config(self) -> None:
        client = _make_client()
        sh = NoveumLlamaIndexSpanHandler(client=client, trace_name_prefix="li")
        eh = NoveumLlamaIndexEventHandler(span_handler=sh, capture_inputs=True)
        assert sh.client is client
        assert sh.trace_name_prefix == "li"
        assert eh.span_handler is sh
        assert eh.capture_inputs is True

    def test_captures_everything_by_default(self) -> None:
        eh = NoveumLlamaIndexEventHandler()
        assert eh.capture_inputs is True
        assert eh.capture_outputs is True
        assert eh.capture_llm_messages is True
        assert eh.capture_cost is True

    def test_embedding_chunk_capture_is_opt_in(self) -> None:
        # Indexing a corpus emits one embedding call per batch, so the chunk
        # text stays behind an explicit opt-in even though everything else is
        # captured by default.
        eh = NoveumLlamaIndexEventHandler()
        assert eh.capture_embedding_chunks is False

    def test_capture_flags_can_be_disabled(self) -> None:
        eh = NoveumLlamaIndexEventHandler(
            capture_inputs=False,
            capture_outputs=False,
            capture_llm_messages=False,
            capture_cost=False,
        )
        assert eh.capture_inputs is False
        assert eh.capture_outputs is False
        assert eh.capture_llm_messages is False
        assert eh.capture_cost is False


# ---------------------------------------------------------------------------
# Span mapping
# ---------------------------------------------------------------------------


class TestSpanMapping:
    def test_root_span_opens_trace(self) -> None:
        client = _make_client()
        sh = NoveumLlamaIndexSpanHandler(client=client)
        root = _open(sh, _RID)
        assert root is not None
        assert root.is_root is True
        client.start_trace.assert_called_once()
        assert (
            root.noveum_span.attributes["llamaindex.operation"]
            == "RetrieverQueryEngine.query"
        )
        assert root.noveum_span.attributes["llamaindex.span_type"] == "query"

    def test_child_span_linked_to_parent(self) -> None:
        client = _make_client()
        sh = NoveumLlamaIndexSpanHandler(client=client)
        root = _open(sh, _RID)
        child = _open(sh, _CID, parent=_RID)
        assert child is not None
        assert child.is_root is False
        assert child.noveum_trace is root.noveum_trace
        child_call = client._trace.create_span.call_args_list[-1]
        assert child_call.kwargs["parent_span_id"] == root.noveum_span.span_id

    def test_missing_parent_becomes_new_root(self) -> None:
        client = _make_client()
        sh = NoveumLlamaIndexSpanHandler(client=client)
        orphan = _open(sh, _CID, parent="never-opened")
        assert orphan is not None
        assert orphan.is_root is True

    def test_no_client_returns_none(self) -> None:
        sh = NoveumLlamaIndexSpanHandler()  # no client, global not initialised
        assert (
            sh.new_span(id_=_RID, bound_args=_BOUND_ARGS, parent_span_id=None) is None
        )


# ---------------------------------------------------------------------------
# Event enrichment
# ---------------------------------------------------------------------------


class TestEventEnrichment:
    def _setup(self, **eh_kwargs):
        client = _make_client()
        sh = NoveumLlamaIndexSpanHandler(client=client)
        eh = NoveumLlamaIndexEventHandler(span_handler=sh, **eh_kwargs)
        _open(sh, _RID)
        child = _open(sh, _CID, parent=_RID)
        return sh, eh, child

    def test_llm_events_map_model_and_tokens(self) -> None:
        sh, eh, child = self._setup(capture_outputs=False)
        eh.handle(
            _event("LLMChatStartEvent", span_id=_CID, model_dict={"model": "gpt-4o"})
        )
        response = type("R", (), {})()
        response.raw = {"usage": {"prompt_tokens": 12, "completion_tokens": 7}}
        response.message = _message("assistant", "hello")
        eh.handle(_event("LLMChatEndEvent", span_id=_CID, response=response))

        attrs = child.noveum_span.attributes
        assert attrs["llamaindex.span_type"] == "llm"
        assert attrs["llm.model"] == "gpt-4o"
        assert attrs["llm.provider"] == "openai"
        assert attrs["llm.input_tokens"] == 12
        assert attrs["llm.output_tokens"] == 7
        assert attrs["llm.total_tokens"] == 19
        assert "llm.output" not in attrs  # capture_outputs is off by default

    def test_capture_outputs_and_messages(self) -> None:
        sh, eh, child = self._setup(capture_outputs=True, capture_llm_messages=True)
        eh.handle(
            _event(
                "LLMChatStartEvent",
                span_id=_CID,
                model_dict={"model": "gpt-4o"},
                messages=[_message("user", "hi")],
            )
        )
        response = type("R", (), {})()
        response.message = _message("assistant", "hello there")
        eh.handle(_event("LLMChatEndEvent", span_id=_CID, response=response))

        attrs = child.noveum_span.attributes
        assert attrs["llm.input"] == [{"role": "user", "content": "hi"}]
        assert attrs["llm.output"] == "hello there"

    def test_retrieval_end_records_nodes(self) -> None:
        sh, eh, child = self._setup()
        node = type("NodeWithScore", (), {"score": 0.9})()
        eh.handle(_event("RetrievalEndEvent", span_id=_CID, nodes=[node, node]))
        attrs = child.noveum_span.attributes
        assert attrs["retrieval.node_count"] == 2
        assert attrs["retrieval.scores"] == [0.9, 0.9]

    def test_embedding_end_records_counts(self) -> None:
        sh, eh, child = self._setup()
        eh.handle(
            _event(
                "EmbeddingEndEvent",
                span_id=_CID,
                chunks=["a", "b", "c"],
                embeddings=[[0.1], [0.2], [0.3]],
            )
        )
        attrs = child.noveum_span.attributes
        assert attrs["embedding.chunk_count"] == 3
        assert attrs["embedding.vector_count"] == 3

    def test_event_for_unknown_span_is_noop(self) -> None:
        sh, eh, _child = self._setup()
        eh.handle(_event("LLMChatStartEvent", span_id="not-open"))  # must not raise

    def test_exception_event_marks_error(self) -> None:
        sh, eh, child = self._setup()
        eh.handle(_event("ExceptionEvent", span_id=_CID, exception=ValueError("boom")))
        attrs = child.noveum_span.attributes
        assert attrs["llamaindex.status"] == "error"
        assert attrs["error.type"] == "ValueError"
        child.noveum_span.set_status.assert_called_once()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_exit_finishes_span_and_trace(self) -> None:
        client = _make_client()
        sh = NoveumLlamaIndexSpanHandler(client=client)
        root = _open(sh, _RID)
        child = _open(sh, _CID, parent=_RID)

        sh.prepare_to_exit_span(id_=_CID, bound_args=_BOUND_ARGS)
        assert child.noveum_span.finish.called
        assert not client.finish_trace.called  # child is not root

        sh.prepare_to_exit_span(id_=_RID, bound_args=_BOUND_ARGS)
        assert root.noveum_span.finish.called
        client.finish_trace.assert_called_once_with(root.noveum_trace)

    def test_drop_marks_error_and_finishes(self) -> None:
        client = _make_client()
        sh = NoveumLlamaIndexSpanHandler(client=client)
        root = _open(sh, _RID)
        sh.prepare_to_drop_span(
            id_=_RID, bound_args=_BOUND_ARGS, err=RuntimeError("nope")
        )
        assert root.noveum_span.attributes["llamaindex.status"] == "error"
        assert root.noveum_span.finish.called
        client.finish_trace.assert_called_once()

    def test_exit_unknown_span_is_noop(self) -> None:
        sh = NoveumLlamaIndexSpanHandler(client=_make_client())
        assert sh.prepare_to_exit_span(id_="unknown", bound_args=_BOUND_ARGS) is None


# ---------------------------------------------------------------------------
# Setup factory
# ---------------------------------------------------------------------------


class TestSetup:
    def test_setup_registers_on_dispatcher(self) -> None:
        client = _make_client()
        dispatcher = MagicMock()
        handler = setup_llamaindex_tracing(client=client, dispatcher=dispatcher)
        assert isinstance(handler, NoveumLlamaIndexSpanHandler)
        dispatcher.add_span_handler.assert_called_once()
        dispatcher.add_event_handler.assert_called_once()

    def test_setup_requires_initialization(self, monkeypatch) -> None:
        import noveum_trace

        monkeypatch.setattr(noveum_trace, "is_initialized", lambda: False)
        with pytest.raises(RuntimeError):
            setup_llamaindex_tracing()  # no client, SDK not initialised

    def test_setup_with_explicit_client_skips_init_check(self, monkeypatch) -> None:
        import noveum_trace

        monkeypatch.setattr(noveum_trace, "is_initialized", lambda: False)
        handler = setup_llamaindex_tracing(
            client=_make_client(), dispatcher=MagicMock()
        )
        assert isinstance(handler, NoveumLlamaIndexSpanHandler)


# ---------------------------------------------------------------------------
# Full-payload capture: nodes, rerank, tools, cost, embeddings
# ---------------------------------------------------------------------------


def _node(text: str, score: float, node_id: str = "n1", metadata=None):
    inner = type("TextNode", (), {})()
    inner.node_id = node_id
    inner.text = text
    inner.metadata = metadata or {}
    inner.get_content = lambda: text
    wrapper = type("NodeWithScore", (), {})()
    wrapper.node = inner
    wrapper.score = score
    return wrapper


class TestRichCapture:
    def _setup(self, **eh_kwargs):
        client = _make_client()
        sh = NoveumLlamaIndexSpanHandler(client=client)
        eh = NoveumLlamaIndexEventHandler(span_handler=sh, **eh_kwargs)
        _open(sh, _RID)
        child = _open(sh, _CID, parent=_RID)
        return sh, eh, child

    def test_retrieved_nodes_carry_text_score_and_metadata(self) -> None:
        sh, eh, child = self._setup()
        nodes = [
            _node("chunk one", 0.91, "a", {"file": "a.md"}),
            _node("chunk two", 0.42, "b", {"file": "b.md"}),
        ]
        eh.handle(_event("RetrievalEndEvent", span_id=_CID, nodes=nodes))

        attrs = child.noveum_span.attributes
        assert attrs["retrieval.node_count"] == 2
        assert attrs["retrieval.scores"] == [0.91, 0.42]
        serialized = attrs["retrieval.nodes"]
        assert serialized[0]["id"] == "a"
        assert serialized[0]["text"] == "chunk one"
        assert serialized[0]["score"] == 0.91
        assert serialized[0]["metadata"] == {"file": "a.md"}

    def test_node_text_is_not_truncated(self) -> None:
        sh, eh, child = self._setup()
        huge = "y" * 40_000
        eh.handle(_event("RetrievalEndEvent", span_id=_CID, nodes=[_node(huge, 0.5)]))
        assert child.noveum_span.attributes["retrieval.nodes"][0]["text"] == huge

    def test_rerank_captures_input_and_ranked_output(self) -> None:
        sh, eh, child = self._setup()
        before = [_node("low", 0.30, "a"), _node("high", 0.20, "b")]
        after = [_node("high", 0.99, "b"), _node("low", 0.10, "a")]
        eh.handle(
            _event(
                "ReRankStartEvent",
                span_id=_CID,
                nodes=before,
                top_n=2,
                model_name="rerank-v3",
                query="which one?",
            )
        )
        eh.handle(_event("ReRankEndEvent", span_id=_CID, nodes=after))

        attrs = child.noveum_span.attributes
        assert attrs["rerank.model"] == "rerank-v3"
        assert attrs["rerank.query"] == "which one?"
        assert attrs["rerank.input_node_count"] == 2
        assert attrs["rerank.input_scores"] == [0.30, 0.20]
        assert attrs["rerank.output_node_count"] == 2
        assert attrs["rerank.output_scores"] == [0.99, 0.10]
        assert [n["id"] for n in attrs["rerank.input_nodes"]] == ["a", "b"]
        assert [n["id"] for n in attrs["rerank.output_nodes"]] == ["b", "a"]

    def test_embedding_records_counts_and_dimensions_not_vectors(self) -> None:
        sh, eh, child = self._setup()
        eh.handle(
            _event(
                "EmbeddingEndEvent",
                span_id=_CID,
                chunks=["alpha", "beta"],
                embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            )
        )
        attrs = child.noveum_span.attributes
        assert attrs["embedding.chunk_count"] == 2
        assert attrs["embedding.vector_count"] == 2
        assert attrs["embedding.dimensions"] == 3
        assert "embedding.chunks" not in attrs
        assert not any("0.1" in str(v) for v in attrs.values())

    def test_embedding_chunks_captured_when_opted_in(self) -> None:
        sh, eh, child = self._setup(capture_embedding_chunks=True)
        eh.handle(
            _event(
                "EmbeddingEndEvent",
                span_id=_CID,
                chunks=["alpha", "beta"],
                embeddings=[[0.1], [0.2]],
            )
        )
        assert child.noveum_span.attributes["embedding.chunks"] == ["alpha", "beta"]

    def test_llm_cost_estimated_from_start_event_model(self) -> None:
        sh, eh, child = self._setup()
        eh.handle(
            _event("LLMChatStartEvent", span_id=_CID, model_dict={"model": "gpt-4o"})
        )
        response = type("R", (), {})()
        response.raw = {"usage": {"prompt_tokens": 1000, "completion_tokens": 500}}
        response.message = _message("assistant", "hi")
        eh.handle(_event("LLMChatEndEvent", span_id=_CID, response=response))

        attrs = child.noveum_span.attributes
        assert attrs["llm.cost.total"] > 0
        assert attrs["llm.cost.currency"] == "USD"

    def test_llm_start_captures_system_prompt_and_tools(self) -> None:
        sh, eh, child = self._setup()
        tool = type("ToolMetadata", (), {})()
        tool.name = "search"
        tool.description = "Search the docs"
        eh.handle(
            _event(
                "LLMChatStartEvent",
                span_id=_CID,
                model_dict={"model": "gpt-4o"},
                additional_kwargs={"tools": [tool]},
                messages=[
                    _message("system", "You are terse."),
                    _message("user", "hello"),
                ],
            )
        )
        attrs = child.noveum_span.attributes
        assert attrs["llm.system_prompt"] == "You are terse."
        assert attrs["llm.available_tool_count"] == 1
        assert attrs["llm.available_tools"][0]["name"] == "search"

    def test_agent_tool_call_event_is_mapped(self) -> None:
        sh, eh, child = self._setup()
        tool = type("ToolMetadata", (), {})()
        tool.name = "get_weather"
        tool.description = "Look up weather"
        eh.handle(
            _event(
                "AgentToolCallEvent",
                span_id=_CID,
                tool=tool,
                arguments='{"city":"Paris"}',
            )
        )
        attrs = child.noveum_span.attributes
        assert attrs["tool.name"] == "get_weather"
        assert attrs["tool.description"] == "Look up weather"
        assert attrs["tool.input"] == '{"city":"Paris"}'

    def test_query_end_captures_response_and_source_nodes(self) -> None:
        sh, eh, child = self._setup()
        response = type("Response", (), {"__str__": lambda self: "the answer"})()
        response.source_nodes = [_node("src", 0.7, "s1")]
        eh.handle(_event("QueryEndEvent", span_id=_CID, response=response))

        attrs = child.noveum_span.attributes
        assert attrs["query.response"] == "the answer"
        assert attrs["query.source_nodes"][0]["id"] == "s1"

    def test_top_k_read_from_retriever_instance(self) -> None:
        client = _make_client()
        sh = NoveumLlamaIndexSpanHandler(client=client)
        retriever = type("VectorIndexRetriever", (), {})()
        retriever.similarity_top_k = 7
        span = sh.new_span(
            id_="VectorIndexRetriever.retrieve-33333333-3333-3333-3333-333333333333",
            bound_args=_BOUND_ARGS,
            instance=retriever,
        )
        assert span.noveum_span.attributes["retrieval.top_k"] == 7
