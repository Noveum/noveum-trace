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

    def test_privacy_safe_defaults(self) -> None:
        eh = NoveumLlamaIndexEventHandler()
        assert eh.capture_inputs is False
        assert eh.capture_outputs is False
        assert eh.capture_llm_messages is False


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
        sh, eh, child = self._setup()
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
