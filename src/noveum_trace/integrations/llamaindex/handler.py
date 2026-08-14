"""
Noveum instrumentation handlers for LlamaIndex.

Bridges LlamaIndex's modern instrumentation system
(``llama_index.core.instrumentation``) into Noveum:

* :class:`NoveumLlamaIndexSpanHandler` maps every LlamaIndex span (query engine,
  retriever, synthesizer, LLM, embedding calls) onto Noveum traces/spans,
  preserving the parent-child hierarchy that LlamaIndex threads through its
  ``active_span_id`` context variable.
* :class:`NoveumLlamaIndexEventHandler` enriches the corresponding open Noveum
  span with LLM / embedding / retrieval / rerank attributes from the events that
  fire inside each span.

Register both on the root dispatcher with :func:`setup_llamaindex_tracing`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
from llama_index.core.instrumentation.span.base import BaseSpan
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler

from noveum_trace.integrations._common import (
    estimate_cost_safe,
    finish_span,
    probe,
    set_span_attributes,
    stringify,
)
from noveum_trace.integrations.llamaindex import constants as C
from noveum_trace.integrations.llamaindex.utils import (
    classify_operation,
    derive_provider,
    extract_model_name,
    extract_node_scores,
    extract_query_str,
    extract_response_text,
    extract_system_prompt,
    extract_token_usage,
    extract_tool_metadata,
    operation_from_span_id,
    serialize_messages,
    serialize_nodes,
    vector_dimensions,
)

logger = logging.getLogger(__name__)

_LLM_START_EVENTS = frozenset({"LLMChatStartEvent", "LLMCompletionStartEvent"})
_LLM_END_EVENTS = frozenset(
    {"LLMChatEndEvent", "LLMCompletionEndEvent", "LLMPredictEndEvent"}
)


# ---------------------------------------------------------------------------
# Span attribute helpers (module-level, never raise)
# ---------------------------------------------------------------------------


_set_attrs = set_span_attributes
_finish_span = finish_span


def _apply_error(span: Any, err: Any) -> None:
    if span is None:
        return
    attrs: dict[str, Any] = {C.ATTR_STATUS: C.STATUS_ERROR}
    if err is not None:
        attrs[C.ATTR_ERROR_TYPE] = type(err).__name__
        attrs[C.ATTR_ERROR_MESSAGE] = stringify(err)
    try:
        span.set_attributes(attrs)
        from noveum_trace.core.span import SpanStatus

        span.set_status(SpanStatus.ERROR, str(err) if err else "error")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("failed to mark span error: %s", exc)


# ---------------------------------------------------------------------------
# Span type
# ---------------------------------------------------------------------------


class NoveumLlamaIndexSpan(BaseSpan):
    """A LlamaIndex span carrying its mapped Noveum trace/span handles."""

    noveum_span: Any = None
    noveum_trace: Any = None
    is_root: bool = False


# ---------------------------------------------------------------------------
# Span handler
# ---------------------------------------------------------------------------


class NoveumLlamaIndexSpanHandler(BaseSpanHandler[NoveumLlamaIndexSpan]):
    """
    Maps LlamaIndex spans onto Noveum traces/spans.

    Each top-level LlamaIndex span (``parent_span_id is None``) opens a new Noveum
    trace; nested spans become Noveum child spans under the same trace. The Noveum
    handles ride on the returned :class:`NoveumLlamaIndexSpan`, so a child resolves
    its parent purely from ``open_spans`` — no separate bookkeeping.
    """

    client: Any = None
    trace_name_prefix: str = C.DEFAULT_TRACE_NAME_PREFIX

    def __init__(
        self,
        client: Any = None,
        trace_name_prefix: str = C.DEFAULT_TRACE_NAME_PREFIX,
        **kwargs: Any,
    ) -> None:
        # ``BaseSpanHandler`` defines a custom ``__init__`` whose signature does
        # not accept extra fields, so bridge our config in explicitly.
        super().__init__(**kwargs)
        self.client = client
        self.trace_name_prefix = trace_name_prefix

    @classmethod
    def class_name(cls) -> str:
        return "NoveumLlamaIndexSpanHandler"

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from noveum_trace import get_client, is_initialized

            if is_initialized():
                return get_client()
        except Exception:  # pragma: no cover - defensive
            return None
        return None

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[NoveumLlamaIndexSpan]:
        try:
            client = self._get_client()
            if client is None:
                return None

            operation = operation_from_span_id(id_)
            attributes: dict[str, Any] = {
                C.ATTR_OPERATION: operation,
                C.ATTR_SPAN_TYPE: classify_operation(operation),
            }
            # The retriever instance is the only place the configured top-k is
            # visible; the retrieval events do not carry it.
            top_k = probe(instance, "similarity_top_k", "top_k")
            if isinstance(top_k, int):
                attributes[C.ATTR_RETRIEVAL_TOP_K] = top_k

            parent = self.open_spans.get(parent_span_id) if parent_span_id else None
            if parent is not None and getattr(parent, "noveum_trace", None) is not None:
                noveum_trace_obj = parent.noveum_trace
                parent_noveum_span = getattr(parent, "noveum_span", None)
                parent_noveum_span_id = getattr(parent_noveum_span, "span_id", None)
                noveum_span = noveum_trace_obj.create_span(
                    name=operation,
                    parent_span_id=parent_noveum_span_id,
                    attributes=attributes,
                )
                return NoveumLlamaIndexSpan(
                    id_=id_,
                    parent_id=parent_span_id,
                    tags=tags or {},
                    noveum_span=noveum_span,
                    noveum_trace=noveum_trace_obj,
                    is_root=False,
                )

            # Root span → open a new Noveum trace.
            attributes[C.ATTR_FRAMEWORK] = C.FRAMEWORK_NAME
            noveum_trace_obj = client.start_trace(
                name=operation or f"{self.trace_name_prefix}.trace",
                attributes={C.ATTR_FRAMEWORK: C.FRAMEWORK_NAME},
                set_as_current=False,
            )
            noveum_span = noveum_trace_obj.create_span(
                name=operation,
                parent_span_id=None,
                attributes=attributes,
            )
            return NoveumLlamaIndexSpan(
                id_=id_,
                parent_id=parent_span_id,
                tags=tags or {},
                noveum_span=noveum_span,
                noveum_trace=noveum_trace_obj,
                is_root=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("new_span failed: %s", exc, exc_info=True)
            return None

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[NoveumLlamaIndexSpan]:
        span = self.open_spans.get(id_)
        if span is None:
            return None
        try:
            noveum_span = getattr(span, "noveum_span", None)
            if noveum_span is not None:
                _set_attrs(noveum_span, {C.ATTR_STATUS: C.STATUS_OK})
                _finish_span(noveum_span)
            if getattr(span, "is_root", False):
                self._finish_trace(span)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("prepare_to_exit_span failed: %s", exc, exc_info=True)
        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[NoveumLlamaIndexSpan]:
        span = self.open_spans.get(id_)
        if span is None:
            return None
        try:
            noveum_span = getattr(span, "noveum_span", None)
            if noveum_span is not None:
                _apply_error(noveum_span, err)
                _finish_span(noveum_span)
            if getattr(span, "is_root", False):
                self._finish_trace(span)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("prepare_to_drop_span failed: %s", exc, exc_info=True)
        return span

    def _finish_trace(self, span: NoveumLlamaIndexSpan) -> None:
        noveum_trace_obj = getattr(span, "noveum_trace", None)
        if noveum_trace_obj is None:
            return
        try:
            client = self._get_client()
            if client is not None:
                client.finish_trace(noveum_trace_obj)
            else:
                noveum_trace_obj.finish()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("finish trace failed: %s", exc)


# ---------------------------------------------------------------------------
# Event handler
# ---------------------------------------------------------------------------


class NoveumLlamaIndexEventHandler(BaseEventHandler):
    """
    Enriches open Noveum spans with data from LlamaIndex events.

    Correlates each event to its span via ``event.span_id`` (which equals the
    enclosing span's ``id_``) and attaches LLM / embedding / retrieval / rerank
    attributes to that span's Noveum span.
    """

    span_handler: Any = None
    capture_inputs: bool = True
    capture_outputs: bool = True
    capture_llm_messages: bool = True
    capture_cost: bool = True
    capture_embedding_chunks: bool = False

    @classmethod
    def class_name(cls) -> str:
        return "NoveumLlamaIndexEventHandler"

    def handle(self, event: Any, **kwargs: Any) -> Any:
        try:
            self._handle(event)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("event handle failed: %s", exc, exc_info=True)
        return None

    def _resolve_span(self, event: Any) -> Any:
        span_handler = self.span_handler
        if span_handler is None:
            return None
        span_id = getattr(event, "span_id", None)
        if span_id is None:
            return None
        mapped = span_handler.open_spans.get(span_id)
        return getattr(mapped, "noveum_span", None) if mapped is not None else None

    def _handle(self, event: Any) -> None:
        noveum_span = self._resolve_span(event)
        if noveum_span is None:
            return
        name = type(event).__name__

        if name == "ExceptionEvent":
            _apply_error(noveum_span, getattr(event, "exception", None))
            return

        attrs: dict[str, Any] = {}
        if name in _LLM_START_EVENTS:
            self._map_llm_start(event, attrs)
        elif name in _LLM_END_EVENTS:
            self._map_llm_end(event, attrs, noveum_span)
        elif name == "AgentToolCallEvent":
            self._map_tool_call(event, attrs)
        elif name == "EmbeddingStartEvent":
            attrs[C.ATTR_SPAN_TYPE] = C.SPAN_TYPE_EMBEDDING
            model = extract_model_name(getattr(event, "model_dict", None))
            if model:
                attrs[C.ATTR_EMBEDDING_MODEL] = model
        elif name == "EmbeddingEndEvent":
            self._map_embedding_end(event, attrs)
        elif name == "RetrievalStartEvent":
            attrs[C.ATTR_SPAN_TYPE] = C.SPAN_TYPE_RETRIEVAL
            if self.capture_inputs:
                query = extract_query_str(getattr(event, "str_or_query_bundle", None))
                if query:
                    attrs[C.ATTR_RETRIEVAL_QUERY] = query
        elif name == "RetrievalEndEvent":
            self._map_retrieval_end(event, attrs)
        elif name == "QueryStartEvent":
            attrs[C.ATTR_SPAN_TYPE] = C.SPAN_TYPE_QUERY
            if self.capture_inputs:
                query = extract_query_str(getattr(event, "query", None))
                if query:
                    attrs[C.ATTR_QUERY_TEXT] = query
        elif name == "QueryEndEvent":
            self._map_query_end(event, attrs)
        elif name == "ReRankStartEvent":
            self._map_rerank_start(event, attrs)
        elif name == "ReRankEndEvent":
            self._map_rerank_end(event, attrs)

        _set_attrs(noveum_span, attrs)

    def _map_llm_start(self, event: Any, attrs: dict[str, Any]) -> None:
        attrs[C.ATTR_SPAN_TYPE] = C.SPAN_TYPE_LLM
        model = extract_model_name(getattr(event, "model_dict", None))
        if model:
            attrs[C.ATTR_LLM_MODEL] = model
            provider = derive_provider(model)
            if provider:
                attrs[C.ATTR_LLM_PROVIDER] = provider

        model_dict = getattr(event, "model_dict", None)
        additional = getattr(event, "additional_kwargs", None)
        tools = probe(additional, "tools") or probe(model_dict, "tools")
        if tools and self.capture_llm_messages:
            schemas = [
                extract_tool_metadata(tool) or {"tool": stringify(tool)}
                for tool in tools
            ]
            attrs[C.ATTR_LLM_AVAILABLE_TOOLS] = schemas
            attrs[C.ATTR_LLM_AVAILABLE_TOOL_COUNT] = len(schemas)

        if self.capture_llm_messages:
            raw_messages = getattr(event, "messages", None)
            messages = serialize_messages(raw_messages)
            if messages:
                attrs[C.ATTR_LLM_INPUT] = messages
                system_prompt = extract_system_prompt(raw_messages)
                if system_prompt:
                    attrs[C.ATTR_LLM_SYSTEM_PROMPT] = system_prompt
            prompt = getattr(event, "prompt", None)
            if prompt is not None:
                attrs[C.ATTR_LLM_INPUT] = stringify(prompt)

    def _map_llm_end(self, event: Any, attrs: dict[str, Any], noveum_span: Any) -> None:
        response = getattr(event, "response", None)
        usage = extract_token_usage(response)
        if usage["input_tokens"] is not None:
            attrs[C.ATTR_LLM_INPUT_TOKENS] = usage["input_tokens"]
        if usage["output_tokens"] is not None:
            attrs[C.ATTR_LLM_OUTPUT_TOKENS] = usage["output_tokens"]
        if usage["total_tokens"] is not None:
            attrs[C.ATTR_LLM_TOTAL_TOKENS] = usage["total_tokens"]
        if usage["cached_input_tokens"] is not None:
            attrs[C.ATTR_LLM_CACHED_INPUT_TOKENS] = usage["cached_input_tokens"]
        if usage["reasoning_tokens"] is not None:
            attrs[C.ATTR_LLM_REASONING_TOKENS] = usage["reasoning_tokens"]

        if self.capture_cost:
            self._apply_cost(usage, attrs, noveum_span)

        tool_calls = self._response_tool_calls(response)
        if tool_calls:
            attrs[C.ATTR_LLM_TOOL_CALLS] = tool_calls
            attrs[C.ATTR_LLM_TOOL_CALL_COUNT] = len(tool_calls)

        if self.capture_outputs:
            text = extract_response_text(response)
            if text is None:
                output = getattr(event, "output", None)
                text = stringify(output) if output is not None else None
            if text:
                attrs[C.ATTR_LLM_OUTPUT] = text

    def _apply_cost(
        self, usage: dict[str, Any], attrs: dict[str, Any], noveum_span: Any
    ) -> None:
        """
        Estimate cost from the model recorded by the matching start event.

        LlamaIndex splits a call across ``LLMChatStartEvent`` (which carries the
        model) and ``LLMChatEndEvent`` (which carries usage), so the model is
        read back off the span the start event already annotated.
        """
        model = attrs.get(C.ATTR_LLM_MODEL)
        if not model:
            existing = getattr(noveum_span, "attributes", None)
            if isinstance(existing, dict):
                model = existing.get(C.ATTR_LLM_MODEL)
        if not model:
            return
        cost = estimate_cost_safe(
            str(model), usage["input_tokens"], usage["output_tokens"]
        )
        if cost.get("total"):
            attrs[C.ATTR_LLM_COST_INPUT] = cost.get("input")
            attrs[C.ATTR_LLM_COST_OUTPUT] = cost.get("output")
            attrs[C.ATTR_LLM_COST_TOTAL] = cost.get("total")
            attrs[C.ATTR_LLM_COST_CURRENCY] = cost.get("currency")

    def _response_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """Pull tool calls off a ``ChatResponse`` message's additional kwargs."""
        message = probe(response, "message")
        additional = (
            probe(message, "additional_kwargs") if message is not None else None
        )
        raw_calls = probe(additional, "tool_calls") if additional else None
        calls: list[dict[str, Any]] = []
        for raw in raw_calls or []:
            function = probe(raw, "function")
            name = (
                probe(function, "name") if function is not None else probe(raw, "name")
            )
            arguments = (
                probe(function, "arguments")
                if function is not None
                else probe(raw, "arguments")
            )
            if name is None and arguments is None:
                continue
            entry: dict[str, Any] = {"name": stringify(name) if name else None}
            call_id = probe(raw, "id", "tool_call_id")
            if call_id is not None:
                entry["call_id"] = stringify(call_id)
            if arguments is not None:
                entry["arguments"] = stringify(arguments)
            calls.append(entry)
        return calls

    def _map_tool_call(self, event: Any, attrs: dict[str, Any]) -> None:
        """Map an agent tool invocation (``AgentToolCallEvent``)."""
        metadata = extract_tool_metadata(getattr(event, "tool", None))
        if metadata.get("name"):
            attrs[C.ATTR_TOOL_NAME] = metadata["name"]
        if metadata.get("description"):
            attrs[C.ATTR_TOOL_DESCRIPTION] = metadata["description"]
        if self.capture_inputs:
            arguments = getattr(event, "arguments", None)
            if arguments is not None:
                attrs[C.ATTR_TOOL_INPUT] = stringify(arguments)

    def _map_embedding_end(self, event: Any, attrs: dict[str, Any]) -> None:
        """
        Map an embedding call.

        Counts and vector width are always recorded; the chunk *text* is behind
        ``capture_embedding_chunks`` because indexing a corpus emits one
        embedding call per batch, and capturing their text would copy the whole
        source corpus into the trace store. The embedding vectors themselves are
        never attached — a float array is not usable in a trace viewer.
        """
        chunks = getattr(event, "chunks", None)
        if chunks is not None:
            attrs[C.ATTR_EMBEDDING_CHUNK_COUNT] = len(chunks)
            if self.capture_embedding_chunks:
                attrs[C.ATTR_EMBEDDING_CHUNKS] = [stringify(c) for c in chunks]
        embeddings = getattr(event, "embeddings", None)
        if embeddings is not None:
            attrs[C.ATTR_EMBEDDING_VECTOR_COUNT] = len(embeddings)
            dimensions = vector_dimensions(embeddings)
            if dimensions is not None:
                attrs[C.ATTR_EMBEDDING_DIMENSIONS] = dimensions

    def _map_retrieval_end(self, event: Any, attrs: dict[str, Any]) -> None:
        nodes = getattr(event, "nodes", None)
        if nodes is None:
            return
        attrs[C.ATTR_RETRIEVAL_NODE_COUNT] = len(nodes)
        # Ordered best-match first: the retriever's own top-k similarity scores.
        attrs[C.ATTR_RETRIEVAL_SCORES] = extract_node_scores(nodes)
        if self.capture_outputs:
            attrs[C.ATTR_RETRIEVAL_NODES] = serialize_nodes(nodes)

    def _map_query_end(self, event: Any, attrs: dict[str, Any]) -> None:
        response = getattr(event, "response", None)
        if response is None:
            return
        if self.capture_outputs:
            attrs[C.ATTR_QUERY_RESPONSE] = stringify(response)
            source_nodes = getattr(response, "source_nodes", None)
            if source_nodes:
                attrs[C.ATTR_QUERY_SOURCE_NODES] = serialize_nodes(source_nodes)

    def _map_rerank_start(self, event: Any, attrs: dict[str, Any]) -> None:
        attrs[C.ATTR_SPAN_TYPE] = C.SPAN_TYPE_RERANK
        model = getattr(event, "model_name", None)
        if model:
            attrs[C.ATTR_RERANK_MODEL] = str(model)
        top_n = getattr(event, "top_n", None)
        if top_n is not None:
            attrs[C.ATTR_RERANK_TOP_N] = top_n
        if self.capture_inputs:
            query = extract_query_str(getattr(event, "query", None))
            if query:
                attrs[C.ATTR_RERANK_QUERY] = query
        nodes = getattr(event, "nodes", None)
        if nodes is not None:
            attrs[C.ATTR_RERANK_INPUT_NODE_COUNT] = len(nodes)
            attrs[C.ATTR_RERANK_INPUT_SCORES] = extract_node_scores(nodes)
            if self.capture_inputs:
                attrs[C.ATTR_RERANK_INPUT_NODES] = serialize_nodes(nodes)

    def _map_rerank_end(self, event: Any, attrs: dict[str, Any]) -> None:
        """
        Map the reranked result set.

        An LLM-backed reranker (``LLMRerank`` and friends) runs its scoring
        prompt through the dispatcher as a nested LLM span, so its reasoning is
        captured there as ``llm.output`` rather than on this span — the rerank
        events themselves carry only the node lists.
        """
        nodes = getattr(event, "nodes", None)
        if nodes is None:
            return
        attrs[C.ATTR_RERANK_OUTPUT_NODE_COUNT] = len(nodes)
        attrs[C.ATTR_RERANK_OUTPUT_SCORES] = extract_node_scores(nodes)
        if self.capture_outputs:
            attrs[C.ATTR_RERANK_OUTPUT_NODES] = serialize_nodes(nodes)


def setup_llamaindex_tracing(
    client: Any = None,
    *,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    capture_llm_messages: bool = True,
    capture_cost: bool = True,
    capture_embedding_chunks: bool = False,
    trace_name_prefix: str = C.DEFAULT_TRACE_NAME_PREFIX,
    dispatcher: Any = None,
) -> NoveumLlamaIndexSpanHandler:
    """
    Register Noveum span + event handlers on LlamaIndex's dispatcher.

    Requires ``noveum_trace.init(...)`` to have been called (or an explicit
    ``client=``). Handlers are added to the root dispatcher by default, so all
    LlamaIndex activity is traced.

    Args:
        client: Explicit Noveum client. Defaults to the globally initialised one.
        capture_inputs: Capture query text, retrieval and rerank query text,
            rerank input nodes, and agent tool arguments (default on).
        capture_outputs: Capture LLM response text (``llm.output``), query
            response text, retrieved node content, query source nodes and
            reranked output nodes (default on).
        capture_llm_messages: Capture LLM prompt messages (``llm.input``),
            system prompts and available tool schemas (default on). LLM
            *responses* are governed by ``capture_outputs``, not this flag.
        capture_cost: Estimate LLM cost from the model and token counts
            (default on).
        capture_embedding_chunks: Capture the *text* being embedded (default
            **off**). Indexing a corpus emits an embedding call per batch, so
            turning this on copies the whole source corpus into the trace
            store — useful when building evaluation datasets from a small
            index, expensive on a large one. Counts, vector counts and vector
            width are recorded either way; the embedding vectors themselves are
            never attached.
        trace_name_prefix: Prefix used when an operation name cannot be derived.
        dispatcher: Dispatcher to register on. Defaults to the root dispatcher
            (``get_dispatcher()``).

    Returns:
        The registered :class:`NoveumLlamaIndexSpanHandler`.

    Raises:
        RuntimeError: If ``noveum_trace.init()`` has not been called and no
            explicit ``client`` is provided (otherwise LlamaIndex activity would
            be silently untraced).
    """
    if client is None:
        try:
            from noveum_trace import is_initialized
        except ImportError as exc:  # pragma: no cover - noveum_trace always present
            raise RuntimeError(
                "noveum_trace is not installed. Install with: pip install noveum-trace"
            ) from exc
        if not is_initialized():
            raise RuntimeError(
                "Noveum tracing is not initialized. Call noveum_trace.init() "
                "before setup_llamaindex_tracing() (or pass an explicit client=)."
            )

    span_handler = NoveumLlamaIndexSpanHandler(
        client=client,
        trace_name_prefix=trace_name_prefix,
    )
    event_handler = NoveumLlamaIndexEventHandler(
        span_handler=span_handler,
        capture_inputs=capture_inputs,
        capture_outputs=capture_outputs,
        capture_llm_messages=capture_llm_messages,
        capture_cost=capture_cost,
        capture_embedding_chunks=capture_embedding_chunks,
    )
    target = dispatcher if dispatcher is not None else get_dispatcher()
    target.add_span_handler(span_handler)
    target.add_event_handler(event_handler)
    return span_handler
