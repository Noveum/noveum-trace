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

from noveum_trace.integrations.llamaindex import constants as C
from noveum_trace.integrations.llamaindex.utils import (
    classify_operation,
    derive_provider,
    extract_model_name,
    extract_node_contents,
    extract_node_scores,
    extract_query_str,
    extract_response_text,
    extract_token_usage,
    operation_from_span_id,
    serialize_messages,
    truncate_text,
)

logger = logging.getLogger(__name__)

_LLM_START_EVENTS = frozenset({"LLMChatStartEvent", "LLMCompletionStartEvent"})
_LLM_END_EVENTS = frozenset(
    {"LLMChatEndEvent", "LLMCompletionEndEvent", "LLMPredictEndEvent"}
)


# ---------------------------------------------------------------------------
# Span attribute helpers (module-level, never raise)
# ---------------------------------------------------------------------------


def _set_attrs(span: Any, attrs: dict[str, Any]) -> None:
    if not attrs or span is None:
        return
    try:
        span.set_attributes(attrs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("failed to set span attributes: %s", exc)


def _finish_span(span: Any) -> None:
    if span is None:
        return
    try:
        if not span.is_finished():
            span.finish()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("failed to finish span: %s", exc)


def _apply_error(span: Any, err: Any) -> None:
    if span is None:
        return
    attrs: dict[str, Any] = {C.ATTR_STATUS: C.STATUS_ERROR}
    if err is not None:
        attrs[C.ATTR_ERROR_TYPE] = type(err).__name__
        attrs[C.ATTR_ERROR_MESSAGE] = truncate_text(str(err), C.MAX_TEXT_LENGTH)
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
    capture_inputs: bool = False
    capture_outputs: bool = False
    capture_llm_messages: bool = False

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
            self._map_llm_end(event, attrs)
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
                    attrs[C.ATTR_RETRIEVAL_QUERY] = truncate_text(query)
        elif name == "RetrievalEndEvent":
            self._map_retrieval_end(event, attrs)
        elif name == "QueryStartEvent":
            attrs[C.ATTR_SPAN_TYPE] = C.SPAN_TYPE_QUERY
            if self.capture_inputs:
                query = extract_query_str(getattr(event, "query", None))
                if query:
                    attrs[C.ATTR_QUERY_TEXT] = truncate_text(query)
        elif name == "QueryEndEvent":
            if self.capture_outputs:
                response = getattr(event, "response", None)
                if response is not None:
                    attrs[C.ATTR_QUERY_RESPONSE] = truncate_text(str(response))
        elif name == "ReRankStartEvent":
            self._map_rerank_start(event, attrs)
        elif name == "ReRankEndEvent":
            nodes = getattr(event, "nodes", None)
            if nodes is not None:
                attrs[C.ATTR_RERANK_OUTPUT_NODE_COUNT] = len(nodes)

        _set_attrs(noveum_span, attrs)

    def _map_llm_start(self, event: Any, attrs: dict[str, Any]) -> None:
        attrs[C.ATTR_SPAN_TYPE] = C.SPAN_TYPE_LLM
        model = extract_model_name(getattr(event, "model_dict", None))
        if model:
            attrs[C.ATTR_LLM_MODEL] = model
            provider = derive_provider(model)
            if provider:
                attrs[C.ATTR_LLM_PROVIDER] = provider
        if self.capture_llm_messages:
            messages = serialize_messages(getattr(event, "messages", None))
            if messages:
                attrs[C.ATTR_LLM_INPUT] = messages
            prompt = getattr(event, "prompt", None)
            if prompt is not None:
                attrs[C.ATTR_LLM_INPUT] = truncate_text(prompt)

    def _map_llm_end(self, event: Any, attrs: dict[str, Any]) -> None:
        response = getattr(event, "response", None)
        usage = extract_token_usage(response)
        if usage["input_tokens"] is not None:
            attrs[C.ATTR_LLM_INPUT_TOKENS] = usage["input_tokens"]
        if usage["output_tokens"] is not None:
            attrs[C.ATTR_LLM_OUTPUT_TOKENS] = usage["output_tokens"]
        if usage["total_tokens"] is not None:
            attrs[C.ATTR_LLM_TOTAL_TOKENS] = usage["total_tokens"]
        if self.capture_outputs:
            text = extract_response_text(response)
            if text is None:
                output = getattr(event, "output", None)
                text = str(output) if output is not None else None
            if text:
                attrs[C.ATTR_LLM_OUTPUT] = truncate_text(text)

    def _map_embedding_end(self, event: Any, attrs: dict[str, Any]) -> None:
        chunks = getattr(event, "chunks", None)
        if chunks is not None:
            attrs[C.ATTR_EMBEDDING_CHUNK_COUNT] = len(chunks)
        embeddings = getattr(event, "embeddings", None)
        if embeddings is not None:
            attrs[C.ATTR_EMBEDDING_VECTOR_COUNT] = len(embeddings)

    def _map_retrieval_end(self, event: Any, attrs: dict[str, Any]) -> None:
        nodes = getattr(event, "nodes", None)
        if nodes is not None:
            attrs[C.ATTR_RETRIEVAL_NODE_COUNT] = len(nodes)
            attrs[C.ATTR_RETRIEVAL_SCORES] = extract_node_scores(nodes)
            if self.capture_outputs:
                attrs[C.ATTR_RETRIEVAL_NODES] = extract_node_contents(
                    nodes, C.MAX_NODE_CONTENT_LENGTH
                )

    def _map_rerank_start(self, event: Any, attrs: dict[str, Any]) -> None:
        attrs[C.ATTR_SPAN_TYPE] = C.SPAN_TYPE_RERANK
        model = getattr(event, "model_name", None)
        if model:
            attrs[C.ATTR_RERANK_MODEL] = str(model)
        top_n = getattr(event, "top_n", None)
        if top_n is not None:
            attrs[C.ATTR_RERANK_TOP_N] = top_n
        nodes = getattr(event, "nodes", None)
        if nodes is not None:
            attrs[C.ATTR_RERANK_INPUT_NODE_COUNT] = len(nodes)


def setup_llamaindex_tracing(
    client: Any = None,
    *,
    capture_inputs: bool = False,
    capture_outputs: bool = False,
    capture_llm_messages: bool = False,
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
        capture_inputs: Capture query / retrieval query text (default off).
        capture_outputs: Capture response text and retrieved node content
            (default off).
        capture_llm_messages: Capture full LLM prompt/response messages
            (default off — the most sensitive payload).
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
    )
    target = dispatcher if dispatcher is not None else get_dispatcher()
    target.add_span_handler(span_handler)
    target.add_event_handler(event_handler)
    return span_handler
