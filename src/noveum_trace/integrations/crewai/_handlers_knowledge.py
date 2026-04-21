"""
Knowledge-retrieval event handler mixin for NoveumCrewAIListener.

CrewAI Knowledge is the RAG-style knowledge base that agents can search
and query.  It is distinct from Memory (short/long-term session state):
Knowledge is a pre-loaded, static (or semi-static) corpus of documents,
PDFs, URLs, or custom data sources attached to a Crew.

Handles CrewAI ``BaseEventListener`` knowledge events:

  Retrieval (bulk context pull for a task or agent):
  - ``on_knowledge_retrieval_started``    → open ``crewai.memory_op`` span
                                             tagged as ``memory.type = "knowledge"``,
                                             ``memory.operation = "retrieval"``;
                                             capture agent_role, task context
  - ``on_knowledge_retrieval_completed``  → close as SUCCESS; write sources list,
                                             result_count, full retrieved content,
                                             score min/max and per-result ``scores`` when present

  Query (targeted search within the knowledge base):
  - ``on_knowledge_query_started``        → open span;
                                             capture query text, sources filter,
                                             top_k parameter
  - ``on_knowledge_query_completed``      → close as SUCCESS; write results,
                                             result_count, score min/max / per-result scores, duration_ms
  - ``on_knowledge_query_failed``         → close as ERROR

  Search-query failure (variant fired by the search layer, not the query layer):
  - ``on_knowledge_search_query_failed``  → annotate nearest open knowledge span
                                             or agent span with search error details

All knowledge spans use ``SPAN_KNOWLEDGE`` (``crewai.knowledge``) so
knowledge operations are distinguishable from memory operations while still
emitting ``memory.*`` attributes for compatibility.

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown,
    _agent_spans, _task_spans, _memory_op_spans, _memory_op_start_times
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_AGENT_ROLE,
    ATTR_ERROR_STACKTRACE,
    ATTR_MEMORY_DURATION_MS,
    ATTR_MEMORY_OP_ID,
    ATTR_MEMORY_OPERATION,
    ATTR_MEMORY_QUERY,
    ATTR_MEMORY_RESULT_COUNT,
    ATTR_MEMORY_STATUS,
    ATTR_MEMORY_TYPE,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    MAX_DESCRIPTION_LENGTH,
    SPAN_KNOWLEDGE,
)
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase
from noveum_trace.integrations.crewai.crewai_utils import (
    finish_span_common,
    monotonic_now,
)
from noveum_trace.integrations.crewai.crewai_utils import (
    resolve_agent_id as _resolve_agent_id,
)
from noveum_trace.integrations.crewai.crewai_utils import (
    safe_getattr,
    safe_json_dumps,
    set_span_attributes,
    truncate_str,
)

logger = logging.getLogger(__name__)

# Canonical memory.type value for all knowledge events
_KNOWLEDGE_TYPE = "knowledge"

# Operation strings
_OP_RETRIEVAL = "retrieval"
_OP_QUERY = "query"


class _KnowledgeHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI Knowledge retrieval and query events.

    All public methods match the ``BaseEventListener`` callback signature::

        def on_knowledge_retrieval_started(self, source, event): ...

    ``source`` is the knowledge store object or Agent; ``event`` carries the
    per-operation payload.  Every method is fully exception-shielded.
    """

    # =========================================================================
    # RETRIEVAL — started / completed
    # =========================================================================

    def on_knowledge_retrieval_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.memory_op`` span for a knowledge bulk-retrieval operation.

        This fires when CrewAI hydrates an agent's context with knowledge
        base content for the current task.  It may encompass multiple
        underlying queries internally.

        Attributes set at span open
        ---------------------------
        - ``memory.op_id``          — unique operation identifier
        - ``memory.type``           — ``"knowledge"``
        - ``memory.operation``      — ``"retrieval"``
        - ``agent.role``            — role of the agent being hydrated
        - ``knowledge.sources``     — JSON list of source names / identifiers
        - ``task.id``               — id of the task requesting knowledge context
        """
        if not self._is_active():
            return
        try:
            op_id = _resolve_op_id(event, source)
            agent_id = _resolve_agent_id(source, event)

            attrs: dict[str, Any] = {
                ATTR_MEMORY_OP_ID: op_id,
                ATTR_MEMORY_TYPE: _KNOWLEDGE_TYPE,
                ATTR_MEMORY_OPERATION: _OP_RETRIEVAL,
            }

            agent_role = safe_getattr(event, "agent_role") or safe_getattr(
                source, "role"
            )
            if agent_role:
                attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

            task_id = safe_getattr(event, "task_id") or safe_getattr(
                safe_getattr(source, "task"), "id"
            )
            if task_id:
                attrs["task.id"] = str(task_id)

            sources = (
                safe_getattr(event, "sources")
                or safe_getattr(event, "knowledge_sources")
                or safe_getattr(source, "sources")
            )
            if sources is not None:
                attrs["knowledge.sources"] = safe_json_dumps(sources)

            self._open_knowledge_span(op_id, agent_id, attrs)
            logger.debug(
                "Knowledge retrieval span opened: op_id=%s agent_id=%s",
                op_id,
                agent_id,
            )
        except Exception:
            logger.debug(
                "on_knowledge_retrieval_started error:\n%s", traceback.format_exc()
            )

    def on_knowledge_retrieval_completed(self, source: Any, event: Any) -> None:
        """
        Close the knowledge retrieval span as OK.

        Attributes written
        ------------------
        - ``memory.result_count``         — number of chunks / documents returned
        - ``knowledge.content_preview``   — full retrieved text (string results)
        - ``knowledge.results_preview``   — full JSON-serialized results (structured results)
        - ``knowledge.score_min`` / ``max`` / ``scores`` — relevance rollups and
          per-result scores (JSON list, ``null`` when a row has no score)
        - ``knowledge.sources_used``      — JSON list of sources that returned results
        - ``memory.status``               — ``"ok"``
        - ``memory.duration_ms``          — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            op_id = _resolve_op_id(event, source)
            extra: dict[str, Any] = {}

            results = (
                safe_getattr(event, "results")
                or safe_getattr(event, "content")
                or safe_getattr(event, "chunks")
            )
            if results is not None:
                _enrich_results(extra, results)
                _enrich_scores(extra, results)

            sources_used = safe_getattr(event, "sources_used") or safe_getattr(
                event, "matched_sources"
            )
            if sources_used is not None:
                extra["knowledge.sources_used"] = safe_json_dumps(sources_used)

            self._finish_knowledge_span(op_id, ATTR_STATUS_SUCCESS, None, extra)
        except Exception:
            logger.debug(
                "on_knowledge_retrieval_completed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # QUERY — started / completed / failed
    # =========================================================================

    def on_knowledge_query_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.memory_op`` span for a targeted knowledge query.

        A query is a single retrieval call with an explicit search string,
        as opposed to the bulk retrieval that hydrates all relevant knowledge
        for a task.

        Attributes set at span open
        ---------------------------
        - ``memory.op_id``          — unique operation identifier
        - ``memory.type``           — ``"knowledge"``
        - ``memory.operation``      — ``"query"``
        - ``memory.query``          — search query text (≤ MAX_DESCRIPTION_LENGTH)
        - ``knowledge.sources``     — sources scoped to this query (filter)
        - ``knowledge.top_k``       — maximum number of results requested
        - ``agent.role``            — executing agent's role (correlation)
        """
        if not self._is_active():
            return
        try:
            op_id = _resolve_op_id(event, source)
            agent_id = _resolve_agent_id(source, event)

            attrs: dict[str, Any] = {
                ATTR_MEMORY_OP_ID: op_id,
                ATTR_MEMORY_TYPE: _KNOWLEDGE_TYPE,
                ATTR_MEMORY_OPERATION: _OP_QUERY,
            }

            query = (
                safe_getattr(event, "query")
                or safe_getattr(event, "search_query")
                or safe_getattr(event, "text")
                or safe_getattr(event, "input")
            )
            if query:
                attrs[ATTR_MEMORY_QUERY] = truncate_str(
                    str(query), MAX_DESCRIPTION_LENGTH
                )

            sources = safe_getattr(event, "sources") or safe_getattr(
                event, "knowledge_sources"
            )
            if sources is not None:
                attrs["knowledge.sources"] = safe_json_dumps(sources)

            top_k = safe_getattr(event, "top_k") or safe_getattr(event, "limit")
            if top_k is not None:
                try:
                    attrs["knowledge.top_k"] = int(top_k)
                except (TypeError, ValueError):
                    pass

            agent_role = safe_getattr(event, "agent_role") or safe_getattr(
                source, "role"
            )
            if agent_role:
                attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

            self._open_knowledge_span(op_id, agent_id, attrs)
            logger.debug("Knowledge query span opened: op_id=%s", op_id)
        except Exception:
            logger.debug(
                "on_knowledge_query_started error:\n%s", traceback.format_exc()
            )

    def on_knowledge_query_completed(self, source: Any, event: Any) -> None:
        """
        Close the knowledge query span as OK.

        Attributes written
        ------------------
        - ``memory.result_count``       — number of results returned
        - ``knowledge.content_preview`` — full retrieved text (string results)
        - ``knowledge.results_preview`` — full JSON-serialized results (structured results)
        - ``knowledge.score_min``       — lowest relevance score in results
        - ``knowledge.score_max``       — highest relevance score in results
        - ``knowledge.scores``          — JSON list of scores, one per result in order
                                          (``null`` entries when a row has no ``score``)
        - ``memory.status``             — ``"ok"``
        - ``memory.duration_ms``        — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            op_id = _resolve_op_id(event, source)
            extra: dict[str, Any] = {}

            results = (
                safe_getattr(event, "results")
                or safe_getattr(event, "chunks")
                or safe_getattr(event, "documents")
            )
            if results is not None:
                _enrich_results(extra, results)
                _enrich_scores(extra, results)

            self._finish_knowledge_span(op_id, ATTR_STATUS_SUCCESS, None, extra)
        except Exception:
            logger.debug(
                "on_knowledge_query_completed error:\n%s", traceback.format_exc()
            )

    def on_knowledge_query_failed(self, source: Any, event: Any) -> None:
        """Close the knowledge query span as ERROR."""
        if not self._is_active():
            return
        try:
            op_id = _resolve_op_id(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_knowledge_span(op_id, ATTR_STATUS_ERROR, error)
        except Exception:
            logger.debug("on_knowledge_query_failed error:\n%s", traceback.format_exc())

    # =========================================================================
    # Search-query failure (search-layer variant)
    # =========================================================================

    def on_knowledge_search_query_failed(self, source: Any, event: Any) -> None:
        """
        Annotate the nearest open knowledge / agent span with a search error.

        ``KnowledgeSearchQueryFailed`` fires from the search layer (vector DB,
        embedding lookup, etc.) which may be *below* the query span.  If the
        query span is already closed or was never opened, the error is written
        to the agent span so it is not silently dropped.

        Attributes written
        ------------------
        - ``knowledge.search_error``       — error message string
        - ``knowledge.search_error.type``  — exception class name
        - ``knowledge.search_query``       — the query that failed (if available)
        """
        if not self._is_active():
            return
        try:
            op_id = _resolve_op_id(event, source)
            agent_id = _resolve_agent_id(source, event)

            # Prefer open knowledge span; fall back to agent span
            span = self._get_knowledge_or_agent_span(op_id, agent_id)
            if span is None:
                logger.debug(
                    "on_knowledge_search_query_failed: no open span "
                    "for op_id=%s agent_id=%s — error dropped",
                    op_id,
                    agent_id,
                )
                return

            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            error_str = (
                str(error) if error else str(safe_getattr(event, "message") or "")
            )

            search_attrs: dict[str, Any] = {}
            if error_str:
                search_attrs["knowledge.search_error"] = truncate_str(error_str, 1024)
            if error is not None:
                search_attrs["knowledge.search_error.type"] = type(error).__name__
                tb = getattr(error, "__traceback__", None)
                if tb:
                    search_attrs[ATTR_ERROR_STACKTRACE] = "".join(
                        traceback.format_tb(tb)
                    )

            query = safe_getattr(event, "query") or safe_getattr(event, "search_query")
            if query:
                search_attrs["knowledge.search_query"] = truncate_str(
                    str(query), MAX_DESCRIPTION_LENGTH
                )

            set_span_attributes(span, search_attrs)
            logger.debug(
                "knowledge.search_error attached: op_id=%s agent_id=%s",
                op_id,
                agent_id,
            )
        except Exception:
            logger.debug(
                "on_knowledge_search_query_failed error:\n%s",
                traceback.format_exc(),
            )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _open_knowledge_span(
        self,
        op_id: str,
        agent_id: Optional[str],
        attrs: dict[str, Any],
    ) -> None:
        """Create a knowledge ``crewai.knowledge`` span and register it."""
        start_t = monotonic_now()
        parent_span = self._best_parent_for_knowledge(agent_id)

        span = self._create_child_span(
            SPAN_KNOWLEDGE,
            parent_span=parent_span,
            attributes=attrs,
        )

        with self._lock:
            self._memory_op_spans[op_id] = span
            self._memory_op_start_times[op_id] = start_t

    def _best_parent_for_knowledge(self, agent_id: Optional[str]) -> Any:
        """
        Return the most contextually appropriate parent span.

        Priority: agent span → None.
        Knowledge ops are agent-initiated (an agent queries its knowledge base
        for the current task).
        """
        with self._lock:
            if agent_id and agent_id in self._agent_spans:
                return self._agent_spans[agent_id]
            logger.debug(
                "_best_parent_for_knowledge: no matching agent span for agent_id=%s",
                agent_id,
            )
        return None

    def _get_knowledge_or_agent_span(self, op_id: str, agent_id: Optional[str]) -> Any:
        """Return open knowledge span for *op_id*, else agent span."""
        with self._lock:
            span = self._memory_op_spans.get(op_id)
            if span is not None:
                return span
            if agent_id:
                return self._agent_spans.get(agent_id)
        return None

    def _finish_knowledge_span(
        self,
        op_id: str,
        status: str,
        error: Any,
        extra_attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write final attributes onto the knowledge span and close it."""
        with self._lock:
            span = self._memory_op_spans.pop(op_id, None)
            start_t = self._memory_op_start_times.pop(op_id, None)

        if span is None:
            logger.debug("_finish_knowledge_span: no open span for op_id=%s", op_id)
            return

        finish_span_common(
            span,
            start_t=start_t,
            status=status,
            status_attr=ATTR_MEMORY_STATUS,
            duration_attr=ATTR_MEMORY_DURATION_MS,
            error=error,
            extra_attrs=extra_attrs,
            log_label="_finish_knowledge_span",
        )

        logger.debug("Knowledge span closed: op_id=%s status=%s", op_id, status)


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _resolve_op_id(event: Any, source: Any) -> str:
    """Return a stable string key for pairing knowledge started ↔ completed/failed.

    Same contract as memory events: CrewAI sets ``started_event_id`` on closing
    events (see ``crewai.events.event_bus``). Do not use ``id(event)`` before
    ``event_id`` or pairing breaks and spans live until ``Trace.finish()``.
    """
    started_link = safe_getattr(event, "started_event_id")
    if isinstance(started_link, str) and started_link.strip():
        return started_link
    return str(
        safe_getattr(event, "knowledge_op_id")
        or safe_getattr(event, "op_id")
        or safe_getattr(event, "query_id")
        or safe_getattr(event, "run_id")
        or safe_getattr(event, "event_id")
        or id(event)
    )


def _enrich_results(attrs: dict[str, Any], results: Any) -> None:
    """
    Write ``memory.result_count`` and the full retrieved payload into *attrs*.

    String results go to ``knowledge.content_preview``; structured results to
    ``knowledge.results_preview``. Neither field is truncated.

    Handles list/tuple of chunks, a plain string, or an unknown object
    without raising.
    """
    try:
        if isinstance(results, str):
            attrs["memory.result_count"] = 1
            attrs["knowledge.content_preview"] = results
            return

        if hasattr(results, "__len__"):
            attrs[ATTR_MEMORY_RESULT_COUNT] = len(results)

        serialized = safe_json_dumps(results)
        if serialized:
            attrs["knowledge.results_preview"] = serialized
    except Exception as exc:
        logger.debug("_enrich_results failed: %s", exc)


def _enrich_scores(attrs: dict[str, Any], results: Any) -> None:
    """
    Extract relevance scores from the results list and write min/max plus
    ``knowledge.scores`` — a JSON array aligned with *results* order.

    CrewAI knowledge results are typically
    ``[{"content": "...", "score": 0.87}, ...]``
    but the exact schema varies by knowledge source.  Rows without a numeric
    ``score`` become JSON ``null`` so indices still line up with ``results``.
    """
    try:
        if not isinstance(results, (list, tuple)):
            return
        scores: list[float] = []
        per_result: list[Optional[float]] = []
        for item in results:
            raw = safe_getattr(item, "score") or (
                item.get("score") if isinstance(item, dict) else None
            )
            if raw is None:
                per_result.append(None)
                continue
            try:
                f = float(raw)
                scores.append(f)
                per_result.append(f)
            except (TypeError, ValueError):
                per_result.append(None)
        if per_result and any(s is not None for s in per_result):
            attrs["knowledge.scores"] = safe_json_dumps(per_result)
        if scores:
            attrs["knowledge.score_min"] = round(min(scores), 6)
            attrs["knowledge.score_max"] = round(max(scores), 6)
    except Exception as exc:
        logger.debug("_enrich_scores failed: %s", exc)
