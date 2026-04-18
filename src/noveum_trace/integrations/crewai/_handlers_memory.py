"""
Memory-operation event handler mixin for NoveumCrewAIListener.

Handles CrewAI ``BaseEventListener`` memory events across all three memory
subsystems: short-term, long-term, and entity memory.

Each operation family follows the same started → completed/failed lifecycle:

  Query (read / search):
  - ``on_memory_query_started``     → open ``crewai.memory_op`` span;
                                       capture memory_type, query text, agent_role
  - ``on_memory_query_completed``   → close as SUCCESS; write results_count,
                                       duration_ms, serialised results preview
  - ``on_memory_query_failed``      → close as ERROR

  Save (write / store):
  - ``on_memory_save_started``      → open ``crewai.memory_op`` span;
                                       capture memory_type, value, metadata
  - ``on_memory_save_completed``    → close as SUCCESS; write duration_ms
  - ``on_memory_save_failed``       → close as ERROR

  Retrieval (bulk context pull — distinct from single-item query):
  - ``on_memory_retrieval_started``  → open ``crewai.memory_op`` span;
                                        capture task_id (the requesting task)
  - ``on_memory_retrieval_completed``→ close as SUCCESS; write memory_content
                                        preview and duration_ms
  - ``on_memory_retrieval_failed``   → close as ERROR

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
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_STACKTRACE,
    ATTR_ERROR_TYPE,
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
    SPAN_MEMORY_QUERY,
    SPAN_MEMORY_RETRIEVAL,
    SPAN_MEMORY_SAVE,
)
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase
from noveum_trace.integrations.crewai.crewai_utils import (
    duration_ms_monotonic,
    monotonic_now,
    safe_getattr,
    safe_json_dumps,
    truncate_str,
)

logger = logging.getLogger(__name__)

# Memory operation type strings used for ATTR_MEMORY_OPERATION
_OP_QUERY = "query"
_OP_SAVE = "save"
_OP_RETRIEVAL = "retrieval"

# Preview lengths — kept short; full results can be large
_MEMORY_VALUE_PREVIEW_LEN = 1_024
_MEMORY_RESULTS_PREVIEW_LEN = 2_048


class _MemoryHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI memory operation events.

    All public methods match the ``BaseEventListener`` callback signature::

        def on_memory_query_started(self, source, event): ...

    ``source`` is typically the memory object or the Agent; ``event`` carries
    the per-operation payload.  Every method is fully exception-shielded.

    All handlers no-op when ``capture_memory`` is ``False`` on the listener.
    """

    # =========================================================================
    # QUERY — started / completed / failed
    # =========================================================================

    def on_memory_query_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.memory_op`` span for a memory search/query operation.

        Attributes set at span open
        ---------------------------
        - ``memory.op_id``     — unique operation identifier
        - ``memory.type``      — ``"short_term"`` | ``"long_term"`` | ``"entity"``
                                 | ``"external"`` | ``"unknown"``
        - ``memory.operation`` — ``"query"``
        - ``memory.query``     — search query text (≤ MAX_DESCRIPTION_LENGTH)
        - ``agent.role``       — role of the querying agent (correlation)
        """
        if not self._is_active() or not self.capture_memory:
            return
        try:
            op_id = _resolve_op_id(event, source)
            memory_type = _resolve_memory_type(source, event)
            agent_id = _resolve_agent_id(source, event)

            attrs: dict[str, Any] = {
                ATTR_MEMORY_OP_ID: op_id,
                ATTR_MEMORY_TYPE: memory_type,
                ATTR_MEMORY_OPERATION: _OP_QUERY,
            }

            query = (
                safe_getattr(event, "query")
                or safe_getattr(event, "search_query")
                or safe_getattr(event, "text")
            )
            if query:
                attrs[ATTR_MEMORY_QUERY] = truncate_str(
                    str(query), MAX_DESCRIPTION_LENGTH
                )

            agent_role = safe_getattr(event, "agent_role") or safe_getattr(
                source, "role"
            )
            if agent_role:
                attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

            # memory.limit — max results requested
            limit = safe_getattr(event, "limit") or safe_getattr(event, "top_k")
            if limit is not None:
                try:
                    attrs["memory.limit"] = int(limit)
                except (TypeError, ValueError):
                    pass

            # memory.score_threshold — minimum relevance score requested
            score_threshold = (
                safe_getattr(event, "score_threshold")
                or safe_getattr(event, "threshold")
                or safe_getattr(event, "min_score")
            )
            if score_threshold is not None:
                try:
                    attrs["memory.score_threshold"] = float(score_threshold)
                except (TypeError, ValueError):
                    pass

            self._open_memory_span(op_id, agent_id, attrs, span_name=SPAN_MEMORY_QUERY)
            logger.debug(
                "Memory query span opened: op_id=%s type=%s", op_id, memory_type
            )
        except Exception:
            logger.debug("on_memory_query_started error:\n%s", traceback.format_exc())

    def on_memory_query_completed(self, source: Any, event: Any) -> None:
        """
        Close the memory query span as SUCCESS.

        Attributes written
        ------------------
        - ``memory.result_count``    — number of results returned
        - ``memory.results_preview`` — truncated JSON preview of results
        - ``memory.status``          — ``"success"``
        - ``memory.duration_ms``     — wall-clock duration
        """
        if not self._is_active() or not self.capture_memory:
            return
        try:
            op_id = _resolve_op_id(event, source)
            extra: dict[str, Any] = {}

            results = safe_getattr(event, "results") or safe_getattr(event, "memories")
            if results is not None:
                try:
                    count = len(results) if hasattr(results, "__len__") else None
                    if count is not None:
                        extra[ATTR_MEMORY_RESULT_COUNT] = count
                    preview = safe_json_dumps(results)
                    if preview:
                        extra["memory.results_preview"] = truncate_str(
                            preview, _MEMORY_RESULTS_PREVIEW_LEN
                        )
                except Exception:
                    pass

            self._finish_memory_span(op_id, ATTR_STATUS_SUCCESS, None, extra)
        except Exception:
            logger.debug("on_memory_query_completed error:\n%s", traceback.format_exc())

    def on_memory_query_failed(self, source: Any, event: Any) -> None:
        """Close the memory query span as ERROR."""
        if not self._is_active() or not self.capture_memory:
            return
        try:
            op_id = _resolve_op_id(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_memory_span(op_id, ATTR_STATUS_ERROR, error)
        except Exception:
            logger.debug("on_memory_query_failed error:\n%s", traceback.format_exc())

    # =========================================================================
    # SAVE — started / completed / failed
    # =========================================================================

    def on_memory_save_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.memory_op`` span for a memory write/save operation.

        Attributes set at span open
        ---------------------------
        - ``memory.op_id``        — unique operation identifier
        - ``memory.type``         — memory subsystem type
        - ``memory.operation``    — ``"save"``
        - ``memory.value``        — the value being stored (preview, truncated)
        - ``memory.metadata``     — JSON of associated metadata when available
        - ``agent.role``          — role of the saving agent (correlation)
        """
        if not self._is_active() or not self.capture_memory:
            return
        try:
            op_id = _resolve_op_id(event, source)
            memory_type = _resolve_memory_type(source, event)
            agent_id = _resolve_agent_id(source, event)

            attrs: dict[str, Any] = {
                ATTR_MEMORY_OP_ID: op_id,
                ATTR_MEMORY_TYPE: memory_type,
                ATTR_MEMORY_OPERATION: _OP_SAVE,
            }

            value = (
                safe_getattr(event, "value")
                or safe_getattr(event, "content")
                or safe_getattr(event, "memory")
            )
            if value is not None:
                if isinstance(value, str):
                    attrs["memory.value"] = truncate_str(
                        value, _MEMORY_VALUE_PREVIEW_LEN
                    )
                else:
                    attrs["memory.value"] = truncate_str(
                        safe_json_dumps(value), _MEMORY_VALUE_PREVIEW_LEN
                    )

            metadata = safe_getattr(event, "metadata") or safe_getattr(event, "meta")
            if metadata is not None:
                attrs["memory.metadata"] = truncate_str(safe_json_dumps(metadata), 512)

            agent_role = safe_getattr(event, "agent_role") or safe_getattr(
                source, "role"
            )
            if agent_role:
                attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

            self._open_memory_span(op_id, agent_id, attrs, span_name=SPAN_MEMORY_SAVE)
            logger.debug(
                "Memory save span opened: op_id=%s type=%s", op_id, memory_type
            )
        except Exception:
            logger.debug("on_memory_save_started error:\n%s", traceback.format_exc())

    def on_memory_save_completed(self, source: Any, event: Any) -> None:
        """Close the memory save span as SUCCESS."""
        if not self._is_active() or not self.capture_memory:
            return
        try:
            op_id = _resolve_op_id(event, source)
            self._finish_memory_span(op_id, ATTR_STATUS_SUCCESS, None)
        except Exception:
            logger.debug("on_memory_save_completed error:\n%s", traceback.format_exc())

    def on_memory_save_failed(self, source: Any, event: Any) -> None:
        """Close the memory save span as ERROR."""
        if not self._is_active() or not self.capture_memory:
            return
        try:
            op_id = _resolve_op_id(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_memory_span(op_id, ATTR_STATUS_ERROR, error)
        except Exception:
            logger.debug("on_memory_save_failed error:\n%s", traceback.format_exc())

    # =========================================================================
    # RETRIEVAL — started / completed / failed
    # =========================================================================

    def on_memory_retrieval_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.memory_op`` span for a bulk context retrieval.

        Retrieval is distinct from query: CrewAI fires this when it pulls the
        full relevant memory context for a task (not a targeted search).  It
        may span multiple memory subsystems internally.

        Attributes set at span open
        ---------------------------
        - ``memory.op_id``        — unique operation identifier
        - ``memory.type``         — memory subsystem or ``"all"``
        - ``memory.operation``    — ``"retrieval"``
        - ``task.id``             — id of the requesting task (correlation)
        - ``agent.role``          — role of the agent context being hydrated
        """
        if not self._is_active() or not self.capture_memory:
            return
        try:
            op_id = _resolve_op_id(event, source)
            memory_type = _resolve_memory_type(source, event, default="all")
            agent_id = _resolve_agent_id(source, event)

            attrs: dict[str, Any] = {
                ATTR_MEMORY_OP_ID: op_id,
                ATTR_MEMORY_TYPE: memory_type,
                ATTR_MEMORY_OPERATION: _OP_RETRIEVAL,
            }

            task_id = (
                safe_getattr(event, "task_id")
                or safe_getattr(event, "id")
                or safe_getattr(safe_getattr(source, "task"), "id")
            )
            if task_id:
                attrs["task.id"] = str(task_id)
                attrs["memory.task_id"] = str(task_id)  # spec key

            agent_role = safe_getattr(event, "agent_role") or safe_getattr(
                source, "role"
            )
            if agent_role:
                attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

            self._open_memory_span(
                op_id, agent_id, attrs, span_name=SPAN_MEMORY_RETRIEVAL
            )
            logger.debug(
                "Memory retrieval span opened: op_id=%s type=%s", op_id, memory_type
            )
        except Exception:
            logger.debug(
                "on_memory_retrieval_started error:\n%s", traceback.format_exc()
            )

    def on_memory_retrieval_completed(self, source: Any, event: Any) -> None:
        """
        Close the memory retrieval span as SUCCESS.

        Attributes written
        ------------------
        - ``memory.content_preview`` — truncated preview of retrieved memory context
        - ``memory.result_count``    — number of memory items retrieved (when countable)
        - ``memory.status``          — ``"success"``
        - ``memory.duration_ms``     — wall-clock duration
        """
        if not self._is_active() or not self.capture_memory:
            return
        try:
            op_id = _resolve_op_id(event, source)
            extra: dict[str, Any] = {}

            memory_content = (
                safe_getattr(event, "memory_content")
                or safe_getattr(event, "context")
                or safe_getattr(event, "memories")
                or safe_getattr(event, "results")
            )
            if memory_content is not None:
                if isinstance(memory_content, str):
                    extra["memory.content_preview"] = truncate_str(
                        memory_content, _MEMORY_RESULTS_PREVIEW_LEN
                    )
                else:
                    try:
                        count = (
                            len(memory_content)
                            if hasattr(memory_content, "__len__")
                            else None
                        )
                        if count is not None:
                            extra[ATTR_MEMORY_RESULT_COUNT] = count
                    except Exception:
                        pass
                    extra["memory.content_preview"] = truncate_str(
                        safe_json_dumps(memory_content), _MEMORY_RESULTS_PREVIEW_LEN
                    )

            self._finish_memory_span(op_id, ATTR_STATUS_SUCCESS, None, extra)
        except Exception:
            logger.debug(
                "on_memory_retrieval_completed error:\n%s", traceback.format_exc()
            )

    def on_memory_retrieval_failed(self, source: Any, event: Any) -> None:
        """Close the memory retrieval span as ERROR."""
        if not self._is_active() or not self.capture_memory:
            return
        try:
            op_id = _resolve_op_id(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_memory_span(op_id, ATTR_STATUS_ERROR, error)
        except Exception:
            logger.debug(
                "on_memory_retrieval_failed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _open_memory_span(
        self,
        op_id: str,
        agent_id: Optional[str],
        attrs: dict[str, Any],
        span_name: str = SPAN_MEMORY_QUERY,
    ) -> None:
        """Create a memory span with the given *span_name* and register it in state."""
        start_t = monotonic_now()

        # Best parent: agent span → task span (any open task) → None
        parent_span = self._best_parent_for_memory(agent_id)

        span = self._create_child_span(
            span_name,
            parent_span=parent_span,
            attributes=attrs,
        )

        with self._lock:
            self._memory_op_spans[op_id] = span
            self._memory_op_start_times[op_id] = start_t

    def _best_parent_for_memory(self, agent_id: Optional[str]) -> Any:
        """
        Return the most contextually appropriate parent span for a memory op.

        Priority: agent span → first open task span → None.
        """
        with self._lock:
            if agent_id and agent_id in self._agent_spans:
                return self._agent_spans[agent_id]
            # Pick any open task span (memory is task-scoped)
            for span in self._task_spans.values():
                return span
        return None

    def _finish_memory_span(
        self,
        op_id: str,
        status: str,
        error: Any,
        extra_attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write final attributes onto the memory op span and close it."""
        with self._lock:
            span = self._memory_op_spans.pop(op_id, None)
            start_t = self._memory_op_start_times.pop(op_id, None)

        if span is None:
            logger.debug("_finish_memory_span: no open span for op_id=%s", op_id)
            return

        attrs: dict[str, Any] = {ATTR_MEMORY_STATUS: status}

        if start_t is not None:
            attrs[ATTR_MEMORY_DURATION_MS] = duration_ms_monotonic(start_t)

        if error is not None:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = str(error)
            tb = getattr(error, "__traceback__", None)
            if tb is not None:
                attrs[ATTR_ERROR_STACKTRACE] = "".join(traceback.format_tb(tb))

        if extra_attrs:
            attrs.update(extra_attrs)

        try:
            if hasattr(span, "set_attributes"):
                span.set_attributes(attrs)
            elif hasattr(span, "attributes"):
                span.attributes.update(attrs)

            if status == ATTR_STATUS_ERROR and hasattr(span, "set_status"):
                try:
                    from noveum_trace.core.span import SpanStatus

                    span.set_status(SpanStatus.ERROR, str(error) if error else "")
                except Exception:
                    pass

            if hasattr(span, "finish"):
                span.finish()
        except Exception:
            logger.debug(
                "_finish_memory_span span.finish() error:\n%s",
                traceback.format_exc(),
            )

        logger.debug("Memory op span closed: op_id=%s status=%s", op_id, status)


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _resolve_op_id(event: Any, source: Any) -> str:
    """Return a stable string key for pairing memory started ↔ completed/failed.

    CrewAI's event bus assigns ``started_event_id`` on scope-ending events to the
    ``event_id`` of the matching ``*_started`` event (see ``event_bus._prepare_event``).

    Falling back to Python ``id(event)`` **breaks** pairing because ``started`` and
    ``completed`` are different event instances: the completed handler never finds the
    open span, so the span stays in ``Trace.active_spans`` until ``Trace.finish()``,
    which force-finishes it with the **trace** end time — producing multi‑minute
    ``memory.*`` durations that track the whole kickoff instead of the real op.
    """
    started_link = safe_getattr(event, "started_event_id")
    if isinstance(started_link, str) and started_link.strip():
        return started_link
    return str(
        safe_getattr(event, "memory_op_id")
        or safe_getattr(event, "op_id")
        or safe_getattr(event, "run_id")
        or safe_getattr(event, "event_id")
        or id(event)
    )


def _resolve_agent_id(source: Any, event: Any) -> Optional[str]:
    """Return the agent_id associated with this memory event, or ``None``."""
    raw = (
        safe_getattr(event, "agent_id")
        or safe_getattr(source, "id")
        or safe_getattr(source, "agent_id")
    )
    return str(raw) if raw is not None else None


def _resolve_memory_type(source: Any, event: Any, default: str = "unknown") -> str:
    """
    Resolve the memory subsystem type from event, source class name, or default.

    Maps raw strings to canonical values:
      ``"short_term"``  | ``"long_term"`` | ``"entity"`` |
      ``"external"``    | ``"all"``       | ``"unknown"``
    """
    raw = (
        safe_getattr(event, "memory_type")
        or safe_getattr(event, "type")
        or safe_getattr(source, "memory_type")
        or safe_getattr(source, "type")
    )
    if raw is None:
        # Infer from class name of source
        class_name = type(source).__name__.lower()
        if "short" in class_name:
            return "short_term"
        if "long" in class_name:
            return "long_term"
        if "entity" in class_name:
            return "entity"
        if "external" in class_name:
            return "external"
        return default

    raw_str = str(raw).lower().replace("-", "_").replace(" ", "_")
    for canonical in ("short_term", "long_term", "entity", "external", "all"):
        if canonical in raw_str:
            return canonical
    return raw_str or default
