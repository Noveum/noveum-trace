"""
Agent-lifecycle event handler mixin for NoveumCrewAIListener.

Handles CrewAI ``BaseEventListener`` agent events:

  Regular agents (CrewAI full Agent):
  - ``on_agent_execution_started``   → open ``crewai.agent`` child span under the
                                        current task span; capture role, goal, backstory,
                                        agent_id, task_prompt, serialized tool
                                        names+descriptions, max_iter, allow_delegation
  - ``on_agent_execution_completed`` → close span as SUCCESS; write output text
                                        and iteration count
  - ``on_agent_execution_error``     → close span as ERROR; attach exception details

  LiteAgent (CrewAI lightweight agent):
  - ``on_lite_agent_started``        → same open pattern, tagged ``agent.type=lite``
  - ``on_lite_agent_completed``      → close as SUCCESS
  - ``on_lite_agent_error``          → close as ERROR

  Agent evaluation:
  - ``on_agent_evaluation_started``  → attach evaluator model name
  - ``on_agent_evaluation_completed``→ attach numeric quality score + feedback
  - ``on_agent_evaluation_error``    → attach error to existing span

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown,
    _task_spans, _agent_spans, _agent_start_times
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_AGENT_ALLOW_DELEGATION,
    ATTR_AGENT_BACKSTORY,
    ATTR_AGENT_DURATION_MS,
    ATTR_AGENT_GOAL,
    ATTR_AGENT_ID,
    ATTR_AGENT_LLM_MODEL,
    ATTR_AGENT_MAX_ITER,
    ATTR_AGENT_MAX_RPM,
    ATTR_AGENT_ROLE,
    ATTR_AGENT_STATUS,
    ATTR_AGENT_TOOL_NAMES,
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_STACKTRACE,
    ATTR_ERROR_TYPE,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    MAX_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
    SPAN_AGENT,
)
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase
from noveum_trace.integrations.crewai.crewai_utils import (
    duration_ms_monotonic,
    extract_llm_model_from_agent,
    merge_available_tools_attributes,
    monotonic_now,
    safe_getattr,
    safe_json_dumps,
    truncate_str,
)

logger = logging.getLogger(__name__)


class _AgentHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI Agent-lifecycle events (full Agent + LiteAgent).

    All public methods match ``BaseEventListener`` callback signature::

        def on_agent_execution_started(self, source, event): ...

    ``source`` is the ``Agent`` / ``LiteAgent`` object; ``event`` carries the
    event payload.  Every method is fully exception-shielded.
    """

    # =========================================================================
    # Full Agent — started
    # =========================================================================

    def on_agent_execution_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.agent`` span as a child of the owning task span.

        Attributes captured at span-open time
        --------------------------------------
        - ``agent.id``               — agent UUID
        - ``agent.role``             — agent role string
        - ``agent.goal``             — agent goal (≤ MAX_DESCRIPTION_LENGTH)
        - ``agent.backstory``        — agent backstory (≤ MAX_DESCRIPTION_LENGTH)
        - ``agent.llm_model``        — backing LLM model name
        - ``agent.tool_names``       — JSON list of tool names
        - ``agent.tools``            — JSON list of {name, description} dicts
        - ``agent.available_tools.*``— count, names, descriptions, schemas JSON
        - ``agent.allow_delegation`` — bool
        - ``agent.max_iter``         — max reasoning iterations
        - ``agent.max_rpm``          — max requests per minute
        - ``agent.type``             — ``"full"``
        - ``agent.task_prompt``      — compiled prompt the agent received
                                       (≤ MAX_TEXT_LENGTH)
        """
        if not self._is_active():
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            task_id = _resolve_task_id(source, event)
            attrs = _build_agent_attributes(source, event, agent_type="full")
            self._open_agent_span(agent_id, task_id, attrs)
        except Exception:
            logger.debug(
                "on_agent_execution_started error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Full Agent — completed
    # =========================================================================

    def on_agent_execution_completed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.agent`` span as SUCCESS.

        Attributes written
        ------------------
        - ``agent.output``         — final agent output text (≤ MAX_TEXT_LENGTH)
        - ``agent.step``           — number of reasoning iterations used
        - ``agent.status``         — ``"success"``
        - ``agent.duration_ms``    — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            output = safe_getattr(event, "output")
            iterations = safe_getattr(event, "iterations") or safe_getattr(
                event, "step"
            )
            extra: dict[str, Any] = {}
            if iterations is not None:
                try:
                    extra["agent.iterations"] = int(iterations)
                except (TypeError, ValueError):
                    pass
            self._finish_agent_span(
                agent_id=agent_id,
                status=ATTR_STATUS_SUCCESS,
                output=output,
                error=None,
                extra_attrs=extra,
            )
        except Exception:
            logger.debug(
                "on_agent_execution_completed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Full Agent — error
    # =========================================================================

    def on_agent_execution_error(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.agent`` span as ERROR.

        Attributes written
        ------------------
        - ``error.type``        — exception class name
        - ``error.message``     — exception message
        - ``error.stacktrace``  — formatted traceback
        - ``agent.status``      — ``"error"``
        - ``agent.duration_ms`` — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_agent_span(
                agent_id=agent_id,
                status=ATTR_STATUS_ERROR,
                output=None,
                error=error,
            )
        except Exception:
            logger.debug("on_agent_execution_error error:\n%s", traceback.format_exc())

    # =========================================================================
    # LiteAgent — started / completed / error
    # =========================================================================

    def on_lite_agent_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.agent`` span for a LiteAgent execution.

        Same attribute set as ``on_agent_execution_started`` plus
        ``agent.type = "lite"``.
        """
        if not self._is_active():
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            task_id = _resolve_task_id(source, event)
            attrs = _build_agent_attributes(source, event, agent_type="lite")
            self._open_agent_span(agent_id, task_id, attrs)
        except Exception:
            logger.debug("on_lite_agent_started error:\n%s", traceback.format_exc())

    def on_lite_agent_completed(self, source: Any, event: Any) -> None:
        """Close the LiteAgent span as SUCCESS."""
        if not self._is_active():
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            output = safe_getattr(event, "output")
            self._finish_agent_span(
                agent_id=agent_id,
                status=ATTR_STATUS_SUCCESS,
                output=output,
                error=None,
            )
        except Exception:
            logger.debug("on_lite_agent_completed error:\n%s", traceback.format_exc())

    def on_lite_agent_error(self, source: Any, event: Any) -> None:
        """Close the LiteAgent span as ERROR."""
        if not self._is_active():
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_agent_span(
                agent_id=agent_id,
                status=ATTR_STATUS_ERROR,
                output=None,
                error=error,
            )
        except Exception:
            logger.debug("on_lite_agent_error error:\n%s", traceback.format_exc())

    # =========================================================================
    # Agent evaluation
    # =========================================================================

    def on_agent_evaluation_started(self, source: Any, event: Any) -> None:
        """
        Annotate the agent span with evaluation metadata at evaluation start.

        Attributes written
        ------------------
        - ``agent.evaluation.model``    — LLM used as evaluator judge
        - ``agent.evaluation.criteria`` — JSON list of evaluation criteria names
        """
        if not self._is_active():
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            span = self._get_agent_span(agent_id)
            if span is None:
                return

            eval_attrs: dict[str, Any] = {}
            model = safe_getattr(event, "model") or safe_getattr(
                event, "evaluator_model"
            )
            if model:
                eval_attrs["agent.evaluation.model"] = str(model)

            criteria = safe_getattr(event, "criteria")
            if criteria:
                try:
                    if isinstance(criteria, (list, tuple)):
                        eval_attrs["agent.evaluation.criteria"] = safe_json_dumps(
                            list(criteria)
                        )
                    else:
                        eval_attrs["agent.evaluation.criteria"] = str(criteria)
                except Exception:
                    pass

            _set_span_attributes(span, eval_attrs)
        except Exception:
            logger.debug(
                "on_agent_evaluation_started error:\n%s", traceback.format_exc()
            )

    def on_agent_evaluation_completed(self, source: Any, event: Any) -> None:
        """
        Attach quality score and feedback to the agent span.

        Attributes written
        ------------------
        - ``agent.evaluation.score``    — float quality score
        - ``agent.evaluation.feedback`` — evaluator text feedback (≤ 2048 chars)
        - ``agent.evaluation.passed``   — bool (when the evaluator returns a pass/fail)
        """
        if not self._is_active():
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            span = self._get_agent_span(agent_id)
            if span is None:
                # May have already closed; best-effort attach to task/crew span
                task_id = _resolve_task_id(source, event)
                with self._lock:
                    span = (
                        self._task_spans.get(task_id) if task_id is not None else None
                    )

            if span is None:
                return

            eval_attrs = _extract_agent_evaluation_attributes(event)
            _set_span_attributes(span, eval_attrs)
        except Exception:
            logger.debug(
                "on_agent_evaluation_completed error:\n%s", traceback.format_exc()
            )

    def on_agent_evaluation_error(self, source: Any, event: Any) -> None:
        """Attach evaluation error information to the agent span."""
        if not self._is_active():
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            span = self._get_agent_span(agent_id)
            if span is None:
                return

            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            if error:
                _set_span_attributes(
                    span,
                    {
                        "agent.evaluation.error": str(error),
                        "agent.evaluation.error_type": type(error).__name__,
                    },
                )
        except Exception:
            logger.debug("on_agent_evaluation_error error:\n%s", traceback.format_exc())

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _open_agent_span(
        self,
        agent_id: str,
        task_id: Optional[str],
        attrs: dict[str, Any],
    ) -> None:
        """Create a ``crewai.agent`` span and register it in state."""
        start_t = monotonic_now()

        # Parent: task span if available, else None (span will link to trace root)
        parent_span: Any = None
        if task_id:
            with self._lock:
                parent_span = self._task_spans.get(task_id)

        span = self._create_child_span(
            SPAN_AGENT,
            parent_span=parent_span,
            attributes=attrs,
            # Provide task_id so _create_child_span can look up the owning
            # crew via _task_to_crew_id when two crew lifetimes overlap.
            task_id=task_id or None,
        )

        with self._lock:
            self._agent_spans[agent_id] = span
            self._agent_start_times[agent_id] = start_t

        logger.debug("Agent span opened: agent_id=%s task_id=%s", agent_id, task_id)

    def _get_agent_span(self, agent_id: Optional[str]) -> Any:
        """Return the open span for *agent_id*, or ``None``."""
        if not agent_id:
            return None
        with self._lock:
            return self._agent_spans.get(agent_id)

    def _close_orphan_llm_spans(
        self,
        agent_id: str,
        status: str,
        error: Any,
    ) -> None:
        """
        Force-close any ``crewai.llm`` spans that were opened for *agent_id*
        but never received a ``LLMCallCompletedEvent``.

        This happens when CrewAI's agentic tool-calling loop does not emit
        ``LLMCallCompletedEvent`` for intermediate LLM rounds (e.g. the model
        returns a tool-call block rather than a final answer).  Without this,
        those spans remain open and are only closed (without any token or
        response data) when the crew span itself finishes.

        We call ``_finish_llm_span`` with ``event=None`` so the helper uses
        whatever the monkey-patch already buffered in ``_llm_usage_by_call_id``
        and marks the span with the same status as the agent.
        """
        with self._lock:
            orphan_call_ids = [
                call_id
                for call_id, entry in self._llm_call_spans.items()
                if entry.get("agent_id") == agent_id
            ]

        for call_id in orphan_call_ids:
            try:
                logger.debug(
                    "_close_orphan_llm_spans: force-closing call_id=%s for agent_id=%s",
                    call_id,
                    agent_id,
                )
                self._finish_llm_span(
                    call_id=call_id,
                    event=None,
                    status=status,
                    error=error,
                )
            except Exception:
                logger.debug(
                    "_close_orphan_llm_spans error for call_id=%s:\n%s",
                    call_id,
                    traceback.format_exc(),
                )

    def _close_orphan_tool_spans(
        self,
        agent_id: str,
        status: str,
    ) -> None:
        """
        Force-close any ``crewai.tool`` spans opened by *agent_id* that never
        received a ``ToolUsageFinishedEvent``.

        This is common when the LLM makes multiple parallel tool calls in a
        single response (batch tool-call): CrewAI fires one
        ``ToolUsageStartedEvent`` per tool but ``ToolUsageFinishedEvent`` may
        not fire for every individual tool in the batch.  Without this, those
        spans remain open and are silently closed (without ``tool.output`` /
        ``tool.status`` / ``tool.duration_ms``) when the crew span finishes.
        """
        with self._lock:
            orphan_run_ids = [
                run_id
                for run_id, aid in self._tool_run_id_to_agent_id.items()
                if aid == agent_id
            ]

        for run_id in orphan_run_ids:
            try:
                logger.debug(
                    "_close_orphan_tool_spans: force-closing run_id=%s agent_id=%s",
                    run_id,
                    agent_id,
                )
                self._finish_tool_span(
                    run_id=run_id,
                    status=status,
                    output=None,
                    error=None,
                )
            except Exception:
                logger.debug(
                    "_close_orphan_tool_spans error for run_id=%s:\n%s",
                    run_id,
                    traceback.format_exc(),
                )

    def _finish_agent_span(
        self,
        agent_id: str,
        status: str,
        output: Any,
        error: Any,
        extra_attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write final attributes onto the agent span and close it."""
        # Force-close any LLM spans opened during this agent's run that never
        # received a LLMCallCompletedEvent (common with tool-calling agentic loops
        # where CrewAI does not always emit the completion event per round).
        self._close_orphan_llm_spans(agent_id, status, error)
        # Force-close any tool spans that never received a ToolUsageFinishedEvent
        # (common when the LLM issues multiple parallel tool calls in one batch).
        self._close_orphan_tool_spans(agent_id, status)
        self._close_orphan_observation_spans(agent_id, status, error)
        self._close_orphan_reasoning_spans(agent_id, status, error)

        with self._lock:
            span = self._agent_spans.pop(agent_id, None)
            start_t = self._agent_start_times.pop(agent_id, None)

        if span is None:
            logger.debug("_finish_agent_span: no open span for agent_id=%s", agent_id)
            return

        attrs: dict[str, Any] = {ATTR_AGENT_STATUS: status}

        if start_t is not None:
            attrs[ATTR_AGENT_DURATION_MS] = duration_ms_monotonic(start_t)

        if output is not None:
            raw = _extract_output_text(output)
            if raw:
                attrs["agent.output"] = truncate_str(raw, MAX_TEXT_LENGTH)

        if error is not None:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = str(error)
            tb = getattr(error, "__traceback__", None)
            if tb is not None:
                attrs[ATTR_ERROR_STACKTRACE] = "".join(traceback.format_tb(tb))

        if extra_attrs:
            attrs.update(extra_attrs)

        try:
            _set_span_attributes(span, attrs)

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
                "_finish_agent_span span.finish() error:\n%s",
                traceback.format_exc(),
            )

        logger.debug("Agent span closed: agent_id=%s status=%s", agent_id, status)


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _resolve_agent_id(source: Any, event: Any) -> str:
    """Return a stable string key for the agent."""
    return str(
        safe_getattr(event, "agent_id")
        or safe_getattr(event, "id")
        or safe_getattr(source, "id")
        or id(source)
    )


def _resolve_task_id(source: Any, event: Any) -> Optional[str]:
    """Return the task_id associated with this agent event, or ``None``.

    CrewAI emits AgentExecutionStartedEvent with ``task=<Task>`` kwarg (not
    ``from_task=``), so ``event.task_id`` is always ``None``.  We must drill
    into ``event.task.id`` directly.
    """
    raw = (
        safe_getattr(event, "task_id")
        or safe_getattr(safe_getattr(event, "task"), "id")
        or safe_getattr(source, "task_id")
        or safe_getattr(safe_getattr(source, "task"), "id")
    )
    return str(raw) if raw is not None else None


def _build_agent_attributes(
    source: Any, event: Any, agent_type: str = "full"
) -> dict[str, Any]:
    """
    Collect span attributes for the opening ``crewai.agent`` span.

    Probes both *source* (the Agent object) and *event* (payload) so the
    handler works regardless of which CrewAI version populates which object.
    """
    attrs: dict[str, Any] = {"agent.type": agent_type}

    # Identity
    agent_id = str(
        safe_getattr(event, "agent_id")
        or safe_getattr(event, "id")
        or safe_getattr(source, "id")
        or ""
    )
    if agent_id:
        attrs[ATTR_AGENT_ID] = agent_id

    for attr_name, span_key, max_len in (
        ("role", ATTR_AGENT_ROLE, 256),
        ("goal", ATTR_AGENT_GOAL, MAX_DESCRIPTION_LENGTH),
        ("backstory", ATTR_AGENT_BACKSTORY, MAX_DESCRIPTION_LENGTH),
    ):
        val = safe_getattr(event, attr_name) or safe_getattr(source, attr_name)
        if val:
            attrs[span_key] = truncate_str(str(val), max_len)

    # LLM model
    model = safe_getattr(event, "llm_model") or extract_llm_model_from_agent(source)
    if model:
        attrs[ATTR_AGENT_LLM_MODEL] = str(model)

    # Numeric / bool config
    for attr_name, span_key in (
        ("allow_delegation", ATTR_AGENT_ALLOW_DELEGATION),
        ("max_iter", ATTR_AGENT_MAX_ITER),
        ("max_rpm", ATTR_AGENT_MAX_RPM),
    ):
        val = safe_getattr(source, attr_name)
        if val is None:
            val = safe_getattr(event, attr_name)
        if val is not None:
            attrs[span_key] = val

    # Task prompt: the compiled prompt that was passed into the agent.
    # CrewAI stores this on the event as ``task_prompt`` or ``prompt``.
    task_prompt = (
        safe_getattr(event, "task_prompt")
        or safe_getattr(event, "prompt")
        or safe_getattr(event, "input")
    )
    if task_prompt:
        attrs["agent.task_prompt"] = truncate_str(str(task_prompt), MAX_TEXT_LENGTH)

    # Tools
    tools = safe_getattr(source, "tools") or safe_getattr(event, "tools")
    if tools:
        _attach_tool_attributes(attrs, tools)

    return attrs


def _attach_tool_attributes(attrs: dict[str, Any], tools: Any) -> None:
    """
    Serialize agent tools into span attributes:

    - ``agent.tool_names``  — JSON list of bare tool name strings (compact)
    - ``agent.tools``       — JSON list of {name, description} dicts (detailed)
    - ``agent.available_tools.*`` — count, names, descriptions, schemas (full serialisation)
    """
    try:
        tool_list = tools if isinstance(tools, (list, tuple)) else [tools]

        # Compact name list
        names = [str(safe_getattr(t, "name") or t) for t in tool_list if t is not None]
        if names:
            attrs[ATTR_AGENT_TOOL_NAMES] = safe_json_dumps(names)

        # Detailed schema (name + description, no args_schema to keep it lean)
        detailed = []
        for tool in tool_list:
            if tool is None:
                continue
            entry: dict[str, Any] = {}
            name = safe_getattr(tool, "name")
            if name:
                entry["name"] = str(name)
            desc = safe_getattr(tool, "description")
            if desc:
                entry["description"] = truncate_str(str(desc), 512)
            if entry:
                detailed.append(entry)

        if detailed:
            attrs["agent.tools"] = safe_json_dumps(detailed)

        merge_available_tools_attributes(attrs, tools, "agent")

    except Exception as exc:
        logger.debug("_attach_tool_attributes failed: %s", exc)


def _extract_agent_evaluation_attributes(event: Any) -> dict[str, Any]:
    """Extract evaluation score + feedback from an agent evaluation event."""
    attrs: dict[str, Any] = {}

    score = safe_getattr(event, "score")
    if score is not None:
        try:
            attrs["agent.evaluation.score"] = float(score)
        except (TypeError, ValueError):
            attrs["agent.evaluation.score"] = str(score)

    feedback = safe_getattr(event, "feedback") or safe_getattr(event, "suggestion")
    if feedback:
        attrs["agent.evaluation.feedback"] = truncate_str(str(feedback), 2048)

    passed = safe_getattr(event, "passed")
    if passed is not None:
        attrs["agent.evaluation.passed"] = bool(passed)

    model = safe_getattr(event, "model") or safe_getattr(event, "evaluator_model")
    if model:
        attrs["agent.evaluation.model"] = str(model)

    # Nested result object fallback
    result_obj = safe_getattr(event, "result")
    if result_obj is not None and score is None:
        nested_score = safe_getattr(result_obj, "score")
        if nested_score is not None:
            try:
                attrs["agent.evaluation.score"] = float(nested_score)
            except (TypeError, ValueError):
                pass
        nested_feedback = safe_getattr(result_obj, "feedback")
        if nested_feedback and "agent.evaluation.feedback" not in attrs:
            attrs["agent.evaluation.feedback"] = truncate_str(
                str(nested_feedback), 2048
            )

    return attrs


def _extract_output_text(output: Any) -> Optional[str]:
    """Best-effort extraction of text from an agent output value."""
    if output is None:
        return None
    if isinstance(output, str):
        return output

    # AgentFinish / TaskOutput .raw
    raw = safe_getattr(output, "raw")
    if isinstance(raw, str) and raw:
        return raw

    # AgentFinish .return_values (LangChain pattern used by CrewAI internally)
    return_values = safe_getattr(output, "return_values")
    if isinstance(return_values, dict):
        output_val = return_values.get("output")
        if isinstance(output_val, str):
            return output_val

    # Pydantic model
    pydantic_obj = safe_getattr(output, "pydantic")
    if pydantic_obj is not None:
        for method in ("model_dump_json", "json"):
            fn = getattr(pydantic_obj, method, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass

    try:
        return str(output)
    except Exception:
        return None


def _set_span_attributes(span: Any, attrs: dict[str, Any]) -> None:
    """Write *attrs* to *span* via ``set_attributes`` or direct dict update."""
    if not attrs:
        return
    try:
        if hasattr(span, "set_attributes"):
            span.set_attributes(attrs)
        elif hasattr(span, "attributes"):
            span.attributes.update(attrs)
    except Exception as exc:
        logger.debug("_set_span_attributes failed: %s", exc)
