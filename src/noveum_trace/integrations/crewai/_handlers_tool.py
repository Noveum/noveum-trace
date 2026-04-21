"""
Tool-usage event handler mixin for NoveumCrewAIListener.

Handles CrewAI ``BaseEventListener`` tool events:

  - ``on_tool_usage_started``        → open ``crewai.tool`` child span under the
                                        agent span; capture tool_name, tool_class,
                                        tool_args (JSON), agent_role, task_name,
                                        run_attempts counter, delegations count
  - ``on_tool_usage_finished``       → close span as SUCCESS; write output text,
                                        duration_ms, result
  - ``on_tool_usage_error``          → close span as ERROR; write error message

  Error-only events (no span lifecycle — annotate the nearest open span):
  - ``on_tool_validate_input_error`` → attach validation error to agent/tool span
  - ``on_tool_selection_error``      → attach selection error to agent span
  - ``on_tool_execution_error``      → attach execution error to tool or agent span

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown,
    _agent_spans, _tool_spans, _tool_start_times
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
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    ATTR_TOOL_DESCRIPTION,
    ATTR_TOOL_DURATION_MS,
    ATTR_TOOL_ERROR,
    ATTR_TOOL_INPUT,
    ATTR_TOOL_NAME,
    ATTR_TOOL_OUTPUT,
    ATTR_TOOL_RUN_ID,
    ATTR_TOOL_STATUS,
    MAX_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
    SPAN_TOOL,
)
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase
from noveum_trace.integrations.crewai.crewai_utils import (
    duration_ms_monotonic,
    extract_tool_result,
    monotonic_now,
)
from noveum_trace.integrations.crewai.crewai_utils import (
    resolve_agent_id as _resolve_agent_id,
)
from noveum_trace.integrations.crewai.crewai_utils import (
    safe_getattr,
    safe_json_dumps,
    truncate_str,
)

logger = logging.getLogger(__name__)


class _ToolHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI Tool-usage events.

    All public methods match the ``BaseEventListener`` callback signature::

        def on_tool_usage_started(self, source, event): ...

    ``source`` is the ``Agent`` executing the tool; ``event`` carries the
    per-invocation payload.  Every method is fully exception-shielded.
    """

    # =========================================================================
    # Tool usage started
    # =========================================================================

    def on_tool_usage_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.tool`` span as a child of the executing agent span.

        Attributes set at span open
        ---------------------------
        - ``tool.run_id``        — unique tool invocation identifier
        - ``tool.name``          — tool name string
        - ``tool.class``         — tool implementation class name
        - ``tool.description``   — tool description (≤ MAX_DESCRIPTION_LENGTH)
        - ``tool.input``         — serialized tool arguments (JSON or str)
        - ``agent.role``         — role of the agent using the tool (correlation)
        - ``task.name``          — name/description of the current task
        - ``tool.run_attempts``  — how many times this tool has been retried
        - ``tool.delegations``   — delegation depth counter (hierarchical crews)
        """
        if not self._is_active():
            return
        try:
            # Use event.event_id as the stable key — ToolUsageFinishedEvent
            # carries this same ID in its started_event_id field, providing
            # reliable start↔finish correlation without relying on id(event).
            run_id = _resolve_started_run_id(event, source)
            agent_id = _resolve_agent_id(source, event)
            attrs = _build_tool_start_attributes(source, event, run_id)
            start_t = monotonic_now()

            # Parent: agent span (most common)
            parent_span = self._get_agent_span(agent_id)

            span = self._create_child_span(
                SPAN_TOOL,
                parent_span=parent_span,
                attributes=attrs,
            )

            with self._lock:
                self._tool_spans[run_id] = span
                self._tool_start_times[run_id] = start_t
                # Record the owning agent so orphan-close can find this span.
                if agent_id:
                    self._tool_run_id_to_agent_id[run_id] = agent_id

            logger.debug(
                "Tool span opened: run_id=%s agent_id=%s tool=%s",
                run_id,
                agent_id,
                attrs.get(ATTR_TOOL_NAME, "?"),
            )

        except Exception:
            logger.debug("on_tool_usage_started error:\n%s", traceback.format_exc())

    # =========================================================================
    # Tool usage finished (success)
    # =========================================================================

    def on_tool_usage_finished(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.tool`` span as SUCCESS.

        Attributes written
        ------------------
        - ``tool.output``       — tool result text (≤ MAX_TEXT_LENGTH)
        - ``tool.result``       — raw result before truncation (same source)
        - ``tool.status``       — ``"success"``
        - ``tool.duration_ms``  — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            # Match the key that on_tool_usage_started wrote: started_event_id
            # on the finished event equals event_id on the started event.
            run_id = _resolve_finished_run_id(event, source)
            output = safe_getattr(event, "output") or safe_getattr(event, "result")
            self._finish_tool_span(
                run_id=run_id,
                status=ATTR_STATUS_SUCCESS,
                output=output,
                error=None,
            )
        except Exception:
            logger.debug("on_tool_usage_finished error:\n%s", traceback.format_exc())

    # =========================================================================
    # Tool usage error (generic)
    # =========================================================================

    def on_tool_usage_error(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.tool`` span as ERROR.

        Attributes written
        ------------------
        - ``error.type``        — exception class name
        - ``error.message``     — error message string
        - ``error.stacktrace``  — formatted traceback when available
        - ``tool.error``        — same message (duplicate for quick filtering)
        - ``tool.status``       — ``"error"``
        - ``tool.duration_ms``  — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            run_id = _resolve_finished_run_id(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_tool_span(
                run_id=run_id,
                status=ATTR_STATUS_ERROR,
                output=None,
                error=error,
            )
        except Exception:
            logger.debug("on_tool_usage_error error:\n%s", traceback.format_exc())

    # =========================================================================
    # Validate-input error (no span lifecycle — annotate nearest open span)
    # =========================================================================

    def on_tool_validate_input_error(self, source: Any, event: Any) -> None:
        """
        Attach a validation error to the nearest open span.

        CrewAI fires ``ToolValidateInputErrorEvent`` before a tool span is
        opened, so we annotate the agent span (or open tool span if present).

        Attribute written: ``tool.validate_input_error``
        """
        if not self._is_active():
            return
        try:
            run_id = _resolve_run_id(event, source)
            agent_id = _resolve_agent_id(source, event)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            error_str = (
                str(error) if error else str(safe_getattr(event, "message") or "")
            )

            # Try the open tool span first; fall back to agent span
            span = self._get_tool_or_agent_span(run_id, agent_id)
            if span is None:
                return

            _annotate_span_error(span, "tool.validate_input_error", error_str, error)
            logger.debug(
                "tool.validate_input_error attached: run_id=%s agent_id=%s",
                run_id,
                agent_id,
            )
        except Exception:
            logger.debug(
                "on_tool_validate_input_error error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Tool selection error
    # =========================================================================

    def on_tool_selection_error(self, source: Any, event: Any) -> None:
        """
        Attach a tool-selection error to the agent span.

        Fires when CrewAI fails to match the agent's chosen tool name to any
        registered tool (e.g. hallucinated tool name).

        Attributes written
        ------------------
        - ``tool.selection_error``         — error / mismatch description
        - ``tool.selection_error.chosen``  — the tool name the agent tried to use
        """
        if not self._is_active():
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            span = self._get_agent_span(agent_id)
            if span is None:
                return

            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            error_str = (
                str(error) if error else str(safe_getattr(event, "message") or "")
            )
            _annotate_span_error(span, "tool.selection_error", error_str, error)

            # The tool name the agent attempted to invoke (may be hallucinated)
            chosen_tool = safe_getattr(event, "tool_name") or safe_getattr(
                event, "chosen_tool"
            )
            if chosen_tool:
                _set_span_attr(span, "tool.selection_error.chosen", str(chosen_tool))

            logger.debug(
                "tool.selection_error attached: agent_id=%s chosen=%s",
                agent_id,
                chosen_tool,
            )
        except Exception:
            logger.debug("on_tool_selection_error error:\n%s", traceback.format_exc())

    # =========================================================================
    # Tool execution error (distinct from usage error — execution-phase only)
    # =========================================================================

    def on_tool_execution_error(self, source: Any, event: Any) -> None:
        """
        Attach an execution-phase error to the open tool span, or agent span
        if the tool span has already closed / was never opened.

        Attributes written
        ------------------
        - ``tool.execution_error``       — error description
        - ``tool.execution_error.type``  — exception class name
        """
        if not self._is_active():
            return
        try:
            run_id = _resolve_run_id(event, source)
            agent_id = _resolve_agent_id(source, event)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            error_str = (
                str(error) if error else str(safe_getattr(event, "message") or "")
            )

            span = self._get_tool_or_agent_span(run_id, agent_id)
            if span is None:
                return

            _annotate_span_error(span, "tool.execution_error", error_str, error)
            logger.debug("tool.execution_error attached: run_id=%s", run_id)
        except Exception:
            logger.debug("on_tool_execution_error error:\n%s", traceback.format_exc())

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_agent_span(self, agent_id: Optional[str]) -> Any:
        """Return the open agent span for *agent_id*, or ``None``."""
        if not agent_id:
            return None
        with self._lock:
            return self._agent_spans.get(agent_id)

    def _get_tool_or_agent_span(
        self, run_id: Optional[str], agent_id: Optional[str]
    ) -> Optional[Any]:
        """Return the open tool span, falling back to the agent span."""
        with self._lock:
            if run_id is not None:
                tool_span = self._tool_spans.get(run_id)
                if tool_span is not None:
                    return tool_span
            if agent_id:
                return self._agent_spans.get(agent_id)
        return None

    def _finish_tool_span(
        self,
        run_id: str,
        status: str,
        output: Any,
        error: Any,
    ) -> None:
        """Write final attributes onto the tool span and close it."""
        with self._lock:
            span = self._tool_spans.pop(run_id, None)
            start_t = self._tool_start_times.pop(run_id, None)
            self._tool_run_id_to_agent_id.pop(run_id, None)

        if span is None:
            logger.debug("_finish_tool_span: no open span for run_id=%s", run_id)
            return

        attrs: dict[str, Any] = {ATTR_TOOL_STATUS: status}

        if start_t is not None:
            attrs[ATTR_TOOL_DURATION_MS] = duration_ms_monotonic(start_t)

        if output is not None:
            result_str = extract_tool_result(output)
            if result_str:
                attrs[ATTR_TOOL_OUTPUT] = result_str

        if error is not None:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = str(error)
            attrs[ATTR_TOOL_ERROR] = str(error)
            tb = getattr(error, "__traceback__", None)
            if tb is not None:
                attrs[ATTR_ERROR_STACKTRACE] = "".join(traceback.format_tb(tb))

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
                "_finish_tool_span span.finish() error:\n%s",
                traceback.format_exc(),
            )

        logger.debug("Tool span closed: run_id=%s status=%s", run_id, status)


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _resolve_started_run_id(event: Any, source: Any) -> str:
    """
    Return the stable key to store when a tool span opens.

    CrewAI's tool events carry no explicit ``run_id``; the unique identifier
    is ``event.event_id`` (set on every ``BaseEvent``).  The corresponding
    ``ToolUsageFinishedEvent`` / ``ToolUsageErrorEvent`` expose the same UUID
    via their ``started_event_id`` field, giving us a reliable start↔finish
    correlation without relying on ``id(event)`` (which changes per object).
    """
    return str(
        safe_getattr(event, "run_id")  # explicit field (may exist in future)
        or safe_getattr(event, "tool_run_id")
        or safe_getattr(event, "event_id")  # unique ID of THIS started event
        or safe_getattr(event, "id")
        or safe_getattr(event, "tool_call_id")
        or id(event)
    )


def _resolve_finished_run_id(event: Any, source: Any) -> str:
    """
    Return the key that matches the open tool span for a finished/error event.

    ``ToolUsageFinishedEvent.started_event_id`` equals the ``event_id`` that
    was written by ``_resolve_started_run_id`` when the span was opened.
    """
    return str(
        safe_getattr(event, "run_id")
        or safe_getattr(event, "tool_run_id")
        or safe_getattr(event, "started_event_id")  # matches started event's event_id
        or safe_getattr(event, "event_id")
        or safe_getattr(event, "id")
        or safe_getattr(event, "tool_call_id")
        or id(event)
    )


def _resolve_run_id(event: Any, source: Any) -> str:
    """Legacy alias — used only by error-annotation helpers that don't open spans."""
    return _resolve_finished_run_id(event, source)


def _build_tool_start_attributes(
    source: Any, event: Any, run_id: str
) -> dict[str, Any]:
    """Collect span attributes for the opening ``crewai.tool`` span."""
    attrs: dict[str, Any] = {ATTR_TOOL_RUN_ID: run_id}

    # --- Tool identity -------------------------------------------------------
    tool = safe_getattr(event, "tool") or source  # source may BE the tool
    tool_name = (
        safe_getattr(event, "tool_name")
        or safe_getattr(tool, "name")
        or safe_getattr(event, "name")
    )
    if tool_name:
        attrs[ATTR_TOOL_NAME] = truncate_str(str(tool_name), 256)

    # Tool implementation class
    tool_class = safe_getattr(event, "tool_class") or type(tool).__name__
    if tool_class:
        attrs["tool.class"] = str(tool_class)

    # Tool description
    tool_desc = safe_getattr(event, "tool_description") or safe_getattr(
        tool, "description"
    )
    if tool_desc:
        attrs[ATTR_TOOL_DESCRIPTION] = truncate_str(
            str(tool_desc), MAX_DESCRIPTION_LENGTH
        )

    # --- Tool arguments (input) ---------------------------------------------
    args = (
        safe_getattr(event, "tool_input")
        or safe_getattr(event, "arguments")
        or safe_getattr(event, "args")
        or safe_getattr(event, "tool_args")
    )
    if args is not None:
        if isinstance(args, str):
            attrs[ATTR_TOOL_INPUT] = truncate_str(args, MAX_TEXT_LENGTH)
        else:
            attrs[ATTR_TOOL_INPUT] = truncate_str(
                safe_json_dumps(args), MAX_TEXT_LENGTH
            )

    # --- Agent / task correlation -------------------------------------------
    agent_role = safe_getattr(source, "role") or safe_getattr(event, "agent_role")
    if agent_role:
        attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

    task_name = (
        safe_getattr(event, "task_name")
        or safe_getattr(event, "task_description")
        or safe_getattr(safe_getattr(source, "task"), "description")
        or safe_getattr(safe_getattr(source, "task"), "name")
    )
    if task_name:
        attrs["task.name"] = truncate_str(str(task_name), 512)

    # --- Retry / delegation counters ----------------------------------------
    # Use explicit None-checks so integer 0 (first attempt) is not swallowed
    # by a falsy `or` short-circuit.
    _run_attempts = safe_getattr(event, "run_attempts")
    if _run_attempts is None:
        _run_attempts = safe_getattr(source, "run_attempts")
    if _run_attempts is not None:
        try:
            attrs["tool.run_attempts"] = int(_run_attempts)
        except (TypeError, ValueError):
            pass

    _delegations = safe_getattr(event, "delegations")
    if _delegations is None:
        _delegations = safe_getattr(source, "delegations")
    if _delegations is not None:
        try:
            attrs["tool.delegations"] = int(_delegations)
        except (TypeError, ValueError):
            pass

    return attrs


def _annotate_span_error(
    span: Any,
    attr_key: str,
    error_str: str,
    error_obj: Any = None,
) -> None:
    """
    Write a named error attribute onto *span*.

    Also writes ``{attr_key}.type`` when *error_obj* carries type information.
    Uses ``set_attributes`` when available, falls back to direct dict write.
    """
    attrs: dict[str, Any] = {}
    if error_str:
        attrs[attr_key] = truncate_str(error_str, 1024)
    if error_obj is not None:
        attrs[f"{attr_key}.type"] = type(error_obj).__name__

    if not attrs:
        return
    try:
        if hasattr(span, "set_attributes"):
            span.set_attributes(attrs)
        elif hasattr(span, "attributes"):
            span.attributes.update(attrs)
    except Exception as exc:
        logger.debug("_annotate_span_error failed: %s", exc)


def _set_span_attr(span: Any, key: str, value: Any) -> None:
    """Single-attribute write helper."""
    try:
        if hasattr(span, "set_attribute"):
            span.set_attribute(key, value)
        elif hasattr(span, "attributes"):
            span.attributes[key] = value
    except Exception as exc:
        logger.debug("_set_span_attr failed key=%s: %s", key, exc)
