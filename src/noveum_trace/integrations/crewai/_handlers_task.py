"""
Task-lifecycle event handler mixin for NoveumCrewAIListener.

Handles CrewAI ``BaseEventListener`` task events:

  - ``on_task_started``     → open ``crewai.task`` child span under the crew span;
                               capture task snapshot: description, expected_output,
                               assigned agent role, human_input flag, and the full
                               context chain (upstream task descriptions = RAG chain)
  - ``on_task_completed``   → close span as SUCCESS; write task output text
  - ``on_task_failed``      → close span as ERROR; attach exception details
  - ``on_task_evaluation``  → attach quality / evaluation score to the span

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown,
    _crew_spans, _task_spans, _task_start_times
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_STACKTRACE,
    ATTR_ERROR_TYPE,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    ATTR_TASK_AGENT_ROLE,
    ATTR_TASK_ASYNC,
    ATTR_TASK_CONTEXT,
    ATTR_TASK_DESCRIPTION,
    ATTR_TASK_DURATION_MS,
    ATTR_TASK_EXPECTED_OUTPUT,
    ATTR_TASK_HUMAN_INPUT,
    ATTR_TASK_ID,
    ATTR_TASK_NAME,
    ATTR_TASK_OUTPUT,
    ATTR_TASK_OUTPUT_FILE,
    ATTR_TASK_STATUS,
    MAX_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
    SPAN_TASK,
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


class _TaskHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI Task-lifecycle events.

    Every public method matches the ``BaseEventListener`` callback signature::

        def on_task_started(self, source, event): ...

    ``source`` is the ``Task`` object; ``event`` carries event-specific payload.
    All methods are fully exception-shielded.
    """

    # =========================================================================
    # Task started
    # =========================================================================

    def on_task_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.task`` span as a child of the owning crew span.

        Captures at span-open time:

        Attributes set
        --------------
        - ``task.id``               — unique task id (UUID)
        - ``task.name``             — optional human-readable task name
        - ``task.description``      — task description (≤ MAX_DESCRIPTION_LENGTH)
        - ``task.expected_output``  — what the task should produce (≤ MAX_DESCRIPTION_LENGTH)
        - ``task.agent_role``       — role of the assigned agent
        - ``task.human_input``      — bool: requires human sign-off?
        - ``task.async_execution``  — bool: runs asynchronously?
        - ``task.output_file``      — output file path when configured
        - ``task.context_tasks``    — JSON list of upstream task descriptions
                                      (the RAG / context chain fed into this task);
                                      only when ``capture_inputs`` is enabled

        When ``capture_inputs`` is false, description, expected output, output
        file path, and context chain are omitted; structural fields (ids, name,
        agent role, flags) are still recorded.
        """
        if not self._is_active():
            return
        try:
            task_id = _resolve_task_id(source, event)
            crew_id = _resolve_crew_id(source, event)

            attrs = _build_task_attributes(
                source, event, capture_inputs=self.capture_inputs
            )
            start_t = monotonic_now()

            # Locate the parent crew span (may be None when crew span creation
            # failed or tasks are used outside a Crew)
            parent_span = self._get_parent_span(crew_id)

            span = self._create_child_span(
                SPAN_TASK,
                parent_span=parent_span,
                attributes=attrs,
                # Pass hints so _create_child_span can unambiguously pick the
                # correct crew trace when multiple crews overlap in time.
                crew_id=crew_id or None,
                task_id=task_id or None,
            )

            with self._lock:
                self._task_spans[task_id] = span
                self._task_start_times[task_id] = start_t

            logger.debug("Task span opened: task_id=%s crew_id=%s", task_id, crew_id)

        except Exception:
            logger.debug("on_task_started error:\n%s", traceback.format_exc())

    # =========================================================================
    # Task completed (success)
    # =========================================================================

    def on_task_completed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.task`` span as SUCCESS.

        Attributes written
        ------------------
        - ``task.output``       — final task output text (≤ MAX_TEXT_LENGTH);
                                only when ``capture_outputs`` is enabled
        - ``task.status``       — ``"success"``
        - ``task.duration_ms``  — wall-clock duration since ``on_task_started``
        """
        if not self._is_active():
            return
        try:
            task_id = _resolve_task_id(source, event)
            output = safe_getattr(event, "output")
            self._finish_task_span(
                task_id=task_id,
                status=ATTR_STATUS_SUCCESS,
                output=output,
                error=None,
            )
        except Exception:
            logger.debug("on_task_completed error:\n%s", traceback.format_exc())

    # =========================================================================
    # Task failed (error)
    # =========================================================================

    def on_task_failed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.task`` span as ERROR.

        Attributes written
        ------------------
        - ``error.type``        — exception class name
        - ``error.message``     — exception message
        - ``error.stacktrace``  — formatted traceback (when available)
        - ``task.status``       — ``"error"``
        - ``task.duration_ms``  — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            task_id = _resolve_task_id(source, event)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_task_span(
                task_id=task_id,
                status=ATTR_STATUS_ERROR,
                output=None,
                error=error,
            )
        except Exception:
            logger.debug("on_task_failed error:\n%s", traceback.format_exc())

    # =========================================================================
    # Task evaluation (quality / judge score)
    # =========================================================================

    def on_task_evaluation(self, source: Any, event: Any) -> None:
        """
        Attach an automated evaluation score to the task span.

        CrewAI's ``TaskEvaluated`` event (fired when ``evaluate=True`` or an
        evaluator callback is configured) provides a numeric quality score and
        optional evaluator feedback.

        Attributes written
        ------------------
        - ``task.evaluation.score``     — float quality score (0–1 or 0–10 scale,
                                          provider-dependent)
        - ``task.evaluation.feedback``  — evaluator text feedback (truncated)
        - ``task.evaluation.model``     — LLM used for evaluation (when available)
        - ``task.evaluation.criteria``  — list of criteria names (JSON, when provided)
        """
        if not self._is_active():
            return
        try:
            task_id = _resolve_task_id(source, event)

            with self._lock:
                span = self._task_spans.get(task_id)

            if span is None:
                logger.debug(
                    "on_task_evaluation: no open span for task_id=%s — "
                    "evaluation arrived after task span closed; attaching to "
                    "crew span if available",
                    task_id,
                )
                # Fall back to the crew span so the score is not lost
                crew_id = _resolve_crew_id(source, event)
                span = self._get_parent_span(crew_id)

            if span is None:
                return

            eval_attrs = _extract_evaluation_attributes(event)
            try:
                if hasattr(span, "set_attributes"):
                    span.set_attributes(eval_attrs)
                elif hasattr(span, "attributes"):
                    span.attributes.update(eval_attrs)
            except Exception:
                logger.debug(
                    "on_task_evaluation span.set_attributes error:\n%s",
                    traceback.format_exc(),
                )

        except Exception:
            logger.debug("on_task_evaluation error:\n%s", traceback.format_exc())

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_parent_span(self, crew_id: Optional[str]) -> Any:
        """Return the open crew Span for *crew_id*, or ``None``."""
        if not crew_id:
            return None
        with self._lock:
            entry = self._crew_spans.get(crew_id)
        return entry["span"] if entry else None

    def _finish_task_span(
        self,
        task_id: str,
        status: str,
        output: Any,
        error: Any,
    ) -> None:
        """
        Write final attributes onto the task span and close it.

        Removes the span and start-time entries from state dicts atomically.
        """
        with self._lock:
            span = self._task_spans.pop(task_id, None)
            start_t = self._task_start_times.pop(task_id, None)

        if span is None:
            logger.debug("_finish_task_span: no open span for task_id=%s", task_id)
            return

        attrs: dict[str, Any] = {ATTR_TASK_STATUS: status}

        # Duration
        if start_t is not None:
            attrs[ATTR_TASK_DURATION_MS] = duration_ms_monotonic(start_t)

        # Task output (privacy: respect capture_outputs)
        if output is not None and getattr(self, "capture_outputs", True):
            raw = _extract_task_output_text(output)
            if raw:
                attrs[ATTR_TASK_OUTPUT] = truncate_str(raw, MAX_TEXT_LENGTH)

        # Error details
        if error is not None:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = str(error)
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
                "_finish_task_span span.finish() error:\n%s",
                traceback.format_exc(),
            )

        logger.debug("Task span closed: task_id=%s status=%s", task_id, status)


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _resolve_task_id(source: Any, event: Any) -> str:
    """Return a stable string key for the task from event or source."""
    return str(
        safe_getattr(event, "task_id")
        or safe_getattr(event, "id")
        or safe_getattr(source, "id")
        or id(source)
    )


def _resolve_crew_id(source: Any, event: Any) -> Optional[str]:
    """Return the crew_id associated with this task event, or ``None``."""
    raw = (
        safe_getattr(event, "crew_id")
        or safe_getattr(source, "crew_id")
        or safe_getattr(safe_getattr(source, "crew"), "id")
    )
    return str(raw) if raw is not None else None


def _build_task_attributes(
    source: Any, event: Any, *, capture_inputs: bool = True
) -> dict[str, Any]:
    """
    Collect span attributes for the opening ``crewai.task`` span.

    Probes both *source* (the ``Task`` object) and *event* (the event payload)
    so the handler works regardless of which object CrewAI populates.

    When *capture_inputs* is false, task prompt-style fields (description,
    expected output, output file path, upstream context chain) are omitted so
    listener privacy settings are respected.
    """
    attrs: dict[str, Any] = {}

    task_id = str(
        safe_getattr(event, "task_id")
        or safe_getattr(event, "id")
        or safe_getattr(source, "id")
        or ""
    )
    if task_id:
        attrs[ATTR_TASK_ID] = task_id

    name = safe_getattr(source, "name") or safe_getattr(event, "name")
    if name:
        attrs[ATTR_TASK_NAME] = str(name)

    if capture_inputs:
        description = safe_getattr(event, "description") or safe_getattr(
            source, "description"
        )
        if description:
            attrs[ATTR_TASK_DESCRIPTION] = truncate_str(
                str(description), MAX_DESCRIPTION_LENGTH
            )

        expected_output = safe_getattr(event, "expected_output") or safe_getattr(
            source, "expected_output"
        )
        if expected_output:
            attrs[ATTR_TASK_EXPECTED_OUTPUT] = truncate_str(
                str(expected_output), MAX_DESCRIPTION_LENGTH
            )

    # Assigned agent role
    agent = safe_getattr(source, "agent") or safe_getattr(event, "agent")
    if agent is not None:
        role = safe_getattr(agent, "role")
        if role:
            attrs[ATTR_TASK_AGENT_ROLE] = str(role)

    # Boolean flags
    for flag_attr, span_key in (
        ("human_input", ATTR_TASK_HUMAN_INPUT),
        ("async_execution", ATTR_TASK_ASYNC),
    ):
        val = safe_getattr(source, flag_attr)
        if val is None:
            val = safe_getattr(event, flag_attr)
        if val is not None:
            attrs[span_key] = bool(val)

    if capture_inputs:
        output_file = safe_getattr(source, "output_file")
        if output_file:
            attrs[ATTR_TASK_OUTPUT_FILE] = str(output_file)

        # Context chain (RAG chain): list of upstream tasks whose outputs are
        # fed as context into this task.  We serialise each upstream task as its
        # description so the attribute is human-readable without the full Task object.
        # Read only from source (the Task object) — BaseEvent may also expose a
        # `context` attribute with unrelated data, so we deliberately skip event here.
        context_tasks = safe_getattr(source, "context")
        if context_tasks:
            context_descriptions = _extract_context_chain(context_tasks)
            if context_descriptions:
                attrs[ATTR_TASK_CONTEXT] = safe_json_dumps(context_descriptions)

    return attrs


def _extract_context_chain(context_tasks: Any) -> list[str]:
    """
    Serialise the upstream task context chain to a list of human-readable strings.

    Each element is either:
    - The task ``description`` (preferred — captures intent)
    - The task ``id`` (fallback when description is absent)

    Tasks with neither a readable description nor an id are silently skipped
    to avoid sentinel noise like "NOT_SPECIFIED".

    This is the "RAG chain" — the ordered sequence of upstream task outputs
    that are injected as context into the current task.
    """
    result: list[str] = []
    try:
        iterable = (
            context_tasks
            if isinstance(context_tasks, (list, tuple))
            else [context_tasks]
        )
        for ctx_task in iterable:
            if ctx_task is None:
                continue
            description = safe_getattr(ctx_task, "description")
            if description:
                result.append(truncate_str(str(description), MAX_DESCRIPTION_LENGTH))
                continue
            task_id = safe_getattr(ctx_task, "id")
            if task_id:
                result.append(str(task_id))
                continue
            # Skip entries that have no readable description or id — str(ctx_task)
            # often produces sentinel values like "NOT_SPECIFIED" that add noise.
    except Exception as exc:
        logger.debug("_extract_context_chain failed: %s", exc)
    return result


def _extract_evaluation_attributes(event: Any) -> dict[str, Any]:
    """
    Extract evaluation / quality-score attributes from a task-evaluation event.

    Returns a dict ready to be written to a span via ``set_attributes``.
    Uses flat spec keys: task.evaluation_score, task.evaluation_feedback, etc.
    """
    attrs: dict[str, Any] = {}

    score = safe_getattr(event, "score")
    if score is not None:
        try:
            attrs["task.evaluation_score"] = float(score)
        except (TypeError, ValueError):
            attrs["task.evaluation_score"] = str(score)

    feedback = safe_getattr(event, "feedback") or safe_getattr(event, "suggestion")
    if feedback:
        attrs["task.evaluation_feedback"] = truncate_str(str(feedback), 2048)

    model = safe_getattr(event, "model") or safe_getattr(event, "evaluator_model")
    if model:
        attrs["task.evaluation_model"] = str(model)

    criteria = safe_getattr(event, "criteria")
    if criteria:
        try:
            if isinstance(criteria, (list, tuple)):
                attrs["task.evaluation_criteria"] = safe_json_dumps(list(criteria))
            else:
                attrs["task.evaluation_criteria"] = str(criteria)
        except Exception:
            pass

    # Some evaluators return a structured result object
    result_obj = safe_getattr(event, "result")
    if result_obj is not None and score is None:
        nested_score = safe_getattr(result_obj, "score")
        if nested_score is not None:
            try:
                attrs["task.evaluation_score"] = float(nested_score)
            except (TypeError, ValueError):
                pass
        nested_feedback = safe_getattr(result_obj, "feedback")
        if nested_feedback and "task.evaluation_feedback" not in attrs:
            attrs["task.evaluation_feedback"] = truncate_str(str(nested_feedback), 2048)

    return attrs


def _extract_task_output_text(output: Any) -> Optional[str]:
    """
    Best-effort extraction of text from a ``TaskOutput`` or plain value.

    ``TaskOutput`` exposes ``.raw`` (str), ``.pydantic``, and ``.json_dict``.
    Falls back to ``str(output)`` for unknown types.
    """
    if output is None:
        return None
    if isinstance(output, str):
        return output

    raw = safe_getattr(output, "raw")
    if isinstance(raw, str) and raw:
        return raw

    pydantic_obj = safe_getattr(output, "pydantic")
    if pydantic_obj is not None:
        for method in ("model_dump_json", "json"):
            fn = getattr(pydantic_obj, method, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass

    json_dict = safe_getattr(output, "json_dict")
    if isinstance(json_dict, dict):
        try:
            import json

            return json.dumps(json_dict, default=str)
        except Exception:
            pass

    try:
        return str(output)
    except Exception:
        return None
