"""
Agent reasoning (ReAct loop) event handler mixin for NoveumCrewAIListener.

CrewAI agents follow a **ReAct** (Reasoning + Acting) loop:

    Thought → Action (tool call) → Observation → Thought → ...

This module maps the internal reasoning-loop events to spans:

  Span hierarchy (within a ``crewai.agent`` span)::

    crewai.agent
      crewai.reasoning   ← one per reasoning attempt (Thought block)
        crewai.step_observation  ← one per Action → Observation pair

Reasoning events:
  - ``on_agent_reasoning_started``    → open ``crewai.reasoning`` span;
                                         capture agent_role, task_id, plan text,
                                         attempt number, is_ready flag
  - ``on_agent_reasoning_completed``  → close as SUCCESS; write final plan
  - ``on_agent_reasoning_failed``     → close as ERROR

Step/observation events:
  - ``on_step_observation_started``   → open ``crewai.step_observation`` child span;
                                         capture step index, action name, action input
  - ``on_step_observation_completed`` → close as SUCCESS; write observation text
  - ``on_step_observation_failed``    → close as ERROR; write error

Mid-reasoning annotations (no span lifecycle — annotate open reasoning span):
  - ``on_plan_refinement``            → write refined_steps list, refinement_count
  - ``on_plan_replan_triggered``      → write replan reason, replan_count,
                                         completed_steps summary
  - ``on_goal_achieved_early``        → write steps_remaining, early_exit flag

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown,
    _agent_spans,
    _reasoning_spans (dict[reasoning_id, {span, start_t}]),
    _observation_spans (dict[obs_id, {span, start_t}])

Note: ``_reasoning_spans`` and ``_observation_spans`` are not declared in
``_CrewAIObserverState``; this mixin stores them in ``_flow_method_spans``
and ``_memory_op_spans`` respectively under namespaced keys to avoid adding
new state dicts.  (Alternatively, subclasses may extend the state class.)
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_AGENT_ROLE,
    ATTR_AGENT_STEP,
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_STACKTRACE,
    ATTR_ERROR_TYPE,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    ATTR_TASK_ID,
    MAX_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
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

# Span name constants (not in crewai_constants.py — reasoning is internal)
_SPAN_REASONING = "crewai.reasoning"
_SPAN_STEP_OBSERVATION = "crewai.step_observation"

# Namespace prefixes for reusing existing state dicts
# _flow_method_spans  → reasoning spans  (key: "rsn::{reasoning_id}")
# _memory_op_spans    → observation spans (key: "obs::{obs_id}")
_RSN_PREFIX = "rsn::"
_OBS_PREFIX = "obs::"


class _ReasoningHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI agent reasoning (ReAct loop) events.

    All public methods match the ``BaseEventListener`` callback signature::

        def on_agent_reasoning_started(self, source, event): ...

    ``source`` is the ``Agent``; ``event`` carries the per-reasoning payload.
    Every method is fully exception-shielded.
    """

    # =========================================================================
    # Reasoning — started / completed / failed
    # =========================================================================

    def on_agent_reasoning_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.reasoning`` child span under the owning agent span.

        One reasoning span is created per **Thought** iteration in the ReAct
        loop.  A single agent execution may contain multiple reasoning spans
        (one per attempt up to ``max_iter``).

        Attributes set at span open
        ---------------------------
        - ``reasoning.id``         — unique identifier for this reasoning attempt
        - ``agent.role``           — role of the reasoning agent
        - ``task.id``              — current task identifier
        - ``reasoning.attempt``    — iteration number (1-based)
        - ``reasoning.plan``       — the agent's current plan / thought text
        - ``reasoning.is_ready``   — bool: does the agent think the task is done?
        - ``reasoning.tool_name``  — tool the agent plans to use (when decided)
        - ``reasoning.tool_input`` — planned tool input (when decided)
        """
        if not self._is_active():
            return
        try:
            reasoning_id = _resolve_reasoning_id(event, source)
            agent_id = _resolve_agent_id(source, event)
            key = _RSN_PREFIX + reasoning_id

            attrs = _build_reasoning_start_attributes(source, event, reasoning_id)
            start_t = monotonic_now()

            parent_span = self._get_agent_span(agent_id)

            span = self._create_child_span(
                _SPAN_REASONING,
                parent_span=parent_span,
                attributes=attrs,
            )

            with self._lock:
                self._flow_method_spans[key] = {
                    "span": span,
                    "start_t": start_t,
                    "agent_id": agent_id,
                }

            logger.debug(
                "Reasoning span opened: reasoning_id=%s agent_id=%s attempt=%s",
                reasoning_id,
                agent_id,
                attrs.get("reasoning.attempt", "?"),
            )
        except Exception:
            logger.debug(
                "on_agent_reasoning_started error:\n%s", traceback.format_exc()
            )

    def on_agent_reasoning_completed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.reasoning`` span as SUCCESS.

        Attributes written
        ------------------
        - ``reasoning.final_plan``   — the agent's final thought/plan text
        - ``reasoning.is_ready``     — bool: task considered complete?
        - ``reasoning.status``       — ``"success"``
        - ``reasoning.duration_ms``  — wall-clock duration of this thought block
        """
        if not self._is_active():
            return
        try:
            reasoning_id = _resolve_reasoning_id(event, source)
            extra: dict[str, Any] = {}

            final_plan = (
                safe_getattr(event, "plan")
                or safe_getattr(event, "thought")
                or safe_getattr(event, "output")
            )
            if final_plan:
                extra["reasoning.final_plan"] = truncate_str(
                    str(final_plan), MAX_TEXT_LENGTH
                )

            is_ready = safe_getattr(event, "is_ready")
            if is_ready is not None:
                extra["reasoning.is_ready"] = bool(is_ready)

            self._finish_reasoning_span(
                reasoning_id, ATTR_STATUS_SUCCESS, None, extra
            )
        except Exception:
            logger.debug(
                "on_agent_reasoning_completed error:\n%s", traceback.format_exc()
            )

    def on_agent_reasoning_failed(self, source: Any, event: Any) -> None:
        """Close the ``crewai.reasoning`` span as ERROR."""
        if not self._is_active():
            return
        try:
            reasoning_id = _resolve_reasoning_id(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_reasoning_span(reasoning_id, ATTR_STATUS_ERROR, error)
        except Exception:
            logger.debug(
                "on_agent_reasoning_failed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Step observation — started / completed / failed
    # =========================================================================

    def on_step_observation_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.step_observation`` span for an Action → Observation step.

        Each step corresponds to one tool call + its result within the ReAct
        loop.  The observation span is a child of the active reasoning span
        (or agent span as fallback).

        Attributes set at span open
        ---------------------------
        - ``step.obs_id``          — unique observation identifier
        - ``step.index``           — 0-based step counter within this agent run
        - ``step.action``          — action name (typically the tool name)
        - ``step.action_input``    — input passed to the action (JSON / str)
        - ``step.tool``            — tool name (may differ from action name)
        - ``agent.role``           — executing agent's role (correlation)
        """
        if not self._is_active():
            return
        try:
            obs_id = _resolve_obs_id(event, source)
            reasoning_id = _resolve_reasoning_id(event, source)
            agent_id = _resolve_agent_id(source, event)
            key = _OBS_PREFIX + obs_id

            attrs = _build_observation_start_attributes(source, event, obs_id)
            start_t = monotonic_now()

            # Parent: active reasoning span → agent span → None
            parent_span = (
                self._get_reasoning_span(reasoning_id)
                or self._get_agent_span(agent_id)
            )

            span = self._create_child_span(
                _SPAN_STEP_OBSERVATION,
                parent_span=parent_span,
                attributes=attrs,
            )

            with self._lock:
                self._memory_op_spans[key] = span
                self._memory_op_start_times[key] = start_t

            logger.debug(
                "Observation span opened: obs_id=%s step=%s",
                obs_id,
                attrs.get("step.index", "?"),
            )
        except Exception:
            logger.debug(
                "on_step_observation_started error:\n%s", traceback.format_exc()
            )

    def on_step_observation_completed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.step_observation`` span as SUCCESS.

        Attributes written
        ------------------
        - ``step.observation``     — observation text returned by the tool / env
        - ``step.status``          — ``"success"``
        - ``step.duration_ms``     — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            obs_id = _resolve_obs_id(event, source)
            extra: dict[str, Any] = {}

            observation = (
                safe_getattr(event, "observation")
                or safe_getattr(event, "result")
                or safe_getattr(event, "output")
            )
            if observation is not None:
                raw = (
                    observation
                    if isinstance(observation, str)
                    else safe_json_dumps(observation)
                )
                extra["step.observation"] = truncate_str(raw, MAX_TEXT_LENGTH)

            self._finish_observation_span(obs_id, ATTR_STATUS_SUCCESS, None, extra)
        except Exception:
            logger.debug(
                "on_step_observation_completed error:\n%s", traceback.format_exc()
            )

    def on_step_observation_failed(self, source: Any, event: Any) -> None:
        """Close the ``crewai.step_observation`` span as ERROR."""
        if not self._is_active():
            return
        try:
            obs_id = _resolve_obs_id(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_observation_span(obs_id, ATTR_STATUS_ERROR, error)
        except Exception:
            logger.debug(
                "on_step_observation_failed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Mid-reasoning annotations (no span lifecycle)
    # =========================================================================

    def on_plan_refinement(self, source: Any, event: Any) -> None:
        """
        Annotate the active reasoning span when the agent refines its plan.

        Plan refinement happens when the agent revises its next steps without
        triggering a full replan (e.g. after a partial tool result).

        Attributes written
        ------------------
        - ``reasoning.refined_steps``      — JSON list of updated planned steps
        - ``reasoning.refinement_count``   — cumulative refinements in this attempt
        - ``reasoning.refinement_reason``  — why the plan was refined (if provided)
        """
        if not self._is_active():
            return
        try:
            reasoning_id = _resolve_reasoning_id(event, source)
            agent_id = _resolve_agent_id(source, event)
            span = (
                self._get_reasoning_span(reasoning_id)
                or self._get_agent_span(agent_id)
            )
            if span is None:
                return

            attrs: dict[str, Any] = {}

            refined_steps = (
                safe_getattr(event, "refined_steps")
                or safe_getattr(event, "steps")
                or safe_getattr(event, "plan")
            )
            if refined_steps is not None:
                if isinstance(refined_steps, (list, tuple)):
                    attrs["reasoning.refined_steps"] = truncate_str(
                        safe_json_dumps(list(refined_steps)), MAX_TEXT_LENGTH
                    )
                else:
                    attrs["reasoning.refined_steps"] = truncate_str(
                        str(refined_steps), MAX_TEXT_LENGTH
                    )

            refinements = safe_getattr(event, "refinements") or safe_getattr(
                event, "refinement_count"
            )
            if refinements is not None:
                try:
                    attrs["reasoning.refinement_count"] = int(refinements)
                except (TypeError, ValueError):
                    pass

            reason = safe_getattr(event, "reason") or safe_getattr(event, "message")
            if reason:
                attrs["reasoning.refinement_reason"] = truncate_str(
                    str(reason), MAX_DESCRIPTION_LENGTH
                )

            _set_span_attributes(span, attrs)
            logger.debug("Plan refinement annotated: reasoning_id=%s", reasoning_id)
        except Exception:
            logger.debug(
                "on_plan_refinement error:\n%s", traceback.format_exc()
            )

    def on_plan_replan_triggered(self, source: Any, event: Any) -> None:
        """
        Annotate the active reasoning (or agent) span when a full replan fires.

        A replan is triggered when the agent decides its current plan is
        unachievable (e.g. a tool failed, a constraint was violated) and it
        must start a fresh chain-of-thought.

        Attributes written
        ------------------
        - ``reasoning.replan_triggered``      — ``True``
        - ``reasoning.replan_reason``         — why the replan was triggered
        - ``reasoning.replan_count``          — cumulative replans in this agent run
        - ``reasoning.completed_steps``       — JSON list of steps completed before
                                                 the replan (context for debugging)
        """
        if not self._is_active():
            return
        try:
            reasoning_id = _resolve_reasoning_id(event, source)
            agent_id = _resolve_agent_id(source, event)
            span = (
                self._get_reasoning_span(reasoning_id)
                or self._get_agent_span(agent_id)
            )
            if span is None:
                return

            attrs: dict[str, Any] = {"plan.replan_triggered": True}

            reason = safe_getattr(event, "reason") or safe_getattr(event, "message")
            if reason:
                attrs["plan.replan_reason"] = truncate_str(
                    str(reason), MAX_DESCRIPTION_LENGTH
                )

            replan_count = safe_getattr(event, "replan_count") or safe_getattr(
                event, "replans"
            )
            if replan_count is not None:
                try:
                    attrs["plan.replan_count"] = int(replan_count)
                except (TypeError, ValueError):
                    pass

            completed = (
                safe_getattr(event, "completed_steps")
                or safe_getattr(event, "steps_done")
            )
            if completed is not None:
                if isinstance(completed, (list, tuple)):
                    attrs["plan.completed_steps"] = truncate_str(
                        safe_json_dumps(list(completed)), MAX_TEXT_LENGTH
                    )
                else:
                    attrs["plan.completed_steps"] = truncate_str(
                        str(completed), MAX_TEXT_LENGTH
                    )

            _set_span_attributes(span, attrs)
            logger.debug(
                "Replan triggered annotation: reasoning_id=%s count=%s",
                reasoning_id,
                replan_count,
            )
        except Exception:
            logger.debug(
                "on_plan_replan_triggered error:\n%s", traceback.format_exc()
            )

    def on_goal_achieved_early(self, source: Any, event: Any) -> None:
        """
        Annotate the active reasoning (or agent) span when the goal is met
        before exhausting all planned steps.

        This fires when an agent detects that the task is already complete
        before running all allocated iterations (``max_iter`` not reached).

        Attributes written
        ------------------
        - ``reasoning.goal_achieved_early``  — ``True``
        - ``agent.step``                     — step at which goal was achieved
        - ``reasoning.steps_remaining``      — steps *not* executed due to early exit
        - ``reasoning.early_exit_reason``    — optional rationale from the agent
        """
        if not self._is_active():
            return
        try:
            reasoning_id = _resolve_reasoning_id(event, source)
            agent_id = _resolve_agent_id(source, event)
            span = (
                self._get_reasoning_span(reasoning_id)
                or self._get_agent_span(agent_id)
            )
            if span is None:
                return

            attrs: dict[str, Any] = {"reasoning.goal_achieved_early": True}

            steps_remaining = (
                safe_getattr(event, "steps_remaining")
                or safe_getattr(event, "remaining_steps")
            )
            if steps_remaining is not None:
                try:
                    attrs["reasoning.steps_remaining"] = int(steps_remaining)
                except (TypeError, ValueError):
                    pass

            current_step = safe_getattr(event, "step") or safe_getattr(
                event, "current_step"
            )
            if current_step is not None:
                try:
                    attrs[ATTR_AGENT_STEP] = int(current_step)
                except (TypeError, ValueError):
                    pass

            reason = (
                safe_getattr(event, "reason")
                or safe_getattr(event, "answer")
                or safe_getattr(event, "message")
            )
            if reason:
                attrs["reasoning.early_exit_reason"] = truncate_str(
                    str(reason), MAX_DESCRIPTION_LENGTH
                )

            _set_span_attributes(span, attrs)
            logger.debug(
                "Goal achieved early: reasoning_id=%s steps_remaining=%s",
                reasoning_id,
                steps_remaining,
            )
        except Exception:
            logger.debug(
                "on_goal_achieved_early error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_agent_span(self, agent_id: Optional[str]) -> Any:
        """Return the open agent span for *agent_id*, or ``None``."""
        if not agent_id:
            return None
        with self._lock:
            return self._agent_spans.get(agent_id)

    def _get_reasoning_span(self, reasoning_id: str) -> Any:
        """Return the open reasoning span for *reasoning_id*, or ``None``."""
        key = _RSN_PREFIX + reasoning_id
        with self._lock:
            entry = self._flow_method_spans.get(key)
        return entry["span"] if entry else None

    def _finish_reasoning_span(
        self,
        reasoning_id: str,
        status: str,
        error: Any,
        extra_attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write final attributes onto the reasoning span and close it."""
        key = _RSN_PREFIX + reasoning_id
        with self._lock:
            entry = self._flow_method_spans.pop(key, None)

        if entry is None:
            logger.debug(
                "_finish_reasoning_span: no open entry for reasoning_id=%s",
                reasoning_id,
            )
            return

        span = entry["span"]
        start_t = entry.get("start_t")

        attrs: dict[str, Any] = {"reasoning.status": status}
        if start_t is not None:
            attrs["reasoning.duration_ms"] = duration_ms_monotonic(start_t)
        if error is not None:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = str(error)
            tb = getattr(error, "__traceback__", None)
            if tb is not None:
                attrs[ATTR_ERROR_STACKTRACE] = "".join(traceback.format_tb(tb))
        if extra_attrs:
            attrs.update(extra_attrs)

        _set_span_attributes(span, attrs)

        if status == ATTR_STATUS_ERROR and hasattr(span, "set_status"):
            try:
                from noveum_trace.core.span import SpanStatus
                span.set_status(SpanStatus.ERROR, str(error) if error else "")
            except Exception:
                pass

        try:
            if hasattr(span, "finish"):
                span.finish()
        except Exception:
            logger.debug(
                "_finish_reasoning_span span.finish() error:\n%s",
                traceback.format_exc(),
            )
        logger.debug(
            "Reasoning span closed: reasoning_id=%s status=%s", reasoning_id, status
        )

    def _finish_observation_span(
        self,
        obs_id: str,
        status: str,
        error: Any,
        extra_attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write final attributes onto the observation span and close it."""
        key = _OBS_PREFIX + obs_id
        with self._lock:
            span = self._memory_op_spans.pop(key, None)
            start_t = self._memory_op_start_times.pop(key, None)

        if span is None:
            logger.debug(
                "_finish_observation_span: no open span for obs_id=%s", obs_id
            )
            return

        attrs: dict[str, Any] = {"step.status": status}
        if start_t is not None:
            attrs["step.duration_ms"] = duration_ms_monotonic(start_t)
        if error is not None:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = str(error)
            tb = getattr(error, "__traceback__", None)
            if tb is not None:
                attrs[ATTR_ERROR_STACKTRACE] = "".join(traceback.format_tb(tb))
        if extra_attrs:
            attrs.update(extra_attrs)

        _set_span_attributes(span, attrs)

        if status == ATTR_STATUS_ERROR and hasattr(span, "set_status"):
            try:
                from noveum_trace.core.span import SpanStatus
                span.set_status(SpanStatus.ERROR, str(error) if error else "")
            except Exception:
                pass

        try:
            if hasattr(span, "finish"):
                span.finish()
        except Exception:
            logger.debug(
                "_finish_observation_span span.finish() error:\n%s",
                traceback.format_exc(),
            )
        logger.debug("Observation span closed: obs_id=%s status=%s", obs_id, status)


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _resolve_reasoning_id(event: Any, source: Any) -> str:
    """Return a stable key for a reasoning attempt."""
    return str(
        safe_getattr(event, "reasoning_id")
        or safe_getattr(event, "thought_id")
        or safe_getattr(event, "id")
        or safe_getattr(event, "run_id")
        or id(event)
    )


def _resolve_obs_id(event: Any, source: Any) -> str:
    """Return a stable key for a step observation."""
    return str(
        safe_getattr(event, "obs_id")
        or safe_getattr(event, "step_id")
        or safe_getattr(event, "observation_id")
        or safe_getattr(event, "id")
        or id(event)
    )


def _resolve_agent_id(source: Any, event: Any) -> Optional[str]:
    """Return the agent_id for this reasoning event, or ``None``."""
    raw = (
        safe_getattr(event, "agent_id")
        or safe_getattr(source, "id")
        or safe_getattr(source, "agent_id")
    )
    return str(raw) if raw is not None else None


def _build_reasoning_start_attributes(
    source: Any, event: Any, reasoning_id: str
) -> dict[str, Any]:
    """Collect span attributes for the opening ``crewai.reasoning`` span."""
    attrs: dict[str, Any] = {"reasoning.id": reasoning_id}

    agent_role = (
        safe_getattr(event, "agent_role")
        or safe_getattr(source, "role")
    )
    if agent_role:
        attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

    task_id = (
        safe_getattr(event, "task_id")
        or safe_getattr(safe_getattr(source, "task"), "id")
    )
    if task_id:
        attrs[ATTR_TASK_ID] = str(task_id)

    # Current plan / thought text
    plan = (
        safe_getattr(event, "plan")
        or safe_getattr(event, "thought")
        or safe_getattr(event, "reasoning")
    )
    if plan:
        attrs["reasoning.plan"] = truncate_str(str(plan), MAX_TEXT_LENGTH)

    # Iteration attempt number
    attempt = (
        safe_getattr(event, "attempt")
        or safe_getattr(event, "iteration")
        or safe_getattr(event, "step")
    )
    if attempt is not None:
        try:
            attrs["reasoning.attempt"] = int(attempt)
        except (TypeError, ValueError):
            pass

    # is_ready: agent signals it believes the task is done
    is_ready = safe_getattr(event, "is_ready")
    if is_ready is not None:
        attrs["reasoning.is_ready"] = bool(is_ready)

    # Planned action (when the thought already includes a tool decision)
    tool_name = (
        safe_getattr(event, "tool_name")
        or safe_getattr(event, "action")
    )
    if tool_name:
        attrs["reasoning.tool_name"] = truncate_str(str(tool_name), 256)

    tool_input = (
        safe_getattr(event, "tool_input")
        or safe_getattr(event, "action_input")
        or safe_getattr(event, "arguments")
    )
    if tool_input is not None:
        raw = (
            tool_input
            if isinstance(tool_input, str)
            else safe_json_dumps(tool_input)
        )
        attrs["reasoning.tool_input"] = truncate_str(raw, MAX_DESCRIPTION_LENGTH)

    return attrs


def _build_observation_start_attributes(
    source: Any, event: Any, obs_id: str
) -> dict[str, Any]:
    """Collect span attributes for the opening ``crewai.step_observation`` span."""
    attrs: dict[str, Any] = {"step.obs_id": obs_id}

    # step.number — 1-based (spec); step.index — 0-based alias for convenience
    step_number = (
        safe_getattr(event, "step_number")
        or safe_getattr(event, "number")
    )
    if step_number is not None:
        try:
            attrs["step.number"] = int(step_number)
        except (TypeError, ValueError):
            pass

    step_index = (
        safe_getattr(event, "step")
        or safe_getattr(event, "step_index")
        or safe_getattr(event, "index")
    )
    if step_index is not None:
        try:
            attrs["step.index"] = int(step_index)
            # Derive step.number from index if not already set
            if "step.number" not in attrs:
                attrs["step.number"] = int(step_index) + 1
        except (TypeError, ValueError):
            pass

    # step.description — human-readable description of this step
    description = safe_getattr(event, "description") or safe_getattr(event, "step_name")
    if description:
        attrs["step.description"] = truncate_str(str(description), 512)

    # Action name (matches tool name in most cases)
    action = (
        safe_getattr(event, "action")
        or safe_getattr(event, "tool_name")
        or safe_getattr(event, "function_name")
    )
    if action:
        attrs["step.action"] = truncate_str(str(action), 256)
        attrs["step.tool"] = attrs["step.action"]

    # Action input
    action_input = (
        safe_getattr(event, "action_input")
        or safe_getattr(event, "tool_input")
        or safe_getattr(event, "arguments")
    )
    if action_input is not None:
        raw = (
            action_input
            if isinstance(action_input, str)
            else safe_json_dumps(action_input)
        )
        attrs["step.action_input"] = truncate_str(raw, MAX_DESCRIPTION_LENGTH)

    # Completion and plan validity flags
    completed_ok = safe_getattr(event, "completed_ok") or safe_getattr(
        event, "success"
    )
    if completed_ok is not None:
        attrs["step.completed_ok"] = bool(completed_ok)

    plan_still_valid = safe_getattr(event, "plan_still_valid")
    if plan_still_valid is not None:
        attrs["step.plan_still_valid"] = bool(plan_still_valid)

    needs_replan = safe_getattr(event, "needs_replan")
    if needs_replan is not None:
        attrs["step.needs_replan"] = bool(needs_replan)

    # Key info extracted from this step's result (summary / insight)
    key_info = safe_getattr(event, "key_info") or safe_getattr(event, "insight")
    if key_info:
        attrs["step.key_info"] = truncate_str(str(key_info), 512)

    # Refinements applied to the plan at this step
    refinements = safe_getattr(event, "refinements")
    if refinements is not None:
        try:
            attrs["step.refinements"] = int(refinements)
        except (TypeError, ValueError):
            pass

    # Agent correlation
    agent_role = (
        safe_getattr(event, "agent_role")
        or safe_getattr(source, "role")
    )
    if agent_role:
        attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

    return attrs

def _set_span_attributes(span: Any, attrs: dict[str, Any]) -> None:
    """Write *attrs* to *span* via ``set_attributes`` or direct dict update."""
    if not attrs or span is None:
        return
    try:
        if hasattr(span, "set_attributes"):
            span.set_attributes(attrs)
        elif hasattr(span, "attributes"):
            span.attributes.update(attrs)
    except Exception as exc:
        logger.debug("_set_span_attributes failed: %s", exc)