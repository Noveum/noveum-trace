"""
Crew-lifecycle event handler mixin for NoveumCrewAIListener.

Handles the top-level Crew events emitted by CrewAI's ``BaseEventListener``:

  - ``on_crew_kickoff_started``   → open root trace + ``crewai.crew`` span;
                                     capture full crew snapshot (name, id, process,
                                     all agents with roles/goals/backstories/tools,
                                     all tasks with descriptions/expected_outputs)
  - ``on_crew_kickoff_completed`` → close trace as SUCCESS; write crew output and
                                     aggregated token/cost totals
  - ``on_crew_kickoff_failed``    → close trace as ERROR; attach exception details

  - ``on_crew_test_started``      → mark span as a test run; optional ``crew.test.inputs``,
                                     ``crew.test.crew_name``
  - ``on_crew_test_completed``    → close test span as SUCCESS
  - ``on_crew_train_started``     → mark span as a training run; optional ``crew.train.crew_name``,
                                     ``crew.train.inputs``
  - ``on_crew_train_completed``   → close training span as SUCCESS; may set
                                     ``crew.train.n_iterations`` / ``crew.train.filename``
                                     from the completion event when not set earlier

State consumed / mutated (all declared in _CrewAIObserverState):
    _lock, _is_shutdown, _crew_spans,
    _total_tokens_by_crew, _total_cost_by_crew,
    _task_start_times, _agent_start_times
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_CREW_AGENT_COUNT,
    ATTR_CREW_AGENT_ROLES,
    ATTR_CREW_AVAILABLE_AGENT_COUNT,
    ATTR_CREW_AVAILABLE_AGENTS,
    ATTR_CREW_DURATION_MS,
    ATTR_CREW_ID,
    ATTR_CREW_MAX_RPM,
    ATTR_CREW_MEMORY,
    ATTR_CREW_NAME,
    ATTR_CREW_OUTPUT,
    ATTR_CREW_PROCESS,
    ATTR_CREW_STATUS,
    ATTR_CREW_TASK_COUNT,
    ATTR_CREW_TOTAL_COST,
    ATTR_CREW_TOTAL_TOKENS,
    ATTR_CREW_VERBOSE,
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_STACKTRACE,
    ATTR_ERROR_TYPE,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    MAX_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
    SPAN_CREW,
)
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase
from noveum_trace.integrations.crewai.crewai_utils import (
    duration_ms_monotonic,
    extract_llm_model_from_agent,
    monotonic_now,
    safe_getattr,
    safe_json_dumps,
    truncate_str,
)

logger = logging.getLogger(__name__)

# Sentinel for ``_finish_crew_span(..., finish_trace_client=...)`` default only.
_FINISH_TRACE_CLIENT_UNSPECIFIED = object()


def _maybe_set_event_crew_name(span: Any, event: Any, attr_key: str) -> None:
    """Copy ``CrewBaseEvent.crew_name`` onto the span when present and non-empty."""
    cn = safe_getattr(event, "crew_name")
    if cn is not None and str(cn).strip() != "":
        span.set_attribute(attr_key, str(cn))


class _CrewHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI Crew-lifecycle events.

    All public methods match the ``BaseEventListener`` callback signature::

        def on_crew_kickoff_started(self, source, event): ...

    ``source`` is the ``Crew`` object; ``event`` carries event-specific payload.
    Every method is fully exception-shielded — a bug here can never propagate
    into user application code.
    """

    # =========================================================================
    # Crew kickoff — start
    # =========================================================================

    def on_crew_kickoff_started(self, source: Any, event: Any) -> None:
        """
        Open the root Noveum trace and ``crewai.crew`` span.

        Captures a full snapshot of the Crew at kickoff time (subject to
        ``capture_inputs``, ``capture_agent_snapshot``, and ``capture_crew_snapshot``):
        - Crew identity: ``name``, ``id``, ``process``, ``memory``, ``verbose``
        - Agent roster: roles, goals, backstories, tool names, LLM model, delegation
        - Task list: descriptions, expected outputs, assigned agent roles
        """
        if not self._is_active():
            return
        try:
            # crew_id may come from the event or from source.id
            crew_id = str(
                safe_getattr(event, "crew_id")
                or safe_getattr(source, "id")
                or id(source)
            )

            client = self._get_client()
            if client is None:
                logger.debug(
                    "NoveumCrewAIListener: no client available — skipping crew trace"
                )
                return

            # Gather span attributes from the Crew object
            attrs = _build_crew_attributes(
                source,
                event,
                capture_inputs=self.capture_inputs,
                capture_agent_snapshot=self.capture_agent_snapshot,
                capture_crew_snapshot=self.capture_crew_snapshot,
            )

            # start_time for duration calculation
            start_t = monotonic_now()

            # Create a root trace and the crew span
            crew_name = attrs.get(ATTR_CREW_NAME) or "crew"
            prefix = (self.trace_name_prefix or "").strip().rstrip(".") or "crewai"
            trace_name = f"{prefix}.{crew_name}.kickoff"

            # set_as_current=False avoids polluting thread-local trace context;
            # CrewAI uses threads and each crew must own its own context.
            trace = client.start_trace(name=trace_name, set_as_current=False)
            span = trace.create_span(name=SPAN_CREW, attributes=attrs)

            with self._lock:
                self._crew_spans[crew_id] = {
                    "trace": trace,
                    "span": span,
                    "start_t": start_t,
                }
                # Ensure accumulator buckets exist
                self._total_tokens_by_crew.setdefault(crew_id, 0)
                self._total_cost_by_crew.setdefault(crew_id, 0.0)

                # Build task_id → crew_id reverse map so that task/agent/llm
                # spans can resolve their correct parent trace even when two
                # crew lifetimes overlap.
                tasks = safe_getattr(source, "tasks") or []
                for t_obj in tasks:
                    t_id = safe_getattr(t_obj, "id")
                    if t_id:
                        self._task_to_crew_id[str(t_id)] = crew_id

            logger.debug(
                "CrewAI trace started: crew_id=%s trace=%s",
                crew_id,
                getattr(trace, "trace_id", "?"),
            )

        except Exception:
            logger.debug("on_crew_kickoff_started error:\n%s", traceback.format_exc())

    # =========================================================================
    # Crew kickoff — completed (success)
    # =========================================================================

    def on_crew_kickoff_completed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.crew`` span and root trace as SUCCESS.

        Writes:
        - ``crew.output``        — final CrewOutput text (truncated to MAX_TEXT_LENGTH)
        - ``crew.total_tokens``  — sum of all LLM call token counts under this crew
        - ``crew.total_cost``    — sum of all LLM call costs under this crew (USD)
        - ``crew.duration_ms``   — wall-clock duration of the kickoff
        - ``crew.status``        — ``"success"``
        """
        if not self._is_active():
            return
        try:
            crew_id = str(
                safe_getattr(event, "crew_id")
                or safe_getattr(source, "id")
                or id(source)
            )
            self._finish_crew_span(
                crew_id=crew_id,
                status=ATTR_STATUS_SUCCESS,
                output=safe_getattr(event, "output"),
                error=None,
            )
        except Exception:
            logger.debug("on_crew_kickoff_completed error:\n%s", traceback.format_exc())

    # =========================================================================
    # Crew kickoff — failed (error)
    # =========================================================================

    def on_crew_kickoff_failed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.crew`` span and root trace as ERROR.

        Writes:
        - ``error.type``         — exception class name
        - ``error.message``      — exception message
        - ``error.stacktrace``   — full traceback (when available on event)
        - ``crew.status``        — ``"error"``
        - ``crew.duration_ms``   — wall-clock duration until failure
        """
        if not self._is_active():
            return
        try:
            crew_id = str(
                safe_getattr(event, "crew_id")
                or safe_getattr(source, "id")
                or id(source)
            )
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_crew_span(
                crew_id=crew_id,
                status=ATTR_STATUS_ERROR,
                output=None,
                error=error,
            )
        except Exception:
            logger.debug("on_crew_kickoff_failed error:\n%s", traceback.format_exc())

    # =========================================================================
    # Test handlers
    # =========================================================================

    def on_crew_test_started(self, source: Any, event: Any) -> None:
        """
        ``CrewTestStarted``: annotate the crew span as a test run.

        Records ``crew.mode = "test"`` and, when available,
        ``crew.test.n_iterations`` on the open crew span.

        When ``capture_inputs`` is true and ``event.inputs`` is set, writes
        ``crew.test.inputs`` (JSON, truncated like ``crew.inputs`` on kickoff)
        so test-run inputs are distinguishable from kickoff ``crew.inputs``.

        When ``CrewBaseEvent.crew_name`` is set, writes ``crew.test.crew_name``
        (event label; kickoff may already have ``crew.name`` from ``source``).
        """
        if not self._is_active():
            return
        try:
            crew_id = str(
                safe_getattr(event, "crew_id")
                or safe_getattr(source, "id")
                or id(source)
            )
            span = self._get_crew_span(crew_id)
            if span is None:
                return
            span.set_attribute("crew.mode", "test")
            _maybe_set_event_crew_name(span, event, "crew.test.crew_name")
            n_iter = safe_getattr(event, "n_iterations")
            if n_iter is not None:
                span.set_attribute("crew.test.n_iterations", int(n_iter))
            eval_llm = safe_getattr(event, "eval_llm")
            if eval_llm:
                span.set_attribute("crew.test.eval_llm", str(eval_llm))
            if self.capture_inputs:
                test_inputs = safe_getattr(event, "inputs")
                if test_inputs is not None:
                    span.set_attribute(
                        "crew.test.inputs",
                        truncate_str(safe_json_dumps(test_inputs), MAX_TEXT_LENGTH),
                    )
        except Exception:
            logger.debug("on_crew_test_started error:\n%s", traceback.format_exc())

    def on_crew_test_completed(self, source: Any, event: Any) -> None:
        """``CrewTestCompletedEvent``: close the crew span as SUCCESS.

        Quality score and model come from ``CrewTestResultEvent`` (a separate
        event fired before this one), so we only handle span teardown here.
        """
        if not self._is_active():
            return
        try:
            crew_id = str(
                safe_getattr(event, "crew_id")
                or safe_getattr(source, "id")
                or id(source)
            )
            self._finish_crew_span(
                crew_id=crew_id,
                status=ATTR_STATUS_SUCCESS,
                output=safe_getattr(event, "output"),
                error=None,
                extra_attrs={"crew.mode": "test"},
            )
        except Exception:
            logger.debug("on_crew_test_completed error:\n%s", traceback.format_exc())

    def on_crew_test_result(self, source: Any, event: Any) -> None:
        """``CrewTestResultEvent``: write quality score and eval model to the span.

        ``CrewTestResultEvent`` carries:
          - ``quality``           — float score (0–10)
          - ``execution_duration``— float seconds
          - ``model``             — the evaluator LLM model string
          - ``crew_name``         — str (written as ``crew.test.crew_name``)

        Fires *before* ``CrewTestCompletedEvent``, so the span is still open.
        """
        if not self._is_active():
            return
        try:
            crew_id = str(
                safe_getattr(event, "crew_id")
                or safe_getattr(source, "id")
                or id(source)
            )
            span = self._get_crew_span(crew_id)
            if span is None:
                return
            _maybe_set_event_crew_name(span, event, "crew.test.crew_name")
            quality = safe_getattr(event, "quality")
            if quality is not None:
                try:
                    span.set_attribute("crew.quality_score", float(quality))
                except (TypeError, ValueError):
                    pass
            exec_dur = safe_getattr(event, "execution_duration")
            if exec_dur is not None:
                try:
                    span.set_attribute(
                        "crew.test.execution_duration_s", float(exec_dur)
                    )
                except (TypeError, ValueError):
                    pass
            model = safe_getattr(event, "model")
            if model:
                span.set_attribute("crew.test_model", str(model))
        except Exception:
            logger.debug("on_crew_test_result error:\n%s", traceback.format_exc())

    # =========================================================================
    # Training handlers
    # =========================================================================

    def on_crew_train_started(self, source: Any, event: Any) -> None:
        """``CrewTrainStarted``: annotate the crew span as a training run.

        When ``CrewBaseEvent.crew_name`` is set, writes ``crew.train.crew_name``.

        When ``capture_inputs`` is true and ``event.inputs`` is set, writes
        ``crew.train.inputs`` (JSON, truncated like ``crew.test.inputs``).
        """
        if not self._is_active():
            return
        try:
            crew_id = str(
                safe_getattr(event, "crew_id")
                or safe_getattr(source, "id")
                or id(source)
            )
            span = self._get_crew_span(crew_id)
            if span is None:
                return
            span.set_attribute("crew.mode", "train")
            _maybe_set_event_crew_name(span, event, "crew.train.crew_name")
            n_iter = safe_getattr(event, "n_iterations")
            if n_iter is not None:
                span.set_attribute("crew.train.n_iterations", int(n_iter))
            filename = safe_getattr(event, "filename")
            if filename:
                span.set_attribute("crew.train.filename", str(filename))
            if self.capture_inputs:
                train_inputs = safe_getattr(event, "inputs")
                if train_inputs is not None:
                    span.set_attribute(
                        "crew.train.inputs",
                        truncate_str(safe_json_dumps(train_inputs), MAX_TEXT_LENGTH),
                    )
        except Exception:
            logger.debug("on_crew_train_started error:\n%s", traceback.format_exc())

    def on_crew_train_completed(self, source: Any, event: Any) -> None:
        """``CrewTrainCompleted``: close the crew span as SUCCESS.

        Copies ``n_iterations`` and ``filename`` from the completion event onto
        the span (with ``crew.mode = "train"``) so values are present even when
        ``on_crew_train_started`` did not run or could not attach to a span.
        """
        if not self._is_active():
            return
        try:
            crew_id = str(
                safe_getattr(event, "crew_id")
                or safe_getattr(source, "id")
                or id(source)
            )
            extra_attrs: dict[str, Any] = {"crew.mode": "train"}
            n_iter = safe_getattr(event, "n_iterations")
            if n_iter is not None:
                try:
                    extra_attrs["crew.train.n_iterations"] = int(n_iter)
                except (TypeError, ValueError):
                    pass
            filename = safe_getattr(event, "filename")
            if filename:
                extra_attrs["crew.train.filename"] = str(filename)
            self._finish_crew_span(
                crew_id=crew_id,
                status=ATTR_STATUS_SUCCESS,
                output=None,
                error=None,
                extra_attrs=extra_attrs,
            )
        except Exception:
            logger.debug("on_crew_train_completed error:\n%s", traceback.format_exc())

    def on_crew_train_failed(self, source: Any, event: Any) -> None:
        """``CrewTrainFailed``: close the crew span as ERROR."""
        if not self._is_active():
            return
        try:
            crew_id = str(
                safe_getattr(event, "crew_id")
                or safe_getattr(source, "id")
                or id(source)
            )
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_crew_span(
                crew_id=crew_id,
                status=ATTR_STATUS_ERROR,
                output=None,
                error=error,
                extra_attrs={"crew.mode": "train"},
            )
        except Exception:
            logger.debug("on_crew_train_failed error:\n%s", traceback.format_exc())

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _finish_crew_span(
        self,
        crew_id: str,
        status: str,
        output: Any,
        error: Any,
        extra_attrs: Optional[dict[str, Any]] = None,
        *,
        finish_trace_client: Any = _FINISH_TRACE_CLIENT_UNSPECIFIED,
    ) -> None:
        """
        Finish the crew span + root trace for *crew_id*.

        Writes aggregated token/cost totals, duration, output, and status onto
        the span before closing it and the trace.  The span and trace entries
        are removed from the state dicts atomically under the lock.

        Pass ``finish_trace_client`` when ``_get_client()`` would return ``None``
        but a client is still required (e.g. listener shutdown force-close).
        """
        with self._lock:
            entry = self._crew_spans.pop(crew_id, None)
            total_tokens = self._total_tokens_by_crew.pop(crew_id, 0)
            total_cost = self._total_cost_by_crew.pop(crew_id, 0.0)
            # Remove task→crew mappings for this crew so they don't linger
            stale_task_ids = [
                tid for tid, cid in self._task_to_crew_id.items() if cid == crew_id
            ]
            for tid in stale_task_ids:
                self._task_to_crew_id.pop(tid, None)

        if entry is None:
            logger.debug("_finish_crew_span: no open span for crew_id=%s", crew_id)
            return

        span = entry["span"]
        trace = entry["trace"]
        start_t = entry.get("start_t")

        attrs: dict[str, Any] = {ATTR_CREW_STATUS: status}

        # Duration
        if start_t is not None:
            attrs[ATTR_CREW_DURATION_MS] = duration_ms_monotonic(start_t)

        # Aggregated token / cost totals
        if total_tokens:
            attrs[ATTR_CREW_TOTAL_TOKENS] = total_tokens
        if total_cost:
            attrs[ATTR_CREW_TOTAL_COST] = round(total_cost, 8)

        # Crew output
        if output is not None and getattr(self, "capture_outputs", True):
            raw = _extract_crew_output_text(output)
            if raw:
                attrs[ATTR_CREW_OUTPUT] = truncate_str(raw, MAX_TEXT_LENGTH)

        # Error details
        if error is not None:
            if isinstance(error, BaseException):
                attrs[ATTR_ERROR_TYPE] = type(error).__name__
                msg = str(error)
                attrs[ATTR_ERROR_MESSAGE] = msg
                attrs["crew.error"] = msg
                tb = getattr(error, "__traceback__", None)
                if tb is not None:
                    attrs[ATTR_ERROR_STACKTRACE] = "".join(traceback.format_tb(tb))
            else:
                # CrewAI sometimes emits ``event.error`` as a plain string (no
                # exception object), which would otherwise record error.type="str".
                msg = str(error)
                attrs["crew.error"] = msg
                attrs[ATTR_ERROR_MESSAGE] = msg
                if ":" in msg:
                    head, tail = msg.split(":", 1)
                    head_stripped = head.strip()
                    if head_stripped.isidentifier() and head_stripped[:1].isalpha():
                        attrs[ATTR_ERROR_TYPE] = head_stripped
                        attrs[ATTR_ERROR_MESSAGE] = tail.strip() or msg
                    else:
                        attrs[ATTR_ERROR_TYPE] = "Error"
                else:
                    attrs[ATTR_ERROR_TYPE] = "Error"

        # Caller-supplied extras (e.g. crew.mode)
        if extra_attrs:
            attrs.update(extra_attrs)

        # Write and close span
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
                "_finish_crew_span span.finish() error:\n%s",
                traceback.format_exc(),
            )

        # Close the root trace
        try:
            if finish_trace_client is not _FINISH_TRACE_CLIENT_UNSPECIFIED:
                client = finish_trace_client
            else:
                client = self._get_client()
            if client and trace is not None and hasattr(client, "finish_trace"):
                client.finish_trace(trace)
        except Exception:
            logger.debug(
                "_finish_crew_span client.finish_trace() error:\n%s",
                traceback.format_exc(),
            )

        logger.debug(
            "CrewAI trace finished: crew_id=%s status=%s tokens=%d cost=%.6f",
            crew_id,
            status,
            total_tokens,
            total_cost,
        )


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _crew_memory_enabled(crew: Any) -> bool:
    """
    Whether crew memory is turned on, as a plain bool for span attributes.

    CrewAI's ``memory`` field may be ``True``, ``False``, ``None``, or a
    ``Memory`` / ``MemoryScope`` / ``MemorySlice`` instance.  Writing the
    instance onto a span can break JSON export and show up as ``false`` in
    UIs; we always normalize to ``True``/``False``.

    After kickoff materialization, ``_memory`` may be set even when the public
    field reads ``False`` in edge cases — treat a non-``None`` ``_memory`` as
    enabled.
    """
    try:
        mem = safe_getattr(crew, "memory")
        if mem is True:
            return True
        if mem not in (False, None):
            return True
        return safe_getattr(crew, "_memory") is not None
    except Exception:
        return False


def _build_crew_attributes(
    source: Any,
    event: Any,
    *,
    capture_inputs: bool = True,
    capture_agent_snapshot: bool = True,
    capture_crew_snapshot: bool = True,
) -> dict[str, Any]:
    """
    Collect all span attributes for the root ``crewai.crew`` span.

    Snapshots the Crew at kickoff time so attribute values reflect the
    configured state, not post-run mutations.

    ``capture_*`` mirror :class:`NoveumCrewAIListener` flags so users can omit
    sensitive or bulky inputs, agent/task JSON snapshots, etc.
    """
    attrs: dict[str, Any] = {}

    # --- Crew identity -------------------------------------------------------
    crew_id = str(safe_getattr(event, "crew_id") or safe_getattr(source, "id") or "")
    if crew_id:
        attrs[ATTR_CREW_ID] = crew_id

    name = safe_getattr(source, "name")
    if name:
        attrs[ATTR_CREW_NAME] = str(name)

    process = safe_getattr(source, "process")
    if process is not None:
        attrs[ATTR_CREW_PROCESS] = str(safe_getattr(process, "value") or process)

    # --- Kickoff inputs --------------------------------------------------
    if capture_inputs:
        inputs = safe_getattr(event, "inputs") or safe_getattr(source, "inputs")
        if inputs is not None:
            attrs["crew.inputs"] = truncate_str(
                safe_json_dumps(inputs), MAX_TEXT_LENGTH
            )

    attrs[ATTR_CREW_MEMORY] = _crew_memory_enabled(source)

    for flag_attr, span_key in (
        ("verbose", ATTR_CREW_VERBOSE),
        ("max_rpm", ATTR_CREW_MAX_RPM),
    ):
        val = safe_getattr(source, flag_attr)
        if val is not None:
            attrs[span_key] = val

    # --- Agent roster — crew.agents_snapshot JSON array -------------------
    if capture_agent_snapshot:
        agents = safe_getattr(source, "agents") or []
        if agents:
            attrs[ATTR_CREW_AGENT_COUNT] = len(agents)
            agents_snapshot = []
            roles = []
            available_agents: list[str] = []
            for agent in agents:
                role = safe_getattr(agent, "role")
                entry: dict[str, Any] = {}
                if safe_getattr(agent, "id") is not None:
                    entry["id"] = str(safe_getattr(agent, "id"))
                # Stable "available agent" label for quick scanning/filtering.
                label = (
                    safe_getattr(agent, "role")
                    or safe_getattr(agent, "name")
                    or safe_getattr(agent, "id")
                )
                if label is not None:
                    available_agents.append(str(label))
                if role:
                    entry["role"] = str(role)
                    roles.append(str(role))
                for sub_attr in ("goal", "backstory"):
                    val = safe_getattr(agent, sub_attr)
                    if val:
                        entry[sub_attr] = truncate_str(str(val), MAX_DESCRIPTION_LENGTH)
                for sub_attr in ("allow_delegation", "max_iter"):
                    val = safe_getattr(agent, sub_attr)
                    if val is not None:
                        entry[sub_attr] = val
                model = extract_llm_model_from_agent(agent)
                if model:
                    entry["llm_model"] = model
                tools = safe_getattr(agent, "tools") or []
                entry["tools_names"] = [
                    str(safe_getattr(t, "name") or t) for t in tools if t is not None
                ]
                agents_snapshot.append(entry)
            if agents_snapshot:
                attrs["crew.agents_snapshot"] = truncate_str(
                    safe_json_dumps(agents_snapshot), MAX_TEXT_LENGTH
                )
            if roles:
                attrs[ATTR_CREW_AGENT_ROLES] = safe_json_dumps(roles)
            if available_agents:
                attrs[ATTR_CREW_AVAILABLE_AGENTS] = safe_json_dumps(available_agents)
                attrs[ATTR_CREW_AVAILABLE_AGENT_COUNT] = len(available_agents)

    # --- Task list — crew.tasks_snapshot JSON array -----------------------
    if capture_crew_snapshot:
        tasks = safe_getattr(source, "tasks") or []
        if tasks:
            attrs[ATTR_CREW_TASK_COUNT] = len(tasks)
            tasks_snapshot = []
            for task in tasks:
                t_entry: dict[str, Any] = {}
                task_id = safe_getattr(task, "id")
                if task_id:
                    t_entry["id"] = str(task_id)
                task_name = safe_getattr(task, "name")
                if task_name:
                    t_entry["name"] = str(task_name)
                description = safe_getattr(task, "description")
                if description:
                    t_entry["description"] = truncate_str(
                        str(description), MAX_DESCRIPTION_LENGTH
                    )
                expected_output = safe_getattr(task, "expected_output")
                if expected_output:
                    t_entry["expected_output"] = truncate_str(
                        str(expected_output), MAX_DESCRIPTION_LENGTH
                    )
                agent = safe_getattr(task, "agent")
                if agent is not None:
                    agent_role = safe_getattr(agent, "role")
                    if agent_role:
                        t_entry["agent_role"] = str(agent_role)
                for flag_attr in ("human_input", "async_execution"):
                    val = safe_getattr(task, flag_attr)
                    if val is not None:
                        t_entry[flag_attr] = val
                t_tools = safe_getattr(task, "tools") or []
                t_entry["tools_names"] = [
                    str(safe_getattr(tt, "name") or tt)
                    for tt in t_tools
                    if tt is not None
                ]
                context = safe_getattr(task, "context") or []
                if context:
                    try:
                        ctx_descs = []
                        for ctx_task in context:
                            desc = safe_getattr(ctx_task, "description")
                            if desc:
                                ctx_descs.append(str(desc))
                        t_entry["context_tasks"] = ctx_descs
                    except Exception:
                        pass
                tasks_snapshot.append(t_entry)
            if tasks_snapshot:
                attrs["crew.tasks_snapshot"] = truncate_str(
                    safe_json_dumps(tasks_snapshot), MAX_TEXT_LENGTH
                )

    return attrs


def _extract_crew_output_text(output: Any) -> Optional[str]:
    """
    Best-effort extraction of text from a ``CrewOutput`` object or plain value.

    ``CrewOutput`` exposes ``.raw`` (str), ``.pydantic`` (Pydantic model), and
    ``.json_dict`` (dict).  Falls back to ``str(output)`` for any other type.
    """
    if output is None:
        return None
    if isinstance(output, str):
        return output

    # CrewOutput.raw is the primary serialised result
    raw = safe_getattr(output, "raw")
    if isinstance(raw, str) and raw:
        return raw

    # Pydantic model result
    pydantic_obj = safe_getattr(output, "pydantic")
    if pydantic_obj is not None:
        for method in ("model_dump_json", "json"):
            fn = getattr(pydantic_obj, method, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass

    # Dict result
    json_dict = safe_getattr(output, "json_dict")
    if isinstance(json_dict, dict):
        try:
            import json

            return json.dumps(json_dict, default=str)
        except Exception:
            pass

    # Final fallback
    try:
        return str(output)
    except Exception:
        return None
