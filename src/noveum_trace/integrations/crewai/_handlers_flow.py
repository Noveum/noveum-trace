"""
Flow-lifecycle event handler mixin for NoveumCrewAIListener.

Handles CrewAI ``BaseEventListener`` Flow events.  CrewAI Flows are
event-driven pipelines built with ``@start``, ``@listen``, and ``@router``
decorators; they can pause for human input and resume.

Span hierarchy::

    crewai.flow                   ← one per Flow.kickoff() call
      crewai.flow.method          ← one per @start / @listen / @router method

Event families handled:

  Flow lifecycle:
  - ``on_flow_started``                → open ``crewai.flow`` span;
                                          capture flow_name, flow_id, inputs, structure
  - ``on_flow_finished``               → close as SUCCESS; write result
  - ``on_flow_failed``                 → close as ERROR
  - ``on_flow_plot``                   → annotate open flow span (plot marker + structure)

  Method execution (per decorated method):
  - ``on_method_execution_started``    → open ``crewai.flow.method`` child span;
                                          capture method_name, flow_name,
                                          method_type (start/listen/router), params
  - ``on_method_execution_finished``   → close method span as SUCCESS; write output
  - ``on_method_execution_failed``     → close method span as ERROR

  Pause / resume (human-in-the-loop):
  - ``on_flow_paused``                 → annotate flow span with state snapshot,
                                          pause message, possible_outcomes list
  - ``on_flow_input_requested``        → annotate span with input prompt + context
  - ``on_flow_input_received``         → annotate span with received value
  - ``on_human_feedback_requested``    → annotate span with feedback prompt
  - ``on_human_feedback_received``     → annotate span with feedback text

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown,
    _flow_spans, _flow_method_spans, _flow_start_times
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_STACKTRACE,
    ATTR_ERROR_TYPE,
    ATTR_FLOW_DURATION_MS,
    ATTR_FLOW_ID,
    ATTR_FLOW_METHOD_DURATION_MS,
    ATTR_FLOW_METHOD_ID,
    ATTR_FLOW_METHOD_NAME,
    ATTR_FLOW_METHOD_STATUS,
    ATTR_FLOW_METHOD_TRIGGER,
    ATTR_FLOW_NAME,
    ATTR_FLOW_PLOT_EMITTED,
    ATTR_FLOW_STATE,
    ATTR_FLOW_STATUS,
    ATTR_FLOW_STRUCTURE,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    MAX_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
    SPAN_FLOW,
    SPAN_FLOW_METHOD,
)
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase
from noveum_trace.integrations.crewai.crewai_utils import (
    duration_ms_monotonic,
    monotonic_now,
    safe_getattr,
    safe_json_dumps,
    set_span_attributes,
    truncate_str,
)

logger = logging.getLogger(__name__)

_FINISH_TRACE_CLIENT_UNSPECIFIED = object()

# Key used in _flow_method_spans: "{flow_id}::{method_name}::{method_id}"
_METHOD_KEY_SEP = "::"


class _FlowHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI Flow-lifecycle events.

    All public methods match the ``BaseEventListener`` callback signature::

        def on_flow_started(self, source, event): ...

    ``source`` is the ``Flow`` instance; ``event`` carries event-specific
    payload.  Every method is fully exception-shielded.
    """

    # =========================================================================
    # Flow lifecycle — started / finished / failed
    # =========================================================================

    def on_flow_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.flow`` root span for a Flow execution.

        Attributes set at span open
        ---------------------------
        - ``flow.id``      — unique flow run identifier
        - ``flow.name``    — Flow class name or configured name
        - ``flow.inputs``     — JSON of initial inputs passed to ``kickoff()``
        - ``flow.state``      — JSON snapshot of the initial Flow state object
        - ``flow.structure``  — JSON graph from ``build_flow_structure`` (nodes, edges, …)
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            attrs = _build_flow_start_attributes(
                source,
                event,
                flow_id,
                capture_inputs=getattr(self, "capture_inputs", True),
                capture_outputs=getattr(self, "capture_outputs", True),
            )
            start_t = monotonic_now()

            client = self._get_client()
            if client is None:
                logger.debug("NoveumCrewAIListener: no client — skipping flow trace")
                return

            # Flows may run inside a Crew (nested) or standalone.
            # For standalone flows, create a root trace; nested flows attach
            # as a child span of the open crew span.
            crew_id = _resolve_crew_id(source, event)
            parent_span = self._get_crew_span_for_flow(crew_id)

            if parent_span is None:
                # Standalone flow — create its own root trace.
                # set_as_current=False avoids thread-local pollution.
                flow_name = attrs.get(ATTR_FLOW_NAME) or "flow"
                trace = client.start_trace(
                    name=f"crewai.flow.{flow_name}", set_as_current=False
                )
                # Create the flow root span directly from the trace
                # (no parent span yet — this IS the root span).
                span = trace.create_span(name=SPAN_FLOW, attributes=attrs)
            else:
                trace = None  # Embedded in existing crew trace
                span = self._create_child_span(
                    SPAN_FLOW,
                    parent_span=parent_span,
                    attributes=attrs,
                )

            with self._lock:
                self._flow_spans[flow_id] = {
                    "span": span,
                    "trace": trace,  # None for nested flows
                    "start_t": start_t,
                }

            logger.debug(
                "Flow span opened: flow_id=%s name=%s standalone=%s",
                flow_id,
                attrs.get(ATTR_FLOW_NAME, "?"),
                trace is not None,
            )
        except Exception:
            logger.debug("on_flow_started error:\n%s", traceback.format_exc())

    def on_flow_finished(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.flow`` span as SUCCESS.

        Attributes written
        ------------------
        - ``flow.result``      — final flow result (truncated)
        - ``flow.state``       — final state snapshot (JSON)
        - ``flow.status``      — ``"success"``
        - ``flow.duration_ms`` — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            result = safe_getattr(event, "result") or safe_getattr(event, "output")
            state = safe_getattr(event, "state") or safe_getattr(source, "state")
            extra: dict[str, Any] = {}
            if result is not None and getattr(self, "capture_outputs", True):
                extra["flow.result"] = truncate_str(
                    result if isinstance(result, str) else safe_json_dumps(result),
                    MAX_TEXT_LENGTH,
                )
            if state is not None:
                extra[ATTR_FLOW_STATE] = truncate_str(
                    safe_json_dumps(state), MAX_TEXT_LENGTH
                )
            self._finish_flow_span(flow_id, ATTR_STATUS_SUCCESS, None, extra)
        except Exception:
            logger.debug("on_flow_finished error:\n%s", traceback.format_exc())

    def on_flow_failed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.flow`` span as ERROR.

        Attributes written
        ------------------
        - ``error.type``       — exception class name
        - ``error.message``    — error message
        - ``error.stacktrace`` — formatted traceback
        - ``flow.status``      — ``"error"``
        - ``flow.duration_ms`` — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_flow_span(flow_id, ATTR_STATUS_ERROR, error)
        except Exception:
            logger.debug("on_flow_failed error:\n%s", traceback.format_exc())

    def on_flow_plot(self, source: Any, event: Any) -> None:
        """
        Annotate the open ``crewai.flow`` span when CrewAI emits ``FlowPlotEvent``.

        ``FlowPlotEvent`` does not carry graph data; we still attach a fresh
        ``flow.structure`` from ``build_flow_structure(source)`` when available,
        and set ``flow.plot_emitted`` so backends can tell ``plot()`` ran.
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            with self._lock:
                entry = self._flow_spans.get(flow_id)
            if not entry:
                return
            span = entry.get("span")
            if span is None:
                return
            payload: dict[str, Any] = {ATTR_FLOW_PLOT_EMITTED: True}
            structure_json = _try_flow_structure_json(source)
            if structure_json:
                payload[ATTR_FLOW_STRUCTURE] = structure_json
            set_span_attributes(span, payload)
        except Exception:
            logger.debug("on_flow_plot error:\n%s", traceback.format_exc())

    # =========================================================================
    # Method execution — started / finished / failed
    # =========================================================================

    def on_method_execution_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.flow.method`` child span for a single method execution.

        Attributes set at span open
        ---------------------------
        - ``flow.method.id``       — stable key for this execution instance
        - ``flow.method.name``     — decorated method name
        - ``flow.name``            — parent Flow class / name
        - ``flow.method.type``     — ``"start"`` | ``"listen"`` | ``"router"``
        - ``flow.method.trigger``  — event or method that triggered this listener
        - ``flow.method.params``   — JSON of method call parameters
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            method_id = _resolve_method_id(event)
            method_key = _make_method_key(flow_id, method_id)

            attrs = _build_method_start_attributes(
                source,
                event,
                method_id,
                flow_id,
                capture_inputs=getattr(self, "capture_inputs", True),
                capture_outputs=getattr(self, "capture_outputs", True),
            )
            start_t = monotonic_now()

            # Parent: flow span
            parent_span = self._get_flow_span(flow_id)

            span = self._create_child_span(
                SPAN_FLOW_METHOD,
                parent_span=parent_span,
                attributes=attrs,
            )

            with self._lock:
                self._flow_method_spans[method_key] = {
                    "span": span,
                    "start_t": start_t,
                }

            logger.debug(
                "Flow method span opened: method_key=%s type=%s",
                method_key,
                attrs.get("flow.method.type", "?"),
            )
        except Exception:
            logger.debug(
                "on_method_execution_started error:\n%s", traceback.format_exc()
            )

    def on_method_execution_finished(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.flow.method`` span as SUCCESS.

        Attributes written
        ------------------
        - ``flow.method.output``      — method return value (JSON / str, truncated)
        - ``flow.method.status``      — ``"success"``
        - ``flow.method.duration_ms`` — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            method_id = _resolve_method_id(event)
            method_key = _make_method_key(flow_id, method_id)
            output = safe_getattr(event, "output") or safe_getattr(event, "result")
            self._finish_method_span(method_key, ATTR_STATUS_SUCCESS, output, None)
        except Exception:
            logger.debug(
                "on_method_execution_finished error:\n%s", traceback.format_exc()
            )

    def on_method_execution_failed(self, source: Any, event: Any) -> None:
        """Close the ``crewai.flow.method`` span as ERROR."""
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            method_id = _resolve_method_id(event)
            method_key = _make_method_key(flow_id, method_id)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_method_span(method_key, ATTR_STATUS_ERROR, None, error)
        except Exception:
            logger.debug(
                "on_method_execution_failed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Pause / resume — human-in-the-loop
    # =========================================================================

    def on_flow_paused(self, source: Any, event: Any) -> None:
        """
        Annotate the flow span when it pauses waiting for external input.

        Attributes written
        ------------------
        - ``flow.pause``                 — ``True``
        - ``flow.pause_message``         — human-readable pause reason
        - ``flow.pause_state``           — JSON snapshot of current state
        - ``flow.pause_possible_outcomes`` — JSON list of valid next steps
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            span = self._get_flow_span(flow_id)
            if span is None:
                return

            pause_attrs: dict[str, Any] = {"flow.pause": True}

            message = safe_getattr(event, "message") or safe_getattr(event, "reason")
            if message:
                pause_attrs["flow.pause_message"] = truncate_str(
                    str(message), MAX_DESCRIPTION_LENGTH
                )

            state = safe_getattr(event, "state") or safe_getattr(source, "state")
            if state is not None:
                pause_attrs["flow.pause_state"] = truncate_str(
                    safe_json_dumps(state), MAX_TEXT_LENGTH
                )

            outcomes = safe_getattr(event, "possible_outcomes") or safe_getattr(
                event, "outcomes"
            )
            if outcomes is not None:
                try:
                    outcomes_list = (
                        list(outcomes)
                        if isinstance(outcomes, (list, tuple))
                        else [str(outcomes)]
                    )
                    pause_attrs["flow.pause_possible_outcomes"] = safe_json_dumps(
                        outcomes_list
                    )
                except Exception:
                    pause_attrs["flow.pause_possible_outcomes"] = str(outcomes)

            set_span_attributes(span, pause_attrs)
            logger.debug("Flow paused: flow_id=%s", flow_id)
        except Exception:
            logger.debug("on_flow_paused error:\n%s", traceback.format_exc())

    def on_flow_input_requested(self, source: Any, event: Any) -> None:
        """
        Annotate the flow span when it requests structured input.

        Attributes written
        ------------------
        - ``flow.input_requested``        — ``True``
        - ``flow.input_prompt``           — prompt shown to the user
        - ``flow.input_context``          — additional context JSON
        - ``flow.input_field``            — state field name expecting the input
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            span = self._get_flow_span(flow_id)
            if span is None:
                return

            req_attrs: dict[str, Any] = {"flow.input_requested": True}

            prompt = safe_getattr(event, "prompt") or safe_getattr(event, "message")
            if prompt:
                req_attrs["flow.input_prompt"] = truncate_str(
                    str(prompt), MAX_DESCRIPTION_LENGTH
                )

            context = safe_getattr(event, "context")
            if context is not None:
                req_attrs["flow.input_context"] = truncate_str(
                    safe_json_dumps(context), 1024
                )

            field = safe_getattr(event, "field") or safe_getattr(event, "field_name")
            if field:
                req_attrs["flow.input_field"] = str(field)

            set_span_attributes(span, req_attrs)
        except Exception:
            logger.debug("on_flow_input_requested error:\n%s", traceback.format_exc())

    def on_flow_input_received(self, source: Any, event: Any) -> None:
        """
        Annotate the flow span when user input is received.

        Attributes written
        ------------------
        - ``flow.input_received``   — ``True``
        - ``flow.input_value``      — the value provided (truncated)
        - ``flow.input_field``      — state field that was populated
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            span = self._get_flow_span(flow_id)
            if span is None:
                return

            recv_attrs: dict[str, Any] = {"flow.input_received": True}

            value = safe_getattr(event, "value")
            if value is None:
                value = safe_getattr(event, "input")
            if value is not None and getattr(self, "capture_inputs", True):
                raw = value if isinstance(value, str) else safe_json_dumps(value)
                recv_attrs["flow.input_value"] = truncate_str(raw, 1024)

            field = safe_getattr(event, "field") or safe_getattr(event, "field_name")
            if field:
                recv_attrs["flow.input_field"] = str(field)

            set_span_attributes(span, recv_attrs)
        except Exception:
            logger.debug("on_flow_input_received error:\n%s", traceback.format_exc())

    def on_human_feedback_requested(self, source: Any, event: Any) -> None:
        """
        Annotate the flow (or method) span when free-form human feedback is requested.

        Attributes written
        ------------------
        - ``flow.feedback_requested``  — ``True``
        - ``flow.feedback_prompt``     — prompt / question shown to the human
        - ``flow.feedback_context``    — additional context provided
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            span = self._get_flow_span(flow_id) or self._get_any_method_span(flow_id)
            if span is None:
                return

            fb_attrs: dict[str, Any] = {"flow.feedback_requested": True}

            prompt = (
                safe_getattr(event, "prompt")
                or safe_getattr(event, "question")
                or safe_getattr(event, "message")
            )
            if prompt:
                fb_attrs["flow.feedback_prompt"] = truncate_str(
                    str(prompt), MAX_DESCRIPTION_LENGTH
                )

            context = safe_getattr(event, "context")
            if context is not None:
                fb_attrs["flow.feedback_context"] = truncate_str(
                    safe_json_dumps(context), 1024
                )

            set_span_attributes(span, fb_attrs)
        except Exception:
            logger.debug(
                "on_human_feedback_requested error:\n%s", traceback.format_exc()
            )

    def on_human_feedback_received(self, source: Any, event: Any) -> None:
        """
        Annotate the flow (or method) span when human feedback arrives.

        Attributes written
        ------------------
        - ``flow.feedback_received``   — ``True``
        - ``flow.feedback_text``       — the human's response (truncated)
        - ``flow.feedback_outcome``    — optional structured outcome / result
        """
        if not self._is_active():
            return
        try:
            flow_id = _resolve_flow_id(source, event)
            span = self._get_flow_span(flow_id) or self._get_any_method_span(flow_id)
            if span is None:
                return

            feedback = None
            for _attr in ("feedback", "response", "value", "text"):
                val = safe_getattr(event, _attr)
                if val is not None:
                    feedback = val
                    break
            recv_attrs: dict[str, Any] = {"flow.feedback_received": True}
            if feedback is not None and getattr(self, "capture_inputs", True):
                raw = (
                    feedback if isinstance(feedback, str) else safe_json_dumps(feedback)
                )
                recv_attrs["flow.feedback_text"] = truncate_str(raw, 2048)

            outcome = safe_getattr(event, "outcome")
            if outcome is None:
                outcome = safe_getattr(event, "result")
            if outcome is not None:
                recv_attrs["flow.feedback_outcome"] = truncate_str(
                    outcome if isinstance(outcome, str) else safe_json_dumps(outcome),
                    1024,
                )

            set_span_attributes(span, recv_attrs)
        except Exception:
            logger.debug(
                "on_human_feedback_received error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_flow_span(self, flow_id: str) -> Any:
        """Return the open Span for *flow_id*, or ``None``."""
        with self._lock:
            entry = self._flow_spans.get(flow_id)
        return entry["span"] if entry else None

    def _get_any_method_span(self, flow_id: str) -> Any:
        """Return any open method span belonging to *flow_id*, or ``None``."""
        prefix = flow_id + _METHOD_KEY_SEP
        with self._lock:
            for key, entry in self._flow_method_spans.items():
                if key.startswith(prefix):
                    return entry["span"]
        return None

    def _get_crew_span_for_flow(self, crew_id: Optional[str]) -> Any:
        """Return the open crew span to use as a flow's parent, or ``None``."""
        if not crew_id:
            return None
        with self._lock:
            entry = self._crew_spans.get(crew_id)
        return entry["span"] if entry else None

    def _finish_flow_span(
        self,
        flow_id: str,
        status: str,
        error: Any,
        extra_attrs: Optional[dict[str, Any]] = None,
        *,
        finish_trace_client: Any = _FINISH_TRACE_CLIENT_UNSPECIFIED,
    ) -> None:
        """Write final attributes onto the flow span and close it (+trace).

        ``finish_trace_client`` overrides ``_get_client()`` for ``finish_trace``
        (e.g. shutdown force-close when ``_is_shutdown`` is already true).
        """
        with self._lock:
            entry = self._flow_spans.pop(flow_id, None)

        if entry is None:
            logger.debug("_finish_flow_span: no open entry for flow_id=%s", flow_id)
            return

        span = entry["span"]
        trace = entry.get("trace")
        start_t = entry.get("start_t")

        attrs: dict[str, Any] = {ATTR_FLOW_STATUS: status}
        if start_t is not None:
            attrs[ATTR_FLOW_DURATION_MS] = duration_ms_monotonic(start_t)
        if error is not None:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = truncate_str(str(error), MAX_DESCRIPTION_LENGTH)
            tb = getattr(error, "__traceback__", None)
            if tb is not None:
                attrs[ATTR_ERROR_STACKTRACE] = truncate_str(
                    "".join(traceback.format_tb(tb)),
                    MAX_TEXT_LENGTH,
                )
        if extra_attrs:
            attrs.update(extra_attrs)

        set_span_attributes(span, attrs)

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
                "_finish_flow_span span.finish() error:\n%s",
                traceback.format_exc(),
            )

        # Close standalone trace (nested flows share the crew trace)
        if trace is not None:
            try:
                if finish_trace_client is not _FINISH_TRACE_CLIENT_UNSPECIFIED:
                    client = finish_trace_client
                else:
                    client = self._get_client()
                if client and hasattr(client, "finish_trace"):
                    client.finish_trace(trace)
            except Exception:
                logger.debug(
                    "_finish_flow_span finish_trace error:\n%s",
                    traceback.format_exc(),
                )

        logger.debug("Flow span closed: flow_id=%s status=%s", flow_id, status)

    def _finish_method_span(
        self,
        method_key: str,
        status: str,
        output: Any,
        error: Any,
    ) -> None:
        """Write final attributes onto a flow method span and close it."""
        with self._lock:
            entry = self._flow_method_spans.pop(method_key, None)

        if entry is None:
            logger.debug("_finish_method_span: no open entry for key=%s", method_key)
            return

        span = entry["span"]
        start_t = entry.get("start_t")

        attrs: dict[str, Any] = {ATTR_FLOW_METHOD_STATUS: status}
        if start_t is not None:
            attrs[ATTR_FLOW_METHOD_DURATION_MS] = duration_ms_monotonic(start_t)

        if output is not None and getattr(self, "capture_outputs", True):
            raw = output if isinstance(output, str) else safe_json_dumps(output)
            attrs["flow.method.output"] = truncate_str(raw, MAX_TEXT_LENGTH)

        if error is not None:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = truncate_str(str(error), MAX_DESCRIPTION_LENGTH)
            tb = getattr(error, "__traceback__", None)
            if tb is not None:
                attrs[ATTR_ERROR_STACKTRACE] = truncate_str(
                    "".join(traceback.format_tb(tb)),
                    MAX_TEXT_LENGTH,
                )

        set_span_attributes(span, attrs)

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
                "_finish_method_span span.finish() error:\n%s",
                traceback.format_exc(),
            )

        logger.debug("Flow method span closed: key=%s status=%s", method_key, status)


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _resolve_flow_id(source: Any, event: Any) -> str:
    """Return a stable string key for the flow run."""
    return str(
        safe_getattr(event, "flow_id")
        or safe_getattr(event, "id")
        or safe_getattr(source, "flow_id")
        or safe_getattr(source, "id")
        or id(source)
    )


def _resolve_crew_id(source: Any, event: Any) -> Optional[str]:
    """Return the crew_id for a flow if it runs inside a Crew."""
    raw = (
        safe_getattr(event, "crew_id")
        or safe_getattr(source, "crew_id")
        or safe_getattr(safe_getattr(source, "crew"), "id")
    )
    return str(raw) if raw is not None else None


def _resolve_method_id(event: Any) -> str:
    """Return a stable key for a single method execution instance."""
    return str(
        safe_getattr(event, "method_id")
        or safe_getattr(event, "execution_id")
        or safe_getattr(event, "id")
        or id(event)
    )


def _make_method_key(flow_id: str, method_id: str) -> str:
    """Compose the composite key used in ``_flow_method_spans``."""
    return f"{flow_id}{_METHOD_KEY_SEP}{method_id}"


_HEAVY_FLOW_NODE_KEYS = frozenset({"source_code", "source_lines", "source_start_line"})
_MAX_METHOD_SIGNATURE_CHARS = 2000


def _flow_structure_to_plain(structure: Any) -> dict[str, Any]:
    """Turn CrewAI ``FlowStructure`` (or dict) into a plain dict for JSON."""
    if structure is None:
        return {}
    md = getattr(structure, "model_dump", None)
    if callable(md):
        try:
            out = md()
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    legacy_dict = getattr(structure, "dict", None)
    if callable(legacy_dict):
        try:
            out = legacy_dict()
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    if isinstance(structure, dict):
        return dict(structure)
    try:
        out = dict(structure)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _sanitize_flow_structure_for_trace(structure: Any) -> dict[str, Any]:
    """Drop heavy per-node fields before serializing to a span attribute."""
    data = _flow_structure_to_plain(structure)
    if not data:
        return {}

    nodes = data.get("nodes")
    if isinstance(nodes, dict):
        slim_nodes: dict[str, Any] = {}
        for key, meta in nodes.items():
            if isinstance(meta, dict):
                slim = {k: v for k, v in meta.items() if k not in _HEAVY_FLOW_NODE_KEYS}
                sig = slim.get("method_signature")
                if isinstance(sig, str) and len(sig) > _MAX_METHOD_SIGNATURE_CHARS:
                    slim["method_signature"] = sig[:_MAX_METHOD_SIGNATURE_CHARS] + "..."
                slim_nodes[key] = slim
            else:
                slim_nodes[key] = meta
        data = {**data, "nodes": slim_nodes}
    return data


def _try_flow_structure_json(source: Any) -> Optional[str]:
    """Return truncated JSON for ``flow.structure``, or ``None`` if unavailable."""
    try:
        from crewai.flow.visualization import build_flow_structure
    except ImportError:
        return None
    try:
        built = build_flow_structure(source)
    except Exception:
        logger.debug("build_flow_structure failed", exc_info=True)
        return None
    slim = _sanitize_flow_structure_for_trace(built)
    if not slim:
        return None
    try:
        raw = safe_json_dumps(slim)
    except Exception:
        return None
    if not raw:
        return None
    return truncate_str(raw, MAX_TEXT_LENGTH)


def _build_flow_start_attributes(
    source: Any,
    event: Any,
    flow_id: str,
    *,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
) -> dict[str, Any]:
    """Collect span attributes for the opening ``crewai.flow`` span."""
    attrs: dict[str, Any] = {ATTR_FLOW_ID: flow_id}

    # Flow name: prefer explicit name, fall back to class name
    name = (
        safe_getattr(event, "flow_name")
        or safe_getattr(event, "name")
        or safe_getattr(source, "name")
        or type(source).__name__
    )
    if name:
        attrs[ATTR_FLOW_NAME] = str(name)

    # Inputs passed to kickoff()
    if capture_inputs:
        inputs = safe_getattr(event, "inputs") or safe_getattr(event, "kwargs")
        if inputs is not None:
            attrs["flow.inputs"] = truncate_str(
                safe_json_dumps(inputs), MAX_TEXT_LENGTH
            )

    # Initial state snapshot
    state = safe_getattr(event, "state") or safe_getattr(source, "state")
    if state is not None:
        attrs[ATTR_FLOW_STATE] = truncate_str(safe_json_dumps(state), MAX_TEXT_LENGTH)

    structure_json = _try_flow_structure_json(source)
    if structure_json:
        attrs[ATTR_FLOW_STRUCTURE] = structure_json

    return attrs


def _build_method_start_attributes(
    source: Any,
    event: Any,
    method_id: str,
    flow_id: str,
    *,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
) -> dict[str, Any]:
    """Collect span attributes for a ``crewai.flow.method`` span."""
    attrs: dict[str, Any] = {ATTR_FLOW_METHOD_ID: method_id}

    method_name = (
        safe_getattr(event, "method_name")
        or safe_getattr(event, "name")
        or safe_getattr(event, "function_name")
    )
    if method_name:
        attrs[ATTR_FLOW_METHOD_NAME] = str(method_name)

    # Flow name for cross-referencing
    flow_name = (
        safe_getattr(event, "flow_name")
        or safe_getattr(source, "name")
        or type(source).__name__
    )
    if flow_name:
        attrs[ATTR_FLOW_NAME] = str(flow_name)

    # Method type: start | listen | router  (inferred from decorator or event)
    method_type = (
        safe_getattr(event, "method_type")
        or safe_getattr(event, "decorator_type")
        or safe_getattr(event, "trigger_type")
    )
    if method_type:
        attrs["flow.method.type"] = str(method_type).lower()
    else:
        # Infer from Flow class metadata populated by @start / @listen / @router.
        # CrewAI stores method names in class-level sets:
        #   _start_methods: set of method names decorated with @start()
        #   _routers:       set of method names decorated with @router()
        # Everything else is a @listen method.
        mn = str(method_name) if method_name else ""
        start_methods: Any = safe_getattr(source, "_start_methods") or set()
        router_methods: Any = safe_getattr(source, "_routers") or set()
        if mn and mn in start_methods:
            inferred_type = "start"
        elif mn and mn in router_methods:
            inferred_type = "router"
        else:
            inferred_type = "listen"
        attrs["flow.method.type"] = inferred_type

    # Triggering event / upstream method name
    trigger = (
        safe_getattr(event, "trigger")
        or safe_getattr(event, "triggered_by")
        or safe_getattr(event, "trigger_event")
    )
    if trigger:
        attrs[ATTR_FLOW_METHOD_TRIGGER] = truncate_str(str(trigger), 256)

    # Method call parameters
    if capture_inputs:
        params = (
            safe_getattr(event, "params")
            or safe_getattr(event, "kwargs")
            or safe_getattr(event, "arguments")
        )
        if params is not None:
            attrs["flow.method.params"] = truncate_str(
                safe_json_dumps(params), MAX_DESCRIPTION_LENGTH
            )

    return attrs
