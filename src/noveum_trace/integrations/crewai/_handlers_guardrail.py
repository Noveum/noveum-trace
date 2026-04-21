"""
LLM guardrail event handler mixin for NoveumCrewAIListener.

CrewAI guardrails validate LLM outputs before they are accepted as the result
of an agent step.  When a guardrail rejects an output, CrewAI retries the LLM
call up to a configurable maximum, so each guardrail check may be associated
with multiple retry attempts.

Handles CrewAI ``BaseEventListener`` guardrail events:

  - ``on_llm_guardrail_started``    → open ``crewai.guardrail`` child span;
                                       capture guardrail name/type, input being
                                       validated, retry_count (which attempt this is),
                                       the call_id of the LLM call being guarded
  - ``on_llm_guardrail_completed``  → close span; write validation_success bool,
                                       results list, retry_count at completion,
                                       the accepted/rejected output text
  - ``on_llm_guardrail_failed``     → close span as ERROR; write error details

Span hierarchy::

    crewai.agent
      crewai.llm
        crewai.guardrail   ← one per guardrail check (including retries)

Guardrail spans are children of the LLM span they are guarding.  If the LLM
span has already closed (guardrail result arrives after the LLM span finishes),
the span falls back to the agent span as its parent so data is never lost.

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown,
    _llm_call_spans, _agent_spans,
    _guardrail_spans (``guardrail_id`` → ``{span, start_t}``)
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_AGENT_ROLE,
    ATTR_LLM_CALL_ID,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    MAX_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
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
    truncate_str,
)

logger = logging.getLogger(__name__)

# Span name (internal — not in crewai_constants to keep public API lean)
_SPAN_GUARDRAIL = "crewai.guardrail"


class _GuardrailHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI LLM guardrail events.

    All public methods match the ``BaseEventListener`` callback signature::

        def on_llm_guardrail_started(self, source, event): ...

    ``source`` is the guardrail object or the Agent; ``event`` carries the
    per-check payload.  Every method is fully exception-shielded.
    """

    # =========================================================================
    # Guardrail started
    # =========================================================================

    def on_llm_guardrail_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.guardrail`` span as a child of the guarded LLM span.

        Attributes set at span open
        ---------------------------
        - ``guardrail.id``          — unique identifier for this check instance
        - ``guardrail.name``        — guardrail name / identifier
        - ``guardrail.type``        — guardrail class / implementation type
        - ``guardrail.description`` — human-readable description of the guardrail
        - ``guardrail.input``       — LLM output text being validated (truncated)
        - ``guardrail.retry_count`` — which retry attempt this check is (0 = first)
        - ``llm.call_id``           — call_id of the LLM call being guarded
        - ``agent.role``            — executing agent's role (correlation)
        """
        if not self._is_active() or not getattr(self, "capture_guardrails", False):
            return
        try:
            guardrail_id = _resolve_guardrail_id(event, source)
            call_id = _resolve_call_id(event)
            agent_id = _resolve_agent_id(source, event)

            attrs = _build_guardrail_start_attributes(
                source, event, guardrail_id, call_id
            )
            start_t = monotonic_now()

            # Parent: open LLM span → agent span → None
            parent_span = self._get_llm_or_agent_span(call_id, agent_id)

            span = self._create_child_span(
                _SPAN_GUARDRAIL,
                parent_span=parent_span,
                attributes=attrs,
            )

            with self._lock:
                self._guardrail_spans[guardrail_id] = {"span": span, "start_t": start_t}

            logger.debug(
                "Guardrail span opened: guardrail_id=%s call_id=%s retry=%s",
                guardrail_id,
                call_id,
                attrs.get("guardrail.retry_count", 0),
            )
        except Exception:
            logger.debug("on_llm_guardrail_started error:\n%s", traceback.format_exc())

    # =========================================================================
    # Guardrail completed (validation ran — may pass or fail validation)
    # =========================================================================

    def on_llm_guardrail_completed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.guardrail`` span.

        Note: "completed" means the guardrail *ran to completion*, not that
        validation *passed*.  Check ``guardrail.validation_success`` to
        distinguish acceptance from rejection.

        Attributes written
        ------------------
        - ``guardrail.validation_success`` — bool; True = output accepted,
                                              False = output rejected / retry
        - ``guardrail.retry_count``        — retry number at completion time
        - ``guardrail.results``            — JSON list of per-check result dicts
        - ``guardrail.result_summary``     — human-readable summary (if provided)
        - ``guardrail.output``             — the validated / corrected output text
        - ``guardrail.status``             — ``"success"`` (span ran without error)
        - ``guardrail.duration_ms``        — wall-clock duration of the check
        """
        if not self._is_active() or not getattr(self, "capture_guardrails", False):
            return
        try:
            guardrail_id = _resolve_guardrail_id(event, source)
            extra: dict[str, Any] = {}

            validation_success = safe_getattr(event, "validation_success")
            if validation_success is None:
                # First non-None among legacy flags (``or`` would skip explicit False).
                for alt_attr in ("passed", "accepted", "valid"):
                    alt_val = safe_getattr(event, alt_attr)
                    if alt_val is not None:
                        validation_success = alt_val
                        break
            if validation_success is not None:
                extra["guardrail.validation_success"] = bool(validation_success)

            retry_count = _get_retry_count(event)
            if retry_count is not None:
                extra["guardrail.retry_count"] = retry_count

            results = safe_getattr(event, "results") or safe_getattr(event, "checks")
            if results is not None:
                extra["guardrail.results"] = truncate_str(
                    safe_json_dumps(results), MAX_TEXT_LENGTH
                )

            summary = safe_getattr(event, "result_summary") or safe_getattr(
                event, "summary"
            )
            if summary:
                extra["guardrail.result_summary"] = truncate_str(
                    str(summary), MAX_DESCRIPTION_LENGTH
                )

            output = (
                safe_getattr(event, "output")
                or safe_getattr(event, "corrected_output")
                or safe_getattr(event, "accepted_output")
            )
            if output:
                extra["guardrail.output"] = truncate_str(str(output), MAX_TEXT_LENGTH)

            self._finish_guardrail_span(guardrail_id, ATTR_STATUS_SUCCESS, None, extra)
        except Exception:
            logger.debug(
                "on_llm_guardrail_completed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Guardrail failed (the guardrail itself raised an exception)
    # =========================================================================

    def on_llm_guardrail_failed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.guardrail`` span as ERROR.

        This fires when the guardrail *implementation* raises an exception
        (not when it rejects the LLM output — that is captured by
        ``on_llm_guardrail_completed`` with ``validation_success=False``).

        Attributes written
        ------------------
        - ``error.type``           — exception class name
        - ``error.message``        — exception message
        - ``error.stacktrace``     — formatted traceback
        - ``guardrail.retry_count``— which retry this failure occurred on
        - ``guardrail.status``     — ``"error"``
        - ``guardrail.duration_ms``— wall-clock duration
        """
        if not self._is_active() or not getattr(self, "capture_guardrails", False):
            return
        try:
            guardrail_id = _resolve_guardrail_id(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            extra: dict[str, Any] = {}

            retry_count = _get_retry_count(event)
            if retry_count is not None:
                extra["guardrail.retry_count"] = retry_count

            self._finish_guardrail_span(guardrail_id, ATTR_STATUS_ERROR, error, extra)
        except Exception:
            logger.debug("on_llm_guardrail_failed error:\n%s", traceback.format_exc())

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_llm_or_agent_span(
        self, call_id: Optional[str], agent_id: Optional[str]
    ) -> Any:
        """Return the open LLM span, falling back to the agent span."""
        with self._lock:
            if call_id:
                entry = self._llm_call_spans.get(call_id)
                if entry:
                    return entry.get("span")
            if agent_id:
                return self._agent_spans.get(agent_id)
        return None

    def _finish_guardrail_span(
        self,
        guardrail_id: str,
        status: str,
        error: Any,
        extra_attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write final attributes onto the guardrail span and close it."""
        with self._lock:
            entry = self._guardrail_spans.pop(guardrail_id, None)
        span = entry.get("span") if entry else None
        start_t = entry.get("start_t") if entry else None

        if span is None:
            logger.debug(
                "_finish_guardrail_span: no open span for guardrail_id=%s",
                guardrail_id,
            )
            return

        attrs = finish_span_common(
            span,
            start_t=start_t,
            status=status,
            status_attr="guardrail.status",
            duration_attr="guardrail.duration_ms",
            error=error,
            extra_attrs=extra_attrs,
            log_label="_finish_guardrail_span",
        )

        logger.debug(
            "Guardrail span closed: guardrail_id=%s status=%s validation=%s",
            guardrail_id,
            status,
            attrs.get("guardrail.validation_success", "?"),
        )


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _resolve_guardrail_id(event: Any, source: Any) -> str:
    """Return a stable string key for the guardrail check instance.

    CrewAI's event bus sets ``started_event_id`` on completed/failed events to the
    ``event_id`` of the matching ``*_started`` event.  When that link is present it
    must win so lifecycle handlers close the same span opened in ``*_started``.

    Otherwise we fall back to explicit guardrail identifiers, then ``event_id`` (so
    started events align with the bus link), then ``id(event)`` as a last resort.
    """
    started_link = safe_getattr(event, "started_event_id")
    if isinstance(started_link, str) and started_link.strip():
        return started_link.strip()

    raw = (
        safe_getattr(event, "guardrail_id")
        or safe_getattr(event, "event_id")
        or safe_getattr(event, "check_id")
        or safe_getattr(event, "id")
        or safe_getattr(event, "run_id")
        or id(event)
    )
    return str(raw)


def _resolve_call_id(event: Any) -> Optional[str]:
    """Return the LLM call_id being guarded, or ``None``."""
    raw = (
        safe_getattr(event, "call_id")
        or safe_getattr(event, "llm_call_id")
        or safe_getattr(event, "run_id")
    )
    return str(raw) if raw is not None else None


def _get_retry_count(event: Any) -> Optional[int]:
    """Extract retry_count from the event as an int, or ``None``."""
    val = next(
        (
            v
            for v in (
                safe_getattr(event, "retry_count"),
                safe_getattr(event, "attempt"),
                safe_getattr(event, "retries"),
            )
            if v is not None
        ),
        None,
    )
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _build_guardrail_start_attributes(
    source: Any,
    event: Any,
    guardrail_id: str,
    call_id: Optional[str],
) -> dict[str, Any]:
    """Collect span attributes for the opening ``crewai.guardrail`` span."""
    attrs: dict[str, Any] = {"guardrail.id": guardrail_id}

    # Guardrail identity
    gname = (
        safe_getattr(event, "guardrail_name")
        or safe_getattr(event, "name")
        or safe_getattr(source, "name")
        or type(source).__name__
    )
    if gname:
        attrs["guardrail.name"] = truncate_str(str(gname), 256)

    gtype = (
        safe_getattr(event, "guardrail_type")
        or safe_getattr(event, "type")
        or type(source).__name__
    )
    if gtype:
        attrs["guardrail.type"] = str(gtype)

    description = safe_getattr(event, "description") or safe_getattr(
        source, "description"
    )
    if description:
        attrs["guardrail.description"] = truncate_str(
            str(description), MAX_DESCRIPTION_LENGTH
        )

    # The LLM output being validated
    input_text = (
        safe_getattr(event, "input")
        or safe_getattr(event, "output")  # CrewAI names it 'output' from LLM POV
        or safe_getattr(event, "llm_output")
        or safe_getattr(event, "text")
    )
    if input_text:
        attrs["guardrail.input"] = truncate_str(str(input_text), MAX_TEXT_LENGTH)

    # Retry counter
    retry_count = _get_retry_count(event)
    if retry_count is not None:
        attrs["guardrail.retry_count"] = retry_count

    # Max retries allowed (useful context for understanding retry loops)
    max_retries = safe_getattr(event, "max_retries") or safe_getattr(
        source, "max_retries"
    )
    if max_retries is not None:
        try:
            attrs["guardrail.max_retries"] = int(max_retries)
        except (TypeError, ValueError):
            pass

    # Guardrail configuration / rules (compact snapshot)
    config = safe_getattr(event, "config") or safe_getattr(source, "config")
    if config is not None:
        attrs["guardrail.config"] = truncate_str(safe_json_dumps(config), 512)

    # LLM call correlation
    if call_id:
        attrs[ATTR_LLM_CALL_ID] = call_id

    # Agent correlation
    agent_role = safe_getattr(event, "agent_role") or safe_getattr(source, "role")
    if agent_role:
        attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

    return attrs
