"""
LLM-call event handler mixin for NoveumCrewAIListener.

Handles CrewAI ``BaseEventListener`` LLM events:

  - ``on_llm_call_started``    → open ``crewai.llm`` child span under the owning
                                  agent span; capture model, call_id, full messages
                                  list as ``llm.messages`` JSON, system prompt
                                  extracted separately as ``llm.system_prompt``,
                                  tool definitions as ``llm.tools``;
                                  ``llm.available_tools.*`` (count, names,
                                  descriptions, schemas); plus ``agent.role`` and
                                  ``task.name`` for correlation.
                                  NOTE: ``call_type`` is NOT available here — it
                                  arrives only with the completed event.

  - ``on_llm_call_completed``  → write ``llm.call_type`` from event;
                                  join streaming buffer → ``llm.streaming_response``;
                                  join thinking buffer → ``llm.thinking_text``;
                                  extract token counts from ``_llm_usage_by_call_id``
                                  (populated by monkey-patch) and write directly to
                                  ``span.attributes`` dict (span may already be
                                  finished at this point);
                                  calculate cost; add to per-crew accumulators;
                                  write ``llm.response`` and finish span.

  - ``on_llm_call_failed``     → set ERROR status + message; finish span.

  - ``on_llm_stream_chunk``    → append ``event.chunk`` to
                                  ``_llm_stream_chunks[call_id]``.

  - ``on_llm_thinking_chunk``  → append to ``_llm_thinking_chunks[call_id]``.

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown,
    _agent_spans, _llm_call_spans, _llm_call_start_times,
    _llm_stream_chunks, _llm_thinking_chunks, _pending_llm_metadata,
    _total_tokens_by_crew, _total_cost_by_crew
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
    ATTR_LLM_CACHE_CREATION_TOKENS,
    ATTR_LLM_CACHE_READ_TOKENS,
    ATTR_LLM_CALL_ID,
    ATTR_LLM_COST_CURRENCY,
    ATTR_LLM_COST_INPUT,
    ATTR_LLM_COST_OUTPUT,
    ATTR_LLM_COST_TOTAL,
    ATTR_LLM_DURATION_MS,
    ATTR_LLM_FINISH_REASON,
    ATTR_LLM_INPUT_MESSAGES,
    ATTR_LLM_INPUT_TOKENS,
    ATTR_LLM_MAX_TOKENS,
    ATTR_LLM_MODEL,
    ATTR_LLM_OUTPUT_TEXT,
    ATTR_LLM_OUTPUT_TOKENS,
    ATTR_LLM_PROVIDER,
    ATTR_LLM_REASONING_TOKENS,
    ATTR_LLM_SEED,
    ATTR_LLM_STREAMING,
    ATTR_LLM_SYSTEM_PROMPT,
    ATTR_LLM_TEMPERATURE,
    ATTR_LLM_THINKING_TEXT,
    ATTR_LLM_TOOLS,
    ATTR_LLM_TOP_P,
    ATTR_LLM_TOTAL_TOKENS,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    DEFAULT_LLM_MODEL,
    MAX_SYSTEM_PROMPT_LENGTH,
    MAX_TEXT_LENGTH,
    SPAN_LLM,
)
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase
from noveum_trace.integrations.crewai.crewai_utils import (
    calculate_llm_cost,
    duration_ms_monotonic,
    extract_response_text,
    extract_system_prompt,
    extract_token_usage,
    merge_available_tools_attributes,
    messages_to_json,
    monotonic_now,
)
from noveum_trace.integrations.crewai.crewai_utils import (
    resolve_agent_id as _resolve_agent_id,
)
from noveum_trace.integrations.crewai.crewai_utils import (
    safe_getattr,
    serialize_tool_schema,
    set_span_attributes,
    truncate_str,
)

logger = logging.getLogger(__name__)

# Canonical keys merged onto LLM token_usage dicts in :meth:`_finish_llm_span`.
_TOKEN_USAGE_FIELD_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
)


def _first_non_none_llm(*values: Any) -> Any:
    """Return the first argument that is not ``None`` (preserves ``0`` and ``False``)."""
    for v in values:
        if v is not None:
            return v
    return None


def _merge_missing_token_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Copy *source* into *target* only for keys that are still ``None`` in *target*."""
    for key in _TOKEN_USAGE_FIELD_KEYS:
        val = source.get(key)
        if val is None:
            continue
        if target.get(key) is None:
            target[key] = val


def _token_usage_from_event_usage_dict(event_usage: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``event.usage`` to canonical token_usage keys."""
    return {
        "input_tokens": _first_non_none_llm(
            event_usage.get("input_tokens"),
            event_usage.get("prompt_tokens"),
            event_usage.get("prompt_token_count"),
        ),
        "output_tokens": _first_non_none_llm(
            event_usage.get("output_tokens"),
            event_usage.get("completion_tokens"),
            event_usage.get("candidates_token_count"),
        ),
        "total_tokens": _first_non_none_llm(
            event_usage.get("total_tokens"),
            event_usage.get("total_token_count"),
        ),
        "cache_read_tokens": _first_non_none_llm(
            event_usage.get("cached_tokens"),
            event_usage.get("cached_prompt_tokens"),
            event_usage.get("cache_read_input_tokens"),
            event_usage.get("cache_read_tokens"),
        ),
        "cache_creation_tokens": _first_non_none_llm(
            event_usage.get("cache_creation_input_tokens"),
            event_usage.get("cache_creation_tokens"),
        ),
        "reasoning_tokens": event_usage.get("reasoning_tokens"),
    }


def _token_usage_from_monkey_dict(monkey: dict[str, Any]) -> dict[str, Any]:
    """Normalize monkey-patch buffer dict to canonical token_usage keys."""
    return {
        "input_tokens": _first_non_none_llm(
            monkey.get("input_tokens"),
            monkey.get("prompt_tokens"),
        ),
        "output_tokens": _first_non_none_llm(
            monkey.get("output_tokens"),
            monkey.get("completion_tokens"),
        ),
        "total_tokens": monkey.get("total_tokens"),
        "cache_read_tokens": _first_non_none_llm(
            monkey.get("cache_read_tokens"),
            monkey.get("cache_read_input_tokens"),
        ),
        "cache_creation_tokens": _first_non_none_llm(
            monkey.get("cache_creation_tokens"),
            monkey.get("cache_creation_input_tokens"),
        ),
        "reasoning_tokens": monkey.get("reasoning_tokens"),
    }


def _token_usage_from_event_top_level(event: Any) -> dict[str, Any]:
    """Fallback token fields read directly from the event object."""
    return {
        "input_tokens": _first_non_none_llm(
            safe_getattr(event, "prompt_tokens"),
            safe_getattr(event, "input_tokens"),
        ),
        "output_tokens": _first_non_none_llm(
            safe_getattr(event, "completion_tokens"),
            safe_getattr(event, "output_tokens"),
        ),
        "total_tokens": safe_getattr(event, "total_tokens"),
        "cache_read_tokens": None,
        "cache_creation_tokens": None,
        "reasoning_tokens": None,
    }


class _LLMHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI LLM-call events.

    All public methods match the ``BaseEventListener`` callback signature::

        def on_llm_call_started(self, source, event): ...

    ``source`` is typically the ``Agent`` executing the call; ``event`` carries
    the per-call payload.  Every method is fully exception-shielded.
    """

    # =========================================================================
    # LLM call started
    # =========================================================================

    def on_llm_call_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.llm`` child span under the owning agent or task span.

        Parent resolution prefers agent → task → explicit crew (via ``task_id`` /
        ``_task_to_crew_id`` and :meth:`_create_child_span` hints), not dict order
        of open crews.

        Attributes set at span open
        ---------------------------
        - ``llm.call_id``            — unique call identifier (for correlation)
        - ``llm.model``              — model name (from event or agent LLM)
        - ``llm.provider``           — inferred provider name
        - ``llm.system_prompt``      — extracted from the messages list
        - ``llm.messages``           — full messages list serialized as JSON
        - ``llm.tools``              — tool schema JSON (when tools provided)
        - ``llm.available_tools.*``  — count, names, descriptions, schemas JSON
        - ``agent.role``             — role of the executing agent (correlation)
        - ``task.name``              — name/description of the current task
        - ``llm.streaming``          — bool, True when streaming mode detected

        Message / tool payload on the span respects ``capture_llm_messages`` and
        ``capture_tool_schemas``. Stream and thinking buffers are only allocated
        when ``capture_streaming`` / ``capture_thinking`` are enabled.

        Note: ``llm.call_type`` is intentionally NOT set here — it is only
        available on the completed event and is written there instead.
        """
        if not self._is_active():
            return
        try:
            call_id = _resolve_call_id(event)
            agent_id = _resolve_agent_id(source, event)
            crew_id = _resolve_crew_id(source, event)
            task_id = _resolve_task_id(source, event)

            # Prefer task→crew mapping (populated at kickoff) over guessing from
            # dict iteration order when multiple crews overlap.
            if not crew_id and task_id:
                with self._lock:
                    mapped = self._task_to_crew_id.get(str(task_id))
                if mapped:
                    crew_id = mapped

            attrs = _build_llm_start_attributes(
                source,
                event,
                call_id,
                capture_llm_messages=self.capture_llm_messages,
                capture_tool_schemas=self.capture_tool_schemas,
            )
            start_t = monotonic_now()

            # Parent: agent span, else task span, else crew root (never "most recent")
            parent_span = self._get_agent_or_task_span(agent_id, task_id)
            if parent_span is None and crew_id:
                parent_span = self._get_crew_span(crew_id)

            span = self._create_child_span(
                SPAN_LLM,
                parent_span=parent_span,
                attributes=attrs,
                crew_id=crew_id,
                task_id=task_id,
            )

            with self._lock:
                self._llm_call_spans[call_id] = {
                    "span": span,
                    "crew_id": crew_id,
                    "agent_id": agent_id,
                    "task_id": task_id,
                }
                self._llm_call_start_times[call_id] = start_t
                # Pre-initialise buffers only when the corresponding capture flag is on
                if self.capture_streaming:
                    self._llm_stream_chunks.setdefault(call_id, [])
                if self.capture_thinking:
                    self._llm_thinking_chunks.setdefault(call_id, [])

            logger.debug("LLM span opened: call_id=%s agent_id=%s", call_id, agent_id)

        except Exception:
            logger.debug("on_llm_call_started error:\n%s", traceback.format_exc())

    # =========================================================================
    # LLM stream chunk (streaming mode only)
    # =========================================================================

    def on_llm_stream_chunk(self, source: Any, event: Any) -> None:
        """
        Append a streaming text chunk to ``_llm_stream_chunks[call_id]``.

        The buffer is joined and written to ``llm.streaming_response`` when the
        call completes.  Buffering is bounded by MAX_TEXT_LENGTH total chars;
        excess chunks are silently dropped to prevent unbounded memory growth.

        No-op when ``capture_streaming`` is ``False``.
        """
        if not self._is_active():
            return
        if not self.capture_streaming:
            return
        try:
            call_id = _resolve_call_id(event)
            chunk = (
                safe_getattr(event, "chunk")
                or safe_getattr(event, "text")
                or safe_getattr(event, "delta")
                or ""
            )
            if not chunk:
                return
            chunk_str = str(chunk)

            with self._lock:
                buf = self._llm_stream_chunks.setdefault(call_id, [])
                current_len = sum(len(c) for c in buf)
                if current_len + len(chunk_str) <= MAX_TEXT_LENGTH:
                    buf.append(chunk_str)
                # Silently drop when buffer is full — span will show truncated text

        except Exception:
            logger.debug("on_llm_stream_chunk error:\n%s", traceback.format_exc())

    # =========================================================================
    # LLM thinking / reasoning chunk (extended thinking models)
    # =========================================================================

    def on_llm_thinking_chunk(self, source: Any, event: Any) -> None:
        """
        Append a chain-of-thought / thinking chunk to ``_llm_thinking_chunks[call_id]``.

        Supports Anthropic Claude extended thinking, o1-series reasoning tokens,
        and any provider that emits thinking tokens separately from the response.
        The accumulated buffer is written to ``llm.thinking_text`` on completion.

        No-op when ``capture_thinking`` is ``False``.
        """
        if not self._is_active():
            return
        if not self.capture_thinking:
            return
        try:
            call_id = _resolve_call_id(event)
            chunk = (
                safe_getattr(event, "chunk")
                or safe_getattr(event, "thinking")
                or safe_getattr(event, "reasoning")
                or ""
            )
            if not chunk:
                return
            chunk_str = str(chunk)

            with self._lock:
                buf = self._llm_thinking_chunks.setdefault(call_id, [])
                current_len = sum(len(c) for c in buf)
                if current_len + len(chunk_str) <= MAX_TEXT_LENGTH:
                    buf.append(chunk_str)

        except Exception:
            logger.debug("on_llm_thinking_chunk error:\n%s", traceback.format_exc())

    # =========================================================================
    # LLM call completed
    # =========================================================================

    def on_llm_call_completed(self, source: Any, event: Any) -> None:
        """
        Finalise the ``crewai.llm`` span.

        Actions performed (in order)
        -----------------------------
        1. Drain and join ``_llm_stream_chunks[call_id]`` →
           ``llm.streaming_response`` (streaming mode).
        2. Drain and join ``_llm_thinking_chunks[call_id]`` →
           ``llm.thinking_text`` (extended thinking).
        3. Write ``llm.call_type`` from ``event.call_type`` (not available at
           start time).
        4. Write ``llm.response`` from ``event.response`` / response object.
        5. Extract token counts:
           a. Try ``event.response`` usage fields via
              :func:`~crewai_utils.extract_token_usage`.
           b. If counts are missing, check ``_llm_usage_by_call_id[call_id]``
              (populated by the class-level monkey-patch on ``BaseLLM``).
           Write counts directly to ``span.attributes`` (span may be finished).
        6. Calculate cost via :func:`~crewai_utils.calculate_llm_cost`.
        7. Add token/cost deltas to per-crew accumulators via
           :meth:`_accumulate_tokens`.
        8. Write ``llm.duration_ms`` and finish span.
        """
        if not self._is_active():
            return
        try:
            call_id = _resolve_call_id(event)
            self._finish_llm_span(
                call_id=call_id,
                event=event,
                status=ATTR_STATUS_SUCCESS,
                error=None,
            )
        except Exception:
            logger.debug("on_llm_call_completed error:\n%s", traceback.format_exc())

    # =========================================================================
    # LLM call failed
    # =========================================================================

    def on_llm_call_failed(self, source: Any, event: Any) -> None:
        """
        Set ERROR status on the ``crewai.llm`` span and finish it.

        Attributes written
        ------------------
        - ``error.type``       — exception class name
        - ``error.message``    — exception message string
        - ``error.stacktrace`` — formatted traceback (when available)
        - ``llm.duration_ms``  — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            call_id = _resolve_call_id(event)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_llm_span(
                call_id=call_id,
                event=event,
                status=ATTR_STATUS_ERROR,
                error=error,
            )
        except Exception:
            logger.debug("on_llm_call_failed error:\n%s", traceback.format_exc())

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_agent_or_crew_span(
        self, agent_id: Optional[str], crew_id: Optional[str] = None
    ) -> Optional[Any]:
        """Return the best available parent span: agent → crew → None."""
        with self._lock:
            if agent_id and agent_id in self._agent_spans:
                return self._agent_spans[agent_id]
            if crew_id and crew_id in self._crew_spans:
                return self._crew_spans[crew_id].get("span")
        return None

    def _finish_llm_span(
        self,
        call_id: str,
        event: Any,
        status: str,
        error: Any,
    ) -> None:
        """
        Core span-close logic shared by completed and failed paths.

        Drains stream/thinking buffers, resolves token counts and cost, then
        writes all attributes + finishes the span.
        """
        # --- Pop state atomically --------------------------------------------
        with self._lock:
            entry = self._llm_call_spans.pop(call_id, None)
            start_t = self._llm_call_start_times.pop(call_id, None)
            stream_chunks = self._llm_stream_chunks.pop(call_id, [])
            thinking_chunks = self._llm_thinking_chunks.pop(call_id, [])
            # Token counts injected by monkey-patch (may be empty)
            monkey_usage: dict[str, Any] = getattr(
                self, "_llm_usage_by_call_id", {}
            ).pop(call_id, {})

        if entry is None:
            logger.debug("_finish_llm_span: no open entry for call_id=%s", call_id)
            return

        span = entry["span"]
        crew_id: Optional[str] = entry.get("crew_id")
        if not crew_id:
            task_id_fin = entry.get("task_id")
            if task_id_fin:
                with self._lock:
                    crew_id = self._task_to_crew_id.get(str(task_id_fin))

        attrs: dict[str, Any] = {}

        # --- Duration --------------------------------------------------------
        if start_t is not None:
            attrs[ATTR_LLM_DURATION_MS] = duration_ms_monotonic(start_t)

        # --- call_type (only available on the completed event) ---------------
        call_type = safe_getattr(event, "call_type")
        if call_type:
            # Use .name to get bare "TOOL_CALL" / "LLM_CALL" instead of
            # the full enum repr "LLMCallType.TOOL_CALL".
            ct_str = (
                getattr(call_type, "name", None)
                or getattr(call_type, "value", None)
                or str(call_type)
            )
            attrs["llm.call_type"] = ct_str

        # --- Response text ---------------------------------------------------
        response_obj = safe_getattr(event, "response")
        if status == ATTR_STATUS_SUCCESS and response_obj is not None:
            resp_text = _extract_llm_response_text(response_obj, event)
            if resp_text:
                attrs[ATTR_LLM_OUTPUT_TEXT] = truncate_str(resp_text, MAX_TEXT_LENGTH)

        finish_reason = safe_getattr(
            event, "finish_reason"
        ) or _extract_finish_reason_from_response(response_obj)
        if finish_reason:
            attrs[ATTR_LLM_FINISH_REASON] = str(finish_reason)

        # --- Streaming text --------------------------------------------------
        if stream_chunks and self.capture_streaming:
            attrs["llm.streaming_response"] = truncate_str(
                "".join(stream_chunks), MAX_TEXT_LENGTH
            )
            attrs[ATTR_LLM_STREAMING] = True

        # --- Thinking text ---------------------------------------------------
        if thinking_chunks and self.capture_thinking:
            attrs[ATTR_LLM_THINKING_TEXT] = truncate_str(
                "".join(thinking_chunks), MAX_TEXT_LENGTH
            )

        # --- Token counts ----------------------------------------------------
        # Layered merge (each source fills only keys still ``None`` in *token_usage*):
        #   1. ``event.response`` usage           (:func:`extract_token_usage`)
        #   2. ``event.usage`` dict               (CrewAI ``LLMCallCompletedEvent``)
        #   3. monkey-patch buffer                (``pop`` above — always drained)
        #   4. top-level event fields             (edge-case event shapes)
        # ``0`` is treated as a real value, not "missing". ``total_tokens == 0`` is
        # recomputed from input+output when both are known (APIs sometimes emit 0).
        tu0 = extract_token_usage(response_obj)
        token_usage: dict[str, Any] = {
            "input_tokens": tu0.get("input_tokens"),
            "output_tokens": tu0.get("output_tokens"),
            "total_tokens": tu0.get("total_tokens"),
            "cache_read_tokens": None,
            "cache_creation_tokens": None,
            "reasoning_tokens": None,
        }

        event_usage = safe_getattr(event, "usage")
        if isinstance(event_usage, dict) and event_usage:
            _merge_missing_token_usage(
                token_usage, _token_usage_from_event_usage_dict(event_usage)
            )

        _merge_missing_token_usage(
            token_usage, _token_usage_from_monkey_dict(monkey_usage)
        )
        _merge_missing_token_usage(
            token_usage, _token_usage_from_event_top_level(event)
        )

        input_tokens: Optional[int] = _to_int(token_usage.get("input_tokens"))
        output_tokens: Optional[int] = _to_int(token_usage.get("output_tokens"))
        total_tokens: Optional[int] = _to_int(token_usage.get("total_tokens"))

        if (
            (total_tokens is None or total_tokens == 0)
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = input_tokens + output_tokens

        if input_tokens is not None:
            attrs[ATTR_LLM_INPUT_TOKENS] = input_tokens
        if output_tokens is not None:
            attrs[ATTR_LLM_OUTPUT_TOKENS] = output_tokens
        if total_tokens is not None:
            attrs[ATTR_LLM_TOTAL_TOKENS] = total_tokens

        # --- Provider token extras (Anthropic cache, OpenAI reasoning) --------
        _extract_provider_token_extras(attrs, token_usage, response_obj, monkey_usage)

        # --- Cost ------------------------------------------------------------
        _span_attrs = getattr(span, "attributes", None) or {}
        model = (
            (_span_attrs.get(ATTR_LLM_MODEL) if isinstance(_span_attrs, dict) else None)
            or safe_getattr(event, "model")
            or DEFAULT_LLM_MODEL
        )
        cost_info: dict[str, Any] = {}
        if model and (input_tokens or output_tokens):
            cost_info = calculate_llm_cost(str(model), input_tokens, output_tokens)
        if cost_info:
            attrs[ATTR_LLM_COST_INPUT] = cost_info.get("input", 0.0)
            attrs[ATTR_LLM_COST_OUTPUT] = cost_info.get("output", 0.0)
            attrs[ATTR_LLM_COST_TOTAL] = cost_info.get("total", 0.0)
            attrs[ATTR_LLM_COST_CURRENCY] = cost_info.get("currency", "USD")

        # --- Error details ---------------------------------------------------
        if error is not None:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = str(error)
            tb = getattr(error, "__traceback__", None)
            if tb is not None:
                attrs[ATTR_ERROR_STACKTRACE] = "".join(traceback.format_tb(tb))

        # --- Write to span ---------------------------------------------------
        # NOTE: We write directly to span.attributes dict because the span
        # might already be in a "finished" state (some proxy patterns call
        # finish() before firing the completed event).  Both paths are safe:
        # set_attributes() on a live span and dict update on a closed one.
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
                "_finish_llm_span span.finish() error:\n%s",
                traceback.format_exc(),
            )

        # --- Per-crew accumulators -------------------------------------------
        if crew_id and total_tokens:
            total_cost = float(cost_info.get("total", 0.0))
            self._accumulate_tokens(crew_id, total_tokens, total_cost)

        logger.debug(
            "LLM span closed: call_id=%s status=%s tokens=%s cost=%.6f",
            call_id,
            status,
            total_tokens,
            cost_info.get("total", 0.0),
        )


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _resolve_call_id(event: Any) -> str:
    """Return a stable call identifier from the event."""
    return str(
        safe_getattr(event, "call_id")
        or safe_getattr(event, "id")
        or safe_getattr(event, "run_id")
        or id(event)
    )


def _resolve_crew_id(source: Any, event: Any) -> Optional[str]:
    raw = (
        safe_getattr(event, "crew_id")
        or safe_getattr(source, "crew_id")
        or safe_getattr(safe_getattr(source, "crew"), "id")
    )
    return str(raw) if raw is not None else None


def _resolve_task_id(source: Any, event: Any) -> Optional[str]:
    """Return the task_id for this LLM event, or ``None``.

    Aligns with :func:`noveum_trace.integrations.crewai._handlers_agent._resolve_task_id`
    so LLM calls opened from an agent still resolve ``event.task.id`` when
    ``event.task_id`` is unset.
    """
    raw = (
        safe_getattr(event, "task_id")
        or safe_getattr(safe_getattr(event, "task"), "id")
        or safe_getattr(source, "task_id")
        or safe_getattr(safe_getattr(source, "task"), "id")
    )
    return str(raw) if raw is not None else None


def _build_llm_start_attributes(
    source: Any,
    event: Any,
    call_id: str,
    *,
    capture_llm_messages: bool = True,
    capture_tool_schemas: bool = True,
) -> dict[str, Any]:
    """
    Build the full set of span attributes for an LLM call start event.

    Note: ``call_type`` is intentionally absent — it is not available until
    the completed event fires.

    ``capture_llm_messages`` / ``capture_tool_schemas`` mirror listener flags so
    prompts, message bodies, and tool definitions can be omitted when disabled.
    """
    attrs: dict[str, Any] = {ATTR_LLM_CALL_ID: call_id}

    # --- Model / provider ---------------------------------------------------
    model = (
        safe_getattr(event, "model")
        or safe_getattr(source, "model_name")
        or safe_getattr(source, "model")
        or DEFAULT_LLM_MODEL
    )
    attrs[ATTR_LLM_MODEL] = str(model)

    provider = _infer_provider(model, source, event)
    if provider:
        attrs[ATTR_LLM_PROVIDER] = provider

    _event_inputs = safe_getattr(event, "inputs")

    # --- Messages -----------------------------------------------------------
    if capture_llm_messages:
        messages = safe_getattr(event, "messages") or (
            _event_inputs.get("messages") if isinstance(_event_inputs, dict) else None
        )

        if messages:
            # System prompt — extracted separately for discoverability
            system_prompt = extract_system_prompt(messages)
            if system_prompt:
                attrs[ATTR_LLM_SYSTEM_PROMPT] = truncate_str(
                    system_prompt, MAX_SYSTEM_PROMPT_LENGTH
                )

            # Full messages serialized to JSON
            msgs_json = messages_to_json(messages)
            if msgs_json:
                attrs[ATTR_LLM_INPUT_MESSAGES] = msgs_json

    # --- Tools --------------------------------------------------------------
    if capture_tool_schemas:
        # CrewAI often puts ``tools`` on ``event.inputs`` when ``source`` is the LLM
        # object (no ``.tools``), matching how ``messages`` may appear only in inputs.
        tools = safe_getattr(event, "tools") or safe_getattr(source, "tools")
        if not tools and isinstance(_event_inputs, dict):
            tools = _event_inputs.get("tools")
        if tools:
            tool_json = serialize_tool_schema(tools)
            if tool_json:
                attrs[ATTR_LLM_TOOLS] = tool_json
            merge_available_tools_attributes(attrs, tools, "llm")

    # --- Agent / task correlation -------------------------------------------
    agent_role = safe_getattr(event, "agent_role") or safe_getattr(source, "role")
    if agent_role:
        attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

    task_name = (
        safe_getattr(event, "task_name")
        or safe_getattr(event, "task_description")
        or safe_getattr(safe_getattr(source, "task"), "name")
        or safe_getattr(safe_getattr(source, "task"), "description")
    )
    if task_name:
        attrs["task.name"] = truncate_str(str(task_name), 512)

    # --- Resolve LLM object (used by both streaming flag + config params) ---
    # source may be an Agent (source.llm holds params) OR a BaseLLM (source IS params).
    llm_obj = safe_getattr(source, "llm") or source

    # --- Streaming flag -----------------------------------------------------
    # CrewAI BaseLLM uses either ``stream`` or ``streaming`` as the field name.
    streaming = (
        safe_getattr(event, "streaming")
        or safe_getattr(event, "stream")
        or safe_getattr(llm_obj, "streaming")
        or safe_getattr(llm_obj, "stream")
        or safe_getattr(source, "streaming")
        or safe_getattr(source, "stream")
    )
    if streaming is not None:
        attrs[ATTR_LLM_STREAMING] = bool(streaming)

    # --- LLM config params (temperature, max_tokens, top_p, seed) ----------
    # Use explicit is-None checks so that 0 / 0.0 values are never lost.
    for obj_attr, span_attr in (
        ("temperature", ATTR_LLM_TEMPERATURE),
        ("max_tokens", ATTR_LLM_MAX_TOKENS),
        ("top_p", ATTR_LLM_TOP_P),
        ("seed", ATTR_LLM_SEED),
    ):
        val = safe_getattr(event, obj_attr)
        if val is None:
            val = safe_getattr(llm_obj, obj_attr)
        if val is None and llm_obj is not source:
            # Direct fallback when source IS the BaseLLM
            val = safe_getattr(source, obj_attr)
        if val is None:
            # Last-resort: some providers store config in a kwargs dict
            for kw_attr in (
                "litellm_kwargs",
                "llm_kwargs",
                "extra_kwargs",
                "model_kwargs",
            ):
                kw = safe_getattr(llm_obj, kw_attr) or safe_getattr(source, kw_attr)
                if isinstance(kw, dict) and obj_attr in kw:
                    val = kw[obj_attr]
                    break
        if val is not None:
            attrs[span_attr] = val
            logger.debug(
                "LLM param captured: %s=%r (source=%s)",
                span_attr,
                val,
                type(source).__name__,
            )

    return attrs


def _infer_provider(model: str, source: Any, event: Any) -> Optional[str]:
    """
    Infer the LLM provider from the model name, source object, or event.

    Falls back to ``DEFAULT_LLM_PROVIDER`` only when nothing else matches.
    """
    # Explicit provider on event or source
    provider = safe_getattr(event, "provider") or safe_getattr(source, "provider")
    if provider:
        return str(provider)

    # Infer from model name prefix
    model_lower = model.lower() if model else ""
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "google"
    if "llama" in model_lower:
        return "meta"
    if "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    if "deepseek" in model_lower:
        return "deepseek"
    if "command" in model_lower:
        return "cohere"

    # Class name of the LLM wrapper sometimes carries the provider
    llm = safe_getattr(source, "llm")
    if llm is not None:
        class_name = type(llm).__name__.lower()
        for keyword, prov in (
            ("openai", "openai"),
            ("anthropic", "anthropic"),
            ("google", "google"),
            ("gemini", "google"),
            ("bedrock", "aws"),
            ("mistral", "mistral"),
            ("cohere", "cohere"),
            ("litellm", "litellm"),
        ):
            if keyword in class_name:
                return prov

    return None


def _extract_llm_response_text(response_obj: Any, event: Any) -> Optional[str]:
    """
    Extract the generated text from either the response object or the event.

    Tries ``event.text`` / ``event.output`` first (most direct).  For
    ``LLM_CALL`` type responses CrewAI puts the final answer directly on
    ``event.response`` as a plain Python string — return it immediately.
    Falls back to :func:`extract_response_text` for provider-specific parsing
    of richer response objects (ChatCompletion, ModelResponse, etc.).
    """
    # Direct event fields (CrewAI may populate these at the event level)
    for attr in ("text", "output", "content", "result"):
        val = safe_getattr(event, attr)
        if isinstance(val, str) and val:
            return val

    # When CrewAI fires LLMCallCompletedEvent with call_type=LLM_CALL,
    # event.response is a plain string containing the final answer.
    if isinstance(response_obj, str) and response_obj:
        return response_obj

    # Provider-agnostic response object parsing (ChatCompletion / ModelResponse)
    return extract_response_text(response_obj)


def _extract_finish_reason_from_response(response_obj: Any) -> Optional[str]:
    """Extract stop/finish reason from a response object."""
    if response_obj is None:
        return None
    # OpenAI choices
    choices = safe_getattr(response_obj, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        reason = safe_getattr(choices[0], "finish_reason")
        if reason:
            return str(reason)
    # Anthropic
    reason = safe_getattr(response_obj, "stop_reason")
    if reason:
        return str(reason)
    # Google
    candidates = safe_getattr(response_obj, "candidates")
    if isinstance(candidates, (list, tuple)) and candidates:
        reason = safe_getattr(candidates[0], "finish_reason")
        if reason:
            return str(reason)
    return None


def _extract_provider_token_extras(
    attrs: dict[str, Any],
    token_usage: dict[str, Any],
    response_obj: Any,
    monkey_usage: dict[str, Any],
) -> None:
    """
    Extract provider-specific token counts and write them to *attrs*.

    Supports:
    - Anthropic cache_read_input_tokens / cache_creation_input_tokens
    - OpenAI o-series reasoning_tokens (inside completion_tokens_details)
    """
    # --- Anthropic cache tokens ------------------------------------------
    _resp_usage = (
        safe_getattr(response_obj, "usage") if response_obj is not None else None
    )

    cache_read = _first_non_none_llm(
        token_usage.get("cache_read_tokens"),
        token_usage.get("cache_read_input_tokens"),
        monkey_usage.get("cache_read_tokens"),
        monkey_usage.get("cache_read_input_tokens"),
        (
            safe_getattr(_resp_usage, "cache_read_input_tokens")
            if _resp_usage is not None
            else None
        ),
    )
    if cache_read is not None:
        val = _to_int(cache_read)
        if val is not None:
            attrs[ATTR_LLM_CACHE_READ_TOKENS] = val

    cache_creation = _first_non_none_llm(
        token_usage.get("cache_creation_tokens"),
        token_usage.get("cache_creation_input_tokens"),
        monkey_usage.get("cache_creation_tokens"),
        monkey_usage.get("cache_creation_input_tokens"),
        (
            safe_getattr(_resp_usage, "cache_creation_input_tokens")
            if _resp_usage is not None
            else None
        ),
    )
    if cache_creation is not None:
        val = _to_int(cache_creation)
        if val is not None:
            attrs[ATTR_LLM_CACHE_CREATION_TOKENS] = val

    # --- OpenAI reasoning tokens (o1 / o3 series) ------------------------
    reasoning = _first_non_none_llm(
        token_usage.get("reasoning_tokens"),
        monkey_usage.get("reasoning_tokens"),
    )
    if reasoning is None and response_obj is not None:
        # OpenAI: response.usage.completion_tokens_details.reasoning_tokens
        usage = safe_getattr(response_obj, "usage")
        if usage is not None:
            details = safe_getattr(usage, "completion_tokens_details")
            if details is not None:
                reasoning = safe_getattr(details, "reasoning_tokens")
    if reasoning is not None:
        val = _to_int(reasoning)
        if val is not None:
            attrs[ATTR_LLM_REASONING_TOKENS] = val


def _to_int(value: Any) -> Optional[int]:
    """Coerce *value* to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
