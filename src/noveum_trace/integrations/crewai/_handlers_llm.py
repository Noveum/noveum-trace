"""
LLM-call event handler mixin for NoveumCrewAIListener.

Handles CrewAI ``BaseEventListener`` LLM events:

  - ``on_llm_call_started``    → open ``crewai.llm`` child span under the owning
                                  agent span; capture model, call_id, full messages
                                  list as ``llm.messages`` JSON, system prompt
                                  extracted separately as ``llm.system_prompt``,
                                  tool definitions as ``llm.tools``;
                                  ``llm.available_tools.*`` (count, names,
                                  descriptions, schemas); available
                                  functions as ``llm.available_functions``; plus
                                  ``agent.role`` and ``task.name`` for correlation.
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
    DEFAULT_LLM_PROVIDER,
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
    safe_getattr,
    safe_json_dumps,
    serialize_tool_schema,
    truncate_str,
)

logger = logging.getLogger(__name__)


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
        Open a ``crewai.llm`` child span under the owning agent span.

        Attributes set at span open
        ---------------------------
        - ``llm.call_id``            — unique call identifier (for correlation)
        - ``llm.model``              — model name (from event or agent LLM)
        - ``llm.provider``           — inferred provider name
        - ``llm.system_prompt``      — extracted from the messages list
        - ``llm.messages``           — full messages list serialized as JSON
        - ``llm.tools``              — tool schema JSON (when tools provided)
        - ``llm.available_tools.*``  — count, names, descriptions, schemas JSON
        - ``llm.available_functions``— function names list JSON
        - ``agent.role``             — role of the executing agent (correlation)
        - ``task.name``              — name/description of the current task
        - ``llm.streaming``          — bool, True when streaming mode detected

        Note: ``llm.call_type`` is intentionally NOT set here — it is only
        available on the completed event and is written there instead.
        """
        if not self._is_active():
            return
        try:
            call_id = _resolve_call_id(event)
            agent_id = _resolve_agent_id(source, event)
            crew_id = _resolve_crew_id(source, event)

            # CrewAI events carry no crew_id — fall back to the active crew when
            # there is exactly one, or the most-recently started one otherwise.
            if not crew_id:
                with self._lock:
                    if self._crew_spans:
                        crew_id = next(reversed(self._crew_spans))

            attrs = _build_llm_start_attributes(source, event, call_id)
            start_t = monotonic_now()

            # Parent: agent span (most common), else crew span, else None
            parent_span = self._get_agent_or_crew_span(agent_id, crew_id)

            span = self._create_child_span(
                SPAN_LLM,
                parent_span=parent_span,
                attributes=attrs,
            )

            with self._lock:
                self._llm_call_spans[call_id] = {
                    "span": span,
                    "crew_id": crew_id,
                    "agent_id": agent_id,
                }
                self._llm_call_start_times[call_id] = start_t
                # Pre-initialise stream / thinking buffers
                self._llm_stream_chunks.setdefault(call_id, [])
                self._llm_thinking_chunks.setdefault(call_id, [])

            logger.debug(
                "LLM span opened: call_id=%s agent_id=%s", call_id, agent_id
            )

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
        """
        if not self._is_active():
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
        """
        if not self._is_active():
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
            logger.debug(
                "on_llm_thinking_chunk error:\n%s", traceback.format_exc()
            )

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
            logger.debug(
                "on_llm_call_completed error:\n%s", traceback.format_exc()
            )

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
        self, agent_id: Optional[str], crew_id: Optional[str]
    ) -> Any:
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
            logger.debug(
                "_finish_llm_span: no open entry for call_id=%s", call_id
            )
            return

        span = entry["span"]
        crew_id: Optional[str] = entry.get("crew_id")

        attrs: dict[str, Any] = {}

        # --- Duration --------------------------------------------------------
        if start_t is not None:
            attrs[ATTR_LLM_DURATION_MS] = duration_ms_monotonic(start_t)

        # --- call_type (only available on the completed event) ---------------
        call_type = safe_getattr(event, "call_type")
        if call_type:
            # Use .name to get bare "TOOL_CALL" / "LLM_CALL" instead of
            # the full enum repr "LLMCallType.TOOL_CALL".
            ct_str = getattr(call_type, "name", None) or getattr(call_type, "value", None) or str(call_type)
            attrs["llm.call_type"] = ct_str

        # --- Response text ---------------------------------------------------
        response_obj = safe_getattr(event, "response")
        if status == ATTR_STATUS_SUCCESS and response_obj is not None:
            resp_text = _extract_llm_response_text(response_obj, event)
            if resp_text:
                attrs[ATTR_LLM_OUTPUT_TEXT] = truncate_str(resp_text, MAX_TEXT_LENGTH)

        finish_reason = (
            safe_getattr(event, "finish_reason")
            or _extract_finish_reason_from_response(response_obj)
        )
        if finish_reason:
            attrs[ATTR_LLM_FINISH_REASON] = str(finish_reason)

        # --- Streaming text --------------------------------------------------
        if stream_chunks:
            attrs["llm.streaming_response"] = truncate_str(
                "".join(stream_chunks), MAX_TEXT_LENGTH
            )
            attrs[ATTR_LLM_STREAMING] = True

        # --- Thinking text ---------------------------------------------------
        if thinking_chunks:
            attrs[ATTR_LLM_THINKING_TEXT] = truncate_str(
                "".join(thinking_chunks), MAX_TEXT_LENGTH
            )

        # --- Token counts ----------------------------------------------------
        # Priority order (first non-None wins):
        #   1. event.response object usage fields  (LiteLLM ModelResponse / ChatCompletion)
        #   2. event.usage dict                    (CrewAI passes usage= on LLMCallCompletedEvent)
        #   3. monkey-patch buffer                 (_track_token_usage_internal intercept)
        #   4. direct top-level event fields       (fallback for edge-case event shapes)
        token_usage = extract_token_usage(response_obj)

        if token_usage.get("input_tokens") is None:
            # Try event.usage dict (CrewAI 1.x sets this on LLMCallCompletedEvent)
            event_usage = safe_getattr(event, "usage")
            if isinstance(event_usage, dict) and event_usage:
                token_usage = {
                    "input_tokens": event_usage.get("input_tokens")
                        or event_usage.get("prompt_tokens")
                        or event_usage.get("prompt_token_count"),
                    "output_tokens": event_usage.get("output_tokens")
                        or event_usage.get("completion_tokens")
                        or event_usage.get("candidates_token_count"),
                    "total_tokens": event_usage.get("total_tokens")
                        or event_usage.get("total_token_count"),
                    "cache_read_tokens": event_usage.get("cached_tokens")
                        or event_usage.get("cached_prompt_tokens")
                        or event_usage.get("cache_read_input_tokens"),
                    "cache_creation_tokens": event_usage.get("cache_creation_input_tokens"),
                    "reasoning_tokens": event_usage.get("reasoning_tokens"),
                }
            elif monkey_usage:
                # Monkey-patch buffer (_track_token_usage_internal intercept)
                token_usage = {
                    "input_tokens": monkey_usage.get("input_tokens")
                        or monkey_usage.get("prompt_tokens"),
                    "output_tokens": monkey_usage.get("output_tokens")
                        or monkey_usage.get("completion_tokens"),
                    "total_tokens": monkey_usage.get("total_tokens"),
                    "cache_read_tokens": monkey_usage.get("cache_read_tokens"),
                    "cache_creation_tokens": monkey_usage.get("cache_creation_tokens"),
                    "reasoning_tokens": monkey_usage.get("reasoning_tokens"),
                }
            else:
                # Last resort: top-level event fields
                token_usage = {
                    "input_tokens": safe_getattr(event, "prompt_tokens")
                        or safe_getattr(event, "input_tokens"),
                    "output_tokens": safe_getattr(event, "completion_tokens")
                        or safe_getattr(event, "output_tokens"),
                    "total_tokens": safe_getattr(event, "total_tokens"),
                }

        # Also drain monkey-patch buffer even when event.usage already provided
        # (ensures the buffer doesn't grow unbounded)
        if monkey_usage and token_usage.get("input_tokens") is not None:
            pass  # already consumed via pop() above

        input_tokens: Optional[int] = _to_int(token_usage.get("input_tokens"))
        output_tokens: Optional[int] = _to_int(token_usage.get("output_tokens"))
        total_tokens: Optional[int] = _to_int(token_usage.get("total_tokens"))

        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        if input_tokens is not None:
            attrs[ATTR_LLM_INPUT_TOKENS] = input_tokens
        if output_tokens is not None:
            attrs[ATTR_LLM_OUTPUT_TOKENS] = output_tokens
        if total_tokens is not None:
            attrs[ATTR_LLM_TOTAL_TOKENS] = total_tokens

        # --- Provider token extras (Anthropic cache, OpenAI reasoning) --------
        _extract_provider_token_extras(attrs, token_usage, response_obj,
                                       monkey_usage)

        # --- Cost ------------------------------------------------------------
        _span_attrs = getattr(span, "attributes", None) or {}
        model = (
            _span_attrs.get(ATTR_LLM_MODEL) if isinstance(_span_attrs, dict) else None
        ) or safe_getattr(event, "model") or DEFAULT_LLM_MODEL
        cost_info: dict[str, Any] = {}
        if model and (input_tokens or output_tokens):
            cost_info = calculate_llm_cost(
                str(model), input_tokens, output_tokens
            )
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
        _write_attrs_to_span(span, attrs)

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


def _resolve_agent_id(source: Any, event: Any) -> Optional[str]:
    raw = (
        safe_getattr(event, "agent_id")
        or safe_getattr(source, "id")
        or safe_getattr(source, "agent_id")
    )
    return str(raw) if raw is not None else None


def _resolve_crew_id(source: Any, event: Any) -> Optional[str]:
    raw = (
        safe_getattr(event, "crew_id")
        or safe_getattr(source, "crew_id")
        or safe_getattr(safe_getattr(source, "crew"), "id")
    )
    return str(raw) if raw is not None else None


def _build_llm_start_attributes(
    source: Any, event: Any, call_id: str
) -> dict[str, Any]:
    """
    Build the full set of span attributes for an LLM call start event.

    Note: ``call_type`` is intentionally absent — it is not available until
    the completed event fires.
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

    # --- Messages -----------------------------------------------------------
    _event_inputs = safe_getattr(event, "inputs")
    messages = (
        safe_getattr(event, "messages")
        or (_event_inputs.get("messages") if isinstance(_event_inputs, dict) else None)
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

    # --- Tools / functions --------------------------------------------------
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

    available_functions = safe_getattr(event, "available_functions") or safe_getattr(
        event, "functions"
    )
    if not available_functions and isinstance(_event_inputs, dict):
        available_functions = _event_inputs.get("available_functions") or _event_inputs.get(
            "functions"
        )
    if available_functions:
        try:
            if isinstance(available_functions, (list, tuple)):
                fn_names = []
                for fn in available_functions:
                    name = safe_getattr(fn, "name") or (fn if isinstance(fn, str) else str(fn))
                    fn_names.append(name)
                attrs["llm.available_functions"] = safe_json_dumps(fn_names)
            else:
                attrs["llm.available_functions"] = str(available_functions)
        except Exception:
            pass

    # --- Agent / task correlation -------------------------------------------
    agent_role = (
        safe_getattr(event, "agent_role")
        or safe_getattr(source, "role")
    )
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
            for kw_attr in ("litellm_kwargs", "llm_kwargs", "extra_kwargs", "model_kwargs"):
                kw = safe_getattr(llm_obj, kw_attr) or safe_getattr(source, kw_attr)
                if isinstance(kw, dict) and obj_attr in kw:
                    val = kw[obj_attr]
                    break
        if val is not None:
            attrs[span_attr] = val
            logger.debug("LLM param captured: %s=%r (source=%s)", span_attr, val, type(source).__name__)

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


def _write_attrs_to_span(span: Any, attrs: dict[str, Any]) -> None:
    """
    Write *attrs* to *span*.

    Attempts ``span.set_attributes()`` first (live span).  If the span is
    already finished or does not implement ``set_attributes``, falls back to
    writing directly into ``span.attributes`` dict so token counts and cost
    are never silently dropped.
    """
    if not attrs or span is None:
        return
    try:
        if hasattr(span, "set_attributes"):
            span.set_attributes(attrs)
            return
    except Exception:
        pass
    # Direct dict write — works on finished Span objects
    try:
        if hasattr(span, "attributes") and isinstance(span.attributes, dict):
            span.attributes.update(attrs)
    except Exception as exc:
        logger.debug("_write_attrs_to_span fallback failed: %s", exc)


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
    _resp_usage = safe_getattr(response_obj, "usage") if response_obj is not None else None

    cache_read = (
        token_usage.get("cache_read_tokens")
        or token_usage.get("cache_read_input_tokens")
        or monkey_usage.get("cache_read_tokens")
        or monkey_usage.get("cache_read_input_tokens")
        or (safe_getattr(_resp_usage, "cache_read_input_tokens") if _resp_usage is not None else None)
    )
    if cache_read is not None:
        val = _to_int(cache_read)
        if val is not None:
            attrs[ATTR_LLM_CACHE_READ_TOKENS] = val

    cache_creation = (
        token_usage.get("cache_creation_tokens")
        or token_usage.get("cache_creation_input_tokens")
        or monkey_usage.get("cache_creation_tokens")
        or monkey_usage.get("cache_creation_input_tokens")
        or (safe_getattr(_resp_usage, "cache_creation_input_tokens") if _resp_usage is not None else None)
    )
    if cache_creation is not None:
        val = _to_int(cache_creation)
        if val is not None:
            attrs[ATTR_LLM_CACHE_CREATION_TOKENS] = val

    # --- OpenAI reasoning tokens (o1 / o3 series) ------------------------
    reasoning = (
        token_usage.get("reasoning_tokens")
        or monkey_usage.get("reasoning_tokens")
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
