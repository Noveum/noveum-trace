"""
LLM, function-call, thought, and summarization frame handler mixin for
NoveumTraceObserver.

Handles:
  - LLMContextFrame / OpenAILLMContextFrame  — stash input messages + tools (merge)
  - LLMMessagesFrame / Update / Append       — legacy message stash
  - LLMSetToolsFrame / LLMSetToolChoiceFrame — tools + tool_choice stash
  - LLMContextSummaryRequestFrame            — duplicate request + context on turn/trace
  - LLMFullResponseStartFrame                — open pipecat.llm span, flush stash
  - LLMTextFrame / VisionTextFrame           — accumulate assistant text chunks
  - LLMFullResponseEndFrame                  — close pipecat.llm span; write thought
                                               and function-call attribute lists
  - LLMThoughtStartFrame                     — clear thought buffer (no child span)
  - LLMThoughtTextFrame                      — accumulate thought text chunks
  - LLMThoughtEndFrame                       — append completed thought to llm.thoughts list
  - FunctionCallsStartedFrame                — debug log (no span / no event)
  - FunctionCallInProgressFrame              — record call on the requesting llm span
  - FunctionCallResultFrame                  — record result on the requesting llm span
  - FunctionCallCancelFrame                  — record cancelled result on the requesting llm span
  - LLMContextSummaryResultFrame             — write summary to turn/trace

Thought blocks and function calls are stored as flat attribute lists on the
pipecat.llm span rather than as child spans:
  llm.thoughts                — list[str], one entry per thought block
  llm.thought_signatures      — list[str], one entry per thought block (may be "")
  llm.function_calls          — list[dict], one entry per FunctionCallInProgressFrame
  llm.function_call_results   — list[dict], one entry per result/cancel frame
"""

from __future__ import annotations

import json
import logging
from typing import Any

from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
from noveum_trace.integrations.pipecat.pipecat_constants import (
    MAX_TEXT_BUFFER_LENGTH,
    SPAN_LLM,
)
from noveum_trace.integrations.pipecat.pipecat_utils import (
    derive_provider,
    extract_function_call_data,
    extract_llm_context_data,
    extract_service_settings,
    json_dumps_messages,
    merge_appended_messages_json,
    merge_llm_pending_stash,
    serialize_tool_choice_field,
    serialize_tools_field,
    truncate_for_trace_attr,
)

logger = logging.getLogger(__name__)


def _append_json_list_attr(span: Any, key: str, item: Any) -> None:
    """Append ``item`` to a JSON-encoded list attribute on ``span``.

    The attribute is always stored as a JSON **string** (consistent with
    ``llm.tools`` / ``llm.input`` per TRACE_DESIGN §5.4) and supports incremental
    appends, so function calls/results can be written to their owning span the
    moment their frames arrive rather than being deferred to ``LLMFullResponseEnd``.
    """
    raw = span.attributes.get(key)
    if isinstance(raw, str):
        try:
            current = json.loads(raw)
        except Exception:  # pragma: no cover - defensive
            current = []
        if not isinstance(current, list):
            current = []
    elif isinstance(raw, list):
        current = raw
    else:
        current = []
    current.append(item)
    span.attributes[key] = json.dumps(current, default=str)


def _thought_signature_of(message: Any) -> Any:
    """Return the signature string if ``message`` is a Gemini thought-signature
    append message, else ``None`` (B8/B9).

    Gemini routes thought signatures through ``LLMMessagesAppendFrame`` as a
    provider-specific message ``{"type": "thought_signature", "signature": ...}``
    (wrapped in an ``LLMSpecificMessage`` whose ``.message`` holds that dict).
    Returns ``""`` for a thought-signature message with an empty/missing signature
    so the caller can still drop it from ``llm.input`` even when there is nothing
    to capture; returns ``None`` for any other message.
    """
    msg = getattr(message, "message", message)
    if isinstance(msg, dict) and msg.get("type") == "thought_signature":
        return str(msg.get("signature") or "")
    return None


# Settings keys to copy from _settings → llm.* span attributes
_LLM_SETTINGS_MAP: tuple[tuple[str, str], ...] = (
    ("model", "llm.model"),
    ("system_instruction", "llm.system_prompt"),
    ("temperature", "llm.temperature"),
    ("max_tokens", "llm.max_tokens"),
    ("max_completion_tokens", "llm.max_tokens"),  # OpenAI alias, may overwrite
    ("top_p", "llm.top_p"),
    ("top_k", "llm.top_k"),
    ("frequency_penalty", "llm.frequency_penalty"),
    ("presence_penalty", "llm.presence_penalty"),
    ("seed", "llm.seed"),
    # Gemini thinking / reasoning config (D3)
    ("thinking_budget", "llm.thinking_budget"),
    ("thinking_level", "llm.thinking_level"),
    ("include_thoughts", "llm.include_thoughts"),
    ("thinking_enabled", "llm.thinking_enabled"),
    ("thinking_config_source", "llm.thinking_config_source"),
)


class _LLMHandlersMixin(_PipecatObserverMixinBase):
    """Handler methods for LLM response, thought, function-call, and summarization frames."""

    # State attributes declared in NoveumTraceObserver.__init__:
    #   _trace, _capture_text, _capture_function_calls,
    #   _llm_text_buffer, _active_llm_span, _current_turn_span,
    #   _pending_function_calls, _function_call_owner, _resolved_function_call_ids,
    #   _using_external_turn_tracking, _pending_llm_context,
    #   _llm_thought_buffer, _llm_thoughts_list, _llm_thought_signatures_list
    # Helpers: _create_child_span(), _start_new_turn()

    # ---------------------------------------------------------------------- #
    # Context frame (stash input + tools)                                     #
    # ---------------------------------------------------------------------- #

    def _merge_pending_llm(self, updates: dict[str, Any]) -> None:
        """Merge non-empty stash keys into ``_pending_llm_context``."""
        if not updates:
            return

        merge_llm_pending_stash(self._pending_llm_context, updates)

    async def _handle_llm_context(self, data: Any) -> None:
        """
        ``LLMContextFrame`` / ``OpenAILLMContextFrame``: stash messages and tools.

        Merges into ``_pending_llm_context`` (does not wipe alternate-path keys).
        Flushed into the ``pipecat.llm`` span when ``LLMFullResponseStartFrame`` fires.
        """
        frame = data.frame
        context = getattr(frame, "context", None)
        if context is None:
            return
        try:
            extracted = extract_llm_context_data(context)
            self._merge_pending_llm(extracted)
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Failed to handle LLM context frame: %s", e)

    async def _handle_llm_messages_replace(self, data: Any) -> None:
        """``LLMMessagesFrame`` / ``LLMMessagesUpdateFrame``: replace stashed messages."""
        frame = data.frame
        messages = getattr(frame, "messages", None)
        dumped = json_dumps_messages(messages)
        if dumped:
            self._merge_pending_llm({"messages": dumped})

    async def _handle_llm_messages_append(self, data: Any) -> None:
        """``LLMMessagesAppendFrame``: append to stashed messages JSON.

        B8/B9: Gemini delivers thought signatures through this frame as
        ``{"type": "thought_signature", "signature": ...}`` messages. Those are
        captured into ``_pending_thought_signatures`` (flushed to
        ``llm.thought_signatures`` at response end) and kept OUT of the stashed
        context — otherwise the opaque signature blob leaks into the next LLM
        span's ``llm.input``.
        """
        frame = data.frame
        new_msgs = getattr(frame, "messages", None)
        if not new_msgs:
            return

        kept = []
        for m in new_msgs:
            sig = _thought_signature_of(m)
            if sig is not None:
                if sig:
                    self._pending_thought_signatures.append(sig)
                continue
            kept.append(m)
        if not kept:
            return

        prev = self._pending_llm_context.get("messages")
        merged = merge_appended_messages_json(prev, kept)
        if merged:

            self._pending_llm_context["messages"] = merged

    async def _handle_llm_set_tools(self, data: Any) -> None:
        """``LLMSetToolsFrame``: stash tool definitions JSON."""
        frame = data.frame
        tools = getattr(frame, "tools", None)
        dumped = serialize_tools_field(tools)
        if dumped:
            self._merge_pending_llm({"tools": dumped})

    async def _handle_llm_set_tool_choice(self, data: Any) -> None:
        """``LLMSetToolChoiceFrame``: stash tool choice for ``llm.tool_choice``."""
        frame = data.frame
        choice = getattr(frame, "tool_choice", None)
        dumped = serialize_tool_choice_field(choice)
        if dumped:
            self._merge_pending_llm({"tool_choice": dumped})

    async def _handle_llm_summary_request(self, data: Any) -> None:
        """
        ``LLMContextSummaryRequestFrame``: duplicate request parameters and full
        context (messages + tools) onto the active turn or trace.
        """
        frame = data.frame

        target = self._current_turn_span or self._trace
        if not target:
            return

        req_id = getattr(frame, "request_id", None)
        if req_id is not None:
            target.attributes["llm.summary.request_id"] = str(req_id)

        mink = getattr(frame, "min_messages_to_keep", None)
        if mink is not None:
            target.attributes["llm.summary.request.min_messages_to_keep"] = int(mink)

        tgt_tok = getattr(frame, "target_context_tokens", None)
        if tgt_tok is not None:
            target.attributes["llm.summary.request.target_context_tokens"] = int(
                tgt_tok
            )

        prompt = getattr(frame, "summarization_prompt", None)
        if prompt:
            target.attributes["llm.summary.request.summarization_prompt"] = (
                truncate_for_trace_attr(str(prompt))
            )

        timeout = getattr(frame, "summarization_timeout", None)
        if timeout is not None:
            target.attributes["llm.summary.request.summarization_timeout_sec"] = float(
                timeout
            )

        ctx = getattr(frame, "context", None)
        if ctx is not None:
            try:
                extracted = extract_llm_context_data(ctx)
                if extracted.get("messages"):
                    target.attributes["llm.summary.request.input"] = extracted[
                        "messages"
                    ]
                if extracted.get("tools"):
                    target.attributes["llm.summary.request.tools"] = extracted["tools"]
            except Exception:  # pylint: disable=broad-except
                pass

    # ---------------------------------------------------------------------- #
    # LLM response span lifecycle                                             #
    # ---------------------------------------------------------------------- #

    async def _handle_llm_response_start(self, data: Any) -> None:
        """
        ``LLMFullResponseStartFrame``: open a ``pipecat.llm`` child span.

        Attributes set:
        - From ``data.source._settings``: ``llm.model``, ``llm.system_prompt``,
          ``llm.temperature``, ``llm.max_tokens``, ``llm.top_p``, ``llm.top_k``,
          ``llm.frequency_penalty``, ``llm.presence_penalty``, ``llm.seed``.
        - From stash (context / message / tool frames): ``llm.input``, ``llm.tools``,
          ``llm.tool_choice`` (JSON when set).
        """
        if not self._trace:
            return

        if self._current_turn_span is None and not self._using_external_turn_tracking:
            await self._start_new_turn()

        self._llm_text_buffer.clear()
        # A new LLM span is opening — the stale backref is no longer valid.
        self._last_llm_span = None

        attributes: dict[str, Any] = {}

        # Extract all available settings from the source processor
        source = getattr(data, "source", None)
        if source:
            settings = extract_service_settings(source)
            for settings_key, attr_key in _LLM_SETTINGS_MAP:
                val = settings.get(settings_key)
                if val is not None:
                    attributes[attr_key] = val
            provider = derive_provider(source, settings.get("model"))
            if provider:
                attributes["llm.provider"] = provider

        # Flush stashed context data (Path A + Path B frames)
        pending = self._pending_llm_context
        if pending:
            if pending.get("messages"):
                attributes["llm.input"] = pending["messages"]
            if pending.get("tools"):
                attributes["llm.tools"] = pending["tools"]
            if pending.get("tool_choice"):
                attributes["llm.tool_choice"] = pending["tool_choice"]
            self._pending_llm_context = {}

        self._active_llm_span = (
            self._create_child_span(  # pylint: disable=assignment-from-no-return
                SPAN_LLM,
                parent_span=self._current_turn_span,
                attributes=attributes,
            )
        )

    async def _handle_llm_text(self, data: Any) -> None:
        """
        ``LLMTextFrame`` / ``VisionTextFrame``: accumulate assistant text chunks.

        The buffer is written to ``llm.output`` when the response ends.
        Guarded to prevent runaway memory usage.
        """
        if not self._capture_text:
            return
        frame = data.frame
        text = getattr(frame, "text", None)
        if text:

            self._llm_text_buffer.append(str(text))

            if sum(len(t) for t in self._llm_text_buffer) > MAX_TEXT_BUFFER_LENGTH:

                self._llm_text_buffer = self._llm_text_buffer[-100:]

    async def _handle_llm_response_end(self, _data: Any) -> None:
        """
        ``LLMFullResponseEndFrame``: finish the active ``pipecat.llm`` span.

        Attributes written:
        - ``llm.output`` — joined text buffer (when ``capture_text=True``)
        - ``llm.thoughts`` — list of thought-text strings accumulated this response
        - ``llm.thought_signatures`` — matching list of signature strings

        Note: ``llm.function_calls`` / ``llm.function_call_results`` are written to
        their owning span in the function-call handlers (not here), so they survive
        even when no follow-up LLM response fires and regardless of frame ordering.
        """
        # Defensive: flush any unclosed thought block into the list
        if self._llm_thought_buffer:

            thought_text = "".join(self._llm_thought_buffer)

            self._llm_thoughts_list.append(thought_text)
            self._llm_thought_signatures_list.append("")
        self._llm_thought_buffer.clear()

        span = self._active_llm_span
        if not span:
            self._llm_thoughts_list.clear()
            self._llm_thought_signatures_list.clear()
            self._pending_thought_signatures.clear()
            self._llm_text_buffer.clear()
            return
        self._active_llm_span = None
        # Keep a backref so MetricsFrame data (token counts, processing time) arriving
        # after this span closes can still be attached to the right span, and so that
        # FunctionCallInProgressFrame arriving between span 1 and span 2 can be written
        # directly to span 1.  Cleared when the next LLM span opens.
        self._last_llm_span = span

        if self._capture_text and self._llm_text_buffer:
            span.attributes["llm.output"] = "".join(self._llm_text_buffer)
        self._llm_text_buffer.clear()

        # Write thought attribute lists.
        # B8: Gemini delivers thought signatures out-of-band via
        # LLMMessagesAppendFrame (collected in _pending_thought_signatures), NOT on
        # LLMThoughtEndFrame (which is bare, so _llm_thought_signatures_list holds
        # ""). When such signatures exist, emit a flat arrival-order list of the real
        # ones — no thought-block<->signature alignment is asserted (Gemini bookmarks
        # reference response parts, not thought blocks, and a signature can appear
        # with no thought block at all). Otherwise keep the per-block list aligned
        # with llm.thoughts (Anthropic sets frame.signature per block).
        if self._pending_thought_signatures:
            signatures = [s for s in self._llm_thought_signatures_list if s]
            signatures.extend(self._pending_thought_signatures)
        else:
            signatures = list(self._llm_thought_signatures_list)
        if self._llm_thoughts_list:
            span.attributes["llm.thoughts"] = list(self._llm_thoughts_list)
            span.attributes["llm.thought_signatures"] = signatures
        elif any(signatures):
            span.attributes["llm.thought_signatures"] = signatures
        self._llm_thoughts_list.clear()
        self._llm_thought_signatures_list.clear()
        self._pending_thought_signatures.clear()

        # Function calls/results are recorded on their owning span in the
        # function-call handlers; the per-tool-call state (pending / owner /
        # resolved) is cleared at turn boundaries (interruption / conversation
        # finish), NOT here — so a result arriving after this response ends (the
        # common case, and any InProgress→End→Result ordering) is never dropped.

        span.attributes["pipecat_span_status"] = "ok"
        span.finish()

    # ---------------------------------------------------------------------- #
    # LLM thought accumulation (flattened onto the LLM span)                 #
    # ---------------------------------------------------------------------- #

    async def _handle_llm_thought_start(self, _data: Any) -> None:
        """
        ``LLMThoughtStartFrame``: begin a new thought block.

        Clears the thought buffer so the next ``LLMThoughtTextFrame`` chunks
        accumulate cleanly. No child span is created; the completed thought is
        appended to ``llm.thoughts`` on the parent ``pipecat.llm`` span.
        """
        self._llm_thought_buffer.clear()

    async def _handle_llm_thought_text(self, data: Any) -> None:
        """``LLMThoughtTextFrame``: accumulate thought text chunks."""
        if not self._capture_text:
            return
        frame = data.frame
        text = getattr(frame, "text", None)
        if text:
            self._llm_thought_buffer.append(str(text))

    async def _handle_llm_thought_end(self, data: Any) -> None:
        """
        ``LLMThoughtEndFrame``: complete the current thought block.

        Appends accumulated text to ``_llm_thoughts_list`` and the frame's
        ``signature`` (used by Anthropic extended thinking) to
        ``_llm_thought_signatures_list``. Both lists are written to the
        ``pipecat.llm`` span as ``llm.thoughts`` / ``llm.thought_signatures``
        when ``LLMFullResponseEndFrame`` fires.
        """
        thought_text = "".join(self._llm_thought_buffer)
        self._llm_thought_buffer.clear()

        if not self._capture_text:
            return

        frame = data.frame
        sig = getattr(frame, "signature", None)

        self._llm_thoughts_list.append(thought_text)
        self._llm_thought_signatures_list.append(str(sig) if sig is not None else "")

    # ---------------------------------------------------------------------- #
    # Function call handlers                                                  #
    # ---------------------------------------------------------------------- #

    async def _handle_function_calls_started(self, data: Any) -> None:
        """
        ``FunctionCallsStartedFrame``: log the batch start at debug level.

        Individual calls are tracked via ``FunctionCallInProgressFrame``; this
        frame is informational only.
        """
        if not self._capture_function_calls:
            return
        frame = data.frame
        func_calls = getattr(frame, "function_calls", None) or []
        names = [getattr(fc, "function_name", "") for fc in func_calls]
        logger.debug("Function calls started: %s", names)

    def _owning_llm_span(self) -> Any:
        """The span a function call/result belongs to.

        Function-call frames arrive after the requesting response has ended, so the
        active span is usually ``None`` and the call belongs to the most recent
        (requesting) LLM span via the backref. Fall back to the active span when a
        result arrives while the span is still open.
        """
        return self._active_llm_span or self._last_llm_span

    def _function_call_target_span(self, tool_call_id: str) -> Any:
        """The span a tool result/cancel belongs to.

        Uses the span that REQUESTED the call (recorded at ``FunctionCallInProgress``
        time), so a straggler result still lands on the requesting span even after a
        follow-up LLM span has opened (the parallel-tool case). Falls back to the
        current owning span when the requester was not recorded.
        """
        return self._function_call_owner.get(tool_call_id) or self._owning_llm_span()

    async def _handle_function_call_start(self, data: Any) -> None:
        """
        ``FunctionCallInProgressFrame``: record the call on its owning ``pipecat.llm`` span.

        Writes the call to ``llm.function_calls`` on the owning span immediately, so
        it is captured regardless of whether a follow-up LLM response fires. Also
        stashed in ``_pending_function_calls`` so the matching result frame can be
        enriched with name/arguments. Dict keys: ``name``, ``tool_call_id``, ``arguments``.
        """
        if not self._capture_function_calls:
            return

        frame = data.frame
        fc_data = extract_function_call_data(frame)
        # tool_call_id is a required typed field on FunctionCallInProgressFrame;
        # fall back to "" so the result handler (which also uses "") can still match.
        tool_call_id = fc_data.get("tool_call_id", "")

        # Deduplicate: pipecat pushes FunctionCallInProgressFrame both upstream and
        # downstream so the observer sees it twice with the same tool_call_id.
        if tool_call_id in self._pending_function_calls:
            logger.debug(
                "FunctionCallInProgressFrame with duplicate tool_call_id=%r; skipping",
                tool_call_id,
            )
            return

        call_dict: dict[str, Any] = {"tool_call_id": tool_call_id}
        if fc_data.get("function_name"):
            call_dict["name"] = fc_data["function_name"]
        if fc_data.get("arguments"):
            call_dict["arguments"] = fc_data["arguments"]

        self._pending_function_calls[tool_call_id] = call_dict

        # Write the call to its owning LLM span immediately and remember that span,
        # so the matching result/cancel lands on the SAME (requesting) span even if
        # a follow-up LLM span opens first (parallel tools).
        span = self._owning_llm_span()
        if span is not None:
            self._function_call_owner[tool_call_id] = span
            _append_json_list_attr(span, "llm.function_calls", call_dict)

    def _build_result_identity(
        self, tool_call_id: str, fc_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Base dict for a result/cancel record: prefer the stashed call (has
        name/arguments), else reconstruct identity from the frame's own data so the
        record is never lost when the pending entry is already gone (e.g. a service
        emitting FunctionCallInProgress before LLMFullResponseEnd)."""
        call_dict = self._pending_function_calls.pop(tool_call_id, None)
        if call_dict is not None:
            return dict(call_dict)
        rebuilt: dict[str, Any] = {"tool_call_id": tool_call_id}
        if fc_data.get("function_name"):
            rebuilt["name"] = fc_data["function_name"]
        if fc_data.get("arguments"):
            rebuilt["arguments"] = fc_data["arguments"]
        return rebuilt

    async def _handle_function_call_result(self, data: Any) -> None:
        """
        ``FunctionCallResultFrame``: record the tool result on the requesting
        ``pipecat.llm`` span (JSON-encoded), immediately and independently of any
        follow-up LLM response or frame ordering.

        Result dict keys: ``name``, ``tool_call_id``, ``arguments``, ``result``,
        ``run_llm`` (when present).
        """
        if not self._capture_function_calls:
            return

        frame = data.frame
        fc_data = extract_function_call_data(frame)
        tool_call_id = fc_data.get("tool_call_id", "")

        # Dedup double-broadcast frames by tool_call_id (not by pop-returns-None,
        # which would also drop a genuine result whose pending entry was already
        # cleared).
        if tool_call_id in self._resolved_function_call_ids:
            return

        result_dict = self._build_result_identity(tool_call_id, fc_data)
        if "result" in fc_data:
            result_dict["result"] = fc_data["result"]
        if "run_llm" in fc_data:
            result_dict["run_llm"] = fc_data["run_llm"]

        self._resolved_function_call_ids.add(tool_call_id)
        span = self._function_call_target_span(tool_call_id)
        if span is not None:
            _append_json_list_attr(span, "llm.function_call_results", result_dict)

    async def _handle_function_call_cancel(self, data: Any) -> None:
        """
        ``FunctionCallCancelFrame``: record a cancelled tool call on the requesting
        ``pipecat.llm`` span. Same keys as the result handler plus ``cancelled: True``.
        """
        if not self._capture_function_calls:
            return

        frame = data.frame
        tool_call_id = getattr(frame, "tool_call_id", "")

        # Dedup double-broadcast cancel frames (same tool_call_id, distinct ids).
        if tool_call_id in self._resolved_function_call_ids:
            return

        fc_data = extract_function_call_data(frame)
        result_dict = self._build_result_identity(tool_call_id, fc_data)
        result_dict["cancelled"] = True

        self._resolved_function_call_ids.add(tool_call_id)
        span = self._function_call_target_span(tool_call_id)
        if span is not None:
            _append_json_list_attr(span, "llm.function_call_results", result_dict)

    # ---------------------------------------------------------------------- #
    # Context summarization                                                   #
    # ---------------------------------------------------------------------- #

    async def _handle_llm_summary_result(self, data: Any) -> None:
        """
        ``LLMContextSummaryResultFrame``: write summarization output to the active
        turn span (or trace if no turn is open).

        Attributes set: ``llm.summary.text``, ``llm.summary.request_id``,
        ``llm.summary.last_summarized_index``, ``llm.summary.error`` (if present).
        """
        frame = data.frame

        target = self._current_turn_span or self._trace
        if not target:
            return

        error = getattr(frame, "error", None)
        if error:
            target.attributes["llm.summary.error"] = str(error)
            return

        summary = getattr(frame, "summary", None)
        if summary:
            target.attributes["llm.summary.text"] = str(summary)
        req_id = getattr(frame, "request_id", None)
        if req_id is not None:
            target.attributes["llm.summary.request_id"] = str(req_id)
        last_idx = getattr(frame, "last_summarized_index", None)
        if last_idx is not None:
            target.attributes["llm.summary.last_summarized_index"] = int(last_idx)
