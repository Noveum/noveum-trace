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
  - FunctionCallInProgressFrame              — stash call dict in _pending_function_calls
  - FunctionCallResultFrame                  — move to _function_call_results list
  - FunctionCallCancelFrame                  — move to _function_call_results with cancelled=True
  - LLMContextSummaryResultFrame             — write summary to turn/trace

Thought blocks and function calls are stored as flat attribute lists on the
pipecat.llm span rather than as child spans:
  llm.thoughts                — list[str], one entry per thought block
  llm.thought_signatures      — list[str], one entry per thought block (may be "")
  llm.function_calls          — list[dict], one entry per FunctionCallInProgressFrame
  llm.function_call_results   — list[dict], one entry per result/cancel frame
"""

from __future__ import annotations

import logging
from typing import Any

from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
from noveum_trace.integrations.pipecat.pipecat_constants import (
    MAX_TEXT_BUFFER_LENGTH,
    SPAN_LLM,
)
from noveum_trace.integrations.pipecat.pipecat_utils import (
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
)


class _LLMHandlersMixin(_PipecatObserverMixinBase):
    """Handler methods for LLM response, thought, function-call, and summarization frames."""

    # State attributes declared in NoveumTraceObserver.__init__:
    #   _trace, _capture_text, _capture_function_calls,
    #   _llm_text_buffer, _active_llm_span, _current_turn_span,
    #   _pending_function_calls, _function_call_results,
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
        """``LLMMessagesAppendFrame``: append to stashed messages JSON."""
        frame = data.frame
        new_msgs = getattr(frame, "messages", None)
        if not new_msgs:
            return
        prev = self._pending_llm_context.get("messages")
        merged = merge_appended_messages_json(prev, new_msgs)
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
        - ``llm.function_calls`` — list of function-call dicts (name/tool_call_id/arguments)
        - ``llm.function_call_results`` — list of result/cancel dicts
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
            self._pending_function_calls.clear()
            self._function_call_results.clear()

            self._pre_span_function_call_ids.clear()
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

        # Write thought attribute lists
        if self._llm_thoughts_list:
            span.attributes["llm.thoughts"] = list(self._llm_thoughts_list)
            span.attributes["llm.thought_signatures"] = list(
                self._llm_thought_signatures_list
            )
        self._llm_thoughts_list.clear()
        self._llm_thought_signatures_list.clear()

        # Write function-call attribute lists.
        # Exclude pre-span IDs — those were already written directly to the previous
        # span's attributes in _handle_function_call_start.

        if self._pending_function_calls or self._function_call_results:
            all_calls = [
                v
                for k, v in self._pending_function_calls.items()
                if k not in self._pre_span_function_call_ids
            ]
            if all_calls:
                span.attributes["llm.function_calls"] = all_calls
            if self._function_call_results:
                span.attributes["llm.function_call_results"] = list(
                    self._function_call_results
                )
        self._pending_function_calls.clear()
        self._function_call_results.clear()
        self._pre_span_function_call_ids.clear()

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

    async def _handle_function_call_start(self, data: Any) -> None:
        """
        ``FunctionCallInProgressFrame``: stash call details in ``_pending_function_calls``.

        Dict keys: ``name``, ``tool_call_id``, ``arguments``.
        Written to ``llm.function_calls`` on the ``pipecat.llm`` span when
        ``LLMFullResponseEndFrame`` fires.
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

        # If span 1 has already closed but function-call frames arrive before span 2
        # opens, write the call directly to span 1 via the backref so it is not lost.

        if self._active_llm_span is None and self._last_llm_span is not None:
            existing = list(
                self._last_llm_span.attributes.get("llm.function_calls", [])
            )
            existing.append(call_dict)

            self._last_llm_span.attributes["llm.function_calls"] = existing
            self._pre_span_function_call_ids.add(tool_call_id)

    async def _handle_function_call_result(self, data: Any) -> None:
        """
        ``FunctionCallResultFrame``: move pending call to ``_function_call_results``.

        Result dict keys: ``name``, ``tool_call_id``, ``arguments``,
        ``result``, ``run_llm`` (when present).
        Written to ``llm.function_call_results`` on the ``pipecat.llm`` span
        when ``LLMFullResponseEndFrame`` fires.
        """
        if not self._capture_function_calls:
            return

        frame = data.frame
        fc_data = extract_function_call_data(frame)
        tool_call_id = fc_data.get("tool_call_id", "")

        # Use None sentinel so we can distinguish "not found" from an empty call dict.
        # Pipecat pushes FunctionCallResultFrame both upstream and downstream; the
        # observer sees it twice.  The second time the id is no longer in pending, so
        # call_dict is None and we silently drop the duplicate.
        call_dict = self._pending_function_calls.pop(tool_call_id, None)
        if call_dict is None:
            logger.debug(
                "FunctionCallResultFrame with no matching pending call "
                "(tool_call_id=%r); dropping duplicate",
                tool_call_id,
            )
            return

        result_dict: dict[str, Any] = {**call_dict}
        if "result" in fc_data:
            result_dict["result"] = fc_data["result"]
        if "run_llm" in fc_data:
            result_dict["run_llm"] = fc_data["run_llm"]

        self._function_call_results.append(result_dict)

    async def _handle_function_call_cancel(self, data: Any) -> None:
        """
        ``FunctionCallCancelFrame``: mark pending call as cancelled in results list.

        Result dict keys: same as ``_handle_function_call_result`` plus
        ``cancelled: True``.
        """
        if not self._capture_function_calls:
            return

        frame = data.frame
        tool_call_id = getattr(frame, "tool_call_id", "")
        call_dict = self._pending_function_calls.pop(tool_call_id, {})
        result_dict: dict[str, Any] = {**call_dict, "cancelled": True}
        self._function_call_results.append(result_dict)

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
