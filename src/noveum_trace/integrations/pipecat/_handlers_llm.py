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
import time
from typing import Any

from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.pipecat._llm_operation_registry import (
    LLMOperationAlreadyActiveError,
    LLMOperationRecord,
)
from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
from noveum_trace.integrations.pipecat._processor_registry import PROCESSOR_ROLE_LLM
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

    # Per-invocation content and correlation live in _llm_operations. Scalar
    # _active/_last LLM span fields remain compatibility aliases only.

    # ---------------------------------------------------------------------- #
    # Context frame (stash input + tools)                                     #
    # ---------------------------------------------------------------------- #

    def _llm_source(self, data: Any) -> Any:
        """Return and register the exact processor that emitted an LLM frame."""
        source = getattr(data, "source", None)
        if source is not None:
            self._processor_registry.set_explicit_role(source, PROCESSOR_ROLE_LLM)
            record = self._processor_registry.get(source)
            self._llm_operations.register_processor(
                source, record.name if record is not None else type(source).__name__
            )
        return source

    def _llm_destination(self, data: Any) -> Any:
        """Return the destination LLM for a context/control frame, when known."""
        destination = getattr(data, "destination", None)
        if destination is None:
            return None
        record = self._processor_registry.register(destination)
        if not record.has_role(PROCESSOR_ROLE_LLM):
            return None
        self._llm_operations.register_processor(destination, record.name)
        return destination

    def _resolve_llm_operation(
        self, data: Any, *, include_metrics_pending: bool = False
    ) -> Any:
        """Resolve an LLM frame to the exact emitting processor's operation."""
        source = getattr(data, "source", None)
        if source is not None:
            operation = (
                self._llm_operations.get_metrics_target(source)
                if include_metrics_pending
                else self._llm_operations.get_active(source)
            )
            if operation is None:
                return None
            frame_id = getattr(getattr(data, "frame", None), "id", None)
            if (
                operation.phase == "active"
                and operation.settled_predecessor_at_start
                and isinstance(frame_id, int)
                and isinstance(operation.start_frame_id, int)
                and frame_id < operation.start_frame_id
            ):
                return None
            return operation

        candidates = list(self._llm_operations.active_operations)
        if include_metrics_pending:
            candidates.extend(self._llm_operations.metrics_pending_operations)
        return candidates[0] if len(candidates) == 1 else None

    def _record_unmatched_llm_frame(self, data: Any) -> None:
        target = self._current_turn_span or self._trace
        if target is None:
            return
        frame_name = type(getattr(data, "frame", None)).__name__
        key = f"pipecat.unmatched_llm_frames.{frame_name}"
        target.attributes[key] = int(target.attributes.get(key, 0)) + 1

    def _merge_pending_llm(
        self, updates: dict[str, Any], destination: Any = None
    ) -> None:
        """Merge context into one destination or the broadcast fallback."""
        if not updates:
            return
        if destination is not None:
            self._llm_operations.merge_pending_input(destination, updates)
        else:
            merge_llm_pending_stash(self._pending_llm_context, updates)
            self._global_llm_context_generation += 1

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
            self._merge_pending_llm(extracted, self._llm_destination(data))
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Failed to handle LLM context frame: %s", e)

    async def _handle_llm_messages_replace(self, data: Any) -> None:
        """``LLMMessagesFrame`` / ``LLMMessagesUpdateFrame``: replace stashed messages."""
        frame = data.frame
        messages = getattr(frame, "messages", None)
        dumped = json_dumps_messages(messages)
        if messages is not None and dumped is None:
            dumped = "[]"
        if dumped is not None:
            self._merge_pending_llm({"messages": dumped}, self._llm_destination(data))

    async def _handle_llm_messages_append(self, data: Any) -> None:
        """``LLMMessagesAppendFrame``: append to stashed messages JSON.

        B8/B9: Gemini delivers thought signatures through this frame as
        ``{"type": "thought_signature", "signature": ...}`` messages. Those are
        attached to the exact emitting operation's ``thought_signatures`` and kept
        OUT of the stashed context — otherwise the opaque signature blob leaks
        into the next LLM span's ``llm.input``. A signature whose operation cannot
        be resolved has no correlation key and is dropped: parking it in shared
        state would hand it to whichever operation happens to end next.
        """
        frame = data.frame
        new_msgs = getattr(frame, "messages", None)
        if not new_msgs:
            return

        kept = []
        operation = self._resolve_llm_operation(data, include_metrics_pending=True)
        for m in new_msgs:
            sig = _thought_signature_of(m)
            if sig is not None:
                if sig:
                    if operation is not None:
                        try:
                            empty_index = operation.thought_signatures.index("")
                        except ValueError:
                            operation.thought_signatures.append(sig)
                        else:
                            operation.thought_signatures[empty_index] = sig
                    else:
                        logger.debug(
                            "Dropping thought signature with no resolvable LLM "
                            "operation (source=%r)",
                            getattr(data, "source", None),
                        )
                continue
            kept.append(m)
        if not kept:
            return

        destination = self._llm_destination(data)
        pending = (
            self._llm_operations.pending_input_for(destination)
            if destination is not None
            else self._pending_llm_context
        ) or {}
        prev = pending.get("messages")
        merged = merge_appended_messages_json(prev, kept)
        if merged:
            self._merge_pending_llm({"messages": merged}, destination)

    async def _handle_llm_set_tools(self, data: Any) -> None:
        """``LLMSetToolsFrame``: stash tool definitions JSON."""
        frame = data.frame
        tools = getattr(frame, "tools", None)
        dumped = serialize_tools_field(tools)
        if tools is not None and dumped is None:
            dumped = "[]"
        if dumped is not None:
            self._merge_pending_llm({"tools": dumped}, self._llm_destination(data))

    async def _handle_llm_set_tool_choice(self, data: Any) -> None:
        """``LLMSetToolChoiceFrame``: stash tool choice for ``llm.tool_choice``."""
        frame = data.frame
        choice = getattr(frame, "tool_choice", None)
        dumped = serialize_tool_choice_field(choice)
        if dumped:
            self._merge_pending_llm(
                {"tool_choice": dumped}, self._llm_destination(data)
            )

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

        attributes: dict[str, Any] = {}

        # Extract all available settings from the source processor
        source = self._llm_source(data)
        if source is None:
            self._record_unmatched_llm_frame(data)
            return
        settings = extract_service_settings(source)
        for settings_key, attr_key in _LLM_SETTINGS_MAP:
            val = settings.get(settings_key)
            if val is not None:
                attributes[attr_key] = val
        provider = derive_provider(source, settings.get("model"))
        if provider:
            attributes["llm.provider"] = provider

        existing = self._llm_operations.get_active(source)
        if existing is not None:
            logger.warning(
                "Unsupported overlapping LLM responses from processor %s; "
                "finalizing %s as incomplete",
                existing.processor_name,
                existing.operation_id,
            )
            self._finalize_llm_operation(
                existing,
                complete=False,
                termination_reason="same_source_overlap",
                terminal_status="cancelled",
            )

        span = self._create_child_span(
            SPAN_LLM,
            parent_span=self._current_turn_span,
            attributes=attributes,
        )
        if span is None:
            return

        record = self._processor_registry.get(source)
        try:
            operation = self._llm_operations.start(
                source,
                span=span,
                parent_span=self._current_turn_span,
                processor_name=record.name if record else type(source).__name__,
                provider=provider,
                model=attributes.get("llm.model"),
                started_at=time.monotonic(),
                start_frame_id=getattr(data.frame, "id", None),
            )
        except LLMOperationAlreadyActiveError:
            span.attributes["pipecat_span_status"] = "error"
            span.attributes["pipecat_span_status_message"] = (
                "same processor already had an active LLM operation"
            )
            span.set_status(
                SpanStatus.ERROR,
                "same processor already had an active LLM operation",
            )
            self._finish_managed_span(span)
            return

        # Source-less context frames are broadcasts. Each registered LLM consumes
        # each broadcast generation at most once, while destination-scoped input
        # belongs only to that exact processor.
        pending: dict[str, Any] = {}
        source_key = id(source)
        consumed_generation = self._global_llm_context_consumed.get(source_key, 0)
        if (
            self._pending_llm_context
            and consumed_generation < self._global_llm_context_generation
        ):
            pending.update(self._pending_llm_context)
            self._global_llm_context_consumed[source_key] = (
                self._global_llm_context_generation
            )
        if operation.pending_input_was_set and operation.pending_input is not None:
            pending.update(operation.pending_input)
        if "messages" in pending:
            span.attributes["llm.input"] = pending["messages"]
        if "tools" in pending:
            span.attributes["llm.tools"] = pending["tools"]
        if "tool_choice" in pending:
            span.attributes["llm.tool_choice"] = pending["tool_choice"]
        span.attributes["llm.operation_id"] = operation.operation_id

        self._active_llm_span = span
        self._last_llm_span = None

    async def _handle_llm_text(self, data: Any) -> None:
        """
        ``LLMTextFrame`` / ``VisionTextFrame``: accumulate assistant text chunks.

        The buffer is written to ``llm.output`` when the response ends.
        Guarded to prevent runaway memory usage.
        """
        if not self._capture_text:
            return
        operation = self._resolve_llm_operation(data)
        if operation is None:
            self._record_unmatched_llm_frame(data)
            return
        frame = data.frame
        text = getattr(frame, "text", None)
        if text:
            operation.output_chunks.append(str(text))
            while (
                sum(len(chunk) for chunk in operation.output_chunks)
                > MAX_TEXT_BUFFER_LENGTH
                and len(operation.output_chunks) > 1
            ):
                operation.output_chunks.pop(0)

    async def _handle_llm_response_end(self, data: Any) -> None:
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
        operation = self._resolve_llm_operation(data)
        if operation is None:
            return
        self._finalize_llm_operation(
            operation,
            complete=True,
            termination_reason="response_end",
            terminal_status="ok",
        )

    def _finalize_llm_operation(
        self,
        operation: LLMOperationRecord,
        *,
        complete: bool,
        termination_reason: str,
        terminal_status: str,
    ) -> None:
        """Flush one operation and finish its span exactly once."""
        if operation.phase != "active":
            return
        operation.finish_open_thought()
        existing_status = operation.span.attributes.get("pipecat_span_status")
        if operation.error or existing_status == "error":
            terminal_status = "error"

        completed = self._llm_operations.complete(
            operation.source_processor,
            logical_end_at=time.monotonic(),
            output_complete=complete,
            termination_reason=termination_reason,
            terminal_status=terminal_status,
        )
        if completed is None:
            return
        span = completed.span
        if self._capture_text and completed.output_chunks:
            span.attributes["llm.output"] = completed.output_text
        if completed.thoughts:
            span.attributes["llm.thoughts"] = list(completed.thoughts)
        if completed.thought_signatures:
            span.attributes["llm.thought_signatures"] = list(
                completed.thought_signatures
            )
        if completed.requested_function_calls:
            span.attributes["llm.function_calls"] = json.dumps(
                list(completed.requested_function_calls.values()), default=str
            )
        if completed.function_call_results:
            span.attributes["llm.function_call_results"] = json.dumps(
                completed.function_call_results, default=str
            )
        if completed.markers:
            span.attributes["llm.markers"] = list(completed.markers)
        span.attributes["llm.output.complete"] = complete
        span.attributes["llm.termination_reason"] = termination_reason
        span.attributes["pipecat_span_status"] = terminal_status
        if hasattr(span, "set_status"):
            if terminal_status == "error":
                span.set_status(SpanStatus.ERROR)
            elif terminal_status == "cancelled":
                # Native SDK status has no cancelled value; keep cancellation in
                # Pipecat attributes and upload an accepted native status.
                span.set_status(SpanStatus.OK)
        if getattr(span, "is_finished", lambda: False)() is not True:
            self._finish_managed_span(span)
        if self._active_llm_span is span:
            self._active_llm_span = None
        self._last_llm_span = span

    # ---------------------------------------------------------------------- #
    # LLM thought accumulation (flattened onto the LLM span)                 #
    # ---------------------------------------------------------------------- #

    async def _handle_llm_marker(self, data: Any) -> None:
        """Capture a Pipecat 1.x sideband marker without treating it as speech."""
        frame = data.frame
        marker = getattr(frame, "marker", None)
        if marker is None:
            return
        marker_data = {
            "marker": str(marker),
            "append_to_context_immediately": bool(
                getattr(frame, "append_to_context_immediately", True)
            ),
        }
        operation = self._resolve_llm_operation(data)
        if operation is not None:
            if len(operation.markers) < 100:
                operation.markers.append(marker_data)
            return
        target = self._current_turn_span
        if target is not None:
            markers = list(target.attributes.get("llm.markers", []))
            if len(markers) < 100:
                markers.append(marker_data)
                target.attributes["llm.markers"] = markers

    async def _handle_llm_thought_start(self, data: Any) -> None:
        """
        ``LLMThoughtStartFrame``: begin a new thought block.

        Clears the thought buffer so the next ``LLMThoughtTextFrame`` chunks
        accumulate cleanly. No child span is created; the completed thought is
        appended to ``llm.thoughts`` on the parent ``pipecat.llm`` span.
        """
        operation = self._resolve_llm_operation(data)
        if operation is not None:
            operation.thought_chunks.clear()

    async def _handle_llm_thought_text(self, data: Any) -> None:
        """``LLMThoughtTextFrame``: accumulate thought text chunks."""
        if not self._capture_text:
            return
        operation = self._resolve_llm_operation(data)
        if operation is None:
            self._record_unmatched_llm_frame(data)
            return
        frame = data.frame
        text = getattr(frame, "text", None)
        if text:
            operation.thought_chunks.append(str(text))

    async def _handle_llm_thought_end(self, data: Any) -> None:
        """
        ``LLMThoughtEndFrame``: complete the current thought block.

        Appends accumulated text to ``_llm_thoughts_list`` and the frame's
        ``signature`` (used by Anthropic extended thinking) to
        ``_llm_thought_signatures_list``. Both lists are written to the
        ``pipecat.llm`` span as ``llm.thoughts`` / ``llm.thought_signatures``
        when ``LLMFullResponseEndFrame`` fires.
        """
        if not self._capture_text:
            return
        operation = self._resolve_llm_operation(data)
        if operation is None:
            self._record_unmatched_llm_frame(data)
            return
        frame = data.frame
        sig = getattr(frame, "signature", None)
        operation.finish_open_thought(str(sig) if sig is not None else "")

    # ---------------------------------------------------------------------- #
    # Function call handlers                                                  #
    # ---------------------------------------------------------------------- #

    def _operation_for_tool_frame(self, data: Any, tool_call_id: str = "") -> Any:
        if tool_call_id:
            owner = self._function_call_owner.get(tool_call_id)
            if isinstance(owner, LLMOperationRecord):
                return owner
        operation = self._resolve_llm_operation(data, include_metrics_pending=True)
        if operation is not None:
            return operation
        if tool_call_id:
            candidates = [
                candidate
                for candidate in (
                    *self._llm_operations.active_operations,
                    *self._llm_operations.metrics_pending_operations,
                )
                if tool_call_id in candidate.requested_function_calls
            ]
            if len(candidates) == 1:
                return candidates[0]
        return None

    @staticmethod
    def _mint_tool_call_id(operation: LLMOperationRecord, ordinal: int) -> str:
        """Mint ``<operation>:tool-<n>`` and advance the per-operation counter.

        The counter only ever moves forward, so a later mint can never collide
        with an ID minted positionally from a FunctionCallsStartedFrame batch.
        """
        operation.synthesized_call_count = max(
            operation.synthesized_call_count, ordinal
        )
        return f"{operation.operation_id}:tool-{ordinal}"

    def _resolve_tool_call_id(
        self, operation: LLMOperationRecord, original_id: str
    ) -> str:
        """Return the operation-scoped ID for an in-progress call frame.

        Frames carrying an ID keep it. An ID-less frame resolves positionally to
        the next unclaimed ID minted by FunctionCallsStartedFrame; once that batch
        is consumed a fresh ID is minted so the two paths never diverge.
        """
        if original_id:
            return original_id
        if operation.unclaimed_synthesized_call_ids:
            return operation.unclaimed_synthesized_call_ids.pop(0)
        return self._mint_tool_call_id(operation, operation.synthesized_call_count + 1)

    @staticmethod
    def _uncorrelated_call_dict(
        tool_call_id: str, fc_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a result entry from the frame alone when no request was recorded."""
        call_dict: dict[str, Any] = {"tool_call_id": tool_call_id}
        if fc_data.get("function_name"):
            call_dict["name"] = fc_data["function_name"]
        if fc_data.get("arguments") is not None:
            call_dict["arguments"] = fc_data["arguments"]
        return call_dict

    @staticmethod
    def _write_function_state(operation: LLMOperationRecord) -> None:
        """Late-enrich a logically completed span with operation-owned tool state."""
        if operation.requested_function_calls:
            operation.span.attributes["llm.function_calls"] = json.dumps(
                list(operation.requested_function_calls.values()), default=str
            )
        if operation.function_call_results:
            operation.span.attributes["llm.function_call_results"] = json.dumps(
                operation.function_call_results, default=str
            )

    async def _handle_function_calls_started(self, data: Any) -> None:
        """Capture the ordered model-requested function-call batch."""
        if not self._capture_function_calls:
            return
        frame = data.frame
        func_calls = getattr(frame, "function_calls", None) or []
        operation = self._operation_for_tool_frame(data)
        if operation is None:
            self._record_unmatched_llm_frame(data)
            return
        for request_order, function_call in enumerate(func_calls):
            call_data = extract_function_call_data(function_call)
            original_id = call_data.get("tool_call_id", "")
            tool_call_id = original_id or self._mint_tool_call_id(
                operation, request_order + 1
            )
            if tool_call_id in operation.requested_function_calls:
                # Double-broadcast of the same batch: keep the first record and
                # do not queue the minted ID a second time.
                continue
            call_dict: dict[str, Any] = {
                "tool_call_id": tool_call_id,
                "request_order": request_order,
            }
            if call_data.get("function_name"):
                call_dict["name"] = call_data["function_name"]
            if call_data.get("arguments") is not None:
                call_dict["arguments"] = call_data["arguments"]
            if not original_id:
                call_dict["original_tool_call_id"] = ""
                operation.unclaimed_synthesized_call_ids.append(tool_call_id)
            operation.requested_function_calls[tool_call_id] = call_dict
            self._function_call_owner[tool_call_id] = operation
        if operation.phase != "active":
            self._write_function_state(operation)

    async def _handle_function_call_start(self, data: Any) -> None:
        """
        ``FunctionCallInProgressFrame``: attach call details to its operation.
        """
        if not self._capture_function_calls:
            return

        frame = data.frame
        fc_data = extract_function_call_data(frame)
        original_tool_call_id = fc_data.get("tool_call_id", "")
        operation = self._operation_for_tool_frame(data, original_tool_call_id)
        if operation is None:
            self._record_unmatched_llm_frame(data)
            return
        tool_call_id = self._resolve_tool_call_id(operation, original_tool_call_id)

        if tool_call_id in operation.requested_function_calls:
            existing = operation.requested_function_calls[tool_call_id]
            if fc_data.get("function_name"):
                existing["name"] = fc_data["function_name"]
            if fc_data.get("arguments") is not None:
                existing["arguments"] = fc_data["arguments"]
            return

        call_dict: dict[str, Any] = {"tool_call_id": tool_call_id}
        if fc_data.get("function_name"):
            call_dict["name"] = fc_data["function_name"]
        if fc_data.get("arguments"):
            call_dict["arguments"] = fc_data["arguments"]
        if not original_tool_call_id:
            call_dict["original_tool_call_id"] = ""
        operation.requested_function_calls[tool_call_id] = call_dict
        self._function_call_owner[tool_call_id] = operation
        if operation.phase != "active":
            self._write_function_state(operation)

    async def _handle_function_call_result(self, data: Any) -> None:
        """
        ``FunctionCallResultFrame``: attach a result to the owning operation.
        """
        if not self._capture_function_calls:
            return

        frame = data.frame
        fc_data = extract_function_call_data(frame)
        tool_call_id = fc_data.get("tool_call_id", "")
        operation = self._operation_for_tool_frame(data, tool_call_id)
        if operation is None:
            return
        call_dict = operation.requested_function_calls.get(tool_call_id)
        if call_dict is None:
            # No recorded request to correlate with (ID-less frame, or an ID the
            # request path never saw). Keep the result rather than dropping it;
            # it is built from the frame alone so it carries no request_order.
            logger.debug(
                "FunctionCallResultFrame with no recorded request "
                "(tool_call_id=%r); recording uncorrelated result",
                tool_call_id,
            )
            call_dict = self._uncorrelated_call_dict(tool_call_id, fc_data)
        result_dict: dict[str, Any] = {**call_dict}
        if "result" in fc_data:
            result_dict["result"] = fc_data["result"]
        if "run_llm" in fc_data:
            result_dict["run_llm"] = fc_data["run_llm"]
        if result_dict not in operation.function_call_results:
            operation.function_call_results.append(result_dict)
        if operation.phase != "active":
            self._write_function_state(operation)

    async def _handle_function_call_cancel(self, data: Any) -> None:
        """
        ``FunctionCallCancelFrame``: mark a call cancelled on its operation.
        """
        if not self._capture_function_calls:
            return

        frame = data.frame
        fc_data = extract_function_call_data(frame)
        tool_call_id = fc_data.get("tool_call_id", "")
        operation = self._operation_for_tool_frame(data, tool_call_id)
        if operation is None:
            return
        call_dict = operation.requested_function_calls.get(tool_call_id)
        if call_dict is None:
            # Same shape as an uncorrelated result: never a bare {"cancelled": True}.
            call_dict = self._uncorrelated_call_dict(tool_call_id, fc_data)
        result_dict: dict[str, Any] = {**call_dict, "cancelled": True}
        if result_dict not in operation.function_call_results:
            operation.function_call_results.append(result_dict)
        if operation.phase != "active":
            self._write_function_state(operation)

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
