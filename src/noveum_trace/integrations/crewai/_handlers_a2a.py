"""
Agent-to-Agent (A2A) delegation & communication event handler mixin for NoveumCrewAIListener.

Handles CrewAI A2A events across multiple span types:

  Delegation lifecycle:
  - ``on_a2a_delegation_started``    → open ``crewai.a2a.delegation`` span under
                                        current agent/task; capture delegating agent,
                                        receiving agent, task description, endpoint,
                                        context_id, protocol version
  - ``on_a2a_delegation_completed``  → close as SUCCESS; write result text,
                                        total turns, duration
  - ``on_a2a_delegation_failed``     → close as ERROR; attach exception details

  Conversation lifecycle:
  - ``on_a2a_conversation_started``  → open ``crewai.a2a.conversation`` span;
                                        capture participants, context_id
  - ``on_a2a_conversation_completed``→ close as SUCCESS; write final context
  - ``on_a2a_conversation_failed``   → close as ERROR

  Message-level events (no span lifecycle — annotate open conversation):
  - ``on_a2a_message_sent``          → append to message buffer
  - ``on_a2a_message_received``      → append to message buffer

  Streaming (raw chunks, separate from message dict buffer):
  - ``on_a2a_streaming_started``     → initialize ``_a2a_streaming_chunks`` for conversation
  - ``on_a2a_streaming_chunk``       → append string chunk to ``_a2a_streaming_chunks``
  - ``on_a2a_streaming_completed``   → flush chunks to ``a2a.streaming_content`` on the span

  Polling (status polling for async delegation):
  - ``on_a2a_polling_started``       → record polling start
  - ``on_a2a_polling_status``        → update status in span

  Other:
  - ``on_a2a_artifact_received``     → record received artifact metadata
  - ``on_a2a_server_task``           → MCP server task within A2A context
  - ``on_a2a_context_updated``       → record context changes
  - ``on_a2a_auth_failed``           → authentication error
  - ``on_a2a_connection_error``      → connection failure

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown, _agent_spans, _task_spans,
    _a2a_spans, _a2a_stream_buffers, _a2a_streaming_chunks, _a2a_streaming_lengths,
    _a2a_start_times
    (composite keys ``(context_id, "delegation")`` / ``(context_id, "conversation")``)
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_A2A_CONTEXT_ID,
    ATTR_A2A_DELEGATING_AGENT,
    ATTR_A2A_RECEIVING_AGENT,
    ATTR_A2A_RESULT,
    ATTR_A2A_STATUS,
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_STACKTRACE,
    ATTR_ERROR_TYPE,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    MAX_A2A_CONVERSATION_MESSAGES,
    MAX_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
    SPAN_A2A_CONVERSATION,
    SPAN_A2A_DELEGATION,
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

# Composite dict keys so delegation and conversation spans for the same context_id
# do not overwrite each other.
_A2A_SPAN_DELEGATION = "delegation"
_A2A_SPAN_CONVERSATION = "conversation"


def _a2a_entry_key(context_id: str, span_type: str) -> tuple[str, str]:
    return (context_id, span_type)


class _A2AHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI Agent-to-Agent delegation & communication events.

    All public methods match the ``BaseEventListener`` callback signature::

        def on_a2a_delegation_started(self, source, event): ...

    ``source`` is typically the delegating Agent; ``event`` carries event payload.
    Every method is fully exception-shielded.
    """

    # =========================================================================
    # DELEGATION — started / completed / failed
    # =========================================================================

    def on_a2a_delegation_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.a2a.delegation`` span for agent-to-agent delegation.

        Attributes set at span open
        ---------------------------
        - ``a2a.context_id``           — unique delegation/conversation context
        - ``a2a.delegating_agent``     — role of delegating agent
        - ``a2a.receiving_agent``      — role of receiving agent
        - ``a2a.endpoint``             — A2A server endpoint (URL or identifier)
        - ``a2a.task_description``     — delegated task description (truncated)
        - ``a2a.protocol_version``     — A2A protocol version (e.g., "1.0")
        - ``a2a.provider_info``        — provider/server identifier
        - ``a2a.skill_id``             — skill being delegated (if applicable)
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)
            agent_id = _resolve_agent_id(source, event)
            task_id = _resolve_task_id(source, event)

            attrs = _build_a2a_delegation_start_attributes(source, event, context_id)
            start_t = monotonic_now()

            # Parent: agent span or task span (most common in delegation)
            parent_span = self._get_agent_or_task_span(agent_id, task_id)

            span = self._create_child_span(
                SPAN_A2A_DELEGATION,
                parent_span=parent_span,
                attributes=attrs,
            )

            with self._lock:
                dk = _a2a_entry_key(context_id, _A2A_SPAN_DELEGATION)
                self._a2a_spans[dk] = {
                    "span": span,
                    "type": "delegation",
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "start_t": start_t,
                }
                self._a2a_start_times[dk] = start_t

            logger.debug(
                "A2A delegation span opened: context_id=%s delegating=%s receiving=%s",
                context_id,
                attrs.get(ATTR_A2A_DELEGATING_AGENT, "?"),
                attrs.get(ATTR_A2A_RECEIVING_AGENT, "?"),
            )

        except Exception:
            logger.debug("on_a2a_delegation_started error:\n%s", traceback.format_exc())

    def on_a2a_delegation_completed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.a2a.delegation`` span as SUCCESS.

        Attributes written
        ------------------
        - ``a2a.status``         — ``"completed"`` or ``"input_required"``
        - ``a2a.result``         — delegated task result (truncated)
        - ``a2a.total_turns``    — total conversation turns/iterations
        - ``a2a.duration_ms``    — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)
            status = safe_getattr(event, "status") or ATTR_STATUS_SUCCESS

            result = safe_getattr(event, "result")
            total_turns = safe_getattr(event, "total_turns")
            extra: dict[str, Any] = {}

            if result:
                extra[ATTR_A2A_RESULT] = truncate_str(str(result), MAX_TEXT_LENGTH)
            if total_turns is not None:
                try:
                    extra["a2a.total_turns"] = int(total_turns)
                except (TypeError, ValueError):
                    pass

            self._finish_a2a_span(
                context_id=context_id,
                span_type=_A2A_SPAN_DELEGATION,
                status=status,
                error=None,
                extra_attrs=extra,
            )
        except Exception:
            logger.debug(
                "on_a2a_delegation_completed error:\n%s", traceback.format_exc()
            )

    def on_a2a_delegation_failed(self, source: Any, event: Any) -> None:
        """
        Close the ``crewai.a2a.delegation`` span as ERROR.

        Attributes written
        ------------------
        - ``error.type``        — exception class name
        - ``error.message``     — exception message
        - ``error.stacktrace``  — formatted traceback
        - ``a2a.status``        — ``"failed"``
        - ``a2a.duration_ms``   — wall-clock duration
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_a2a_span(
                context_id=context_id,
                span_type=_A2A_SPAN_DELEGATION,
                status=ATTR_STATUS_ERROR,
                error=error,
            )
        except Exception:
            logger.debug("on_a2a_delegation_failed error:\n%s", traceback.format_exc())

    # =========================================================================
    # CONVERSATION — started / completed / failed
    # =========================================================================

    def on_a2a_conversation_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.a2a.conversation`` span for multi-turn A2A conversation.

        Attributes set at span open
        ---------------------------
        - ``a2a.context_id``     — conversation context ID
        - ``a2a.delegating_agent`` — initiating agent role
        - ``a2a.receiving_agent``  — target/receiving agent role
        - ``a2a.endpoint``        — server endpoint
        - ``a2a.protocol_version``— protocol version in use
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)
            agent_id = _resolve_agent_id(source, event)
            task_id = _resolve_task_id(source, event)

            attrs = _build_a2a_conversation_start_attributes(source, event, context_id)
            start_t = monotonic_now()

            parent_span = self._get_agent_or_task_span(agent_id, task_id)

            span = self._create_child_span(
                SPAN_A2A_CONVERSATION,
                parent_span=parent_span,
                attributes=attrs,
            )

            with self._lock:
                ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
                self._a2a_spans[ck] = {
                    "span": span,
                    "type": "conversation",
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "start_t": start_t,
                }
                self._a2a_stream_buffers.setdefault(ck, [])
                self._a2a_streaming_chunks.setdefault(ck, [])
                self._a2a_streaming_lengths.setdefault(ck, 0)
                self._a2a_start_times[ck] = start_t

            logger.debug("A2A conversation span opened: context_id=%s", context_id)

        except Exception:
            logger.debug(
                "on_a2a_conversation_started error:\n%s", traceback.format_exc()
            )

    def on_a2a_conversation_completed(self, source: Any, event: Any) -> None:
        """Close the conversation span as SUCCESS."""
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)
            total_turns = safe_getattr(event, "total_turns")
            extra: dict[str, Any] = {}

            if total_turns is not None:
                try:
                    extra["a2a.total_turns"] = int(total_turns)
                except (TypeError, ValueError):
                    pass

            self._finish_a2a_span(
                context_id=context_id,
                span_type=_A2A_SPAN_CONVERSATION,
                status=ATTR_STATUS_SUCCESS,
                error=None,
                extra_attrs=extra,
            )
        except Exception:
            logger.debug(
                "on_a2a_conversation_completed error:\n%s", traceback.format_exc()
            )

    def on_a2a_conversation_failed(self, source: Any, event: Any) -> None:
        """Close the conversation span as ERROR."""
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            self._finish_a2a_span(
                context_id=context_id,
                span_type=_A2A_SPAN_CONVERSATION,
                status=ATTR_STATUS_ERROR,
                error=error,
            )
        except Exception:
            logger.debug(
                "on_a2a_conversation_failed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # MESSAGES — sent / received (no span lifecycle — buffer for conversation)
    # =========================================================================

    def on_a2a_message_sent(self, source: Any, event: Any) -> None:
        """
        Append a sent message to the conversation buffer.

        Attributes recorded (buffered)
        ------------------------------
        - ``a2a.turn_number``      — turn in the conversation
        - ``a2a.message_content``  — message text (truncated)
        - ``a2a.message_sender``   — agent role sending the message
        - ``a2a.message_type``     — message type (e.g., "query", "response")
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)
            turn_number = safe_getattr(event, "turn_number")
            message = safe_getattr(event, "message") or safe_getattr(event, "content")
            sender = safe_getattr(event, "sender") or safe_getattr(event, "agent_role")
            message_type = safe_getattr(event, "message_type")
            if message_type is None:
                message_type = safe_getattr(event, "msg_type")

            msg_entry = {
                "type": "sent",
                "turn_number": turn_number,
                "content": (
                    truncate_str(str(message), MAX_TEXT_LENGTH) if message else ""
                ),
                "sender": str(sender) if sender else "unknown",
            }
            if message_type is not None:
                msg_entry["message_type"] = truncate_str(
                    str(message_type), MAX_TEXT_LENGTH
                )

            ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
            with self._lock:
                buf = self._a2a_stream_buffers.get(ck, [])
                if len(buf) < MAX_A2A_CONVERSATION_MESSAGES:
                    buf.append(msg_entry)
                    self._a2a_stream_buffers[ck] = buf

            logger.debug(
                "A2A message sent: context_id=%s turn=%s",
                context_id,
                turn_number,
            )

        except Exception:
            logger.debug("on_a2a_message_sent error:\n%s", traceback.format_exc())

    def on_a2a_message_received(self, source: Any, event: Any) -> None:
        """
        Append a received message to the conversation buffer.

        Attributes recorded (buffered)
        ------------------------------
        - ``a2a.turn_number``      — turn in the conversation
        - ``a2a.message_content``  — message text (truncated)
        - ``a2a.message_sender``   — agent role sending the message
        - ``a2a.message_type``     — message type (e.g., "query", "response")
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)
            turn_number = safe_getattr(event, "turn_number")
            message = safe_getattr(event, "message") or safe_getattr(event, "content")
            sender = safe_getattr(event, "sender") or safe_getattr(
                event, "from_agent_role"
            )
            message_type = safe_getattr(event, "message_type")
            if message_type is None:
                message_type = safe_getattr(event, "msg_type")

            msg_entry = {
                "type": "received",
                "turn_number": turn_number,
                "content": (
                    truncate_str(str(message), MAX_TEXT_LENGTH) if message else ""
                ),
                "sender": str(sender) if sender else "unknown",
            }
            if message_type is not None:
                msg_entry["message_type"] = truncate_str(
                    str(message_type), MAX_TEXT_LENGTH
                )

            ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
            with self._lock:
                buf = self._a2a_stream_buffers.get(ck, [])
                if len(buf) < MAX_A2A_CONVERSATION_MESSAGES:
                    buf.append(msg_entry)
                    self._a2a_stream_buffers[ck] = buf

            logger.debug(
                "A2A message received: context_id=%s turn=%s",
                context_id,
                turn_number,
            )

        except Exception:
            logger.debug("on_a2a_message_received error:\n%s", traceback.format_exc())

    # =========================================================================
    # STREAMING — started / chunk / completed
    # =========================================================================

    def on_a2a_streaming_started(self, source: Any, event: Any) -> None:
        """Initialize raw streaming chunk buffer (separate from message dict buffer)."""
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)
            ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
            with self._lock:
                if ck not in self._a2a_streaming_chunks:
                    self._a2a_streaming_chunks[ck] = []
                    self._a2a_streaming_lengths[ck] = 0
            logger.debug("A2A streaming started: context_id=%s", context_id)
        except Exception:
            logger.debug("on_a2a_streaming_started error:\n%s", traceback.format_exc())

    def on_a2a_streaming_chunk(self, source: Any, event: Any) -> None:
        """
        Append a raw text chunk to ``_a2a_streaming_chunks`` (not the message buffer).

        Attributes captured
        -------------------
        - ``a2a.chunk_text``      — text chunk
        - ``a2a.is_final_chunk``  — bool: is this the final chunk?
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)
            chunk = (
                safe_getattr(event, "chunk")
                or safe_getattr(event, "text")
                or safe_getattr(event, "delta")
                or ""
            )
            is_final = safe_getattr(event, "is_final") or False
            chunk_str = str(chunk)

            ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
            with self._lock:
                chunks = self._a2a_streaming_chunks.setdefault(ck, [])
                current_len = self._a2a_streaming_lengths.setdefault(ck, 0)
                add_len = len(chunk_str)
                if current_len + add_len <= MAX_TEXT_LENGTH:
                    chunks.append(chunk_str)
                    self._a2a_streaming_lengths[ck] = current_len + add_len

            if is_final:
                logger.debug("A2A streaming final chunk: context_id=%s", context_id)

        except Exception:
            logger.debug("on_a2a_streaming_chunk error:\n%s", traceback.format_exc())

    def on_a2a_streaming_completed(self, source: Any, event: Any) -> None:
        """Flush raw streaming chunks to the span (message buffer is left intact)."""
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)

            ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
            with self._lock:
                raw_chunks = self._a2a_streaming_chunks.pop(ck, [])
                self._a2a_streaming_lengths.pop(ck, None)
                span_entry = self._a2a_spans.get(ck)

            if not span_entry:
                return

            span = span_entry.get("span")
            if not span or not raw_chunks:
                return

            content = truncate_str("".join(raw_chunks), MAX_TEXT_LENGTH)

            try:
                span.set_attribute("a2a.streaming_content", content)
            except Exception:
                # Fallback: direct dict write (post-close)
                if hasattr(span, "attributes"):
                    span.attributes["a2a.streaming_content"] = content

            logger.debug(
                "A2A streaming completed: context_id=%s content_len=%d",
                context_id,
                len(content),
            )

        except Exception:
            logger.debug(
                "on_a2a_streaming_completed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # POLLING — started / status (async delegation tracking)
    # =========================================================================

    def on_a2a_polling_started(self, source: Any, event: Any) -> None:
        """Record polling start for async delegation."""
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)

            dk = _a2a_entry_key(context_id, _A2A_SPAN_DELEGATION)
            with self._lock:
                span_entry = self._a2a_spans.get(dk)

            if not span_entry:
                return

            span = span_entry.get("span")
            if span:
                try:
                    span.set_attribute("a2a.polling_started", True)
                    interval = safe_getattr(event, "polling_interval")
                    if interval:
                        span.set_attribute("a2a.polling_interval_ms", int(interval))
                except Exception:
                    pass

            logger.debug("A2A polling started: context_id=%s", context_id)

        except Exception:
            logger.debug("on_a2a_polling_started error:\n%s", traceback.format_exc())

    def on_a2a_polling_status(self, source: Any, event: Any) -> None:
        """
        Update polling status in the span.

        Attributes written
        ------------------
        - ``a2a.polling_status``    — ``"pending"`` | ``"ready"`` | ``"failed"``
        - ``a2a.polling_attempt``   — attempt number
        - ``a2a.polling_message``   — status message (truncated)
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)

            dk = _a2a_entry_key(context_id, _A2A_SPAN_DELEGATION)
            with self._lock:
                span_entry = self._a2a_spans.get(dk)

            if not span_entry:
                return

            span = span_entry.get("span")
            if not span:
                return

            status = safe_getattr(event, "status")
            attempt = safe_getattr(event, "attempt")
            message = safe_getattr(event, "message")

            attrs: dict[str, Any] = {}
            if status:
                attrs["a2a.polling_status"] = str(status)
            if attempt is not None:
                try:
                    attrs["a2a.polling_attempt"] = int(attempt)
                except (TypeError, ValueError):
                    pass
            if message:
                attrs["a2a.polling_message"] = truncate_str(
                    str(message), MAX_DESCRIPTION_LENGTH
                )

            try:
                if hasattr(span, "set_attributes"):
                    span.set_attributes(attrs)
                else:
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
            except Exception:
                # Post-close fallback
                if hasattr(span, "attributes"):
                    span.attributes.update(attrs)

        except Exception:
            logger.debug("on_a2a_polling_status error:\n%s", traceback.format_exc())

    # =========================================================================
    # ARTIFACTS — received
    # =========================================================================

    def on_a2a_artifact_received(self, source: Any, event: Any) -> None:
        """
        Record metadata of artifact received from delegation.

        Attributes written
        ------------------
        - ``a2a.artifact_name``      — artifact name/filename
        - ``a2a.artifact_mime_type`` — MIME type (e.g., "application/json")
        - ``a2a.artifact_size_bytes``— size in bytes
        - ``a2a.artifact_id``        — unique artifact identifier
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)

            dk = _a2a_entry_key(context_id, _A2A_SPAN_DELEGATION)
            with self._lock:
                span_entry = self._a2a_spans.get(dk)

            if not span_entry:
                return

            span = span_entry.get("span")
            if not span:
                return

            artifact_name = safe_getattr(event, "artifact_name") or safe_getattr(
                event, "name"
            )
            mime_type = safe_getattr(event, "mime_type") or safe_getattr(
                event, "content_type"
            )
            size_bytes = safe_getattr(event, "size_bytes") or safe_getattr(
                event, "size"
            )
            artifact_id = safe_getattr(event, "artifact_id") or safe_getattr(
                event, "id"
            )

            attrs: dict[str, Any] = {}
            if artifact_name:
                attrs["a2a.artifact_name"] = str(artifact_name)
            if mime_type:
                attrs["a2a.artifact_mime_type"] = str(mime_type)
            if size_bytes is not None:
                try:
                    attrs["a2a.artifact_size_bytes"] = int(size_bytes)
                except (TypeError, ValueError):
                    pass
            if artifact_id:
                attrs["a2a.artifact_id"] = str(artifact_id)

            try:
                if hasattr(span, "set_attributes"):
                    span.set_attributes(attrs)
                else:
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
            except Exception:
                if hasattr(span, "attributes"):
                    span.attributes.update(attrs)

            logger.debug(
                "A2A artifact received: context_id=%s name=%s size=%s",
                context_id,
                artifact_name,
                size_bytes,
            )

        except Exception:
            logger.debug("on_a2a_artifact_received error:\n%s", traceback.format_exc())

    # =========================================================================
    # SERVER TASK — MCP server-side task within A2A context
    # =========================================================================

    def on_a2a_server_task(self, source: Any, event: Any) -> None:
        """
        Record an MCP server task executed as part of A2A delegation.

        Attributes written
        ------------------
        - ``a2a.server_task_name`` — task name
        - ``a2a.server_task_input``— input JSON (truncated)
        - ``a2a.server_task_result`` — result (truncated)
        - ``a2a.server_task_status`` — "success" or "failed"
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)

            dk = _a2a_entry_key(context_id, _A2A_SPAN_DELEGATION)
            with self._lock:
                span_entry = self._a2a_spans.get(dk)

            if not span_entry:
                return

            span = span_entry.get("span")
            if not span:
                return

            task_name = safe_getattr(event, "task_name") or safe_getattr(event, "name")
            task_input = safe_getattr(event, "input")
            task_result = safe_getattr(event, "result") or safe_getattr(event, "output")
            task_status = safe_getattr(event, "status") or "unknown"

            attrs: dict[str, Any] = {}
            if task_name:
                attrs["a2a.server_task_name"] = str(task_name)
            if task_input:
                attrs["a2a.server_task_input"] = truncate_str(
                    safe_json_dumps(task_input), MAX_DESCRIPTION_LENGTH
                )
            if task_result:
                attrs["a2a.server_task_result"] = truncate_str(
                    safe_json_dumps(task_result), MAX_TEXT_LENGTH
                )
            attrs["a2a.server_task_status"] = str(task_status)

            try:
                if hasattr(span, "set_attributes"):
                    span.set_attributes(attrs)
                else:
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
            except Exception:
                if hasattr(span, "attributes"):
                    span.attributes.update(attrs)

        except Exception:
            logger.debug("on_a2a_server_task error:\n%s", traceback.format_exc())

    # =========================================================================
    # CONTEXT — updated / shared
    # =========================================================================

    def on_a2a_context_updated(self, source: Any, event: Any) -> None:
        """Record context state changes during A2A conversation."""
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)

            ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
            with self._lock:
                span_entry = self._a2a_spans.get(ck)

            if not span_entry:
                return

            span = span_entry.get("span")
            if not span:
                return

            context_update = safe_getattr(event, "context") or safe_getattr(
                event, "update"
            )
            reason = safe_getattr(event, "reason")

            attrs: dict[str, Any] = {}
            if context_update:
                attrs["a2a.context_update"] = truncate_str(
                    safe_json_dumps(context_update), MAX_TEXT_LENGTH
                )
            if reason:
                attrs["a2a.context_update_reason"] = str(reason)

            try:
                if hasattr(span, "set_attributes"):
                    span.set_attributes(attrs)
                else:
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
            except Exception:
                if hasattr(span, "attributes"):
                    span.attributes.update(attrs)

        except Exception:
            logger.debug("on_a2a_context_updated error:\n%s", traceback.format_exc())

    def on_a2a_context_shared(self, source: Any, event: Any) -> None:
        """Record context shared between agents."""
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)

            ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
            with self._lock:
                span_entry = self._a2a_spans.get(ck)

            if not span_entry:
                return

            span = span_entry.get("span")
            if not span:
                return

            shared_with = safe_getattr(event, "shared_with")
            shared_data = safe_getattr(event, "data")

            attrs: dict[str, Any] = {}
            if shared_with:
                attrs["a2a.context_shared_with"] = str(shared_with)
            if shared_data:
                attrs["a2a.context_shared_data_size"] = len(
                    safe_json_dumps(shared_data)
                )

            try:
                if hasattr(span, "set_attributes"):
                    span.set_attributes(attrs)
                else:
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
            except Exception:
                if hasattr(span, "attributes"):
                    span.attributes.update(attrs)

        except Exception:
            logger.debug("on_a2a_context_shared error:\n%s", traceback.format_exc())

    # =========================================================================
    # ERROR HANDLING — authentication / connection
    # =========================================================================

    def on_a2a_auth_failed(self, source: Any, event: Any) -> None:
        """
        Record authentication failure in A2A context.

        Attributes written
        ------------------
        - ``a2a.auth_error``       — error message
        - ``a2a.auth_method``      — auth method attempted
        - ``a2a.auth_server``      — server where auth failed
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)

            dk = _a2a_entry_key(context_id, _A2A_SPAN_DELEGATION)
            ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
            with self._lock:
                span_entry = self._a2a_spans.get(dk) or self._a2a_spans.get(ck)

            if not span_entry:
                return

            span = span_entry.get("span")
            if not span:
                return

            error_msg = safe_getattr(event, "error") or safe_getattr(event, "message")
            auth_method = safe_getattr(event, "auth_method") or safe_getattr(
                event, "method"
            )
            auth_server = safe_getattr(event, "server") or safe_getattr(
                event, "endpoint"
            )

            attrs: dict[str, Any] = {}
            if error_msg:
                attrs["a2a.auth_error"] = truncate_str(
                    str(error_msg), MAX_DESCRIPTION_LENGTH
                )
            if auth_method:
                attrs["a2a.auth_method"] = str(auth_method)
            if auth_server:
                attrs["a2a.auth_server"] = str(auth_server)

            attrs[ATTR_ERROR_TYPE] = "AuthenticationError"

            try:
                if hasattr(span, "set_attributes"):
                    span.set_attributes(attrs)
                else:
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
            except Exception:
                if hasattr(span, "attributes"):
                    span.attributes.update(attrs)

            logger.debug(
                "A2A auth failed: context_id=%s method=%s",
                context_id,
                auth_method,
            )

        except Exception:
            logger.debug("on_a2a_auth_failed error:\n%s", traceback.format_exc())

    def on_a2a_connection_error(self, source: Any, event: Any) -> None:
        """
        Record connection error in A2A communication.

        Attributes written
        ------------------
        - ``error.type``        — "ConnectionError"
        - ``error.message``     — connection error message
        - ``a2a.endpoint``      — endpoint where connection failed
        - ``a2a.connection_attempt`` — retry attempt number
        """
        if not self._is_active():
            return
        try:
            context_id = _resolve_context_id(event, source)

            dk = _a2a_entry_key(context_id, _A2A_SPAN_DELEGATION)
            ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
            with self._lock:
                span_entry = self._a2a_spans.get(dk) or self._a2a_spans.get(ck)

            if not span_entry:
                return

            span = span_entry.get("span")
            if not span:
                return

            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            endpoint = safe_getattr(event, "endpoint")
            attempt = safe_getattr(event, "attempt")

            attrs: dict[str, Any] = {
                ATTR_ERROR_TYPE: "ConnectionError",
            }

            if error:
                attrs[ATTR_ERROR_MESSAGE] = truncate_str(
                    str(error), MAX_DESCRIPTION_LENGTH
                )

            if endpoint:
                attrs["a2a.endpoint"] = str(endpoint)

            if attempt is not None:
                try:
                    attrs["a2a.connection_attempt"] = int(attempt)
                except (TypeError, ValueError):
                    pass

            try:
                if hasattr(span, "set_attributes"):
                    span.set_attributes(attrs)
                else:
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
            except Exception:
                if hasattr(span, "attributes"):
                    span.attributes.update(attrs)

            logger.debug(
                "A2A connection error: context_id=%s endpoint=%s attempt=%s",
                context_id,
                endpoint,
                attempt,
            )

        except Exception:
            logger.debug("on_a2a_connection_error error:\n%s", traceback.format_exc())

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _finish_a2a_span(
        self,
        context_id: str,
        span_type: str,
        status: str,
        error: Optional[Any] = None,
        extra_attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Close an A2A span with given status.

        Args:
            context_id:  Unique context ID
            span_type:   ``"delegation"`` or ``"conversation"`` (composite dict key)
            status:      "success" or "error"
            error:       Exception object (if error status)
            extra_attrs: Additional attributes to write
        """
        key = _a2a_entry_key(context_id, span_type)
        with self._lock:
            entry = self._a2a_spans.pop(key, None)
            start_t = self._a2a_start_times.pop(key, None)
            buf = self._a2a_stream_buffers.pop(key, None)
            stream_chunks = self._a2a_streaming_chunks.pop(key, None)
            self._a2a_streaming_lengths.pop(key, None)

        if entry is None:
            logger.debug(
                "_finish_a2a_span: no open entry for context_id=%s span_type=%s",
                context_id,
                span_type,
            )
            return

        span = entry.get("span")
        if span is None:
            return

        attrs: dict[str, Any] = {ATTR_A2A_STATUS: status}

        # Duration
        if start_t is not None:
            attrs["a2a.duration_ms"] = duration_ms_monotonic(start_t)

        # Merge extra attributes
        if extra_attrs:
            attrs.update(extra_attrs)

        # Message dicts (on_a2a_message_*) and any leftover raw chunks (if streaming
        # completed before span close did not flush) — buffers cleared under the lock above.
        if buf:
            attrs["a2a.messages"] = truncate_str(
                safe_json_dumps(buf),
                MAX_TEXT_LENGTH,
            )
        if stream_chunks:
            joined = "".join(stream_chunks)
            if joined:
                attrs["a2a.streaming_content"] = truncate_str(
                    joined,
                    MAX_TEXT_LENGTH,
                )

        # Error details
        if status == ATTR_STATUS_ERROR and error:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = truncate_str(str(error), MAX_DESCRIPTION_LENGTH)
            if hasattr(error, "__traceback__"):
                import traceback as tb_module

                attrs[ATTR_ERROR_STACKTRACE] = truncate_str(
                    "".join(tb_module.format_tb(error.__traceback__)),
                    MAX_TEXT_LENGTH,
                )

        # Write attributes
        try:
            if hasattr(span, "set_attributes"):
                span.set_attributes(attrs)
            else:
                for k, v in attrs.items():
                    span.set_attribute(k, v)
        except Exception:
            # Post-close fallback
            if hasattr(span, "attributes"):
                span.attributes.update(attrs)

        # Finish span
        try:
            if hasattr(span, "finish"):
                span.finish()
            elif hasattr(span, "end"):
                span.end()
        except Exception:
            pass

        logger.debug(
            "A2A span finished: context_id=%s span_type=%s status=%s duration_ms=%s",
            context_id,
            span_type,
            status,
            attrs.get("a2a.duration_ms", "?"),
        )


# ============================================================================
# Helper functions
# ============================================================================


def _resolve_context_id(event: Any, source: Any) -> str:
    """Extract or generate unique context ID for A2A interaction."""
    context_id = (
        safe_getattr(event, "context_id")
        or safe_getattr(event, "conversation_id")
        or safe_getattr(event, "delegation_id")
        or safe_getattr(source, "context_id")
    )
    return str(context_id) if context_id else f"a2a_{id(event)}"


def _resolve_agent_id(source: Any, event: Any) -> Optional[str]:
    """Extract agent ID from delegating agent."""
    agent_id = (
        safe_getattr(source, "id")
        or safe_getattr(event, "agent_id")
        or safe_getattr(event, "delegating_agent_id")
    )
    return str(agent_id) if agent_id else None


def _resolve_task_id(source: Any, event: Any) -> Optional[str]:
    """Extract task ID from context."""
    task_id = safe_getattr(event, "task_id") or safe_getattr(source, "task_id")
    return str(task_id) if task_id else None


def _build_a2a_delegation_start_attributes(
    source: Any, event: Any, context_id: str
) -> dict[str, Any]:
    """Build attributes for delegation start."""
    attrs: dict[str, Any] = {
        ATTR_A2A_CONTEXT_ID: context_id,
    }

    delegating = safe_getattr(event, "delegating_agent") or safe_getattr(source, "role")
    if delegating:
        attrs[ATTR_A2A_DELEGATING_AGENT] = str(delegating)

    receiving = safe_getattr(event, "receiving_agent") or safe_getattr(
        event, "target_agent"
    )
    if receiving:
        attrs[ATTR_A2A_RECEIVING_AGENT] = str(receiving)

    endpoint = safe_getattr(event, "endpoint") or safe_getattr(event, "server")
    if endpoint:
        attrs["a2a.endpoint"] = str(endpoint)

    task_desc = safe_getattr(event, "task_description") or safe_getattr(event, "task")
    if task_desc:
        attrs["a2a.task_description"] = truncate_str(
            str(task_desc), MAX_DESCRIPTION_LENGTH
        )

    protocol_version = safe_getattr(event, "protocol_version") or safe_getattr(
        event, "version"
    )
    if protocol_version:
        attrs["a2a.protocol_version"] = str(protocol_version)

    provider_info = safe_getattr(event, "provider_info") or safe_getattr(
        event, "provider"
    )
    if provider_info:
        attrs["a2a.provider_info"] = str(provider_info)

    skill_id = safe_getattr(event, "skill_id")
    if skill_id:
        attrs["a2a.skill_id"] = str(skill_id)

    return attrs


def _build_a2a_conversation_start_attributes(
    source: Any, event: Any, context_id: str
) -> dict[str, Any]:
    """Build attributes for conversation start."""
    # Reuse delegation attributes as base
    attrs = _build_a2a_delegation_start_attributes(source, event, context_id)

    # Add conversation-specific fields
    max_turns = safe_getattr(event, "max_turns")
    if max_turns is not None:
        try:
            attrs["a2a.max_turns"] = int(max_turns)
        except (TypeError, ValueError):
            pass

    return attrs
