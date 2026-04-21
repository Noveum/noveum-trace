"""
Agent-to-Agent (A2A) delegation & communication event handler mixin for NoveumCrewAIListener.

Handles CrewAI A2A events across multiple span types:

  Delegation lifecycle:
  - ``on_a2a_delegation_started``    → open ``crewai.a2a.delegation`` span under
                                        current agent/task; capture delegating agent,
                                        receiving agent, task description, endpoint,
                                        context_id, protocol version
  - ``on_a2a_delegation_completed``  → close delegation span; write result / error
                                        from ``status`` (``failed`` → ERROR span)
  - ``on_a2a_delegation_failed``     → close as ERROR; attach exception details

  Conversation lifecycle:
  - ``on_a2a_conversation_started``  → open ``crewai.a2a.conversation`` span;
                                        capture participants, context_id
  - ``on_a2a_conversation_completed``→ close conversation span; ``status=failed``
                                        closes as ERROR with ``event.error``
  - ``on_a2a_conversation_failed``   → close as ERROR

  Message-level events (no span lifecycle — annotate open conversation):
  - ``on_a2a_message_sent``          → append to message buffer
  - ``on_a2a_message_received``      → append to message buffer

  Streaming (raw chunks, separate from message dict buffer):
  - ``on_a2a_streaming_started``     → initialize ``_a2a_streaming_chunks`` for conversation
  - ``on_a2a_streaming_chunk``       → append chunk; flush early when ``final`` /
                                        ``is_final`` is true (CrewAI streaming)
  - ``on_a2a_streaming_completed``   → flush chunks (also wired if ``A2AStreamingCompletedEvent`` exists)

  Polling (status polling for async delegation):
  - ``on_a2a_polling_started``       → record polling start
  - ``on_a2a_polling_status``        → update status in span

  Other:
  - ``on_a2a_artifact_received``     → record received artifact metadata; for ``image/*``
                                        payloads (bytes or base64 on the event / metadata),
                                        queue an image upload via ``NoveumClient.export_image``
  - ``on_a2a_server_task_*``         → A2A server task lifecycle (started / completed /
                                        failed / canceled; CrewAI ``A2AServerTask*Event``)
  - ``on_a2a_response_received``     → buffer inbound A2A agent text (``A2AResponseReceivedEvent``)
  - ``on_a2a_context_lifecycle_event``→ context created / completed / expired / idle / pruned
  - ``on_a2a_auth_failed``           → authentication error
  - ``on_a2a_connection_error``      → connection failure

  ``setup_listeners()`` wires CrewAI event classes to these handlers where the installed
  CrewAI version exports them (each subscription is isolated in ``ImportError`` blocks).

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown, _agent_spans, _task_spans,
    _a2a_spans, _a2a_stream_buffers, _a2a_streaming_chunks, _a2a_streaming_lengths,
    _a2a_start_times
    (composite keys ``(context_id, "delegation")`` / ``(context_id, "conversation")``)
"""

from __future__ import annotations

import base64
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
    MAX_A2A_IMAGE_BUFFER_BYTES,
    MAX_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
    SPAN_A2A_CONVERSATION,
    SPAN_A2A_DELEGATION,
)
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase
from noveum_trace.integrations.crewai.crewai_utils import (
    duration_ms_monotonic,
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

# Composite dict keys so delegation and conversation spans for the same context_id
# do not overwrite each other.
_A2A_SPAN_DELEGATION = "delegation"
_A2A_SPAN_CONVERSATION = "conversation"


def _a2a_entry_key(context_id: str, span_type: str) -> tuple[str, str]:
    return (context_id, span_type)


def _a2a_artifact_image_buffer_key(context_id: str, artifact_id: Any) -> Optional[str]:
    if artifact_id is None or str(artifact_id).strip() == "":
        return None
    return f"{context_id}::{artifact_id}"


def _mime_type_is_image(mime: Any) -> bool:
    if not isinstance(mime, str):
        return False
    return mime.lower().strip().startswith("image/")


def _image_subtype_from_mime(mime: Optional[str]) -> str:
    if isinstance(mime, str) and "/" in mime:
        sub = mime.split("/", 1)[1].split(";")[0].strip().lower()
        if sub:
            return sub
    return "png"


def _artifact_last_chunk_is_final(last_chunk: Any) -> bool:
    """Treat absent ``last_chunk`` as a single final chunk (CrewAI one-shot artifacts)."""
    if last_chunk is None:
        return True
    return bool(last_chunk)


def _coerce_artifact_bytes(value: Any) -> Optional[bytes]:
    if value is None:
        return None
    if isinstance(value, memoryview):
        try:
            return value.tobytes()
        except Exception:
            return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("data:image/"):
            try:
                from noveum_trace.utils.image_utils import parse_base64_image

                return parse_base64_image(s)["image_data"]
            except Exception:
                return None
        try:
            return base64.b64decode(s, validate=False)
        except Exception:
            return None
    return None


def _extract_a2a_artifact_binary_payload(event: Any) -> Optional[bytes]:
    """Best-effort binary payload for artifact upload (field names vary by CrewAI / runtime)."""
    for name in (
        "artifact_content",
        "artifact_bytes",
        "content",
        "data",
        "bytes",
        "body",
        "chunk",
        "raw",
        "payload",
    ):
        got = _coerce_artifact_bytes(safe_getattr(event, name))
        if got:
            return got
    meta = safe_getattr(event, "metadata")
    if isinstance(meta, dict):
        for key in (
            "content",
            "data",
            "bytes",
            "artifact_bytes",
            "image",
            "base64",
            "body",
        ):
            if key in meta:
                got = _coerce_artifact_bytes(meta.get(key))
                if got:
                    return got
    return None


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
        if not getattr(self, "capture_a2a", False):
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
        Close the ``crewai.a2a.delegation`` span when delegation ends.

        ``A2ADelegationCompletedEvent`` may report ``status`` of ``completed``,
        ``input_required``, ``failed``, etc. Failed statuses are closed as ERROR
        with ``event.error`` when present.
        """
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
            return
        try:
            context_id = _resolve_context_id(event, source)
            status = safe_getattr(event, "status") or ATTR_STATUS_SUCCESS
            status_lower = str(status).lower()
            is_failed = status_lower == "failed"

            result = safe_getattr(event, "result")
            err_msg = safe_getattr(event, "error")
            total_turns = safe_getattr(event, "total_turns")
            extra: dict[str, Any] = {}

            if result and not is_failed:
                extra[ATTR_A2A_RESULT] = truncate_str(str(result), MAX_TEXT_LENGTH)
            if is_failed and err_msg:
                extra["a2a.delegation_error"] = truncate_str(
                    str(err_msg), MAX_DESCRIPTION_LENGTH
                )
            if total_turns is not None:
                try:
                    extra["a2a.total_turns"] = int(total_turns)
                except (TypeError, ValueError):
                    pass

            self._finish_a2a_span(
                context_id=context_id,
                span_type=_A2A_SPAN_DELEGATION,
                status=ATTR_STATUS_ERROR if is_failed else status,
                error=err_msg if is_failed else None,
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
        if not getattr(self, "capture_a2a", False):
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
        if not getattr(self, "capture_a2a", False):
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
        """
        Close the conversation span when CrewAI reports conversation end.

        ``A2AConversationCompletedEvent`` uses ``status`` of ``completed`` or ``failed``;
        failed runs are finished as ERROR with ``event.error`` when present.
        """
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
            return
        try:
            context_id = _resolve_context_id(event, source)
            raw_status = safe_getattr(event, "status")
            status_str = str(raw_status).lower() if raw_status is not None else ""
            is_failed = status_str == "failed"

            total_turns = safe_getattr(event, "total_turns")
            extra: dict[str, Any] = {}

            if total_turns is not None:
                try:
                    extra["a2a.total_turns"] = int(total_turns)
                except (TypeError, ValueError):
                    pass

            final_result = safe_getattr(event, "final_result")
            if final_result is not None and not is_failed:
                extra["a2a.final_result"] = truncate_str(
                    str(final_result), MAX_TEXT_LENGTH
                )

            err_msg = safe_getattr(event, "error")
            if is_failed and err_msg:
                extra["a2a.conversation_error"] = truncate_str(
                    str(err_msg), MAX_DESCRIPTION_LENGTH
                )

            self._flush_a2a_streaming_chunks_for_context(context_id)

            self._finish_a2a_span(
                context_id=context_id,
                span_type=_A2A_SPAN_CONVERSATION,
                status=ATTR_STATUS_ERROR if is_failed else ATTR_STATUS_SUCCESS,
                error=err_msg if is_failed else None,
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
        if not getattr(self, "capture_a2a", False):
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
        if not getattr(self, "capture_a2a", False):
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
        if not getattr(self, "capture_a2a", False):
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

    def on_a2a_response_received(self, source: Any, event: Any) -> None:
        """
        Buffer an inbound response from the remote A2A agent (``A2AResponseReceivedEvent``).

        Writes the same conversation message buffer as ``on_a2a_message_received``,
        with ``type`` = ``response_received`` and optional ``status`` / ``final`` fields.
        """
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
            return
        try:
            context_id = _resolve_context_id(event, source)
            turn_number = safe_getattr(event, "turn_number")
            text = safe_getattr(event, "response") or safe_getattr(event, "message")
            sender = safe_getattr(event, "agent_role") or safe_getattr(
                event, "a2a_agent_name"
            )
            status = safe_getattr(event, "status")
            final = safe_getattr(event, "final")

            msg_entry: dict[str, Any] = {
                "type": "response_received",
                "turn_number": turn_number,
                "content": (
                    truncate_str(str(text), MAX_TEXT_LENGTH) if text is not None else ""
                ),
                "sender": str(sender) if sender else "a2a_agent",
            }
            if status is not None:
                msg_entry["status"] = truncate_str(str(status), 256)
            if final is not None:
                msg_entry["final"] = bool(final)

            ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
            with self._lock:
                buf = self._a2a_stream_buffers.get(ck, [])
                if len(buf) < MAX_A2A_CONVERSATION_MESSAGES:
                    buf.append(msg_entry)
                    self._a2a_stream_buffers[ck] = buf

            logger.debug(
                "A2A response received: context_id=%s turn=%s",
                context_id,
                turn_number,
            )
        except Exception:
            logger.debug("on_a2a_response_received error:\n%s", traceback.format_exc())

    # =========================================================================
    # STREAMING — started / chunk / completed
    # =========================================================================

    def _flush_a2a_streaming_chunks_for_context(self, context_id: str) -> None:
        """Join buffered streaming chunks onto the open conversation span (if any)."""
        ck = _a2a_entry_key(context_id, _A2A_SPAN_CONVERSATION)
        with self._lock:
            raw_chunks = self._a2a_streaming_chunks.pop(ck, [])
            self._a2a_streaming_lengths.pop(ck, None)
            span_entry = self._a2a_spans.get(ck)

        if not span_entry or not raw_chunks:
            return

        span = span_entry.get("span")
        if span is None:
            return

        content = truncate_str("".join(raw_chunks), MAX_TEXT_LENGTH)
        try:
            span.set_attribute("a2a.streaming_content", content)
        except Exception:
            if hasattr(span, "attributes"):
                span.attributes["a2a.streaming_content"] = content

        logger.debug(
            "A2A streaming flushed: context_id=%s content_len=%d",
            context_id,
            len(content),
        )

    def on_a2a_streaming_started(self, source: Any, event: Any) -> None:
        """Initialize raw streaming chunk buffer (separate from message dict buffer)."""
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
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
        if not getattr(self, "capture_a2a", False):
            return
        try:
            context_id = _resolve_context_id(event, source)
            chunk = (
                safe_getattr(event, "chunk")
                or safe_getattr(event, "text")
                or safe_getattr(event, "delta")
                or ""
            )
            is_final = bool(
                safe_getattr(event, "is_final")
                or safe_getattr(event, "final")
                or safe_getattr(event, "is_final_chunk")
            )
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
                logger.debug(
                    "A2A streaming final chunk: context_id=%s (flush)", context_id
                )
                self._flush_a2a_streaming_chunks_for_context(context_id)

        except Exception:
            logger.debug("on_a2a_streaming_chunk error:\n%s", traceback.format_exc())

    def on_a2a_streaming_completed(self, source: Any, event: Any) -> None:
        """Flush raw streaming chunks to the span (message buffer is left intact)."""
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
            return
        try:
            context_id = _resolve_context_id(event, source)
            self._flush_a2a_streaming_chunks_for_context(context_id)
            logger.debug("A2A streaming completed: context_id=%s", context_id)
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
        if not getattr(self, "capture_a2a", False):
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
        if not getattr(self, "capture_a2a", False):
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
        - ``a2a.artifact_name``         — artifact name/filename
        - ``a2a.artifact_mime_type``    — MIME type (e.g., ``image/png``)
        - ``a2a.artifact_size_bytes``   — size in bytes
        - ``a2a.artifact_id``             — unique artifact identifier
        - ``a2a.artifact_image_uuid``   — when ``mime_type`` is ``image/*`` and binary
                                          payload is present on the event (or in ``metadata``),
                                          bytes are queued via ``export_image`` and this UUID is set
        """
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
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
            artifact_desc = safe_getattr(event, "artifact_description")
            task_id_art = safe_getattr(event, "task_id")
            endpoint_art = safe_getattr(event, "endpoint")
            a2a_agent = safe_getattr(event, "a2a_agent_name")
            turn_art = safe_getattr(event, "turn_number")
            is_multi = safe_getattr(event, "is_multiturn")
            append = safe_getattr(event, "append")
            last_chunk = safe_getattr(event, "last_chunk")

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
            if artifact_desc:
                attrs["a2a.artifact_description"] = truncate_str(
                    str(artifact_desc), MAX_DESCRIPTION_LENGTH
                )
            if task_id_art:
                attrs["a2a.artifact_task_id"] = str(task_id_art)
            if endpoint_art:
                attrs["a2a.artifact_endpoint"] = truncate_str(str(endpoint_art), 512)
            if a2a_agent:
                attrs["a2a.artifact_a2a_agent_name"] = truncate_str(str(a2a_agent), 256)
            if turn_art is not None:
                try:
                    attrs["a2a.artifact_turn_number"] = int(turn_art)
                except (TypeError, ValueError):
                    attrs["a2a.artifact_turn_number"] = str(turn_art)
            if is_multi is not None:
                attrs["a2a.artifact_is_multiturn"] = bool(is_multi)
            if append is not None:
                attrs["a2a.artifact_append"] = bool(append)
            if last_chunk is not None:
                attrs["a2a.artifact_last_chunk"] = bool(last_chunk)

            if _mime_type_is_image(mime_type):
                image_uuid = self._maybe_export_a2a_artifact_image(
                    context_id=context_id,
                    span=span,
                    event=event,
                    mime_type=mime_type,
                    artifact_id=artifact_id,
                    artifact_name=artifact_name,
                    last_chunk=last_chunk,
                )
                if image_uuid:
                    attrs["a2a.artifact_image_uuid"] = image_uuid

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
                "A2A artifact received: context_id=%s name=%s size=%s image_uuid=%s",
                context_id,
                artifact_name,
                size_bytes,
                attrs.get("a2a.artifact_image_uuid"),
            )

        except Exception:
            logger.debug("on_a2a_artifact_received error:\n%s", traceback.format_exc())

    # =========================================================================
    # SERVER TASK — A2A server-side task lifecycle (CrewAI A2AServerTask*Event)
    # =========================================================================

    def _write_attrs_on_open_delegation_span(
        self, context_id: str, attrs: dict[str, Any]
    ) -> None:
        """Merge *attrs* onto the open ``crewai.a2a.delegation`` span for *context_id*."""
        dk = _a2a_entry_key(context_id, _A2A_SPAN_DELEGATION)
        with self._lock:
            span_entry = self._a2a_spans.get(dk)
        if not span_entry:
            return
        span = span_entry.get("span")
        if not span or not attrs:
            return
        try:
            if hasattr(span, "set_attributes"):
                span.set_attributes(attrs)
            else:
                for k, v in attrs.items():
                    span.set_attribute(k, v)
        except Exception:
            if hasattr(span, "attributes"):
                span.attributes.update(attrs)

    def _server_task_base_attrs(
        self, source: Any, event: Any, context_id: Optional[str]
    ) -> dict[str, Any]:
        tid = safe_getattr(event, "task_id")
        meta = safe_getattr(event, "metadata")
        attrs: dict[str, Any] = {}
        if context_id:
            attrs["a2a.context_id"] = str(context_id)
        if tid is not None:
            attrs["a2a.server_task.task_id"] = str(tid)
        if meta is not None:
            attrs["a2a.server_task.metadata"] = truncate_str(
                safe_json_dumps(meta), MAX_DESCRIPTION_LENGTH
            )
        # Best-effort extras: these are not guaranteed by A2AServerTask* events,
        # but some runtimes/adapters may attach them.
        endpoint = safe_getattr(event, "endpoint") or safe_getattr(source, "endpoint")
        if endpoint:
            attrs["a2a.server_task.endpoint"] = truncate_str(str(endpoint), 512)
        task_name = safe_getattr(event, "task_name") or safe_getattr(event, "name")
        if task_name:
            attrs["a2a.server_task.task_name"] = truncate_str(str(task_name), 256)
        input_payload = safe_getattr(event, "input") or safe_getattr(event, "payload")
        if input_payload is not None:
            raw = (
                input_payload
                if isinstance(input_payload, str)
                else safe_json_dumps(input_payload)
            )
            attrs["a2a.server_task.input"] = truncate_str(raw, MAX_TEXT_LENGTH)
        return attrs

    def on_a2a_server_task_started(self, source: Any, event: Any) -> None:
        """``A2AServerTaskStartedEvent``: task_id, context_id, metadata on delegation span."""
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
            return
        try:
            context_id = _resolve_context_id(event, source)
            attrs = self._server_task_base_attrs(source, event, context_id)
            attrs["a2a.server_task.phase"] = "started"
            self._write_attrs_on_open_delegation_span(context_id, attrs)
        except Exception:
            logger.debug(
                "on_a2a_server_task_started error:\n%s", traceback.format_exc()
            )

    def on_a2a_server_task_completed(self, source: Any, event: Any) -> None:
        """``A2AServerTaskCompletedEvent``: includes string ``result`` when provided."""
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
            return
        try:
            context_id = _resolve_context_id(event, source)
            attrs = self._server_task_base_attrs(source, event, context_id)
            attrs["a2a.server_task.phase"] = "completed"
            res = safe_getattr(event, "result")
            if res is not None:
                attrs["a2a.server_task.result"] = truncate_str(
                    str(res), MAX_TEXT_LENGTH
                )
            self._write_attrs_on_open_delegation_span(context_id, attrs)
        except Exception:
            logger.debug(
                "on_a2a_server_task_completed error:\n%s", traceback.format_exc()
            )

    def on_a2a_server_task_failed(self, source: Any, event: Any) -> None:
        """``A2AServerTaskFailedEvent``: records ``error`` string."""
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
            return
        try:
            context_id = _resolve_context_id(event, source)
            attrs = self._server_task_base_attrs(source, event, context_id)
            attrs["a2a.server_task.phase"] = "failed"
            err = safe_getattr(event, "error")
            if err is not None:
                attrs["a2a.server_task.error"] = truncate_str(
                    str(err), MAX_DESCRIPTION_LENGTH
                )
            self._write_attrs_on_open_delegation_span(context_id, attrs)
        except Exception:
            logger.debug("on_a2a_server_task_failed error:\n%s", traceback.format_exc())

    def on_a2a_server_task_canceled(self, source: Any, event: Any) -> None:
        """``A2AServerTaskCanceledEvent``."""
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
            return
        try:
            context_id = _resolve_context_id(event, source)
            attrs = self._server_task_base_attrs(source, event, context_id)
            attrs["a2a.server_task.phase"] = "canceled"
            self._write_attrs_on_open_delegation_span(context_id, attrs)
        except Exception:
            logger.debug(
                "on_a2a_server_task_canceled error:\n%s", traceback.format_exc()
            )

    def _write_attrs_on_open_conversation_or_delegation(
        self, context_id: str, attrs: dict[str, Any]
    ) -> None:
        """Apply *attrs* to the first open A2A span: conversation, else delegation."""
        if not attrs:
            return
        for span_type in (_A2A_SPAN_CONVERSATION, _A2A_SPAN_DELEGATION):
            key = _a2a_entry_key(context_id, span_type)
            with self._lock:
                span_entry = self._a2a_spans.get(key)
            if not span_entry:
                continue
            span = span_entry.get("span")
            if not span:
                continue
            try:
                if hasattr(span, "set_attributes"):
                    span.set_attributes(attrs)
                else:
                    for k, v in attrs.items():
                        span.set_attribute(k, v)
            except Exception:
                if hasattr(span, "attributes"):
                    span.attributes.update(attrs)
            return

    def on_a2a_context_lifecycle_event(self, source: Any, event: Any) -> None:
        """
        Record CrewAI A2A context lifecycle (created / completed / expired / idle / pruned).

        Subscribed from ``setup_listeners()`` for each concrete event class that exists
        in the installed CrewAI version.
        """
        if not self._is_active():
            return
        if not getattr(self, "capture_a2a", False):
            return
        try:
            context_id = _resolve_context_id(event, source)
            event_type = safe_getattr(event, "type") or type(event).__name__
            snap: dict[str, Any] = {"event_type": str(event_type)}
            for key in (
                "context_id",
                "created_at",
                "total_tasks",
                "duration_seconds",
                "age_seconds",
                "task_count",
                "idle_seconds",
                "metadata",
            ):
                val = safe_getattr(event, key)
                if val is not None:
                    snap[key] = val
            attrs = {
                "a2a.context_lifecycle.type": str(event_type),
                "a2a.context_lifecycle.snapshot": truncate_str(
                    safe_json_dumps(snap), MAX_TEXT_LENGTH
                ),
            }
            self._write_attrs_on_open_conversation_or_delegation(context_id, attrs)
        except Exception:
            logger.debug(
                "on_a2a_context_lifecycle_event error:\n%s", traceback.format_exc()
            )

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
        if not getattr(self, "capture_a2a", False):
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
        if not getattr(self, "capture_a2a", False):
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

    def _clear_a2a_artifact_image_buffers_for_context(self, context_id: str) -> None:
        """Drop incomplete chunked image buffers when the A2A delegation span closes."""
        prefix = f"{context_id}::"
        with self._lock:
            dead = [k for k in self._a2a_artifact_image_buffers if k.startswith(prefix)]
            for k in dead:
                self._a2a_artifact_image_buffers.pop(k, None)

    def _maybe_export_a2a_artifact_image(
        self,
        *,
        context_id: str,
        span: Any,
        event: Any,
        mime_type: Any,
        artifact_id: Any,
        artifact_name: Any,
        last_chunk: Any,
    ) -> Optional[str]:
        """
        If the event carries image bytes, call ``client.export_image`` and return UUID.

        Chunked artifacts: when ``last_chunk`` is false, bytes are accumulated under
        ``_a2a_artifact_image_buffers`` until a final chunk arrives (requires ``artifact_id``).
        """
        chunk = _extract_a2a_artifact_binary_payload(event)
        if not chunk:
            return None
        client = self._get_client()
        if client is None:
            return None
        # ``MagicMock`` makes bare ``getattr(..., False)`` truthy; only skip real shutdown.
        if getattr(client, "_shutdown", False) is True:
            return None
        trace_id = getattr(span, "trace_id", None)
        span_id = getattr(span, "span_id", None)
        if not trace_id or not span_id:
            return None

        final = _artifact_last_chunk_is_final(last_chunk)
        buf_key = _a2a_artifact_image_buffer_key(context_id, artifact_id)

        with self._lock:
            if not final:
                if buf_key is None:
                    logger.debug(
                        "A2A image artifact non-final chunk without artifact_id; cannot buffer"
                    )
                    return None
                acc = self._a2a_artifact_image_buffers.setdefault(buf_key, bytearray())
                new_size = len(acc) + len(chunk)
                if new_size > MAX_A2A_IMAGE_BUFFER_BYTES:
                    logger.warning(
                        "A2A image artifact buffer cap exceeded; dropping chunk and buffer "
                        "(buf_key=%s, current_bytes=%d, incoming_bytes=%d, max_bytes=%d)",
                        buf_key,
                        len(acc),
                        len(chunk),
                        MAX_A2A_IMAGE_BUFFER_BYTES,
                    )
                    self._a2a_artifact_image_buffers.pop(buf_key, None)
                    return None
                acc.extend(chunk)
                return None
            pending: Optional[bytearray] = None
            if buf_key is not None:
                pending = self._a2a_artifact_image_buffers.pop(buf_key, None)

        parts: list[bytes] = []
        if pending:
            parts.append(bytes(pending))
        parts.append(chunk)
        full = b"".join(parts)
        if len(full) > MAX_A2A_IMAGE_BUFFER_BYTES:
            logger.warning(
                "A2A image artifact assembled payload exceeds cap; dropping export "
                "(buf_key=%s, len=%d, max_bytes=%d)",
                buf_key,
                len(full),
                MAX_A2A_IMAGE_BUFFER_BYTES,
            )
            return None

        from noveum_trace.utils.image_utils import generate_image_uuid

        image_uuid = generate_image_uuid()
        fmt = _image_subtype_from_mime(
            mime_type if isinstance(mime_type, str) else None
        )
        meta: dict[str, Any] = {
            "format": fmt,
            "source": "crewai.a2a.artifact",
        }
        if artifact_id is not None:
            meta["artifact_id"] = str(artifact_id)
        if artifact_name:
            meta["artifact_name"] = str(artifact_name)
        try:
            client.export_image(
                image_data=full,
                trace_id=str(trace_id),
                span_id=str(span_id),
                image_uuid=image_uuid,
                metadata=meta,
            )
        except Exception:
            logger.debug("A2A artifact image export_image failed", exc_info=True)
            return None
        return image_uuid

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

        if span_type == _A2A_SPAN_DELEGATION:
            self._clear_a2a_artifact_image_buffers_for_context(context_id)

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

        if hasattr(span, "set_status"):
            try:
                from noveum_trace.core.span import SpanStatus

                if status == ATTR_STATUS_ERROR:
                    err_type = attrs.get(ATTR_ERROR_TYPE)
                    err_msg = attrs.get(ATTR_ERROR_MESSAGE)
                    detail_parts: list[str] = []
                    if err_type:
                        detail_parts.append(str(err_type))
                    if err_msg:
                        detail_parts.append(str(err_msg))
                    status_msg = (
                        ": ".join(detail_parts)
                        if detail_parts
                        else (str(error) if error else "")
                    )
                    span.set_status(SpanStatus.ERROR, status_msg or None)
                else:
                    span.set_status(SpanStatus.OK)
            except Exception:
                pass

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


def _resolve_task_id(source: Any, event: Any) -> Optional[str]:
    """Extract task ID from context."""
    task_id = safe_getattr(event, "task_id") or safe_getattr(source, "task_id")
    return str(task_id) if task_id else None


def _build_a2a_delegation_start_attributes(
    source: Any, event: Any, context_id: str
) -> dict[str, Any]:
    """Build attributes for delegation / conversation start (CrewAI A2A event fields)."""
    attrs: dict[str, Any] = {
        ATTR_A2A_CONTEXT_ID: context_id,
    }

    delegating = (
        safe_getattr(event, "delegating_agent")
        or safe_getattr(event, "agent_role")
        or safe_getattr(source, "role")
    )
    if delegating:
        attrs[ATTR_A2A_DELEGATING_AGENT] = str(delegating)

    receiving = (
        safe_getattr(event, "receiving_agent")
        or safe_getattr(event, "target_agent")
        or safe_getattr(event, "a2a_agent_name")
    )
    if receiving:
        attrs[ATTR_A2A_RECEIVING_AGENT] = str(receiving)

    remote_agent_id = safe_getattr(event, "agent_id")
    if remote_agent_id:
        attrs["a2a.remote_agent_id"] = str(remote_agent_id)

    a2a_name = safe_getattr(event, "a2a_agent_name")
    if a2a_name:
        attrs["a2a.a2a_agent_name"] = truncate_str(str(a2a_name), 256)

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
    if provider_info is not None:
        if isinstance(provider_info, (dict, list)):
            attrs["a2a.provider_info"] = truncate_str(
                safe_json_dumps(provider_info), MAX_DESCRIPTION_LENGTH
            )
        else:
            attrs["a2a.provider_info"] = truncate_str(
                str(provider_info), MAX_DESCRIPTION_LENGTH
            )

    skill_id = safe_getattr(event, "skill_id")
    if skill_id:
        attrs["a2a.skill_id"] = str(skill_id)

    agent_card = safe_getattr(event, "agent_card")
    if agent_card is not None:
        attrs["a2a.agent_card"] = truncate_str(
            safe_json_dumps(agent_card), MAX_TEXT_LENGTH
        )

    ref_ids = safe_getattr(event, "reference_task_ids")
    if ref_ids is not None:
        attrs["a2a.reference_task_ids"] = truncate_str(
            safe_json_dumps(ref_ids), MAX_DESCRIPTION_LENGTH
        )

    meta = safe_getattr(event, "metadata")
    if meta is not None:
        attrs["a2a.metadata"] = truncate_str(
            safe_json_dumps(meta), MAX_DESCRIPTION_LENGTH
        )

    extensions = safe_getattr(event, "extensions")
    if extensions is not None:
        attrs["a2a.extensions"] = truncate_str(
            safe_json_dumps(extensions), MAX_DESCRIPTION_LENGTH
        )

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
