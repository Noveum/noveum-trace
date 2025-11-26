"""
LiveKit AgentSession and RealtimeSession integration for noveum-trace.

This module provides automatic tracing for LiveKit agent sessions, creating
traces at session.start() and spans for each event that fires.
"""

import asyncio
import functools
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional

from noveum_trace.core.trace import Trace

from noveum_trace.core.context import (
    get_current_span,
    get_current_trace,
    set_current_trace,
)
from noveum_trace.core.span import SpanStatus

logger = logging.getLogger(__name__)

try:
    from livekit.agents.voice.events import (
        AgentStateChangedEvent,
        CloseEvent,
        CloseReason,
        ConversationItemAddedEvent,
        ErrorEvent,
        FunctionToolsExecutedEvent,
        MetricsCollectedEvent,
        SpeechCreatedEvent,
        UserInputTranscribedEvent,
        UserStateChangedEvent,
    )
    from livekit.agents.llm.realtime import (
        GenerationCreatedEvent,
        InputSpeechStartedEvent,
        InputSpeechStoppedEvent,
        InputTranscriptionCompleted,
        RealtimeSessionReconnectedEvent,
    )

    LIVEKIT_AVAILABLE = True
except ImportError as e:
    LIVEKIT_AVAILABLE = False
    logger.error(
        "LiveKit is not importable. LiveKit session tracing features will not be available. "
        "Install it with: pip install livekit livekit-agents",
        exc_info=e,
    )


def _serialize_event_data(event: Any, prefix: str = "") -> dict[str, Any]:
    """
    Serialize event data to a dictionary for span attributes.

    Handles Pydantic models, dataclasses, and nested objects recursively.

    Args:
        event: Event object to serialize
        prefix: Optional prefix for attribute keys

    Returns:
        Dictionary of serialized attributes
    """
    if event is None:
        return {}

    result: dict[str, Any] = {}

    try:
        # Handle Pydantic models (v2 uses model_dump, v1 uses dict)
        if hasattr(event, "model_dump"):
            data = event.model_dump()
        elif hasattr(event, "dict"):
            data = event.dict()
        # Handle dataclasses
        elif is_dataclass(event) and not isinstance(event, type):
            # Type guard: is_dataclass ensures event is a dataclass instance, not a class
            data = asdict(event)
        # Handle objects with __dict__
        elif hasattr(event, "__dict__"):
            data = {k: v for k, v in event.__dict__.items()
                    if not k.startswith("_")}
        # Handle dictionaries
        elif isinstance(event, dict):
            data = event
        else:
            # Fallback: try to convert to string
            return {prefix: str(event)} if prefix else {"value": str(event)}

        # Recursively serialize nested structures
        for key, value in data.items():
            # Skip excluded fields (Pydantic models may have exclude in model_dump)
            if value is None:
                continue

            attr_key = f"{prefix}.{key}" if prefix else key

            # Handle nested objects
            if isinstance(value, (dict, list, tuple)):
                serialized = _serialize_value(value, attr_key)
                if isinstance(serialized, dict):
                    result.update(serialized)
                else:
                    result[attr_key] = serialized
            elif isinstance(value, (str, int, float, bool)):
                result[attr_key] = value
            elif (
                is_dataclass(value)
                or hasattr(value, "model_dump")
                or hasattr(value, "dict")
            ):
                # Recursively serialize nested objects
                nested = _serialize_event_data(value, attr_key)
                result.update(nested)
            elif hasattr(value, "__dict__"):
                nested = _serialize_event_data(value, attr_key)
                result.update(nested)
            else:
                # Convert to string as fallback
                result[attr_key] = str(value)

    except Exception as e:
        logger.warning(f"Failed to serialize event data: {e}")
        result[prefix or "event"] = str(event)

    return result


def _serialize_value(value: Any, prefix: str = "") -> Any:
    """
    Serialize a value (handles lists, tuples, dicts recursively).

    Args:
        value: Value to serialize
        prefix: Optional prefix for keys

    Returns:
        Serialized value
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, dict):
        result = {}
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else k
            serialized = _serialize_value(v, key)
            if isinstance(serialized, dict):
                result.update(serialized)
            else:
                result[key] = serialized
        return result
    elif isinstance(value, (list, tuple)):
        # For lists/tuples, convert to indexed attributes
        result = {}
        for i, item in enumerate(value):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            serialized = _serialize_value(item, key)
            if isinstance(serialized, dict):
                result.update(serialized)
            else:
                result[key] = serialized
        return result
    elif is_dataclass(value) or hasattr(value, "model_dump") or hasattr(value, "dict"):
        return _serialize_event_data(value, prefix)
    elif hasattr(value, "__dict__"):
        return _serialize_event_data(value, prefix)
    else:
        return str(value)


def _serialize_chat_items(chat_items: list[Any]) -> dict[str, Any]:
    """
    Serialize chat items (messages, function calls, outputs) into span attributes.
    Flattens all items directly into attributes without nesting.

    Args:
        chat_items: List of ChatItem objects (ChatMessage, FunctionCall, FunctionCallOutput)

    Returns:
        Dictionary of serialized attributes
    """
    if not chat_items:
        return {}

    result: dict[str, Any] = {"speech.chat_items.count": len(chat_items)}

    # Collect all messages, function calls, and outputs
    messages = []
    function_calls = []
    function_outputs = []

    for item in chat_items:
        # Determine item type
        item_type = getattr(item, "type", None)
        if not item_type:
            # Try to infer from class name or attributes
            if hasattr(item, "content") and hasattr(item, "role"):
                item_type = "message"
            elif hasattr(item, "name") and hasattr(item, "arguments"):
                item_type = "function_call"
            elif hasattr(item, "name") and hasattr(item, "output"):
                item_type = "function_call_output"
            else:
                item_type = "unknown"

        if item_type == "message":
            # ChatMessage - extract text content
            text_content = None
            if hasattr(item, "text_content"):
                text_content = str(item.text_content)
            elif hasattr(item, "content"):
                content = item.content
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif hasattr(part, "text"):
                            text_parts.append(str(part.text))
                        elif isinstance(part, dict) and "text" in part:
                            text_parts.append(str(part["text"]))
                    text_content = "\n".join(
                        text_parts) if text_parts else None
                elif isinstance(content, str):
                    text_content = content

            if text_content:
                messages.append(
                    {
                        "role": str(item.role) if hasattr(item, "role") else None,
                        "content": text_content,
                        "interrupted": (
                            bool(item.interrupted)
                            if hasattr(item, "interrupted")
                            else False
                        ),
                    }
                )

        elif item_type == "function_call":
            # FunctionCall
            function_calls.append(
                {
                    "name": str(item.name) if hasattr(item, "name") else None,
                    "arguments": (
                        str(item.arguments) if hasattr(
                            item, "arguments") else None
                    ),
                }
            )

        elif item_type == "function_call_output":
            # FunctionCallOutput
            function_outputs.append(
                {
                    "name": str(item.name) if hasattr(item, "name") else None,
                    "output": str(item.output) if hasattr(item, "output") else None,
                    "is_error": (
                        bool(item.is_error) if hasattr(
                            item, "is_error") else False
                    ),
                }
            )

    # Add flattened results
    if messages:
        result["speech.messages"] = messages
    if function_calls:
        result["speech.function_calls"] = function_calls
    if function_outputs:
        result["speech.function_outputs"] = function_outputs

    return result


async def _update_speech_span_with_chat_items(
    speech_handle: Any,
    span: Any,
    manager: Any,
) -> None:
    """
    Wait for speech playout to complete, then update span with chat_items.

    Args:
        speech_handle: SpeechHandle instance
        span: Span to update
        manager: _LiveKitTracingManager instance
    """
    try:
        # Wait for speech to complete (all tasks done, playout finished)
        await speech_handle.wait_for_playout()

        # Now chat_items should be fully populated
        chat_items = speech_handle.chat_items

        if chat_items:
            # Serialize chat_items
            chat_attributes = _serialize_chat_items(chat_items)

            # Directly modify span.attributes (bypassing set_attribute since span is finished)
            span.attributes.update(chat_attributes)

            logger.debug(
                f"Updated speech span {span.span_id} with {len(chat_items)} chat items"
            )

        # Remove from tracking
        speech_id = speech_handle.id
        if speech_id in manager._speech_spans:
            del manager._speech_spans[speech_id]

    except Exception as e:
        logger.warning(
            f"Failed to update speech span with chat_items: {e}", exc_info=True
        )
        # Still remove from tracking to prevent memory leak
        speech_id = speech_handle.id
        if speech_id in manager._speech_spans:
            del manager._speech_spans[speech_id]


def _create_event_span(
    event_type: str, event_data: Any, manager: Optional[Any] = None
) -> Optional[Any]:
    """
    Create a span for an event with explicit parent resolution.

    Args:
        event_type: Type of event (e.g., "user_state_changed")
        event_data: Event object to serialize
        manager: Optional _LiveKitTracingManager instance for tracking parent spans

    Returns:
        The created Span instance, or None if creation failed
    """
    try:
        # Get current trace (should exist from session.start())
        trace = get_current_trace()
        if trace is None:
            logger.debug(
                f"No active trace for event {event_type}, skipping span creation"
            )
            return None

        # Get client
        from noveum_trace import get_client

        try:
            client = get_client()
        except Exception as e:
            logger.warning(f"Failed to get Noveum client: {e}")
            return None

        # Serialize event data
        attributes = _serialize_event_data(event_data, event_type)

        # Add event type as attribute
        attributes["event.type"] = event_type

        # Create span name
        span_name = f"livekit.{event_type}"

        # Determine parent span ID
        # metrics_collected events should use the latest agent_state_changed span as parent
        # and should not be set as current span
        # speech_created events should also not be set as current (finished immediately)
        is_metrics_event = (
            event_type == "metrics_collected"
            or event_type == "realtime.metrics_collected"
        )
        is_speech_event = event_type == "speech_created"

        if is_metrics_event:
            # Use the latest agent_state_changed span as parent, or None if none exists yet
            if manager and manager._last_agent_state_changed_span_id:
                parent_span_id = manager._last_agent_state_changed_span_id
                use_direct_create = (
                    False  # Use client.start_span() with explicit parent
                )
            else:
                # No agent_state_changed yet, create as direct child of trace
                # Bypass client.start_span() to avoid its None fallback to context
                parent_span_id = None
                use_direct_create = True  # Use trace.create_span() directly
            set_as_current = False  # Don't set as current span
        elif is_speech_event:
            # speech_created: use context-based parent resolution
            current_span = get_current_span()
            if current_span and (
                current_span.name == "livekit.metrics_collected"
                or current_span.name == "livekit.realtime.metrics_collected"
            ):
                # Current span is metrics_collected, use latest agent_state_changed as parent
                if manager and manager._last_agent_state_changed_span_id:
                    parent_span_id = manager._last_agent_state_changed_span_id
                else:
                    parent_span_id = None
            else:
                # Use current span as parent (or None if no current span)
                parent_span_id = current_span.span_id if current_span else None
            use_direct_create = False  # Use normal client.start_span()
            # Set as current (will finish immediately anyway)
            set_as_current = True
        else:
            # For other events, check if current span is metrics_collected
            # If so, use the latest agent_state_changed span as parent
            current_span = get_current_span()
            if current_span and (
                current_span.name == "livekit.metrics_collected"
                or current_span.name == "livekit.realtime.metrics_collected"
            ):
                # Current span is metrics_collected, use latest agent_state_changed as parent
                if manager and manager._last_agent_state_changed_span_id:
                    parent_span_id = manager._last_agent_state_changed_span_id
                    use_direct_create = False
                else:
                    # No agent_state_changed yet, create as direct child of trace
                    parent_span_id = None
                    use_direct_create = True
            else:
                # Use current span as parent (or None if no current span)
                parent_span_id = current_span.span_id if current_span else None
                use_direct_create = False  # Use normal client.start_span()
            set_as_current = True  # Set as current for other events

        # Create span
        if use_direct_create:
            # Bypass client.start_span() to avoid its None fallback to context
            # Create span directly via trace
            span = trace.create_span(
                name=span_name,
                parent_span_id=None,  # Explicitly no parent
                attributes=attributes,
            )
            # Don't set as current (metrics_collected should never be current)
        else:
            # Use normal client.start_span() which handles context properly
            span = client.start_span(
                name=span_name,
                attributes=attributes,
                parent_span_id=parent_span_id,
                set_as_current=set_as_current,
            )

        # Track agent_state_changed spans for use as parent for metrics_collected
        if event_type == "agent_state_changed" and manager:
            manager._last_agent_state_changed_span_id = span.span_id

        # Set status for error events
        if event_type == "error":
            span.set_status(
                SpanStatus.ERROR,
                (
                    str(event_data.error)
                    if hasattr(event_data, "error")
                    else "Error occurred"
                ),
            )
        elif LIVEKIT_AVAILABLE:
            # Only check isinstance if LiveKit types are available
            try:
                if (
                    isinstance(event_data, ErrorEvent)
                    and hasattr(event_data, "error")
                    and event_data.error
                ):
                    span.set_status(SpanStatus.ERROR, str(event_data.error))
            except (NameError, TypeError):
                # ErrorEvent not available, skip isinstance check
                pass

        # Finish span immediately (events are instantaneous)
        # Note: We don't need to restore context for metrics_collected since we never set it
        client.finish_span(span)

        return span

    except Exception as e:
        logger.warning(
            f"Failed to create span for event {event_type}: {e}", exc_info=True
        )
        return None


class _LiveKitTracingManager:
    """Manages tracing for a LiveKit AgentSession."""

    def __init__(
        self,
        session: Any,
        enabled: bool = True,
        trace_name_prefix: Optional[str] = None,
    ):
        """
        Initialize tracing manager.

        Args:
            session: AgentSession instance
            enabled: Whether tracing is enabled
            trace_name_prefix: Optional prefix for trace names
        """
        if not LIVEKIT_AVAILABLE:
            logger.error(
                "Cannot initialize LiveKit tracing manager: LiveKit is not available. "
                "Install it with: pip install livekit livekit-agents"
            )
            return

        self.session = session
        self.enabled = enabled
        self.trace_name_prefix = trace_name_prefix or "livekit"
        self._original_start: Optional[Callable[..., Any]] = None
        self._trace: Optional[Trace] = None
        # List of (event_type, handler)
        self._event_handlers: list[tuple[str, Any]] = []
        self._realtime_session: Optional[Any] = None
        self._realtime_handlers: list[tuple[str, Any]] = []
        self._wrapped = False
        # Track the latest agent_state_changed span ID to use as parent for metrics_collected
        self._last_agent_state_changed_span_id: Optional[str] = None
        # Track finished speech spans by speech_handle.id for later attribute updates
        self._speech_spans: dict[str, Any] = {}  # speech_id -> Span

    def _wrap_start_method(self) -> None:
        """Wrap session.start() method to create trace."""
        if self._wrapped:
            return

        self._original_start = self.session.start
        assert self._original_start is not None, "session.start must be callable"

        @functools.wraps(self._original_start)
        async def wrapped_start(agent: Any, **kwargs: Any) -> Any:
            """Wrapped start method that creates trace."""
            if not self.enabled:
                assert self._original_start is not None
                return await self._original_start(agent, **kwargs)

            try:
                # Get client
                from noveum_trace import get_client

                client = get_client()

                # Create trace name
                trace_name = f"{self.trace_name_prefix}.agent_session"

                # Try to get session ID or job context for trace name
                try:
                    from livekit.agents import get_job_context

                    job_ctx = get_job_context()
                    if job_ctx:
                        trace_name = (
                            f"{self.trace_name_prefix}.agent_session.{job_ctx.job.id}"
                        )
                except Exception:
                    pass

                # Create trace attributes
                attributes: dict[str, Any] = {
                    "livekit.session_type": "agent_session",
                }

                # Add agent label if available
                if hasattr(agent, "label"):
                    attributes["livekit.agent.label"] = agent.label

                # Add job context if available
                try:
                    from livekit.agents import get_job_context

                    job_ctx = get_job_context()
                    if job_ctx:
                        attributes["livekit.job.id"] = job_ctx.job.id
                        attributes["livekit.room.name"] = (
                            job_ctx.room.name if hasattr(
                                job_ctx, "room") else None
                        )
                except Exception:
                    pass

                # Create trace
                self._trace = client.start_trace(
                    name=trace_name,
                    attributes=attributes,
                )

                # Set trace in context
                set_current_trace(self._trace)

                # Call original start
                try:
                    assert self._original_start is not None
                    result = await self._original_start(agent, **kwargs)

                    # Check for RealtimeSession after start (with retry)
                    # agent_activity might not be immediately available
                    # Small delay to let session initialize
                    await asyncio.sleep(0.1)
                    self._setup_realtime_handlers()

                    return result
                except Exception as e:
                    # End trace on error
                    if self._trace:
                        self._trace.set_status(SpanStatus.ERROR, str(e))
                        client.finish_trace(self._trace)
                        set_current_trace(None)
                        self._trace = None
                    raise

            except Exception as e:
                logger.warning(
                    f"Failed to create trace in session.start(): {e}", exc_info=True
                )
                # Fallback to original start without tracing
                assert self._original_start is not None
                return await self._original_start(agent, **kwargs)

        # Replace method
        self.session.start = wrapped_start
        self._wrapped = True

    def _setup_realtime_handlers(self) -> None:
        """Setup handlers for RealtimeSession events if available."""
        if not self.enabled:
            return

        # Skip if already set up
        if self._realtime_session is not None:
            return

        try:
            # Check if session has agent_activity with realtime session
            if hasattr(self.session, "agent_activity"):
                agent_activity = self.session.agent_activity
                if agent_activity and hasattr(agent_activity, "realtime_llm_session"):
                    rt_session = agent_activity.realtime_llm_session()
                    if rt_session:
                        self._realtime_session = rt_session
                        self._register_realtime_handlers(rt_session)
                        logger.debug("RealtimeSession handlers registered")
        except Exception as e:
            logger.debug(
                f"RealtimeSession not available or failed to setup: {e}")

    def _try_setup_realtime_handlers_later(self) -> None:
        """Try to setup RealtimeSession handlers later (called from event handlers)."""
        if self._realtime_session is None:
            self._setup_realtime_handlers()

    def _register_realtime_handlers(self, rt_session: Any) -> None:
        """Register handlers for RealtimeSession events."""
        handlers = [
            ("input_speech_started", self._on_input_speech_started),
            ("input_speech_stopped", self._on_input_speech_stopped),
            (
                "input_audio_transcription_completed",
                self._on_input_audio_transcription_completed,
            ),
            ("generation_created", self._on_generation_created),
            ("session_reconnected", self._on_session_reconnected),
            ("metrics_collected", self._on_realtime_metrics_collected),
            ("error", self._on_realtime_error),
        ]

        for event_type, handler in handlers:
            try:
                rt_session.on(event_type, handler)
                self._realtime_handlers.append((event_type, handler))
            except Exception as e:
                logger.warning(
                    f"Failed to register RealtimeSession handler for {event_type}: {e}"
                )

    def _register_agent_session_handlers(self) -> None:
        """Register handlers for AgentSession events."""
        handlers = [
            ("user_state_changed", self._on_user_state_changed),
            ("agent_state_changed", self._on_agent_state_changed),
            ("user_input_transcribed", self._on_user_input_transcribed),
            ("conversation_item_added", self._on_conversation_item_added),
            ("agent_false_interruption", self._on_agent_false_interruption),
            ("function_tools_executed", self._on_function_tools_executed),
            ("metrics_collected", self._on_metrics_collected),
            ("speech_created", self._on_speech_created),
            ("error", self._on_error),
            ("close", self._on_close),
        ]

        for event_type, handler in handlers:
            try:
                self.session.on(event_type, handler)
                self._event_handlers.append((event_type, handler))
            except Exception as e:
                logger.warning(
                    f"Failed to register AgentSession handler for {event_type}: {e}"
                )

    def _create_async_handler(
        self, event_type: str, additional_work: Optional[Any] = None
    ) -> Callable[[Any], None]:
        """
        Create a synchronous handler that runs async code via asyncio.create_task.

        Args:
            event_type: Type of event (e.g., "user_state_changed")
            additional_work: Optional async function to call after creating span
        """

        def handler(ev: Any) -> None:
            async def _handle() -> None:
                try:
                    span = _create_event_span(event_type, ev, manager=self)

                    # Special handling for speech_created: store span and start background task
                    if event_type == "speech_created" and span is not None:
                        speech_handle = ev.speech_handle
                        # Store span for later attribute updates
                        self._speech_spans[speech_handle.id] = span
                        # Start background task to update span with chat_items after playout completes
                        asyncio.create_task(
                            _update_speech_span_with_chat_items(
                                speech_handle, span, self
                            )
                        )

                    if additional_work:
                        await additional_work()
                except Exception as e:
                    logger.warning(
                        f"Error in event handler for {event_type}: {e}", exc_info=True
                    )

            try:
                asyncio.create_task(_handle())
            except Exception as e:
                logger.warning(
                    f"Failed to create task for {event_type}: {e}", exc_info=True
                )

        return handler

    # AgentSession event handlers (synchronous, use asyncio.create_task internally)
    def _on_user_state_changed(self, ev: UserStateChangedEvent) -> None:
        """Handle user_state_changed event."""
        self._create_async_handler("user_state_changed")(ev)

    def _on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        """Handle agent_state_changed event."""

        async def _additional_work() -> None:
            # Try to setup realtime handlers if not already done (session might be ready now)
            self._try_setup_realtime_handlers_later()

        self._create_async_handler("agent_state_changed", _additional_work)(ev)

    def _on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        """Handle user_input_transcribed event."""
        self._create_async_handler("user_input_transcribed")(ev)

    def _on_conversation_item_added(self, ev: ConversationItemAddedEvent) -> None:
        """Handle conversation_item_added event."""
        self._create_async_handler("conversation_item_added")(ev)

    def _on_agent_false_interruption(self, ev: Any) -> None:
        """Handle agent_false_interruption event."""
        self._create_async_handler("agent_false_interruption")(ev)

    def _on_function_tools_executed(self, ev: FunctionToolsExecutedEvent) -> None:
        """Handle function_tools_executed event."""
        self._create_async_handler("function_tools_executed")(ev)

    def _on_metrics_collected(self, ev: MetricsCollectedEvent) -> None:
        """Handle metrics_collected event."""
        self._create_async_handler("metrics_collected")(ev)

    def _on_speech_created(self, ev: SpeechCreatedEvent) -> None:
        """Handle speech_created event."""
        self._create_async_handler("speech_created")(ev)

    def _on_error(self, ev: ErrorEvent) -> None:
        """Handle error event."""
        self._create_async_handler("error")(ev)

    def _on_close(self, ev: CloseEvent) -> None:
        """Handle close event and end trace."""

        async def _handle_close() -> None:
            try:
                _create_event_span("close", ev, manager=self)

                # Clean up pending speech spans (background tasks will handle their own cleanup,
                # but we clear the dict to prevent memory leaks)
                self._speech_spans.clear()

                # End trace
                if self._trace:
                    try:
                        from noveum_trace import get_client

                        client = get_client()

                        # Set trace status based on close reason
                        if LIVEKIT_AVAILABLE:
                            try:
                                if (
                                    ev.reason == CloseReason.ERROR
                                    and hasattr(ev, "error")
                                    and ev.error
                                ):
                                    self._trace.set_status(
                                        SpanStatus.ERROR, str(ev.error)
                                    )
                                elif hasattr(ev, "reason") and ev.reason in (
                                    CloseReason.JOB_SHUTDOWN,
                                    CloseReason.PARTICIPANT_DISCONNECTED,
                                ):
                                    self._trace.set_status(SpanStatus.OK)
                                else:
                                    self._trace.set_status(SpanStatus.OK)
                            except (NameError, AttributeError):
                                # CloseReason not available, use default status
                                self._trace.set_status(SpanStatus.OK)
                        else:
                            # LiveKit not available, use default status
                            if hasattr(ev, "error") and ev.error:
                                self._trace.set_status(
                                    SpanStatus.ERROR, str(ev.error))
                            else:
                                self._trace.set_status(SpanStatus.OK)

                        client.finish_trace(self._trace)
                        set_current_trace(None)
                        self._trace = None
                    except Exception as e:
                        logger.warning(
                            f"Failed to end trace on close: {e}", exc_info=True
                        )
            except Exception as e:
                logger.warning(f"Error in close handler: {e}", exc_info=True)

        try:
            asyncio.create_task(_handle_close())
        except Exception as e:
            logger.warning(
                f"Failed to create task for close event: {e}", exc_info=True)

    # RealtimeSession event handlers (synchronous, use asyncio.create_task internally)
    def _on_input_speech_started(self, ev: InputSpeechStartedEvent) -> None:
        """Handle input_speech_started event."""
        self._create_async_handler("realtime.input_speech_started")(ev)

    def _on_input_speech_stopped(self, ev: InputSpeechStoppedEvent) -> None:
        """Handle input_speech_stopped event."""
        self._create_async_handler("realtime.input_speech_stopped")(ev)

    def _on_input_audio_transcription_completed(
        self, ev: InputTranscriptionCompleted
    ) -> None:
        """Handle input_audio_transcription_completed event."""
        self._create_async_handler(
            "realtime.input_audio_transcription_completed")(ev)

    def _on_generation_created(self, ev: GenerationCreatedEvent) -> None:
        """Handle generation_created event."""
        self._create_async_handler("realtime.generation_created")(ev)

    def _on_session_reconnected(self, ev: RealtimeSessionReconnectedEvent) -> None:
        """Handle session_reconnected event."""
        self._create_async_handler("realtime.session_reconnected")(ev)

    def _on_realtime_metrics_collected(self, ev: Any) -> None:
        """Handle realtime metrics_collected event."""
        self._create_async_handler("realtime.metrics_collected")(ev)

    def _on_realtime_error(self, ev: Any) -> None:
        """Handle realtime error event."""
        self._create_async_handler("realtime.error")(ev)

    def cleanup(self) -> None:
        """Cleanup event handlers and restore original methods."""
        # Remove event handlers
        for event_type, handler in self._event_handlers:
            try:
                self.session.off(event_type, handler)
            except Exception:
                pass

        # Remove realtime handlers
        if self._realtime_session:
            for event_type, handler in self._realtime_handlers:
                try:
                    self._realtime_session.off(event_type, handler)
                except Exception:
                    pass  # Ignore errors during cleanup

        # Restore original start method
        if self._original_start and self._wrapped:
            self.session.start = self._original_start
            self._wrapped = False


def setup_livekit_tracing(
    session: Any,
    *,
    enabled: bool = True,
    trace_name_prefix: Optional[str] = None,
) -> _LiveKitTracingManager:
    """
    Setup automatic tracing for a LiveKit AgentSession.

    This function hooks into the session's event system to automatically:
    - Create a trace when session.start(agent) is called
    - Create spans for each AgentSession event
    - Create spans for RealtimeSession events (if using RealtimeModel)
    - End trace on close or error events

    Args:
        session: The AgentSession to trace
        enabled: Whether tracing is enabled (default: True)
        trace_name_prefix: Optional prefix for trace names (default: "livekit")

    Returns:
        Tracing manager instance for cleanup if needed

    Example:
        >>> from livekit.agents import AgentSession, Agent
        >>> from noveum_trace.integrations.livekit import setup_livekit_tracing
        >>>
        >>> session = AgentSession()
        >>> manager = setup_livekit_tracing(session)
        >>>
        >>> agent = Agent(instructions="You are helpful.")
        >>> await session.start(agent)  # Trace is automatically created
    """
    manager = _LiveKitTracingManager(
        session, enabled=enabled, trace_name_prefix=trace_name_prefix
    )

    # Wrap start method
    manager._wrap_start_method()

    # Register AgentSession event handlers
    manager._register_agent_session_handlers()

    return manager
