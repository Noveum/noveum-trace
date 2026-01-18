"""
LiveKit AgentSession and RealtimeSession integration for noveum-trace.

This module provides automatic tracing for LiveKit agent sessions, creating
traces at session.start() and spans for each event that fires.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import uuid
from typing import Any, Callable, Optional

from noveum_trace.core.context import set_current_trace
from noveum_trace.core.span import SpanStatus
from noveum_trace.core.trace import Trace
from noveum_trace.integrations.livekit.livekit_constants import (
    MAX_AUDIO_FRAMES,
    MAX_CONVERSATION_HISTORY,
    MAX_PENDING_FUNCTION_CALLS,
    MAX_PENDING_FUNCTION_OUTPUTS,
)
from noveum_trace.integrations.livekit.livekit_llm import (
    extract_available_tools,
    serialize_tools_for_attributes,
)
from noveum_trace.integrations.livekit.livekit_utils import (
    create_event_span,
    update_speech_span_with_chat_items,
    upload_audio_frames,
)

logger = logging.getLogger(__name__)

try:
    from livekit.agents.llm.realtime import (
        GenerationCreatedEvent,
        InputSpeechStartedEvent,
        InputSpeechStoppedEvent,
        InputTranscriptionCompleted,
        RealtimeSessionReconnectedEvent,
    )
    from livekit.agents.voice.events import (
        AgentStateChangedEvent,
        CloseEvent,
        CloseReason,
        ConversationItemAddedEvent,
        FunctionToolsExecutedEvent,
        MetricsCollectedEvent,
        SpeechCreatedEvent,
        UserInputTranscribedEvent,
        UserStateChangedEvent,
    )

    LIVEKIT_AVAILABLE = True
except ImportError as e:
    LIVEKIT_AVAILABLE = False
    logger.debug(
        "LiveKit is not importable. LiveKit session tracing features will not be available. "
        "Install it with: pip install livekit livekit-agents",
        exc_info=e,
    )


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
        # Always initialize fields so manager is safe to use when LiveKit is unavailable
        self.session = session
        self.enabled = enabled and LIVEKIT_AVAILABLE
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

        # Available tools extracted from agent
        self._available_tools: list[dict[str, Any]] = []

        # Conversation history tracking
        self._conversation_history: list[dict[str, Any]] = []

        # Pending function calls/outputs (to merge with generation span)
        self._pending_generation_span: Optional[Any] = None
        self._pending_function_calls: list[dict[str, Any]] = []
        self._pending_function_outputs: list[dict[str, Any]] = []

        # Audio frames collection for full conversation audio
        self._collected_stt_frames: list[Any] = []
        self._collected_tts_frames: list[Any] = []

        if not LIVEKIT_AVAILABLE:
            logger.error(
                "Cannot initialize LiveKit tracing manager: LiveKit is not available. "
                "Install it with: pip install livekit livekit-agents"
            )
            return

    def _wrap_start_method(self) -> None:
        """Wrap session.start() method to create trace."""
        if not self.enabled:
            return

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

            # Try to create trace first (separate from calling original start)
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
                except Exception as e:
                    logger.debug(
                        f"Failed to get job context for trace name: {e}", exc_info=True
                    )

                # Create trace attributes
                attributes: dict[str, Any] = {
                    "livekit.session_type": "agent_session",
                }

                # Add agent label if available
                if hasattr(agent, "label"):
                    attributes["livekit.agent.label"] = agent.label

                # Extract agent instructions (system prompt) if available
                if hasattr(agent, "instructions") and agent.instructions:
                    attributes["llm.system_prompt"] = agent.instructions
                elif hasattr(agent, "_instructions") and agent._instructions:
                    attributes["llm.system_prompt"] = agent._instructions

                # Extract available tools from agent
                self._available_tools = extract_available_tools(agent)
                if self._available_tools:
                    tool_attrs = serialize_tools_for_attributes(self._available_tools)
                    attributes.update(tool_attrs)
                    logger.debug(
                        f"Extracted {len(self._available_tools)} tools from agent"
                    )

                # Add job context if available
                try:
                    from livekit.agents import get_job_context

                    job_ctx = get_job_context()
                    if job_ctx:
                        attributes["livekit.job.id"] = job_ctx.job.id
                        attributes["livekit.room.name"] = (
                            job_ctx.room.name if hasattr(job_ctx, "room") else None
                        )
                except Exception as e:
                    logger.debug(
                        f"Failed to get job context for trace attributes: {e}",
                        exc_info=True,
                    )

                # Create trace
                self._trace = client.start_trace(
                    name=trace_name,
                    attributes=attributes,
                )

                # Set trace in context
                set_current_trace(self._trace)

            except Exception as e:
                logger.warning(
                    f"Failed to create trace in session.start(): {e}", exc_info=True
                )
                # Fallback to original start without tracing
                assert self._original_start is not None
                return await self._original_start(agent, **kwargs)

            # Trace was successfully created, now call original start
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
                # End trace on error from original start
                if self._trace:
                    self._trace.set_status(SpanStatus.ERROR, str(e))
                    client.finish_trace(self._trace)
                    set_current_trace(None)
                    self._trace = None
                raise

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
            logger.debug(f"RealtimeSession not available or failed to setup: {e}")

    def _try_setup_realtime_handlers_later(self) -> None:
        """Try to setup RealtimeSession handlers later (called from event handlers)."""
        if self._realtime_session is None:
            self._setup_realtime_handlers()

    def _register_realtime_handlers(self, rt_session: Any) -> None:
        """Register handlers for RealtimeSession events."""
        if not self.enabled:
            return

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
        if not self.enabled:
            return

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
                    span = create_event_span(event_type, ev, manager=self)

                    # Special handling for speech_created: store span and start background task
                    if event_type == "speech_created" and span is not None:
                        speech_handle = ev.speech_handle
                        # Store span for later attribute updates
                        self._speech_spans[speech_handle.id] = span
                        # Start background task to update span with chat_items after playout completes
                        asyncio.create_task(
                            update_speech_span_with_chat_items(
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
        """Handle conversation_item_added event and track history."""

        async def _handle_with_history() -> None:
            try:
                # Track conversation history
                item = ev.item if hasattr(ev, "item") else ev
                self._track_conversation_item(item)

                # Create span
                create_event_span("conversation_item_added", ev, manager=self)
            except Exception as e:
                logger.warning(
                    f"Error in conversation_item_added handler: {e}", exc_info=True
                )

        try:
            asyncio.create_task(_handle_with_history())
        except Exception as e:
            logger.warning(
                f"Failed to create task for conversation_item_added: {e}", exc_info=True
            )

    def _track_conversation_item(self, item: Any) -> None:
        """Track a conversation item in history."""
        try:
            item_dict: dict[str, Any] = {}

            # Determine item type
            item_type = getattr(item, "type", None)
            if not item_type:
                if hasattr(item, "content") and hasattr(item, "role"):
                    item_type = "message"
                elif hasattr(item, "name") and hasattr(item, "arguments"):
                    item_type = "function_call"
                elif hasattr(item, "name") and hasattr(item, "output"):
                    item_type = "function_call_output"
                else:
                    item_type = "unknown"

            if item_type == "message":
                # Extract role
                if hasattr(item, "role"):
                    role = item.role
                    if hasattr(role, "value"):
                        item_dict["role"] = str(role.value)
                    else:
                        item_dict["role"] = str(role)

                # Extract content
                if hasattr(item, "text_content"):
                    item_dict["content"] = str(item.text_content)
                elif hasattr(item, "content"):
                    content = item.content
                    if isinstance(content, str):
                        item_dict["content"] = content
                    elif isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, str):
                                text_parts.append(part)
                            elif hasattr(part, "text"):
                                text_parts.append(str(part.text))
                        item_dict["content"] = "\n".join(text_parts)

                item_dict["type"] = "message"
                self._conversation_history.append(item_dict)
                # Enforce cap to prevent memory growth
                if len(self._conversation_history) > MAX_CONVERSATION_HISTORY:
                    self._conversation_history = self._conversation_history[
                        -MAX_CONVERSATION_HISTORY:
                    ]

            elif item_type == "function_call":
                item_dict = {
                    "type": "function_call",
                    "name": str(item.name) if hasattr(item, "name") else "unknown",
                    "arguments": (
                        str(item.arguments) if hasattr(item, "arguments") else ""
                    ),
                }
                self._conversation_history.append(item_dict)
                # Enforce cap to prevent memory growth
                if len(self._conversation_history) > MAX_CONVERSATION_HISTORY:
                    self._conversation_history = self._conversation_history[
                        -MAX_CONVERSATION_HISTORY:
                    ]
                # Also track as pending for merging
                self._pending_function_calls.append(item_dict)
                if len(self._pending_function_calls) > MAX_PENDING_FUNCTION_CALLS:
                    self._pending_function_calls = self._pending_function_calls[
                        -MAX_PENDING_FUNCTION_CALLS:
                    ]

            elif item_type == "function_call_output":
                item_dict = {
                    "type": "function_call_output",
                    "name": str(item.name) if hasattr(item, "name") else "unknown",
                    "output": str(item.output) if hasattr(item, "output") else "",
                    "is_error": (
                        bool(item.is_error) if hasattr(item, "is_error") else False
                    ),
                }
                self._conversation_history.append(item_dict)
                # Enforce cap to prevent memory growth
                if len(self._conversation_history) > MAX_CONVERSATION_HISTORY:
                    self._conversation_history = self._conversation_history[
                        -MAX_CONVERSATION_HISTORY:
                    ]
                # Also track as pending for merging
                self._pending_function_outputs.append(item_dict)
                if len(self._pending_function_outputs) > MAX_PENDING_FUNCTION_OUTPUTS:
                    self._pending_function_outputs = self._pending_function_outputs[
                        -MAX_PENDING_FUNCTION_OUTPUTS:
                    ]

        except Exception as e:
            logger.debug(f"Failed to track conversation item: {e}")

    def _on_agent_false_interruption(self, ev: Any) -> None:
        """Handle agent_false_interruption event."""
        self._create_async_handler("agent_false_interruption")(ev)

    def _on_function_tools_executed(self, ev: FunctionToolsExecutedEvent) -> None:
        """Handle function_tools_executed event - merge into pending generation span."""

        async def _handle_function_tools() -> None:
            try:
                # Extract function call results
                if hasattr(ev, "results") and ev.results:
                    for result in ev.results:
                        output_dict = {
                            "name": (
                                str(result.name)
                                if hasattr(result, "name")
                                else "unknown"
                            ),
                            "output": (
                                str(result.output) if hasattr(result, "output") else ""
                            ),
                            "is_error": (
                                bool(result.is_error)
                                if hasattr(result, "is_error")
                                else False
                            ),
                        }
                        self._pending_function_outputs.append(output_dict)
                        if (
                            len(self._pending_function_outputs)
                            > MAX_PENDING_FUNCTION_OUTPUTS
                        ):
                            self._pending_function_outputs = (
                                self._pending_function_outputs[
                                    -MAX_PENDING_FUNCTION_OUTPUTS:
                                ]
                            )

                # If we have a pending generation span, update it with function data
                if self._pending_generation_span:
                    self._finalize_generation_span_with_functions()
                else:
                    # No pending span, create a minimal event span
                    # (fallback, should rarely happen)
                    create_event_span("function_tools_executed", ev, manager=self)
                    # Avoid leaking outputs/calls into a later generation
                    self._pending_function_outputs.clear()
                    self._pending_function_calls.clear()

            except Exception as e:
                logger.warning(
                    f"Error in function_tools_executed handler: {e}", exc_info=True
                )

        try:
            asyncio.create_task(_handle_function_tools())
        except Exception as e:
            logger.warning(
                f"Failed to create task for function_tools_executed: {e}", exc_info=True
            )

    def _finalize_generation_span_with_functions(self) -> None:
        """Finalize the pending generation span with function call data and finish it."""
        if not self._pending_generation_span:
            return

        try:
            from noveum_trace import get_client

            client = get_client()
            if not client:
                # Can't finish span without client, clear pending data anyway
                self._pending_function_calls.clear()
                self._pending_function_outputs.clear()
                self._pending_generation_span = None
                return

            span = self._pending_generation_span

            # Add function calls to span
            if self._pending_function_calls:
                span.attributes["llm.function_calls.count"] = len(
                    self._pending_function_calls
                )
                try:
                    span.attributes["llm.function_calls"] = json.dumps(
                        self._pending_function_calls, default=str
                    )
                except Exception:
                    pass

            # Add function outputs to span
            if self._pending_function_outputs:
                span.attributes["llm.function_outputs.count"] = len(
                    self._pending_function_outputs
                )
                try:
                    span.attributes["llm.function_outputs"] = json.dumps(
                        self._pending_function_outputs, default=str
                    )
                except Exception:
                    pass

            # Now finish the span with all merged data
            span.set_status(SpanStatus.OK)
            client.finish_span(span)

            # Clear pending data
            self._pending_function_calls.clear()
            self._pending_function_outputs.clear()
            self._pending_generation_span = None

            logger.debug("Finalized generation span with function data")

        except Exception as e:
            logger.warning(f"Failed to finalize generation span: {e}", exc_info=True)

    def _on_metrics_collected(self, ev: MetricsCollectedEvent) -> None:
        """Handle metrics_collected event."""
        self._create_async_handler("metrics_collected")(ev)

    def _on_speech_created(self, ev: SpeechCreatedEvent) -> None:
        """Handle speech_created event."""
        self._create_async_handler("speech_created")(ev)

    def _on_error(self, ev: Any) -> None:
        """Handle error event."""
        self._create_async_handler("error")(ev)

    def _on_close(self, ev: CloseEvent) -> None:
        """Handle close event, add final data, and end trace."""

        async def _handle_close() -> None:
            try:
                create_event_span("close", ev, manager=self)

                # Clean up pending speech spans (background tasks will handle their own cleanup,
                # but we clear the dict to prevent memory leaks)
                self._speech_spans.clear()

                # Flush any pending generation span before ending the trace
                # This ensures function call data is persisted even if no
                # function_tools_executed event was received
                self._finalize_generation_span_with_functions()

                # End trace
                if self._trace:
                    try:
                        from noveum_trace import get_client

                        client = get_client()

                        # Add final conversation history to trace (bounded snapshot)
                        if self._conversation_history:
                            history_snapshot = self._conversation_history[
                                -MAX_CONVERSATION_HISTORY:
                            ]
                            self._trace.set_attributes(
                                {
                                    "conversation.history.message_count": len(
                                        history_snapshot
                                    ),
                                }
                            )
                            try:
                                self._trace.set_attributes(
                                    {
                                        "conversation.history": json.dumps(
                                            history_snapshot, default=str
                                        ),
                                    }
                                )
                            except Exception:
                                pass

                        # Add available tools summary to trace
                        if self._available_tools:
                            self._trace.set_attributes(
                                {
                                    "agent.available_tools.count": len(
                                        self._available_tools
                                    ),
                                    "agent.available_tools.names": [
                                        t.get("name", "unknown")
                                        for t in self._available_tools
                                    ],
                                }
                            )

                        # Upload full conversation audio if we have collected frames
                        await self._upload_full_conversation_audio()

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
                                self._trace.set_status(SpanStatus.ERROR, str(ev.error))
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
            logger.warning(f"Failed to create task for close event: {e}", exc_info=True)

    async def _upload_full_conversation_audio(self) -> None:
        """Upload full conversation audio (combined STT + TTS frames)."""
        if not self._trace:
            return

        try:
            from noveum_trace import get_client

            client = get_client()
            if not client:
                return

            # Upload combined STT audio if available
            if self._collected_stt_frames:
                frame_count = len(self._collected_stt_frames)
                audio_uuid = str(uuid.uuid4())
                span = client.start_span(
                    name="audio.full_conversation.stt",
                    attributes={
                        "audio.type": "stt",
                        "audio.uuid": audio_uuid,
                        "audio.frame_count": frame_count,
                    },
                )
                upload_audio_frames(
                    self._collected_stt_frames,
                    audio_uuid,
                    "stt_full",
                    span.trace_id,
                    span.span_id,
                )
                span.set_status(SpanStatus.OK)
                client.finish_span(span)
                # Clear buffer after upload to free memory
                self._collected_stt_frames.clear()
                logger.debug(f"Uploaded full STT audio: {frame_count} frames")

            # Upload combined TTS audio if available
            if self._collected_tts_frames:
                frame_count = len(self._collected_tts_frames)
                audio_uuid = str(uuid.uuid4())
                span = client.start_span(
                    name="audio.full_conversation.tts",
                    attributes={
                        "audio.type": "tts",
                        "audio.uuid": audio_uuid,
                        "audio.frame_count": frame_count,
                    },
                )
                upload_audio_frames(
                    self._collected_tts_frames,
                    audio_uuid,
                    "tts_full",
                    span.trace_id,
                    span.span_id,
                )
                span.set_status(SpanStatus.OK)
                client.finish_span(span)
                # Clear buffer after upload to free memory
                self._collected_tts_frames.clear()
                logger.debug(f"Uploaded full TTS audio: {frame_count} frames")

        except Exception as e:
            # Clear buffers on exception to free memory even if upload failed
            self._collected_stt_frames.clear()
            self._collected_tts_frames.clear()
            logger.warning(
                f"Failed to upload full conversation audio: {e}", exc_info=True
            )

    def collect_stt_frames(self, frames: list[Any]) -> None:
        """Collect STT audio frames for full conversation audio (capped to prevent OOM)."""
        if not frames:
            return
        # Cap frame collection to prevent OOM on long sessions
        if len(self._collected_stt_frames) >= MAX_AUDIO_FRAMES:
            return  # Already at cap, skip new frames
        remaining = MAX_AUDIO_FRAMES - len(self._collected_stt_frames)
        self._collected_stt_frames.extend(frames[:remaining])

    def collect_tts_frames(self, frames: list[Any]) -> None:
        """Collect TTS audio frames for full conversation audio (capped to prevent OOM)."""
        if not frames:
            return
        # Cap frame collection to prevent OOM on long sessions
        if len(self._collected_tts_frames) >= MAX_AUDIO_FRAMES:
            return  # Already at cap, skip new frames
        remaining = MAX_AUDIO_FRAMES - len(self._collected_tts_frames)
        self._collected_tts_frames.extend(frames[:remaining])

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
        self._create_async_handler("realtime.input_audio_transcription_completed")(ev)

    def _on_generation_created(self, ev: GenerationCreatedEvent) -> None:
        """Handle generation_created event with tools and history."""

        async def _handle_generation() -> None:
            try:
                from noveum_trace import get_client
                from noveum_trace.core.context import get_current_trace

                trace = get_current_trace()
                if not trace:
                    return

                client = get_client()
                if not client:
                    return

                # Serialize event data
                from noveum_trace.integrations.livekit.livekit_utils import (
                    _serialize_event_data,
                    create_constants_metadata,
                )

                attributes = _serialize_event_data(ev, "realtime.generation_created")
                attributes["event.type"] = "realtime.generation_created"

                # Add available tools
                if self._available_tools:
                    tool_attrs = serialize_tools_for_attributes(self._available_tools)
                    attributes.update(tool_attrs)

                # Add conversation history snapshot (bounded to prevent oversized attributes)
                if self._conversation_history:
                    history_snapshot = self._conversation_history[
                        -MAX_CONVERSATION_HISTORY:
                    ]
                    attributes["llm.conversation.message_count"] = len(history_snapshot)
                    try:
                        attributes["llm.conversation.history"] = json.dumps(
                            history_snapshot, default=str
                        )
                    except Exception:
                        pass

                # Add constants metadata
                attributes["metadata"] = create_constants_metadata()

                # Create span (do NOT finish yet - wait for function call data)
                span = client.start_span(
                    name="livekit.realtime.generation_created",
                    attributes=attributes,
                )

                # Store as pending for function call merging
                # The span will be finished in _finalize_generation_span_with_functions
                # after function calls/outputs are merged
                self._pending_generation_span = span

            except Exception as e:
                logger.warning(
                    f"Error in generation_created handler: {e}", exc_info=True
                )

        try:
            asyncio.create_task(_handle_generation())
        except Exception as e:
            logger.warning(
                f"Failed to create task for generation_created: {e}", exc_info=True
            )

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
            except Exception as e:
                logger.debug(
                    f"Failed to remove event handler for {event_type}: {e}",
                    exc_info=True,
                )

        # Remove realtime handlers
        if self._realtime_session:
            for event_type, handler in self._realtime_handlers:
                try:
                    self._realtime_session.off(event_type, handler)
                except Exception as e:
                    logger.debug(
                        f"Failed to remove realtime handler for {event_type}: {e}",
                        exc_info=True,
                    )

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

    if not manager.enabled:
        return manager

    # Wrap start method
    manager._wrap_start_method()

    # Register AgentSession event handlers
    manager._register_agent_session_handlers()

    return manager
