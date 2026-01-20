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
from pathlib import Path
from typing import Any, Callable, Optional

from noveum_trace.core.context import set_current_span, set_current_trace
from noveum_trace.core.span import SpanStatus
from noveum_trace.core.trace import Trace
from noveum_trace.integrations.livekit.livekit_constants import (
    MAX_CONVERSATION_HISTORY,
)
from noveum_trace.integrations.livekit.livekit_utils import (
    create_event_span,
    extract_available_tools,
    get_conversation_history_from_session,
    get_recorder_audio_path,
    serialize_function_calls,
    serialize_tools_for_attributes,
    update_speech_span_with_chat_items,
    upload_audio_file,
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
        # Track the latest agent_state_changed span ID to use as parent for metrics_collected
        self._last_agent_state_changed_span_id: Optional[str] = None
        # Track finished speech spans by speech_handle.id for later attribute updates
        self._speech_spans: dict[str, Any] = {}  # speech_id -> Span

        # Track pending LLM metrics by speech_id (to merge into speech_created spans)
        # Format: {speech_id: {"prompt_tokens": N, "completion_tokens": N, "model": "...", ...}}
        self._pending_llm_metrics: dict[str, dict[str, Any]] = {}

        # Available tools extracted from agent
        self._available_tools: list[dict[str, Any]] = []

        # Pending generation span (to merge function calls/outputs into)
        # Function calls/outputs come directly from function_tools_executed event
        self._pending_generation_span: Optional[Any] = None

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

        if self._original_start is not None:
            return  # Already wrapped

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

                # Ensure RecorderIO actually starts when record=True in console mode.
                # LiveKit console requires --record, but tracing expects recording if record=True.
                await self._ensure_recorder_started(bool(kwargs.get("record", False)))

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

    async def _ensure_recorder_started(self, record_enabled: bool) -> None:
        """Start RecorderIO if record=True but console didn't start it."""
        if not record_enabled:
            return
        if not LIVEKIT_AVAILABLE:
            return

        recorder_io = getattr(self.session, "_recorder_io", None)
        if recorder_io is None:
            return

        # If already recording or output_path is set, nothing to do
        if getattr(recorder_io, "recording", False) or getattr(
            recorder_io, "output_path", None
        ):
            return

        try:
            from livekit.agents import get_job_context

            job_ctx = get_job_context()
            if not job_ctx or not hasattr(job_ctx, "session_directory"):
                return

            output_path = Path(job_ctx.session_directory) / "audio.ogg"
            await recorder_io.start(output_path=output_path)
            logger.debug(
                "RecorderIO started by tracing helper for console mode: %s",
                output_path,
            )
        except Exception as e:
            logger.debug(
                f"Failed to start RecorderIO from tracing helper: {e}", exc_info=True
            )

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

    def _on_conversation_item_added(self, ev: Any) -> None:
        """Handle conversation_item_added event."""
        self._create_async_handler("conversation_item_added")(ev)

    def _on_agent_false_interruption(self, ev: Any) -> None:
        """Handle agent_false_interruption event."""
        self._create_async_handler("agent_false_interruption")(ev)

    def _on_function_tools_executed(self, ev: FunctionToolsExecutedEvent) -> None:
        """Handle function_tools_executed event - merge into pending generation span.

        LiveKit provides both function_calls and function_call_outputs in this event,
        so we use them directly instead of manually tracking.
        """

        async def _handle_function_tools() -> None:
            try:
                # If we have a pending generation span, update it with function data
                if self._pending_generation_span:
                    self._finalize_generation_span_with_functions(ev)
                else:
                    # No pending span, create a minimal event span
                    # (fallback, should rarely happen)
                    create_event_span("function_tools_executed", ev, manager=self)

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

    def _finalize_generation_span_with_functions(
        self, ev: Optional[FunctionToolsExecutedEvent] = None
    ) -> None:
        """Finalize the pending generation span with function call data and finish it.

        Args:
            ev: Optional FunctionToolsExecutedEvent containing function_calls and function_call_outputs.
                If provided, uses data from the event directly. Otherwise, span is finished without function data.
        """
        if not self._pending_generation_span:
            return

        try:
            from noveum_trace import get_client

            client = get_client()
            if not client:
                # Can't finish span without client, clear pending data anyway
                self._pending_generation_span = None
                return

            span = self._pending_generation_span

            # Extract function calls and outputs from event (LiveKit provides both)
            if ev and hasattr(ev, "function_calls") and ev.function_calls:
                serialized_calls = serialize_function_calls(ev.function_calls)
                if serialized_calls:
                    span.attributes["llm.function_calls.count"] = len(serialized_calls)
                    try:
                        span.attributes["llm.function_calls"] = json.dumps(
                            serialized_calls, default=str
                        )
                    except Exception:
                        pass

            if ev and hasattr(ev, "function_call_outputs") and ev.function_call_outputs:
                # Serialize function outputs
                serialized_outputs: list[dict[str, Any]] = []
                for output in ev.function_call_outputs:
                    if output is None:
                        continue
                    try:
                        output_dict: dict[str, Any] = {}
                        if hasattr(output, "name"):
                            output_dict["name"] = str(output.name)
                        if hasattr(output, "output"):
                            output_dict["output"] = str(output.output)
                        if hasattr(output, "is_error"):
                            output_dict["is_error"] = bool(output.is_error)
                        if output_dict:
                            serialized_outputs.append(output_dict)
                    except Exception:
                        continue

                if serialized_outputs:
                    span.attributes["llm.function_outputs.count"] = len(
                        serialized_outputs
                    )
                    try:
                        span.attributes["llm.function_outputs"] = json.dumps(
                            serialized_outputs, default=str
                        )
                    except Exception:
                        pass

            # Now finish the span with all merged data
            span.set_status(SpanStatus.OK)
            client.finish_span(span)

            # Clear pending span
            self._pending_generation_span = None

            logger.debug("Finalized generation span with function data")

        except Exception as e:
            logger.warning(f"Failed to finalize generation span: {e}", exc_info=True)

    def _on_metrics_collected(self, ev: MetricsCollectedEvent) -> None:
        """Handle metrics_collected event and extract LLM metrics."""

        async def _handle_with_llm_metrics() -> None:
            try:
                # Create the event span (standard handling)
                create_event_span("metrics_collected", ev, manager=self)

                # Extract LLM metrics if this is an llm_metrics type event
                metrics = getattr(ev, "metrics", None)
                if metrics is None:
                    return

                # Check if this is an LLM metrics event
                metrics_type = getattr(metrics, "type", None)
                if metrics_type != "llm_metrics":
                    return

                # Get speech_id to correlate with speech_created span
                speech_id = getattr(metrics, "speech_id", None)
                if not speech_id:
                    return

                # Extract LLM metrics
                llm_metrics: dict[str, Any] = {}

                # Token usage
                prompt_tokens = getattr(metrics, "prompt_tokens", None)
                completion_tokens = getattr(metrics, "completion_tokens", None)
                total_tokens = getattr(metrics, "total_tokens", None)

                if prompt_tokens is not None:
                    llm_metrics["llm.input_tokens"] = prompt_tokens
                if completion_tokens is not None:
                    llm_metrics["llm.output_tokens"] = completion_tokens
                if total_tokens is not None:
                    llm_metrics["llm.total_tokens"] = total_tokens

                # Model info from metadata
                metadata = getattr(metrics, "metadata", None)
                if metadata:
                    model_name = getattr(metadata, "model_name", None)
                    model_provider = getattr(metadata, "model_provider", None)
                    if model_name:
                        llm_metrics["llm.model"] = model_name
                    if model_provider:
                        llm_metrics["llm.provider"] = model_provider

                # Performance metrics
                ttft = getattr(metrics, "ttft", None)
                tokens_per_second = getattr(metrics, "tokens_per_second", None)
                duration = getattr(metrics, "duration", None)
                request_id = getattr(metrics, "request_id", None)
                cancelled = getattr(metrics, "cancelled", None)

                if ttft is not None:
                    llm_metrics["llm.time_to_first_token_ms"] = (
                        ttft * 1000
                    )  # Convert to ms
                if tokens_per_second is not None:
                    llm_metrics["llm.tokens_per_second"] = tokens_per_second
                if duration is not None:
                    llm_metrics["llm.latency_ms"] = duration * 1000  # Convert to ms
                if request_id is not None:
                    llm_metrics["llm.request_id"] = request_id
                if cancelled is not None:
                    llm_metrics["llm.cancelled"] = cancelled

                # Calculate cost using the same utility as LangChain
                if (
                    llm_metrics.get("llm.model")
                    and llm_metrics.get("llm.input_tokens") is not None
                ):
                    try:
                        from noveum_trace.utils.llm_utils import estimate_cost

                        cost_info = estimate_cost(
                            llm_metrics["llm.model"],
                            input_tokens=llm_metrics.get("llm.input_tokens", 0),
                            output_tokens=llm_metrics.get("llm.output_tokens", 0),
                        )
                        llm_metrics["llm.cost.input"] = cost_info.get("input_cost", 0)
                        llm_metrics["llm.cost.output"] = cost_info.get("output_cost", 0)
                        llm_metrics["llm.cost.total"] = cost_info.get("total_cost", 0)
                        llm_metrics["llm.cost.currency"] = cost_info.get(
                            "currency", "USD"
                        )
                    except Exception as cost_err:
                        logger.debug(f"Could not calculate LLM cost: {cost_err}")

                # Store metrics for merging with speech span
                if llm_metrics:
                    # If we already have metrics for this speech_id, merge them
                    # (there can be multiple LLM calls for follow-up tool calls)
                    if speech_id in self._pending_llm_metrics:
                        existing = self._pending_llm_metrics[speech_id]
                        existing_model = existing.get("llm.model")
                        new_model = llm_metrics.get("llm.model")
                        # Accumulate tokens
                        for key in [
                            "llm.input_tokens",
                            "llm.output_tokens",
                            "llm.total_tokens",
                        ]:
                            if key in llm_metrics and key in existing:
                                llm_metrics[key] += existing[key]
                        # Accumulate costs
                        for key in [
                            "llm.cost.input",
                            "llm.cost.output",
                            "llm.cost.total",
                        ]:
                            if key in llm_metrics and key in existing:
                                llm_metrics[key] += existing[key]
                        if existing_model and new_model and existing_model != new_model:
                            existing_models = existing.get("llm.models")
                            if existing_models:
                                if isinstance(existing_models, list):
                                    models = list(existing_models)
                                else:
                                    models = [existing_models]
                            else:
                                models = [existing_model]
                            models.append(new_model)
                            deduped_models = []
                            for model_name in models:
                                if model_name and model_name not in deduped_models:
                                    deduped_models.append(model_name)
                            llm_metrics["llm.models"] = deduped_models
                            # Preserve the original model for compatibility.
                            llm_metrics["llm.model"] = existing_model
                        elif existing.get("llm.models") and new_model:
                            models = existing["llm.models"]
                            if not isinstance(models, list):
                                models = [models]
                            if new_model not in models:
                                llm_metrics["llm.models"] = models + [new_model]
                        # Merge (new values override except for accumulated)
                        existing.update(llm_metrics)
                    else:
                        self._pending_llm_metrics[speech_id] = llm_metrics

                    logger.debug(
                        f"Stored LLM metrics for speech {speech_id}: "
                        f"tokens={llm_metrics.get('llm.total_tokens')}, "
                        f"model={llm_metrics.get('llm.model')}"
                    )

                    # Try to update the speech span if it already exists
                    if speech_id in self._speech_spans:
                        span = self._speech_spans[speech_id]
                        try:
                            span.attributes.update(llm_metrics)
                            logger.debug(
                                f"Updated speech span {span.span_id} with LLM metrics"
                            )
                        except Exception as update_err:
                            logger.debug(
                                f"Could not update speech span with LLM metrics: {update_err}"
                            )

            except Exception as e:
                logger.warning(
                    f"Error extracting LLM metrics from metrics_collected: {e}",
                    exc_info=True,
                )

        try:
            asyncio.create_task(_handle_with_llm_metrics())
        except Exception as e:
            logger.warning(
                f"Failed to create task for metrics_collected: {e}", exc_info=True
            )

    def _on_speech_created(self, ev: SpeechCreatedEvent) -> None:
        """Handle speech_created event."""
        self._create_async_handler("speech_created")(ev)

    def _on_error(self, ev: Any) -> None:
        """Handle error event."""
        self._create_async_handler("error")(ev)

    def _on_close(self, ev: CloseEvent) -> None:
        """Handle close event, add final data, and end trace."""
        logger.info(
            f"_on_close handler triggered: reason={getattr(ev, 'reason', 'unknown')}"
        )

        async def _handle_close() -> None:
            try:
                if self._trace:
                    # Ensure trace context is available in this task
                    set_current_trace(self._trace)
                    set_current_span(None)
                create_event_span("close", ev, manager=self)

                # Clean up pending speech spans (background tasks will handle their own cleanup,
                # but we clear the dict to prevent memory leaks)
                self._speech_spans.clear()

                # Clean up pending LLM metrics
                self._pending_llm_metrics.clear()

                # Flush any pending generation span before ending the trace
                # This ensures function call data is persisted even if no
                # function_tools_executed event was received
                self._finalize_generation_span_with_functions()

                # End trace
                if self._trace:
                    try:
                        from noveum_trace import get_client

                        client = get_client()

                        # Get conversation history from LiveKit's built-in ChatContext
                        history_dict = get_conversation_history_from_session(
                            self.session
                        )
                        if history_dict:
                            items = history_dict.get("items", [])
                            self._trace.set_attributes(
                                {
                                    "conversation.history.message_count": len(items),
                                }
                            )
                            try:
                                self._trace.set_attributes(
                                    {
                                        "conversation.history": json.dumps(
                                            history_dict, default=str
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
                                    "agent.available_tools.descriptions": [
                                        t.get("description", "")
                                        for t in self._available_tools
                                    ],
                                }
                            )

                        # Upload full conversation audio from LiveKit's RecorderIO
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
                        logger.info("Trace finished successfully in close handler")
                    except Exception as e:
                        logger.warning(
                            f"Failed to end trace on close: {e}", exc_info=True
                        )
            except Exception as e:
                logger.warning(f"Error in close handler: {e}", exc_info=True)

        # Try to run the close handler - use different strategies based on event loop state
        try:
            loop = asyncio.get_running_loop()
            # Event loop is running, create task and try to ensure it runs
            task = loop.create_task(_handle_close())
            # Add a callback to log when task completes
            task.add_done_callback(
                lambda t: logger.debug(
                    f"Close handler task completed: cancelled={t.cancelled()}, "
                    f"exception={t.exception() if not t.cancelled() else 'N/A'}"
                )
            )
        except RuntimeError:
            # No running event loop - try to run synchronously
            logger.debug("No running event loop, running close handler synchronously")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_handle_close())
                finally:
                    loop.close()
            except Exception as e:
                logger.warning(
                    f"Failed to run close handler synchronously: {e}", exc_info=True
                )

    async def _upload_full_conversation_audio(self) -> None:
        """Upload full conversation audio from LiveKit's RecorderIO.

        LiveKit's RecorderIO automatically records the entire conversation as a
        stereo OGG/Opus file (left channel = user audio, right channel = agent audio).
        This provides a properly synchronized recording without manual frame collection.
        """
        # Finalize the recorder before reading the audio file to ensure
        # the OGG encoding is fully flushed and complete
        recorder_io = getattr(self.session, "_recorder_io", None)
        if recorder_io is not None:
            try:
                # Check if recorder is already closed (various ways it might be indicated)
                is_closed = getattr(recorder_io, "closed", False) or getattr(
                    recorder_io, "_closed", False
                )
                if not is_closed and hasattr(recorder_io, "aclose"):
                    logger.debug("Finalizing RecorderIO before reading audio file")
                    await recorder_io.aclose()
                    logger.debug("RecorderIO finalized successfully")
            except Exception as e:
                logger.warning(
                    f"Failed to finalize RecorderIO: {e}. "
                    "Audio file may be incomplete or corrupted.",
                    exc_info=True,
                )

        # Get the recording path from LiveKit's RecorderIO
        audio_path = get_recorder_audio_path(self.session)
        if audio_path is None:
            # RecorderIO may have an output_path but the file isn't visible yet.
            recorder_io = getattr(self.session, "_recorder_io", None)
            recorder_path = getattr(recorder_io, "output_path", None)
            if recorder_path is not None:
                audio_path = (
                    recorder_path
                    if isinstance(recorder_path, Path)
                    else Path(recorder_path)
                )

        # Wait briefly for the recorder to flush the file to disk
        if audio_path is not None:
            for _ in range(25):  # ~5s max wait
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    break
                await asyncio.sleep(0.2)

        if audio_path is None or not audio_path.exists():
            logger.debug(
                "_upload_full_conversation_audio: No recording available. "
                "Ensure session.start(record=True) was called."
            )
            return

        if not self._trace:
            logger.warning(
                "_upload_full_conversation_audio: No trace available, skipping"
            )
            return

        try:
            trace = self._trace
            if trace.is_finished():
                logger.warning(
                    "_upload_full_conversation_audio: Trace already finished, skipping"
                )
                return

            audio_uuid = str(uuid.uuid4())

            logger.info(
                f"Uploading full conversation audio from RecorderIO: "
                f"path={audio_path}, uuid={audio_uuid}"
            )

            # Build attributes for the full conversation audio span
            # Include crucial details: system prompt, tools, and chat history
            attributes: dict[str, Any] = {
                "stt.audio_uuid": audio_uuid,
                "stt.audio_format": "ogg",
                "stt.audio_channels": "stereo",
                "stt.audio_channel_left": "user",
                "stt.audio_channel_right": "agent",
                "stt.audio_source": "livekit_recorder_io",
                "stt.audio_description": "Full conversation - stereo recording (left=user, right=agent)",
            }

            # Add available tools
            if self._available_tools:
                tool_attrs = serialize_tools_for_attributes(self._available_tools)
                attributes.update(tool_attrs)

            # Add conversation history from LiveKit's built-in ChatContext
            history_dict = get_conversation_history_from_session(self.session)
            if history_dict:
                items = history_dict.get("items", [])
                # Bound to prevent oversized attributes
                if len(items) > MAX_CONVERSATION_HISTORY:
                    items = items[-MAX_CONVERSATION_HISTORY:]
                    history_dict = {"items": items}
                attributes["llm.conversation.message_count"] = len(items)
                try:
                    attributes["llm.conversation.history"] = json.dumps(
                        history_dict, default=str
                    )
                except Exception:
                    pass

            # Try to get system prompt synchronously if agent_activity is available
            system_prompt = None
            try:
                if hasattr(self.session, "agent_activity"):
                    agent_activity = self.session.agent_activity
                    if agent_activity and hasattr(agent_activity, "_agent"):
                        agent = agent_activity._agent
                        if hasattr(agent, "instructions") and agent.instructions:
                            system_prompt = agent.instructions
                        elif hasattr(agent, "_instructions") and agent._instructions:
                            system_prompt = agent._instructions
            except Exception:
                pass  # Will try via background task if not available

            if system_prompt:
                attributes["llm.system_prompt"] = system_prompt

            # Create span for the full conversation audio
            # UI expects this name for full-conversation playback
            span = trace.create_span(
                name="livekit.full_conversation",
                attributes=attributes,
            )

            # If system prompt not already added, start background task to update span
            if "llm.system_prompt" not in attributes:
                from noveum_trace.integrations.livekit.livekit_utils import (
                    _update_span_with_system_prompt,
                )

                asyncio.create_task(_update_span_with_system_prompt(span, self))

            # Track upload success to set appropriate status
            upload_success = False

            try:
                # Upload the OGG audio file directly
                upload_success = upload_audio_file(
                    audio_path,
                    audio_uuid,
                    "stt",  # Use "stt" type so UI recognizes it
                    span.trace_id,
                    span.span_id,
                    content_type="audio/ogg",
                )

                if upload_success:
                    logger.info(
                        f"Successfully uploaded conversation audio: {audio_path}"
                    )
                else:
                    logger.warning(f"Failed to upload conversation audio: {audio_path}")

            except Exception as e:
                # Set error status on exception
                span.set_status(SpanStatus.ERROR, str(e))
                logger.warning(
                    f"Exception while uploading full conversation audio: {e}",
                    exc_info=True,
                )
                # Re-raise to let outer handler log it
                raise
            finally:
                # Always finish the span, setting status based on upload result
                if upload_success:
                    span.set_status(SpanStatus.OK)
                else:
                    # Only set error status if not already set (i.e., upload returned False)
                    if span.status != SpanStatus.ERROR:
                        span.set_status(SpanStatus.ERROR, "Upload failed")
                trace.finish_span(span.span_id)

        except Exception as e:
            logger.warning(
                f"Failed to upload full conversation audio: {e}", exc_info=True
            )

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

                # Try to get system prompt synchronously if agent_activity is available
                system_prompt = None
                try:
                    if hasattr(self.session, "agent_activity"):
                        agent_activity = self.session.agent_activity
                        if agent_activity and hasattr(agent_activity, "_agent"):
                            agent = agent_activity._agent
                            if hasattr(agent, "instructions") and agent.instructions:
                                system_prompt = agent.instructions
                            elif (
                                hasattr(agent, "_instructions") and agent._instructions
                            ):
                                system_prompt = agent._instructions
                except Exception:
                    pass  # Will try via background task if not available

                if system_prompt:
                    attributes["llm.system_prompt"] = system_prompt

                # Add conversation history from LiveKit's built-in ChatContext
                history_dict = get_conversation_history_from_session(self.session)
                if history_dict:
                    items = history_dict.get("items", [])
                    # Bound to prevent oversized attributes
                    if len(items) > MAX_CONVERSATION_HISTORY:
                        items = items[-MAX_CONVERSATION_HISTORY:]
                        history_dict = {"items": items}
                    attributes["llm.conversation.message_count"] = len(items)
                    try:
                        attributes["llm.conversation.history"] = json.dumps(
                            history_dict, default=str
                        )
                    except Exception:
                        pass

                # Add constants metadata
                attributes["metadata"] = create_constants_metadata()

                # Finalize any existing pending span before creating a new one
                # This prevents span leaks when generation events fire in quick succession
                if self._pending_generation_span is not None:
                    self._finalize_generation_span_with_functions()

                # Create span (do NOT finish yet - wait for function call data)
                span = client.start_span(
                    name="livekit.realtime.generation_created",
                    attributes=attributes,
                )

                # Store as pending for function call merging
                # The span will be finished in _finalize_generation_span_with_functions
                # after function calls/outputs are merged
                self._pending_generation_span = span

                # If system prompt not already added, start background task to update span
                # (waiting for agent_activity to become available)
                if "llm.system_prompt" not in attributes:
                    from noveum_trace.integrations.livekit.livekit_utils import (
                        _update_span_with_system_prompt,
                    )

                    asyncio.create_task(_update_span_with_system_prompt(span, self))

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
        if self._original_start is not None:
            self.session.start = self._original_start


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
    - Upload full conversation audio from LiveKit's RecorderIO (when record=True)
    - Export conversation history from LiveKit's ChatContext

    Note:
        For full conversation audio, ensure session.start(record=True) is called.
        LiveKit's RecorderIO automatically records as a stereo OGG file
        (left channel = user, right channel = agent).

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
        >>> session = AgentSession(stt=stt, tts=tts, llm=llm)
        >>> setup_livekit_tracing(session)
        >>>
        >>> agent = Agent(instructions="You are helpful.")
        >>> await session.start(agent, record=True)  # Enables audio recording
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
