"""
Utility functions for LiveKit STT/TTS integration.

This module provides helper functions for audio handling, file management,
and context extraction for LiveKit integration with noveum-trace.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from noveum_trace.core.context import get_current_span, get_current_trace
from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.livekit.livekit_constants import (
    AUDIO_DURATION_MS_DEFAULT_VALUE,
    STT_CONFIDENCE_DEFAULT_VALUE,
    STT_END_TIME_DEFAULT_VALUE,
    STT_IS_PRIMARY_SPEAKER_DEFAULT_VALUE,
    STT_LANGUAGE_DEFAULT_VALUE,
    STT_SPEAKER_ID_DEFAULT_VALUE,
    STT_START_TIME_DEFAULT_VALUE,
    STT_TRANSCRIPT_DEFAULT_VALUE,
    SYSTEM_PROMPT_CHECK_INTERVAL_SECONDS,
    SYSTEM_PROMPT_MAX_WAIT_SECONDS,
    TTS_DELTA_TEXT_DEFAULT_VALUE,
    TTS_INPUT_TEXT_DEFAULT_VALUE,
    TTS_NUM_CHANNELS_DEFAULT_VALUE,
    TTS_REQUEST_ID_DEFAULT_VALUE,
    TTS_SAMPLE_RATE_DEFAULT_VALUE,
    TTS_SEGMENT_ID_DEFAULT_VALUE,
)

if TYPE_CHECKING:
    from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)

try:
    from livekit import rtc
    from livekit.agents.utils import AudioBuffer

    LIVEKIT_AVAILABLE = True
except ImportError as e:
    LIVEKIT_AVAILABLE = False
    logger.debug(
        "LiveKit is not importable. LiveKit utility functions will not work properly. "
        "Install it with: pip install livekit livekit-agents",
        exc_info=e,
    )
    # Define a dummy type for when LiveKit is not available
    AudioBuffer = Any


def create_constants_metadata() -> dict[str, Any]:
    """
    Create metadata dictionary containing all LiveKit constants as defaults.

    Returns:
        Dictionary with structure: {"config": {"defaults": {...all constants...}}}
    """
    return {
        "config": {
            "defaults": {
                # STT constants
                "STT_TRANSCRIPT_DEFAULT_VALUE": STT_TRANSCRIPT_DEFAULT_VALUE,
                "STT_CONFIDENCE_DEFAULT_VALUE": STT_CONFIDENCE_DEFAULT_VALUE,
                "STT_LANGUAGE_DEFAULT_VALUE": STT_LANGUAGE_DEFAULT_VALUE,
                "STT_START_TIME_DEFAULT_VALUE": STT_START_TIME_DEFAULT_VALUE,
                "STT_END_TIME_DEFAULT_VALUE": STT_END_TIME_DEFAULT_VALUE,
                "STT_SPEAKER_ID_DEFAULT_VALUE": STT_SPEAKER_ID_DEFAULT_VALUE,
                "STT_IS_PRIMARY_SPEAKER_DEFAULT_VALUE": STT_IS_PRIMARY_SPEAKER_DEFAULT_VALUE,
                # TTS constants
                "TTS_INPUT_TEXT_DEFAULT_VALUE": TTS_INPUT_TEXT_DEFAULT_VALUE,
                "TTS_SEGMENT_ID_DEFAULT_VALUE": TTS_SEGMENT_ID_DEFAULT_VALUE,
                "TTS_REQUEST_ID_DEFAULT_VALUE": TTS_REQUEST_ID_DEFAULT_VALUE,
                "TTS_DELTA_TEXT_DEFAULT_VALUE": TTS_DELTA_TEXT_DEFAULT_VALUE,
                "TTS_SAMPLE_RATE_DEFAULT_VALUE": TTS_SAMPLE_RATE_DEFAULT_VALUE,
                "TTS_NUM_CHANNELS_DEFAULT_VALUE": TTS_NUM_CHANNELS_DEFAULT_VALUE,
                # Audio duration constants
                "AUDIO_DURATION_MS_DEFAULT_VALUE": AUDIO_DURATION_MS_DEFAULT_VALUE,
                # System prompt timing constants
                "SYSTEM_PROMPT_MAX_WAIT_SECONDS": SYSTEM_PROMPT_MAX_WAIT_SECONDS,
                "SYSTEM_PROMPT_CHECK_INTERVAL_SECONDS": SYSTEM_PROMPT_CHECK_INTERVAL_SECONDS,
            }
        }
    }


def save_audio_frames(frames: list[Any], output_path: Path) -> None:
    """
    Combine audio frames and save as WAV file.

    Args:
        frames: List of rtc.AudioFrame objects
        output_path: Path where the WAV file will be saved

    Raises:
        IOError: If file cannot be written
    """
    if not LIVEKIT_AVAILABLE:
        logger.error(
            "Cannot save audio frames: LiveKit is not available. "
            "Install it with: pip install livekit"
        )
        return

    if not frames:
        # Create empty WAV file for empty frames
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"")
        return

    # Combine frames using LiveKit's utility
    combined = rtc.combine_audio_frames(frames)

    # Convert to WAV bytes
    wav_bytes = combined.to_wav_bytes()

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    output_path.write_bytes(wav_bytes)


def save_audio_buffer(buffer: AudioBuffer, output_path: Path) -> None:
    """
    Save AudioBuffer (list of frames) as WAV file.

    Args:
        buffer: AudioBuffer containing audio frames
        output_path: Path where the WAV file will be saved

    Raises:
        ImportError: If livekit package is not installed
        IOError: If file cannot be written
    """
    # AudioBuffer is essentially a list of AudioFrame objects
    save_audio_frames(list(buffer), output_path)


def calculate_audio_duration_ms(frames: list[Any]) -> float:
    """
    Calculate total duration of audio frames in milliseconds.

    Args:
        frames: List of rtc.AudioFrame objects

    Returns:
        Total duration in milliseconds
    """
    if not frames:
        return AUDIO_DURATION_MS_DEFAULT_VALUE

    total_duration_sec = sum(frame.duration for frame in frames)
    return total_duration_sec * 1000.0


def upload_audio_frames(
    frames: list[Any],
    audio_uuid: str,
    audio_type: str,
    trace_id: str,
    span_id: str,
) -> bool:
    """
    Upload audio frames to Noveum platform.

    This utility function handles the complete audio upload workflow:
    - Converts audio frames to WAV bytes
    - Gets the current client
    - Calculates audio duration
    - Uploads audio with metadata

    Args:
        frames: List of rtc.AudioFrame objects
        audio_uuid: Unique identifier for the audio file
        audio_type: Type of audio ('stt' or 'tts')
        trace_id: Trace ID to associate audio with
        span_id: Span ID to associate audio with

    Returns:
        True if upload was successful, False otherwise
    """
    try:
        if not LIVEKIT_AVAILABLE:
            logger.debug("LiveKit not available, skipping audio upload")
            return False

        if not frames:
            logger.debug("No frames to upload")
            return False

        # Convert frames to WAV bytes
        combined = rtc.combine_audio_frames(frames)
        audio_bytes = combined.to_wav_bytes()

        # Get client
        from noveum_trace import get_client

        client = get_client()
        if not client:
            logger.info("No client available, skipping audio upload")
            return False

        # Calculate duration
        duration_ms = calculate_audio_duration_ms(frames)

        # Prepare metadata
        metadata = {
            "duration_ms": duration_ms,
            "format": "wav",
            "type": audio_type,
        }

        # Export audio (non-blocking, queued)
        client.export_audio(
            audio_data=audio_bytes,
            trace_id=trace_id,
            span_id=span_id,
            audio_uuid=audio_uuid,
            metadata=metadata,
        )

        logger.debug(f"Queued audio upload: {audio_uuid}")
        return True

    except Exception as e:  # noqa: S110 - broad exception for graceful degradation
        logger.warning(f"Failed to export audio {audio_uuid}: {e}", exc_info=True)
        return False


def ensure_audio_directory(session_id: str, base_dir: Optional[Path] = None) -> Path:
    """
    Ensure audio storage directory exists for a session.

    Args:
        session_id: Session identifier
        base_dir: Base directory for audio files (defaults to 'audio_files' in current dir)

    Returns:
        Path to the session's audio directory
    """
    if base_dir is None:
        base_dir = Path("audio_files")

    audio_dir = base_dir / session_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


def generate_audio_filename(
    prefix: str, counter: int, timestamp: Optional[int] = None
) -> str:
    """
    Generate a standardized audio filename.

    Args:
        prefix: File prefix (e.g., 'stt' or 'tts')
        counter: Sequence counter
        timestamp: Timestamp in milliseconds (defaults to current time)

    Returns:
        Formatted filename like 'stt_0001_1732386400000.wav'
    """
    if timestamp is None:
        timestamp = int(time.time() * 1000)

    return f"{prefix}_{counter:04d}_{timestamp}.wav"


def _is_mock_object(obj: Any) -> bool:
    """Check if an object is a mock object."""
    obj_str = str(obj)
    return (
        "<Mock" in obj_str
        or "<MagicMock" in obj_str
        or "<AsyncMock" in obj_str
        or obj_str.startswith("mock.")
    )


async def _safe_str(obj: Any, default: str = "unknown") -> str:
    """
    Safely convert an object to string, filtering out mocks and handling coroutines.

    Supports both sync properties (old LiveKit SDK) and async properties (new LiveKit SDK v2+).

    Args:
        obj: Object to convert (can be a coroutine or regular value)
        default: Default value if object is mock or None

    Returns:
        String representation or default
    """
    if obj is None:
        return default

    # Handle coroutines (async properties from LiveKit SDK v2+)
    # Await them to get the actual value
    if inspect.iscoroutine(obj):
        obj = await obj

    str_val = str(obj)
    if _is_mock_object(obj):
        return default

    return str_val


async def extract_job_context(ctx: Any) -> dict[str, Any]:
    """
    Extract serializable fields from LiveKit JobContext.

    Filters out mock objects to prevent "<Mock object>" strings in traces.
    Handles both sync properties (old LiveKit SDK) and async properties (new LiveKit SDK v2+).

    Args:
        ctx: LiveKit JobContext or similar object

    Returns:
        Dictionary of serializable context fields
    """
    context: dict[str, Any] = {}

    # Job info
    if hasattr(ctx, "job") and ctx.job and not _is_mock_object(ctx.job):
        if hasattr(ctx.job, "id"):
            job_id = await _safe_str(ctx.job.id)
            if job_id != "unknown":
                context["job_id"] = job_id

        if (
            hasattr(ctx.job, "room")
            and ctx.job.room
            and not _is_mock_object(ctx.job.room)
        ):
            if hasattr(ctx.job.room, "sid"):
                room_sid = await _safe_str(ctx.job.room.sid)
                if room_sid != "unknown":
                    context["job_room_sid"] = room_sid
            if hasattr(ctx.job.room, "name"):
                room_name = await _safe_str(ctx.job.room.name)
                if room_name != "unknown":
                    context["job_room_name"] = room_name

    # Room info
    if hasattr(ctx, "room") and ctx.room and not _is_mock_object(ctx.room):
        if hasattr(ctx.room, "name"):
            room_name = await _safe_str(ctx.room.name)
            if room_name != "unknown":
                context["room_name"] = room_name
        if hasattr(ctx.room, "sid"):
            room_sid = await _safe_str(ctx.room.sid)
            if room_sid != "unknown":
                context["room_sid"] = room_sid

    # Agent info
    if hasattr(ctx, "agent") and ctx.agent and not _is_mock_object(ctx.agent):
        if hasattr(ctx.agent, "id"):
            agent_id = await _safe_str(ctx.agent.id)
            if agent_id != "unknown":
                context["agent_id"] = agent_id

    # Worker info
    if hasattr(ctx, "worker_id") and not _is_mock_object(ctx.worker_id):
        worker_id = await _safe_str(ctx.worker_id)
        if worker_id != "unknown":
            context["worker_id"] = worker_id

    # Participant info
    if (
        hasattr(ctx, "participant")
        and ctx.participant
        and not _is_mock_object(ctx.participant)
    ):
        if hasattr(ctx.participant, "identity"):
            identity = await _safe_str(ctx.participant.identity)
            if identity != "unknown":
                context["participant_identity"] = identity
        if hasattr(ctx.participant, "sid"):
            sid = await _safe_str(ctx.participant.sid)
            if sid != "unknown":
                context["participant_sid"] = sid

    return context


def create_span_attributes(
    provider: str,
    model: str,
    operation_type: str,
    audio_uuid: str,
    audio_duration_ms: float,
    job_context: dict[str, Any],
    **extra_attributes: Any,
) -> dict[str, Any]:
    """
    Create standardized span attributes for STT/TTS operations.

    Args:
        provider: Provider name (e.g., 'deepgram', 'cartesia')
        model: Model identifier
        operation_type: 'stt' or 'tts'
        audio_uuid: UUID of the audio file
        audio_duration_ms: Audio duration in milliseconds
        job_context: Job context dictionary
        **extra_attributes: Additional operation-specific attributes

    Returns:
        Dictionary of span attributes
    """
    attributes = {
        f"{operation_type}.provider": provider,
        f"{operation_type}.model": model,
        f"{operation_type}.audio_uuid": audio_uuid,
        f"{operation_type}.audio_duration_ms": audio_duration_ms,
    }

    # Add job context with 'job.' prefix
    for key, value in job_context.items():
        # If key already has 'job.' prefix with dot, use as-is
        if key.startswith("job."):
            attributes[key] = value
        # If key has 'job_' prefix with underscore, convert to 'job.'
        elif key.startswith("job_"):
            attributes[f"job.{key[4:]}"] = value
        # Otherwise, add 'job.' prefix
        else:
            attributes[f"job.{key}"] = value

    # Add constants metadata
    constants_metadata = create_constants_metadata()
    attributes["metadata"] = constants_metadata

    # Add extra attributes
    attributes.update(extra_attributes)

    return attributes


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
            data = {k: v for k, v in event.__dict__.items() if not k.startswith("_")}
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
                    text_content = "\n".join(text_parts) if text_parts else None
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
                        str(item.arguments) if hasattr(item, "arguments") else None
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
                        bool(item.is_error) if hasattr(item, "is_error") else False
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


async def update_speech_span_with_chat_items(
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


async def _update_span_with_system_prompt(
    span: Any,
    manager: Any,
    max_wait_seconds: float = SYSTEM_PROMPT_MAX_WAIT_SECONDS,
    check_interval: float = SYSTEM_PROMPT_CHECK_INTERVAL_SECONDS,
) -> None:
    """
    Wait for agent_activity to become available, then update span with system prompt.

    Args:
        span: Span to update
        manager: _LiveKitTracingManager instance
        max_wait_seconds: Maximum time to wait for agent_activity
        check_interval: Interval between checks
    """
    try:
        start_time = asyncio.get_event_loop().time()

        # Wait for agent_activity to become available
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_wait_seconds:
                return

            # Check if agent_activity is available (try both property and attribute)
            agent_activity = None

            # Try property first
            if hasattr(manager.session, "agent_activity"):
                try:
                    agent_activity = manager.session.agent_activity
                except Exception as e:
                    logger.debug(
                        f"Failed to access agent_activity property: {e}", exc_info=True
                    )

            # Fallback to checking _activity attribute directly
            if not agent_activity and hasattr(manager.session, "_activity"):
                try:
                    agent_activity = manager.session._activity
                except Exception as e:
                    logger.debug(
                        f"Failed to access _activity attribute: {e}", exc_info=True
                    )

            if agent_activity:
                if hasattr(agent_activity, "_agent") and agent_activity._agent:
                    agent = agent_activity._agent

                    # Extract system prompt from agent.instructions (most reliable source)
                    system_prompt = None
                    if hasattr(agent, "instructions") and agent.instructions:
                        system_prompt = agent.instructions
                    elif hasattr(agent, "_instructions") and agent._instructions:
                        system_prompt = agent._instructions

                    if system_prompt:
                        # Directly modify span.attributes (bypassing set_attribute since span may be finished)
                        # This follows the same pattern as update_speech_span_with_chat_items
                        span.attributes["llm.system_prompt"] = system_prompt
                        return

            # Wait before next check
            await asyncio.sleep(check_interval)

    except Exception as e:
        logger.debug(f"Failed to update span with system prompt: {e}", exc_info=True)


def create_event_span(
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

        # Add constants metadata
        constants_metadata = create_constants_metadata()
        attributes["metadata"] = constants_metadata

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
        # Check event_type == "error" or hasattr(event_data, "error") to avoid
        # referencing ErrorEvent when LIVEKIT_AVAILABLE is False
        if event_type == "error" or (hasattr(event_data, "error") and event_data.error):
            span.set_status(
                SpanStatus.ERROR,
                (
                    str(event_data.error)
                    if hasattr(event_data, "error")
                    else "Error occurred"
                ),
            )

        # Finish span immediately (events are instantaneous)
        # Note: We don't need to restore context for metrics_collected since we never set it
        client.finish_span(span)

        # For speech_created events, start background task to update span with system prompt
        # (waiting for agent_activity to become available)
        if manager and event_type == "speech_created":
            # Check if we already have system prompt in attributes
            if "llm.system_prompt" not in attributes:
                # Start background task to wait for agent_activity and update span
                asyncio.create_task(_update_span_with_system_prompt(span, manager))

        return span

    except Exception as e:
        logger.warning(
            f"Failed to create span for event {event_type}: {e}", exc_info=True
        )
        return None
