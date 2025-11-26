"""
Utility functions for LiveKit STT/TTS integration.

This module provides helper functions for audio handling, file management,
and context extraction for LiveKit integration with noveum-trace.
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)

try:
    from livekit import rtc
    from livekit.agents.utils import AudioBuffer

    LIVEKIT_AVAILABLE = True
except ImportError as e:
    LIVEKIT_AVAILABLE = False
    logger.error(
        "LiveKit is not importable. LiveKit utility functions will not work properly. "
        "Install it with: pip install livekit livekit-agents",
        exc_info=e,
    )
    # Define a dummy type for when LiveKit is not available
    AudioBuffer = Any  # type: ignore


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


def save_audio_buffer(buffer: "AudioBuffer", output_path: Path) -> None:
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
        return 0.0

    total_duration_sec = sum(frame.duration for frame in frames)
    return total_duration_sec * 1000.0


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


def _safe_str(obj: Any, default: str = "unknown") -> str:
    """
    Safely convert an object to string, filtering out mocks.

    Args:
        obj: Object to convert
        default: Default value if object is mock or None

    Returns:
        String representation or default
    """
    if obj is None:
        return default

    str_val = str(obj)
    if _is_mock_object(obj):
        return default

    return str_val


def extract_job_context(ctx: Any) -> dict[str, Any]:
    """
    Extract serializable fields from LiveKit JobContext.

    Filters out mock objects to prevent "<Mock object>" strings in traces.

    Args:
        ctx: LiveKit JobContext or similar object

    Returns:
        Dictionary of serializable context fields
    """
    context: dict[str, Any] = {}

    # Job info
    if hasattr(ctx, "job") and ctx.job and not _is_mock_object(ctx.job):
        if hasattr(ctx.job, "id"):
            job_id = _safe_str(ctx.job.id)
            if job_id != "unknown":
                context["job_id"] = job_id

        if (
            hasattr(ctx.job, "room")
            and ctx.job.room
            and not _is_mock_object(ctx.job.room)
        ):
            if hasattr(ctx.job.room, "sid"):
                room_sid = _safe_str(ctx.job.room.sid)
                if room_sid != "unknown":
                    context["job_room_sid"] = room_sid
            if hasattr(ctx.job.room, "name"):
                room_name = _safe_str(ctx.job.room.name)
                if room_name != "unknown":
                    context["job_room_name"] = room_name

    # Room info
    if hasattr(ctx, "room") and ctx.room and not _is_mock_object(ctx.room):
        if hasattr(ctx.room, "name"):
            room_name = _safe_str(ctx.room.name)
            if room_name != "unknown":
                context["room_name"] = room_name
        if hasattr(ctx.room, "sid"):
            room_sid = _safe_str(ctx.room.sid)
            if room_sid != "unknown":
                context["room_sid"] = room_sid

    # Agent info
    if hasattr(ctx, "agent") and ctx.agent and not _is_mock_object(ctx.agent):
        if hasattr(ctx.agent, "id"):
            agent_id = _safe_str(ctx.agent.id)
            if agent_id != "unknown":
                context["agent_id"] = agent_id

    # Worker info
    if hasattr(ctx, "worker_id") and not _is_mock_object(ctx.worker_id):
        worker_id = _safe_str(ctx.worker_id)
        if worker_id != "unknown":
            context["worker_id"] = worker_id

    # Participant info
    if (
        hasattr(ctx, "participant")
        and ctx.participant
        and not _is_mock_object(ctx.participant)
    ):
        if hasattr(ctx.participant, "identity"):
            identity = _safe_str(ctx.participant.identity)
            if identity != "unknown":
                context["participant_identity"] = identity
        if hasattr(ctx.participant, "sid"):
            sid = _safe_str(ctx.participant.sid)
            if sid != "unknown":
                context["participant_sid"] = sid

    return context


def create_span_attributes(
    provider: str,
    model: str,
    operation_type: str,
    audio_file: str,
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
        audio_file: Filename of saved audio
        audio_duration_ms: Audio duration in milliseconds
        job_context: Job context dictionary
        **extra_attributes: Additional operation-specific attributes

    Returns:
        Dictionary of span attributes
    """
    attributes = {
        f"{operation_type}.provider": provider,
        f"{operation_type}.model": model,
        f"{operation_type}.audio_file": audio_file,
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

    # Add extra attributes
    attributes.update(extra_attributes)

    return attributes
