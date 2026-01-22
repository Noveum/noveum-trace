"""
LiveKit STT integration for noveum-trace.

This module provides wrapper classes that automatically trace LiveKit STT
operations, capturing audio files and metadata as span attributes.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional, Union

from noveum_trace.core.context import get_current_trace
from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.livekit.livekit_constants import (
    STT_CONFIDENCE_DEFAULT_VALUE,
    STT_END_TIME_DEFAULT_VALUE,
    STT_IS_PRIMARY_SPEAKER_DEFAULT_VALUE,
    STT_LANGUAGE_DEFAULT_VALUE,
    STT_SPEAKER_ID_DEFAULT_VALUE,
    STT_START_TIME_DEFAULT_VALUE,
    STT_TRANSCRIPT_DEFAULT_VALUE,
)
from noveum_trace.integrations.livekit.livekit_utils import (
    calculate_audio_duration_ms,
    create_span_attributes,
    upload_audio_frames,
)

logger = logging.getLogger(__name__)

try:
    from livekit.agents.stt import SpeechEvent, SpeechEventType, STTCapabilities
    from livekit.agents.utils import AudioBuffer

    LIVEKIT_AVAILABLE = True
except ImportError as e:
    LIVEKIT_AVAILABLE = False
    logger.debug(
        "LiveKit is not importable. LiveKit STT integration features will not be available. "
        "Install it with: pip install livekit livekit-agents",
        exc_info=e,
    )


class LiveKitSTTWrapper:
    """
    Wrapper for LiveKit STT providers that automatically creates spans for transcription.

    This wrapper captures audio frames, saves them to disk, and creates spans with
    metadata for each transcription operation (both streaming and batch modes).

    Example:
        >>> import noveum_trace
        >>> from livekit.plugins import deepgram
        >>> from noveum_trace.integrations.livekit import LiveKitSTTWrapper
        >>>
        >>> # Initialize noveum-trace (done elsewhere)
        >>> noveum_trace.init(project="livekit-agents")
        >>>
        >>> # Wrap STT provider
        >>> base_stt = deepgram.STT(...)
        >>> traced_stt = LiveKitSTTWrapper(
        ...     stt=base_stt,
        ...     session_id="session_123",
        ...     job_context={"job_id": "job_abc"}
        ... )
        >>>
        >>> # Use in streaming mode
        >>> stream = traced_stt.stream()
        >>> async for event in stream:
        ...     if event.type == SpeechEventType.FINAL_TRANSCRIPT:
        ...         print(event.alternatives[0].text)
    """

    def __init__(
        self,
        stt: Any,  # noqa: F811 - parameter shadows import
        session_id: str,
        job_context: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize STT wrapper.

        Args:
            stt: Base LiveKit STT provider instance
            session_id: Session identifier for organizing audio files
            job_context: Dictionary of job context information to attach to spans
        """
        # Always initialize fields so wrapper is safe to use when LiveKit is unavailable
        self._base_stt = stt
        self._session_id = session_id
        self._job_context = job_context or {}
        self._counter_ref = [0]  # Mutable reference for sharing with streams

        if not LIVEKIT_AVAILABLE:
            logger.error(
                "Cannot initialize LiveKitSTTWrapper: LiveKit is not available. "
                "Install it with: pip install livekit livekit-agents"
            )
            return

    @property
    def capabilities(self) -> STTCapabilities:
        """Get STT capabilities from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_stt is None:
            raise RuntimeError(
                "LiveKit is not available. Cannot access capabilities. "
                "Install it with: pip install livekit livekit-agents"
            )
        return self._base_stt.capabilities

    @property
    def model(self) -> str:
        """Get model name from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_stt is None:
            return "unknown"
        return getattr(self._base_stt, "model", "unknown")

    @property
    def provider(self) -> str:
        """Get provider name from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_stt is None:
            return "unknown"
        return getattr(self._base_stt, "provider", "unknown")

    @property
    def label(self) -> str:
        """Get label from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_stt is None:
            return "LiveKitSTTWrapper"
        return getattr(self._base_stt, "label", self._base_stt.__class__.__name__)

    async def _recognize_impl(self, buffer: AudioBuffer, **kwargs: Any) -> SpeechEvent:
        """
        Batch recognition implementation with tracing.

        Args:
            buffer: Audio buffer to recognize
            **kwargs: Additional arguments passed to base STT

        Returns:
            SpeechEvent with recognition results
        """
        # Increment counter
        self._counter_ref[0] += 1

        # Call base STT (access to protected member is intentional for wrapping)
        event = await self._base_stt._recognize_impl(buffer, **kwargs)  # noqa: SLF001

        # Calculate audio duration
        duration_ms = calculate_audio_duration_ms(list(buffer))

        # Create span if trace exists
        trace = get_current_trace()
        if trace:
            from noveum_trace import get_client

            try:
                client = get_client()

                # Generate audio UUID
                audio_uuid = str(uuid.uuid4())

                # Get transcript text and additional attributes
                transcript = STT_TRANSCRIPT_DEFAULT_VALUE
                confidence = STT_CONFIDENCE_DEFAULT_VALUE
                language = STT_LANGUAGE_DEFAULT_VALUE
                start_time = STT_START_TIME_DEFAULT_VALUE
                end_time = STT_END_TIME_DEFAULT_VALUE
                speaker_id = STT_SPEAKER_ID_DEFAULT_VALUE
                is_primary_speaker = STT_IS_PRIMARY_SPEAKER_DEFAULT_VALUE

                if event.alternatives and len(event.alternatives) > 0:
                    first_alternative = event.alternatives[0]
                    transcript = first_alternative.text
                    confidence = first_alternative.confidence
                    language = first_alternative.language
                    start_time = first_alternative.start_time
                    end_time = first_alternative.end_time
                    speaker_id = first_alternative.speaker_id
                    is_primary_speaker = first_alternative.is_primary_speaker

                # Build additional attributes
                additional_attrs = {
                    "stt.transcript": transcript,
                    "stt.confidence": confidence,
                    "stt.is_final": True,
                    "stt.mode": "batch",
                    "stt.event_type": event.type.value,
                }

                if event.request_id:
                    additional_attrs["stt.request_id"] = event.request_id

                if language:
                    additional_attrs["stt.language"] = language

                if start_time is not None:
                    additional_attrs["stt.start_time"] = start_time

                if end_time is not None:
                    additional_attrs["stt.end_time"] = end_time

                if speaker_id is not None:
                    additional_attrs["stt.speaker_id"] = speaker_id

                if is_primary_speaker is not None:
                    additional_attrs["stt.is_primary_speaker"] = is_primary_speaker

                if event.recognition_usage and event.recognition_usage.audio_duration:
                    additional_attrs["stt.recognition_usage.audio_duration"] = (
                        event.recognition_usage.audio_duration
                    )

                # Create span attributes
                attributes = create_span_attributes(
                    provider=self.provider,
                    model=self.model,
                    operation_type="stt",
                    audio_uuid=audio_uuid,
                    audio_duration_ms=duration_ms,
                    job_context=self._job_context,
                    **additional_attrs,
                )

                # Create and finish span
                span = client.start_span(name="stt.recognize", attributes=attributes)

                # Upload audio frames with explicit span context
                upload_audio_frames(
                    list(buffer), audio_uuid, "stt", span.trace_id, span.span_id
                )

                span.set_status(SpanStatus.OK)
                client.finish_span(span)

            except (
                Exception
            ) as e:  # noqa: S110 - broad exception for graceful degradation
                # Gracefully handle span creation errors
                logger.warning(
                    f"Failed to create span for STT recognition: {e}", exc_info=True
                )

        return event

    async def recognize(self, buffer: AudioBuffer, **kwargs: Any) -> SpeechEvent:
        """
        Public recognition API.

        Args:
            buffer: Audio buffer to recognize
            **kwargs: Additional arguments

        Returns:
            SpeechEvent with recognition results
        """
        return await self._recognize_impl(buffer, **kwargs)

    def stream(self, **kwargs: Any) -> _WrappedSpeechStream:
        """
        Create a streaming recognition interface.

        Args:
            **kwargs: Additional arguments passed to base STT

        Returns:
            Wrapped speech stream
        """
        base_stream = self._base_stt.stream(**kwargs)
        return _WrappedSpeechStream(
            base_stream=base_stream,
            session_id=self._session_id,
            job_context=self._job_context,
            provider=self.provider,
            model=self.model,
            counter_ref=self._counter_ref,
        )

    async def aclose(self) -> None:
        """Close the STT provider."""
        if hasattr(self._base_stt, "aclose"):
            await self._base_stt.aclose()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base STT."""
        return getattr(self._base_stt, name)


class _WrappedSpeechStream:
    """Wrapper for STT streaming that captures frames and creates spans."""

    def __init__(
        self,
        base_stream: Any,
        session_id: str,
        job_context: dict[str, Any],
        provider: str,
        model: str,
        counter_ref: list[int],
    ):
        self._base_stream = base_stream
        self._session_id = session_id
        self._job_context = job_context
        self._provider = provider
        self._model = model
        self._counter_ref = counter_ref

        # State management
        self._buffered_frames: list[Any] = []
        self._current_request_id: Optional[str] = None

    def push_frame(self, frame: Any) -> None:
        """
        Push an audio frame to the stream.

        Args:
            frame: rtc.AudioFrame to push
        """
        self._buffered_frames.append(frame)
        self._base_stream.push_frame(frame)

    async def __anext__(self) -> SpeechEvent:
        """
        Get next speech event from the stream.

        Returns:
            SpeechEvent from the base stream
        """
        event = await self._base_stream.__anext__()

        # Only create span on FINAL transcripts
        if event.type == SpeechEventType.FINAL_TRANSCRIPT:
            # Increment counter
            self._counter_ref[0] += 1

            # Calculate duration
            duration_ms = calculate_audio_duration_ms(self._buffered_frames)

            # Create span if trace exists
            trace = get_current_trace()
            if trace:
                from noveum_trace import get_client

                try:
                    client = get_client()

                    # Generate audio UUID
                    audio_uuid = str(uuid.uuid4())

                    # Get transcript text and additional attributes
                    transcript = STT_TRANSCRIPT_DEFAULT_VALUE
                    confidence = STT_CONFIDENCE_DEFAULT_VALUE
                    language = STT_LANGUAGE_DEFAULT_VALUE
                    start_time = STT_START_TIME_DEFAULT_VALUE
                    end_time = STT_END_TIME_DEFAULT_VALUE
                    speaker_id = STT_SPEAKER_ID_DEFAULT_VALUE
                    is_primary_speaker = STT_IS_PRIMARY_SPEAKER_DEFAULT_VALUE

                    if event.alternatives and len(event.alternatives) > 0:
                        first_alternative = event.alternatives[0]
                        transcript = first_alternative.text
                        confidence = first_alternative.confidence
                        language = first_alternative.language
                        start_time = first_alternative.start_time
                        end_time = first_alternative.end_time
                        speaker_id = first_alternative.speaker_id
                        is_primary_speaker = first_alternative.is_primary_speaker

                    # Build additional attributes
                    additional_attrs = {
                        "stt.transcript": transcript,
                        "stt.confidence": confidence,
                        "stt.is_final": True,
                        "stt.mode": "streaming",
                        "stt.event_type": event.type.value,
                    }

                    if event.request_id:
                        additional_attrs["stt.request_id"] = event.request_id

                    if language:
                        additional_attrs["stt.language"] = language

                    if start_time is not None:
                        additional_attrs["stt.start_time"] = start_time

                    if end_time is not None:
                        additional_attrs["stt.end_time"] = end_time

                    if speaker_id is not None:
                        additional_attrs["stt.speaker_id"] = speaker_id

                    if is_primary_speaker is not None:
                        additional_attrs["stt.is_primary_speaker"] = is_primary_speaker

                    if (
                        event.recognition_usage
                        and event.recognition_usage.audio_duration
                    ):
                        additional_attrs["stt.recognition_usage.audio_duration"] = (
                            event.recognition_usage.audio_duration
                        )

                    # Create span attributes
                    attributes = create_span_attributes(
                        provider=self._provider,
                        model=self._model,
                        operation_type="stt",
                        audio_uuid=audio_uuid,
                        audio_duration_ms=duration_ms,
                        job_context=self._job_context,
                        **additional_attrs,
                    )

                    # Create and finish span
                    span = client.start_span(name="stt.stream", attributes=attributes)

                    # Upload audio frames with explicit span context
                    upload_audio_frames(
                        self._buffered_frames,
                        audio_uuid,
                        "stt",
                        span.trace_id,
                        span.span_id,
                    )

                    span.set_status(SpanStatus.OK)
                    client.finish_span(span)

                except (
                    Exception
                ) as e:  # noqa: S110 - broad exception for graceful degradation
                    # Gracefully handle span creation errors
                    logger.warning(
                        f"Failed to create span for STT streaming: {e}", exc_info=True
                    )

            # Clear buffer for next utterance
            # Note: Full conversation audio is handled by LiveKit's RecorderIO
            self._buffered_frames = []

        return event

    def __aiter__(self) -> _WrappedSpeechStream:
        """Return self as async iterator."""
        return self

    async def __aenter__(self) -> _WrappedSpeechStream:
        """Enter async context manager."""
        # If base stream is an async context manager, enter it
        if hasattr(self._base_stream, "__aenter__"):
            await self._base_stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Union[type[BaseException], None],
        exc: Union[BaseException, None],
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        # If base stream is an async context manager, exit it
        if hasattr(self._base_stream, "__aexit__"):
            await self._base_stream.__aexit__(exc_type, exc, exc_tb)
        else:
            # Fallback to aclose if no context manager support
            await self.aclose()

    async def flush(self) -> None:
        """Flush the stream."""
        if hasattr(self._base_stream, "flush"):
            await self._base_stream.flush()

    async def aclose(self) -> None:
        """Close the stream."""
        if hasattr(self._base_stream, "aclose"):
            await self._base_stream.aclose()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base stream."""
        return getattr(self._base_stream, name)
