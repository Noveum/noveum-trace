"""
LiveKit TTS integration for noveum-trace.

This module provides wrapper classes that automatically trace LiveKit TTS
operations, capturing audio files and metadata as span attributes.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional, Union

from noveum_trace.core.context import get_current_trace
from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.livekit.livekit_constants import (
    TTS_DELTA_TEXT_DEFAULT_VALUE,
    TTS_INPUT_TEXT_DEFAULT_VALUE,
    TTS_NUM_CHANNELS_DEFAULT_VALUE,
    TTS_REQUEST_ID_DEFAULT_VALUE,
    TTS_SAMPLE_RATE_DEFAULT_VALUE,
    TTS_SEGMENT_ID_DEFAULT_VALUE,
)
from noveum_trace.integrations.livekit.livekit_utils import (
    calculate_audio_duration_ms,
    create_span_attributes,
    ensure_audio_directory,
    upload_audio_frames,
)

logger = logging.getLogger(__name__)

try:
    from livekit.agents.tts import SynthesizedAudio, TTSCapabilities

    LIVEKIT_AVAILABLE = True
except ImportError as e:
    LIVEKIT_AVAILABLE = False
    logger.debug(
        "LiveKit is not importable. LiveKit TTS integration features will not be available. "
        "Install it with: pip install livekit livekit-agents",
        exc_info=e,
    )


class LiveKitTTSWrapper:
    """
    Wrapper for LiveKit TTS providers that automatically creates spans for synthesis.

    This wrapper captures synthesized audio frames, saves them to disk, and creates
    spans with metadata for each synthesis operation (both streaming and batch modes).

    Example:
        >>> import noveum_trace
        >>> from livekit.plugins import cartesia
        >>> from noveum_trace.integrations.livekit import LiveKitTTSWrapper
        >>>
        >>> # Initialize noveum-trace (done elsewhere)
        >>> noveum_trace.init(project="livekit-agents")
        >>>
        >>> # Wrap TTS provider
        >>> base_tts = cartesia.TTS(...)
        >>> traced_tts = LiveKitTTSWrapper(
        ...     tts=base_tts,
        ...     session_id="session_123",
        ...     job_context={"job_id": "job_abc"}
        ... )
        >>>
        >>> # Use in streaming mode
        >>> stream = traced_tts.stream()
        >>> stream.push_text("Hello, world!")
        >>> async for audio in stream:
        ...     play_audio(audio.frame)
    """

    def __init__(
        self,
        tts: Any,  # noqa: F811 - parameter shadows import
        session_id: str,
        job_context: Optional[dict[str, Any]] = None,
        audio_base_dir: Optional[Path] = None,
    ):
        """
        Initialize TTS wrapper.

        Args:
            tts: Base LiveKit TTS provider instance
            session_id: Session identifier for organizing audio files
            job_context: Dictionary of job context information to attach to spans
            audio_base_dir: Base directory for audio files (defaults to 'audio_files')
        """
        # Always initialize fields so wrapper is safe to use when LiveKit is unavailable
        self._base_tts = tts
        self._session_id = session_id
        self._job_context = job_context or {}
        self._counter_ref = [0]  # Mutable reference for sharing with streams

        # Always create audio directory (doesn't require LiveKit)
        self._audio_dir = ensure_audio_directory(session_id, audio_base_dir)

        if not LIVEKIT_AVAILABLE:
            logger.error(
                "Cannot initialize LiveKitTTSWrapper: LiveKit is not available. "
                "Install it with: pip install livekit livekit-agents"
            )
            return

    @property
    def capabilities(self) -> TTSCapabilities:
        """Get TTS capabilities from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_tts is None:
            raise RuntimeError(
                "LiveKit is not available. Cannot access capabilities. "
                "Install it with: pip install livekit livekit-agents"
            )
        return self._base_tts.capabilities

    @property
    def model(self) -> str:
        """Get model name from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_tts is None:
            return "unknown"
        return getattr(self._base_tts, "model", "unknown")

    @property
    def provider(self) -> str:
        """Get provider name from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_tts is None:
            return "unknown"
        return getattr(self._base_tts, "provider", "unknown")

    @property
    def label(self) -> str:
        """Get label from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_tts is None:
            return "LiveKitTTSWrapper"
        return getattr(self._base_tts, "label", self._base_tts.__class__.__name__)

    @property
    def sample_rate(self) -> int:
        """Get sample rate from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_tts is None:
            raise RuntimeError(
                "LiveKit is not available. Cannot access sample_rate. "
                "Install it with: pip install livekit livekit-agents"
            )
        return self._base_tts.sample_rate

    @property
    def num_channels(self) -> int:
        """Get number of channels from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_tts is None:
            raise RuntimeError(
                "LiveKit is not available. Cannot access num_channels. "
                "Install it with: pip install livekit livekit-agents"
            )
        return self._base_tts.num_channels

    def synthesize(self, text: str, **kwargs: Any) -> _WrappedChunkedStream:
        """
        Synthesize text to speech (batch mode).

        Args:
            text: Text to synthesize
            **kwargs: Additional arguments passed to base TTS

        Returns:
            Wrapped chunked stream
        """
        base_stream = self._base_tts.synthesize(text, **kwargs)
        return _WrappedChunkedStream(
            base_stream=base_stream,
            input_text=text,
            session_id=self._session_id,
            job_context=self._job_context,
            provider=self.provider,
            model=self.model,
            counter_ref=self._counter_ref,
            audio_dir=self._audio_dir,
        )

    def stream(self, **kwargs: Any) -> _WrappedSynthesizeStream:
        """
        Create a streaming synthesis interface.

        Args:
            **kwargs: Additional arguments passed to base TTS

        Returns:
            Wrapped synthesize stream
        """
        base_stream = self._base_tts.stream(**kwargs)
        return _WrappedSynthesizeStream(
            base_stream=base_stream,
            session_id=self._session_id,
            job_context=self._job_context,
            provider=self.provider,
            model=self.model,
            counter_ref=self._counter_ref,
            audio_dir=self._audio_dir,
        )

    def prewarm(self) -> None:
        """Pre-warm connection to TTS service."""
        if hasattr(self._base_tts, "prewarm"):
            self._base_tts.prewarm()

    async def aclose(self) -> None:
        """Close the TTS provider."""
        if hasattr(self._base_tts, "aclose"):
            await self._base_tts.aclose()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base TTS."""
        return getattr(self._base_tts, name)


class _WrappedSynthesizeStream:
    """Wrapper for TTS streaming that captures frames and creates spans."""

    def __init__(
        self,
        base_stream: Any,
        session_id: str,
        job_context: dict[str, Any],
        provider: str,
        model: str,
        counter_ref: list[int],
        audio_dir: Path,
    ):
        self._base_stream = base_stream
        self._session_id = session_id
        self._job_context = job_context
        self._provider = provider
        self._model = model
        self._counter_ref = counter_ref
        self._audio_dir = audio_dir

        # State management
        self._buffered_frames: list[Any] = []
        self._input_text = ""
        self._segment_id: Optional[str] = None
        self._current_request_id: Optional[str] = None

    def push_text(self, text: str) -> None:
        """
        Push text to synthesize.

        Args:
            text: Text to synthesize
        """
        # Accumulate text across multiple push_text calls for the same segment
        # Just concatenate directly without any processing
        self._input_text += text
        self._base_stream.push_text(text)

    async def __anext__(self) -> SynthesizedAudio:
        """
        Get next synthesized audio chunk.

        Returns:
            SynthesizedAudio from the base stream
        """
        audio = await self._base_stream.__anext__()

        # Buffer all frames
        self._buffered_frames.append(audio.frame)
        self._segment_id = audio.segment_id
        self._current_request_id = audio.request_id

        # Create span when synthesis is complete
        if audio.is_final:
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

                    # Use accumulated input_text if available, otherwise use fallback
                    input_text = (
                        self._input_text.strip()
                        if self._input_text
                        else TTS_INPUT_TEXT_DEFAULT_VALUE
                    )

                    # Try to get text from audio.delta_text if input_text is empty
                    delta_text = TTS_DELTA_TEXT_DEFAULT_VALUE
                    if (
                        not input_text
                        and hasattr(audio, "delta_text")
                        and audio.delta_text
                    ):
                        # Fallback: try to use delta_text if available
                        delta_text = audio.delta_text.strip()
                        if delta_text:
                            input_text = delta_text

                    # Final fallback: use "unknown" if we still have no text
                    if not input_text:
                        input_text = "unknown"

                    # Get additional attributes from audio
                    segment_id = (
                        self._segment_id
                        if self._segment_id
                        else TTS_SEGMENT_ID_DEFAULT_VALUE
                    )
                    request_id = (
                        self._current_request_id
                        if self._current_request_id
                        else TTS_REQUEST_ID_DEFAULT_VALUE
                    )
                    sample_rate = (
                        audio.frame.sample_rate
                        if hasattr(audio.frame, "sample_rate")
                        else TTS_SAMPLE_RATE_DEFAULT_VALUE
                    )
                    num_channels = (
                        audio.frame.num_channels
                        if hasattr(audio.frame, "num_channels")
                        else TTS_NUM_CHANNELS_DEFAULT_VALUE
                    )

                    # Build additional attributes
                    additional_attrs = {
                        "tts.input_text": input_text,
                        "tts.mode": "streaming",
                        "tts.is_final": audio.is_final,
                    }

                    if segment_id is not None:
                        additional_attrs["tts.segment_id"] = segment_id

                    if request_id is not None:
                        additional_attrs["tts.request_id"] = request_id

                    if delta_text is not None:
                        additional_attrs["tts.delta_text"] = delta_text

                    if sample_rate is not None:
                        additional_attrs["tts.sample_rate"] = sample_rate

                    if num_channels is not None:
                        additional_attrs["tts.num_channels"] = num_channels

                    # Create span attributes
                    attributes = create_span_attributes(
                        provider=self._provider,
                        model=self._model,
                        operation_type="tts",
                        audio_uuid=audio_uuid,
                        audio_duration_ms=duration_ms,
                        job_context=self._job_context,
                        **additional_attrs,
                    )

                    # Create and finish span
                    span = client.start_span(name="tts.stream", attributes=attributes)

                    # Upload audio frames with explicit span context
                    upload_audio_frames(
                        self._buffered_frames,
                        audio_uuid,
                        "tts",
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
                        f"Failed to create span for TTS streaming: {e}", exc_info=True
                    )

            # Clear buffer for next segment
            self._buffered_frames = []
            self._input_text = ""

        return audio

    def __aiter__(self) -> _WrappedSynthesizeStream:
        """Return self as async iterator."""
        return self

    async def __aenter__(self) -> _WrappedSynthesizeStream:
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


class _WrappedChunkedStream:
    """Wrapper for TTS batch synthesis that captures frames and creates spans."""

    def __init__(
        self,
        base_stream: Any,
        input_text: str,
        session_id: str,
        job_context: dict[str, Any],
        provider: str,
        model: str,
        counter_ref: list[int],
        audio_dir: Path,
    ):
        self._base_stream = base_stream
        self._input_text = input_text
        self._session_id = session_id
        self._job_context = job_context
        self._provider = provider
        self._model = model
        self._counter_ref = counter_ref
        self._audio_dir = audio_dir

        # State management
        self._buffered_frames: list[Any] = []
        self._first_audio: Optional[SynthesizedAudio] = None
        self._span_created = False

    async def __anext__(self) -> SynthesizedAudio:
        """
        Get next synthesized audio chunk.

        Returns:
            SynthesizedAudio from the base stream
        """
        audio = await self._base_stream.__anext__()

        # Buffer frames
        self._buffered_frames.append(audio.frame)

        # Store first audio for metadata
        if self._first_audio is None:
            self._first_audio = audio

        # Create span after collecting all frames (on final frame)
        if audio.is_final and not self._span_created:
            self._create_span()

        return audio

    def _create_span(self) -> None:
        """Create span for the synthesize operation."""
        if self._span_created:
            return

        self._span_created = True

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

                # Get additional attributes from first audio
                input_text = (
                    self._input_text.strip()
                    if self._input_text
                    else TTS_INPUT_TEXT_DEFAULT_VALUE
                )
                request_id = (
                    self._first_audio.request_id
                    if self._first_audio and self._first_audio.request_id
                    else TTS_REQUEST_ID_DEFAULT_VALUE
                )
                segment_id = (
                    self._first_audio.segment_id
                    if self._first_audio and self._first_audio.segment_id
                    else TTS_SEGMENT_ID_DEFAULT_VALUE
                )
                delta_text = (
                    self._first_audio.delta_text
                    if self._first_audio and self._first_audio.delta_text
                    else TTS_DELTA_TEXT_DEFAULT_VALUE
                )
                sample_rate = (
                    self._first_audio.frame.sample_rate
                    if self._first_audio
                    and hasattr(self._first_audio.frame, "sample_rate")
                    else TTS_SAMPLE_RATE_DEFAULT_VALUE
                )
                num_channels = (
                    self._first_audio.frame.num_channels
                    if self._first_audio
                    and hasattr(self._first_audio.frame, "num_channels")
                    else TTS_NUM_CHANNELS_DEFAULT_VALUE
                )

                # Build additional attributes
                additional_attrs = {
                    "tts.input_text": input_text,
                    "tts.mode": "batch",
                    "tts.is_final": (
                        self._first_audio.is_final if self._first_audio else True
                    ),
                }

                if request_id is not None:
                    additional_attrs["tts.request_id"] = request_id

                if segment_id is not None:
                    additional_attrs["tts.segment_id"] = segment_id

                if delta_text is not None:
                    additional_attrs["tts.delta_text"] = delta_text

                if sample_rate is not None:
                    additional_attrs["tts.sample_rate"] = sample_rate

                if num_channels is not None:
                    additional_attrs["tts.num_channels"] = num_channels

                # Create span attributes
                attributes = create_span_attributes(
                    provider=self._provider,
                    model=self._model,
                    operation_type="tts",
                    audio_uuid=audio_uuid,
                    audio_duration_ms=duration_ms,
                    job_context=self._job_context,
                    **additional_attrs,
                )

                # Create and finish span
                span = client.start_span(name="tts.synthesize", attributes=attributes)

                # Upload audio frames with explicit span context
                upload_audio_frames(
                    self._buffered_frames,
                    audio_uuid,
                    "tts",
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
                    f"Failed to create/finish tts.synthesize span: {e}", exc_info=True
                )

    def __aiter__(self) -> _WrappedChunkedStream:
        """Return self as async iterator."""
        return self

    async def aclose(self) -> None:
        """Close the stream."""
        # Create span if not already created (e.g., if iteration stopped early)
        if not self._span_created and self._buffered_frames:
            self._create_span()

        if hasattr(self._base_stream, "aclose"):
            await self._base_stream.aclose()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base stream."""
        return getattr(self._base_stream, name)
