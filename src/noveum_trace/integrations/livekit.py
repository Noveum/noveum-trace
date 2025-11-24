"""
LiveKit STT/TTS integration for noveum-trace.

This module provides wrapper classes that automatically trace LiveKit STT and TTS
operations, capturing audio files and metadata as span attributes.
"""

from pathlib import Path
from typing import Any, Optional

from noveum_trace.core.context import get_current_trace
from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.livekit_utils import (
    calculate_audio_duration_ms,
    create_span_attributes,
    ensure_audio_directory,
    generate_audio_filename,
    save_audio_buffer,
    save_audio_frames,
)

try:
    from livekit.agents.stt import SpeechEvent, SpeechEventType, STTCapabilities
    from livekit.agents.tts import SynthesizedAudio, TTSCapabilities
    from livekit.agents.utils import AudioBuffer

    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    # Placeholders for when LiveKit is not installed
    SpeechEvent = Any  # type: ignore
    SpeechEventType = Any  # type: ignore
    STTCapabilities = Any  # type: ignore
    SynthesizedAudio = Any  # type: ignore
    TTSCapabilities = Any  # type: ignore
    AudioBuffer = Any  # type: ignore


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
        audio_base_dir: Optional[Path] = None
    ):
        """
        Initialize STT wrapper.

        Args:
            stt: Base LiveKit STT provider instance
            session_id: Session identifier for organizing audio files
            job_context: Dictionary of job context information to attach to spans
            audio_base_dir: Base directory for audio files (defaults to 'audio_files')

        Raises:
            ImportError: If livekit package is not installed
        """
        if not LIVEKIT_AVAILABLE:
            raise ImportError(
                "livekit package is required for LiveKit integration. "
                "Install it with: pip install livekit livekit-agents"
            )

        self._base_stt = stt
        self._session_id = session_id
        self._job_context = job_context or {}
        self._audio_dir = ensure_audio_directory(session_id, audio_base_dir)
        self._counter_ref = [0]  # Mutable reference for sharing with streams

    @property
    def capabilities(self) -> STTCapabilities:
        """Get STT capabilities from base provider."""
        return self._base_stt.capabilities

    @property
    def model(self) -> str:
        """Get model name from base provider."""
        return getattr(self._base_stt, 'model', 'unknown')

    @property
    def provider(self) -> str:
        """Get provider name from base provider."""
        return getattr(self._base_stt, 'provider', 'unknown')

    @property
    def label(self) -> str:
        """Get label from base provider."""
        return getattr(self._base_stt, 'label', self._base_stt.__class__.__name__)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        **kwargs: Any
    ) -> SpeechEvent:
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

        # Generate audio filename
        audio_filename = generate_audio_filename('stt', self._counter_ref[0])
        audio_path = self._audio_dir / audio_filename

        # Save audio buffer
        try:
            save_audio_buffer(buffer, audio_path)
        except Exception:  # noqa: S110 - broad exception for graceful degradation
            # Log but don't fail if audio save fails
            pass

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

                # Get transcript text
                transcript = ""
                confidence = 0.0
                if event.alternatives and len(event.alternatives) > 0:
                    transcript = event.alternatives[0].text
                    confidence = event.alternatives[0].confidence

                # Create span attributes
                attributes = create_span_attributes(
                    provider=self.provider,
                    model=self.model,
                    operation_type='stt',
                    audio_file=audio_filename,
                    audio_duration_ms=duration_ms,
                    job_context=self._job_context,
                    **{
                        'stt.transcript': transcript,
                        'stt.confidence': confidence,
                        'stt.is_final': True,
                        'stt.mode': 'batch'
                    }
                )

                # Create and finish span
                span = client.start_span(
                    name="stt.recognize", attributes=attributes)
                span.set_status(SpanStatus.OK)
                client.finish_span(span)

            except Exception:  # noqa: S110 - broad exception for graceful degradation
                # Gracefully handle span creation errors
                pass

        return event

    async def recognize(
        self,
        buffer: AudioBuffer,
        **kwargs: Any
    ) -> SpeechEvent:
        """
        Public recognition API.

        Args:
            buffer: Audio buffer to recognize
            **kwargs: Additional arguments

        Returns:
            SpeechEvent with recognition results
        """
        return await self._recognize_impl(buffer, **kwargs)

    def stream(self, **kwargs: Any) -> "_WrappedSpeechStream":
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
            audio_dir=self._audio_dir
        )

    async def aclose(self) -> None:
        """Close the STT provider."""
        if hasattr(self._base_stt, 'aclose'):
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
        audio_dir: Path
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

            # Generate audio filename
            audio_filename = generate_audio_filename(
                'stt', self._counter_ref[0])
            audio_path = self._audio_dir / audio_filename

            # Save buffered audio
            try:
                if self._buffered_frames:
                    save_audio_frames(self._buffered_frames, audio_path)
            except Exception:  # noqa: S110 - broad exception for graceful degradation
                # Log but don't fail if audio save fails
                pass

            # Calculate duration
            duration_ms = calculate_audio_duration_ms(self._buffered_frames)

            # Create span if trace exists
            trace = get_current_trace()
            if trace:
                from noveum_trace import get_client

                try:
                    client = get_client()

                    # Get transcript text
                    transcript = ""
                    confidence = 0.0
                    if event.alternatives and len(event.alternatives) > 0:
                        transcript = event.alternatives[0].text
                        confidence = event.alternatives[0].confidence

                    # Create span attributes
                    attributes = create_span_attributes(
                        provider=self._provider,
                        model=self._model,
                        operation_type='stt',
                        audio_file=audio_filename,
                        audio_duration_ms=duration_ms,
                        job_context=self._job_context,
                        **{
                            'stt.transcript': transcript,
                            'stt.confidence': confidence,
                            'stt.is_final': True,
                            'stt.mode': 'streaming',
                            'stt.request_id': event.request_id
                        }
                    )

                    # Create and finish span
                    span = client.start_span(
                        name="stt.stream", attributes=attributes)
                    span.set_status(SpanStatus.OK)
                    client.finish_span(span)

                except Exception:  # noqa: S110 - broad exception for graceful degradation
                    # Gracefully handle span creation errors
                    pass

            # Clear buffer for next utterance
            self._buffered_frames = []

        return event

    def __aiter__(self) -> "_WrappedSpeechStream":
        """Return self as async iterator."""
        return self

    async def __aenter__(self) -> "_WrappedSpeechStream":
        """Enter async context manager."""
        # If base stream is an async context manager, enter it
        if hasattr(self._base_stream, '__aenter__'):
            await self._base_stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        # If base stream is an async context manager, exit it
        if hasattr(self._base_stream, '__aexit__'):
            await self._base_stream.__aexit__(exc_type, exc, exc_tb)
        else:
            # Fallback to aclose if no context manager support
            await self.aclose()

    async def flush(self) -> None:
        """Flush the stream."""
        if hasattr(self._base_stream, 'flush'):
            await self._base_stream.flush()

    async def aclose(self) -> None:
        """Close the stream."""
        if hasattr(self._base_stream, 'aclose'):
            await self._base_stream.aclose()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base stream."""
        return getattr(self._base_stream, name)


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
        audio_base_dir: Optional[Path] = None
    ):
        """
        Initialize TTS wrapper.

        Args:
            tts: Base LiveKit TTS provider instance
            session_id: Session identifier for organizing audio files
            job_context: Dictionary of job context information to attach to spans
            audio_base_dir: Base directory for audio files (defaults to 'audio_files')

        Raises:
            ImportError: If livekit package is not installed
        """
        if not LIVEKIT_AVAILABLE:
            raise ImportError(
                "livekit package is required for LiveKit integration. "
                "Install it with: pip install livekit livekit-agents"
            )

        self._base_tts = tts
        self._session_id = session_id
        self._job_context = job_context or {}
        self._audio_dir = ensure_audio_directory(session_id, audio_base_dir)
        self._counter_ref = [0]  # Mutable reference for sharing with streams

    @property
    def capabilities(self) -> TTSCapabilities:
        """Get TTS capabilities from base provider."""
        return self._base_tts.capabilities

    @property
    def model(self) -> str:
        """Get model name from base provider."""
        return getattr(self._base_tts, 'model', 'unknown')

    @property
    def provider(self) -> str:
        """Get provider name from base provider."""
        return getattr(self._base_tts, 'provider', 'unknown')

    @property
    def label(self) -> str:
        """Get label from base provider."""
        return getattr(self._base_tts, 'label', self._base_tts.__class__.__name__)

    @property
    def sample_rate(self) -> int:
        """Get sample rate from base provider."""
        return self._base_tts.sample_rate

    @property
    def num_channels(self) -> int:
        """Get number of channels from base provider."""
        return self._base_tts.num_channels

    def synthesize(self, text: str, **kwargs: Any) -> "_WrappedChunkedStream":
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
            audio_dir=self._audio_dir
        )

    def stream(self, **kwargs: Any) -> "_WrappedSynthesizeStream":
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
            audio_dir=self._audio_dir
        )

    def prewarm(self) -> None:
        """Pre-warm connection to TTS service."""
        if hasattr(self._base_tts, 'prewarm'):
            self._base_tts.prewarm()

    async def aclose(self) -> None:
        """Close the TTS provider."""
        if hasattr(self._base_tts, 'aclose'):
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
        audio_dir: Path
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

            # Generate audio filename
            audio_filename = generate_audio_filename(
                'tts', self._counter_ref[0])
            audio_path = self._audio_dir / audio_filename

            # Save buffered audio
            try:
                if self._buffered_frames:
                    save_audio_frames(self._buffered_frames, audio_path)
            except Exception:  # noqa: S110 - broad exception for graceful degradation
                # Log but don't fail if audio save fails
                pass

            # Calculate duration
            duration_ms = calculate_audio_duration_ms(self._buffered_frames)

            # Create span if trace exists
            trace = get_current_trace()
            if trace:
                from noveum_trace import get_client

                try:
                    client = get_client()

                    # Use accumulated input_text if available, otherwise use fallback
                    input_text = self._input_text.strip() if self._input_text else ""

                    # Try to get text from audio.delta_text if input_text is empty
                    if not input_text and hasattr(audio, 'delta_text') and audio.delta_text:
                        # Fallback: try to use delta_text if available
                        delta_text = audio.delta_text.strip()
                        if delta_text:
                            input_text = delta_text

                    # Final fallback: use "unknown" if we still have no text
                    if not input_text:
                        input_text = "unknown"

                    # Create span attributes
                    attributes = create_span_attributes(
                        provider=self._provider,
                        model=self._model,
                        operation_type='tts',
                        audio_file=audio_filename,
                        audio_duration_ms=duration_ms,
                        job_context=self._job_context,
                        **{
                            'tts.input_text': input_text,
                            'tts.segment_id': self._segment_id or '',
                            'tts.request_id': self._current_request_id or '',
                            'tts.mode': 'streaming'
                        }
                    )

                    # Create and finish span
                    span = client.start_span(
                        name="tts.stream", attributes=attributes)
                    span.set_status(SpanStatus.OK)
                    client.finish_span(span)

                except Exception:  # noqa: S110 - broad exception for graceful degradation
                    # Gracefully handle span creation errors
                    pass

            # Clear buffer for next segment
            self._buffered_frames = []
            self._input_text = ""

        return audio

    def __aiter__(self) -> "_WrappedSynthesizeStream":
        """Return self as async iterator."""
        return self

    async def __aenter__(self) -> "_WrappedSynthesizeStream":
        """Enter async context manager."""
        # If base stream is an async context manager, enter it
        if hasattr(self._base_stream, '__aenter__'):
            await self._base_stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        # If base stream is an async context manager, exit it
        if hasattr(self._base_stream, '__aexit__'):
            await self._base_stream.__aexit__(exc_type, exc, exc_tb)
        else:
            # Fallback to aclose if no context manager support
            await self.aclose()

    async def flush(self) -> None:
        """Flush the stream."""
        if hasattr(self._base_stream, 'flush'):
            await self._base_stream.flush()

    async def aclose(self) -> None:
        """Close the stream."""
        if hasattr(self._base_stream, 'aclose'):
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
        audio_dir: Path
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

        # Generate audio filename
        audio_filename = generate_audio_filename('tts', self._counter_ref[0])
        audio_path = self._audio_dir / audio_filename

        # Save buffered audio
        try:
            if self._buffered_frames:
                save_audio_frames(self._buffered_frames, audio_path)
        except Exception:  # noqa: S110 - broad exception for graceful degradation
            # Log but don't fail if audio save fails
            pass

        # Calculate duration
        duration_ms = calculate_audio_duration_ms(self._buffered_frames)

        # Create span if trace exists
        trace = get_current_trace()
        if trace:
            from noveum_trace import get_client

            try:
                client = get_client()

                # Create span attributes
                attributes = create_span_attributes(
                    provider=self._provider,
                    model=self._model,
                    operation_type='tts',
                    audio_file=audio_filename,
                    audio_duration_ms=duration_ms,
                    job_context=self._job_context,
                    **{
                        'tts.input_text': self._input_text,
                        'tts.mode': 'batch'
                    }
                )

                # Create and finish span
                span = client.start_span(
                    name="tts.synthesize", attributes=attributes)
                span.set_status(SpanStatus.OK)
                client.finish_span(span)

            except Exception:  # noqa: S110 - broad exception for graceful degradation
                # Gracefully handle span creation errors
                pass

    def __aiter__(self) -> "_WrappedChunkedStream":
        """Return self as async iterator."""
        return self

    async def aclose(self) -> None:
        """Close the stream."""
        # Create span if not already created (e.g., if iteration stopped early)
        if not self._span_created and self._buffered_frames:
            self._create_span()

        if hasattr(self._base_stream, 'aclose'):
            await self._base_stream.aclose()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base stream."""
        return getattr(self._base_stream, name)
