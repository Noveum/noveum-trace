"""
Unit tests for LiveKit STT/TTS wrapper classes.

Tests the wrapper classes in livekit.py:
- LiveKitSTTWrapper
- LiveKitTTSWrapper
- _WrappedSpeechStream
- _WrappedSynthesizeStream
- _WrappedChunkedStream
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Skip all tests if LiveKit is not available
try:
    from noveum_trace.integrations.livekit.livekit_stt import LiveKitSTTWrapper
    from noveum_trace.integrations.livekit.livekit_tts import LiveKitTTSWrapper

    LIVEKIT_WRAPPER_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    LIVEKIT_WRAPPER_AVAILABLE = False


@pytest.fixture
def mock_stt_provider():
    """Create a mock STT provider."""
    stt = Mock()
    stt.capabilities = Mock()
    stt.model = "nova-2"
    stt.provider = "deepgram"
    stt.label = "Deepgram STT"
    stt._recognize_impl = AsyncMock()
    stt.stream = Mock()
    stt.aclose = AsyncMock()
    return stt


@pytest.fixture
def mock_tts_provider():
    """Create a mock TTS provider."""
    tts = Mock()
    tts.capabilities = Mock()
    tts.model = "sonic"
    tts.provider = "cartesia"
    tts.label = "Cartesia TTS"
    tts.sample_rate = 24000
    tts.num_channels = 1
    tts.synthesize = Mock()
    tts.stream = Mock()
    return tts


@pytest.fixture
def mock_trace():
    """Create a mock trace."""
    trace = Mock()
    trace.trace_id = "test_trace_123"
    return trace


@pytest.fixture
def mock_client():
    """Create a mock noveum client."""
    client = Mock()
    mock_span = Mock()
    mock_span.span_id = "test_span_456"
    client.start_span = Mock(return_value=mock_span)
    client.finish_span = Mock()
    return client


@pytest.mark.skipif(
    not LIVEKIT_WRAPPER_AVAILABLE, reason="LiveKit wrappers not available"
)
class TestLiveKitSTTWrapper:
    """Test LiveKitSTTWrapper class."""

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    def test_init_with_livekit_available(self, mock_stt_provider, tmp_path):
        """Test STT wrapper initialization when LiveKit is available."""
        wrapper = LiveKitSTTWrapper(
            stt=mock_stt_provider,
            session_id="session_123",
            job_context={"job_id": "job_456"},
            audio_base_dir=tmp_path,
        )

        assert wrapper._base_stt == mock_stt_provider
        assert wrapper._session_id == "session_123"
        assert wrapper._job_context == {"job_id": "job_456"}
        assert wrapper._counter_ref == [0]

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", False)
    def test_init_without_livekit(self, mock_stt_provider):
        """Test STT wrapper initialization when LiveKit is not available."""
        # When LiveKit is unavailable, wrapper should initialize gracefully without errors
        wrapper = LiveKitSTTWrapper(
            stt=mock_stt_provider,
            session_id="session_123",
        )
        # Verify wrapper was created successfully
        assert wrapper is not None
        # Verify attributes are initialized
        assert wrapper._base_stt == mock_stt_provider
        assert wrapper._session_id == "session_123"
        # Audio directory is always created regardless of LiveKit availability
        assert wrapper._audio_dir is not None
        assert wrapper._audio_dir.name == "session_123"

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    def test_properties(self, mock_stt_provider, tmp_path):
        """Test STT wrapper property access."""
        wrapper = LiveKitSTTWrapper(
            stt=mock_stt_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        assert wrapper.capabilities == mock_stt_provider.capabilities
        assert wrapper.model == "nova-2"
        assert wrapper.provider == "deepgram"
        assert wrapper.label == "Deepgram STT"

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    def test_properties_with_missing_attributes(self, tmp_path):
        """Test properties with base STT missing some attributes."""
        minimal_stt = Mock(spec=[])  # Empty spec so getattr returns AttributeError
        minimal_stt.capabilities = Mock()
        # Missing model, provider, label - getattr will raise AttributeError

        wrapper = LiveKitSTTWrapper(
            stt=minimal_stt, session_id="session_123", audio_base_dir=tmp_path
        )

        # getattr with default will return "unknown" when attribute doesn't exist
        assert wrapper.model == "unknown"
        assert wrapper.provider == "unknown"
        assert wrapper.label == minimal_stt.__class__.__name__

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_stt.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_stt.upload_audio_frames")
    @patch("noveum_trace.integrations.livekit.livekit_stt.calculate_audio_duration_ms")
    @patch("noveum_trace.integrations.livekit.livekit_stt.create_span_attributes")
    @pytest.mark.asyncio
    async def test_recognize_with_trace(
        self,
        mock_create_attrs,
        mock_calc_duration,
        mock_upload_audio,
        mock_get_client,
        mock_get_trace,
        mock_stt_provider,
        mock_trace,
        mock_client,
        tmp_path,
    ):
        """Test recognize() method with active trace."""
        wrapper = LiveKitSTTWrapper(
            stt=mock_stt_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        # Mock speech event
        mock_event = Mock()
        mock_alternative = Mock()
        mock_alternative.text = "Hello world"
        mock_alternative.confidence = 0.95
        mock_alternative.language = "en"
        mock_alternative.start_time = 0.0
        mock_alternative.end_time = 1.5
        mock_alternative.speaker_id = None
        mock_alternative.is_primary_speaker = None
        mock_event.alternatives = [mock_alternative]
        mock_event.type = Mock()
        mock_event.type.value = "FINAL_TRANSCRIPT"
        mock_event.request_id = "req_123"
        mock_event.recognition_usage = None
        mock_stt_provider._recognize_impl = AsyncMock(return_value=mock_event)

        # Mock audio buffer
        mock_buffer = [Mock(), Mock()]

        # Setup mocks
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client
        mock_upload_audio.return_value = True
        mock_calc_duration.return_value = 1500.0
        mock_create_attrs.return_value = {"stt.provider": "deepgram"}
        
        # Mock span with trace_id and span_id
        mock_span = Mock()
        mock_span.trace_id = "trace-123"
        mock_span.span_id = "span-456"
        mock_client.start_span.return_value = mock_span

        result = await wrapper.recognize(mock_buffer)

        assert result == mock_event
        mock_stt_provider._recognize_impl.assert_called_once_with(mock_buffer)
        mock_upload_audio.assert_called_once()
        mock_client.start_span.assert_called_once()
        mock_client.finish_span.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_stt.get_current_trace")
    @pytest.mark.asyncio
    async def test_recognize_without_trace(
        self, mock_get_trace, mock_stt_provider, tmp_path
    ):
        """Test recognize() method without active trace."""
        wrapper = LiveKitSTTWrapper(
            stt=mock_stt_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        mock_event = Mock()
        mock_stt_provider._recognize_impl = AsyncMock(return_value=mock_event)
        mock_get_trace.return_value = None

        mock_frame = Mock()
        mock_frame.duration = 0.1
        result = await wrapper.recognize([mock_frame])

        assert result == mock_event
        mock_stt_provider._recognize_impl.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    def test_stream(self, mock_stt_provider, tmp_path):
        """Test stream() method creates wrapped stream."""
        wrapper = LiveKitSTTWrapper(
            stt=mock_stt_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        mock_base_stream = Mock()
        mock_stt_provider.stream.return_value = mock_base_stream

        stream = wrapper.stream()

        assert stream._base_stream == mock_base_stream
        assert stream._session_id == "session_123"
        assert stream._provider == "deepgram"
        assert stream._model == "nova-2"
        assert stream._counter_ref == wrapper._counter_ref

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_aclose(self, mock_stt_provider, tmp_path):
        """Test aclose() method."""
        wrapper = LiveKitSTTWrapper(
            stt=mock_stt_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        mock_stt_provider.aclose = AsyncMock()

        await wrapper.aclose()

        mock_stt_provider.aclose.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_aclose_without_method(self, tmp_path):
        """Test aclose() when base STT doesn't have aclose."""
        minimal_stt = Mock()
        del minimal_stt.aclose  # Ensure aclose does not exist
        minimal_stt.capabilities = Mock()
        # No aclose method

        wrapper = LiveKitSTTWrapper(
            stt=minimal_stt, session_id="session_123", audio_base_dir=tmp_path
        )

        # Should not raise error
        await wrapper.aclose()

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    def test_getattr_delegation(self, mock_stt_provider, tmp_path):
        """Test __getattr__ delegates to base STT."""
        wrapper = LiveKitSTTWrapper(
            stt=mock_stt_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        mock_stt_provider.custom_method = Mock(return_value="custom_value")

        result = wrapper.custom_method()

        assert result == "custom_value"
        mock_stt_provider.custom_method.assert_called_once()


@pytest.mark.skipif(
    not LIVEKIT_WRAPPER_AVAILABLE, reason="LiveKit wrappers not available"
)
class TestWrappedSpeechStream:
    """Test _WrappedSpeechStream class."""

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    def test_push_frame(self, tmp_path):
        """Test push_frame() buffers frames."""
        from noveum_trace.integrations.livekit.livekit_stt import _WrappedSpeechStream

        mock_base_stream = Mock()
        mock_base_stream.push_frame = Mock()

        stream = _WrappedSpeechStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="deepgram",
            model="nova-2",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        mock_frame = Mock()
        stream.push_frame(mock_frame)

        assert len(stream._buffered_frames) == 1
        assert stream._buffered_frames[0] == mock_frame
        mock_base_stream.push_frame.assert_called_once_with(mock_frame)

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_anext_non_final_no_span(self, tmp_path):
        """Test __anext__ doesn't create span for non-final events."""
        from noveum_trace.integrations.livekit.livekit_stt import _WrappedSpeechStream

        mock_base_stream = AsyncMock()
        mock_event = Mock()
        mock_event.type = "INTERIM_TRANSCRIPT"  # Not FINAL_TRANSCRIPT
        mock_base_stream.__anext__ = AsyncMock(return_value=mock_event)

        # Patch SpeechEventType to avoid NameError
        mock_speech_event_type = MagicMock()
        mock_speech_event_type.FINAL_TRANSCRIPT = "FINAL_TRANSCRIPT"

        with patch(
            "noveum_trace.integrations.livekit.livekit_stt.SpeechEventType",
            mock_speech_event_type,
            create=True,
        ):
            stream = _WrappedSpeechStream(
                base_stream=mock_base_stream,
                session_id="session_123",
                job_context={},
                provider="deepgram",
                model="nova-2",
                counter_ref=[0],
                audio_dir=tmp_path,
            )

            result = await stream.__anext__()

        assert result == mock_event
        # Buffer should not be cleared for non-final events
        assert len(stream._buffered_frames) == 0  # Was empty to start

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path):
        """Test async context manager support."""
        from noveum_trace.integrations.livekit.livekit_stt import _WrappedSpeechStream

        mock_base_stream = AsyncMock()
        mock_base_stream.__aenter__ = AsyncMock(return_value=mock_base_stream)
        mock_base_stream.__aexit__ = AsyncMock(return_value=None)

        stream = _WrappedSpeechStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="deepgram",
            model="nova-2",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        async with stream:
            pass

        mock_base_stream.__aenter__.assert_called_once()
        mock_base_stream.__aexit__.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_aclose(self, tmp_path):
        """Test aclose() method."""
        from noveum_trace.integrations.livekit.livekit_stt import _WrappedSpeechStream

        mock_base_stream = Mock()
        mock_base_stream.aclose = AsyncMock()

        stream = _WrappedSpeechStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="deepgram",
            model="nova-2",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        await stream.aclose()

        mock_base_stream.aclose.assert_called_once()


@pytest.mark.skipif(
    not LIVEKIT_WRAPPER_AVAILABLE, reason="LiveKit wrappers not available"
)
class TestLiveKitTTSWrapper:
    """Test LiveKitTTSWrapper class."""

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    def test_init_with_livekit_available(self, mock_tts_provider, tmp_path):
        """Test TTS wrapper initialization when LiveKit is available."""
        wrapper = LiveKitTTSWrapper(
            tts=mock_tts_provider,
            session_id="session_123",
            job_context={"job_id": "job_456"},
            audio_base_dir=tmp_path,
        )

        assert wrapper._base_tts == mock_tts_provider
        assert wrapper._session_id == "session_123"
        assert wrapper._job_context == {"job_id": "job_456"}
        assert wrapper._counter_ref == [0]

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    def test_properties(self, mock_tts_provider, tmp_path):
        """Test TTS wrapper property access."""
        wrapper = LiveKitTTSWrapper(
            tts=mock_tts_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        assert wrapper.capabilities == mock_tts_provider.capabilities
        assert wrapper.model == "sonic"
        assert wrapper.provider == "cartesia"
        assert wrapper.label == "Cartesia TTS"
        assert wrapper.sample_rate == 24000
        assert wrapper.num_channels == 1

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    def test_synthesize(self, mock_tts_provider, tmp_path):
        """Test synthesize() method creates wrapped chunked stream."""
        wrapper = LiveKitTTSWrapper(
            tts=mock_tts_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        mock_base_stream = Mock()
        mock_tts_provider.synthesize.return_value = mock_base_stream

        stream = wrapper.synthesize("Hello world")

        assert stream._base_stream == mock_base_stream
        assert stream._input_text == "Hello world"
        assert stream._provider == "cartesia"
        assert stream._model == "sonic"
        assert stream._counter_ref == wrapper._counter_ref

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    def test_stream(self, mock_tts_provider, tmp_path):
        """Test stream() method creates wrapped synthesize stream."""
        wrapper = LiveKitTTSWrapper(
            tts=mock_tts_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        mock_base_stream = Mock()
        mock_tts_provider.stream.return_value = mock_base_stream

        stream = wrapper.stream()

        assert stream._base_stream == mock_base_stream
        assert stream._provider == "cartesia"
        assert stream._model == "sonic"
        assert stream._counter_ref == wrapper._counter_ref

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    def test_prewarm(self, mock_tts_provider, tmp_path):
        """Test prewarm() method."""
        wrapper = LiveKitTTSWrapper(
            tts=mock_tts_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        mock_tts_provider.prewarm = Mock()

        wrapper.prewarm()

        mock_tts_provider.prewarm.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_aclose(self, mock_tts_provider, tmp_path):
        """Test aclose() method."""
        wrapper = LiveKitTTSWrapper(
            tts=mock_tts_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        mock_tts_provider.aclose = AsyncMock()

        await wrapper.aclose()

        mock_tts_provider.aclose.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    def test_getattr_delegation(self, mock_tts_provider, tmp_path):
        """Test __getattr__ delegates to base TTS."""
        wrapper = LiveKitTTSWrapper(
            tts=mock_tts_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        mock_tts_provider.custom_method = Mock(return_value="custom_value")

        result = wrapper.custom_method()

        assert result == "custom_value"
        mock_tts_provider.custom_method.assert_called_once()


@pytest.mark.skipif(
    not LIVEKIT_WRAPPER_AVAILABLE, reason="LiveKit wrappers not available"
)
class TestWrappedSynthesizeStream:
    """Test _WrappedSynthesizeStream class."""

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    def test_push_text(self, tmp_path):
        """Test push_text() accumulates text."""
        from noveum_trace.integrations.livekit.livekit_tts import (
            _WrappedSynthesizeStream,
        )

        mock_base_stream = Mock()
        mock_base_stream.push_text = Mock()

        stream = _WrappedSynthesizeStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        stream.push_text("Hello ")
        stream.push_text("world")

        assert stream._input_text == "Hello world"
        assert mock_base_stream.push_text.call_count == 2

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_tts.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_tts.upload_audio_frames")
    @patch("noveum_trace.integrations.livekit.livekit_tts.calculate_audio_duration_ms")
    @patch("noveum_trace.integrations.livekit.livekit_tts.create_span_attributes")
    @pytest.mark.asyncio
    async def test_anext_final_creates_span(
        self,
        mock_create_attrs,
        mock_calc_duration,
        mock_upload_audio,
        mock_get_client,
        mock_get_trace,
        tmp_path,
        mock_trace,
        mock_client,
    ):
        """Test __anext__ creates span when audio.is_final is True."""
        from noveum_trace.integrations.livekit.livekit_tts import (
            _WrappedSynthesizeStream,
        )

        mock_base_stream = AsyncMock()
        mock_audio = Mock()
        mock_audio.frame = Mock()
        mock_audio.is_final = True
        mock_audio.segment_id = "seg_123"
        mock_audio.request_id = "req_456"
        mock_base_stream.__anext__ = AsyncMock(return_value=mock_audio)

        stream = _WrappedSynthesizeStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        stream._input_text = "Hello world"

        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client
        mock_upload_audio.return_value = True
        mock_calc_duration.return_value = 3000.0
        mock_create_attrs.return_value = {"tts.provider": "cartesia"}
        
        # Mock span with trace_id and span_id
        mock_span = Mock()
        mock_span.trace_id = "trace-123"
        mock_span.span_id = "span-456"
        mock_client.start_span.return_value = mock_span

        result = await stream.__anext__()

        assert result == mock_audio
        mock_upload_audio.assert_called_once()
        mock_client.start_span.assert_called_once()
        assert len(stream._buffered_frames) == 0  # Buffer cleared
        assert stream._input_text == ""  # Text cleared

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_anext_non_final_no_span(self, tmp_path):
        """Test __anext__ doesn't create span when audio.is_final is False."""
        from noveum_trace.integrations.livekit.livekit_tts import (
            _WrappedSynthesizeStream,
        )

        mock_base_stream = AsyncMock()
        mock_audio = Mock()
        mock_audio.frame = Mock()
        mock_audio.is_final = False
        mock_base_stream.__anext__ = AsyncMock(return_value=mock_audio)

        stream = _WrappedSynthesizeStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        result = await stream.__anext__()

        assert result == mock_audio
        assert len(stream._buffered_frames) == 1  # Frame buffered but no span


@pytest.mark.skipif(
    not LIVEKIT_WRAPPER_AVAILABLE, reason="LiveKit wrappers not available"
)
class TestWrappedChunkedStream:
    """Test _WrappedChunkedStream class."""

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_tts.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_tts.upload_audio_frames")
    @patch("noveum_trace.integrations.livekit.livekit_tts.calculate_audio_duration_ms")
    @patch("noveum_trace.integrations.livekit.livekit_tts.create_span_attributes")
    @pytest.mark.asyncio
    async def test_anext_final_creates_span(
        self,
        mock_create_attrs,
        mock_calc_duration,
        mock_upload_audio,
        mock_get_client,
        mock_get_trace,
        tmp_path,
        mock_trace,
        mock_client,
    ):
        """Test __anext__ creates span when audio.is_final is True."""
        from noveum_trace.integrations.livekit.livekit_tts import _WrappedChunkedStream

        mock_base_stream = AsyncMock()
        mock_audio = Mock()
        mock_audio.frame = Mock()
        mock_audio.is_final = True
        mock_base_stream.__anext__ = AsyncMock(return_value=mock_audio)

        stream = _WrappedChunkedStream(
            base_stream=mock_base_stream,
            input_text="Hello world",
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client
        mock_upload_audio.return_value = True
        mock_calc_duration.return_value = 2500.0
        mock_create_attrs.return_value = {"tts.provider": "cartesia"}
        
        # Mock span with trace_id and span_id
        mock_span = Mock()
        mock_span.trace_id = "trace-123"
        mock_span.span_id = "span-456"
        mock_client.start_span.return_value = mock_span

        result = await stream.__anext__()

        assert result == mock_audio
        assert stream._span_created is True
        mock_upload_audio.assert_called_once()
        mock_client.start_span.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_aclose_creates_span_if_not_created(self, tmp_path):
        """Test aclose() creates span if span wasn't created during iteration."""
        from noveum_trace.integrations.livekit.livekit_tts import _WrappedChunkedStream

        mock_base_stream = AsyncMock()
        mock_base_stream.aclose = AsyncMock()

        stream = _WrappedChunkedStream(
            base_stream=mock_base_stream,
            input_text="Hello world",
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        stream._buffered_frames = [Mock(), Mock()]
        stream._span_created = False

        with patch.object(stream, "_create_span") as mock_create_span:
            await stream.aclose()

            mock_create_span.assert_called_once()
            mock_base_stream.aclose.assert_called_once()


@pytest.mark.skipif(
    not LIVEKIT_WRAPPER_AVAILABLE, reason="LiveKit wrappers not available"
)
class TestLiveKitSTTWrapperErrorHandling:
    """Test error handling in LiveKitSTTWrapper."""

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_stt.get_current_trace")
    @patch("noveum_trace.get_client")
    @pytest.mark.asyncio
    async def test_recognize_impl_span_creation_fails(
        self,
        mock_get_client,
        mock_get_trace,
        mock_stt_provider,
        mock_trace,
        mock_client,
        tmp_path,
    ):
        """Test _recognize_impl when span creation fails."""
        wrapper = LiveKitSTTWrapper(
            stt=mock_stt_provider, session_id="session_123", audio_base_dir=tmp_path
        )

        mock_event = Mock()
        mock_alternative = Mock()
        mock_alternative.text = "Hello"
        mock_alternative.confidence = 0.9
        mock_event.alternatives = [mock_alternative]
        mock_stt_provider._recognize_impl = AsyncMock(return_value=mock_event)

        mock_get_trace.return_value = mock_trace
        mock_get_client.side_effect = Exception("Client error")

        # Patch calculate_audio_duration_ms to avoid duration issues
        with patch(
            "noveum_trace.integrations.livekit.livekit_stt.calculate_audio_duration_ms",
            return_value=1000.0,
        ):
            # Should not raise exception
            result = await wrapper.recognize([Mock()])

        assert result == mock_event


@pytest.mark.skipif(
    not LIVEKIT_WRAPPER_AVAILABLE, reason="LiveKit wrappers not available"
)
class TestWrappedSpeechStreamErrorHandling:
    """Test error handling in _WrappedSpeechStream."""

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_stt.get_current_trace")
    @patch("noveum_trace.get_client")
    @pytest.mark.asyncio
    async def test_anext_span_creation_fails(
        self, mock_get_client, mock_get_trace, tmp_path, mock_trace
    ):
        """Test __anext__ when span creation fails."""
        from noveum_trace.integrations.livekit.livekit_stt import _WrappedSpeechStream

        mock_speech_event_type = MagicMock()
        mock_speech_event_type.FINAL_TRANSCRIPT = "FINAL_TRANSCRIPT"

        with patch(
            "noveum_trace.integrations.livekit.livekit_stt.SpeechEventType",
            mock_speech_event_type,
            create=True,
        ):
            mock_base_stream = AsyncMock()
            mock_event = Mock()
            mock_event.type = "FINAL_TRANSCRIPT"
            mock_event.alternatives = []
            mock_base_stream.__anext__ = AsyncMock(return_value=mock_event)

            stream = _WrappedSpeechStream(
                base_stream=mock_base_stream,
                session_id="session_123",
                job_context={},
                provider="deepgram",
                model="nova-2",
                counter_ref=[0],
                audio_dir=tmp_path,
            )

            stream._buffered_frames = [Mock()]

            mock_get_trace.return_value = mock_trace
            mock_get_client.side_effect = Exception("Client error")

            # Patch calculate_audio_duration_ms to avoid duration issues
            with patch(
                "noveum_trace.integrations.livekit.livekit_stt.calculate_audio_duration_ms",
                return_value=1000.0,
            ):
                # Should not raise exception
                result = await stream.__anext__()

            assert result == mock_event

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_aexit_fallback_to_aclose(self, tmp_path):
        """Test __aexit__ fallback to aclose when no __aexit__."""
        from noveum_trace.integrations.livekit.livekit_stt import _WrappedSpeechStream

        mock_base_stream = Mock()
        # No __aexit__ method
        mock_base_stream.aclose = AsyncMock()

        stream = _WrappedSpeechStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="deepgram",
            model="nova-2",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        await stream.__aexit__(None, None, None)

        mock_base_stream.aclose.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_stt.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_flush(self, tmp_path):
        """Test flush method."""
        from noveum_trace.integrations.livekit.livekit_stt import _WrappedSpeechStream

        mock_base_stream = Mock()
        mock_base_stream.flush = AsyncMock()

        stream = _WrappedSpeechStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="deepgram",
            model="nova-2",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        await stream.flush()

        mock_base_stream.flush.assert_called_once()


@pytest.mark.skipif(
    not LIVEKIT_WRAPPER_AVAILABLE, reason="LiveKit wrappers not available"
)
class TestWrappedSynthesizeStreamErrorHandling:
    """Test error handling in _WrappedSynthesizeStream."""

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_tts.get_current_trace")
    @patch("noveum_trace.get_client")
    @pytest.mark.asyncio
    async def test_anext_span_creation_fails(
        self, mock_get_client, mock_get_trace, tmp_path, mock_trace
    ):
        """Test __anext__ when span creation fails."""
        from noveum_trace.integrations.livekit.livekit_tts import (
            _WrappedSynthesizeStream,
        )

        mock_base_stream = AsyncMock()
        mock_audio = Mock()
        mock_audio.frame = Mock()
        mock_audio.is_final = True
        mock_base_stream.__anext__ = AsyncMock(return_value=mock_audio)

        stream = _WrappedSynthesizeStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        stream._input_text = "Hello"
        stream._buffered_frames = [Mock()]

        mock_get_trace.return_value = mock_trace
        mock_get_client.side_effect = Exception("Client error")

        # Patch calculate_audio_duration_ms to avoid duration issues
        with patch(
            "noveum_trace.integrations.livekit.livekit_tts.calculate_audio_duration_ms",
            return_value=1000.0,
        ):
            # Should not raise exception
            result = await stream.__anext__()

        assert result == mock_audio

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_tts.get_current_trace")
    @patch("noveum_trace.get_client")
    @pytest.mark.asyncio
    async def test_anext_with_delta_text_fallback(
        self, mock_get_client, mock_get_trace, tmp_path, mock_trace, mock_client
    ):
        """Test __anext__ with empty input_text and delta_text fallback."""
        from noveum_trace.integrations.livekit.livekit_tts import (
            _WrappedSynthesizeStream,
        )

        mock_base_stream = AsyncMock()
        mock_audio = Mock()
        mock_audio.frame = Mock()
        mock_audio.is_final = True
        mock_audio.delta_text = "Delta text"
        mock_base_stream.__anext__ = AsyncMock(return_value=mock_audio)

        stream = _WrappedSynthesizeStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        stream._input_text = ""  # Empty
        stream._buffered_frames = [Mock()]

        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client

        # Patch calculate_audio_duration_ms to avoid duration issues
        with patch(
            "noveum_trace.integrations.livekit.livekit_tts.calculate_audio_duration_ms",
            return_value=1000.0,
        ):
            result = await stream.__anext__()

        assert result == mock_audio

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_tts.get_current_trace")
    @patch("noveum_trace.get_client")
    @pytest.mark.asyncio
    async def test_anext_with_unknown_fallback(
        self, mock_get_client, mock_get_trace, tmp_path, mock_trace, mock_client
    ):
        """Test __anext__ with input_text='unknown' fallback."""
        from noveum_trace.integrations.livekit.livekit_tts import (
            _WrappedSynthesizeStream,
        )

        mock_base_stream = AsyncMock()
        mock_audio = Mock()
        mock_audio.frame = Mock()
        mock_audio.is_final = True
        # No delta_text
        mock_base_stream.__anext__ = AsyncMock(return_value=mock_audio)

        stream = _WrappedSynthesizeStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        stream._input_text = ""  # Empty
        stream._buffered_frames = [Mock()]

        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client

        # Patch calculate_audio_duration_ms to avoid duration issues
        with patch(
            "noveum_trace.integrations.livekit.livekit_tts.calculate_audio_duration_ms",
            return_value=1000.0,
        ):
            result = await stream.__anext__()

        assert result == mock_audio

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_aexit_fallback_to_aclose(self, tmp_path):
        """Test __aexit__ fallback to aclose when no __aexit__."""
        from noveum_trace.integrations.livekit.livekit_tts import (
            _WrappedSynthesizeStream,
        )

        mock_base_stream = Mock()
        # No __aexit__ method
        mock_base_stream.aclose = AsyncMock()

        stream = _WrappedSynthesizeStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        await stream.__aexit__(None, None, None)

        mock_base_stream.aclose.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_flush(self, tmp_path):
        """Test flush method."""
        from noveum_trace.integrations.livekit.livekit_tts import (
            _WrappedSynthesizeStream,
        )

        mock_base_stream = Mock()
        mock_base_stream.flush = AsyncMock()

        stream = _WrappedSynthesizeStream(
            base_stream=mock_base_stream,
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        await stream.flush()

        mock_base_stream.flush.assert_called_once()


@pytest.mark.skipif(
    not LIVEKIT_WRAPPER_AVAILABLE, reason="LiveKit wrappers not available"
)
class TestWrappedChunkedStreamErrorHandling:
    """Test error handling in _WrappedChunkedStream."""

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_tts.get_current_trace")
    @patch("noveum_trace.get_client")
    @pytest.mark.asyncio
    async def test_create_span_span_creation_fails(
        self, mock_get_client, mock_get_trace, tmp_path, mock_trace
    ):
        """Test _create_span when span creation fails."""
        from noveum_trace.integrations.livekit.livekit_tts import _WrappedChunkedStream

        mock_base_stream = AsyncMock()
        stream = _WrappedChunkedStream(
            base_stream=mock_base_stream,
            input_text="Hello",
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        stream._buffered_frames = [Mock()]

        mock_get_trace.return_value = mock_trace
        mock_get_client.side_effect = Exception("Client error")

        # Patch calculate_audio_duration_ms to avoid duration issues
        with patch(
            "noveum_trace.integrations.livekit.livekit_tts.calculate_audio_duration_ms",
            return_value=1000.0,
        ):
            # Should not raise exception
            stream._create_span()

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_anext_non_final_no_span(self, tmp_path):
        """Test __anext__ with non-final audio (buffer frames but no span)."""
        from noveum_trace.integrations.livekit.livekit_tts import _WrappedChunkedStream

        mock_base_stream = AsyncMock()
        mock_audio = Mock()
        mock_audio.frame = Mock()
        mock_audio.is_final = False
        mock_base_stream.__anext__ = AsyncMock(return_value=mock_audio)

        stream = _WrappedChunkedStream(
            base_stream=mock_base_stream,
            input_text="Hello",
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        result = await stream.__anext__()

        assert result == mock_audio
        assert len(stream._buffered_frames) == 1
        assert stream._span_created is False

    @patch("noveum_trace.integrations.livekit.livekit_tts.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_aclose_when_span_already_created(self, tmp_path):
        """Test aclose when span already created."""
        from noveum_trace.integrations.livekit.livekit_tts import _WrappedChunkedStream

        mock_base_stream = AsyncMock()
        mock_base_stream.aclose = AsyncMock()

        stream = _WrappedChunkedStream(
            base_stream=mock_base_stream,
            input_text="Hello",
            session_id="session_123",
            job_context={},
            provider="cartesia",
            model="sonic",
            counter_ref=[0],
            audio_dir=tmp_path,
        )

        stream._span_created = True

        await stream.aclose()

        mock_base_stream.aclose.assert_called_once()
