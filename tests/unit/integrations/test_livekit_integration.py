"""
Unit tests for LiveKit STT/TTS integration.

Tests the wrapper classes that automatically trace LiveKit STT and TTS operations.
"""

from typing import Any
from unittest.mock import patch

import pytest

import noveum_trace
from noveum_trace.core.context import set_current_trace

# Check if LiveKit is available
try:
    from livekit.agents.stt import (
        SpeechData,
        SpeechEvent,
        SpeechEventType,
        STTCapabilities,
    )
    from livekit.agents.tts import SynthesizedAudio, TTSCapabilities
    from livekit.agents.utils import AudioBuffer

    # Import wrappers (only when LiveKit is available)
    from noveum_trace.integrations.livekit import (
        LiveKitSTTWrapper,
        LiveKitTTSWrapper,
    )
    from noveum_trace.integrations.livekit.livekit_utils import (
        calculate_audio_duration_ms,
        create_span_attributes,
        ensure_audio_directory,
        extract_job_context,
        generate_audio_filename,
    )

    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="LiveKit not installed")


# Mock audio frame
class MockAudioFrame:
    """Mock rtc.AudioFrame for testing."""

    def __init__(self, duration: float = 0.1, data: bytes = b"mock_audio"):
        self.duration = duration
        self.data = data
        self.sample_rate = 16000
        self.num_channels = 1
        self.samples_per_channel = int(duration * self.sample_rate)


# Mock STT provider
class MockSTT:
    """Mock LiveKit STT provider for testing."""

    def __init__(self, streaming: bool = True):
        self.capabilities = STTCapabilities(
            streaming=streaming, interim_results=True, diarization=False
        )
        self.model = "test-model"
        self.provider = "test-provider"
        self.label = "test.MockSTT"
        self._recognize_calls = []
        self._stream_calls = []

    async def _recognize_impl(self, buffer: AudioBuffer, **kwargs: Any) -> SpeechEvent:
        """Mock batch recognition."""
        self._recognize_calls.append({"buffer": buffer, "kwargs": kwargs})

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id="test-request-123",
            alternatives=[
                SpeechData(
                    language="en",
                    text="Hello, this is a test transcript",
                    start_time=0.0,
                    end_time=2.0,
                    confidence=0.95,
                )
            ],
        )

    def stream(self, **kwargs: Any) -> "MockSpeechStream":
        """Mock streaming recognition."""
        self._stream_calls.append(kwargs)
        return MockSpeechStream()


class MockSpeechStream:
    """Mock speech stream for testing."""

    def __init__(self):
        self.frames = []
        self.events = [
            SpeechEvent(
                type=SpeechEventType.INTERIM_TRANSCRIPT,
                request_id="test-request-1",
                alternatives=[SpeechData(language="en", text="Hello", confidence=0.8)],
            ),
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                request_id="test-request-1",
                alternatives=[
                    SpeechData(
                        language="en",
                        text="Hello, this is a test",
                        start_time=0.0,
                        end_time=1.5,
                        confidence=0.95,
                    )
                ],
            ),
        ]
        self._event_index = 0

    def push_frame(self, frame: Any) -> None:
        """Mock push frame."""
        self.frames.append(frame)

    async def __anext__(self) -> SpeechEvent:
        """Mock async iteration."""
        if self._event_index >= len(self.events):
            raise StopAsyncIteration

        event = self.events[self._event_index]
        self._event_index += 1
        return event

    def __aiter__(self):
        return self

    async def flush(self) -> None:
        """Mock flush."""
        pass

    async def aclose(self) -> None:
        """Mock close."""
        pass


# Mock TTS provider
class MockTTS:
    """Mock LiveKit TTS provider for testing."""

    def __init__(self, streaming: bool = True):
        self.capabilities = TTSCapabilities(
            streaming=streaming, aligned_transcript=False
        )
        self.model = "test-tts-model"
        self.provider = "test-tts-provider"
        self.label = "test.MockTTS"
        self.sample_rate = 24000
        self.num_channels = 1
        self._synthesize_calls = []
        self._stream_calls = []

    def synthesize(self, text: str, **kwargs: Any) -> "MockChunkedStream":
        """Mock batch synthesis."""
        self._synthesize_calls.append({"text": text, "kwargs": kwargs})
        return MockChunkedStream(text)

    def stream(self, **kwargs: Any) -> "MockSynthesizeStream":
        """Mock streaming synthesis."""
        self._stream_calls.append(kwargs)
        return MockSynthesizeStream()


class MockChunkedStream:
    """Mock chunked stream for batch TTS."""

    def __init__(self, text: str):
        self.text = text
        self.audio_chunks = [
            SynthesizedAudio(
                frame=MockAudioFrame(duration=0.5),
                request_id="tts-req-1",
                is_final=False,
                segment_id="seg-1",
                delta_text=text[:10],
            ),
            SynthesizedAudio(
                frame=MockAudioFrame(duration=0.5),
                request_id="tts-req-1",
                is_final=True,
                segment_id="seg-1",
                delta_text=text[10:],
            ),
        ]
        self._chunk_index = 0

    async def __anext__(self) -> SynthesizedAudio:
        """Mock async iteration."""
        if self._chunk_index >= len(self.audio_chunks):
            raise StopAsyncIteration

        chunk = self.audio_chunks[self._chunk_index]
        self._chunk_index += 1
        return chunk

    def __aiter__(self):
        return self

    async def aclose(self) -> None:
        """Mock close."""
        pass


class MockSynthesizeStream:
    """Mock synthesize stream for streaming TTS."""

    def __init__(self):
        self.text_pushed = []
        self.audio_chunks = []
        self._started = False

    def push_text(self, text: str) -> None:
        """Mock push text."""
        self.text_pushed.append(text)
        self._started = True
        # Create audio chunks
        self.audio_chunks = [
            SynthesizedAudio(
                frame=MockAudioFrame(duration=0.3),
                request_id="tts-stream-1",
                is_final=False,
                segment_id="stream-seg-1",
                delta_text=text[:5],
            ),
            SynthesizedAudio(
                frame=MockAudioFrame(duration=0.3),
                request_id="tts-stream-1",
                is_final=True,
                segment_id="stream-seg-1",
                delta_text=text[5:],
            ),
        ]
        self._chunk_index = 0

    async def __anext__(self) -> SynthesizedAudio:
        """Mock async iteration."""
        if not self._started or self._chunk_index >= len(self.audio_chunks):
            raise StopAsyncIteration

        chunk = self.audio_chunks[self._chunk_index]
        self._chunk_index += 1
        return chunk

    def __aiter__(self):
        return self

    async def flush(self) -> None:
        """Mock flush."""
        pass

    async def aclose(self) -> None:
        """Mock close."""
        pass


# Fixtures


@pytest.fixture
def temp_audio_dir(tmp_path):
    """Create temporary audio directory."""
    audio_dir = tmp_path / "audio_files"
    audio_dir.mkdir()
    return audio_dir


@pytest.fixture
def mock_stt_provider():
    """Create mock STT provider."""
    return MockSTT(streaming=True)


@pytest.fixture
def mock_tts_provider():
    """Create mock TTS provider."""
    return MockTTS(streaming=True)


@pytest.fixture
def job_context():
    """Create test job context."""
    return {
        "job_id": "test-job-123",
        "room_name": "test-room",
        "agent_id": "agent-456",
        "worker_id": "worker-789",
    }


@pytest.fixture
def initialized_client():
    """Initialize noveum_trace client for testing."""
    noveum_trace.init(project="test-livekit", api_key="test-key")
    yield
    try:
        noveum_trace.shutdown()
    except Exception:
        pass


# Utility Function Tests


def test_generate_audio_filename():
    """Test audio filename generation."""
    filename = generate_audio_filename("stt", 1, timestamp=1732386400000)
    assert filename == "stt_0001_1732386400000.wav"

    filename = generate_audio_filename("tts", 99)
    assert filename.startswith("tts_0099_")
    assert filename.endswith(".wav")


def test_ensure_audio_directory(temp_audio_dir):
    """Test audio directory creation."""
    session_id = "test-session-123"
    audio_dir = ensure_audio_directory(session_id, base_dir=temp_audio_dir)

    assert audio_dir.exists()
    assert audio_dir.is_dir()
    assert audio_dir.name == session_id


def test_calculate_audio_duration_ms():
    """Test audio duration calculation."""
    frames = [
        MockAudioFrame(duration=0.1),
        MockAudioFrame(duration=0.2),
        MockAudioFrame(duration=0.15),
    ]

    duration_ms = calculate_audio_duration_ms(frames)
    # Use approximate comparison for floats
    assert abs(duration_ms - 450.0) < 0.01


def test_extract_job_context():
    """Test job context extraction."""

    class MockJob:
        id = "job-123"

        class room:
            sid = "room-sid-456"
            name = "test-room"

    class MockContext:
        job = MockJob()

        class room:
            name = "test-room-2"
            sid = "room-sid-789"

        worker_id = "worker-123"

    context = extract_job_context(MockContext())

    assert context["job_id"] == "job-123"
    assert context["job_room_sid"] == "room-sid-456"
    assert context["room_name"] == "test-room-2"
    assert context["worker_id"] == "worker-123"


def test_create_span_attributes(job_context):
    """Test span attributes creation."""
    attributes = create_span_attributes(
        provider="deepgram",
        model="nova-2",
        operation_type="stt",
        audio_file="stt_0001_1234.wav",
        audio_duration_ms=1500.0,
        job_context=job_context,
        transcript="Hello world",
        confidence=0.95,
    )

    assert attributes["stt.provider"] == "deepgram"
    assert attributes["stt.model"] == "nova-2"
    assert attributes["stt.audio_file"] == "stt_0001_1234.wav"
    assert attributes["stt.audio_duration_ms"] == 1500.0
    assert attributes["transcript"] == "Hello world"
    assert attributes["confidence"] == 0.95
    # job_context keys: job_id becomes job.id (job_ prefix stripped and job. added)
    assert attributes["job.id"] == "test-job-123"
    # room_name becomes job.room_name (job. prefix added)
    assert attributes["job.room_name"] == "test-room"


# STT Wrapper Tests


@pytest.mark.skipif(not LIVEKIT_AVAILABLE, reason="LiveKit not installed")
def test_stt_wrapper_initialization(mock_stt_provider, temp_audio_dir, job_context):
    """Test STT wrapper initialization."""
    wrapper = LiveKitSTTWrapper(
        stt=mock_stt_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    assert wrapper.model == "test-model"
    assert wrapper.provider == "test-provider"
    assert wrapper.capabilities.streaming is True
    assert wrapper._counter_ref[0] == 0


@pytest.mark.skipif(not LIVEKIT_AVAILABLE, reason="LiveKit not installed")
@pytest.mark.asyncio
async def test_stt_wrapper_batch_recognition(
    mock_stt_provider, temp_audio_dir, job_context, initialized_client
):
    """Test STT batch recognition with tracing."""
    wrapper = LiveKitSTTWrapper(
        stt=mock_stt_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    # Create a trace context
    client = noveum_trace.get_client()
    trace = client.start_trace("test-stt-trace")
    set_current_trace(trace)

    # Mock audio buffer - use list as LiveKit's AudioBuffer is essentially a list
    buffer = [MockAudioFrame(duration=0.5), MockAudioFrame(duration=0.5)]

    # Mock save_audio_buffer to avoid file I/O issues
    with patch("noveum_trace.integrations.livekit.save_audio_buffer"):
        # Call recognize
        event = await wrapper.recognize(buffer)

    # Verify event
    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "Hello, this is a test transcript"

    # Verify counter incremented
    assert wrapper._counter_ref[0] == 1

    # Verify trace has spans
    assert len(trace.spans) > 0

    # Clean up
    set_current_trace(None)
    client.finish_trace(trace)


@pytest.mark.skipif(not LIVEKIT_AVAILABLE, reason="LiveKit not installed")
@pytest.mark.asyncio
async def test_stt_wrapper_streaming(
    mock_stt_provider, temp_audio_dir, job_context, initialized_client
):
    """Test STT streaming with tracing."""
    wrapper = LiveKitSTTWrapper(
        stt=mock_stt_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    # Create a trace context
    client = noveum_trace.get_client()
    trace = client.start_trace("test-stt-stream-trace")
    set_current_trace(trace)

    # Get stream
    stream = wrapper.stream()

    # Push frames
    stream.push_frame(MockAudioFrame(duration=0.1))
    stream.push_frame(MockAudioFrame(duration=0.2))

    # Mock save_audio_frames
    with patch("noveum_trace.integrations.livekit.save_audio_frames"):
        # Consume events
        events = []
        async for event in stream:
            events.append(event)

    # Verify events
    assert len(events) == 2  # interim + final
    assert events[0].type == SpeechEventType.INTERIM_TRANSCRIPT
    assert events[1].type == SpeechEventType.FINAL_TRANSCRIPT

    # Verify counter incremented (only for final)
    assert wrapper._counter_ref[0] == 1

    # Verify trace has spans
    assert len(trace.spans) > 0

    # Clean up
    set_current_trace(None)
    client.finish_trace(trace)


@pytest.mark.skipif(not LIVEKIT_AVAILABLE, reason="LiveKit not installed")
@pytest.mark.asyncio
async def test_stt_wrapper_no_trace(
    mock_stt_provider, temp_audio_dir, job_context, initialized_client
):
    """Test STT wrapper gracefully handles missing trace."""
    wrapper = LiveKitSTTWrapper(
        stt=mock_stt_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    # No trace set
    set_current_trace(None)

    # Mock audio buffer - use list as LiveKit's AudioBuffer is essentially a list
    buffer = [MockAudioFrame(duration=0.5)]

    # Mock save_audio_buffer
    with patch("noveum_trace.integrations.livekit.save_audio_buffer"):
        # Should not raise error
        event = await wrapper.recognize(buffer)

    # Verify event still returned
    assert event.type == SpeechEventType.FINAL_TRANSCRIPT


# TTS Wrapper Tests


@pytest.mark.skipif(not LIVEKIT_AVAILABLE, reason="LiveKit not installed")
def test_tts_wrapper_initialization(mock_tts_provider, temp_audio_dir, job_context):
    """Test TTS wrapper initialization."""
    wrapper = LiveKitTTSWrapper(
        tts=mock_tts_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    assert wrapper.model == "test-tts-model"
    assert wrapper.provider == "test-tts-provider"
    assert wrapper.capabilities.streaming is True
    assert wrapper.sample_rate == 24000
    assert wrapper._counter_ref[0] == 0


@pytest.mark.skipif(not LIVEKIT_AVAILABLE, reason="LiveKit not installed")
@pytest.mark.asyncio
async def test_tts_wrapper_batch_synthesis(
    mock_tts_provider, temp_audio_dir, job_context, initialized_client
):
    """Test TTS batch synthesis with tracing."""
    wrapper = LiveKitTTSWrapper(
        tts=mock_tts_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    # Create a trace context
    client = noveum_trace.get_client()
    trace = client.start_trace("test-tts-trace")
    set_current_trace(trace)

    # Mock save_audio_frames
    with patch("noveum_trace.integrations.livekit.save_audio_frames"):
        # Synthesize
        stream = wrapper.synthesize("Hello, this is a test")

        # Consume audio chunks
        chunks = []
        async for audio in stream:
            chunks.append(audio)

    # Verify chunks
    assert len(chunks) == 2
    assert chunks[1].is_final is True

    # Verify counter incremented
    assert wrapper._counter_ref[0] == 1

    # Verify trace has spans
    assert len(trace.spans) > 0

    # Clean up
    set_current_trace(None)
    client.finish_trace(trace)


@pytest.mark.skipif(not LIVEKIT_AVAILABLE, reason="LiveKit not installed")
@pytest.mark.asyncio
async def test_tts_wrapper_streaming(
    mock_tts_provider, temp_audio_dir, job_context, initialized_client
):
    """Test TTS streaming with tracing."""
    wrapper = LiveKitTTSWrapper(
        tts=mock_tts_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    # Create a trace context
    client = noveum_trace.get_client()
    trace = client.start_trace("test-tts-stream-trace")
    set_current_trace(trace)

    # Get stream
    stream = wrapper.stream()
    stream.push_text("Hello world")

    # Mock save_audio_frames
    with patch("noveum_trace.integrations.livekit.save_audio_frames"):
        # Consume audio chunks
        chunks = []
        async for audio in stream:
            chunks.append(audio)

    # Verify chunks
    assert len(chunks) == 2
    assert chunks[1].is_final is True

    # Verify counter incremented
    assert wrapper._counter_ref[0] == 1

    # Verify trace has spans
    assert len(trace.spans) > 0

    # Clean up
    set_current_trace(None)
    client.finish_trace(trace)


@pytest.mark.skipif(not LIVEKIT_AVAILABLE, reason="LiveKit not installed")
@pytest.mark.asyncio
async def test_tts_wrapper_no_trace(
    mock_tts_provider, temp_audio_dir, job_context, initialized_client
):
    """Test TTS wrapper gracefully handles missing trace."""
    wrapper = LiveKitTTSWrapper(
        tts=mock_tts_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    # No trace set
    set_current_trace(None)

    # Mock save_audio_frames
    with patch("noveum_trace.integrations.livekit.save_audio_frames"):
        # Should not raise error
        stream = wrapper.synthesize("Test text")

        chunks = []
        async for audio in stream:
            chunks.append(audio)

    # Verify audio still returned
    assert len(chunks) > 0


# Integration Tests


@pytest.mark.skipif(not LIVEKIT_AVAILABLE, reason="LiveKit not installed")
@pytest.mark.asyncio
async def test_multiple_stt_operations(
    mock_stt_provider, temp_audio_dir, job_context, initialized_client
):
    """Test multiple STT operations increment counter correctly."""
    wrapper = LiveKitSTTWrapper(
        stt=mock_stt_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    client = noveum_trace.get_client()
    trace = client.start_trace("multi-stt-trace")
    set_current_trace(trace)

    buffer = [MockAudioFrame(duration=0.5)]

    with patch("noveum_trace.integrations.livekit.save_audio_buffer"):
        # First recognition
        await wrapper.recognize(buffer)
        assert wrapper._counter_ref[0] == 1

        # Second recognition
        await wrapper.recognize(buffer)
        assert wrapper._counter_ref[0] == 2

        # Third recognition
        await wrapper.recognize(buffer)
        assert wrapper._counter_ref[0] == 3

    # Verify trace has 3 spans
    assert len(trace.spans) >= 3

    # Clean up
    set_current_trace(None)
    client.finish_trace(trace)


@pytest.mark.skipif(not LIVEKIT_AVAILABLE, reason="LiveKit not installed")
@pytest.mark.asyncio
async def test_stt_and_tts_together(
    mock_stt_provider,
    mock_tts_provider,
    temp_audio_dir,
    job_context,
    initialized_client,
):
    """Test STT and TTS wrappers work together in same session."""
    stt_wrapper = LiveKitSTTWrapper(
        stt=mock_stt_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    tts_wrapper = LiveKitTTSWrapper(
        tts=mock_tts_provider,
        session_id="test-session",
        job_context=job_context,
        audio_base_dir=temp_audio_dir,
    )

    client = noveum_trace.get_client()
    trace = client.start_trace("stt-tts-trace")
    set_current_trace(trace)

    with (
        patch("noveum_trace.integrations.livekit.save_audio_buffer"),
        patch("noveum_trace.integrations.livekit.save_audio_frames"),
    ):

        # STT operation
        buffer = [MockAudioFrame(duration=0.5)]
        stt_event = await stt_wrapper.recognize(buffer)

        # TTS operation
        tts_stream = tts_wrapper.synthesize("Response text")
        chunks = []
        async for audio in tts_stream:
            chunks.append(audio)

    # Verify both operations succeeded
    assert stt_event.alternatives[0].text is not None
    assert len(chunks) > 0

    # Verify counters are independent
    assert stt_wrapper._counter_ref[0] == 1
    assert tts_wrapper._counter_ref[0] == 1

    # Verify trace has spans from both
    assert len(trace.spans) >= 2

    # Clean up
    set_current_trace(None)
    client.finish_trace(trace)
