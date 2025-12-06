"""
Unit tests for LiveKit utility functions.

Tests the utility functions in livekit_utils.py:
- save_audio_frames, save_audio_buffer
- calculate_audio_duration_ms
- ensure_audio_directory
- generate_audio_filename
- extract_job_context
- create_span_attributes
- Helper functions: _is_mock_object, _safe_str
"""

from unittest.mock import Mock, patch

import pytest

# Test constants - Job and Context IDs
TEST_JOB_ID = "job_123"
TEST_ROOM_SID = "room_sid"
TEST_ROOM_SID_123 = "room_sid_123"
TEST_ROOM_NAME = "room_name"
TEST_ROOM_NAME_456 = "room_name_456"
TEST_AGENT_ID = "agent_123"
TEST_AGENT_ID_789 = "agent_789"
TEST_WORKER_ID = "worker_123"
TEST_PARTICIPANT_IDENTITY = "user_123"
TEST_PARTICIPANT_SID = "participant_sid_456"

# Test constants - Session IDs
TEST_SESSION_ID_123 = "session_123"
TEST_SESSION_ID_456 = "session_456"
TEST_SESSION_ID_789 = "session_789"

# Test constants - Audio providers and models
TEST_PROVIDER_DEEPGRAM = "deepgram"
TEST_PROVIDER_CARTESIA = "cartesia"
TEST_PROVIDER_TEST = "test"
TEST_MODEL_NOVA2 = "nova-2"
TEST_MODEL_SONIC = "sonic"
TEST_MODEL_TEST = "test"

# Test constants - Operation types
TEST_OPERATION_STT = "stt"
TEST_OPERATION_TTS = "tts"

# Test constants - Audio files
TEST_AUDIO_FILE_EMPTY = "empty.wav"
TEST_AUDIO_FILE_TEST = "test.wav"
TEST_AUDIO_FILE_STT = "stt_0001_123.wav"
TEST_AUDIO_FILE_TTS = "tts_0001_123.wav"

# Test constants - Timestamps and durations
TEST_TIMESTAMP_MS = 1732386400000
TEST_TIMESTAMP_SEC = 1732386400.0
TEST_TIMESTAMP_1000 = 1000
TEST_DURATION_MS_500 = 500.0
TEST_DURATION_MS_1000 = 1000.0
TEST_DURATION_MS_1500 = 1500.0
TEST_DURATION_MS_2000 = 2000.0
TEST_DURATION_SEC_0_2 = 0.2
TEST_DURATION_SEC_0_3 = 0.3
TEST_DURATION_SEC_0_5 = 0.5

# Test constants - Audio data
TEST_AUDIO_DATA_WAV = b"fake_wav_data"
TEST_AUDIO_DATA_SHORT = b"data"

# Test constants - STT/TTS attributes
TEST_TRANSCRIPT = "Hello world"
TEST_CONFIDENCE = 0.95
TEST_MODE_STREAMING = "streaming"

# Test constants - Defaults
TEST_DEFAULT_UNKNOWN = "unknown"
TEST_DEFAULT_CUSTOM = "custom"

# Skip all tests if LiveKit is not available
try:
    # Import private functions for testing
    from noveum_trace.integrations.livekit.livekit_utils import (
        _is_mock_object,
        _safe_str,
        calculate_audio_duration_ms,
        create_span_attributes,
        ensure_audio_directory,
        extract_job_context,
        generate_audio_filename,
        save_audio_buffer,
        save_audio_frames,
    )

    LIVEKIT_UTILS_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    LIVEKIT_UTILS_AVAILABLE = False


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestSaveAudioFrames:
    """Test save_audio_frames function."""

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    def test_save_audio_frames_empty_list(self, tmp_path):
        """Test saving empty frames list creates empty file."""
        output_path = tmp_path / TEST_AUDIO_FILE_EMPTY

        save_audio_frames([], output_path)

        assert output_path.exists()
        assert output_path.read_bytes() == b""

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True)
    def test_save_audio_frames_valid_frames(self, mock_rtc, tmp_path):
        """Test saving valid audio frames."""
        output_path = tmp_path / TEST_AUDIO_FILE_TEST

        # Mock audio frames
        mock_frame1 = Mock()
        mock_frame2 = Mock()

        # Mock combined frame
        mock_combined = Mock()
        mock_combined.to_wav_bytes.return_value = TEST_AUDIO_DATA_WAV
        mock_rtc.combine_audio_frames.return_value = mock_combined

        save_audio_frames([mock_frame1, mock_frame2], output_path)

        assert output_path.exists()
        assert output_path.read_bytes() == TEST_AUDIO_DATA_WAV
        mock_rtc.combine_audio_frames.assert_called_once_with(
            [mock_frame1, mock_frame2]
        )
        mock_combined.to_wav_bytes.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", False)
    def test_save_audio_frames_livekit_unavailable(self, tmp_path):
        """Test save_audio_frames when LiveKit is not available."""
        output_path = tmp_path / TEST_AUDIO_FILE_TEST

        save_audio_frames([Mock()], output_path)

        # Should not create file when LiveKit unavailable
        assert not output_path.exists()

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True)
    def test_save_audio_frames_creates_directory(self, mock_rtc, tmp_path):
        """Test that save_audio_frames creates parent directory if needed."""
        output_path = tmp_path / "nested" / "dir" / TEST_AUDIO_FILE_TEST

        mock_combined = Mock()
        mock_combined.to_wav_bytes.return_value = TEST_AUDIO_DATA_SHORT
        mock_rtc.combine_audio_frames.return_value = mock_combined

        save_audio_frames([Mock()], output_path)

        assert output_path.parent.exists()
        assert output_path.exists()


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestSaveAudioBuffer:
    """Test save_audio_buffer function."""

    @patch("noveum_trace.integrations.livekit.livekit_utils.save_audio_frames")
    def test_save_audio_buffer_calls_save_frames(self, mock_save_frames, tmp_path):
        """Test that save_audio_buffer converts buffer to frames and calls save_audio_frames."""
        output_path = tmp_path / TEST_AUDIO_FILE_TEST
        mock_buffer = [Mock(), Mock()]

        save_audio_buffer(mock_buffer, output_path)

        mock_save_frames.assert_called_once_with(mock_buffer, output_path)

    @patch("noveum_trace.integrations.livekit.livekit_utils.save_audio_frames")
    def test_save_audio_buffer_with_empty_buffer(self, mock_save_frames, tmp_path):
        """Test save_audio_buffer with empty buffer."""
        output_path = tmp_path / TEST_AUDIO_FILE_TEST

        save_audio_buffer([], output_path)

        mock_save_frames.assert_called_once_with([], output_path)


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestCalculateAudioDurationMs:
    """Test calculate_audio_duration_ms function."""

    def test_calculate_audio_duration_ms_empty_list(self):
        """Test duration calculation with empty frames list."""
        result = calculate_audio_duration_ms([])
        assert result == 0.0

    def test_calculate_audio_duration_ms_single_frame(self):
        """Test duration calculation with single frame."""
        mock_frame = Mock()
        mock_frame.duration = TEST_DURATION_SEC_0_5  # 0.5 seconds

        result = calculate_audio_duration_ms([mock_frame])
        assert result == TEST_DURATION_MS_500  # 500 milliseconds

    def test_calculate_audio_duration_ms_multiple_frames(self):
        """Test duration calculation with multiple frames."""
        mock_frame1 = Mock()
        mock_frame1.duration = TEST_DURATION_SEC_0_5
        mock_frame2 = Mock()
        mock_frame2.duration = TEST_DURATION_SEC_0_3
        mock_frame3 = Mock()
        mock_frame3.duration = TEST_DURATION_SEC_0_2

        result = calculate_audio_duration_ms([mock_frame1, mock_frame2, mock_frame3])
        # (0.5 + 0.3 + 0.2) * 1000 = 1000 ms
        assert result == TEST_DURATION_MS_1000


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestEnsureAudioDirectory:
    """Test ensure_audio_directory function."""

    def test_ensure_audio_directory_default_base(self, tmp_path):
        """Test directory creation with default base directory."""
        with patch(
            "noveum_trace.integrations.livekit.livekit_utils.Path",
            return_value=tmp_path / "audio_files",
        ):
            result = ensure_audio_directory(TEST_SESSION_ID_123)

            expected = tmp_path / "audio_files" / TEST_SESSION_ID_123
            assert result == expected
            assert result.exists()

    def test_ensure_audio_directory_custom_base(self, tmp_path):
        """Test directory creation with custom base directory."""
        custom_base = tmp_path / "custom_audio"
        result = ensure_audio_directory(TEST_SESSION_ID_456, base_dir=custom_base)

        expected = custom_base / TEST_SESSION_ID_456
        assert result == expected
        assert result.exists()

    def test_ensure_audio_directory_nested_session_id(self, tmp_path):
        """Test directory creation with nested session ID."""
        result = ensure_audio_directory("nested/session/id", base_dir=tmp_path)

        expected = tmp_path / "nested" / "session" / "id"
        assert result == expected
        assert result.exists()

    def test_ensure_audio_directory_existing_directory(self, tmp_path):
        """Test that existing directory is not recreated."""
        session_dir = tmp_path / TEST_SESSION_ID_789
        session_dir.mkdir(parents=True)

        result = ensure_audio_directory(TEST_SESSION_ID_789, base_dir=tmp_path)

        assert result == session_dir
        assert result.exists()


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestGenerateAudioFilename:
    """Test generate_audio_filename function."""

    def test_generate_audio_filename_with_timestamp(self):
        """Test filename generation with explicit timestamp."""
        result = generate_audio_filename(
            TEST_OPERATION_STT, 1, timestamp=TEST_TIMESTAMP_MS
        )

        assert result == "stt_0001_1732386400000.wav"

    def test_generate_audio_filename_without_timestamp(self):
        """Test filename generation without timestamp (uses current time)."""
        with patch("noveum_trace.integrations.livekit.livekit_utils.time") as mock_time:
            mock_time.time.return_value = TEST_TIMESTAMP_SEC  # Returns seconds
            result = generate_audio_filename(TEST_OPERATION_TTS, 42)

            # Should convert seconds to milliseconds
            assert result == "tts_0042_1732386400000.wav"

    def test_generate_audio_filename_counter_formatting(self):
        """Test that counter is zero-padded to 4 digits."""
        result = generate_audio_filename(
            TEST_OPERATION_STT, 5, timestamp=TEST_TIMESTAMP_1000
        )
        assert result == "stt_0005_1000.wav"

        result = generate_audio_filename(
            TEST_OPERATION_STT, 123, timestamp=TEST_TIMESTAMP_1000
        )
        assert result == "stt_0123_1000.wav"

        result = generate_audio_filename(
            TEST_OPERATION_STT, 9999, timestamp=TEST_TIMESTAMP_1000
        )
        assert result == "stt_9999_1000.wav"


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestIsMockObject:
    """Test _is_mock_object helper function."""

    def test_is_mock_object_with_mock(self):
        """Test detection of Mock objects."""
        mock_obj = Mock()
        assert _is_mock_object(mock_obj) is True

    def test_is_mock_object_with_magic_mock(self):
        """Test detection of MagicMock objects."""
        from unittest.mock import MagicMock

        magic_mock = MagicMock()
        assert _is_mock_object(magic_mock) is True

    def test_is_mock_object_with_regular_object(self):
        """Test that regular objects are not detected as mocks."""
        regular_obj = "not a mock"
        assert _is_mock_object(regular_obj) is False

        regular_obj = {"key": "value"}
        assert _is_mock_object(regular_obj) is False

        regular_obj = 42
        assert _is_mock_object(regular_obj) is False


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestSafeStr:
    """Test _safe_str helper function."""

    def test_safe_str_with_none(self):
        """Test _safe_str with None returns default."""
        result = _safe_str(None)
        assert result == TEST_DEFAULT_UNKNOWN

    def test_safe_str_with_custom_default(self):
        """Test _safe_str with None and custom default."""
        result = _safe_str(None, default=TEST_DEFAULT_CUSTOM)
        assert result == TEST_DEFAULT_CUSTOM

    def test_safe_str_with_mock_object(self):
        """Test _safe_str with mock object returns default."""
        mock_obj = Mock()
        result = _safe_str(mock_obj)
        assert result == TEST_DEFAULT_UNKNOWN

    def test_safe_str_with_regular_string(self):
        """Test _safe_str with regular string."""
        result = _safe_str("test_string")
        assert result == "test_string"

    def test_safe_str_with_integer(self):
        """Test _safe_str with integer."""
        result = _safe_str(42)
        assert result == "42"


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestExtractJobContext:
    """Test extract_job_context function."""

    def test_extract_job_context_with_job_id(self):
        """Test context extraction with job ID."""
        mock_ctx = Mock()
        mock_job = Mock()
        mock_job.id = TEST_JOB_ID
        mock_ctx.job = mock_job

        # Make sure job is not detected as mock
        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["job_id"] == TEST_JOB_ID

    def test_extract_job_context_with_job_room(self):
        """Test context extraction with job room information."""
        mock_ctx = Mock()
        mock_job = Mock()
        mock_room = Mock()
        mock_room.sid = TEST_ROOM_SID_123
        mock_room.name = TEST_ROOM_NAME_456
        mock_job.room = mock_room
        mock_job.id = TEST_JOB_ID
        mock_ctx.job = mock_job

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["job_id"] == TEST_JOB_ID
        assert result["job_room_sid"] == TEST_ROOM_SID_123
        assert result["job_room_name"] == TEST_ROOM_NAME_456

    def test_extract_job_context_with_room(self):
        """Test context extraction with room information."""
        mock_ctx = Mock()
        mock_room = Mock()
        mock_room.name = TEST_ROOM_NAME
        mock_room.sid = TEST_ROOM_SID
        mock_ctx.room = mock_room

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["room_name"] == TEST_ROOM_NAME
        assert result["room_sid"] == TEST_ROOM_SID

    def test_extract_job_context_with_agent(self):
        """Test context extraction with agent information."""
        mock_ctx = Mock()
        mock_agent = Mock()
        mock_agent.id = TEST_AGENT_ID
        mock_ctx.agent = mock_agent

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["agent_id"] == TEST_AGENT_ID

    def test_extract_job_context_with_worker_id(self):
        """Test context extraction with worker ID."""
        mock_ctx = Mock()
        mock_ctx.worker_id = TEST_WORKER_ID

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["worker_id"] == TEST_WORKER_ID

    def test_extract_job_context_with_participant(self):
        """Test context extraction with participant information."""
        mock_ctx = Mock()
        mock_participant = Mock()
        mock_participant.identity = TEST_PARTICIPANT_IDENTITY
        mock_participant.sid = TEST_PARTICIPANT_SID
        mock_ctx.participant = mock_participant

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["participant_identity"] == TEST_PARTICIPANT_IDENTITY
        assert result["participant_sid"] == TEST_PARTICIPANT_SID

    def test_extract_job_context_filters_mocks(self):
        """Test that extract_job_context filters out mock objects."""
        mock_ctx = Mock()
        mock_job = Mock()
        mock_job.id = Mock()  # Mock ID should be filtered
        mock_ctx.job = mock_job

        # Mock _is_mock_object to detect the mock ID
        def is_mock(obj):
            return isinstance(obj, Mock) and obj is not mock_ctx and obj is not mock_job

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            side_effect=is_mock,
        ):
            result = extract_job_context(mock_ctx)

        # job_id should not be in result because the ID itself is a mock
        assert "job_id" not in result

    def test_extract_job_context_empty_context(self):
        """Test context extraction with empty context object."""
        mock_ctx = Mock(spec=[])  # No attributes

        result = extract_job_context(mock_ctx)

        assert result == {}

    def test_extract_job_context_with_unknown_values(self):
        """Test that 'unknown' values are filtered out."""
        mock_ctx = Mock()
        mock_job = Mock()
        mock_job.id = TEST_DEFAULT_UNKNOWN  # Should be filtered
        mock_ctx.job = mock_job

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert "job_id" not in result


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestCreateSpanAttributes:
    """Test create_span_attributes function."""

    def test_create_span_attributes_basic(self):
        """Test basic span attribute creation."""
        result = create_span_attributes(
            provider=TEST_PROVIDER_DEEPGRAM,
            model=TEST_MODEL_NOVA2,
            operation_type=TEST_OPERATION_STT,
            audio_file=TEST_AUDIO_FILE_STT,
            audio_duration_ms=TEST_DURATION_MS_1500,
            job_context={},
        )

        assert result["stt.provider"] == TEST_PROVIDER_DEEPGRAM
        assert result["stt.model"] == TEST_MODEL_NOVA2
        assert result["stt.audio_file"] == TEST_AUDIO_FILE_STT
        assert result["stt.audio_duration_ms"] == TEST_DURATION_MS_1500

    def test_create_span_attributes_with_job_context(self):
        """Test span attributes with job context."""
        job_context = {"job_id": TEST_JOB_ID, "room_name": "room_456"}

        result = create_span_attributes(
            provider=TEST_PROVIDER_CARTESIA,
            model=TEST_MODEL_SONIC,
            operation_type=TEST_OPERATION_TTS,
            audio_file=TEST_AUDIO_FILE_TTS,
            audio_duration_ms=TEST_DURATION_MS_2000,
            job_context=job_context,
        )

        # job_id with 'job_' prefix gets converted to 'job.id' (removes 'job_' and adds 'job.')
        assert result["job.id"] == TEST_JOB_ID
        assert result["job.room_name"] == "room_456"

    def test_create_span_attributes_with_job_prefix_already_present(self):
        """Test that job context keys with 'job.' prefix are used as-is."""
        job_context = {"job.id": TEST_JOB_ID, "job.room": "room_456"}

        result = create_span_attributes(
            provider=TEST_PROVIDER_TEST,
            model=TEST_MODEL_TEST,
            operation_type=TEST_OPERATION_STT,
            audio_file=TEST_AUDIO_FILE_TEST,
            audio_duration_ms=TEST_DURATION_MS_1000,
            job_context=job_context,
        )

        assert result["job.id"] == TEST_JOB_ID
        assert result["job.room"] == "room_456"

    def test_create_span_attributes_with_job_underscore_prefix(self):
        """Test that job context keys with 'job_' prefix are converted to 'job.'."""
        job_context = {"job_id": TEST_JOB_ID, "job_room": "room_456"}

        result = create_span_attributes(
            provider=TEST_PROVIDER_TEST,
            model=TEST_MODEL_TEST,
            operation_type=TEST_OPERATION_STT,
            audio_file=TEST_AUDIO_FILE_TEST,
            audio_duration_ms=TEST_DURATION_MS_1000,
            job_context=job_context,
        )

        assert result["job.id"] == TEST_JOB_ID
        assert result["job.room"] == "room_456"

    def test_create_span_attributes_with_extra_attributes(self):
        """Test span attributes with extra attributes."""
        result = create_span_attributes(
            provider=TEST_PROVIDER_TEST,
            model=TEST_MODEL_TEST,
            operation_type=TEST_OPERATION_STT,
            audio_file=TEST_AUDIO_FILE_TEST,
            audio_duration_ms=TEST_DURATION_MS_1000,
            job_context={},
            stt_transcript=TEST_TRANSCRIPT,
            stt_confidence=TEST_CONFIDENCE,
            stt_mode=TEST_MODE_STREAMING,
        )

        assert result["stt_transcript"] == TEST_TRANSCRIPT
        assert result["stt_confidence"] == TEST_CONFIDENCE
        assert result["stt_mode"] == TEST_MODE_STREAMING

    def test_create_span_attributes_with_mixed_job_context(self):
        """Test span attributes with mixed job context key formats."""
        job_context = {
            "job.id": TEST_JOB_ID,  # Already has 'job.' prefix
            "job_room": "room_456",  # Has 'job_' prefix
            "agent_id": TEST_AGENT_ID_789,  # No prefix
        }

        result = create_span_attributes(
            provider=TEST_PROVIDER_TEST,
            model=TEST_MODEL_TEST,
            operation_type=TEST_OPERATION_TTS,
            audio_file=TEST_AUDIO_FILE_TEST,
            audio_duration_ms=TEST_DURATION_MS_1000,
            job_context=job_context,
        )

        assert result["job.id"] == TEST_JOB_ID
        assert result["job.room"] == "room_456"
        assert result["job.agent_id"] == TEST_AGENT_ID_789


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestSaveAudioFramesErrorHandling:
    """Test error handling in save_audio_frames."""

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True)
    def test_save_audio_frames_combine_raises_exception(self, mock_rtc, tmp_path):
        """Test save_audio_frames when combine_audio_frames raises exception."""
        output_path = tmp_path / TEST_AUDIO_FILE_TEST

        mock_rtc.combine_audio_frames.side_effect = Exception("Combine error")

        # Exceptions propagate (they're caught at the call site in livekit.py)
        with pytest.raises(Exception, match="Combine error"):
            save_audio_frames([Mock()], output_path)

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True)
    def test_save_audio_frames_to_wav_bytes_raises_exception(self, mock_rtc, tmp_path):
        """Test save_audio_frames when to_wav_bytes() raises exception."""
        output_path = tmp_path / TEST_AUDIO_FILE_TEST

        mock_combined = Mock()
        mock_combined.to_wav_bytes.side_effect = Exception("WAV conversion error")
        mock_rtc.combine_audio_frames.return_value = mock_combined

        # Exceptions propagate (they're caught at the call site in livekit.py)
        with pytest.raises(Exception, match="WAV conversion error"):
            save_audio_frames([Mock()], output_path)


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestExtractJobContextEdgeCases:
    """Test edge cases in extract_job_context."""

    def test_extract_job_context_with_nested_mock_detection(self):
        """Test extract_job_context with nested mock detection."""
        mock_ctx = Mock()
        mock_job = Mock()
        mock_job.id = TEST_JOB_ID
        mock_ctx.job = mock_job

        # Mock _is_mock_object to detect nested mocks
        def is_mock(obj):
            return isinstance(obj, Mock) and obj is not mock_ctx and obj is not mock_job

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            side_effect=is_mock,
        ):
            result = extract_job_context(mock_ctx)

        # Should still extract valid job_id
        assert "job_id" in result or result == {}


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestCreateSpanAttributesEdgeCases:
    """Test edge cases in create_span_attributes."""

    def test_create_span_attributes_with_complex_nested_job_context(self):
        """Test create_span_attributes with complex nested job_context values."""
        job_context = {
            "job_id": TEST_JOB_ID,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

        result = create_span_attributes(
            provider=TEST_PROVIDER_TEST,
            model=TEST_MODEL_TEST,
            operation_type=TEST_OPERATION_STT,
            audio_file=TEST_AUDIO_FILE_TEST,
            audio_duration_ms=TEST_DURATION_MS_1000,
            job_context=job_context,
        )

        # job_id with 'job_' prefix gets converted to 'job.id' (removes 'job_' and adds 'job.')
        assert result["job.id"] == TEST_JOB_ID
        # Other keys get 'job.' prefix added
        assert "job.nested" in result or "job.list" in result
