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
        output_path = tmp_path / "empty.wav"

        save_audio_frames([], output_path)

        assert output_path.exists()
        assert output_path.read_bytes() == b""

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True)
    def test_save_audio_frames_valid_frames(self, mock_rtc, tmp_path):
        """Test saving valid audio frames."""
        output_path = tmp_path / "test.wav"

        # Mock audio frames
        mock_frame1 = Mock()
        mock_frame2 = Mock()

        # Mock combined frame
        mock_combined = Mock()
        mock_combined.to_wav_bytes.return_value = b"fake_wav_data"
        mock_rtc.combine_audio_frames.return_value = mock_combined

        save_audio_frames([mock_frame1, mock_frame2], output_path)

        assert output_path.exists()
        assert output_path.read_bytes() == b"fake_wav_data"
        mock_rtc.combine_audio_frames.assert_called_once_with(
            [mock_frame1, mock_frame2]
        )
        mock_combined.to_wav_bytes.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", False)
    def test_save_audio_frames_livekit_unavailable(self, tmp_path):
        """Test save_audio_frames when LiveKit is not available."""
        output_path = tmp_path / "test.wav"

        save_audio_frames([Mock()], output_path)

        # Should not create file when LiveKit unavailable
        assert not output_path.exists()

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True)
    def test_save_audio_frames_creates_directory(self, mock_rtc, tmp_path):
        """Test that save_audio_frames creates parent directory if needed."""
        output_path = tmp_path / "nested" / "dir" / "test.wav"

        mock_combined = Mock()
        mock_combined.to_wav_bytes.return_value = b"data"
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
        output_path = tmp_path / "test.wav"
        mock_buffer = [Mock(), Mock()]

        save_audio_buffer(mock_buffer, output_path)

        mock_save_frames.assert_called_once_with(mock_buffer, output_path)

    @patch("noveum_trace.integrations.livekit.livekit_utils.save_audio_frames")
    def test_save_audio_buffer_with_empty_buffer(self, mock_save_frames, tmp_path):
        """Test save_audio_buffer with empty buffer."""
        output_path = tmp_path / "test.wav"

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
        mock_frame.duration = 0.5  # 0.5 seconds

        result = calculate_audio_duration_ms([mock_frame])
        assert result == 500.0  # 500 milliseconds

    def test_calculate_audio_duration_ms_multiple_frames(self):
        """Test duration calculation with multiple frames."""
        mock_frame1 = Mock()
        mock_frame1.duration = 0.5
        mock_frame2 = Mock()
        mock_frame2.duration = 0.3
        mock_frame3 = Mock()
        mock_frame3.duration = 0.2

        result = calculate_audio_duration_ms([mock_frame1, mock_frame2, mock_frame3])
        assert result == 1000.0  # (0.5 + 0.3 + 0.2) * 1000 = 1000 ms


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestEnsureAudioDirectory:
    """Test ensure_audio_directory function."""

    def test_ensure_audio_directory_default_base(self, tmp_path):
        """Test directory creation with default base directory."""
        with patch(
            "noveum_trace.integrations.livekit.livekit_utils.Path",
            return_value=tmp_path / "audio_files",
        ):
            result = ensure_audio_directory("session_123")

            expected = tmp_path / "audio_files" / "session_123"
            assert result == expected
            assert result.exists()

    def test_ensure_audio_directory_custom_base(self, tmp_path):
        """Test directory creation with custom base directory."""
        custom_base = tmp_path / "custom_audio"
        result = ensure_audio_directory("session_456", base_dir=custom_base)

        expected = custom_base / "session_456"
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
        session_dir = tmp_path / "session_789"
        session_dir.mkdir(parents=True)

        result = ensure_audio_directory("session_789", base_dir=tmp_path)

        assert result == session_dir
        assert result.exists()


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestGenerateAudioFilename:
    """Test generate_audio_filename function."""

    def test_generate_audio_filename_with_timestamp(self):
        """Test filename generation with explicit timestamp."""
        result = generate_audio_filename("stt", 1, timestamp=1732386400000)

        assert result == "stt_0001_1732386400000.wav"

    def test_generate_audio_filename_without_timestamp(self):
        """Test filename generation without timestamp (uses current time)."""
        with patch("noveum_trace.integrations.livekit.livekit_utils.time") as mock_time:
            mock_time.time.return_value = 1732386400.0  # Returns seconds
            result = generate_audio_filename("tts", 42)

            # Should convert seconds to milliseconds
            assert result == "tts_0042_1732386400000.wav"

    def test_generate_audio_filename_counter_formatting(self):
        """Test that counter is zero-padded to 4 digits."""
        result = generate_audio_filename("stt", 5, timestamp=1000)
        assert result == "stt_0005_1000.wav"

        result = generate_audio_filename("stt", 123, timestamp=1000)
        assert result == "stt_0123_1000.wav"

        result = generate_audio_filename("stt", 9999, timestamp=1000)
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
        assert result == "unknown"

    def test_safe_str_with_custom_default(self):
        """Test _safe_str with None and custom default."""
        result = _safe_str(None, default="custom")
        assert result == "custom"

    def test_safe_str_with_mock_object(self):
        """Test _safe_str with mock object returns default."""
        mock_obj = Mock()
        result = _safe_str(mock_obj)
        assert result == "unknown"

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
        mock_job.id = "job_123"
        mock_ctx.job = mock_job

        # Make sure job is not detected as mock
        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["job_id"] == "job_123"

    def test_extract_job_context_with_job_room(self):
        """Test context extraction with job room information."""
        mock_ctx = Mock()
        mock_job = Mock()
        mock_room = Mock()
        mock_room.sid = "room_sid_123"
        mock_room.name = "room_name_456"
        mock_job.room = mock_room
        mock_job.id = "job_123"
        mock_ctx.job = mock_job

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["job_id"] == "job_123"
        assert result["job_room_sid"] == "room_sid_123"
        assert result["job_room_name"] == "room_name_456"

    def test_extract_job_context_with_room(self):
        """Test context extraction with room information."""
        mock_ctx = Mock()
        mock_room = Mock()
        mock_room.name = "room_name"
        mock_room.sid = "room_sid"
        mock_ctx.room = mock_room

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["room_name"] == "room_name"
        assert result["room_sid"] == "room_sid"

    def test_extract_job_context_with_agent(self):
        """Test context extraction with agent information."""
        mock_ctx = Mock()
        mock_agent = Mock()
        mock_agent.id = "agent_123"
        mock_ctx.agent = mock_agent

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["agent_id"] == "agent_123"

    def test_extract_job_context_with_worker_id(self):
        """Test context extraction with worker ID."""
        mock_ctx = Mock()
        mock_ctx.worker_id = "worker_123"

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["worker_id"] == "worker_123"

    def test_extract_job_context_with_participant(self):
        """Test context extraction with participant information."""
        mock_ctx = Mock()
        mock_participant = Mock()
        mock_participant.identity = "user_123"
        mock_participant.sid = "participant_sid_456"
        mock_ctx.participant = mock_participant

        with patch(
            "noveum_trace.integrations.livekit.livekit_utils._is_mock_object",
            return_value=False,
        ):
            result = extract_job_context(mock_ctx)

        assert result["participant_identity"] == "user_123"
        assert result["participant_sid"] == "participant_sid_456"

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
        mock_job.id = "unknown"  # Should be filtered
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
            provider="deepgram",
            model="nova-2",
            operation_type="stt",
            audio_file="stt_0001_123.wav",
            audio_duration_ms=1500.0,
            job_context={},
        )

        assert result["stt.provider"] == "deepgram"
        assert result["stt.model"] == "nova-2"
        assert result["stt.audio_file"] == "stt_0001_123.wav"
        assert result["stt.audio_duration_ms"] == 1500.0

    def test_create_span_attributes_with_job_context(self):
        """Test span attributes with job context."""
        job_context = {"job_id": "job_123", "room_name": "room_456"}

        result = create_span_attributes(
            provider="cartesia",
            model="sonic",
            operation_type="tts",
            audio_file="tts_0001_123.wav",
            audio_duration_ms=2000.0,
            job_context=job_context,
        )

        # job_id with 'job_' prefix gets converted to 'job.id' (removes 'job_' and adds 'job.')
        assert result["job.id"] == "job_123"
        assert result["job.room_name"] == "room_456"

    def test_create_span_attributes_with_job_prefix_already_present(self):
        """Test that job context keys with 'job.' prefix are used as-is."""
        job_context = {"job.id": "job_123", "job.room": "room_456"}

        result = create_span_attributes(
            provider="test",
            model="test",
            operation_type="stt",
            audio_file="test.wav",
            audio_duration_ms=1000.0,
            job_context=job_context,
        )

        assert result["job.id"] == "job_123"
        assert result["job.room"] == "room_456"

    def test_create_span_attributes_with_job_underscore_prefix(self):
        """Test that job context keys with 'job_' prefix are converted to 'job.'."""
        job_context = {"job_id": "job_123", "job_room": "room_456"}

        result = create_span_attributes(
            provider="test",
            model="test",
            operation_type="stt",
            audio_file="test.wav",
            audio_duration_ms=1000.0,
            job_context=job_context,
        )

        assert result["job.id"] == "job_123"
        assert result["job.room"] == "room_456"

    def test_create_span_attributes_with_extra_attributes(self):
        """Test span attributes with extra attributes."""
        result = create_span_attributes(
            provider="test",
            model="test",
            operation_type="stt",
            audio_file="test.wav",
            audio_duration_ms=1000.0,
            job_context={},
            stt_transcript="Hello world",
            stt_confidence=0.95,
            stt_mode="streaming",
        )

        assert result["stt_transcript"] == "Hello world"
        assert result["stt_confidence"] == 0.95
        assert result["stt_mode"] == "streaming"

    def test_create_span_attributes_with_mixed_job_context(self):
        """Test span attributes with mixed job context key formats."""
        job_context = {
            "job.id": "job_123",  # Already has 'job.' prefix
            "job_room": "room_456",  # Has 'job_' prefix
            "agent_id": "agent_789",  # No prefix
        }

        result = create_span_attributes(
            provider="test",
            model="test",
            operation_type="tts",
            audio_file="test.wav",
            audio_duration_ms=1000.0,
            job_context=job_context,
        )

        assert result["job.id"] == "job_123"
        assert result["job.room"] == "room_456"
        assert result["job.agent_id"] == "agent_789"


@pytest.mark.skipif(not LIVEKIT_UTILS_AVAILABLE, reason="LiveKit utils not available")
class TestSaveAudioFramesErrorHandling:
    """Test error handling in save_audio_frames."""

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True)
    def test_save_audio_frames_combine_raises_exception(self, mock_rtc, tmp_path):
        """Test save_audio_frames when combine_audio_frames raises exception."""
        output_path = tmp_path / "test.wav"

        mock_rtc.combine_audio_frames.side_effect = Exception("Combine error")

        # Exceptions propagate (they're caught at the call site in livekit.py)
        with pytest.raises(Exception, match="Combine error"):
            save_audio_frames([Mock()], output_path)

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True)
    def test_save_audio_frames_to_wav_bytes_raises_exception(self, mock_rtc, tmp_path):
        """Test save_audio_frames when to_wav_bytes() raises exception."""
        output_path = tmp_path / "test.wav"

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
        mock_job.id = "job_123"
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
            "job_id": "job_123",
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

        result = create_span_attributes(
            provider="test",
            model="test",
            operation_type="stt",
            audio_file="test.wav",
            audio_duration_ms=1000.0,
            job_context=job_context,
        )

        # job_id with 'job_' prefix gets converted to 'job.id' (removes 'job_' and adds 'job.')
        assert result["job.id"] == "job_123"
        # Other keys get 'job.' prefix added
        assert "job.nested" in result or "job.list" in result
