"""
Unit tests for LiveKit utility functions.

Tests the utility functions in livekit_utils.py, focusing on audio upload functionality.
"""

from unittest.mock import Mock, patch

import pytest


class TestUploadAudioFrames:
    """Test upload_audio_frames utility function."""

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.calculate_audio_duration_ms")
    def test_successful_upload(
        self, mock_calc_duration, mock_get_client
    ):
        """Test successful audio upload with all parameters."""
        from noveum_trace.integrations.livekit.livekit_utils import upload_audio_frames

        # Create mock frames
        mock_frame1 = Mock()
        mock_frame1.duration = 0.5
        mock_frame2 = Mock()
        mock_frame2.duration = 0.5
        frames = [mock_frame1, mock_frame2]

        # Mock rtc.combine_audio_frames
        mock_combined = Mock()
        mock_combined.to_wav_bytes.return_value = b"fake_wav_data"
        
        with patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True) as mock_rtc:
            mock_rtc.combine_audio_frames.return_value = mock_combined

            # Mock client
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Mock duration calculation
            mock_calc_duration.return_value = 1000.0

            # Test parameters
            audio_uuid = "test-uuid-123"
            audio_type = "stt"
            trace_id = "trace-abc"
            span_id = "span-xyz"

            # Execute
            result = upload_audio_frames(
                frames=frames,
                audio_uuid=audio_uuid,
                audio_type=audio_type,
                trace_id=trace_id,
                span_id=span_id,
            )

            # Verify
            assert result is True
            mock_rtc.combine_audio_frames.assert_called_once_with(frames)
            mock_combined.to_wav_bytes.assert_called_once()
            mock_get_client.assert_called_once()
            mock_calc_duration.assert_called_once_with(frames)

            # Verify client.export_audio was called with correct parameters
            mock_client.export_audio.assert_called_once()
            call_args = mock_client.export_audio.call_args
            assert call_args.kwargs["audio_data"] == b"fake_wav_data"
            assert call_args.kwargs["trace_id"] == trace_id
            assert call_args.kwargs["span_id"] == span_id
            assert call_args.kwargs["audio_uuid"] == audio_uuid
            assert call_args.kwargs["metadata"]["duration_ms"] == 1000.0
            assert call_args.kwargs["metadata"]["format"] == "wav"
            assert call_args.kwargs["metadata"]["type"] == audio_type

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", False)
    def test_livekit_not_available(self):
        """Test upload when LiveKit is not available."""
        from noveum_trace.integrations.livekit.livekit_utils import upload_audio_frames

        frames = [Mock()]
        result = upload_audio_frames(
            frames=frames,
            audio_uuid="test-uuid",
            audio_type="stt",
            trace_id="trace-id",
            span_id="span-id",
        )

        assert result is False

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    def test_empty_frames(self):
        """Test upload with empty frames list."""
        from noveum_trace.integrations.livekit.livekit_utils import upload_audio_frames

        result = upload_audio_frames(
            frames=[],
            audio_uuid="test-uuid",
            audio_type="stt",
            trace_id="trace-id",
            span_id="span-id",
        )

        assert result is False

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.get_client")
    def test_no_client_available(self, mock_get_client):
        """Test upload when no client is available."""
        from noveum_trace.integrations.livekit.livekit_utils import upload_audio_frames

        # Mock frames and rtc
        frames = [Mock()]
        mock_combined = Mock()
        mock_combined.to_wav_bytes.return_value = b"fake_wav_data"
        
        with patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True) as mock_rtc:
            mock_rtc.combine_audio_frames.return_value = mock_combined

            # No client available
            mock_get_client.return_value = None

            result = upload_audio_frames(
                frames=frames,
                audio_uuid="test-uuid",
                audio_type="stt",
                trace_id="trace-id",
                span_id="span-id",
            )

            assert result is False

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.calculate_audio_duration_ms")
    def test_exception_handling(
        self, mock_calc_duration, mock_get_client
    ):
        """Test exception handling during upload."""
        from noveum_trace.integrations.livekit.livekit_utils import upload_audio_frames

        # Create mock frames
        frames = [Mock()]
        mock_combined = Mock()
        mock_combined.to_wav_bytes.return_value = b"fake_wav_data"
        
        with patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True) as mock_rtc:
            mock_rtc.combine_audio_frames.return_value = mock_combined

            # Mock client that raises exception
            mock_client = Mock()
            mock_client.export_audio.side_effect = Exception("Upload failed")
            mock_get_client.return_value = mock_client

            mock_calc_duration.return_value = 1000.0

            result = upload_audio_frames(
                frames=frames,
                audio_uuid="test-uuid",
                audio_type="stt",
                trace_id="trace-id",
                span_id="span-id",
            )

            assert result is False

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.calculate_audio_duration_ms")
    def test_metadata_formatting(
        self, mock_calc_duration, mock_get_client
    ):
        """Test metadata is correctly formatted for different audio types."""
        from noveum_trace.integrations.livekit.livekit_utils import upload_audio_frames

        # Create mock frames
        frames = [Mock()]
        mock_combined = Mock()
        mock_combined.to_wav_bytes.return_value = b"fake_wav_data"
        
        with patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True) as mock_rtc:
            mock_rtc.combine_audio_frames.return_value = mock_combined

            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Test with TTS audio type
            mock_calc_duration.return_value = 2500.0

            result = upload_audio_frames(
                frames=frames,
                audio_uuid="tts-uuid",
                audio_type="tts",
                trace_id="trace-id",
                span_id="span-id",
            )

            assert result is True

            # Verify metadata structure
            call_args = mock_client.export_audio.call_args
            metadata = call_args.kwargs["metadata"]
            assert metadata["duration_ms"] == 2500.0
            assert metadata["format"] == "wav"
            assert metadata["type"] == "tts"

    @patch("noveum_trace.integrations.livekit.livekit_utils.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.calculate_audio_duration_ms")
    def test_trace_and_span_ids_passed_correctly(
        self, mock_calc_duration, mock_get_client
    ):
        """Test that trace_id and span_id are passed correctly to export_audio."""
        from noveum_trace.integrations.livekit.livekit_utils import upload_audio_frames

        # Create mock frames
        frames = [Mock()]
        mock_combined = Mock()
        mock_combined.to_wav_bytes.return_value = b"fake_wav_data"
        
        with patch("noveum_trace.integrations.livekit.livekit_utils.rtc", create=True) as mock_rtc:
            mock_rtc.combine_audio_frames.return_value = mock_combined

            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_calc_duration.return_value = 1000.0

            # Test with specific IDs
            trace_id = "trace-123-abc"
            span_id = "span-456-def"

            result = upload_audio_frames(
                frames=frames,
                audio_uuid="audio-uuid",
                audio_type="stt",
                trace_id=trace_id,
                span_id=span_id,
            )

            assert result is True

            # Verify IDs are passed correctly
            call_args = mock_client.export_audio.call_args
            assert call_args.kwargs["trace_id"] == trace_id
            assert call_args.kwargs["span_id"] == span_id
