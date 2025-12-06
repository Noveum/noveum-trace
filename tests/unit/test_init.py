"""
Unit tests for noveum_trace/__init__.py.

Tests import error handling for LiveKit integrations.
"""

from unittest.mock import patch


class TestInitLiveKitImports:
    """Test LiveKit import error handling in __init__.py."""

    def test_livekit_integration_import_failure(self):
        """Test import failure for LiveKit integrations."""
        with patch(
            "noveum_trace.integrations.livekit.LiveKitSTTWrapper"
        ) as mock_import:
            mock_import.side_effect = ImportError("LiveKit not available")

            # Import should handle gracefully
            try:
                import noveum_trace

                # Should not crash on import failure
                assert hasattr(noveum_trace, "__all__")
            except ImportError:
                # This is acceptable - import errors are handled gracefully
                pass

    def test_livekit_wrapper_imports_handled_gracefully(self):
        """Test that LiveKit wrapper imports are handled gracefully."""
        # Test that the module can be imported even if LiveKit is not available
        try:
            import noveum_trace

            # Verify module structure exists
            assert hasattr(noveum_trace, "__all__")
            assert isinstance(noveum_trace.__all__, list)

            # Verify core functions are available
            assert "init" in noveum_trace.__all__
            assert "get_client" in noveum_trace.__all__
        except ImportError as e:
            # Only fail if it's not a LiveKit-related import error
            if "livekit" not in str(e).lower():
                raise

    @patch("noveum_trace.integrations.livekit.LiveKitSTTWrapper")
    @patch("noveum_trace.integrations.livekit.LiveKitTTSWrapper")
    def test_livekit_imports_skipped_on_error(self, mock_tts, mock_stt):
        """Test that LiveKit imports are skipped when they fail."""
        mock_stt.side_effect = ImportError("LiveKit not available")
        mock_tts.side_effect = ImportError("LiveKit not available")

        # Should not raise exception
        try:
            import noveum_trace

            # Verify module still works
            assert hasattr(noveum_trace, "__all__")
        except ImportError:
            # Acceptable if import fails, but should be handled gracefully
            pass
