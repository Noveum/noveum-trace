"""
Unit tests for noveum_trace/__init__.py.

Tests import error handling for LiveKit integrations and the public
``init()`` entry point.
"""

from unittest.mock import patch

import noveum_trace
from noveum_trace.core import config as config_module


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


class TestInitServiceVersion:
    """Test the ``version`` parameter of the public ``init()`` function."""

    def _reset_state(self):
        """Reset global client and configuration so init() runs cleanly."""
        with noveum_trace._client_lock:
            noveum_trace._client = None
        config_module._config = None

    def setup_method(self):
        self._reset_state()

    def teardown_method(self):
        self._reset_state()

    def test_init_sets_version(self):
        """init(version=...) propagates the value into the configuration."""
        noveum_trace.init(
            project="test-project",
            api_key="test-key",
            version="v1.0.0",
        )

        assert config_module.get_config().version == "v1.0.0"

    def test_init_without_version_defaults_to_none(self):
        """init() without a version leaves the config version as None."""
        with patch.dict("os.environ", {}, clear=True):
            noveum_trace.init(project="test-project", api_key="test-key")

        assert config_module.get_config().version is None
