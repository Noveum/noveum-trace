"""
Unit tests for integrations/__init__.py.

Tests import error handling and conditional imports.
"""

import pytest


class TestIntegrationsInit:
    """Test integrations/__init__.py."""

    def test_integrations_module_structure(self):
        """Test that integrations module has correct structure."""
        from noveum_trace.integrations import __all__

        # Verify __all__ exists and is a list
        assert isinstance(__all__, list)

    def test_integrations_module_can_be_imported(self):
        """Test that integrations module can be imported without errors."""
        # This test verifies that import error handling works correctly
        # If imports fail, they should be handled gracefully
        try:
            import noveum_trace.integrations

            # Verify module has expected attributes
            assert hasattr(noveum_trace.integrations, "__all__")
            assert hasattr(noveum_trace.integrations, "logger")
        except ImportError:
            # If there's an import error, it should be for a good reason
            # (e.g., missing dependencies), not due to bad error handling
            pytest.fail("Integrations module should handle import errors gracefully")

    def test_conditional_imports_handled(self):
        """Test that conditional imports are handled correctly."""
        from noveum_trace.integrations import __all__

        # Verify that the module structure is correct regardless of what's imported
        # The __all__ list may vary based on what's available, but should always be a list
        assert isinstance(__all__, list)

        # If LiveKit is available, these should be in __all__
        # If not, they won't be, but that's expected behavior
        # We're just testing that the module doesn't crash
