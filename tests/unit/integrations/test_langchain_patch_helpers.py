"""
Unit tests for LangChain patch helper functions.

Tests the new helper functions added in the async/parallel execution patch:
- safe_inputs_to_dict()
- get_code_location()
"""

from unittest.mock import Mock, patch

import pytest

# Skip all tests if LangChain is not available
try:
    from noveum_trace.integrations.langchain import (
        get_code_location,
        safe_inputs_to_dict,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestSafeInputsToDict:
    """Test safe_inputs_to_dict helper function."""

    def test_safe_inputs_to_dict_with_dict_input(self):
        """Test conversion of dict input."""
        inputs = {"key1": "value1", "key2": 42, "key3": True}
        result = safe_inputs_to_dict(inputs)

        expected = {"key1": "value1", "key2": "42", "key3": "True"}
        assert result == expected

    def test_safe_inputs_to_dict_with_empty_dict(self):
        """Test conversion of empty dict input."""
        inputs = {}
        result = safe_inputs_to_dict(inputs)

        assert result == {}

    def test_safe_inputs_to_dict_with_list_input(self):
        """Test conversion of list input."""
        inputs = ["item1", 42, True, {"nested": "dict"}]
        result = safe_inputs_to_dict(inputs)

        expected = {
            "item_0": "item1",
            "item_1": "42",
            "item_2": "True",
            "item_3": "{'nested': 'dict'}",
        }
        assert result == expected

    def test_safe_inputs_to_dict_with_empty_list(self):
        """Test conversion of empty list input."""
        inputs = []
        result = safe_inputs_to_dict(inputs)

        assert result == {}

    def test_safe_inputs_to_dict_with_tuple_input(self):
        """Test conversion of tuple input."""
        inputs = ("item1", 42, True)
        result = safe_inputs_to_dict(inputs)

        expected = {"item_0": "item1", "item_1": "42", "item_2": "True"}
        assert result == expected

    def test_safe_inputs_to_dict_with_empty_tuple(self):
        """Test conversion of empty tuple input."""
        inputs = ()
        result = safe_inputs_to_dict(inputs)

        assert result == {}

    def test_safe_inputs_to_dict_with_primitive_input(self):
        """Test conversion of primitive input types."""
        # String input
        result = safe_inputs_to_dict("test_string")
        assert result == {"item": "test_string"}

        # Integer input
        result = safe_inputs_to_dict(42)
        assert result == {"item": "42"}

        # Boolean input
        result = safe_inputs_to_dict(True)
        assert result == {"item": "True"}

        # None input
        result = safe_inputs_to_dict(None)
        assert result == {"item": "None"}

    def test_safe_inputs_to_dict_with_custom_prefix(self):
        """Test conversion with custom prefix."""
        inputs = ["item1", "item2", "item3"]
        result = safe_inputs_to_dict(inputs, prefix="custom")

        expected = {"custom_0": "item1", "custom_1": "item2", "custom_2": "item3"}
        assert result == expected

    def test_safe_inputs_to_dict_with_complex_objects(self):
        """Test conversion of complex objects."""

        class CustomObject:
            def __str__(self):
                return "custom_object_str"

        inputs = [CustomObject(), {"nested": {"deep": "value"}}]
        result = safe_inputs_to_dict(inputs)

        expected = {
            "item_0": "custom_object_str",
            "item_1": "{'nested': {'deep': 'value'}}",
        }
        assert result == expected


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestGetCodeLocation:
    """Test get_code_location helper function."""

    def test_get_code_location_success(self):
        """Test successful code location extraction."""

        def test_function():
            return get_code_location(skip_frames=1)  # Skip this function frame

        result = test_function()

        assert "code.filepath" in result
        assert "code.function" in result
        assert "code.lineno" in result
        assert result["code.function"] == "test_function"
        assert isinstance(result["code.lineno"], int)
        assert result["code.filepath"].endswith(".py")

    def test_get_code_location_with_skip_frames(self):
        """Test code location with different skip_frames values."""

        def outer_function():
            def inner_function():
                return get_code_location(
                    skip_frames=3
                )  # Skip inner + outer + test method frame

            return inner_function()

        result = outer_function()

        # Should skip inner_function and outer_function, pointing to outer or test method
        assert result["code.function"] in [
            "outer_function",
            "test_get_code_location_with_skip_frames",
        ]

    def test_get_code_location_with_default_skip_frames(self):
        """Test code location with default skip_frames (2)."""
        result = get_code_location()  # Default skip_frames=2

        # Should skip get_code_location and point to this test method or pytest frame
        assert result["code.function"] in [
            "test_get_code_location_with_default_skip_frames",
            "pytest_pyfunc_call",
        ]

    def test_get_code_location_exception_handling(self):
        """Test code location when frame inspection fails."""
        with patch("inspect.currentframe", side_effect=Exception("Frame error")):
            result = get_code_location()

            # Should return empty dict on exception
            assert result == {}

    def test_get_code_location_no_frame(self):
        """Test code location when currentframe returns None."""
        with patch("inspect.currentframe", return_value=None):
            result = get_code_location()

            # Should return empty dict when no frame available
            assert result == {}

    def test_get_code_location_frame_traversal_limit(self):
        """Test code location when frame traversal reaches None."""
        # Create a mock frame that becomes None after one f_back access
        mock_frame = Mock()
        mock_frame.f_back = None

        with patch("inspect.currentframe", return_value=mock_frame):
            result = get_code_location(
                skip_frames=5
            )  # Try to skip more frames than available

            # Should return empty dict when frame becomes None during traversal
            assert result == {}

    def test_get_code_location_getframeinfo_exception(self):
        """Test code location when getframeinfo fails."""
        mock_frame = Mock()

        with (
            patch("inspect.currentframe", return_value=mock_frame),
            patch("inspect.getframeinfo", side_effect=Exception("FrameInfo error")),
        ):
            result = get_code_location(skip_frames=0)  # Don't skip any frames

            # Should return empty dict when getframeinfo fails
            assert result == {}

    def test_get_code_location_frame_attributes(self):
        """Test that all expected frame attributes are included."""
        result = get_code_location(skip_frames=1)

        if result:  # Only test if we got a result (frame inspection succeeded)
            assert "code.filepath" in result
            assert "code.function" in result
            assert "code.lineno" in result

            # Verify types
            assert isinstance(result["code.filepath"], str)
            assert isinstance(result["code.function"], str)
            assert isinstance(result["code.lineno"], int)

            # Verify reasonable values
            assert result["code.lineno"] > 0
            assert len(result["code.function"]) > 0
            assert result["code.filepath"].endswith(".py")
