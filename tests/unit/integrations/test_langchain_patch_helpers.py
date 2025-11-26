"""
Unit tests for LangChain patch helper functions.

Tests the new helper functions added in the async/parallel execution patch:
- safe_inputs_to_dict()
"""

import pytest

# Skip all tests if LangChain is not available
try:
    # Import directly from the module to avoid issues with other integrations
    from noveum_trace.integrations.langchain.langchain import (
        safe_inputs_to_dict,
    )

    LANGCHAIN_AVAILABLE = True
except (ImportError, NameError, AttributeError):
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

        expected = {"custom_0": "item1",
                    "custom_1": "item2", "custom_2": "item3"}
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
