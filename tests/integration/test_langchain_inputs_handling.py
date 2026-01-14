"""
Integration tests for LangChain callback handler input type handling.

These tests verify that the callback handlers correctly handle different input types
for on_chain_start, on_tool_start, and on_agent_start.
"""

from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

# Skip all tests if LangChain is not available
try:
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestOnChainStartInputHandling:
    """Test on_chain_start with different input types."""

    def test_chain_start_with_dict_input(self):
        """Test on_chain_start with standard dict input (original behavior)."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Standard dict input
            handler.on_chain_start(
                serialized={"name": "test_chain"},
                inputs={"key1": "value1", "key2": "value2"},
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should use flattened "chain.inputs.{key}" format
            assert "chain.inputs.key1" in attributes
            assert "chain.inputs.key2" in attributes
            assert attributes["chain.inputs.key1"] == "value1"
            assert attributes["chain.inputs.key2"] == "value2"

    def test_chain_start_with_list_of_dicts_input(self):
        """Test on_chain_start with list of dicts (LangGraph prebuilt pattern)."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # List of dicts input (LangGraph pattern)
            handler.on_chain_start(
                serialized={"name": "test_chain"},
                inputs=[
                    {
                        "name": "WebSearch",
                        "args": {"query": "test"},
                        "id": "call_123",
                        "type": "tool_call",
                    },
                    {
                        "name": "Calculator",
                        "args": {"expression": "2+2"},
                        "id": "call_456",
                        "type": "tool_call",
                    },
                ],
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should use flattened "chain.inputs.0.{key}" format for dict items
            assert "chain.inputs.0.name" in attributes
            assert "chain.inputs.0.args" in attributes
            assert "chain.inputs.1.name" in attributes
            assert "chain.inputs.1.args" in attributes
            assert attributes["chain.inputs.0.name"] == "WebSearch"
            assert attributes["chain.inputs.1.name"] == "Calculator"

    def test_chain_start_with_list_of_mixed_types(self):
        """Test on_chain_start with list containing non-dict elements."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # List with non-dict elements
            handler.on_chain_start(
                serialized={"name": "test_chain"},
                inputs=[{"name": "tool1"}, "string_element", 123],
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should use flattened format for dict items and direct format for non-dict items
            assert "chain.inputs.0.name" in attributes  # First item is dict
            assert "chain.inputs.1" in attributes  # Second item is string
            assert "chain.inputs.2" in attributes  # Third item is int
            assert attributes["chain.inputs.0.name"] == "tool1"
            assert attributes["chain.inputs.1"] == "string_element"
            assert attributes["chain.inputs.2"] == "123"

    def test_chain_start_with_empty_list(self):
        """Test on_chain_start with empty list."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Empty list
            handler.on_chain_start(
                serialized={"name": "test_chain"}, inputs=[], run_id=run_id
            )

            # Verify span was created without errors
            mock_client.start_span.assert_called_once()

    def test_chain_start_with_string_input(self):
        """Test on_chain_start with string input (fallback case)."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # String input (fallback case)
            handler.on_chain_start(
                serialized={"name": "test_chain"},
                inputs="string_input",
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should convert to string
            assert "chain.inputs" in attributes
            assert attributes["chain.inputs"] == "string_input"


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestOnToolStartInputHandling:
    """Test on_tool_start with different input types."""

    def test_tool_start_with_dict_input(self):
        """Test on_tool_start with dict input (Case 1: standard behavior)."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Dict input
            handler.on_tool_start(
                serialized={"name": "TestTool"},
                input_str="test input",
                inputs={"query": "test", "limit": 10},
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should create tool.input.{key} attributes
            assert attributes["tool.input.query"] == "test"
            assert attributes["tool.input.limit"] == "10"
            assert attributes["tool.input.argument_count"] == "2"

    def test_tool_start_with_list_of_dicts(self):
        """Test on_tool_start with list of dicts (Case 2)."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # List of dicts
            handler.on_tool_start(
                serialized={"name": "TestTool"},
                input_str="test input",
                inputs=[
                    {"name": "tool1", "arg": "val1"},
                    {"name": "tool2", "arg": "val2"},
                ],
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should create flattened tool.input.{index}.{key} attributes for list of dicts
            assert "tool.input.0.name" in attributes
            assert "tool.input.0.arg" in attributes
            assert "tool.input.1.name" in attributes
            assert "tool.input.1.arg" in attributes
            assert attributes["tool.input.0.name"] == "tool1"
            assert attributes["tool.input.0.arg"] == "val1"
            assert attributes["tool.input.1.name"] == "tool2"
            assert attributes["tool.input.1.arg"] == "val2"
            assert attributes["tool.input.argument_count"] == "2"

    def test_tool_start_with_list_of_mixed_types(self):
        """Test on_tool_start with list of mixed types (Case 3)."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # List of mixed types
            handler.on_tool_start(
                serialized={"name": "TestTool"},
                input_str="test input",
                inputs=["item1", "item2", 123],
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should create tool.input.{index} attributes with string values
            assert attributes["tool.input.0"] == "item1"
            assert attributes["tool.input.1"] == "item2"
            assert attributes["tool.input.2"] == "123"
            assert attributes["tool.input.argument_count"] == "3"

    def test_tool_start_with_tuple(self):
        """Test on_tool_start with tuple input (Case 4)."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Tuple input
            handler.on_tool_start(
                serialized={"name": "TestTool"},
                input_str="test input",
                inputs=("arg1", "arg2", "arg3"),
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should create tool.input.{index} attributes
            assert attributes["tool.input.0"] == "arg1"
            assert attributes["tool.input.1"] == "arg2"
            assert attributes["tool.input.2"] == "arg3"
            assert attributes["tool.input.argument_count"] == "3"

    def test_tool_start_with_string_input(self):
        """Test on_tool_start with string input (Case 5)."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # String input
            handler.on_tool_start(
                serialized={"name": "TestTool"},
                input_str="test input",
                inputs="single_string_value",
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should create tool.input.arg attribute
            assert attributes["tool.input.arg"] == "single_string_value"
            assert attributes["tool.input.argument_count"] == "1"

    def test_tool_start_with_int_input(self):
        """Test on_tool_start with integer input (Case 5)."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Integer input
            handler.on_tool_start(
                serialized={"name": "TestTool"},
                input_str="test input",
                inputs=42,
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should create tool.input.arg attribute with string conversion
            assert attributes["tool.input.arg"] == "42"
            assert attributes["tool.input.argument_count"] == "1"

    def test_tool_start_with_none_input(self):
        """Test on_tool_start with None input."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # None input
            handler.on_tool_start(
                serialized={"name": "TestTool"},
                input_str="test input",
                inputs=None,
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should set argument_count to "0"
            assert attributes["tool.input.argument_count"] == "0"

    def test_tool_start_with_empty_dict(self):
        """Test on_tool_start with empty dict."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Empty dict
            handler.on_tool_start(
                serialized={"name": "TestTool"},
                input_str="test input",
                inputs={},
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should set argument_count to "0"
            assert attributes["tool.input.argument_count"] == "0"

    def test_tool_start_with_empty_list(self):
        """Test on_tool_start with empty list."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Empty list
            handler.on_tool_start(
                serialized={"name": "TestTool"},
                input_str="test input",
                inputs=[],
                run_id=run_id,
            )

            # Verify span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]

            # Should set argument_count to "0" (empty list)
            assert attributes["tool.input.argument_count"] == "0"


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestInputHandlingEdgeCases:
    """Test edge cases and error handling for input type processing."""

    def test_chain_start_no_crash_on_exception(self):
        """Test that on_chain_start doesn't crash on unexpected errors."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Make start_span raise an exception
            mock_client.start_span.side_effect = Exception("Test exception")

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Should not crash
            handler.on_chain_start(
                serialized={"name": "test_chain"},
                inputs={"key": "value"},
                run_id=run_id,
            )

    def test_tool_start_no_crash_on_exception(self):
        """Test that on_tool_start doesn't crash on unexpected errors."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Make start_span raise an exception
            mock_client.start_span.side_effect = Exception("Test exception")

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Should not crash
            handler.on_tool_start(
                serialized={"name": "TestTool"},
                input_str="test",
                inputs={"key": "value"},
                run_id=run_id,
            )

    def test_chain_start_with_nested_structures(self):
        """Test on_chain_start with complex nested data structures."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Nested structure
            handler.on_chain_start(
                serialized={"name": "test_chain"},
                inputs={"nested": {"deep": {"value": "test"}}, "list": [1, 2, 3]},
                run_id=run_id,
            )

            # Verify span was created without errors
            mock_client.start_span.assert_called_once()
