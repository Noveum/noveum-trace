"""
Comprehensive tests for Noveum Trace decorators.

This module contains extensive unit tests for all decorator functionality,
including @trace, @trace_llm, @trace_agent, @trace_tool, and @trace_retrieval.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from noveum_trace.core.span import SpanStatus
from noveum_trace.decorators import (
    trace,
    trace_agent,
    trace_llm,
    trace_retrieval,
    trace_tool,
)
from noveum_trace.decorators.base import _serialize_value, get_trace_config, is_traced


class TestTraceDecorator:
    """Test the base @trace decorator."""

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_decorator_basic(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test basic @trace decorator functionality."""
        # Setup mocks
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        # Create a traced function
        @trace
        def test_function(x: int) -> int:
            return x * 2

        # Call the function
        result = test_function(5)

        # Assertions
        assert result == 10
        assert is_traced(test_function)
        mock_client.start_span.assert_called_once()
        mock_span.set_status.assert_called_with(SpanStatus.OK)
        mock_client.finish_span.assert_called_once_with(mock_span)

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_decorator_not_initialized(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test @trace decorator when SDK is not initialized."""
        mock_initialized.return_value = False

        @trace
        def test_function(x: int) -> int:
            return x * 2

        result = test_function(5)

        assert result == 10
        # Client should not be called when not initialized
        mock_get_client.assert_not_called()

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    @pytest.mark.parametrize(
        "capture_args,capture_result,capture_errors,capture_performance,capture_stack_trace",
        [
            (True, True, True, True, True),
            (False, True, True, True, True),
            (True, False, True, True, True),
            (True, True, False, True, True),
            (True, True, True, False, False),
        ],
    )
    def test_trace_decorator_configurations(
        self,
        mock_initialized,
        mock_get_client,
        mock_get_current_trace,
        capture_args,
        capture_result,
        capture_errors,
        capture_performance,
        capture_stack_trace,
    ):
        """Test @trace decorator with different configurations."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace(
            capture_args=capture_args,
            capture_result=capture_result,
            capture_errors=capture_errors,
            capture_performance=capture_performance,
            capture_stack_trace=capture_stack_trace,
        )
        def test_function(x: int) -> int:
            return x * 2

        result = test_function(5)

        assert result == 10
        config = get_trace_config(test_function)
        assert config is not None
        assert config["capture_args"] == capture_args
        assert config["capture_result"] == capture_result
        assert config["capture_errors"] == capture_errors
        assert config["capture_performance"] == capture_performance

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_decorator_with_exception(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test @trace decorator with exception handling."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace(capture_errors=True)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        # Verify error handling
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_with(SpanStatus.ERROR, "Test error")
        mock_client.finish_span.assert_called_once_with(mock_span)

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_decorator_auto_trace_creation(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test @trace decorator auto-creating traces when none exists."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_current_trace.return_value = None  # No current trace
        mock_new_trace = MagicMock()
        mock_client.start_trace.return_value = mock_new_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace
        def test_function():
            return "result"

        result = test_function()

        assert result == "result"
        # Should auto-create trace
        mock_client.start_trace.assert_called_once()
        mock_client.finish_trace.assert_called_once_with(mock_new_trace)

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    @pytest.mark.parametrize(
        "span_name,metadata,tags",
        [
            ("custom_span", {"key1": "value1"}, {"tag1": "value1"}),
            (None, None, None),
            (
                "complex_span",
                {"key1": "value1", "key2": 123},
                {"env": "test", "version": "1.0"},
            ),
        ],
    )
    def test_trace_decorator_with_metadata(
        self,
        mock_initialized,
        mock_get_client,
        mock_get_current_trace,
        span_name,
        metadata,
        tags,
    ):
        """Test @trace decorator with custom metadata and tags."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace(name=span_name, metadata=metadata, tags=tags)
        def test_function():
            return "result"

        result = test_function()

        assert result == "result"
        # Verify span creation with proper attributes
        mock_client.start_span.assert_called_once()
        call_args = mock_client.start_span.call_args
        attributes = call_args[1]["attributes"]

        # Check metadata
        if metadata:
            for key, _value in metadata.items():
                assert f"metadata.{key}" in attributes
        # Check tags
        if tags:
            for key, _value in tags.items():
                assert f"tag.{key}" in attributes


class TestTraceLLMDecorator:
    """Test the @trace_llm decorator."""

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_llm_basic(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test basic @trace_llm decorator functionality."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_llm
        def llm_call(prompt: str) -> str:
            return f"Response to: {prompt}"

        result = llm_call("Hello")

        assert result == "Response to: Hello"
        mock_client.start_span.assert_called_once()
        call_args = mock_client.start_span.call_args
        attributes = call_args[1]["attributes"]
        assert attributes["function.type"] == "llm_call"

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    @pytest.mark.parametrize(
        "provider,capture_prompts,capture_completions,capture_tokens,estimate_costs",
        [
            ("openai", True, True, True, True),
            ("anthropic", False, True, False, False),
            (None, True, False, True, False),
        ],
    )
    def test_trace_llm_configurations(
        self,
        mock_initialized,
        mock_get_client,
        mock_get_current_trace,
        provider,
        capture_prompts,
        capture_completions,
        capture_tokens,
        estimate_costs,
    ):
        """Test @trace_llm decorator with different configurations."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_llm(
            provider=provider,
            capture_prompts=capture_prompts,
            capture_completions=capture_completions,
            capture_tokens=capture_tokens,
            estimate_costs=estimate_costs,
        )
        def llm_call(prompt: str) -> str:
            return f"Response to: {prompt}"

        result = llm_call("Hello")

        assert result == "Response to: Hello"
        mock_client.start_span.assert_called_once()

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_llm_with_exception(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test @trace_llm decorator with exception handling."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_llm
        def failing_llm_call():
            raise RuntimeError("LLM API error")

        with pytest.raises(RuntimeError, match="LLM API error"):
            failing_llm_call()

        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_with(SpanStatus.ERROR, "LLM API error")

    @patch("noveum_trace.utils.llm_utils.estimate_token_count")
    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_llm_token_estimation(
        self,
        mock_initialized,
        mock_get_client,
        mock_get_current_trace,
        mock_estimate_tokens,
    ):
        """Test @trace_llm decorator with token estimation."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span
        mock_estimate_tokens.return_value = 10

        @trace_llm(capture_tokens=True)
        def llm_call_with_tokens(prompt: str) -> str:
            return "Response"

        result = llm_call_with_tokens("Test prompt")

        assert result == "Response"
        mock_client.start_span.assert_called_once()


class TestTraceAgentDecorator:
    """Test the @trace_agent decorator."""

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_agent_basic(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test basic @trace_agent decorator functionality."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_agent("test_agent")
        def agent_function(task: str) -> dict:
            return {"result": f"Completed: {task}"}

        result = agent_function("test_task")

        assert result == {"result": "Completed: test_task"}
        mock_client.start_span.assert_called_once()
        call_args = mock_client.start_span.call_args
        attributes = call_args[1]["attributes"]
        assert attributes["agent.id"] == "test_agent"
        assert attributes["function.type"] == "agent_operation"

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    @pytest.mark.parametrize(
        "agent_id,role,agent_type,capabilities",
        [
            ("coordinator", "manager", "orchestrator", ["planning", "coordination"]),
            ("researcher", "analyst", "llm_agent", ["web_search", "analysis"]),
            ("simple_agent", None, None, None),
        ],
    )
    def test_trace_agent_configurations(
        self,
        mock_initialized,
        mock_get_client,
        mock_get_current_trace,
        agent_id,
        role,
        agent_type,
        capabilities,
    ):
        """Test @trace_agent decorator with different configurations."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_agent(
            agent_id, role=role, agent_type=agent_type, capabilities=capabilities
        )
        def agent_function():
            return {"status": "complete"}

        result = agent_function()

        assert result == {"status": "complete"}
        mock_client.start_span.assert_called_once()
        call_args = mock_client.start_span.call_args
        attributes = call_args[1]["attributes"]
        assert attributes["agent.id"] == agent_id
        if role:
            assert attributes["agent.role"] == role
        if agent_type:
            assert attributes["agent.type"] == agent_type

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_agent_with_exception(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test @trace_agent decorator with exception handling."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_agent("failing_agent")
        def failing_agent():
            raise ValueError("Agent failed")

        with pytest.raises(ValueError, match="Agent failed"):
            failing_agent()

        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_with(SpanStatus.ERROR, "Agent failed")

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_agent_nested_agents(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test @trace_agent decorator with nested agent calls."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_agent("parent_agent")
        def parent_agent():
            return child_agent()

        @trace_agent("child_agent")
        def child_agent():
            return {"result": "child_complete"}

        result = parent_agent()

        assert result == {"result": "child_complete"}
        # Both agents should create spans
        assert mock_client.start_span.call_count == 2


class TestTraceToolDecorator:
    """Test the @trace_tool decorator."""

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_tool_basic(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test basic @trace_tool decorator functionality."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_tool
        def search_tool(query: str) -> list:
            return [f"Result for: {query}"]

        result = search_tool("test query")

        assert result == ["Result for: test query"]
        mock_client.start_span.assert_called_once()
        call_args = mock_client.start_span.call_args
        attributes = call_args[1]["attributes"]
        assert attributes["function.type"] == "tool_call"

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    @pytest.mark.parametrize(
        "tool_name,tool_type,description",
        [
            ("custom_search", "api", "Search the web using custom API"),
            ("database_query", "database", "Execute SQL queries"),
            (None, "file_system", "File operations"),
        ],
    )
    def test_trace_tool_configurations(
        self,
        mock_initialized,
        mock_get_client,
        mock_get_current_trace,
        tool_name,
        tool_type,
        description,
    ):
        """Test @trace_tool decorator with different configurations."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_tool(tool_name=tool_name, tool_type=tool_type, description=description)
        def tool_function():
            return {"status": "success"}

        result = tool_function()

        assert result == {"status": "success"}
        mock_client.start_span.assert_called_once()
        call_args = mock_client.start_span.call_args
        attributes = call_args[1]["attributes"]
        expected_tool_name = tool_name or "tool_function"
        assert attributes["tool.name"] == expected_tool_name
        if tool_type:
            assert attributes["tool.type"] == tool_type
        if description:
            assert attributes["tool.description"] == description

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_tool_with_exception(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test @trace_tool decorator with exception handling."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_tool
        def failing_tool():
            raise ConnectionError("Tool connection failed")

        with pytest.raises(ConnectionError, match="Tool connection failed"):
            failing_tool()

        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_with(
            SpanStatus.ERROR, "Tool connection failed"
        )

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_tool_input_output_capture(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test @trace_tool decorator capturing inputs and outputs."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_tool(capture_inputs=True, capture_outputs=True)
        def process_tool(data: str, options: dict = None) -> dict:
            return {"processed": data, "options": options}

        result = process_tool("test_data", {"format": "json"})

        assert result == {"processed": "test_data", "options": {"format": "json"}}
        mock_client.start_span.assert_called_once()


class TestTraceRetrievalDecorator:
    """Test the @trace_retrieval decorator."""

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_retrieval_basic(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test basic @trace_retrieval decorator functionality."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_retrieval
        def search_documents(query: str) -> list:
            return [{"text": f"Document about {query}", "score": 0.9}]

        result = search_documents("test query")

        assert len(result) == 1
        assert result[0]["text"] == "Document about test query"
        mock_client.start_span.assert_called_once()
        call_args = mock_client.start_span.call_args
        attributes = call_args[1]["attributes"]
        assert attributes["function.type"] == "retrieval_operation"

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    @pytest.mark.parametrize(
        "retrieval_type,index_name,capture_scores",
        [
            ("vector_search", "documents", True),
            ("keyword_search", "articles", False),
            ("hybrid_search", None, True),
        ],
    )
    def test_trace_retrieval_configurations(
        self,
        mock_initialized,
        mock_get_client,
        mock_get_current_trace,
        retrieval_type,
        index_name,
        capture_scores,
    ):
        """Test @trace_retrieval decorator with different configurations."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_retrieval(
            retrieval_type=retrieval_type,
            index_name=index_name,
            capture_scores=capture_scores,
        )
        def retrieval_function():
            return [{"text": "document", "score": 0.8}]

        result = retrieval_function()

        assert len(result) == 1
        mock_client.start_span.assert_called_once()
        call_args = mock_client.start_span.call_args
        attributes = call_args[1]["attributes"]
        assert attributes["retrieval.type"] == retrieval_type
        if index_name:
            assert attributes["retrieval.index_name"] == index_name

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_trace_retrieval_with_exception(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test @trace_retrieval decorator with exception handling."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_retrieval
        def failing_retrieval():
            raise TimeoutError("Search timeout")

        with pytest.raises(TimeoutError, match="Search timeout"):
            failing_retrieval()

        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_with(SpanStatus.ERROR, "Search timeout")


class TestDecoratorUtilities:
    """Test decorator utility functions."""

    @pytest.mark.parametrize(
        "value,expected_type",
        [
            ("string", str),
            (123, int),
            (3.14, float),
            (True, bool),
            ([1, 2, 3], list),
            ({"key": "value"}, dict),
            (None, type(None)),
        ],
    )
    def test_serialize_value_basic_types(self, value, expected_type):
        """Test _serialize_value with basic types."""
        import json

        result = _serialize_value(value)
        if value is None:
            assert result is None  # _serialize_value returns None for None
        else:
            # Should return the same type for primitives, or appropriate type for collections
            assert isinstance(result, (str, int, float, bool, list, dict))
            if isinstance(value, (str, int, float, bool)):
                assert result == value
            elif isinstance(value, (list, tuple)):
                assert isinstance(result, list)
            elif isinstance(value, dict):
                assert isinstance(result, dict)

    def test_serialize_value_large_collections(self):
        """Test _serialize_value with large collections preserves structure."""
        large_list = list(range(20))
        result = _serialize_value(large_list)
        assert isinstance(result, list)  # Should be list, not string
        assert result == large_list

        large_dict = {f"key_{i}": i for i in range(20)}
        result = _serialize_value(large_dict)
        assert isinstance(result, dict)  # Should be dict, not string
        assert result == large_dict

    def test_serialize_value_no_truncation(self):
        """Test _serialize_value does not truncate or summarize content."""
        import json

        # Test long strings
        long_string = "x" * 2000
        result = _serialize_value(long_string)
        assert len(result) == 2000
        assert result == long_string
        assert isinstance(result, str)

        # Test large lists - should be valid JSON
        large_list = [f"item_{i}" for i in range(100)]
        result = _serialize_value(large_list)
        assert isinstance(result, list)
        assert len(result) == 100

        # Test large dictionaries - should be valid JSON
        large_dict = {f"key_{i}": f"value_{i}" for i in range(50)}
        result = _serialize_value(large_dict)
        assert isinstance(result, dict)
        assert len(result) == 50

    def test_serialize_value_complex_objects(self):
        """Test _serialize_value with complex objects extracts attributes."""

        class CustomObject:
            def __init__(self):
                self.name = "test"
                self.value = 42

        obj = CustomObject()
        result = _serialize_value(obj)
        # Should extract __dict__ as a dictionary
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_is_traced_functionality(self):
        """Test is_traced function."""

        def regular_function():
            pass

        @trace
        def traced_function():
            pass

        assert not is_traced(regular_function)
        assert is_traced(traced_function)

    def test_get_trace_config_functionality(self):
        """Test get_trace_config function."""

        def regular_function():
            pass

        @trace(name="custom", capture_args=False)
        def traced_function():
            pass

        assert get_trace_config(regular_function) is None

        config = get_trace_config(traced_function)
        assert config is not None
        assert config["name"] == "custom"
        assert config["capture_args"] is False


class TestDecoratorEdgeCases:
    """Test edge cases and error conditions."""

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_decorator_with_async_function(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test decorators with async functions."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace
        async def async_function():
            await asyncio.sleep(0)
            return "async_result"

        # Note: We're not actually running the async function in this test
        # Just testing that the decorator can be applied to async functions
        assert hasattr(async_function, "_noveum_traced")
        assert async_function._noveum_traced is True

    def test_decorator_with_sampling(self):
        """Test decorator with custom sampling function."""
        sample_count = 0

        def custom_sampler(*args, **kwargs):
            nonlocal sample_count
            sample_count += 1
            return sample_count % 2 == 0  # Sample every other call

        @trace(sample_fn=custom_sampler)
        def sampled_function():
            return "result"

        # First call should not be sampled
        result1 = sampled_function()
        assert result1 == "result"

        # Second call should be sampled
        result2 = sampled_function()
        assert result2 == "result"

    def test_decorator_metadata_types(self):
        """Test decorators with various metadata types."""
        metadata = {
            "string_value": "test",
            "int_value": 123,
            "float_value": 3.14,
            "bool_value": True,
            "list_value": [1, 2, 3],
            "dict_value": {"nested": "value"},
        }

        @trace(metadata=metadata)
        def test_function():
            return "result"

        config = get_trace_config(test_function)
        assert config["metadata"] == metadata


class TestDecoratorIntegration:
    """Test decorator integration scenarios."""

    @patch("noveum_trace.core.context.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    def test_agent_using_tools(
        self, mock_initialized, mock_get_client, mock_get_current_trace
    ):
        """Test agent using multiple tools."""
        mock_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        @trace_tool(tool_name="search")
        def search_tool(query: str) -> list:
            return [f"Result for {query}"]

        @trace_tool(tool_name="summarize")
        def summarize_tool(data: list) -> str:
            return f"Summary of {len(data)} items"

        @trace_agent("research_agent")
        def research_agent(topic: str) -> dict:
            results = search_tool(topic)
            summary = summarize_tool(results)
            return {"topic": topic, "summary": summary}

        result = research_agent("AI")

        assert result["topic"] == "AI"
        assert "Summary of 1 items" in result["summary"]
        # Should have created spans for agent and both tools
        assert mock_client.start_span.call_count >= 3

    def test_multiple_decorators_on_same_function(self):
        """Test applying multiple decorators to the same function."""

        # This should work but may have undefined behavior
        @trace
        @trace_llm
        def multi_decorated_function():
            return "result"

        # Function should still be callable
        result = multi_decorated_function()
        assert result == "result"

    def test_decorator_inheritance(self):
        """Test decorators work with class methods."""

        class TracedClass:
            @trace
            def instance_method(self):
                return "instance_result"

            @classmethod
            @trace
            def class_method(cls):
                return "class_result"

            @staticmethod
            @trace
            def static_method():
                return "static_result"

        obj = TracedClass()

        # All methods should be traced
        assert is_traced(obj.instance_method)
        assert is_traced(TracedClass.class_method)
        assert is_traced(TracedClass.static_method)

        # And should still work normally
        assert obj.instance_method() == "instance_result"
        assert TracedClass.class_method() == "class_result"
        assert TracedClass.static_method() == "static_result"
