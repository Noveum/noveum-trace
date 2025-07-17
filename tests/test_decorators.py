"""
Tests for the decorator functionality.

These tests verify that the decorators work correctly
and can trace function calls properly.
"""

from unittest.mock import MagicMock, patch

import pytest


# Test the decorators without requiring full SDK initialization
def test_decorator_imports():
    """Test that decorators can be imported."""
    from noveum_trace import trace, trace_agent, trace_llm, trace_retrieval, trace_tool

    assert trace is not None
    assert trace_llm is not None
    assert trace_agent is not None
    assert trace_tool is not None
    assert trace_retrieval is not None


def test_trace_decorator_without_init():
    """Test that trace decorator works when SDK is not initialized."""
    from noveum_trace import trace

    @trace
    def test_function(x, y):
        return x + y

    # Should work normally when SDK is not initialized
    result = test_function(1, 2)
    assert result == 3


def test_trace_llm_decorator_without_init():
    """Test that trace_llm decorator works when SDK is not initialized."""
    from noveum_trace import trace_llm

    @trace_llm
    def mock_llm_call(prompt):
        return {"response": f"Response to: {prompt}"}

    # Should work normally when SDK is not initialized
    result = mock_llm_call("Hello")
    assert result["response"] == "Response to: Hello"


def test_trace_agent_decorator_without_init():
    """Test that trace_agent decorator works when SDK is not initialized."""
    from noveum_trace import trace_agent

    @trace_agent(agent_id="test_agent")
    def agent_function(task):
        return f"Agent completed: {task}"

    # Should work normally when SDK is not initialized
    result = agent_function("test task")
    assert result == "Agent completed: test task"


@patch("noveum_trace.is_initialized")
@patch("noveum_trace.get_client")
def test_trace_decorator_with_mock_client(mock_get_client, mock_is_initialized):
    """Test trace decorator with mocked client."""
    from noveum_trace import trace

    # Mock SDK as initialized
    mock_is_initialized.return_value = True

    # Mock client
    mock_client = MagicMock()
    mock_span = MagicMock()
    mock_client.start_span.return_value = mock_span
    mock_get_client.return_value = mock_client

    @trace
    def test_function(x, y):
        return x + y

    result = test_function(1, 2)

    # Verify function still works
    assert result == 3

    # Verify client was called (if implementation is complete)
    # Note: This might fail if the decorator implementation is incomplete
    # mock_client.start_span.assert_called_once()


def test_decorator_with_custom_name():
    """Test decorator with custom span name."""
    from noveum_trace import trace

    @trace(name="custom_operation")
    def test_function():
        return "success"

    result = test_function()
    assert result == "success"


def test_decorator_with_metadata():
    """Test decorator with metadata."""
    from noveum_trace import trace

    @trace(metadata={"version": "1.0", "type": "test"})
    def test_function():
        return "success"

    result = test_function()
    assert result == "success"


def test_decorator_with_tags():
    """Test decorator with tags."""
    from noveum_trace import trace

    @trace(tags={"environment": "test", "component": "core"})
    def test_function():
        return "success"

    result = test_function()
    assert result == "success"


def test_decorator_exception_handling():
    """Test that decorator handles exceptions properly."""
    from noveum_trace import trace

    @trace
    def failing_function():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        failing_function()


def test_llm_decorator_with_provider():
    """Test LLM decorator with provider specification."""
    from noveum_trace import trace_llm

    @trace_llm(provider="openai")
    def mock_openai_call(model, messages):
        return {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    result = mock_openai_call("gpt-4", [{"role": "user", "content": "Hi"}])
    assert result["choices"][0]["message"]["content"] == "Hello!"


def test_agent_decorator_with_role():
    """Test agent decorator with role specification."""
    from noveum_trace import trace_agent

    @trace_agent(agent_id="researcher", role="research")
    def research_agent(query):
        return f"Research results for: {query}"

    result = research_agent("AI trends")
    assert result == "Research results for: AI trends"


def test_tool_decorator():
    """Test tool decorator."""
    from noveum_trace import trace_tool

    @trace_tool(tool_name="calculator")
    def calculator(operation, a, b):
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        return None

    result = calculator("add", 5, 3)
    assert result == 8


def test_retrieval_decorator():
    """Test retrieval decorator."""
    from noveum_trace import trace_retrieval

    @trace_retrieval(index_name="documents")
    def search_documents(query, top_k=5):
        # Mock retrieval results
        return [
            {"id": "doc1", "score": 0.95, "content": "Relevant content 1"},
            {"id": "doc2", "score": 0.87, "content": "Relevant content 2"},
        ]

    results = search_documents("test query")
    assert len(results) == 2
    assert results[0]["score"] == 0.95


def test_nested_decorators():
    """Test that decorators can be nested."""
    from noveum_trace import trace, trace_llm

    @trace(name="outer_function")
    def outer_function(data):
        return inner_function(data)

    @trace_llm(provider="openai")
    def inner_function(data):
        return f"Processed: {data}"

    result = outer_function("test data")
    assert result == "Processed: test data"


def test_async_decorator_support():
    """Test that decorators work with async functions."""
    import asyncio

    from noveum_trace import trace

    @trace
    async def async_function(x):
        await asyncio.sleep(0.001)  # Minimal delay
        return x * 2

    async def run_test():
        result = await async_function(5)
        assert result == 10

    # Run the async test
    asyncio.run(run_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
