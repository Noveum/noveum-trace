"""
Integration tests for LangChain callback handler.

These tests verify that the NoveumTraceCallbackHandler integrates correctly
with LangChain components and produces the expected traces and spans.
"""

import pytest
from unittest.mock import Mock, patch

# Skip all tests if LangChain is not available
pytest_plugins = []

try:
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestLangChainIntegration:
    """Test LangChain integration functionality."""

    def test_callback_handler_initialization(self):
        """Test that the callback handler initializes correctly."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            assert handler._client == mock_client
            assert handler._trace_stack == []
            assert handler._span_stack == []
            assert handler._current_trace is None

    def test_callback_handler_initialization_no_client(self):
        """Test callback handler initialization when client is not available."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Client not initialized")

            handler = NoveumTraceCallbackHandler()

            assert handler._client is None

    def test_should_create_trace_logic(self):
        """Test the trace creation logic."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            # Chain and agent events should always create traces
            assert handler._should_create_trace("chain_start", {}) is True
            assert handler._should_create_trace("agent_start", {}) is True

            # LLM and retriever events should create traces only if no active traces
            assert handler._should_create_trace("llm_start", {}) is True
            assert handler._should_create_trace("retriever_start", {}) is True

            # When there are active traces, should not create new ones
            handler._trace_stack.append(Mock())
            assert handler._should_create_trace("llm_start", {}) is False
            assert handler._should_create_trace("retriever_start", {}) is False

    def test_operation_name_generation(self):
        """Test operation name generation."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            # Test various operation types
            assert (
                handler._get_operation_name("llm_start", {"name": "gpt-4"})
                == "llm.gpt-4"
            )
            assert (
                handler._get_operation_name("chain_start", {"name": "my_chain"})
                == "chain.my_chain"
            )
            assert (
                handler._get_operation_name("agent_start", {"name": "my_agent"})
                == "agent.my_agent"
            )
            assert (
                handler._get_operation_name("retriever_start", {"name": "vector_store"})
                == "retrieval.vector_store"
            )
            assert (
                handler._get_operation_name("tool_start", {"name": "calculator"})
                == "tool.calculator"
            )

            # Test with unknown name
            assert handler._get_operation_name("llm_start", {}) == "llm.unknown"

            # Test with unknown event type
            assert (
                handler._get_operation_name("custom_start", {"name": "test"})
                == "custom_start.test"
            )

    def test_llm_start_standalone(self):
        """Test LLM start event for standalone call."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            handler.on_llm_start(
                serialized={"name": "gpt-4", "id": ["openai", "gpt-4"]},
                prompts=["Hello world"],
                run_id=run_id,
            )

            # Should create trace for standalone LLM call
            mock_client.start_trace.assert_called_once_with("llm.gpt-4")
            mock_client.start_span.assert_called_once()

            # Check span attributes
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]
            assert attributes["langchain.run_id"] == str(run_id)
            assert attributes["llm.model"] == "gpt-4"
            assert attributes["llm.provider"] == "gpt-4"
            assert attributes["llm.prompts"] == ["Hello world"]
            assert attributes["llm.prompt_count"] == 1

            assert len(handler._trace_stack) == 1
            assert len(handler._span_stack) == 1

    def test_llm_end_success(self):
        """Test LLM end event with successful completion."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._current_trace = mock_trace
            handler._trace_stack = [mock_trace]
            handler._span_stack = [mock_span]

            # Mock LLM response
            mock_response = Mock()
            mock_generation = Mock()
            mock_generation.text = "Paris is the capital of France"
            mock_response.generations = [[mock_generation]]
            mock_response.llm_output = {
                "token_usage": {"total_tokens": 15},
                "finish_reason": "stop",
            }

            run_id = uuid4()

            handler.on_llm_end(response=mock_response, run_id=run_id)

            # Should set span attributes and finish span
            mock_span.set_attributes.assert_called_once()
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once_with(mock_span)

            # Should finish trace since it was standalone
            mock_client.finish_trace.assert_called_once_with(mock_trace)

            assert len(handler._trace_stack) == 0
            assert len(handler._span_stack) == 0
            assert handler._current_trace is None

    def test_llm_error_handling(self):
        """Test LLM error event handling."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._current_trace = mock_trace
            handler._trace_stack = [mock_trace]
            handler._span_stack = [mock_span]

            error = Exception("API key invalid")
            run_id = uuid4()

            handler.on_llm_error(error=error, run_id=run_id)

            # Should record exception and set error status
            mock_span.record_exception.assert_called_once_with(error)
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once_with(mock_span)

            # Should finish trace since it was standalone
            mock_client.finish_trace.assert_called_once_with(mock_trace)

            assert len(handler._trace_stack) == 0
            assert len(handler._span_stack) == 0
            assert handler._current_trace is None

    def test_chain_workflow(self):
        """Test complete chain workflow."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Chain start
            handler.on_chain_start(
                serialized={"name": "llm_chain"}, inputs={"topic": "AI"}, run_id=run_id
            )

            # Should create trace and span
            mock_client.start_trace.assert_called_once_with("chain.llm_chain")
            mock_client.start_span.assert_called_once()

            # Chain end
            handler.on_chain_end(outputs={"text": "AI is fascinating"}, run_id=run_id)

            # Should finish span and trace
            mock_span.set_attributes.assert_called()
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once_with(mock_span)
            mock_client.finish_trace.assert_called_once_with(mock_trace)

    def test_no_client_graceful_handling(self):
        """Test that operations are gracefully handled when no client is available."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Client not initialized")

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # These should not raise exceptions
            handler.on_llm_start(
                serialized={"name": "gpt-4"}, prompts=["Hello"], run_id=run_id
            )

            handler.on_llm_end(response=Mock(), run_id=run_id)
            handler.on_llm_error(error=Exception("test"), run_id=run_id)

            # No traces or spans should be created
            assert len(handler._trace_stack) == 0
            assert len(handler._span_stack) == 0

    def test_repr(self):
        """Test string representation of callback handler."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            repr_str = repr(handler)
            assert "NoveumTraceCallbackHandler" in repr_str
            assert "active_traces=0" in repr_str
            assert "active_spans=0" in repr_str
