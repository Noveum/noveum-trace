"""
Integration tests for LangChain callback handler.

These tests verify that the NoveumTraceCallbackHandler integrates correctly
with LangChain components and produces the expected traces and spans.
"""

from unittest.mock import Mock, patch

import pytest

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
                serialized={
                    "name": "gpt-4",
                    "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                },
                prompts=["Hello world"],
                run_id=run_id,
            )

            # Should create trace for standalone LLM call
            mock_client.start_trace.assert_called_once_with("llm.openai")
            mock_client.start_span.assert_called_once()

            # Check span attributes
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]
            assert attributes["langchain.run_id"] == str(run_id)
            assert attributes["llm.model"] == "gpt-4"
            assert attributes["llm.provider"] == "ChatOpenAI"
            assert attributes["llm.input.prompts"] == ["Hello world"]
            assert attributes["llm.input.prompt_count"] == 1

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

    def test_extract_model_name(self):
        """Test model name extraction from serialized LLM data."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            # Test with model in kwargs
            serialized = {"kwargs": {"model": "gpt-4-turbo"}}
            assert handler._extract_model_name(serialized) == "gpt-4-turbo"

            # Test with provider from id path
            serialized = {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]}
            assert handler._extract_model_name(serialized) == "openai"

            # Test fallback to class name
            serialized = {"name": "GPT4"}
            assert handler._extract_model_name(serialized) == "GPT4"

            # Test empty/None serialized
            assert handler._extract_model_name({}) == "unknown"
            assert handler._extract_model_name(None) == "unknown"

    def test_extract_agent_type(self):
        """Test agent type extraction from serialized agent data."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            # Test with agent type from id path
            serialized = {"id": ["langchain", "agents", "react", "ReActAgent"]}
            assert handler._extract_agent_type(serialized) == "react"

            # Test with different agent type
            serialized = {"id": ["langchain", "agents", "zero_shot", "ZeroShotAgent"]}
            assert handler._extract_agent_type(serialized) == "zero_shot"

            # Test empty/None serialized
            assert handler._extract_agent_type({}) == "unknown"
            assert handler._extract_agent_type(None) == "unknown"

    def test_extract_agent_capabilities(self):
        """Test agent capabilities extraction from tools in serialized data."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            # Test with various tool types
            serialized = {
                "kwargs": {
                    "tools": [
                        {"name": "web_search"},
                        {"name": "calculator"},
                        {"name": "file_reader"},
                        {"name": "api_client"},
                    ]
                }
            }
            capabilities = handler._extract_agent_capabilities(serialized)
            assert "tool_usage" in capabilities
            assert "web_search" in capabilities
            assert "calculation" in capabilities
            assert "file_operations" in capabilities
            assert "api_calls" in capabilities

            # Test with no tools (default reasoning)
            serialized = {"kwargs": {"tools": []}}
            assert handler._extract_agent_capabilities(serialized) == "reasoning"

            # Test empty/None serialized
            assert handler._extract_agent_capabilities({}) == "unknown"
            assert handler._extract_agent_capabilities(None) == "unknown"

    def test_extract_tool_function_name(self):
        """Test tool function name extraction from serialized tool data."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            # Test with function name in kwargs
            serialized = {"kwargs": {"name": "search_web"}}
            assert handler._extract_tool_function_name(serialized) == "search_web"

            # Test fallback to class name
            serialized = {"name": "WebSearchTool"}
            assert handler._extract_tool_function_name(serialized) == "WebSearchTool"

            # Test empty/None serialized
            assert handler._extract_tool_function_name({}) == "unknown"
            assert handler._extract_tool_function_name(None) == "unknown"

    def test_llm_start_with_new_attributes(self):
        """Test LLM start event with new attribute structure."""
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
                serialized={"name": "gpt-4", "kwargs": {"model": "gpt-4-turbo"}},
                prompts=["Hello world", "How are you?"],
                run_id=run_id,
            )

            # Check span attributes include new structure
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]
            assert attributes["llm.operation"] == "completion"
            assert attributes["llm.input.prompts"] == ["Hello world", "How are you?"]
            assert attributes["llm.input.prompt_count"] == 2

    def test_llm_end_with_new_attributes(self):
        """Test LLM end event with new flattened usage attributes."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = [mock_span]

            # Mock LLM response with token usage
            mock_response = Mock()
            mock_generation = Mock()
            mock_generation.text = "Paris is the capital of France"
            mock_response.generations = [[mock_generation]]
            mock_response.llm_output = {
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
                "finish_reason": "stop",
            }

            run_id = uuid4()
            handler.on_llm_end(response=mock_response, run_id=run_id)

            # Check new flattened attributes structure
            call_args = mock_span.set_attributes.call_args
            attributes = call_args[0][0]
            assert attributes["llm.output.response"] == [
                "Paris is the capital of France"
            ]
            assert attributes["llm.output.response_count"] == 1
            assert attributes["llm.output.finish_reason"] == "stop"
            assert attributes["llm.input_tokens"] == 10
            assert attributes["llm.output_tokens"] == 8
            assert attributes["llm.total_tokens"] == 18

    def test_chain_start_with_new_attributes(self):
        """Test chain start event with new attribute structure."""
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

            handler.on_chain_start(
                serialized={"name": "llm_chain"},
                inputs={"topic": "AI", "style": "academic"},
                run_id=run_id,
            )

            # Check span attributes include new structure
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]
            assert attributes["chain.operation"] == "execution"
            assert "chain.inputs" in attributes

    def test_chain_end_with_new_attributes(self):
        """Test chain end event with new output attribute structure."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = [mock_span]

            run_id = uuid4()
            handler.on_chain_end(
                outputs={"text": "AI is fascinating", "confidence": 0.95}, run_id=run_id
            )

            # Check new output attributes structure
            call_args = mock_span.set_attributes.call_args
            attributes = call_args[0][0]
            assert "chain.output.outputs" in attributes

    def test_tool_start_with_new_attributes(self):
        """Test tool start event with enhanced naming and attributes."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            handler.on_tool_start(
                serialized={"name": "WebSearchTool", "kwargs": {"name": "search_web"}},
                input_str="What is the capital of France?",
                run_id=run_id,
            )

            # Check enhanced span name and attributes
            call_args = mock_client.start_span.call_args
            span_name = call_args[1]["name"]
            assert span_name == "tool:WebSearchTool:search_web"

            attributes = call_args[1]["attributes"]
            assert attributes["tool.name"] == "WebSearchTool"
            assert attributes["tool.operation"] == "search_web"
            assert (
                attributes["tool.input.input_str"] == "What is the capital of France?"
            )

    def test_tool_end_with_new_attributes(self):
        """Test tool end event with new output attribute structure."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = [mock_span]

            run_id = uuid4()
            handler.on_tool_end(output="Paris is the capital of France.", run_id=run_id)

            # Check new output attributes structure
            call_args = mock_span.set_attributes.call_args
            attributes = call_args[0][0]
            assert attributes["tool.output.output"] == "Paris is the capital of France."

    def test_agent_start_functionality(self):
        """Test agent start event functionality."""
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

            handler.on_agent_start(
                serialized={
                    "name": "ReActAgent",
                    "id": ["langchain", "agents", "react", "ReActAgent"],
                    "kwargs": {
                        "tools": [{"name": "web_search"}, {"name": "calculator"}]
                    },
                },
                inputs={"input": "What is 2+2 and what's the weather?"},
                run_id=run_id,
            )

            # Should create trace and span for agent
            mock_client.start_trace.assert_called_once()
            mock_client.start_span.assert_called_once()

            # Check span attributes
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]
            assert attributes["agent.name"] == "ReActAgent"
            assert attributes["agent.type"] == "react"
            assert attributes["agent.operation"] == "execution"
            assert "tool_usage" in attributes["agent.capabilities"]
            assert "web_search" in attributes["agent.capabilities"]
            assert "calculation" in attributes["agent.capabilities"]

    def test_agent_action_with_new_attributes(self):
        """Test agent action event with new output attributes."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = [mock_span]

            # Mock agent action
            mock_action = Mock()
            mock_action.tool = "calculator"
            mock_action.tool_input = "2+2"
            mock_action.log = "I need to calculate 2+2"

            run_id = uuid4()
            handler.on_agent_action(action=mock_action, run_id=run_id)

            # Check that attributes were set with new structure
            mock_span.set_attributes.assert_called_once()
            call_args = mock_span.set_attributes.call_args
            attributes = call_args[0][0]
            assert attributes["agent.output.action.tool"] == "calculator"
            assert attributes["agent.output.action.tool_input"] == "2+2"
            assert attributes["agent.output.action.log"] == "I need to calculate 2+2"

    def test_agent_finish_with_new_attributes(self):
        """Test agent finish event with enhanced functionality."""
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

            # Mock agent finish
            mock_finish = Mock()
            mock_finish.return_values = {"output": "The answer is 4"}
            mock_finish.log = "Task completed successfully"

            run_id = uuid4()
            handler.on_agent_finish(finish=mock_finish, run_id=run_id)

            # Check that span was finished with proper attributes
            mock_span.set_attributes.assert_called_once()
            call_args = mock_span.set_attributes.call_args
            attributes = call_args[0][0]
            assert (
                attributes["agent.output.finish.return_values"]["output"]
                == "The answer is 4"
            )
            assert (
                attributes["agent.output.finish.log"] == "Task completed successfully"
            )

            # Should finish span and trace
            mock_client.finish_span.assert_called_once_with(mock_span)
            mock_client.finish_trace.assert_called_once_with(mock_trace)

    def test_retriever_start_with_new_attributes(self):
        """Test retriever start event with new attribute structure."""
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

            handler.on_retriever_start(
                serialized={"name": "VectorStoreRetriever"},
                query="What is machine learning?",
                run_id=run_id,
            )

            # Check span attributes include new structure
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]
            assert attributes["retrieval.type"] == "search"
            assert attributes["retrieval.operation"] == "VectorStoreRetriever"
            assert attributes["retrieval.query"] == "What is machine learning?"

    def test_retriever_end_with_new_attributes(self):
        """Test retriever end event with new output attribute structure."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = [mock_span]

            # Mock documents
            mock_doc1 = Mock()
            mock_doc1.page_content = "Machine learning is a subset of AI"
            mock_doc2 = Mock()
            mock_doc2.page_content = "It involves training algorithms on data"

            documents = [mock_doc1, mock_doc2]

            run_id = uuid4()
            handler.on_retriever_end(documents=documents, run_id=run_id)

            # Check new output attributes structure
            call_args = mock_span.set_attributes.call_args
            attributes = call_args[0][0]
            assert attributes["retrieval.result_count"] == 2
            assert len(attributes["retrieval.sample_results"]) == 2
            assert attributes["retrieval.results_truncated"] is False

    def test_operation_name_with_model_extraction(self):
        """Test operation name generation with model name extraction."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            # Test LLM operation name with model extraction
            serialized = {"name": "ChatOpenAI", "kwargs": {"model": "gpt-4-turbo"}}
            assert (
                handler._get_operation_name("llm_start", serialized)
                == "llm.gpt-4-turbo"
            )

            # Test fallback to provider
            serialized = {
                "name": "ChatOpenAI",
                "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
            }
            assert handler._get_operation_name("llm_start", serialized) == "llm.openai"

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

    def test_chain_error_handling(self):
        """Test chain error event handling."""
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

            error = Exception("Chain execution failed")
            run_id = uuid4()

            result = handler.on_chain_error(error=error, run_id=run_id)

            # Should record exception and set error status
            mock_span.record_exception.assert_called_once_with(error)
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once_with(mock_span)

            # Should finish trace since it was standalone
            mock_client.finish_trace.assert_called_once_with(mock_trace)

            assert len(handler._trace_stack) == 0
            assert len(handler._span_stack) == 0
            assert handler._current_trace is None
            assert result is None

    def test_tool_error_handling(self):
        """Test tool error event handling."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = [mock_span]

            error = Exception("Tool execution failed")
            run_id = uuid4()

            result = handler.on_tool_error(error=error, run_id=run_id)

            # Should record exception and set error status
            mock_span.record_exception.assert_called_once_with(error)
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once_with(mock_span)

            assert len(handler._span_stack) == 0
            assert result is None

    def test_retriever_error_handling(self):
        """Test retriever error event handling."""
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

            error = Exception("Retrieval failed")
            run_id = uuid4()

            result = handler.on_retriever_error(error=error, run_id=run_id)

            # Should record exception and set error status
            mock_span.record_exception.assert_called_once_with(error)
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once_with(mock_span)

            # Should finish trace since it was standalone
            mock_client.finish_trace.assert_called_once_with(mock_trace)

            assert len(handler._trace_stack) == 0
            assert len(handler._span_stack) == 0
            assert handler._current_trace is None
            assert result is None

    def test_agent_error_handling(self):
        """Test agent error event handling - method doesn't exist in LangChain."""
        # The on_agent_error method doesn't exist in the LangChain callback handler
        # This test is kept for completeness but will be skipped
        pytest.skip(
            "on_agent_error method not implemented in LangChain callback handler"
        )

    def test_nested_llm_in_chain(self):
        """Test LLM call within a chain (should not create new trace)."""
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

            # Start chain (creates trace)
            handler.on_chain_start(
                serialized={"name": "llm_chain"}, inputs={"topic": "AI"}, run_id=run_id
            )

            # LLM within chain (should not create new trace)
            handler.on_llm_start(
                serialized={"name": "gpt-4"}, prompts=["Hello"], run_id=run_id
            )

            # Should only have one trace created (for chain)
            assert mock_client.start_trace.call_count == 1
            # Should have two spans (chain + LLM)
            assert mock_client.start_span.call_count == 2

    def test_nested_tool_in_agent(self):
        """Test tool call within an agent (should not create new trace)."""
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

            # Start agent (creates trace)
            handler.on_agent_start(
                serialized={"name": "ReActAgent"},
                inputs={"input": "test"},
                run_id=run_id,
            )

            # Tool within agent (should not create new trace)
            handler.on_tool_start(
                serialized={"name": "calculator"}, input_str="2+2", run_id=run_id
            )

            # Should only have one trace created (for agent)
            assert mock_client.start_trace.call_count == 1
            # Should have two spans (agent + tool)
            assert mock_client.start_span.call_count == 2

    def test_nested_retriever_in_chain(self):
        """Test retriever call within a chain (should not create new trace)."""
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

            # Start chain (creates trace)
            handler.on_chain_start(
                serialized={"name": "rag_chain"},
                inputs={"query": "test"},
                run_id=run_id,
            )

            # Retriever within chain (should not create new trace)
            handler.on_retriever_start(
                serialized={"name": "VectorStoreRetriever"}, query="test", run_id=run_id
            )

            # Should only have one trace created (for chain)
            assert mock_client.start_trace.call_count == 1
            # Should have two spans (chain + retriever)
            assert mock_client.start_span.call_count == 2

    def test_empty_llm_response(self):
        """Test LLM end event with empty response."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = [mock_span]

            # Mock empty LLM response
            mock_response = Mock()
            mock_response.generations = []
            mock_response.llm_output = {}

            run_id = uuid4()
            handler.on_llm_end(response=mock_response, run_id=run_id)

            # Check that attributes were set correctly for empty response
            call_args = mock_span.set_attributes.call_args
            attributes = call_args[0][0]
            assert attributes["llm.output.response"] == []
            assert attributes["llm.output.response_count"] == 0
            assert attributes["llm.output.finish_reason"] is None

    def test_large_input_truncation(self):
        """Test that large inputs are properly truncated."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Test large number of prompts truncation (limited to 5)
            many_prompts = [f"prompt {i}" for i in range(10)]
            handler.on_llm_start(
                serialized={"name": "gpt-4"}, prompts=many_prompts, run_id=run_id
            )

            # Check that prompts were limited to 5
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]
            assert len(attributes["llm.input.prompts"]) <= 5

    def test_large_chain_input_truncation(self):
        """Test that large chain inputs are stored without truncation."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Test large input (no truncation expected)
            large_input = "x" * 300
            handler.on_chain_start(
                serialized={"name": "test_chain"},
                inputs={"large_input": large_input},
                run_id=run_id,
            )

            # Check that input was stored without truncation
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]
            assert len(attributes["chain.inputs"]["large_input"]) == 300
            assert attributes["chain.inputs"]["large_input"] == large_input

    def test_missing_llm_output(self):
        """Test LLM end event with missing llm_output."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = [mock_span]

            # Mock LLM response without llm_output
            mock_response = Mock()
            mock_response.generations = []
            mock_response.llm_output = None

            run_id = uuid4()
            handler.on_llm_end(response=mock_response, run_id=run_id)

            # Should handle gracefully
            mock_span.set_attributes.assert_called_once()
            mock_span.set_status.assert_called_once()

    def test_text_event_handling(self):
        """Test text event handler."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = [mock_span]

            run_id = uuid4()
            handler.on_text(text="Some text output", run_id=run_id)

            # Should add event to current span
            mock_span.add_event.assert_called_once()
            call_args = mock_span.add_event.call_args
            assert call_args[0][0] == "text_output"
            assert "text" in call_args[0][1]

    def test_text_event_large_text_truncation(self):
        """Test text event with large text (no truncation expected)."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = [mock_span]

            run_id = uuid4()
            large_text = "x" * 300
            handler.on_text(text=large_text, run_id=run_id)

            # Check that text was stored without truncation
            call_args = mock_span.add_event.call_args
            event_data = call_args[0][1]
            assert len(event_data["text"]) == 300
            assert event_data["text"] == large_text

    def test_ensure_client_recovery(self):
        """Test _ensure_client method and client recovery."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Create handler with no client initially
            handler = NoveumTraceCallbackHandler()
            handler._client = None

            # Should return True when client is available
            assert handler._ensure_client() is True
            assert handler._client == mock_client

    def test_ensure_client_with_existing_client(self):
        """Test _ensure_client with existing client."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Create handler and manually set client
            handler = NoveumTraceCallbackHandler()
            handler._client = mock_client

            # Should return True immediately
            assert handler._ensure_client() is True
            # Should not call get_client again (it was called once during initialization)
            assert mock_get_client.call_count == 1

    def test_operations_with_no_client(self):
        """Test operations when client is not available."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Client not available")

            handler = NoveumTraceCallbackHandler()
            handler._client = None
            run_id = uuid4()

            # All operations should handle gracefully
            handler.on_llm_start(
                serialized={"name": "gpt-4"}, prompts=["test"], run_id=run_id
            )
            handler.on_llm_end(response=Mock(), run_id=run_id)
            handler.on_chain_start(
                serialized={"name": "test"}, inputs={}, run_id=run_id
            )
            handler.on_chain_end(outputs={}, run_id=run_id)
            handler.on_tool_start(
                serialized={"name": "test"}, input_str="test", run_id=run_id
            )
            handler.on_tool_end(output="test", run_id=run_id)
            handler.on_agent_start(
                serialized={"name": "test"}, inputs={}, run_id=run_id
            )
            handler.on_agent_action(action=Mock(), run_id=run_id)
            handler.on_agent_finish(finish=Mock(), run_id=run_id)
            handler.on_retriever_start(
                serialized={"name": "test"}, query="test", run_id=run_id
            )
            handler.on_retriever_end(documents=[], run_id=run_id)
            handler.on_text(text="test", run_id=run_id)

            # No exceptions should be raised
            assert True

    def test_operations_with_empty_span_stack(self):
        """Test operations when span stack is empty."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler._span_stack = []
            run_id = uuid4()

            # These operations should handle gracefully
            handler.on_llm_end(response=Mock(), run_id=run_id)
            handler.on_chain_end(outputs={}, run_id=run_id)
            handler.on_tool_end(output="test", run_id=run_id)
            handler.on_agent_action(action=Mock(), run_id=run_id)
            handler.on_agent_finish(finish=Mock(), run_id=run_id)
            handler.on_retriever_end(documents=[], run_id=run_id)
            handler.on_text(text="test", run_id=run_id)

            # No exceptions should be raised
            assert True
