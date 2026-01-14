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
    from noveum_trace.integrations.langchain.langchain_utils import (
        build_langgraph_attributes,
        build_routing_attributes,
        extract_agent_capabilities,
        extract_agent_type,
        extract_langgraph_metadata,
        extract_model_name,
        extract_noveum_metadata,
        extract_tool_function_name,
        get_langgraph_operation_name,
        get_operation_name,
    )

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
            assert handler.runs == {}
            assert handler._trace_managed_by_langchain is None

    def test_callback_handler_initialization_no_client(self):
        """Test callback handler initialization when client is not available."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Client not initialized")

            handler = NoveumTraceCallbackHandler()

            assert handler._client is None

    def test_trace_context_management(self):
        """Test the trace context management logic."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            # Test getting or creating trace context when no existing trace
            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch(
                    "noveum_trace.core.context.set_current_trace"
                ) as mock_set_current:
                    mock_get_current.return_value = None
                    mock_client.start_trace.return_value = mock_trace

                    trace, should_manage = handler._get_or_create_trace_context(
                        "test_operation"
                    )

                    assert trace == mock_trace
                    assert should_manage is True
                    mock_client.start_trace.assert_called_once_with(
                        "test_operation")
                    mock_set_current.assert_called_once_with(mock_trace)

            # Test using existing trace
            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch(
                    "noveum_trace.core.context.set_current_trace"
                ) as mock_set_current:
                    mock_get_current.return_value = mock_trace

                    # Reset the mock to clear previous calls
                    mock_client.start_trace.reset_mock()

                    trace, should_manage = handler._get_or_create_trace_context(
                        "test_operation"
                    )

                    assert trace == mock_trace
                    assert should_manage is False  # Don't manage existing trace
                    mock_client.start_trace.assert_not_called()
                    mock_set_current.assert_not_called()

    def test_operation_name_generation(self):
        """Test operation name generation."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            # Test various operation types
            assert get_operation_name(
                "llm_start", {"name": "gpt-4"}) == "llm.gpt-4"
            assert (
                get_operation_name("chain_start", {"name": "my_chain"})
                == "chain.my_chain"
            )
            assert (
                get_operation_name("agent_start", {"name": "my_agent"})
                == "agent.my_agent"
            )
            assert (
                get_operation_name("retriever_start", {"name": "vector_store"})
                == "retrieval.vector_store"
            )
            assert (
                get_operation_name("tool_start", {"name": "calculator"})
                == "tool.calculator"
            )

            # Test with unknown name
            assert get_operation_name("llm_start", {}) == "llm.unknown"

            # Test with unknown event type
            assert (
                get_operation_name("custom_start", {"name": "test"})
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

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = None  # No existing trace

                    handler.on_llm_start(
                        serialized={
                            "name": "gpt-4",
                            "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                        },
                        prompts=["Hello world"],
                        run_id=run_id,
                    )

                    # Should create trace for standalone LLM call
                    mock_client.start_trace.assert_called_once_with(
                        "llm.openai")
                    mock_client.start_span.assert_called_once()

                    # Check span attributes
                    call_args = mock_client.start_span.call_args
                    attributes = call_args[1]["attributes"]
                    assert attributes["langchain.run_id"] == str(run_id)
                    # Uses provider name from ID path (second-to-last element)
                    assert attributes["llm.model"] == "openai"
                    assert attributes["llm.provider"] == "openai"
                    assert attributes["llm.input.prompts"] == ["Hello world"]
                    assert attributes["llm.input.prompt_count"] == 1

                    assert len(handler.runs) == 1
                    assert handler._trace_managed_by_langchain == mock_trace

    def test_llm_end_success(self):
        """Test LLM end event with successful completion."""
        from datetime import datetime, timezone
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()
            # Set up span with required attributes
            mock_span.start_time = datetime.now(timezone.utc)
            mock_span.attributes = {
                "llm.provider": "openai", "llm.model": "gpt-4"}

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler._set_run(run_id, mock_span)
            handler._trace_managed_by_langchain = mock_trace

            # Mock LLM response
            mock_response = Mock()
            mock_generation = Mock()
            mock_generation.text = "Paris is the capital of France"
            mock_response.generations = [[mock_generation]]
            mock_response.llm_output = {
                "token_usage": {"total_tokens": 15},
                "finish_reason": "stop",
            }

            handler.on_llm_end(response=mock_response, run_id=run_id)

            # Should set span attributes and finish span
            mock_span.set_attributes.assert_called_once()
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once()

            # Should finish trace since it was standalone
            mock_client.finish_trace.assert_called_once_with(mock_trace)

            assert len(handler.runs) == 0
            assert handler._trace_managed_by_langchain is None

    def test_llm_error_handling(self):
        """Test LLM error event handling."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler.runs[run_id] = mock_span
            handler._trace_managed_by_langchain = mock_trace

            error = Exception("API key invalid")

            handler.on_llm_error(error=error, run_id=run_id)

            # Should record exception and set error status
            mock_span.record_exception.assert_called_once_with(error)
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once_with(mock_span)

            # Should finish trace since it was standalone
            mock_client.finish_trace.assert_called_once_with(mock_trace)

            assert len(handler.runs) == 0
            assert handler._trace_managed_by_langchain is None

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

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = None  # No existing trace

                    # Chain start
                    handler.on_chain_start(
                        serialized={"name": "llm_chain"},
                        inputs={"topic": "AI"},
                        run_id=run_id,
                    )

                    # Should create trace and span
                    mock_client.start_trace.assert_called_once_with(
                        "chain.llm_chain")
                    mock_client.start_span.assert_called_once()

                    # Chain end
                    handler.on_chain_end(
                        outputs={"text": "AI is fascinating"}, run_id=run_id
                    )

                    # Should finish span and trace
                    mock_span.set_attributes.assert_called()
                    mock_span.set_status.assert_called_once()
                    mock_client.finish_span.assert_called_once_with(mock_span)
                    mock_client.finish_trace.assert_called_once_with(
                        mock_trace)

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
            assert len(handler.runs) == 0
            assert handler._trace_managed_by_langchain is None

    def test_extract_model_name(self):
        """Test model name extraction from serialized LLM data."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            # Test with model in kwargs
            serialized = {"kwargs": {"model": "gpt-4-turbo"}}
            assert extract_model_name(serialized) == "gpt-4-turbo"

            # Test with provider from id path
            serialized = {
                "id": ["langchain", "chat_models", "openai", "ChatOpenAI"]}
            assert extract_model_name(serialized) == "openai"

            # Test fallback to class name
            serialized = {"name": "GPT4"}
            assert extract_model_name(serialized) == "GPT4"

            # Test empty/None serialized
            assert extract_model_name({}) == "unknown"
            assert extract_model_name(None) == "unknown"

            # Test with short id path (edge case)
            serialized = {"id": ["openai"]}
            assert extract_model_name(serialized) == "unknown"

            # Test with no name in serialized
            serialized = {
                "id": ["langchain", "chat_models", "openai", "ChatOpenAI"]}
            assert extract_model_name(serialized) == "openai"

    def test_extract_agent_type(self):
        """Test agent type extraction from serialized agent data."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            # Test with agent type from id path
            serialized = {"id": ["langchain", "agents", "react", "ReActAgent"]}
            assert extract_agent_type(serialized) == "react"

            # Test with different agent type
            serialized = {"id": ["langchain", "agents",
                                 "zero_shot", "ZeroShotAgent"]}
            assert extract_agent_type(serialized) == "zero_shot"

            # Test empty/None serialized
            assert extract_agent_type({}) == "unknown"
            assert extract_agent_type(None) == "unknown"

    def test_extract_agent_capabilities(self):
        """Test agent capabilities extraction from tools in serialized data."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

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
            capabilities = extract_agent_capabilities(serialized)
            assert "tool_usage" in capabilities
            assert "web_search" in capabilities
            assert "calculation" in capabilities
            assert "file_operations" in capabilities
            assert "api_calls" in capabilities

            # Test with no tools (default reasoning)
            serialized = {"kwargs": {"tools": []}}
            assert extract_agent_capabilities(serialized) == "reasoning"

            # Test empty/None serialized
            assert extract_agent_capabilities({}) == "unknown"
            assert extract_agent_capabilities(None) == "unknown"

    def test_extract_tool_function_name(self):
        """Test tool function name extraction from serialized tool data."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            # Test with function name in kwargs
            serialized = {"kwargs": {"name": "search_web"}}
            assert extract_tool_function_name(serialized) == "search_web"

            # Test fallback to class name
            serialized = {"name": "WebSearchTool"}
            assert extract_tool_function_name(serialized) == "WebSearchTool"

            # Test empty/None serialized
            assert extract_tool_function_name({}) == "unknown"
            assert extract_tool_function_name(None) == "unknown"

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
                serialized={"name": "gpt-4",
                            "kwargs": {"model": "gpt-4-turbo"}},
                prompts=["Hello world", "How are you?"],
                run_id=run_id,
            )

            # Check span attributes include new structure
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]
            assert attributes["llm.operation"] == "completion"
            assert attributes["llm.input.prompts"] == [
                "Hello world", "How are you?"]
            assert attributes["llm.input.prompt_count"] == 2

    def test_llm_end_with_new_attributes(self):
        """Test LLM end event with new flattened usage attributes."""
        from datetime import datetime, timezone
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()
            # Set up span with required attributes
            mock_span.start_time = datetime.now(timezone.utc)
            mock_span.attributes = {
                "llm.provider": "openai", "llm.model": "gpt-4"}

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler._set_run(run_id, mock_span)

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

            handler.on_llm_end(response=mock_response, run_id=run_id)

            # Check new flattened attributes structure
            assert mock_span.set_attributes.called
            call_args = mock_span.set_attributes.call_args
            assert call_args, "set_attributes was not called with arguments"
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
            assert "chain.inputs.topic" in attributes
            assert "chain.inputs.style" in attributes
            assert attributes["chain.inputs.topic"] == "AI"
            assert attributes["chain.inputs.style"] == "academic"

    def test_chain_end_with_new_attributes(self):
        """Test chain end event with new output attribute structure."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler.runs[run_id] = mock_span

            handler.on_chain_end(
                outputs={"text": "AI is fascinating", "confidence": 0.95}, run_id=run_id
            )

            # Check new output attributes structure - gather all attributes from all calls
            all_attributes = {}
            for call in mock_span.set_attributes.call_args_list:
                all_attributes.update(call[0][0])
            assert "chain.output.text" in all_attributes
            assert "chain.output.confidence" in all_attributes

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
                serialized={"name": "WebSearchTool",
                            "kwargs": {"name": "search_web"}},
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
            run_id = uuid4()
            handler.runs[run_id] = mock_span

            handler.on_tool_end(
                output="Paris is the capital of France.", run_id=run_id)

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
            run_id = uuid4()
            handler.runs[run_id] = mock_span

            # Mock agent action
            mock_action = Mock()
            mock_action.tool = "calculator"
            mock_action.tool_input = "2+2"
            mock_action.log = "I need to calculate 2+2"

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
            run_id = uuid4()
            handler.runs[run_id] = mock_span
            handler._trace_managed_by_langchain = mock_trace

            # Set proper name for the mock span to avoid being treated as a tool span
            mock_span.name = "agent:test_agent"

            # Mock agent finish
            mock_finish = Mock()
            mock_finish.return_values = {"output": "The answer is 4"}
            mock_finish.log = "Task completed successfully"

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
            run_id = uuid4()
            handler.runs[run_id] = mock_span

            # Mock documents
            mock_doc1 = Mock()
            mock_doc1.page_content = "Machine learning is a subset of AI"
            mock_doc2 = Mock()
            mock_doc2.page_content = "It involves training algorithms on data"

            documents = [mock_doc1, mock_doc2]

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

            NoveumTraceCallbackHandler()

            # Test LLM operation name with model extraction
            serialized = {"name": "ChatOpenAI",
                          "kwargs": {"model": "gpt-4-turbo"}}
            assert get_operation_name(
                "llm_start", serialized) == "llm.gpt-4-turbo"

            # Test fallback to provider
            serialized = {
                "name": "ChatOpenAI",
                "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
            }
            assert get_operation_name("llm_start", serialized) == "llm.openai"

    def test_repr(self):
        """Test string representation of callback handler."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            repr_str = repr(handler)
            assert "NoveumTraceCallbackHandler" in repr_str
            assert "active_runs=0" in repr_str
            assert "managing_trace=False" in repr_str

    def test_chain_error_handling(self):
        """Test chain error event handling."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler.runs[run_id] = mock_span
            handler._trace_managed_by_langchain = mock_trace

            error = Exception("Chain execution failed")

            result = handler.on_chain_error(error=error, run_id=run_id)

            # Should record exception and set error status
            mock_span.record_exception.assert_called_once_with(error)
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once_with(mock_span)

            # Should finish trace since it was standalone
            mock_client.finish_trace.assert_called_once_with(mock_trace)

            assert len(handler.runs) == 0
            assert handler._trace_managed_by_langchain is None
            assert result is None

    def test_tool_error_handling(self):
        """Test tool error event handling."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler.runs[run_id] = mock_span

            error = Exception("Tool execution failed")

            result = handler.on_tool_error(error=error, run_id=run_id)

            # Should record exception and set error status
            mock_span.record_exception.assert_called_once_with(error)
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once_with(mock_span)

            assert len(handler.runs) == 0
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
            run_id = uuid4()
            handler.runs[run_id] = mock_span
            handler._trace_managed_by_langchain = mock_trace

            error = Exception("Retrieval failed")

            result = handler.on_retriever_error(error=error, run_id=run_id)

            # Should record exception and set error status
            mock_span.record_exception.assert_called_once_with(error)
            mock_span.set_status.assert_called_once()
            mock_client.finish_span.assert_called_once_with(mock_span)

            # Should finish trace since it was standalone
            mock_client.finish_trace.assert_called_once_with(mock_trace)

            assert len(handler.runs) == 0
            assert handler._trace_managed_by_langchain is None
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
        from datetime import datetime, timezone
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()
            # Set up span with required attributes
            mock_span.start_time = datetime.now(timezone.utc)
            mock_span.attributes = {
                "llm.provider": "openai", "llm.model": "gpt-4"}

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler._set_run(run_id, mock_span)

            # Mock empty LLM response
            mock_response = Mock()
            mock_response.generations = []
            mock_response.llm_output = {}

            handler.on_llm_end(response=mock_response, run_id=run_id)

            # Check that attributes were set correctly for empty response
            assert mock_span.set_attributes.called
            call_args = mock_span.set_attributes.call_args
            assert call_args, "set_attributes was not called with arguments"
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
            assert "chain.inputs.large_input" in attributes
            assert len(attributes["chain.inputs.large_input"]) == 300
            assert attributes["chain.inputs.large_input"] == large_input

    def test_missing_llm_output(self):
        """Test LLM end event with missing llm_output."""
        from datetime import datetime, timezone
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()
            # Set up span with required attributes
            mock_span.start_time = datetime.now(timezone.utc)
            mock_span.attributes = {
                "llm.provider": "openai", "llm.model": "gpt-4"}

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler._set_run(run_id, mock_span)

            # Mock LLM response without llm_output
            mock_response = Mock()
            mock_response.generations = []
            mock_response.llm_output = None

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
            run_id = uuid4()
            handler.runs[run_id] = mock_span

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
            run_id = uuid4()
            handler.runs[run_id] = mock_span

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

    def test_thread_safe_operations(self):
        """Test thread-safe operations on runs dictionary."""
        import threading
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Test concurrent access to runs dictionary
            def add_run():
                handler._set_run(run_id, mock_span)

            def get_run():
                return handler._get_run(run_id)

            def pop_run():
                return handler._pop_run(run_id)

            # Test thread safety
            thread1 = threading.Thread(target=add_run)
            thread2 = threading.Thread(target=get_run)
            thread3 = threading.Thread(target=pop_run)

            thread1.start()
            thread2.start()
            thread3.start()

            thread1.join()
            thread2.join()
            thread3.join()

            # Verify operations completed without errors
            assert True

    def test_tool_span_creation_error_handling(self):
        """Test error handling in tool span creation."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Mock client to raise exception
            mock_client.start_span.side_effect = Exception(
                "Span creation failed")

            # Mock agent action
            mock_action = Mock()
            mock_action.tool = "calculator"
            mock_action.tool_input = "2+2"
            mock_action.log = "I need to calculate 2+2"

            # Should handle error gracefully
            handler.on_agent_action(action=mock_action, run_id=run_id)

            # Verify error was logged but no exception was raised
            assert True

    def test_complete_tool_spans_error_handling(self):
        """Test error handling in tool span completion."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Add a tool span to runs
            mock_span = Mock()
            tool_run_id = f"{run_id}_tool_test"
            handler._set_run(tool_run_id, mock_span)

            # Mock finish with log containing observation
            mock_finish = Mock()
            mock_finish.log = "Some log\nObservation: The answer is 4\nMore log"

            # Should complete tool spans successfully
            handler._complete_tool_spans_from_finish(mock_finish, run_id)

            # Verify span was processed
            mock_span.set_attributes.assert_called()
            mock_span.set_status.assert_called()
            mock_client.finish_span.assert_called()

    def test_complete_tool_spans_with_final_answer(self):
        """Test tool span completion with final answer in log."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Add a tool span to runs
            mock_span = Mock()
            tool_run_id = f"{run_id}_tool_test"
            handler._set_run(tool_run_id, mock_span)

            # Mock finish with log containing final answer
            mock_finish = Mock()
            mock_finish.log = "Some log\nFinal Answer: The result is 42\nMore log"

            # Should complete tool spans successfully
            handler._complete_tool_spans_from_finish(mock_finish, run_id)

            # Verify span was processed with final answer
            call_args = mock_span.set_attributes.call_args[0][0]
            assert call_args["tool.output.output"] == "The result is 42"

    def test_complete_tool_spans_no_log(self):
        """Test tool span completion with no log."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Add a tool span to runs
            mock_span = Mock()
            tool_run_id = f"{run_id}_tool_test"
            handler._set_run(tool_run_id, mock_span)

            # Mock finish with no log
            mock_finish = Mock()
            mock_finish.log = None

            # Should complete tool spans successfully
            handler._complete_tool_spans_from_finish(mock_finish, run_id)

            # Verify span was processed with default result
            call_args = mock_span.set_attributes.call_args[0][0]
            assert call_args["tool.output.output"] == "Tool execution completed"

    def test_operation_name_generation_edge_cases(self):
        """Test operation name generation with edge cases."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            # Test with None serialized
            name = get_operation_name("llm_start", None)
            assert name == "llm_start.unknown"

            # Test with empty serialized
            name = get_operation_name("chain_start", {})
            assert name == "chain.unknown"

            # Test with missing name
            serialized = {
                "id": ["langchain", "chat_models", "openai", "ChatOpenAI"]}
            name = get_operation_name("llm_start", serialized)
            assert name == "llm.openai"

    def test_agent_capabilities_extraction_edge_cases(self):
        """Test agent capabilities extraction with edge cases."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            # Test with no tools
            serialized = {"name": "test_agent"}
            capabilities = extract_agent_capabilities(serialized)
            assert capabilities == "reasoning"

            # Test with empty tools
            serialized = {"name": "test_agent", "kwargs": {"tools": []}}
            capabilities = extract_agent_capabilities(serialized)
            assert capabilities == "reasoning"

            # Test with tools that have no name
            mock_tool = Mock()
            mock_tool.name = None
            serialized = {"name": "test_agent",
                          "kwargs": {"tools": [mock_tool]}}
            capabilities = extract_agent_capabilities(serialized)
            assert "tool_usage" in capabilities

    def test_llm_end_with_missing_span(self):
        """Test LLM end event when span is missing from runs."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Don't add span to runs - should handle gracefully
            handler.on_llm_end(response=Mock(), run_id=run_id)

            # Should not raise exception
            assert True

    def test_chain_end_with_missing_span(self):
        """Test chain end event when span is missing from runs."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Don't add span to runs - should handle gracefully
            handler.on_chain_end(outputs={}, run_id=run_id)

            # Should not raise exception
            assert True

    def test_agent_finish_with_missing_span(self):
        """Test agent finish event when span is missing from runs."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Don't add span to runs - should handle gracefully
            handler.on_agent_finish(finish=Mock(), run_id=run_id)

            # Should not raise exception
            assert True

    def test_retriever_end_with_missing_span(self):
        """Test retriever end event when span is missing from runs."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Don't add span to runs - should handle gracefully
            handler.on_retriever_end(documents=[], run_id=run_id)

            # Should not raise exception
            assert True

    def test_text_event_with_missing_span(self):
        """Test text event when span is missing from runs."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Don't add span to runs - should handle gracefully
            handler.on_text(text="test", run_id=run_id)

            # Should not raise exception
            assert True

    def test_operations_with_empty_runs(self):
        """Test operations when runs dictionary is empty."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            handler.runs = {}  # Empty runs dictionary
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

    # ===== NEW FEATURE TESTS =====

    # Custom Name Mapping Tests
    def test_custom_name_mapping_thread_safety(self):
        """Test thread-safe custom name mapping operations."""
        import threading

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            results = []

            def set_names():
                for i in range(10):
                    handler._set_name(f"name_{i}", f"span_{i}")

            def get_names():
                for i in range(10):
                    span_id = handler._get_span_id_by_name(f"name_{i}")
                    results.append(span_id)

            # Test concurrent access
            thread1 = threading.Thread(target=set_names)
            thread2 = threading.Thread(target=get_names)

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()

            # Verify thread safety - no exceptions should be raised
            assert True

    def test_custom_name_in_llm_span(self):
        """Test LLM span with custom name from metadata."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()
            mock_span.span_id = "test_span_id"

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = None

                    handler.on_llm_start(
                        serialized={"name": "gpt-4"},
                        prompts=["Hello"],
                        run_id=run_id,
                        metadata={"noveum": {"name": "custom_llm_name"}},
                    )

                    # Verify custom name was used
                    call_args = mock_client.start_span.call_args
                    assert call_args[1]["name"] == "custom_llm_name"

                    # Verify name was stored in names dict
                    assert (
                        handler._get_span_id_by_name("custom_llm_name")
                        == "test_span_id"
                    )

    def test_custom_name_in_chain_span(self):
        """Test chain span with custom name from metadata."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()
            mock_span.span_id = "chain_span_id"

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = None

                    handler.on_chain_start(
                        serialized={"name": "my_chain"},
                        inputs={"input": "test"},
                        run_id=run_id,
                        metadata={"noveum": {"name": "custom_chain_name"}},
                    )

                    # Verify custom name was used
                    call_args = mock_client.start_span.call_args
                    assert call_args[1]["name"] == "custom_chain_name"

                    # Verify name was stored
                    assert (
                        handler._get_span_id_by_name("custom_chain_name")
                        == "chain_span_id"
                    )

    def test_custom_name_in_tool_span(self):
        """Test tool span with custom name from metadata."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()
            mock_span.span_id = "tool_span_id"

            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            handler.on_tool_start(
                serialized={"name": "calculator"},
                input_str="2+2",
                run_id=run_id,
                metadata={"noveum": {"name": "custom_tool_name"}},
            )

            # Verify custom name was used
            call_args = mock_client.start_span.call_args
            assert call_args[1]["name"] == "custom_tool_name"

            # Verify name was stored
            assert handler._get_span_id_by_name(
                "custom_tool_name") == "tool_span_id"

    def test_parent_name_resolution(self):
        """Test parent span resolution by custom name."""

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            # Set up a parent span with custom name
            handler._set_name("parent_span", "parent_span_id")

            # Test resolution
            parent_id = handler._get_parent_span_id_from_name("parent_span")
            assert parent_id == "parent_span_id"

    def test_parent_name_not_found_warning(self):
        """Test warning when parent name not found."""

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            with patch(
                "noveum_trace.integrations.langchain.langchain.logger"
            ) as mock_logger:
                parent_id = handler._get_parent_span_id_from_name(
                    "nonexistent_parent")

                assert parent_id is None
                mock_logger.warning.assert_called_once()
                assert "Parent span with name 'nonexistent_parent' not found" in str(
                    mock_logger.warning.call_args
                )

    # Noveum Metadata Extraction Tests
    def test_extract_noveum_metadata_valid(self):
        """Test valid noveum metadata extraction."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            metadata = {"noveum": {"name": "custom_name",
                                   "parent_name": "parent_span"}}

            result = extract_noveum_metadata(metadata)
            assert result == {"name": "custom_name",
                              "parent_name": "parent_span"}

    def test_extract_noveum_metadata_missing(self):
        """Test noveum metadata extraction with missing data."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            # Test None metadata
            result = extract_noveum_metadata(None)
            assert result == {}

            # Test missing noveum key
            result = extract_noveum_metadata({})
            assert result == {}

            # Test empty noveum dict
            result = extract_noveum_metadata({"noveum": {}})
            assert result == {}

    def test_extract_noveum_metadata_invalid_type(self):
        """Test noveum metadata extraction with invalid type."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            # Test non-dict noveum config
            metadata = {"noveum": "not_a_dict"}
            result = extract_noveum_metadata(metadata)
            assert result == {}

    def test_noveum_metadata_in_all_handlers(self):
        """Test noveum metadata extraction in all event handlers."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()
            mock_span.span_id = "test_span_id"

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            metadata = {"noveum": {
                "name": "test_name", "parent_name": "parent"}}

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = None

                    # Test LLM handler
                    handler.on_llm_start(
                        serialized={"name": "gpt-4"},
                        prompts=["test"],
                        run_id=run_id,
                        metadata=metadata,
                    )

                    # Test chain handler
                    handler.on_chain_start(
                        serialized={"name": "chain"},
                        inputs={},
                        run_id=run_id,
                        metadata=metadata,
                    )

                    # Test tool handler
                    handler.on_tool_start(
                        serialized={"name": "tool"},
                        input_str="test",
                        run_id=run_id,
                        metadata=metadata,
                    )

                    # Test agent handler
                    handler.on_agent_start(
                        serialized={"name": "agent"},
                        inputs={},
                        run_id=run_id,
                        metadata=metadata,
                    )

                    # Test retriever handler
                    handler.on_retriever_start(
                        serialized={"name": "retriever"},
                        query="test",
                        run_id=run_id,
                        metadata=metadata,
                    )

                    # Verify all handlers used custom name
                    assert mock_client.start_span.call_count == 5
                    for call in mock_client.start_span.call_args_list:
                        assert call[1]["name"] == "test_name"

    def test_custom_name_stored_in_names_dict(self):
        """Test that custom names are stored in names dictionary."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()
            mock_span.span_id = "stored_span_id"

            mock_client.start_trace.return_value = mock_trace
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = None

                    handler.on_llm_start(
                        serialized={"name": "gpt-4"},
                        prompts=["test"],
                        run_id=run_id,
                        metadata={"noveum": {"name": "stored_name"}},
                    )

                    # Verify name was stored
                    assert (
                        handler._get_span_id_by_name(
                            "stored_name") == "stored_span_id"
                    )
                    assert "stored_name" in handler.names

    # Parent Span Resolution Tests
    def test_resolve_parent_legacy_mode(self):
        """Test parent resolution in legacy mode (use_langchain_assigned_parent=False)."""

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=False)

            # Set up parent span with custom name
            handler._set_name("parent_span", "parent_span_id")

            # Test with parent_name only
            parent_id = handler._resolve_parent_span_id(None, "parent_span")
            assert parent_id == "parent_span_id"

            # Test with no parent_name
            parent_id = handler._resolve_parent_span_id(None, None)
            assert parent_id is None

    def test_resolve_parent_langchain_mode(self):
        """Test parent resolution in langchain mode (use_langchain_assigned_parent=True)."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True)

            # Set up parent span
            mock_parent_span = Mock()
            mock_parent_span.span_id = "parent_span_id"
            handler._set_run(uuid4(), mock_parent_span)

            parent_run_id = uuid4()
            handler._set_run(parent_run_id, mock_parent_span)

            # Test with parent_run_id
            parent_id = handler._resolve_parent_span_id(parent_run_id, None)
            assert parent_id == "parent_span_id"

    def test_resolve_parent_via_parent_run_id(self):
        """Test parent resolution via parent_run_id lookup."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True)

            # Set up parent span
            mock_parent_span = Mock()
            mock_parent_span.span_id = "parent_span_id"
            parent_run_id = uuid4()
            handler._set_run(parent_run_id, mock_parent_span)

            # Test resolution
            parent_id = handler._resolve_parent_span_id(parent_run_id, None)
            assert parent_id == "parent_span_id"

    def test_resolve_parent_via_parent_name(self):
        """Test parent resolution via parent_name fallback."""

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True)

            # Set up parent span with custom name
            handler._set_name("parent_span", "parent_span_id")

            # Test resolution with parent_name fallback
            parent_id = handler._resolve_parent_span_id(None, "parent_span")
            assert parent_id == "parent_span_id"

    def test_resolve_parent_fallback_to_context(self):
        """Test parent resolution fallback to context with warning."""

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True)

            # Mock current span in context
            mock_current_span = Mock()
            mock_current_span.span_id = "context_span_id"

            with patch(
                "noveum_trace.core.context.get_current_span"
            ) as mock_get_current:
                mock_get_current.return_value = mock_current_span

                with patch(
                    "noveum_trace.integrations.langchain.langchain.logger"
                ) as mock_logger:
                    parent_id = handler._resolve_parent_span_id(None, None)

                    assert parent_id == "context_span_id"
                    mock_logger.warning.assert_called_once()
                    # Updated to match new warning message format
                    assert "Could not resolve parent from parent_run_id" in str(
                        mock_logger.warning.call_args
                    )

    def test_resolve_parent_no_parent_found(self):
        """Test parent resolution when no parent found at all."""

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True)

            with patch(
                "noveum_trace.core.context.get_current_span"
            ) as mock_get_current:
                mock_get_current.return_value = None

                parent_id = handler._resolve_parent_span_id(None, None)
                assert parent_id is None

    # Tests for prioritize_manually_assigned_parents feature
    def test_prioritize_manual_parents_default_false(self):
        """Test default behavior: parent_run_id has priority over parent_name."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True,
                prioritize_manually_assigned_parents=False,
            )

            # Set up both parent_run_id and parent_name
            mock_parent_span_runid = Mock()
            mock_parent_span_runid.span_id = "parent_via_run_id"
            parent_run_id = uuid4()
            handler._set_run(parent_run_id, mock_parent_span_runid)

            handler._set_name("manual_parent", "parent_via_name")

            # When both are available, parent_run_id should win
            parent_id = handler._resolve_parent_span_id(
                parent_run_id, "manual_parent")
            assert parent_id == "parent_via_run_id"

    def test_prioritize_manual_parents_enabled(self):
        """Test prioritize_manually_assigned_parents=True: parent_name has priority."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True,
                prioritize_manually_assigned_parents=True,
            )

            # Set up both parent_run_id and parent_name
            mock_parent_span_runid = Mock()
            mock_parent_span_runid.span_id = "parent_via_run_id"
            parent_run_id = uuid4()
            handler._set_run(parent_run_id, mock_parent_span_runid)

            handler._set_name("manual_parent", "parent_via_name")

            # When both are available, parent_name should win
            parent_id = handler._resolve_parent_span_id(
                parent_run_id, "manual_parent")
            assert parent_id == "parent_via_name"

    def test_prioritize_manual_parents_fallback_to_run_id(self):
        """Test that when parent_name is not found, falls back to parent_run_id."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True,
                prioritize_manually_assigned_parents=True,
            )

            # Set up only parent_run_id
            mock_parent_span_runid = Mock()
            mock_parent_span_runid.span_id = "parent_via_run_id"
            parent_run_id = uuid4()
            handler._set_run(parent_run_id, mock_parent_span_runid)

            # parent_name is provided but not found in names dict
            with patch("noveum_trace.integrations.langchain.langchain.logger"):
                parent_id = handler._resolve_parent_span_id(
                    parent_run_id, "nonexistent_parent"
                )
                # Should fallback to parent_run_id
                assert parent_id == "parent_via_run_id"

    def test_prioritize_manual_parents_fallback_to_name(self):
        """Test default mode: when parent_run_id is not found, falls back to parent_name."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True,
                prioritize_manually_assigned_parents=False,
            )

            # Set up only parent_name
            handler._set_name("manual_parent", "parent_via_name")

            # parent_run_id is provided but not found in runs dict
            from uuid import uuid4

            nonexistent_run_id = uuid4()
            parent_id = handler._resolve_parent_span_id(
                nonexistent_run_id, "manual_parent"
            )
            # Should fallback to parent_name
            assert parent_id == "parent_via_name"

    def test_prioritize_manual_parents_context_fallback(self):
        """Test context fallback with manual override mode indicator in warning."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True,
                prioritize_manually_assigned_parents=True,
            )

            # Mock current span in context
            mock_current_span = Mock()
            mock_current_span.span_id = "context_span_id"

            with patch(
                "noveum_trace.core.context.get_current_span"
            ) as mock_get_current:
                mock_get_current.return_value = mock_current_span

                with patch(
                    "noveum_trace.integrations.langchain.langchain.logger"
                ) as mock_logger:
                    parent_id = handler._resolve_parent_span_id(None, None)

                    assert parent_id == "context_span_id"
                    mock_logger.warning.assert_called_once()
                    # Should include manual override mode indicator
                    assert "(manual override mode)" in str(
                        mock_logger.warning.call_args
                    )

    def test_prioritize_manual_parents_ignored_in_legacy_mode(self):
        """Test that prioritize_manually_assigned_parents is ignored in legacy mode."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Legacy mode with prioritize_manually_assigned_parents=True (should be ignored)
            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=False,
                prioritize_manually_assigned_parents=True,
            )

            # Set up parent_name
            handler._set_name("manual_parent", "parent_via_name")

            # Should still use legacy behavior (only check parent_name)
            parent_id = handler._resolve_parent_span_id(None, "manual_parent")
            assert parent_id == "parent_via_name"

            # Should return None when no parent_name
            parent_id = handler._resolve_parent_span_id(None, None)
            assert parent_id is None

    def test_handler_initialization_defaults(self):
        """Test that handler initializes with correct new defaults."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            # New defaults
            assert handler._use_langchain_assigned_parent is True
            assert handler._prioritize_manually_assigned_parents is False

    def test_handler_repr_includes_new_flag(self):
        """Test that __repr__ includes the prioritize_manual_parents flag."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True,
                prioritize_manually_assigned_parents=True,
            )

            repr_str = repr(handler)
            assert "prioritize_manual_parents=True" in repr_str

    def test_parent_span_id_in_llm_start(self):
        """Test that parent_span_id is passed to start_span in LLM start."""
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

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = None

                    handler.on_llm_start(
                        serialized={"name": "gpt-4"},
                        prompts=["test"],
                        run_id=run_id,
                        metadata={"noveum": {"parent_name": "parent_span"}},
                    )

                    # Verify parent_span_id was passed
                    call_args = mock_client.start_span.call_args
                    assert "parent_span_id" in call_args[1]

    def test_parent_span_id_in_chain_start(self):
        """Test that parent_span_id is passed to start_span in chain start."""
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

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = None

                    handler.on_chain_start(
                        serialized={"name": "chain"},
                        inputs={},
                        run_id=run_id,
                        metadata={"noveum": {"parent_name": "parent_span"}},
                    )

                    # Verify parent_span_id was passed
                    call_args = mock_client.start_span.call_args
                    assert "parent_span_id" in call_args[1]

    # LangGraph Support Tests
    def test_extract_langgraph_metadata_from_metadata(self):
        """Test LangGraph metadata extraction from metadata dict."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            metadata = {
                "langgraph_node": "research_node",
                "langgraph_step": 3,
                "langgraph_graph_name": "research_graph",
                "langgraph_checkpoint_ns": "checkpoint_1",
                "langgraph_path": ["__pregel_pull", "research_node", "other"],
            }

            result = extract_langgraph_metadata(metadata, None, None)

            assert result["is_langgraph"] is True
            assert result["node_name"] == "research_node"
            assert result["step"] == 3
            assert result["graph_name"] == "research_graph"
            assert result["checkpoint_ns"] == "checkpoint_1"
            assert result["execution_type"] == "node"

    def test_extract_langgraph_metadata_from_tags(self):
        """Test LangGraph metadata extraction from tags."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            tags = ["langgraph:analysis_node", "other_tag"]

            result = extract_langgraph_metadata(None, tags, None)

            assert result["is_langgraph"] is True
            assert result["node_name"] == "analysis_node"

    def test_extract_langgraph_metadata_from_serialized(self):
        """Test LangGraph metadata extraction from serialized dict."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            serialized = {
                "id": ["langgraph", "graphs", "research_graph"],
                "name": "ResearchGraph",
            }

            result = extract_langgraph_metadata(None, None, serialized)

            assert result["is_langgraph"] is True
            assert result["graph_name"] == "ResearchGraph"

    def test_extract_langgraph_metadata_none_serialized(self):
        """Test LangGraph metadata extraction with None serialized."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            result = extract_langgraph_metadata(None, None, None)

            assert result["is_langgraph"] is False
            assert result["node_name"] is None
            assert result["step"] is None
            assert result["graph_name"] is None

    def test_extract_langgraph_metadata_safe_fallback(self):
        """Test LangGraph metadata extraction never breaks with invalid data."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            # Test with invalid data types
            result = extract_langgraph_metadata(
                "invalid", "invalid", "invalid")

            # Should return safe defaults
            assert result["is_langgraph"] is False
            assert result["node_name"] is None

    def test_langgraph_operation_name_node(self):
        """Test LangGraph operation name generation with node name."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            langgraph_metadata = {"node_name": "research_node"}
            result = get_langgraph_operation_name(
                langgraph_metadata, "unknown")

            assert result == "graph.node.research_node"

    def test_langgraph_operation_name_graph(self):
        """Test LangGraph operation name generation with graph name."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            langgraph_metadata = {"graph_name": "research_graph"}
            result = get_langgraph_operation_name(
                langgraph_metadata, "unknown")

            assert result == "graph.research_graph"

    def test_langgraph_operation_name_step(self):
        """Test LangGraph operation name generation with step number."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            langgraph_metadata = {"step": 5}
            result = get_langgraph_operation_name(
                langgraph_metadata, "unknown")

            assert result == "graph.node.step_5"

    def test_build_langgraph_attributes(self):
        """Test building LangGraph span attributes."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            langgraph_metadata = {
                "is_langgraph": True,
                "node_name": "research_node",
                "step": 3,
                "graph_name": "research_graph",
                "checkpoint_ns": "checkpoint_1",
                "execution_type": "node",
            }

            result = build_langgraph_attributes(langgraph_metadata)

            assert result["langgraph.is_graph"] is True
            assert result["langgraph.node_name"] == "research_node"
            assert result["langgraph.step"] == 3
            assert result["langgraph.graph_name"] == "research_graph"
            assert result["langgraph.checkpoint_ns"] == "checkpoint_1"
            assert result["langgraph.execution_type"] == "node"

    def test_chain_start_with_langgraph_metadata(self):
        """Test chain start with LangGraph metadata."""
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

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = None

                    handler.on_chain_start(
                        serialized=None,  # LangGraph often passes None
                        inputs={},
                        run_id=run_id,
                        metadata={
                            "langgraph_node": "research_node",
                            "langgraph_step": 3,
                        },
                        tags=["langgraph:research_node"],
                    )

                    # Verify LangGraph operation name was used
                    call_args = mock_client.start_span.call_args
                    assert call_args[1]["name"] == "graph.node.research_node"

                    # Verify LangGraph attributes were added
                    attributes = call_args[1]["attributes"]
                    assert attributes["langgraph.is_graph"] is True
                    assert attributes["langgraph.node_name"] == "research_node"
                    assert attributes["langgraph.step"] == 3

    # Manual Trace Control Tests
    def test_manual_start_trace(self):
        """Test manual trace start functionality."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_trace.trace_id = "test_trace_id"

            mock_client.start_trace.return_value = mock_trace
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch(
                    "noveum_trace.core.context.set_current_trace"
                ) as mock_set_current:
                    mock_get_current.return_value = None

                    handler.start_trace("test_trace")

                    # Verify trace was created and set in context
                    mock_client.start_trace.assert_called_once_with(
                        "test_trace")
                    mock_set_current.assert_called_once_with(mock_trace)

                    # Verify manual control flags
                    assert handler._manual_trace_control is True
                    assert handler._trace_managed_by_langchain == mock_trace

    def test_manual_end_trace(self):
        """Test manual trace end functionality."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_trace.trace_id = "test_trace_id"

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch(
                    "noveum_trace.core.context.set_current_trace"
                ) as mock_set_current:
                    mock_get_current.return_value = mock_trace

                    handler.end_trace()

                    # Verify trace was finished and cleared from context
                    mock_client.finish_trace.assert_called_once_with(
                        mock_trace)
                    mock_set_current.assert_called_once_with(None)

                    # Verify manual control flags reset
                    assert handler._manual_trace_control is False
                    assert handler._trace_managed_by_langchain is None

    def test_manual_trace_error_already_exists(self):
        """Test warning when trying to start trace when one already exists."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_trace.trace_id = "existing_trace_id"

            new_trace = Mock()
            new_trace.trace_id = "new_trace_id"
            mock_client.start_trace.return_value = new_trace

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch(
                    "noveum_trace.core.context.set_current_trace"
                ) as mock_set_current:
                    mock_get_current.return_value = mock_trace

                    # Should log warning but still create new trace
                    handler.start_trace("new_trace")

                    # Verify new trace was created and set
                    mock_client.start_trace.assert_called_once_with(
                        "new_trace")
                    mock_set_current.assert_called_once_with(new_trace)

                    # Verify manual control flags are set
                    assert handler._manual_trace_control is True
                    assert handler._trace_managed_by_langchain == new_trace

    def test_manual_trace_error_no_trace(self):
        """Test error when trying to end trace when none exists."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                mock_get_current.return_value = None

                with patch(
                    "noveum_trace.integrations.langchain.langchain.logger"
                ) as mock_logger:
                    handler.end_trace()

                    # Should log an error instead of raising
                    mock_logger.error.assert_called_once_with(
                        "No active trace to end")

    def test_manual_trace_disables_auto_finish(self):
        """Test that manual trace control disables auto-finish."""
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

            # Start manual trace
            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = None
                    handler.start_trace("manual_trace")

            # Add a span
            handler.runs[run_id] = mock_span
            handler._trace_managed_by_langchain = mock_trace

            # End the span - should NOT auto-finish trace
            handler.on_llm_end(response=Mock(), run_id=run_id)

            # Verify trace was NOT finished
            mock_client.finish_trace.assert_not_called()

    def test_manual_trace_reenables_auto_management(self):
        """Test that manual trace end re-enables auto-management."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            uuid4()

            # Set up manual control
            handler._manual_trace_control = True
            handler._trace_managed_by_langchain = mock_trace

            # End manual trace
            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                with patch("noveum_trace.core.context.set_current_trace"):
                    mock_get_current.return_value = mock_trace
                    handler.end_trace()

            # Verify flags were reset
            assert handler._manual_trace_control is False
            assert handler._trace_managed_by_langchain is None

    def test_manual_trace_control_flag_in_repr(self):
        """Test that __repr__ shows manual trace control state."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Test with manual control enabled
            handler = NoveumTraceCallbackHandler()
            handler._manual_trace_control = True

            repr_str = repr(handler)
            assert "manual_control=True" in repr_str

            # Test with manual control disabled
            handler._manual_trace_control = False
            repr_str = repr(handler)
            assert "manual_control=False" in repr_str

    # Routing Decision Handling Tests
    def test_on_custom_event_routing_decision(self):
        """Test custom event handling for routing decisions."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            with patch.object(handler, "_handle_routing_decision") as mock_handle:
                payload = {"source_node": "research",
                           "target_node": "analysis"}

                handler.on_custom_event(
                    name="langgraph.routing_decision", data=payload, run_id=run_id
                )

                mock_handle.assert_called_once_with(payload, run_id)

    def test_handle_routing_decision_creates_span(self):
        """Test that routing decision creates a routing span."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                mock_get_current.return_value = mock_trace

                payload = {"source_node": "research",
                           "target_node": "analysis"}

                handler._handle_routing_decision(payload, run_id)

                # Verify span was created
                mock_client.start_span.assert_called_once()
                call_args = mock_client.start_span.call_args
                assert call_args[1]["name"] == "routing.research_to_analysis"

                # Verify span was finished immediately
                mock_span.set_attributes.assert_called_once()
                mock_span.set_status.assert_called_once()
                mock_client.finish_span.assert_called_once_with(mock_span)

    def test_routing_span_name_format(self):
        """Test routing span name format."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                mock_get_current.return_value = mock_trace

                payload = {"source_node": "node_a", "target_node": "node_b"}

                handler._handle_routing_decision(payload, run_id)

                call_args = mock_client.start_span.call_args
                assert call_args[1]["name"] == "routing.node_a_to_node_b"

    def test_build_routing_attributes_core(self):
        """Test building core routing attributes."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            payload = {
                "source_node": "research",
                "target_node": "analysis",
                "decision": "analysis",
            }

            result = build_routing_attributes(payload)

            assert result["routing.source_node"] == "research"
            assert result["routing.target_node"] == "analysis"
            assert result["routing.decision"] == "analysis"
            assert result["routing.type"] == "conditional_edge"

    def test_build_routing_attributes_optional(self):
        """Test building optional routing attributes."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            NoveumTraceCallbackHandler()

            payload = {
                "source_node": "research",
                "target_node": "analysis",
                "reason": "Data is ready for analysis",
                "confidence": 0.95,
                "tool_scores": {"analyzer": 0.9, "classifier": 0.7},
                "alternatives": ["research", "review"],
                "state_snapshot": {"data": "processed"},
            }

            result = build_routing_attributes(payload)

            assert result["routing.reason"] == "Data is ready for analysis"
            assert result["routing.confidence"] == 0.95
            assert result["routing.tool_scores"] == str(payload["tool_scores"])
            assert result["routing.score.analyzer"] == 0.9
            assert result["routing.score.classifier"] == 0.7
            assert result["routing.alternatives"] == str(
                payload["alternatives"])
            assert result["routing.alternatives_count"] == 2
            assert result["routing.state_snapshot"] == str(
                payload["state_snapshot"])

    def test_routing_span_with_parent(self):
        """Test routing span creation with parent span."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()
            mock_parent_span = Mock()
            mock_parent_span.span_id = "parent_span_id"

            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            # Set up parent span
            handler._set_run(run_id, mock_parent_span)

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                mock_get_current.return_value = mock_trace

                payload = {"source_node": "research",
                           "target_node": "analysis"}

                handler._handle_routing_decision(payload, run_id)

                # Verify parent_span_id was passed
                call_args = mock_client.start_span.call_args
                assert call_args[1]["parent_span_id"] == "parent_span_id"

    def test_routing_span_without_parent(self):
        """Test routing span creation without parent (root-level)."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_trace = Mock()
            mock_span = Mock()

            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            with patch(
                "noveum_trace.core.context.get_current_trace"
            ) as mock_get_current:
                mock_get_current.return_value = mock_trace

                payload = {"source_node": "research",
                           "target_node": "analysis"}

                handler._handle_routing_decision(payload, run_id)

                # Verify no parent_span_id was passed (root-level span)
                call_args = mock_client.start_span.call_args
                assert call_args[1]["parent_span_id"] is None

    # Enhanced Initialization Tests
    def test_init_with_use_langchain_assigned_parent_false(self):
        """Test initialization with use_langchain_assigned_parent=False (default)."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=False)

            assert handler._use_langchain_assigned_parent is False
            assert handler._manual_trace_control is False
            assert handler._trace_managed_by_langchain is None
            assert handler.runs == {}
            assert handler.names == {}

    def test_init_with_use_langchain_assigned_parent_true(self):
        """Test initialization with use_langchain_assigned_parent=True."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True)

            assert handler._use_langchain_assigned_parent is True
            assert handler._manual_trace_control is False
            assert handler._trace_managed_by_langchain is None
            assert handler.runs == {}
            assert handler.names == {}

    def test_repr_shows_use_langchain_parent_flag(self):
        """Test that __repr__ shows use_langchain_assigned_parent flag."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Test with use_langchain_assigned_parent=True
            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=True)
            repr_str = repr(handler)
            assert "use_langchain_parent=True" in repr_str

            # Test with use_langchain_assigned_parent=False
            handler = NoveumTraceCallbackHandler(
                use_langchain_assigned_parent=False)
            repr_str = repr(handler)
            assert "use_langchain_parent=False" in repr_str

    # Additional Edge Case Tests
    def test_chain_end_with_non_dict_outputs(self):
        """Test chain end with non-dict outputs (string)."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler.runs[run_id] = mock_span

            # Test with string output
            handler.on_chain_end(outputs="Simple string output", run_id=run_id)

            # Verify attributes were set correctly
            call_args = mock_span.set_attributes.call_args
            attributes = call_args[0][0]
            assert attributes["chain.output.outputs"] == "Simple string output"

    def test_tool_start_with_inputs_dict(self):
        """Test tool start with structured inputs dictionary."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()

            handler.on_tool_start(
                serialized={"name": "calculator",
                            "kwargs": {"name": "calculate"}},
                input_str="2+2",
                run_id=run_id,
                inputs={"expression": "2+2", "precision": 2},
            )

            # Verify structured inputs were captured
            call_args = mock_client.start_span.call_args
            attributes = call_args[1]["attributes"]
            assert attributes["tool.input.expression"] == "2+2"
            assert attributes["tool.input.precision"] == "2"
            assert attributes["tool.input.argument_count"] == "2"

    def test_llm_end_without_token_usage(self):
        """Test LLM end without token usage data."""
        from datetime import datetime, timezone
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()
            # Set up span with required attributes
            mock_span.start_time = datetime.now(timezone.utc)
            mock_span.attributes = {
                "llm.provider": "openai", "llm.model": "gpt-4"}

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler._set_run(run_id, mock_span)

            # Mock LLM response without token usage
            mock_response = Mock()
            mock_generation = Mock()
            mock_generation.text = "Test response"
            mock_response.generations = [[mock_generation]]
            mock_response.llm_output = {}  # No token_usage

            handler.on_llm_end(response=mock_response, run_id=run_id)

            # Verify attributes were set without token usage
            assert mock_span.set_attributes.called
            call_args = mock_span.set_attributes.call_args
            assert call_args, "set_attributes was not called with arguments"
            attributes = call_args[0][0]
            assert attributes["llm.output.response"] == ["Test response"]
            assert attributes["llm.output.response_count"] == 1
            # Token usage should be 0 or missing
            assert (
                "llm.input_tokens" not in attributes
                or attributes["llm.input_tokens"] == 0
            )

    def test_retriever_end_truncation(self):
        """Test retriever end with >10 documents (truncation)."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()

            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler.runs[run_id] = mock_span

            # Create 15 mock documents
            documents = []
            for i in range(15):
                mock_doc = Mock()
                mock_doc.page_content = f"Document {i} content"
                documents.append(mock_doc)

            handler.on_retriever_end(documents=documents, run_id=run_id)

            # Verify truncation occurred
            call_args = mock_span.set_attributes.call_args
            attributes = call_args[0][0]
            assert attributes["retrieval.result_count"] == 15
            # Truncated to 10
            assert len(attributes["retrieval.sample_results"]) == 10
            assert attributes["retrieval.results_truncated"] is True

    def test_agent_action_creates_tool_span(self):
        """Test that agent action creates tool span."""
        from uuid import uuid4

        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()
            mock_tool_span = Mock()

            mock_client.start_span.return_value = mock_tool_span
            mock_get_client.return_value = mock_client

            handler = NoveumTraceCallbackHandler()
            run_id = uuid4()
            handler.runs[run_id] = mock_span

            # Mock agent action
            mock_action = Mock()
            mock_action.tool = "calculator"
            mock_action.tool_input = "2+2"
            mock_action.log = "I need to calculate 2+2"

            handler.on_agent_action(action=mock_action, run_id=run_id)

            # Verify tool span was created
            mock_client.start_span.assert_called_once()
            call_args = mock_client.start_span.call_args
            assert "tool:calculator:calculator" in call_args[1]["name"]
            assert call_args[1]["attributes"]["tool.name"] == "calculator"
            assert call_args[1]["attributes"]["tool.operation"] == "calculator"
