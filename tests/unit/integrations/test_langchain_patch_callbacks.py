"""
Unit tests for LangChain patch callback method updates.

Tests the updated callback methods that now use:
- New _get_or_create_trace_context signature with run_id and parent_run_id
- Enhanced input handling with safe_inputs_to_dict()
- Proper integration with root trace tracking
"""

from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

# Skip all tests if LangChain is not available
try:
    # Import directly from the module to avoid issues with other integrations
    from noveum_trace.integrations.langchain.langchain import NoveumTraceCallbackHandler

    LANGCHAIN_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    LANGCHAIN_AVAILABLE = False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestCallbackMethodUpdates:
    """Test updated callback methods with new trace context signature."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    @pytest.fixture
    def mock_trace_context(self):
        """Mock trace context creation."""
        mock_trace = Mock()
        mock_trace.trace_id = "test_trace"
        return mock_trace, True  # (trace, should_manage)

    def test_on_llm_start_with_new_signature(self, handler, mock_trace_context):
        """Test on_llm_start uses new _get_or_create_trace_context signature."""
        run_id = uuid4()
        parent_run_id = uuid4()
        serialized = {
            "name": "test_llm",
            "id": ["langchain", "llms", "openai", "OpenAI"],
        }
        prompts = ["test prompt"]

        # Mock the trace context method
        with (
            patch.object(
                handler, "_get_or_create_trace_context", return_value=mock_trace_context
            ) as mock_context,
            patch.object(handler, "_resolve_parent_span_id", return_value=None),
        ):

            # Mock span creation
            mock_span = Mock()
            handler._client.start_span.return_value = mock_span

            # Call method
            handler.on_llm_start(
                serialized=serialized,
                prompts=prompts,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )

        # Should have called _get_or_create_trace_context with new signature
        mock_context.assert_called_once()
        call_args = mock_context.call_args[0]
        assert len(call_args) == 3  # operation_name, run_id, parent_run_id
        assert call_args[1] == run_id
        assert call_args[2] == parent_run_id

    def test_on_chain_start_with_new_signature(self, handler, mock_trace_context):
        """Test on_chain_start uses new _get_or_create_trace_context signature."""
        run_id = uuid4()
        parent_run_id = uuid4()
        serialized = {"name": "test_chain"}
        inputs = {"input": "test input"}

        # Mock the trace context method
        with (
            patch.object(
                handler, "_get_or_create_trace_context", return_value=mock_trace_context
            ) as mock_context,
            patch.object(handler, "_resolve_parent_span_id", return_value=None),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.extract_langgraph_metadata",
                return_value={},
            ),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.get_operation_name",
                return_value="chain.test_chain",
            ),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.build_langgraph_attributes",
                return_value={},
            ),
        ):

            # Mock span creation
            mock_span = Mock()
            handler._client.start_span.return_value = mock_span

            # Call method
            handler.on_chain_start(
                serialized=serialized,
                inputs=inputs,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )

        # Should have called _get_or_create_trace_context with new signature
        mock_context.assert_called_once()
        call_args = mock_context.call_args[0]
        assert len(call_args) == 3
        assert call_args[1] == run_id
        assert call_args[2] == parent_run_id

    def test_on_tool_start_with_new_signature(self, handler, mock_trace_context):
        """Test on_tool_start uses new _get_or_create_trace_context signature."""
        run_id = uuid4()
        parent_run_id = uuid4()
        serialized = {"name": "test_tool"}
        input_str = "test input"

        # Mock the trace context method
        with (
            patch.object(
                handler, "_get_or_create_trace_context", return_value=mock_trace_context
            ) as mock_context,
            patch.object(handler, "_resolve_parent_span_id", return_value=None),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.extract_tool_function_name",
                return_value="test_func",
            ),
        ):

            # Mock span creation
            mock_span = Mock()
            handler._client.start_span.return_value = mock_span

            # Call method
            handler.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )

        # Should have called _get_or_create_trace_context with new signature
        mock_context.assert_called_once()
        call_args = mock_context.call_args[0]
        assert len(call_args) == 3
        assert call_args[1] == run_id
        assert call_args[2] == parent_run_id

    def test_on_agent_start_with_new_signature(self, handler, mock_trace_context):
        """Test on_agent_start uses new _get_or_create_trace_context signature."""
        run_id = uuid4()
        parent_run_id = uuid4()
        serialized = {"name": "test_agent"}
        inputs = {"input": "test input"}

        # Mock the trace context method
        with (
            patch.object(
                handler, "_get_or_create_trace_context", return_value=mock_trace_context
            ) as mock_context,
            patch.object(handler, "_resolve_parent_span_id", return_value=None),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.get_operation_name",
                return_value="agent.test_agent",
            ),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.extract_agent_type",
                return_value="test_type",
            ),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.extract_agent_capabilities",
                return_value=["test_cap"],
            ),
        ):

            # Mock span creation
            mock_span = Mock()
            handler._client.start_span.return_value = mock_span

            # Call method
            handler.on_agent_start(
                serialized=serialized,
                inputs=inputs,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )

        # Should have called _get_or_create_trace_context with new signature
        mock_context.assert_called_once()
        call_args = mock_context.call_args[0]
        assert len(call_args) == 3
        assert call_args[1] == run_id
        assert call_args[2] == parent_run_id

    def test_on_retriever_start_with_new_signature(self, handler, mock_trace_context):
        """Test on_retriever_start uses new _get_or_create_trace_context signature."""
        run_id = uuid4()
        parent_run_id = uuid4()
        serialized = {"name": "test_retriever"}
        query = "test query"

        # Mock the trace context method
        with (
            patch.object(
                handler, "_get_or_create_trace_context", return_value=mock_trace_context
            ) as mock_context,
            patch.object(handler, "_resolve_parent_span_id", return_value=None),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.get_operation_name",
                return_value="retrieval.test_retriever",
            ),
        ):

            # Mock span creation
            mock_span = Mock()
            handler._client.start_span.return_value = mock_span

            # Call method
            handler.on_retriever_start(
                serialized=serialized,
                query=query,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )

        # Should have called _get_or_create_trace_context with new signature
        mock_context.assert_called_once()
        call_args = mock_context.call_args[0]
        assert len(call_args) == 3
        assert call_args[1] == run_id
        assert call_args[2] == parent_run_id


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestSafeInputsConversion:
    """Test callback methods use safe_inputs_to_dict for input conversion."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_chain_start_uses_safe_inputs_conversion(self, handler):
        """Test on_chain_start handles inputs properly without using safe_inputs_to_dict."""
        run_id = uuid4()
        serialized = {"name": "test_chain"}

        # Test with different input types
        test_inputs = [
            {"key": "value"},  # dict
            ["item1", "item2"],  # list
            ("item1", "item2"),  # tuple
            "single_string",  # primitive
        ]

        for inputs in test_inputs:
            with (
                patch.object(
                    handler, "_get_or_create_trace_context", return_value=(Mock(), True)
                ),
                patch.object(handler, "_resolve_parent_span_id", return_value=None),
                patch(
                    "noveum_trace.integrations.langchain.langchain_utils.extract_langgraph_metadata",
                    return_value={},
                ),
                patch(
                    "noveum_trace.integrations.langchain.langchain_utils.get_operation_name",
                    return_value="chain.test",
                ),
                patch(
                    "noveum_trace.integrations.langchain.langchain_utils.build_langgraph_attributes",
                    return_value={},
                ),
            ):

                # Mock span creation
                mock_span = Mock()
                handler._client.start_span.return_value = mock_span

                # Call method - should not raise exception
                handler.on_chain_start(
                    serialized=serialized, inputs=inputs, run_id=run_id
                )

                # Verify span was created and stored correctly
                assert handler._get_run(run_id) == mock_span

                # Clean up for next iteration
                handler._pop_run(run_id)

    def test_chain_end_uses_safe_inputs_conversion(self, handler):
        """Test on_chain_end handles outputs properly without using safe_inputs_to_dict."""
        run_id = uuid4()

        # Test with different output types
        test_outputs = [
            {"result": "success"},  # dict
            ["result1", "result2"],  # list
            "single_result",  # primitive
        ]

        for outputs in test_outputs:
            # Set up existing span for each test
            mock_span = Mock()
            handler._set_run(run_id, mock_span)

            # Call method - should not raise exception
            handler.on_chain_end(outputs=outputs, run_id=run_id)

            # Verify span was handled correctly (attributes were set and finished)
            mock_span.set_attributes.assert_called_once()
            mock_span.set_status.assert_called_once()
            handler._client.finish_span.assert_called_with(mock_span)

            # Reset mocks for next iteration
            mock_span.reset_mock()
            handler._client.reset_mock()

    def test_agent_start_uses_safe_inputs_conversion(self, handler):
        """Test on_agent_start uses safe_inputs_to_dict for input conversion."""
        run_id = uuid4()
        serialized = {"name": "test_agent"}
        inputs = {"input": "test"}

        with (
            patch.object(
                handler, "_get_or_create_trace_context", return_value=(Mock(), True)
            ),
            patch.object(handler, "_resolve_parent_span_id", return_value=None),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.get_operation_name",
                return_value="agent.test",
            ),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.extract_agent_type",
                return_value="test_type",
            ),
            patch(
                "noveum_trace.integrations.langchain.langchain_utils.extract_agent_capabilities",
                return_value=[],
            ),
            patch(
                "noveum_trace.integrations.langchain.langchain.safe_inputs_to_dict"
            ) as mock_safe_convert,
        ):

            # Mock span creation
            mock_span = Mock()
            handler._client.start_span.return_value = mock_span
            mock_safe_convert.return_value = {"converted": "input"}

            # Call method
            handler.on_agent_start(serialized=serialized, inputs=inputs, run_id=run_id)

            # Should have called safe_inputs_to_dict
            mock_safe_convert.assert_called_once_with(inputs, "input")

    def test_agent_finish_uses_safe_inputs_conversion(self, handler):
        """Test on_agent_finish uses safe_inputs_to_dict for return values conversion."""
        run_id = uuid4()

        # Set up existing span
        mock_span = Mock()
        handler._set_run(run_id, mock_span)

        # Mock AgentFinish
        mock_finish = Mock()
        mock_finish.return_values = {"output": "result"}
        mock_finish.log = "test log"

        with (
            patch.object(handler, "_complete_tool_spans_from_finish"),
            patch(
                "noveum_trace.integrations.langchain.langchain.safe_inputs_to_dict"
            ) as mock_safe_convert,
        ):

            mock_safe_convert.return_value = {"converted": "return"}

            # Call method
            handler.on_agent_finish(finish=mock_finish, run_id=run_id)

            # Should have called safe_inputs_to_dict twice (for attributes and event)
            assert mock_safe_convert.call_count == 2
            for call in mock_safe_convert.call_args_list:
                assert call[0][0] == mock_finish.return_values
                assert call[0][1] == "return"

    def test_tool_start_enhanced_input_handling(self, handler):
        """Test on_tool_start has enhanced input handling for various types."""
        run_id = uuid4()
        serialized = {"name": "test_tool"}
        input_str = "test input"

        # Test with different inputs parameter types
        test_inputs_cases = [
            {"key": "value"},  # dict
            [{"item": "value"}],  # list of dicts
            ["item1", "item2"],  # list of primitives
            ("item1", "item2"),  # tuple
            "single_value",  # primitive
            None,  # None
        ]

        for inputs in test_inputs_cases:
            with (
                patch.object(
                    handler, "_get_or_create_trace_context", return_value=(Mock(), True)
                ),
                patch.object(handler, "_resolve_parent_span_id", return_value=None),
                patch(
                    "noveum_trace.integrations.langchain.langchain_utils.extract_tool_function_name",
                    return_value="test_func",
                ),
            ):

                # Mock span creation
                mock_span = Mock()
                handler._client.start_span.return_value = mock_span

                # Call method - should not raise exception
                handler.on_tool_start(
                    serialized=serialized,
                    input_str=input_str,
                    inputs=inputs,
                    run_id=run_id,
                )

                # Verify span was created and stored
                assert handler._get_run(run_id) == mock_span

                # Clean up for next iteration
                handler._pop_run(run_id)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestCallbackMethodIntegration:
    """Test callback methods integrate properly with root trace tracking."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_callback_methods_store_spans_correctly(self, handler):
        """Test that callback methods store spans in runs dict correctly."""
        run_id = uuid4()
        serialized = {"name": "test_llm"}
        prompts = ["test prompt"]

        with (
            patch.object(
                handler, "_get_or_create_trace_context", return_value=(Mock(), True)
            ),
            patch.object(handler, "_resolve_parent_span_id", return_value=None),
        ):

            # Mock span creation
            mock_span = Mock()
            handler._client.start_span.return_value = mock_span

            # Call method
            handler.on_llm_start(serialized=serialized, prompts=prompts, run_id=run_id)

            # Span should be stored in runs dict
            assert handler._get_run(run_id) == mock_span

    def test_callback_methods_handle_custom_names(self, handler):
        """Test that callback methods handle custom names from metadata."""
        run_id = uuid4()
        serialized = {"name": "test_llm"}
        prompts = ["test prompt"]
        metadata = {"noveum": {"name": "custom_llm_name"}}

        with (
            patch.object(
                handler, "_get_or_create_trace_context", return_value=(Mock(), True)
            ),
            patch.object(handler, "_resolve_parent_span_id", return_value=None),
        ):

            # Mock span creation
            mock_span = Mock()
            mock_span.span_id = "test_span_id"
            handler._client.start_span.return_value = mock_span

            # Call method
            handler.on_llm_start(
                serialized=serialized, prompts=prompts, metadata=metadata, run_id=run_id
            )

            # Should have stored custom name mapping
            assert handler._get_span_id_by_name("custom_llm_name") == "test_span_id"

    def test_callback_methods_manage_trace_lifecycle(self, handler):
        """Test that callback methods manage trace lifecycle correctly."""
        run_id = uuid4()
        serialized = {"name": "test_llm"}
        prompts = ["test prompt"]

        # Mock trace that should be managed
        mock_trace = Mock()
        mock_trace.trace_id = "managed_trace"

        with patch.object(
            handler, "_get_or_create_trace_context", return_value=(mock_trace, True)
        ):

            # Mock span creation
            mock_span = Mock()
            handler._client.start_span.return_value = mock_span

            # Call method
            handler.on_llm_start(serialized=serialized, prompts=prompts, run_id=run_id)

            # Should have stored trace for lifecycle management
            assert handler._trace_managed_by_langchain == mock_trace
