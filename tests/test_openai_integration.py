"""
Test OpenAI integration with mocked Noveum API backend.

This test verifies that the SDK correctly captures OpenAI API calls
and sends trace data to the Noveum backend.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import noveum_trace


class MockNoveumAPI:
    """Mock Noveum API server for testing."""

    def __init__(self):
        self.received_traces: list[dict[str, Any]] = []
        self.received_batches: list[list[dict[str, Any]]] = []

    def receive_trace(self, trace_data: dict[str, Any]) -> dict[str, Any]:
        """Mock endpoint for receiving individual traces."""
        self.received_traces.append(trace_data)
        return {"status": "success", "trace_id": trace_data.get("trace_id")}

    def receive_batch(self, batch_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Mock endpoint for receiving trace batches."""
        self.received_batches.append(batch_data)
        self.received_traces.extend(batch_data)
        return {"status": "success", "processed": len(batch_data)}

    def get_received_traces(self) -> list[dict[str, Any]]:
        """Get all received traces."""
        return self.received_traces

    def clear(self):
        """Clear all received data."""
        self.received_traces.clear()
        self.received_batches.clear()


@pytest.fixture
def mock_noveum_api():
    """Fixture providing a mock Noveum API."""
    return MockNoveumAPI()


@pytest.fixture
def mock_openai_response():
    """Fixture providing a mock OpenAI API response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Hello! How can I help you today?", role="assistant"
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=8, total_tokens=18
    )
    mock_response.model = "gpt-3.5-turbo"
    mock_response.id = "chatcmpl-test123"
    return mock_response


class TestOpenAIIntegration:
    """Test OpenAI integration with Noveum Trace SDK."""

    def setup_method(self):
        """Setup for each test method."""
        # Reset SDK state
        noveum_trace._client = None
        noveum_trace._config = None

    def teardown_method(self):
        """Cleanup after each test method."""
        # Shutdown client if initialized
        if noveum_trace.is_initialized():
            client = noveum_trace.get_client()
            client.shutdown()

        # Reset SDK state
        noveum_trace._client = None
        noveum_trace._config = None

    def test_openai_chat_completion_tracing(
        self, mock_noveum_api, mock_openai_response
    ):
        """Test that OpenAI chat completions are properly traced."""

        # Mock the batch processor to capture sent data
        with patch(
            "noveum_trace.transport.batch_processor.BatchProcessor.add_trace"
        ) as mock_add_trace:
            # Store traces that are added to the batch processor
            captured_traces = []

            def capture_trace(trace_data):
                captured_traces.append(trace_data)
                mock_noveum_api.receive_trace(trace_data)

            mock_add_trace.side_effect = capture_trace

            # Mock OpenAI API call
            with patch(
                "openai.resources.chat.completions.Completions.create"
            ) as mock_openai:
                mock_openai.return_value = mock_openai_response

                # Initialize SDK
                noveum_trace.init(api_key="test_key", project="test_openai_integration")

                # Import OpenAI after SDK initialization
                import openai

                # Create traced OpenAI function
                @noveum_trace.trace_llm
                def call_openai(prompt: str) -> str:
                    client = openai.OpenAI(api_key="test_openai_key")
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content

                # Execute the traced function
                result = call_openai("Hello, how are you?")

                # Verify the result
                assert result == "Hello! How can I help you today?"

                # Verify OpenAI was called
                mock_openai.assert_called_once()
                call_args = mock_openai.call_args
                assert call_args[1]["model"] == "gpt-3.5-turbo"
                assert call_args[1]["messages"][0]["content"] == "Hello, how are you?"

                # Verify trace was captured
                assert (
                    len(captured_traces) > 0
                ), "No traces captured. Expected at least 1."

                # Find the trace for our function
                trace_data = captured_traces[0]  # Should be the auto-created trace

                assert (
                    trace_data is not None
                ), f"Expected trace not found. Captured: {captured_traces}"

                # Verify trace structure
                assert "trace_id" in trace_data
                assert "spans" in trace_data
                assert len(trace_data["spans"]) > 0

                # Find the LLM span
                llm_span = None
                for span in trace_data["spans"]:
                    if span.get("attributes", {}).get("function.type") == "llm_call":
                        llm_span = span
                        break

                assert llm_span is not None, "LLM span not found"

                # Verify LLM span attributes
                attributes = llm_span.get("attributes", {})
                assert attributes.get("function.name") == "call_openai"
                assert attributes.get("function.type") == "llm_call"
                assert attributes.get("llm.operation_type") == "completion"

    def test_openai_integration_with_manual_trace(
        self, mock_noveum_api, mock_openai_response
    ):
        """Test OpenAI integration within a manually created trace."""

        # Mock the batch processor to capture sent data
        with patch(
            "noveum_trace.transport.batch_processor.BatchProcessor.add_trace"
        ) as mock_add_trace:
            captured_traces = []

            def capture_trace(trace_data):
                captured_traces.append(trace_data)
                mock_noveum_api.receive_trace(trace_data)

            mock_add_trace.side_effect = capture_trace

            # Mock OpenAI API call
            with patch(
                "openai.resources.chat.completions.Completions.create"
            ) as mock_openai:
                mock_openai.return_value = mock_openai_response

                # Initialize SDK
                noveum_trace.init(api_key="test_key", project="test_manual_trace")

                import openai

                # Create traced OpenAI function
                @noveum_trace.trace_llm
                def call_openai(prompt: str) -> str:
                    client = openai.OpenAI(api_key="test_openai_key")
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content

                # Create manual trace
                client = noveum_trace.get_client()
                trace = client.start_trace(
                    name="manual_openai_test", attributes={"test_type": "manual_trace"}
                )

                try:
                    # Execute the traced function within the manual trace
                    result = call_openai("What is AI?")

                    # Verify the result
                    assert result == "Hello! How can I help you today?"

                finally:
                    # Finish the trace
                    client.finish_trace(trace)

                # Verify trace was captured
                assert (
                    len(captured_traces) > 0
                ), "No traces captured. Expected at least 1."

                # Find our manual trace
                manual_trace = None
                for trace_data in captured_traces:
                    if trace_data.get("name") == "manual_openai_test":
                        manual_trace = trace_data
                        break

                assert manual_trace is not None, "Manual trace not found"

                # Verify trace has the LLM span
                assert "spans" in manual_trace
                assert len(manual_trace["spans"]) > 0

                # Verify attributes
                assert (
                    manual_trace.get("attributes", {}).get("test_type")
                    == "manual_trace"
                )

    def test_openai_integration_error_handling(self, mock_noveum_api):
        """Test error handling in OpenAI integration."""

        # Mock the batch processor to capture sent data
        with patch(
            "noveum_trace.transport.batch_processor.BatchProcessor.add_trace"
        ) as mock_add_trace:
            captured_traces = []

            def capture_trace(trace_data):
                captured_traces.append(trace_data)
                mock_noveum_api.receive_trace(trace_data)

            mock_add_trace.side_effect = capture_trace

            # Mock OpenAI API to raise an exception
            with patch(
                "openai.resources.chat.completions.Completions.create"
            ) as mock_openai:
                mock_openai.side_effect = Exception("OpenAI API Error")

                # Initialize SDK
                noveum_trace.init(api_key="test_key", project="test_error_handling")

                import openai

                # Create traced OpenAI function
                @noveum_trace.trace_llm
                def call_openai_with_error(prompt: str) -> str:
                    client = openai.OpenAI(api_key="test_openai_key")
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content

                # Execute the function and expect an exception
                with pytest.raises(Exception, match="OpenAI API Error"):
                    call_openai_with_error("This will fail")

                # Verify error trace was captured
                assert (
                    len(captured_traces) > 0
                ), "No traces captured. Expected at least 1."

                # Find the error trace
                error_trace = captured_traces[0]
                assert error_trace is not None, "Error trace not found"

                # Verify error span
                error_span = None
                for span in error_trace.get("spans", []):
                    span_status = span.get("status")
                    if span_status == "error":
                        error_span = span
                        break

                assert (
                    error_span is not None
                ), f"Error span not found. Spans: {error_trace.get('spans', [])}"

                # Verify error details
                assert error_span.get("status_message") == "OpenAI API Error"
                assert (
                    error_span.get("attributes", {}).get("exception.type")
                    == "Exception"
                )

    def test_batch_processing(self, mock_noveum_api, mock_openai_response):
        """Test that multiple traces are batched correctly."""

        # Mock the batch processor to capture sent data
        with patch(
            "noveum_trace.transport.batch_processor.BatchProcessor.add_trace"
        ) as mock_add_trace:
            captured_traces = []

            def capture_trace(trace_data):
                captured_traces.append(trace_data)
                mock_noveum_api.receive_trace(trace_data)

            mock_add_trace.side_effect = capture_trace

            # Mock OpenAI API call
            with patch(
                "openai.resources.chat.completions.Completions.create"
            ) as mock_openai:
                mock_openai.return_value = mock_openai_response

                # Initialize SDK
                noveum_trace.init(api_key="test_key", project="test_batching")

                import openai

                @noveum_trace.trace_llm
                def call_openai(prompt: str) -> str:
                    client = openai.OpenAI(api_key="test_openai_key")
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content

                # Execute multiple calls to trigger batching
                results = []
                for i in range(3):
                    result = call_openai(f"Test prompt {i}")
                    results.append(result)

                # Verify all calls succeeded
                assert len(results) == 3
                assert all(r == "Hello! How can I help you today?" for r in results)

                # Verify multiple traces were captured
                assert (
                    len(captured_traces) >= 3
                ), f"Expected at least 3 traces, got {len(captured_traces)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
