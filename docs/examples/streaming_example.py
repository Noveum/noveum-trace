"""
Streaming Example for Noveum Trace SDK.

This example demonstrates how to trace streaming LLM responses,
where tokens are received incrementally rather than all at once.
"""

import os
import time
from collections.abc import Iterator
from typing import Any

# Load environment variables (install python-dotenv if needed)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "python-dotenv not installed. Environment variables will be read from system only."
    )
    pass

import noveum_trace
from noveum_trace import (
    create_openai_streaming_callback,
    streaming_llm,
    trace_streaming,
)

# Initialize the SDK
noveum_trace.init(
    project="streaming-example",
    api_key=os.getenv("NOVEUM_API_KEY"),
    environment="development",
)


# Mock OpenAI streaming response for demonstration
class MockStreamingResponse:
    """Mock streaming response that simulates token-by-token generation."""

    def __init__(self, content: str, delay: float = 0.05):
        self.content = content
        self.delay = delay
        self.tokens = content.split()
        self.index = 0

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self.index >= len(self.tokens):
            raise StopIteration

        token = self.tokens[self.index]
        self.index += 1

        # Simulate processing delay
        time.sleep(self.delay)

        # Create a mock OpenAI-like response object
        return MockOpenAIChunk(token)


class MockOpenAIChunk:
    """Mock OpenAI streaming chunk."""

    def __init__(self, token: str):
        self.choices = [MockChoice(token)]


class MockChoice:
    """Mock OpenAI streaming choice."""

    def __init__(self, token: str):
        self.delta = MockDelta(token)


class MockDelta:
    """Mock OpenAI streaming delta."""

    def __init__(self, token: str):
        self.content = token


def mock_openai_streaming_call(prompt: str) -> MockStreamingResponse:
    """Simulate an OpenAI streaming API call."""
    response_text = f"This is a simulated response to: {prompt}. It generates tokens one by one with a slight delay to demonstrate streaming tracing capabilities."
    return MockStreamingResponse(response_text)


# Example 1: Using the trace_streaming wrapper
def example_trace_streaming_wrapper():
    """Demonstrate using the trace_streaming wrapper."""
    print("\n=== Example 1: Using trace_streaming wrapper ===")

    # Make a streaming API call
    stream = mock_openai_streaming_call("Tell me about quantum computing")

    # Wrap the stream with tracing
    traced_stream = trace_streaming(
        stream_iterator=stream, model="gpt-4", provider="openai", operation="completion"
    )

    # Process the stream normally
    print("Response: ", end="")
    for chunk in traced_stream:
        token = chunk.choices[0].delta.content
        print(token, end=" ")
    print("\n")


# Example 2: Using the streaming_llm context manager
def example_streaming_context_manager():
    """Demonstrate using the streaming_llm context manager."""
    print("\n=== Example 2: Using streaming_llm context manager ===")

    # Make a streaming API call
    stream = mock_openai_streaming_call("Explain machine learning")

    # Use the context manager for manual token tracking
    with streaming_llm(model="gpt-4", provider="openai") as stream_manager:
        print("Response: ", end="")

        # Process the stream manually
        for chunk in stream:
            token = chunk.choices[0].delta.content

            # Add the token to the trace
            stream_manager.add_token(token)

            # Display the token
            print(token, end=" ")

    print("\n")


# Example 3: Using the OpenAI-specific callback
def example_openai_callback():
    """Demonstrate using the OpenAI-specific streaming callback."""
    print("\n=== Example 3: Using OpenAI-specific callback ===")

    # Create the callback
    trace_stream = create_openai_streaming_callback(
        model="gpt-4", attributes={"prompt": "Describe neural networks"}
    )

    # Make a streaming API call
    stream = mock_openai_streaming_call("Describe neural networks")

    # Apply the callback to the stream
    traced_stream = trace_stream(stream)

    # Process the stream normally
    print("Response: ", end="")
    for chunk in traced_stream:
        token = chunk.choices[0].delta.content
        print(token, end=" ")
    print("\n")


# Example 4: Integrating with real OpenAI (commented out)
def example_real_openai():
    """
    Demonstrate integration with real OpenAI.

    Note: This example is commented out as it requires an OpenAI API key.
    Uncomment and add your API key to run with the actual OpenAI API.
    """
    print("\n=== Example 4: Real OpenAI integration (simulated) ===")

    # Simulate the OpenAI integration
    print("To use with real OpenAI:")
    print(
        """
    from openai import OpenAI

    # Initialize OpenAI client
    client = OpenAI(api_key="your-api-key")

    # Create the streaming callback
    trace_stream = create_openai_streaming_callback("gpt-4")

    # Make a streaming API call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Write a poem about AI"}],
        stream=True
    )

    # Apply tracing to the stream
    traced_response = trace_stream(response)

    # Process the stream normally
    for chunk in traced_response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="")
    """
    )
    print("\n")


if __name__ == "__main__":
    # Run all examples
    example_trace_streaming_wrapper()
    example_streaming_context_manager()
    example_openai_callback()
    example_real_openai()

    # Flush traces before exiting
    noveum_trace.flush()
