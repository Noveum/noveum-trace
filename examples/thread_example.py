"""
Thread Example for Noveum Trace SDK.

This example demonstrates how to trace conversation threads,
allowing tracking of multi-turn conversations and thread context.
"""

import os
import time
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
from noveum_trace.threads import create_thread, trace_thread_llm

# Initialize the SDK
noveum_trace.init(
    project="thread-example",
    api_key=os.getenv("NOVEUM_API_KEY"),
    environment="development",
)


# Mock LLM call for demonstration
def mock_llm_call(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Simulate an LLM API call."""
    # Extract the last user message
    last_message = messages[-1]["content"] if messages else "Hello"

    # Generate a simple response based on the message
    if "weather" in last_message.lower():
        response = "The weather today is sunny with a high of 75°F."
    elif "name" in last_message.lower():
        response = "My name is Noveum Assistant. How can I help you today?"
    elif "help" in last_message.lower():
        response = (
            "I can help with a variety of tasks. What do you need assistance with?"
        )
    else:
        response = (
            f"I received your message: '{last_message}'. How can I assist you further?"
        )

    # Simulate processing delay
    time.sleep(0.5)

    # Return a mock response object
    return {
        "choices": [{"message": {"role": "assistant", "content": response}}],
        "usage": {
            "prompt_tokens": len(last_message.split()),
            "completion_tokens": len(response.split()),
            "total_tokens": len(last_message.split()) + len(response.split()),
        },
    }


# Example 1: Basic thread usage
def example_basic_thread():
    """Demonstrate basic thread usage."""
    print("\n=== Example 1: Basic Thread Usage ===")

    # Create a new thread
    thread = create_thread(name="Customer Support")

    # Use the thread to track conversation
    with thread:
        # Add initial user message
        print("User: Hello, I need help with something.")
        thread.add_message("Hello, I need help with something.", "user")

        # Get LLM response
        messages = thread.get_context()
        response = mock_llm_call(messages)

        # Add assistant message
        assistant_message = response["choices"][0]["message"]["content"]
        print(f"Assistant: {assistant_message}")
        thread.add_message(assistant_message, "assistant")

        # Continue the conversation
        print("User: What's your name?")
        thread.add_message("What's your name?", "user")

        # Get LLM response with updated context
        messages = thread.get_context()
        response = mock_llm_call(messages)

        # Add assistant message
        assistant_message = response["choices"][0]["message"]["content"]
        print(f"Assistant: {assistant_message}")
        thread.add_message(assistant_message, "assistant")

    print("\n")


# Example 2: Using trace_thread_llm for LLM calls
def example_trace_thread_llm():
    """Demonstrate using trace_thread_llm for LLM calls."""
    print("\n=== Example 2: Using trace_thread_llm ===")

    # Create a new thread
    thread = create_thread(name="Technical Support")

    # Use the thread to track conversation
    with thread:
        # Add initial user message
        print("User: How's the weather today?")
        thread.add_message("How's the weather today?", "user")

        # Trace LLM call in thread context
        with trace_thread_llm(thread, model="gpt-4", provider="openai") as span:
            # Get LLM response
            messages = thread.get_context()
            response = mock_llm_call(messages)

            # Add metrics to span
            span.set_attributes(
                {
                    "llm.input_tokens": response["usage"]["prompt_tokens"],
                    "llm.output_tokens": response["usage"]["completion_tokens"],
                    "llm.total_tokens": response["usage"]["total_tokens"],
                }
            )

            # Add assistant message
            assistant_message = response["choices"][0]["message"]["content"]
            print(f"Assistant: {assistant_message}")
            thread.add_message(assistant_message, "assistant")

        # Continue the conversation
        print("User: Can you help me with my account?")
        thread.add_message("Can you help me with my account?", "user")

        # Trace another LLM call
        with trace_thread_llm(thread, model="gpt-4", provider="openai") as span:
            # Get LLM response with updated context
            messages = thread.get_context()
            response = mock_llm_call(messages)

            # Add metrics to span
            span.set_attributes(
                {
                    "llm.input_tokens": response["usage"]["prompt_tokens"],
                    "llm.output_tokens": response["usage"]["completion_tokens"],
                    "llm.total_tokens": response["usage"]["total_tokens"],
                }
            )

            # Add assistant message
            assistant_message = response["choices"][0]["message"]["content"]
            print(f"Assistant: {assistant_message}")
            thread.add_message(assistant_message, "assistant")

    print("\n")


# Example 3: Managing multiple threads
def example_multiple_threads():
    """Demonstrate managing multiple threads."""
    print("\n=== Example 3: Managing Multiple Threads ===")

    # Create multiple threads
    thread1 = create_thread(name="Sales Inquiry")
    thread2 = create_thread(name="Technical Support")

    # Thread 1: Sales inquiry
    with thread1:
        print("--- Sales Inquiry Thread ---")

        # Add messages to thread 1
        print("User: I'm interested in your product pricing.")
        thread1.add_message("I'm interested in your product pricing.", "user")

        # Get LLM response
        with trace_thread_llm(thread1, model="gpt-4", provider="openai"):
            messages = thread1.get_context()
            response = mock_llm_call(messages)

            # Add assistant message
            assistant_message = response["choices"][0]["message"]["content"]
            print(f"Assistant: {assistant_message}")
            thread1.add_message(assistant_message, "assistant")

    # Thread 2: Technical support
    with thread2:
        print("\n--- Technical Support Thread ---")

        # Add messages to thread 2
        print("User: I'm having trouble logging in.")
        thread2.add_message("I'm having trouble logging in.", "user")

        # Get LLM response
        with trace_thread_llm(thread2, model="gpt-4", provider="openai"):
            messages = thread2.get_context()
            response = mock_llm_call(messages)

            # Add assistant message
            assistant_message = response["choices"][0]["message"]["content"]
            print(f"Assistant: {assistant_message}")
            thread2.add_message(assistant_message, "assistant")

    # Continue thread 1
    with thread1:
        print("\n--- Back to Sales Inquiry Thread ---")

        # Add more messages to thread 1
        print("User: Do you offer discounts for bulk orders?")
        thread1.add_message("Do you offer discounts for bulk orders?", "user")

        # Get LLM response with thread 1 context
        with trace_thread_llm(thread1, model="gpt-4", provider="openai"):
            messages = thread1.get_context()
            response = mock_llm_call(messages)

            # Add assistant message
            assistant_message = response["choices"][0]["message"]["content"]
            print(f"Assistant: {assistant_message}")
            thread1.add_message(assistant_message, "assistant")

    print("\n")


# Example 4: Thread metadata and filtering
def example_thread_metadata():
    """Demonstrate thread metadata and filtering."""
    print("\n=== Example 4: Thread Metadata and Filtering ===")

    # Create a thread with metadata
    thread = create_thread(
        name="Support Ticket #12345",
        metadata={
            "customer_id": "cust_123456",
            "priority": "high",
            "category": "billing",
            "ticket_id": "12345",
        },
    )

    # Use the thread
    with thread:
        # Add system message with context
        thread.add_message(
            "This is a high-priority billing inquiry from a premium customer.",
            "system",
            metadata={"visibility": "internal"},
        )

        # Add user message
        print("User: I was charged twice for my subscription.")
        thread.add_message(
            "I was charged twice for my subscription.",
            "user",
            metadata={"sentiment": "negative", "intent": "billing_issue"},
        )

        # Get LLM response
        with trace_thread_llm(
            thread,
            model="gpt-4",
            provider="openai",
            attributes={"priority": "high", "category": "billing"},
        ):
            messages = thread.get_context()
            response = mock_llm_call(messages)

            # Add assistant message
            assistant_message = response["choices"][0]["message"]["content"]
            print(f"Assistant: {assistant_message}")
            thread.add_message(
                assistant_message,
                "assistant",
                metadata={"response_type": "acknowledgment"},
            )

        # Get only user and assistant messages (exclude system)
        # Note: role_filter accepts only a single role, so we filter manually
        all_messages = thread.get_messages()
        user_assistant_messages = [
            msg for msg in all_messages if msg["role"] in ["user", "assistant"]
        ]
        print(f"\nUser and Assistant Messages: {len(user_assistant_messages)}")

        # Get the last 2 messages
        recent_messages = thread.get_messages(limit=2)
        print(f"Recent Messages: {len(recent_messages)}")

    print("\n")


if __name__ == "__main__":
    # Run all examples
    example_basic_thread()
    example_trace_thread_llm()
    example_multiple_threads()
    example_thread_metadata()

    # Flush traces before exiting
    noveum_trace.flush()
