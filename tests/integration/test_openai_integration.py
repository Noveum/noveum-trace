#!/usr/bin/env python3
"""
Test OpenAI integration with Noveum Trace.
"""
import os

import pytest

# Load environment variables
from dotenv import load_dotenv

import noveum_trace

load_dotenv()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_basic():
    """Test basic OpenAI functionality."""
    print("🔍 Testing OpenAI Basic Integration...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        import openai

        # Test with OpenAI
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Say hello and explain what you do in one sentence.",
                },
            ],
            temperature=0.7,
            max_tokens=100,
        )

        print(f"✅ OpenAI Response: {response.choices[0].message.content}")
        print(f"✅ Model: {response.model}")
        print(f"✅ Usage: {response.usage}")

    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_with_tools():
    """Test OpenAI with function calling."""
    print("\n🔧 Testing OpenAI with Tools...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        import openai

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Define a simple function
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            tools=tools,
            tool_choice="auto",
        )

        print(f"✅ OpenAI Tools Response: {response.choices[0].message}")
        if response.choices[0].message.tool_calls:
            print(f"✅ Tool calls: {response.choices[0].message.tool_calls}")

    except Exception as e:
        print(f"❌ OpenAI tools test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_streaming():
    """Test OpenAI streaming functionality."""
    print("\n🌊 Testing OpenAI Streaming...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        import openai

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Count from 1 to 5 slowly."}],
            stream=True,
        )

        print("✅ OpenAI Streaming Response:")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print()

    except Exception as e:
        print(f"❌ OpenAI streaming test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


if __name__ == "__main__":
    print("🚀 Starting OpenAI Integration Tests...")

    results = []
    results.append(test_openai_basic())
    results.append(test_openai_with_tools())
    results.append(test_openai_streaming())

    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")
    if all(results):
        print("🎉 All OpenAI tests passed!")
    else:
        print("❌ Some OpenAI tests failed")
        exit(1)
