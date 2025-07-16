#!/usr/bin/env python3
"""
Test Anthropic integration with Noveum Trace.
"""
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables
from dotenv import load_dotenv

import noveum_trace

load_dotenv()


def test_anthropic_basic():
    """Test basic Anthropic functionality."""
    print("🔍 Testing Anthropic Basic Integration...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        import anthropic

        # Test with Anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": "Say hello and explain what you do in one sentence.",
                }
            ],
        )

        print(f"✅ Anthropic Response: {response.content[0].text}")
        print(f"✅ Model: {response.model}")
        print(f"✅ Usage: {response.usage}")

    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


def test_anthropic_with_system_prompt():
    """Test Anthropic with system prompt."""
    print("\n🎯 Testing Anthropic with System Prompt...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            system="You are a helpful assistant that responds in a friendly, conversational tone.",
            messages=[
                {
                    "role": "user",
                    "content": "Explain quantum computing in simple terms.",
                }
            ],
        )

        print(f"✅ Anthropic System Prompt Response: {response.content[0].text}")
        print(f"✅ Stop reason: {response.stop_reason}")

    except Exception as e:
        print(f"❌ Anthropic system prompt test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


def test_anthropic_multimodal():
    """Test Anthropic with multimodal input (text only for now)."""
    print("\n🖼️ Testing Anthropic Multimodal (text-only)...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe a beautiful sunset scene."}
                    ],
                }
            ],
        )

        print(f"✅ Anthropic Multimodal Response: {response.content[0].text}")

    except Exception as e:
        print(f"❌ Anthropic multimodal test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


def test_anthropic_different_models():
    """Test different Anthropic models."""
    print("\n🤖 Testing Different Anthropic Models...")

    models = [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
    ]

    for model in models:
        print(f"\n  Testing {model}...")

        # Initialize tracer
        noveum_trace.init(
            project_id="test_project", file_logging=True, log_directory="test_traces"
        )

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

            response = client.messages.create(
                model=model,
                max_tokens=50,
                messages=[{"role": "user", "content": "Say hello in a creative way."}],
            )

            print(f"    ✅ {model}: {response.content[0].text[:100]}...")

        except Exception as e:
            print(f"    ❌ {model} failed: {e}")
            return False

        finally:
            noveum_trace.shutdown()

    return True


if __name__ == "__main__":
    print("🚀 Starting Anthropic Integration Tests...")

    results = []
    results.append(test_anthropic_basic())
    results.append(test_anthropic_with_system_prompt())
    results.append(test_anthropic_multimodal())
    results.append(test_anthropic_different_models())

    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")
    if all(results):
        print("🎉 All Anthropic tests passed!")
    else:
        print("❌ Some Anthropic tests failed")
        sys.exit(1)
