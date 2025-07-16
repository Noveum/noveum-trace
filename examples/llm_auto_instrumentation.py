"""
Comprehensive example demonstrating auto-instrumentation with OpenAI and Anthropic SDKs.
"""

import os
import time

from noveum_trace import NoveumTracer, TracerConfig
from noveum_trace.core.tracer import set_current_tracer
from noveum_trace.instrumentation import anthropic, openai
from noveum_trace.sinks.file import FileSink, FileSinkConfig


def main():
    """Main function demonstrating LLM auto-instrumentation."""
    print("Noveum Trace SDK - LLM Auto-Instrumentation Example")
    print("=" * 60)

    # Create file sink for logging traces
    file_sink = FileSink(
        FileSinkConfig(
            directory="./llm_traces",
            file_format="jsonl",
            max_file_size_mb=10,
            include_timestamp=True,
        )
    )

    # Configure tracer
    config = TracerConfig(
        service_name="llm-auto-instrumentation-demo",
        environment="development",
        sinks=[file_sink],
        batch_size=1,  # Process immediately for demo
        batch_timeout_ms=100,
        capture_llm_content=True,  # Capture LLM content for demo
    )

    # Create and set global tracer
    tracer = NoveumTracer(config)
    set_current_tracer(tracer)

    try:
        # Enable auto-instrumentation
        print("1. Enabling auto-instrumentation...")

        try:
            openai.instrument_openai()
            print("✅ OpenAI auto-instrumentation enabled")
        except Exception as e:
            print(f"⚠️  OpenAI instrumentation failed: {e}")

        try:
            anthropic.instrument_anthropic()
            print("✅ Anthropic auto-instrumentation enabled")
        except Exception as e:
            print(f"⚠️  Anthropic instrumentation failed: {e}")

        # Test OpenAI
        print("\n2. Testing OpenAI integration...")
        test_openai()

        # Test Anthropic
        print("\n3. Testing Anthropic integration...")
        test_anthropic()

        # Wait for processing
        print("\n4. Waiting for span processing...")
        time.sleep(1)

        # Flush spans
        print("5. Flushing spans...")
        tracer.flush(timeout_ms=5000)

        # Show file sink stats
        print("\n6. File sink statistics:")
        stats = file_sink.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("\n✅ Example completed successfully!")
        print(f"📁 Traces saved to: {os.path.abspath('./llm_traces')}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\n7. Cleaning up...")

        try:
            openai.uninstrument_openai()
            print("✅ OpenAI auto-instrumentation disabled")
        except Exception:
            pass

        try:
            anthropic.uninstrument_anthropic()
            print("✅ Anthropic auto-instrumentation disabled")
        except Exception:
            pass

        tracer.shutdown()
        print("✅ Tracer shutdown complete")


def test_openai():
    """Test OpenAI integration with auto-instrumentation."""
    try:
        import openai

        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set, skipping OpenAI test")
            return

        client = openai.OpenAI(api_key=api_key)

        print("   Making OpenAI chat completion request...")

        # This call will be automatically traced
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in a creative way!"},
            ],
            max_tokens=50,
            temperature=0.7,
        )

        print(f"   ✅ OpenAI Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"   ❌ OpenAI test failed: {e}")


def test_anthropic():
    """Test Anthropic integration with auto-instrumentation."""
    try:
        import anthropic

        # Check if API key is available
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  ANTHROPIC_API_KEY not set, skipping Anthropic test")
            return

        client = anthropic.Anthropic(api_key=api_key)

        print("   Making Anthropic message request...")

        # This call will be automatically traced
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hello in a creative way!"}],
        )

        content = response.content[0].text if response.content else "No content"
        print(f"   ✅ Anthropic Response: {content}")

    except Exception as e:
        print(f"   ❌ Anthropic test failed: {e}")


if __name__ == "__main__":
    main()
