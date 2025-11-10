"""
Test script to verify call site tracking for LangChain integration.

This script tests that call site information (file, line, function) is correctly
captured for LLM and Tool calls.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

import noveum_trace
from noveum_trace.integrations import NoveumTraceCallbackHandler

load_dotenv()


def test_llm_call_site_tracking():
    """Test that LLM calls capture call site information."""
    print("=== Testing LLM Call Site Tracking ===")

    try:
        from langchain_openai import ChatOpenAI

        # Initialize Noveum Trace with configuration from environment
        init_config = {
            "project": os.getenv("NOVEUM_PROJECT", "call-site-test"),
            "api_key": os.getenv("NOVEUM_API_KEY"),
            "environment": os.getenv("NOVEUM_ENVIRONMENT", "dev-aman"),
            "transport_config": {"batch_size": 1, "batch_timeout": 5.0},
        }
        
        # Add endpoint only if explicitly set
        noveum_endpoint = os.getenv("NOVEUM_ENDPOINT")
        if noveum_endpoint:
            init_config["endpoint"] = noveum_endpoint
        
        noveum_trace.init(**init_config)

        # Create callback handler
        callback_handler = NoveumTraceCallbackHandler()

        # Create LLM with callback
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.7, callbacks=[callback_handler]
        )

        # Make LLM call - this should capture call site info
        print("Making LLM call from test_llm_call_site_tracking()...")
        response = llm.invoke("What is 2+2?")
        print(f"Response: {response.content}")

        # Check if call site info was captured
        # We'll need to inspect the spans to verify
        print("\n✓ LLM call completed - check spans for call_site attributes")

    except ImportError:
        print("Skipping LLM test - langchain-openai not installed")
    except Exception as e:
        print(f"Error in LLM test: {e}")
        import traceback

        traceback.print_exc()


def test_tool_call_site_tracking():
    """Test that Tool calls capture call site information."""
    print("\n=== Testing Tool Call Site Tracking ===")

    try:
        from langchain.agents import AgentType, initialize_agent
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI

        # Create callback handler
        callback_handler = NoveumTraceCallbackHandler()

        # Define custom tool
        def calculator(expression: str) -> str:
            """Simple calculator tool."""
            try:
                result = eval(expression)
                return f"The result is: {result}"
            except Exception as e:
                return f"Error: {str(e)}"

        # Create tools
        tools = [
            Tool(
                name="Calculator",
                func=calculator,
                description="Use this to perform mathematical calculations",
            )
        ]

        # Create LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0, callbacks=[callback_handler]
        )

        # Create agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            callbacks=[callback_handler],
            verbose=True,
        )

        # Use agent with tools - this should capture call site info
        print("Making tool call from test_tool_call_site_tracking()...")
        result = agent.run("Calculate 15 * 23")
        print(f"Agent result: {result}")

        print("\n✓ Tool call completed - check spans for call_site attributes")

    except ImportError:
        print("Skipping tool test - required packages not installed")
    except Exception as e:
        print(f"Error in tool test: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all tests."""
    print("Call Site Tracking Test")
    print("=" * 50)

    # Display configuration
    print("\nConfiguration:")
    print(f"  NOVEUM_PROJECT: {os.getenv('NOVEUM_PROJECT', 'call-site-test')}")
    print(f"  NOVEUM_API_KEY: {'***' if os.getenv('NOVEUM_API_KEY') else 'NOT SET'}")
    print(f"  NOVEUM_ENDPOINT: {os.getenv('NOVEUM_ENDPOINT', 'default (not set)')}")
    print(f"  NOVEUM_ENVIRONMENT: {os.getenv('NOVEUM_ENVIRONMENT', 'development')}")
    print(f"  OPENAI_API_KEY: {'***' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")

    # Check if API keys are set
    if not os.getenv("NOVEUM_API_KEY"):
        print("\n⚠️  Warning: NOVEUM_API_KEY not set. Using mock mode.")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some tests may fail.")

    print()

    # Run tests
    test_llm_call_site_tracking()
    test_tool_call_site_tracking()

    print("\n=== Tests Complete ===")
    print("Check your Noveum dashboard to see the traced operations with call site info!")
    print("\nExpected attributes in spans:")
    print("  - call_site.file")
    print("  - call_site.line")
    print("  - call_site.function")
    print("  - call_site.module")
    print("  - call_site.code_context")

    # Flush any pending traces
    noveum_trace.flush()


if __name__ == "__main__":
    main()

