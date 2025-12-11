"""
LangChain Custom Parent Span Example with Noveum Trace

This example demonstrates how to use metadata.noveum to:
1. Assign custom names to spans
2. Explicitly set parent-child relationships between spans

The example shows two unrelated LLM calls where the second call
is explicitly made a child of the first using custom naming.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from noveum_trace import init as noveum_init
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

# Load environment variables
load_dotenv()


def main():
    """
    Example showing custom span names and explicit parent relationships.
    """
    print("=" * 80)
    print("🎯 LANGCHAIN CUSTOM PARENT SPAN EXAMPLE")
    print("=" * 80)

    # Initialize Noveum Trace
    noveum_init(
        project=os.getenv("NOVEUM_PROJECT", "test-project"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
    )
    print("✅ Noveum Trace initialized")

    # Create callback handler
    handler = NoveumTraceCallbackHandler()
    print("✅ Callback handler created")

    # Manually start a trace
    # This prevents auto-finishing so both calls stay in the same trace
    handler.start_trace("weather_and_recipe_trace")
    print("✅ Trace started manually")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    print("\n" + "=" * 80)
    print("📞 MAKING FIRST LLM CALL (Parent)")
    print("=" * 80)

    # First LLM call - this will be the parent
    # We assign it a custom name "weather_query"
    config_parent = {
        "callbacks": [handler],
        "metadata": {"noveum": {"name": "weather_query"}},  # Custom name for this span
    }

    response1 = llm.invoke(
        [HumanMessage(content="What's the weather like in Tokyo?")],
        config=config_parent,
    )

    print("✅ First call completed")
    print(f"Response: {response1.content[:100]}...")

    print("\n" + "=" * 80)
    print("📞 MAKING SECOND LLM CALL (Child)")
    print("=" * 80)

    # Second LLM call - completely unrelated topic
    # But we explicitly make it a child of the first call using parent_name
    config_child = {
        "callbacks": [handler],
        "metadata": {
            "noveum": {
                "name": "recipe_query",  # Custom name for this span
                "parent_name": "weather_query",  # Reference to parent span
            }
        },
    }

    response2 = llm.invoke(
        [HumanMessage(content="Give me a recipe for chocolate chip cookies")],
        config=config_child,
    )

    print("✅ Second call completed")
    print(f"Response: {response2.content[:100]}...")

    # Manually end the trace
    handler.end_trace()
    print("\n✅ Trace ended manually")

    print("\n" + "=" * 80)
    print("✅ EXAMPLE COMPLETED")
    print("=" * 80)

    print("\n📊 Trace Structure:")
    print("  Trace: weather_and_recipe_trace")
    print("  └── Span: weather_query (First LLM call)")
    print("      └── Span: recipe_query (Second LLM call)")
    print()
    print("💡 Key Points:")
    print("   1. handler.start_trace() creates a trace and disables auto-finishing")
    print("   2. Both LLM calls stay within the same manually controlled trace")
    print(
        "   3. metadata.noveum.parent_name creates explicit parent-child relationship"
    )
    print("   4. handler.end_trace() manually finishes the trace")
    print()
    print(f"Handler state: {handler}")


if __name__ == "__main__":
    main()
