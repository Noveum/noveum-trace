"""
LangChain Name Reuse Example with Noveum Trace

This example demonstrates:
1. Using custom span names with parent relationships
2. Reusing the same name for different spans (name collision)
3. How name mappings get overwritten

Flow:
- Call 1: name="math_question" (no parent)
- Call 2: name="geography_question", parent_name="math_question" (child of Call 1)
- Call 3: name="math_question", parent_name="math_question" (child of Call 1, REUSES name)

Note: Call 3 overwrites the "math_question" mapping, so after Call 3,
self.names["math_question"] points to Call 3's span_id, not Call 1's.
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
    Example showing name reuse and parent relationships.
    """
    print("=" * 80)
    print("🎯 LANGCHAIN NAME REUSE EXAMPLE")
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
    handler.start_trace("name_reuse_demo")
    print("✅ Trace started manually")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # ==========================================================================
    # CALL 1: name="math_question" (Root span)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("📞 CALL 1: name='math_question' (No parent)")
    print("=" * 80)

    config_1 = {
        "callbacks": [handler],
        "metadata": {"noveum": {"name": "math_question"}},
    }

    response1 = llm.invoke([HumanMessage(content="What is 2 + 2?")], config=config_1)

    print("✅ Call 1 completed")
    print("   Name: 'math_question'")
    print(f"   Response: {response1.content[:50]}...")
    print("   Mapping: self.names['math_question'] → span_id_1")

    # ==========================================================================
    # CALL 2: name="geography_question", parent_name="math_question" (Child of Call 1)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("📞 CALL 2: name='geography_question', parent_name='math_question'")
    print("=" * 80)

    config_2 = {
        "callbacks": [handler],
        "metadata": {
            "noveum": {
                "name": "geography_question",
                "parent_name": "math_question",  # References Call 1
            }
        },
    }

    response2 = llm.invoke(
        [HumanMessage(content="What is the capital of France?")], config=config_2
    )

    print("✅ Call 2 completed")
    print("   Name: 'geography_question'")
    print("   Parent: 'math_question' (Call 1)")
    print(f"   Response: {response2.content[:50]}...")
    print("   Mapping: self.names['geography_question'] → span_id_2")

    # ==========================================================================
    # CALL 3: name="math_question", parent_name="math_question" (REUSES name!)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("📞 CALL 3: name='math_question', parent_name='math_question' (REUSES name!)")
    print("=" * 80)

    config_3 = {
        "callbacks": [handler],
        "metadata": {
            "noveum": {
                "name": "math_question",  # SAME NAME as Call 1!
                "parent_name": "math_question",  # References Call 1 (before overwrite)
            }
        },
    }

    response3 = llm.invoke(
        [HumanMessage(content="What is 10 multiplied by 5?")], config=config_3
    )

    print("✅ Call 3 completed")
    print("   Name: 'math_question' (OVERWRITES Call 1's mapping)")
    print("   Parent: 'math_question' (Call 1 - looked up BEFORE overwrite)")
    print(f"   Response: {response3.content[:50]}...")
    print("   Mapping: self.names['math_question'] → span_id_3 (was span_id_1)")

    # Manually end the trace
    handler.end_trace()
    print("\n✅ Trace ended manually")

    print("\n" + "=" * 80)
    print("✅ EXAMPLE COMPLETED")
    print("=" * 80)

    print("\n📊 Final Trace Structure:")
    print("  Trace: name_reuse_demo")
    print("  └── Span: math_question (Call 1 - span_id_1)")
    print("      ├── Span: geography_question (Call 2 - span_id_2)")
    print("      └── Span: math_question (Call 3 - span_id_3) ← SAME NAME!")
    print()
    print("💡 Key Points:")
    print(
        "   1. Call 3 looks up parent_name='math_question' FIRST → finds Call 1's span_id"
    )
    print("   2. Call 3 creates span with Call 1 as parent ✓")
    print("   3. Call 3 THEN stores name='math_question' → overwrites mapping")
    print(
        "   4. After Call 3: self.names['math_question'] points to Call 3, not Call 1"
    )
    print()
    print("⚠️  Name Collision Behavior:")
    print("   - Lookups happen BEFORE storing new mappings")
    print("   - New spans with same name overwrite previous mappings")
    print("   - Original spans remain in trace with correct relationships")
    print("   - Only the name mapping changes, not the trace structure")
    print()
    print(f"Handler state: {handler}")


if __name__ == "__main__":
    main()
