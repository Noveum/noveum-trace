"""
Message Accumulation Demo - Demonstrates Long Message Chains Issue

This example shows how LangGraph agents accumulate messages in state,
leading to very long stringified message chains in traces.

The issue:
- Each iteration adds new messages to the state
- Messages list grows: HumanMessage → AIMessage → ToolMessage → AIMessage → ...
- When traced, the entire messages list is converted to string
- After 5-10 iterations, traces become huge and hard to read

This demo deliberately creates a multi-iteration agent to show the problem.
"""

import json
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

import noveum_trace
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================


def setup_noveum_trace():
    """Initialize Noveum Trace."""
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "message-accumulation-demo"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
    )


# =============================================================================
# STATE DEFINITION - Notice the add_messages reducer
# =============================================================================


class AgentState(TypedDict):
    """
    Agent state with message accumulation.

    The add_messages reducer means messages ACCUMULATE - they never get removed!
    This is what causes the long chains.
    """

    # Messages accumulate here with add_messages reducer
    messages: Annotated[list, add_messages]

    # Iteration counter
    iteration: int

    # Maximum iterations before stopping
    max_iterations: int

    # Task completion flag
    task_complete: bool


# =============================================================================
# FAKE TOOLS - Simulate tool calls with varying output sizes
# =============================================================================


@tool
def search_database(query: str) -> str:
    """
    Search a database. Returns verbose results to increase message size.

    Args:
        query: The search query

    Returns:
        Search results with detailed information
    """
    # Simulate verbose search results
    results = {
        "query": query,
        "results": [
            {
                "id": i,
                "title": f"Result {i} for '{query}'",
                "content": f"This is detailed content for result {i}. " * 5,
                "metadata": {
                    "score": 0.95 - (i * 0.1),
                    "source": f"source_{i}.pdf",
                    "page": i + 1,
                },
            }
            for i in range(3)
        ],
        "total_results": 3,
        "search_time_ms": 125,
    }
    return json.dumps(results, indent=2)


@tool
def analyze_data(data: str) -> str:
    """
    Analyze data. Returns detailed analysis to increase message size.

    Args:
        data: The data to analyze

    Returns:
        Detailed analysis results
    """
    analysis = {
        "input_length": len(data),
        "analysis": {
            "sentiment": "positive",
            "key_topics": ["topic_1", "topic_2", "topic_3"],
            "summary": "This is a detailed summary of the analysis. " * 10,
            "statistics": {
                "word_count": 150,
                "sentence_count": 12,
                "average_word_length": 5.2,
            },
            "recommendations": [
                f"Recommendation {i}: Do something important. " * 3 for i in range(3)
            ],
        },
        "confidence": 0.89,
    }
    return json.dumps(analysis, indent=2)


@tool
def fetch_context(context_id: str) -> str:
    """
    Fetch additional context. Returns large context to increase message size.

    Args:
        context_id: The context identifier

    Returns:
        Context information
    """
    context = {
        "context_id": context_id,
        "description": "This is very detailed context information. " * 20,
        "related_items": [f"item_{i}" for i in range(10)],
        "metadata": {"timestamp": "2026-01-14T10:00:00Z", "version": "1.0.0"},
    }
    return json.dumps(context, indent=2)


# =============================================================================
# AGENT NODES
# =============================================================================


def agent_node(state: AgentState) -> AgentState:
    """
    Agent reasoning node - decides what to do next.

    This node will call tools multiple times, adding messages to the state.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([search_database, analyze_data, fetch_context])

    # Add system message on first iteration
    messages = state["messages"]
    if state["iteration"] == 0:
        messages = [
            SystemMessage(
                content="You are a research assistant. Break down tasks into multiple steps. "
                "Use tools to gather information. After 3-4 tool calls, provide a final answer."
            )
        ] + messages

    print(
        f"\n{'='*80}\n"
        f"🤖 ITERATION {state['iteration'] + 1}\n"
        f"   Current message count: {len(state['messages'])}\n"
        f"   Total characters in messages: {sum(len(str(m)) for m in state['messages'])}\n"
        f"{'='*80}"
    )

    # Call LLM
    response = llm_with_tools.invoke(messages)

    print(f"   AI Response: {response.content[:100]}...")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"   Tool calls requested: {[tc['name'] for tc in response.tool_calls]}")

    return {
        "messages": [response],
        "iteration": state["iteration"] + 1,
    }


def tool_node(state: AgentState) -> AgentState:
    """
    Execute tool calls and return results.

    This adds ToolMessage objects to the state for each tool call.
    """
    from langgraph.prebuilt import ToolNode

    tools = [search_database, analyze_data, fetch_context]
    tool_executor = ToolNode(tools)

    # Get the last message (should be AIMessage with tool calls)
    last_message = state["messages"][-1]

    print("\n🔧 EXECUTING TOOLS:")
    if hasattr(last_message, "tool_calls"):
        for tc in last_message.tool_calls:
            print(f"   - {tc['name']}")

    # Execute tools - this will add ToolMessage objects to messages
    result = tool_executor.invoke(state)

    print(
        f"   Tool results added: {len([m for m in result['messages'] if isinstance(m, ToolMessage)])} ToolMessages"
    )

    return result


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide whether to continue or end."""
    last_message = state["messages"][-1]

    # Check max iterations
    if state["iteration"] >= state["max_iterations"]:
        print(f"\n⚠️  Max iterations ({state['max_iterations']}) reached - ENDING")
        return "end"

    # Check if there are tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    print("\n✅ No more tool calls - ENDING")
    return "end"


# =============================================================================
# BUILD GRAPH
# =============================================================================


def create_agent() -> StateGraph:
    """Create the agent graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def run_demo(max_iterations: int = 5):
    """
    Run the message accumulation demo.

    Args:
        max_iterations: How many iterations to run (more = longer message chains)
    """
    print("\n" + "=" * 80)
    print("📊 MESSAGE ACCUMULATION DEMO")
    print("=" * 80)
    print("\nThis demo shows how messages accumulate in LangGraph state,")
    print("leading to very long stringified message chains in traces.")
    print("=" * 80)

    # Setup
    setup_noveum_trace()
    handler = NoveumTraceCallbackHandler()
    app = create_agent()

    # Initial state - START WITH 5 HUMAN MESSAGES
    initial_state: AgentState = {
        "messages": [
            HumanMessage(
                content="I need you to research the following topic: "
                "'Machine learning model deployment best practices'. "
                "Please search the database, analyze the findings, "
                "fetch additional context, and provide a comprehensive answer."
            ),
            HumanMessage(
                content="Also, please make sure to cover CI/CD pipelines for ML models "
                "and how they differ from traditional software deployment."
            ),
            HumanMessage(
                content="I'm particularly interested in model versioning strategies, "
                "A/B testing frameworks, and monitoring approaches for deployed models."
            ),
            HumanMessage(
                content="Don't forget to include information about containerization (Docker/Kubernetes) "
                "and serverless deployment options for ML models."
            ),
            HumanMessage(
                content="Finally, please address scalability concerns, cost optimization, "
                "and best practices for model retraining and updates in production."
            ),
        ],
        "iteration": 0,
        "max_iterations": max_iterations,
        "task_complete": False,
    }

    print("\n📋 Task: ML deployment best practices research")
    print(f"📊 Max Iterations: {max_iterations}")
    print(f"📝 Starting with: {len(initial_state['messages'])} HumanMessages")
    print("\n💡 Watch how messages accumulate with each iteration!")
    print("=" * 80)

    # Execute
    config = {
        "callbacks": [handler],
        "metadata": {"demo": "message_accumulation"},
        "recursion_limit": 50,
    }

    try:
        final_state = app.invoke(initial_state, config=config)

        # Print statistics
        print("\n" + "=" * 80)
        print("📈 FINAL STATISTICS")
        print("=" * 80)
        print(f"Total iterations: {final_state['iteration']}")
        print(f"Total messages in state: {len(final_state['messages'])}")

        # Count message types
        message_types = {}
        total_chars = 0
        for msg in final_state["messages"]:
            msg_type = type(msg).__name__
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
            total_chars += len(str(msg))

        print("\nMessage breakdown:")
        for msg_type, count in sorted(message_types.items()):
            print(f"  - {msg_type}: {count}")

        print(f"\nTotal characters in all messages: {total_chars:,}")
        print(
            f"Average characters per message: {total_chars // len(final_state['messages']):,}"
        )

        # Show the problem
        print("\n" + "=" * 80)
        print("🚨 THE PROBLEM:")
        print("=" * 80)
        print(
            "When this state is passed to on_chain_start(), the callback handler does:"
        )
        print("  attributes['chain.inputs'] = {k: str(v) for k, v in inputs.items()}")
        print("\nThis means the 'messages' key becomes:")
        print("  '[HumanMessage(...), AIMessage(...), ToolMessage(...), ...]'")
        print(f"\nWith {total_chars:,} characters, this makes traces:")
        print("  ❌ Very large (slow to load)")
        print("  ❌ Hard to read (wall of text)")
        print("  ❌ Expensive to store")
        print("  ❌ Difficult to parse")

        print("\n" + "=" * 80)
        print("💡 SOLUTION IDEAS:")
        print("=" * 80)
        print("1. Truncate message content after N characters")
        print("2. Only store message count + types instead of full content")
        print("3. Store first/last N messages only")
        print("4. Optionally exclude 'messages' key from chain.inputs")
        print("5. Compress messages with summarization")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    # Get max iterations from command line or use default
    max_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"\nRunning with max_iterations={max_iterations}")
    print(f"(Try: python {os.path.basename(__file__)} 10 for more iterations)\n")

    run_demo(max_iterations=max_iterations)

    print("\n\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)
    print("\nCheck your Noveum Trace dashboard to see:")
    print("  • How large the 'chain.inputs.messages' attribute is")
    print("  • How the attribute size grows with each node execution")
    print("  • The full stringified message list in the trace")
    print("\nThis demonstrates the exact issue you're experiencing!")
    print("=" * 80)
