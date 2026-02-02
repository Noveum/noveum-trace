"""
LangGraph Agent and LangChain Chain Multi-Tool Example

This example demonstrates:
1. A LangGraph agent that calls multiple tools (get_weather, calculate)
2. A LangChain chain that calls multiple tools (search_database, format_data)
3. Both examples use Noveum Trace for observability

The agent uses create_react_agent with tools bound, and the chain
demonstrates sequential tool usage in a workflow.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import noveum_trace
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

load_dotenv()

# =============================================================================
# TOOLS DEFINITION
# =============================================================================


@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a location.

    Args:
        location: The city or location name

    Returns:
        Weather information as a string
    """
    print(f"[TOOL] get_weather called with location: {location}")
    # Placeholder implementation
    weather_data = f"Sunny, 22°C in {location}"
    print(f"[TOOL] get_weather returning: {weather_data}")
    return weather_data


@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.

    Args:
        expression: Mathematical expression to evaluate (e.g., "25 * 4")

    Returns:
        Calculation result as a string
    """
    print(f"[TOOL] calculate called with expression: {expression}")
    try:
        result = eval(expression)
        result_str = f"Result: {result}"
        print(f"[TOOL] calculate returning: {result_str}")
        return result_str
    except Exception as e:
        error_str = f"Error: {str(e)}"
        print(f"[TOOL] calculate returning: {error_str}")
        return error_str


@tool
def search_database(query: str) -> str:
    """
    Search a database for information.

    Args:
        query: Search query string

    Returns:
        Search results as a string
    """
    print(f"[TOOL] search_database called with query: {query}")
    # Placeholder implementation
    results = f"Database search results for '{query}': Found 5 relevant records"
    print(f"[TOOL] search_database returning: {results}")
    return results


@tool
def format_data(data: str) -> str:
    """
    Format data into a structured output.

    Args:
        data: Raw data string to format

    Returns:
        Formatted data string
    """
    print(f"[TOOL] format_data called with data length: {len(data)} characters")
    # Placeholder implementation
    formatted = f"FORMATTED: {data.upper()}"
    print(f"[TOOL] format_data returning formatted data")
    return formatted


# =============================================================================
# LANGGRAPH AGENT EXAMPLE
# =============================================================================


def example_langgraph_agent():
    """Example: LangGraph agent calling multiple tools."""
    print("\n" + "=" * 80)
    print("LANGGRAPH AGENT - MULTIPLE TOOLS")
    print("=" * 80)

    try:
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages
        from langchain_core.messages import ToolMessage
        from typing import Annotated, Literal, TypedDict

        # Initialize Noveum Trace
        noveum_trace.init(
            project=os.getenv("NOVEUM_PROJECT", "test-project"),
            api_key=os.getenv("NOVEUM_API_KEY"),
            environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
        )

        # Create callback handler
        handler = NoveumTraceCallbackHandler()
        print("✅ Noveum Trace callback handler created")

        # Define state
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            iteration_count: int
            max_iterations: int

        # Create tool mapping
        tools_map = {
            "get_weather": get_weather,
            "calculate": calculate,
        }

        def agent_node(state: AgentState) -> AgentState:
            """Agent node - LLM with tools bound."""
            print(f"\n  🤖 Agent iteration {state['iteration_count'] + 1}")

            # Create LLM with tools BOUND
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[handler])
            llm_with_tools = llm.bind_tools([get_weather, calculate])

            # Invoke LLM - it decides which tools to call
            response = llm_with_tools.invoke(state["messages"])

            state["iteration_count"] += 1
            return {"messages": [response], "iteration_count": state["iteration_count"]}

        def tool_node(state: AgentState) -> AgentState:
            """Tool execution node - executes tool calls."""
            last_message = state["messages"][-1]
            tool_messages = []

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("args", {})
                    
                    if tool_name in tools_map:
                        print(f"  🔧 Executing tool: {tool_name}")
                        tool = tools_map[tool_name]
                        result = tool.invoke(tool_args)
                        tool_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call.get("id", ""),
                            )
                        )

            return {"messages": tool_messages}

        def should_continue(state: AgentState) -> Literal["tools", "end"]:
            """Decide whether to continue with tool calls or end."""
            last_message = state["messages"][-1]

            if state["iteration_count"] >= state["max_iterations"]:
                print("  ⚠️  Max iterations reached")
                return "end"

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                tool_names = [tc["name"] for tc in last_message.tool_calls]
                print(f"  🔧 Tool calls requested: {tool_names}")
                return "tools"

            print("  ✅ No more tool calls needed")
            return "end"

        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        app = workflow.compile()
        print("✅ LangGraph agent created with tools: get_weather, calculate")

        # Query that requires both tools
        query = (
            "What's the weather in Paris? Also calculate 25 * 4. "
            "Make sure to use both the weather tool and the calculator tool."
        )

        print(f"\n📋 Query: {query}")
        print("\n🚀 Executing agent...")

        # Execute agent with Noveum Trace callback handler
        result = app.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "iteration_count": 0,
                "max_iterations": 5,
            },
            config={
                "callbacks": [handler],
            },
        )

        # Extract final answer
        final_messages = [
            msg
            for msg in result["messages"]
            if hasattr(msg, "content") and msg.content
        ]
        if final_messages:
            print(f"\n✅ Agent Result:")
            print(f"   {final_messages[-1].content[:200]}...")

        print("\n📊 Trace Info:")
        print("   • Agent should have called both get_weather and calculate tools")
        print("   • Check Noveum dashboard for tool call spans")

    except ImportError as e:
        print(f"\n⚠️  Import error: {e}")
        print("   Make sure langgraph is installed: pip install langgraph")
    except Exception as e:
        print(f"\n❌ Error in LangGraph agent example: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# LANGCHAIN CHAIN EXAMPLE
# =============================================================================


def example_langchain_chain():
    """Example: LangChain chain calling multiple tools."""
    print("\n" + "=" * 80)
    print("LANGCHAIN CHAIN - MULTIPLE TOOLS")
    print("=" * 80)

    try:
        from langchain.agents import AgentType, initialize_agent

        # Initialize Noveum Trace
        noveum_trace.init(
            project=os.getenv("NOVEUM_PROJECT", "test-project"),
            api_key=os.getenv("NOVEUM_API_KEY"),
            environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
        )

        # Create callback handler
        handler = NoveumTraceCallbackHandler()
        print("✅ Noveum Trace callback handler created")

        # Create LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, callbacks=[handler]
        )

        # Chain Step 1: Search database using tool
        # Create an agent that uses search_database tool
        search_agent = initialize_agent(
            tools=[search_database],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            callbacks=[handler],
            verbose=False,
        )

        # Chain Step 2: Format data using tool
        # Create an agent that uses format_data tool
        format_agent = initialize_agent(
            tools=[format_data],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            callbacks=[handler],
            verbose=False,
        )

        print("✅ LangChain chain created with two tool-calling agents")
        print("   • Step 1: search_database tool")
        print("   • Step 2: format_data tool")

        query = "Python programming"

        print(f"\n📋 Query: {query}")
        print("\n🚀 Executing chain...")

        # Execute search step (calls search_database tool)
        print("\n   Step 1: Searching database...")
        search_result = search_agent.invoke(
            {"input": f"Use the search_database tool to search for: {query}"},
            config={"callbacks": [handler]},
        )
        search_output = search_result.get("output", str(search_result))
        print(f"   Search result: {search_output[:100]}...")

        # Execute format step (calls format_data tool)
        print("\n   Step 2: Formatting data...")
        formatted_result = format_agent.invoke(
            {"input": f"Use the format_data tool to format this data: {search_output}"},
            config={"callbacks": [handler]},
        )
        formatted_output = formatted_result.get("output", str(formatted_result))
        print(f"   Formatted result: {formatted_output[:100]}...")

        print("\n✅ Chain execution completed")
        print("\n📊 Trace Info:")
        print("   • Chain should have called both search_database and format_data tools")
        print("   • Check Noveum dashboard for tool call spans in chain execution")

    except ImportError as e:
        print(f"\n⚠️  Import error: {e}")
        print("   Make sure langchain is installed: pip install langchain")
    except Exception as e:
        print(f"\n❌ Error in LangChain chain example: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Run both examples."""
    print("=" * 80)
    print("LANGGRAPH AGENT AND LANGCHAIN CHAIN - MULTI-TOOL EXAMPLES")
    print("=" * 80)

    # Check if API keys are set
    if not os.getenv("NOVEUM_API_KEY"):
        print("Warning: NOVEUM_API_KEY not set. Using mock mode.")

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may fail.")

    # Run LangGraph agent example
    example_langgraph_agent()

    # Wait a bit between examples
    import time
    time.sleep(2)

    # Run LangChain chain example
    example_langchain_chain()

    print("\n" + "=" * 80)
    print("✅ ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nCheck your Noveum Trace dashboard to see:")
    print("  • LangGraph agent spans with tool calls (get_weather, calculate)")
    print("  • LangChain chain spans with tool calls (search_database, format_data)")
    print("  • Tool call spans nested under agent/chain spans")

    # Flush any pending traces
    noveum_trace.flush()


if __name__ == "__main__":
    main()
