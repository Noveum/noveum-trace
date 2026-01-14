"""
Comprehensive Tool Calling Guide - Noveum Trace Integration

This guide covers EVERYTHING about tool calling and auto-detection:

TRADITIONAL LANGCHAIN AGENTS (AgentExecutor-based):
├── Pattern 1: OpenAI Functions Agent ✅ Auto-detect works
├── Pattern 2: ReAct Agent (AgentExecutor) ✅ Auto-detect works
└── Pattern 3: Manual Metadata Injection ✅ Most reliable

LANGGRAPH AGENTS (Modern patterns):
├── Pattern 4: Custom StateGraph with bind_tools() ✅ Auto-detect works
├── Pattern 5: langgraph.prebuilt.create_react_agent ✅ Auto-detect works
└── Pattern 6: LangGraph with ToolNode ⚠️ Manual injection recommended

TECHNICAL DEEP-DIVE:
- How the callback handler detects tools
- Where tools are stored in LangChain's data structures
- Why manual tool.invoke() doesn't work
- Production best practices

================================================================================
HOW THE CALLBACK HANDLER DETECTS TOOLS
================================================================================

When tools are bound to an LLM, they flow through LangChain's callback system:

1. llm.bind_tools([tool1, tool2])
   └─> tools stored in invocation_params
       └─> on_llm_start() receives kwargs['invocation_params']['tools']
           └─> callback handler extracts and stores tools

2. create_react_agent(llm, [tool1, tool2])
   └─> tools stored in agent kwargs
       └─> on_agent_start() receives serialized['kwargs']['tools']
           └─> callback handler extracts and stores tools

3. AgentExecutor(agent=agent, tools=[tool1, tool2])
   └─> tools stored in executor kwargs
       └─> on_agent_start() receives serialized['kwargs']['tools']
           └─> callback handler extracts and stores tools

4. metadata['noveum']['available_tools'] = [tool1, tool2]
   └─> manual injection (always works, most reliable)

The callback handler adds these attributes to spans:
- agent.available_tools.count
- agent.available_tools.names
- agent.available_tools[i].name
- agent.available_tools[i].description
- agent.available_tools[i].args_schema

================================================================================
WHY MANUAL tool.invoke() DOESN'T WORK
================================================================================

❌ WRONG:
```python
# Tools called directly as Python functions
result = web_search.invoke({"query": "test"})
```

Problem: The LLM never knows these tools exist!
- No tools in invocation_params
- No tools in serialized data
- Callback handler has nothing to detect

✅ CORRECT:
```python
# Tools bound to LLM
llm_with_tools = llm.bind_tools([web_search])
response = llm_with_tools.invoke(messages)
```

Solution: The LLM is aware of tools, callback handler can detect them!

================================================================================
"""

import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

# Import from langchain_classic for traditional AgentExecutor patterns
# (LangChain 1.x moved old agents to langchain_classic package)
try:
    from langchain_classic.agents import (
        AgentExecutor,
        create_openai_functions_agent,
    )
    from langchain_classic.prompts import (
        ChatPromptTemplate,
        MessagesPlaceholder,
    )
except ImportError:
    # Fallback for older LangChain versions (0.x)
    from langchain.agents import (
        AgentExecutor,
        create_openai_functions_agent,
    )
    from langchain.prompts import (
        ChatPromptTemplate,
        MessagesPlaceholder,
    )

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

import noveum_trace
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

load_dotenv()


# =============================================================================
# SETUP & TOOLS DEFINITION
# =============================================================================


def setup_noveum_trace():
    """Configure Noveum Trace (run once at startup)."""
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "test-project"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
    )


# Simple tools with @tool decorator
@tool
def web_search(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: The search query string

    Returns:
        Search results as a string
    """
    simulated_results = {
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "python": "Python is a high-level, interpreted programming language.",
    }

    query_lower = query.lower()
    for key, result in simulated_results.items():
        if key in query_lower:
            return f"Search results for '{query}': {result}"

    return f"Search results for '{query}': General information about the topic."


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2")

    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression)  # noqa: S307 - example only
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


# Custom tool with Pydantic schema
class WeatherInput(BaseModel):
    """Input schema for weather tool."""

    location: str = Field(description="City name or location")
    unit: str = Field(
        default="celsius", description="Temperature unit (celsius or fahrenheit)"
    )


class WeatherTool(BaseTool):
    """Custom weather tool with Pydantic schema."""

    name: str = "weather_tool"
    description: str = "Get current weather for a location"
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, location: str, unit: str = "celsius") -> str:
        """Get weather for a location."""
        return f"Weather in {location}: 72°{unit[0].upper()}, Sunny ☀️"

    async def _arun(self, location: str, unit: str = "celsius") -> str:
        """Async version."""
        return self._run(location, unit)


# List of all available tools
TOOLS = [web_search, calculator, WeatherTool()]


# =============================================================================
# PATTERN 1: OpenAI Functions Agent
# ✅ Auto-detect works - tools in AgentExecutor constructor
# =============================================================================


def pattern1_openai_functions_agent():
    """
    Pattern 1: OpenAI Functions Agent with AgentExecutor.

    Auto-detection: ✅ WORKS
    How it works:
    - Tools passed as invocation_params['functions'] (OpenAI format)
    - on_llm_start() extracts functions automatically
    - Functions are converted to tool schema format
    - Works in both LangChain 0.x and 1.x
    """
    print("\n" + "=" * 80)
    print("PATTERN 1: OpenAI Functions Agent")
    print("=" * 80)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant with access to tools."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create OpenAI Functions agent
    agent = create_openai_functions_agent(llm, TOOLS, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=TOOLS, verbose=True, max_iterations=3
    )

    # Initialize callback handler
    handler = NoveumTraceCallbackHandler()

    # Execute agent - tools will be auto-detected from invocation_params['functions']
    print("\n🤖 Running agent with question: 'What's the weather in Paris?'")
    result = agent_executor.invoke(
        {"input": "What's the weather in Paris?"},
        config={
            "callbacks": [handler],
            "metadata": {"noveum": {"name": "pattern1_openai_functions"}},
        },
    )

    print(f"\n✅ Result: {result['output']}")
    print("\n📊 Trace Info:")
    print("   • Tools auto-detected from invocation_params['functions']")
    print("   • Check LLM spans: llm.available_tools.count = 3")
    print(
        "   • Check LLM spans: llm.available_tools.names = ['web_search', 'calculator', 'weather_tool']"
    )


# =============================================================================
# PATTERN 2: Manual Metadata Injection (RECOMMENDED FOR PRODUCTION)
# ✅ Most reliable - explicit and always works
# =============================================================================


def pattern2_manual_injection():
    """
    Pattern 2: Manual Metadata Injection.

    Auto-detection: N/A - We're being explicit
    Reliability: ✅ MOST RELIABLE (RECOMMENDED)
    Why:
    - Explicit is better than implicit
    - Works with any agent type
    - No dependency on LangChain internals
    - Production-ready approach
    """
    print("\n" + "=" * 80)
    print("PATTERN 2: Manual Metadata Injection (RECOMMENDED)")
    print("=" * 80)

    # Initialize LLM and agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_functions_agent(llm, TOOLS, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=TOOLS, verbose=True, max_iterations=3
    )

    # Initialize callback handler
    handler = NoveumTraceCallbackHandler()

    # Execute with explicit tool injection
    print("\n🤖 Running agent with question: 'Search for Python tutorials'")
    result = agent_executor.invoke(
        {"input": "Search for Python tutorials"},
        config={
            "callbacks": [handler],
            "metadata": {
                "noveum": {
                    "available_tools": TOOLS,  # ✅ Explicit injection
                    "name": "pattern2_manual_injection",
                }
            },
        },
    )

    print(f"\n✅ Result: {result['output']}")
    print("\n📊 Trace Info:")
    print("   • Tools explicitly injected via metadata.noveum.available_tools")
    print("   • MOST RELIABLE approach - works with ANY agent pattern")
    print("   • Recommended for production use")


# =============================================================================
# PATTERN 3: LangGraph Custom StateGraph with bind_tools()
# ✅ Auto-detect works - tools bound to LLM
# =============================================================================


def pattern3_langgraph_custom_stategraph():
    """
    Pattern 3: LangGraph Custom StateGraph with llm.bind_tools().

    Auto-detection: ✅ WORKS
    How it works:
    - Tools bound to LLM via bind_tools()
    - Tools stored in kwargs['invocation_params']['tools']
    - on_llm_start() extracts tools automatically
    - Full control over agent loop
    """
    print("\n" + "=" * 80)
    print("PATTERN 3: LangGraph Custom StateGraph with bind_tools()")
    print("=" * 80)

    try:
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode

        # Define state
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            iteration_count: int
            max_iterations: int

        def agent_node(state: AgentState) -> AgentState:
            """Agent node - LLM with tools bound."""
            print(f"\n  🤖 Agent iteration {state['iteration_count'] + 1}")

            # Create LLM with tools BOUND
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            llm_with_tools = llm.bind_tools([web_search, calculator])

            # Invoke LLM - it decides which tools to call
            response = llm_with_tools.invoke(state["messages"])

            state["iteration_count"] += 1
            return {"messages": [response], "iteration_count": state["iteration_count"]}

        def should_continue(state: AgentState) -> Literal["tools", "end"]:
            """Decide whether to continue with tool calls or end."""
            last_message = state["messages"][-1]

            if state["iteration_count"] >= state["max_iterations"]:
                print("  ⚠️  Max iterations reached")
                return "end"

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(
                    f"  🔧 Tool calls requested: {[tc['name'] for tc in last_message.tool_calls]}"
                )
                return "tools"

            print("  ✅ No more tool calls needed")
            return "end"

        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode([web_search, calculator]))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        app = workflow.compile()

        # Initialize callback handler
        handler = NoveumTraceCallbackHandler()

        # Execute
        print("\n🤖 Running LangGraph agent with question: 'What is 25 * 4?'")
        result = app.invoke(
            {
                "messages": [HumanMessage(content="What is 25 * 4?")],
                "iteration_count": 0,
                "max_iterations": 5,
            },
            config={
                "callbacks": [handler],
                "metadata": {"noveum": {"name": "pattern3_langgraph_stategraph"}},
            },
        )

        final_message = result["messages"][-1]
        print(
            f"\n✅ Result: {final_message.content if hasattr(final_message, 'content') else final_message}"
        )
        print("\n📊 Trace Info:")
        print("   • Tools auto-detected from llm.bind_tools()")
        print(
            "   • Extracted via kwargs['invocation_params']['tools'] in on_llm_start()"
        )
        print("   • Full custom agent loop with StateGraph")

    except ImportError:
        print("\n⚠️  LangGraph not installed - skipping this pattern")
        print("   Install with: pip install langgraph")


# =============================================================================
# PATTERN 4: langgraph.prebuilt.create_react_agent
# ✅ Auto-detect works - tools bound internally
# =============================================================================


def pattern4_langgraph_react_agent():
    """
    Pattern 4: LangGraph Prebuilt ReAct Agent.

    Auto-detection: ✅ WORKS
    How it works:
    - create_react_agent() binds tools internally
    - Tools stored in agent configuration
    - on_agent_start() or on_llm_start() extracts tools
    - Standard ReAct reasoning pattern
    """
    print("\n" + "=" * 80)
    print("PATTERN 4: langgraph.prebuilt.create_react_agent")
    print("=" * 80)

    try:
        from langgraph.prebuilt import create_react_agent

        # Create LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Create ReAct agent - tools automatically bound!
        agent = create_react_agent(llm, [web_search, calculator])

        # Initialize callback handler
        handler = NoveumTraceCallbackHandler()

        # Execute
        print("\n🤖 Running ReAct agent with question: 'Search for LangGraph'")
        result = agent.invoke(
            {"messages": [HumanMessage(content="Search for LangGraph")]},
            config={
                "callbacks": [handler],
                "metadata": {"noveum": {"name": "pattern4_langgraph_react"}},
            },
        )

        final_messages = [
            msg
            for msg in result["messages"]
            if isinstance(msg, AIMessage) and msg.content
        ]
        if final_messages:
            print(f"\n✅ Result: {final_messages[-1].content}")

        print("\n📊 Trace Info:")
        print("   • Tools auto-detected from create_react_agent()")
        print("   • Prebuilt agent handles tool binding internally")
        print("   • Easiest LangGraph pattern for tool calling")

    except ImportError:
        print("\n⚠️  LangGraph not installed - skipping this pattern")
        print("   Install with: pip install langgraph")


# =============================================================================
# PATTERN 5: Multi-Node Graph with Different Tools Per Node
# ✅ Demonstrates manual injection for complex workflows
# =============================================================================


def pattern5_multinode_different_tools():
    """
    Pattern 5: Multi-Node Graph with Different Tools Per Node.

    Use case: Different agent nodes with specialized tool sets
    - Researcher node: web_search only
    - Calculator node: calculator only
    - Weather node: weather_tool only
    - Router: decides which specialist to call

    Tool tracking: Manual injection per node (recommended for multi-node setups)
    """
    print("\n" + "=" * 80)
    print("PATTERN 5: Multi-Node Graph with Different Tools Per Node")
    print("=" * 80)

    print("\n💡 Use case: Specialized agents with different tool sets")
    print("   • Researcher: web_search")
    print("   • Calculator: calculator")
    print("   • Weather: weather_tool")

    # Initialize components
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    handler = NoveumTraceCallbackHandler()

    # Get individual tools from TOOLS list
    weather_tool = TOOLS[2]  # WeatherTool instance

    # State definition
    class MultiNodeState(TypedDict):
        input: str
        current_specialist: str  # Which specialist to use next
        results: list[str]  # Collect results from multiple specialists
        visited_specialists: list[str]  # Track which specialists we've used

    # Specialist nodes - each with ONE specific tool
    def researcher_node(state: MultiNodeState) -> MultiNodeState:
        """Research specialist - only has web_search."""
        print(f"\n🔍 Researcher handling: '{state['input']}'")
        llm_with_tools = llm.bind_tools([web_search])

        # Invoke with manual tool injection
        response = llm_with_tools.invoke(
            f"Use web_search to answer: {state['input']}",
            config={
                "callbacks": [handler],
                "metadata": {
                    "noveum": {
                        "name": "researcher_node",
                        "available_tools": [web_search],  # ✅ Only web_search
                    }
                },
            },
        )

        new_results = state.get("results", []) + [f"Researcher: {response.content}"]
        new_visited = state.get("visited_specialists", []) + ["researcher"]
        return {
            "results": new_results,
            "visited_specialists": new_visited,
            "current_specialist": "continue",  # Signal to continue routing
        }

    def calculator_node(state: MultiNodeState) -> MultiNodeState:
        """Calculator specialist - only has calculator."""
        print(f"\n🔢 Calculator handling: '{state['input']}'")
        llm_with_tools = llm.bind_tools([calculator])

        response = llm_with_tools.invoke(
            f"Use calculator to solve: {state['input']}",
            config={
                "callbacks": [handler],
                "metadata": {
                    "noveum": {
                        "name": "calculator_node",
                        "available_tools": [calculator],  # ✅ Only calculator
                    }
                },
            },
        )

        new_results = state.get("results", []) + [f"Calculator: {response.content}"]
        new_visited = state.get("visited_specialists", []) + ["calculator"]
        return {
            "results": new_results,
            "visited_specialists": new_visited,
            "current_specialist": "continue",
        }

    def weather_node(state: MultiNodeState) -> MultiNodeState:
        """Weather specialist - only has weather tool."""
        print(f"\n🌤️  Weather handling: '{state['input']}'")
        llm_with_tools = llm.bind_tools([weather_tool])

        response = llm_with_tools.invoke(
            f"Use weather tool to answer: {state['input']}",
            config={
                "callbacks": [handler],
                "metadata": {
                    "noveum": {
                        "name": "weather_node",
                        # ✅ Only weather_tool
                        "available_tools": [weather_tool],
                    }
                },
            },
        )

        new_results = state.get("results", []) + [f"Weather: {response.content}"]
        new_visited = state.get("visited_specialists", []) + ["weather"]
        return {
            "results": new_results,
            "visited_specialists": new_visited,
            "current_specialist": "continue",
        }

    def router_node(state: MultiNodeState) -> MultiNodeState:
        """Routes to specialists based on query - can route to MULTIPLE specialists."""
        query = state["input"].lower()
        visited = state.get("visited_specialists", [])

        # Determine which specialists we need
        needs_weather = "weather" in query or "temperature" in query
        needs_calc = any(
            word in query
            for word in ["calculate", "multiply", "add", "subtract", "*", "+", "math"]
        )
        needs_research = "search" in query or "find" in query or "python" in query

        # Find next specialist to visit
        if needs_calc and "calculator" not in visited:
            next_specialist = "calculator"
        elif needs_weather and "weather" not in visited:
            next_specialist = "weather"
        elif needs_research and "researcher" not in visited:
            next_specialist = "researcher"
        else:
            # If we've visited all needed specialists, or no match, default to researcher once
            if not visited:
                next_specialist = "researcher"
            else:
                next_specialist = "done"

        print(f"\n🎯 Router: Directing to '{next_specialist}' (visited: {visited})")
        return {"current_specialist": next_specialist}

    def should_continue(state: MultiNodeState) -> str:
        """Decide whether to continue routing or end."""
        specialist = state.get("current_specialist", "done")
        if specialist == "done" or specialist == "continue":
            return "done"
        return specialist

    # Build the graph
    workflow = StateGraph(MultiNodeState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("weather", weather_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional edges from router to specialists
    workflow.add_conditional_edges(
        "router",
        should_continue,
        {
            "researcher": "researcher",
            "calculator": "calculator",
            "weather": "weather",
            "done": END,
        },
    )

    # All specialists loop back to router for potential next step
    workflow.add_edge("researcher", "router")
    workflow.add_edge("calculator", "router")
    workflow.add_edge("weather", "router")

    # Compile
    app = workflow.compile()

    # Test with queries that require MULTIPLE specialists
    test_queries = [
        "Calculate 15 * 27 and search for Python tutorials",  # 2 specialists
        "What's the weather in Paris and calculate 100 + 50?",  # 2 specialists
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = app.invoke(
            {"input": query, "results": [], "visited_specialists": []},
            config={
                "callbacks": [handler],
                "metadata": {"noveum": {"name": "pattern5_multinode"}},
            },
        )

        print(f"\n✅ Visited specialists: {result['visited_specialists']}")
        for res in result["results"]:
            print(f"   • {res}")

    print("\n📊 Trace Info:")
    print("   • Each specialist node has DIFFERENT tools")
    print("   • Single query hits MULTIPLE specialists (2+ nodes per trace)")
    print("   • Tools tracked per-node via metadata")
    print("   • Router loops back for multi-step workflows")
    print("   • Check dashboard: Each specialist shows only its specific tool!")
    print("   • Perfect for microservice-style agent architectures!")


# =============================================================================
# MAIN: Run all patterns
# =============================================================================


def main():
    """Run all tool calling pattern examples."""
    print("=" * 80)
    print("COMPREHENSIVE TOOL CALLING GUIDE - NOVEUM TRACE INTEGRATION")
    print("=" * 80)
    print("\nThis guide demonstrates 5 patterns for tool calling with auto-detection.")
    print("\nLegend:")
    print("  ✅ = Auto-detection works reliably")
    print("  🎯 = Recommended for production")
    print("  🔧 = Manual injection (for complex setups)")

    # Setup Noveum Trace
    setup_noveum_trace()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set. Examples will fail.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    try:
        # Traditional LangChain Agents
        print("\n\n" + "=" * 80)
        print("SECTION 1: TRADITIONAL LANGCHAIN AGENTS")
        print("=" * 80)

        pattern1_openai_functions_agent()  # ✅ Auto-detect
        pattern2_manual_injection()  # 🎯 RECOMMENDED

        # LangGraph Agents
        print("\n\n" + "=" * 80)
        print("SECTION 2: LANGGRAPH AGENTS")
        print("=" * 80)

        pattern3_langgraph_custom_stategraph()  # ✅ Auto-detect
        pattern4_langgraph_react_agent()  # ✅ Auto-detect
        pattern5_multinode_different_tools()  # 🔧 Manual per-node

        # Summary
        print("\n\n" + "=" * 80)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 80)

        print("\n🎯 PRODUCTION RECOMMENDATION: Pattern 2 - Manual Injection")
        print("   • Most reliable across all agent types")
        print("   • Explicit and predictable")
        print("   • No dependency on LangChain internals")
        print("   • Code: metadata={'noveum': {'available_tools': TOOLS}}")

        print("\n✅ AUTO-DETECTION WORKS:")
        print("   • Pattern 1: OpenAI Functions Agent (invocation_params['functions'])")
        print("   • Pattern 3: LangGraph Custom StateGraph (llm.bind_tools)")
        print("   • Pattern 4: LangGraph create_react_agent (llm.bind_tools)")

        print("\n🔧 MANUAL INJECTION EXAMPLES:")
        print("   • Pattern 2: Always-reliable manual injection")
        print("   • Pattern 5: Multi-node with different tools per node")

        print("\n📚 HOW IT WORKS:")
        print(
            "   1. llm.bind_tools() → tools in invocation_params['tools'] → on_llm_start() detects"
        )
        print(
            "   2. OpenAI Functions → invocation_params['functions'] → on_llm_start() detects"
        )
        print("   3. metadata.noveum.available_tools → always works (priority 1)")

        print("\n❌ WHAT DOESN'T WORK:")
        print("   • ReAct agents - tools are text in prompt, not in invocation_params")
        print("   • Direct tool.invoke() - LLM never knows about tools")
        print("   • Tools defined but not bound to LLM")

        print("\n💡 WHEN TO USE MANUAL INJECTION:")
        print("   • Multi-node graphs with different tools per node")
        print("   • ReAct-style text-based tool prompting")
        print("   • When you need guaranteed reliability")
        print("   • Custom agent architectures")

        print("\n🔍 CHECK YOUR TRACES FOR:")
        print("   • agent.available_tools.count")
        print("   • agent.available_tools.names")
        print("   • agent.available_tools[i].description")
        print("   • agent.available_tools[i].args_schema")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
