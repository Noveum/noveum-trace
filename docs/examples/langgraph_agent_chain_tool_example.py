"""
LangGraph Agent with Chains and Tools Example

This example demonstrates a complete LangGraph agent system that achieves
the same functionality as the LangChain example but using LangGraph's
state-based approach with explicit graph structure.

Features:
1. State management for agent data flow
2. Node-based tool execution
3. LLM-based chains within nodes
4. Conditional routing based on agent decisions
5. Explicit graph structure vs LangChain's implicit flow

Use Case: A research assistant that can search, summarize, and analyze information.

Prerequisites:
    pip install noveum-trace[langchain]
    pip install langchain langchain-openai langgraph

Environment Variables:
    NOVEUM_API_KEY: Your Noveum API key
    OPENAI_API_KEY: Your OpenAI API key
"""

import os
import time
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

import noveum_trace
from noveum_trace.integrations import NoveumTraceCallbackHandler

load_dotenv()

# =============================================================================
# STATE DEFINITION
# =============================================================================


class AgentState(TypedDict):
    """State for the research agent."""
    
    # The original query from the user
    query: str
    
    # Conversation messages
    messages: Annotated[list, add_messages]
    
    # Results from tools
    search_results: str
    calculation_result: str
    analysis_result: str
    summary: str
    text_stats: dict[str, Any]
    
    # Agent's current plan/thought
    current_thought: str
    
    # Next action to take
    next_action: str
    
    # Final answer
    final_answer: str


# =============================================================================
# TOOLS (Same as LangChain example)
# =============================================================================


def web_search_tool(query: str) -> str:
    """
    Simulate a web search tool.
    """
    print(f"🔍 Searching for: {query}")
    time.sleep(0.5)
    
    search_results = {
        "langchain": """
        LangChain is a framework for developing applications powered by language models.
        Key components: Chains, Agents, Memory, Prompts, and Tools.
        Used for building chatbots, question-answering systems, and AI agents.
        """,
        "transformers": """
        Transformers are a type of neural network architecture introduced in 2017.
        Based on attention mechanisms, they revolutionized NLP.
        Popular models: BERT, GPT, T5, and their variants.
        """,
        "python": """
        Python is a high-level, interpreted programming language.
        Known for its simplicity and extensive ecosystem.
        Popular in data science, web development, and AI.
        """,
    }
    
    for key, content in search_results.items():
        if key.lower() in query.lower():
            return f"Search Results for '{query}':\n{content}"
    
    return f"Search Results for '{query}':\nGeneral information available."


def calculator_tool(expression: str) -> str:
    """
    Calculator tool for mathematical operations.
    """
    print(f"🧮 Calculating: {expression}")
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def text_analyzer_tool(text: str) -> dict[str, Any]:
    """
    Analyze text and return statistics.
    """
    print(f"📊 Analyzing text ({len(text)} chars)...")
    time.sleep(0.3)
    
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": sentences if sentences > 0 else 1,
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
    }


# =============================================================================
# AGENT NODES (LangGraph-specific)
# =============================================================================


def planning_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Planning node: Decide what to do based on the query.
    This replaces the ReAct agent's implicit planning.
    """
    print("\n🧠 PLANNING NODE")
    
    planning_prompt = f"""
    You are a research assistant. Analyze this query and decide what action to take.
    
    Query: {state['query']}
    
    Previous results:
    - Search: {state.get('search_results', 'None')}
    - Calculation: {state.get('calculation_result', 'None')}
    - Analysis: {state.get('analysis_result', 'None')}
    
    Available actions:
    - search: Search for information
    - calculate: Perform calculations
    - analyze: Analyze text
    - summarize: Summarize information
    - finish: Provide final answer
    
    Respond with ONLY ONE of these actions: search, calculate, analyze, summarize, finish
    """
    
    response = llm.invoke([HumanMessage(content=planning_prompt)])
    action = response.content.strip().lower()
    
    state["current_thought"] = f"Planning to: {action}"
    state["next_action"] = action
    state["messages"].append(AIMessage(content=f"Decided to: {action}"))
    
    print(f"Next action: {action}")
    
    return state


def search_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Execute web search tool.
    """
    print("\n🔍 SEARCH NODE")
    
    # Use LLM to extract search query from original query
    extract_prompt = f"""
    Extract the search query from this request:
    {state['query']}
    
    Return ONLY the search query, nothing else.
    """
    
    response = llm.invoke([HumanMessage(content=extract_prompt)])
    search_query = response.content.strip()
    
    # Execute search
    results = web_search_tool(search_query)
    state["search_results"] = results
    state["messages"].append(AIMessage(content=f"Search results: {results}"))
    
    return state


def calculate_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Execute calculator tool.
    """
    print("\n🧮 CALCULATE NODE")
    
    # Use LLM to extract expression from query
    extract_prompt = f"""
    Extract the mathematical expression from this request:
    {state['query']}
    
    Return ONLY the mathematical expression, nothing else.
    Example: If query is "Calculate 5 + 3", return "5 + 3"
    """
    
    response = llm.invoke([HumanMessage(content=extract_prompt)])
    expression = response.content.strip()
    
    # Execute calculation
    result = calculator_tool(expression)
    state["calculation_result"] = result
    state["messages"].append(AIMessage(content=f"Calculation: {result}"))
    
    return state


def analyze_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Execute text analysis tool.
    This demonstrates a "chain" within a LangGraph node.
    """
    print("\n📊 ANALYZE NODE")
    
    # Get text to analyze (from search results or query)
    text_to_analyze = state.get("search_results", state["query"])
    
    # Execute text analysis tool
    stats = text_analyzer_tool(text_to_analyze)
    state["text_stats"] = stats
    
    # Use LLM to create an analysis (chain behavior)
    analysis_prompt = f"""
    Analyze this information and provide insights:
    
    Text: {text_to_analyze}
    
    Statistics: {stats}
    
    Provide:
    1. Main topics covered
    2. Quality assessment
    3. Key insights
    """
    
    response = llm.invoke([HumanMessage(content=analysis_prompt)])
    state["analysis_result"] = response.content
    state["messages"].append(AIMessage(content=f"Analysis: {response.content}"))
    
    return state


def summarize_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Execute summarization.
    This demonstrates another "chain" within a LangGraph node.
    """
    print("\n📝 SUMMARIZE NODE")
    
    # Get content to summarize
    content = state.get("search_results", state.get("analysis_result", state["query"]))
    
    # Use LLM to summarize (chain behavior)
    summarize_prompt = f"""
    Summarize the following content in 2-3 sentences:
    
    {content}
    """
    
    response = llm.invoke([HumanMessage(content=summarize_prompt)])
    state["summary"] = response.content
    state["messages"].append(AIMessage(content=f"Summary: {response.content}"))
    
    return state


def finish_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Generate final answer combining all results.
    """
    print("\n✅ FINISH NODE")
    
    # Compile all results
    final_prompt = f"""
    Generate a comprehensive final answer to this query:
    {state['query']}
    
    Available information:
    - Search Results: {state.get('search_results', 'None')}
    - Calculation: {state.get('calculation_result', 'None')}
    - Analysis: {state.get('analysis_result', 'None')}
    - Summary: {state.get('summary', 'None')}
    - Text Stats: {state.get('text_stats', 'None')}
    
    Provide a clear, complete answer.
    """
    
    response = llm.invoke([HumanMessage(content=final_prompt)])
    state["final_answer"] = response.content
    state["messages"].append(AIMessage(content=f"Final Answer: {response.content}"))
    
    return state


# =============================================================================
# ROUTING LOGIC
# =============================================================================


def route_after_planning(state: AgentState) -> Literal["search", "calculate", "analyze", "summarize", "finish"]:
    """
    Route to the appropriate node based on planning decision.
    """
    action = state["next_action"]
    
    if action in ["search", "calculate", "analyze", "summarize", "finish"]:
        return action
    
    # Default to finish if unknown action
    return "finish"


def should_continue(state: AgentState) -> Literal["planning", "finish"]:
    """
    Decide whether to continue processing or finish.
    This creates the loop structure.
    """
    # Simple heuristic: if we have done multiple actions, finish
    # In a real system, this would be more sophisticated
    
    completed_actions = sum([
        1 if state.get("search_results") else 0,
        1 if state.get("calculation_result") else 0,
        1 if state.get("analysis_result") else 0,
        1 if state.get("summary") else 0,
    ])
    
    if completed_actions >= 2 or state["next_action"] == "finish":
        return "finish"
    
    return "planning"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def create_research_agent_graph(callback_handler: NoveumTraceCallbackHandler) -> StateGraph:
    """
    Create LangGraph research agent.
    
    Graph structure:
        START → planning → [search/calculate/analyze/summarize/finish]
                   ↑            ↓
                   └────────────┘ (loop back if needed)
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        callbacks=[callback_handler]
    )
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes (with LLM bound to each)
    workflow.add_node("planning", lambda state: planning_node(state, llm))
    workflow.add_node("search", lambda state: search_node(state, llm))
    workflow.add_node("calculate", lambda state: calculate_node(state, llm))
    workflow.add_node("analyze", lambda state: analyze_node(state, llm))
    workflow.add_node("summarize", lambda state: summarize_node(state, llm))
    workflow.add_node("finish", lambda state: finish_node(state, llm))
    
    # Set entry point
    workflow.set_entry_point("planning")
    
    # Add conditional edges from planning
    workflow.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "search": "search",
            "calculate": "calculate",
            "analyze": "analyze",
            "summarize": "summarize",
            "finish": "finish",
        }
    )
    
    # After each action, decide whether to continue or finish
    for node in ["search", "calculate", "analyze", "summarize"]:
        workflow.add_conditional_edges(
            node,
            should_continue,
            {
                "planning": "planning",  # Loop back
                "finish": "finish",      # Go to finish
            }
        )
    
    # Finish node goes to END
    workflow.add_edge("finish", END)
    
    return workflow


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


def run_langgraph_agent(query: str, callback_handler: NoveumTraceCallbackHandler):
    """
    Run the LangGraph agent with a query.
    """
    print(f"\nQuery: {query}\n")
    
    # Create agent graph
    workflow = create_research_agent_graph(callback_handler)
    app = workflow.compile()
    
    # Initial state
    initial_state: AgentState = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "search_results": "",
        "calculation_result": "",
        "analysis_result": "",
        "summary": "",
        "text_stats": {},
        "current_thought": "",
        "next_action": "",
        "final_answer": "",
    }
    
    # Execute with callbacks
    config = {
        "callbacks": [callback_handler],
        "metadata": {
            "agent_type": "research",
            "query": query,
        },
        "tags": ["langgraph", "research_agent", "noveum_trace"],
        "recursion_limit": 25,
    }
    
    try:
        # Run the agent
        final_state = app.invoke(initial_state, config=config)
        
        print("\n" + "=" * 80)
        print(f"Final Answer: {final_state['final_answer']}")
        print("=" * 80 + "\n")
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


def example_simple_query():
    """
    Example 1: Simple query with search and summarization.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Query with Search")
    print("=" * 80)
    
    # Initialize Noveum Trace
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "langgraph-example"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment="development",
    )
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Run query
    query = "Search for information about LangChain and summarize it."
    run_langgraph_agent(query, callback_handler)


def example_complex_query():
    """
    Example 2: Complex query with multiple operations.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Complex Query with Multiple Operations")
    print("=" * 80)
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Run query
    query = "Search for transformers in AI, then analyze the results."
    run_langgraph_agent(query, callback_handler)


def example_math_query():
    """
    Example 3: Query with calculation.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Query with Calculation")
    print("=" * 80)
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Run query
    query = "Calculate (150 * 3) + (200 / 4) and search for Python information."
    run_langgraph_agent(query, callback_handler)


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 80)
    print("LangGraph Agent with Chains and Tools - Noveum Trace Integration")
    print("=" * 80)
    print("\nThis example demonstrates the same functionality as the LangChain")
    print("agent example, but using LangGraph's explicit state-based approach.")
    print("=" * 80)
    
    # Check environment variables
    if not os.getenv("NOVEUM_API_KEY"):
        print("⚠️  Warning: NOVEUM_API_KEY not set")
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
    
    # Run examples
    try:
        example_simple_query()
        time.sleep(2)
        
        example_complex_query()
        time.sleep(2)
        
        example_math_query()
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
    
    # Flush traces
    print("\n" + "=" * 80)
    print("Flushing traces...")
    noveum_trace.flush()
    
    print("\n✅ All examples completed!")
    print("\n" + "=" * 80)
    print("COMPARISON: LangChain vs LangGraph")
    print("=" * 80)
    print("\nLangChain Agent:")
    print("  • Implicit control flow (ReAct pattern)")
    print("  • Agent autonomously decides tool usage")
    print("  • Less control over execution path")
    print("  • Easier for simple use cases")
    print("\nLangGraph Agent:")
    print("  • Explicit state management")
    print("  • Clear graph structure with nodes and edges")
    print("  • Full control over execution flow")
    print("  • Better for complex, multi-step workflows")
    print("\nBoth create similar trace structures in Noveum!")
    print("=" * 80 + "\n")
    
    print("\nCheck your Noveum Trace dashboard to see:")
    print("  • Graph/Agent spans showing execution flow")
    print("  • Node spans (planning, search, calculate, analyze, etc.)")
    print("  • Chain spans (LLM calls within nodes)")
    print("  • Tool spans for each operation")
    print("  • State transitions and routing decisions")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

