"""
LangGraph Iterative Research Agent Example with Noveum Trace Integration

This example demonstrates:
1. A complex LangGraph agent with self-loops
2. Integration with Noveum Trace callbacks
3. Multiple node types (LLM, tool, decision)
4. Conditional routing
5. State management
6. Iterative refinement pattern

Agent Flow:
    START → research → evaluate → [sufficient?]
                ↑                      ↓ yes
                └─── refine ←─────── ↓ no
                                      ↓
                                  synthesize → END

The agent can loop back to research multiple times if the information
gathered is insufficient to answer the question.
"""

import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

import noveum_trace

# Import Noveum Trace
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

# At the start of the file or in setup_noveum_trace()
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================


def setup_noveum_trace():
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "test-project"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
    )


# =============================================================================
# STATE DEFINITION
# =============================================================================


class ResearchState(TypedDict):
    """State for the research agent."""

    # The research question
    question: str

    # Conversation history
    messages: Annotated[list, add_messages]

    # Current search query
    current_query: str

    # Search results accumulated
    search_results: list[str]

    # Number of research iterations performed
    iteration_count: int

    # Maximum iterations allowed
    max_iterations: int

    # Quality assessment of current information
    information_quality: str  # "insufficient", "sufficient", "excellent"

    # Final synthesized answer
    final_answer: str


# =============================================================================
# TOOLS
# =============================================================================


@tool
def web_search(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: The search query

    Returns:
        Search results as a string
    """
    # Simulated search results - in production, use real search API
    # (Tavily, SerpAPI, DuckDuckGo, etc.)

    simulated_results = {
        "langchain": """
        LangChain is a framework for developing applications powered by language models.
        It provides tools for chaining together LLM calls, managing prompts, and integrating
        with various data sources. Key features include chains, agents, and memory.
        """,
        "langgraph": """
        LangGraph is a library for building stateful, multi-actor applications with LLMs.
        It extends LangChain with graph-based orchestration, allowing complex workflows
        with cycles, persistence, and human-in-the-loop interactions.
        """,
        "callbacks": """
        Callbacks in LangChain provide hooks for observability and tracing. They include
        events like on_llm_start, on_chain_start, on_tool_start, etc. Callbacks are
        passed through RunnableConfig and propagate to child components.
        """,
        "agents": """
        LangChain agents use LLMs to decide which tools to use and in what order.
        They follow a reasoning loop: observe, think, act, repeat. ReAct is a popular
        agent pattern that combines reasoning traces with actions.
        """,
        "default": """
        General information about AI, machine learning, and natural language processing.
        Large language models like GPT can understand and generate human-like text.
        They are trained on vast amounts of data and can perform various tasks.
        """,
    }

    # Find best matching result
    query_lower = query.lower()
    for key, result in simulated_results.items():
        if key in query_lower:
            return f"Search results for '{query}':\n{result}"

    return f"Search results for '{query}':\n{simulated_results['default']}"


@tool
def check_information_quality(
    question: str, gathered_info: str, min_iterations: int = 0
) -> dict:
    """
    Evaluate if gathered information is sufficient to answer the question.

    Args:
        question: The original research question
        gathered_info: Information gathered so far
        min_iterations: Minimum iterations required before passing (default 0)

    Returns:
        Dictionary with quality assessment
    """
    # Simulated quality check - normal difficulty
    info_length = len(gathered_info)
    num_sources = gathered_info.count("Search results for")

    # Check if minimum iterations requirement is met
    if num_sources < min_iterations:
        return {
            "quality": "insufficient",
            "reason": f"Only {num_sources} iterations completed, need at least {min_iterations}",
            "suggestion": "Continue research to meet minimum iteration requirement",
        }

    # Normal quality thresholds
    if info_length < 100:
        return {
            "quality": "insufficient",
            "reason": "Not enough information gathered",
            "suggestion": "Need more comprehensive search",
        }
    elif info_length < 500:
        return {
            "quality": "sufficient",
            "reason": "Basic information available",
            "suggestion": "Could benefit from more details",
        }
    else:
        return {
            "quality": "excellent",
            "reason": "Comprehensive information gathered",
            "suggestion": "Ready to synthesize answer",
        }


# =============================================================================
# AGENT NODES
# =============================================================================


def research_node(state: ResearchState) -> ResearchState:
    """
    Perform research by searching for information.

    This node can be hit multiple times (self-loop) as the agent
    refines its search queries.
    """
    print(f"\n🔍 RESEARCH NODE (Iteration {state['iteration_count'] + 1})")
    print(f"Query: {state['current_query']}")

    # Perform search
    search_result = web_search.invoke({"query": state["current_query"]})

    # Update state
    state["search_results"].append(search_result)
    state["iteration_count"] += 1

    # Add to message history
    state["messages"].append(
        AIMessage(
            content=f"Searched for: {state['current_query']}\n\nFound: {search_result}"
        )
    )

    print(f"✓ Found {len(search_result)} characters of information")

    return state


def evaluate_node(state: ResearchState) -> ResearchState:
    """
    Evaluate if gathered information is sufficient.

    This is a decision point that determines if we need another
    research iteration or can proceed to synthesis.
    """
    print("\n📊 EVALUATE NODE")

    # Combine all search results
    all_info = "\n\n".join(state["search_results"])

    # Determine minimum iterations based on question complexity
    # For complex questions (example 2), require at least 3 iterations
    min_iterations = (
        3
        if "blockchain" in state["question"].lower()
        or "consensus" in state["question"].lower()
        else 0
    )

    # Check quality
    quality_check = check_information_quality.invoke(
        {
            "question": state["question"],
            "gathered_info": all_info,
            "min_iterations": min_iterations,
        }
    )

    state["information_quality"] = quality_check["quality"]

    print(f"Quality: {quality_check['quality']}")
    print(f"Reason: {quality_check['reason']}")

    # Add evaluation to messages
    state["messages"].append(
        AIMessage(
            content=f"Evaluation: {quality_check['quality']} - {quality_check['reason']}"
        )
    )

    return state


def refine_query_node(state: ResearchState) -> ResearchState:
    """
    Refine the search query for better results.

    This node is reached when information is insufficient,
    leading back to the research node (self-loop).
    """
    print("\n🔧 REFINE QUERY NODE")

    # Use LLM to refine query
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    refine_prompt = f"""
    Original question: {state['question']}

    Previous searches:
    {chr(10).join(f"- {result[:100]}..." for result in state['search_results'])}

    The information gathered is insufficient. Suggest a more specific
    search query to find missing information. Be concise.

    Return ONLY the refined query, nothing else.
    """

    response = llm.invoke([HumanMessage(content=refine_prompt)])
    refined_query = response.content.strip()

    # Update query
    state["current_query"] = refined_query

    print(f"Refined query: {refined_query}")

    # Add to messages
    state["messages"].append(
        AIMessage(content=f"Refining search query to: {refined_query}")
    )

    return state


def synthesize_node(state: ResearchState) -> ResearchState:
    """
    Synthesize final answer from gathered information.

    This is the final node before END.
    """
    print("\n📝 SYNTHESIZE NODE")

    # Use LLM to synthesize answer
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    all_info = "\n\n".join(state["search_results"])

    synthesis_prompt = f"""
    Question: {state['question']}

    Gathered Information:
    {all_info}

    Synthesize a comprehensive answer to the question based on the gathered information.
    Be clear, concise, and cite the information sources when relevant.
    """

    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    final_answer = response.content

    state["final_answer"] = final_answer

    print(f"✓ Generated answer ({len(final_answer)} characters)")

    # Add to messages
    state["messages"].append(AIMessage(content=f"Final Answer:\n{final_answer}"))

    return state


# =============================================================================
# ROUTING LOGIC
# =============================================================================


def should_continue_research(state: ResearchState) -> Literal["refine", "synthesize"]:
    """
    Decide whether to continue researching or synthesize answer.

    This creates the conditional edge that enables self-loops.
    """
    # Check if we've hit max iterations
    if state["iteration_count"] >= state["max_iterations"]:
        print("\n⚠️  Max iterations reached, proceeding to synthesis")
        return "synthesize"

    # Check information quality
    quality = state["information_quality"]

    if quality in ["sufficient", "excellent"]:
        print(f"\n✅ Information quality is {quality}, proceeding to synthesis")
        return "synthesize"
    else:
        print(
            f"\n🔄 Information quality is {quality}, refining query for another iteration"
        )
        return "refine"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def create_research_agent() -> StateGraph:
    """
    Create the research agent graph with self-loops.

    Graph structure:
        START → research → evaluate → [decision]
                  ↑                      ↓
                  └──── refine ←────────┘
                                         ↓
                                    synthesize → END
    """
    # Create graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("refine", refine_query_node)
    workflow.add_node("synthesize", synthesize_node)

    # Set entry point
    workflow.set_entry_point("research")

    # Add edges
    workflow.add_edge("research", "evaluate")

    # Conditional edge from evaluate (this enables the self-loop)
    workflow.add_conditional_edges(
        "evaluate",
        should_continue_research,
        {
            "refine": "refine",  # Loop back via refine
            "synthesize": "synthesize",  # Exit loop
        },
    )

    # Edge from refine back to research (completes the self-loop)
    workflow.add_edge("refine", "research")

    # Edge from synthesize to end
    workflow.add_edge("synthesize", END)

    return workflow


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def run_research_agent(question: str, max_iterations: int = 3):
    """
    Run the research agent with Noveum Trace integration.

    Args:
        question: Research question to answer
        max_iterations: Maximum number of research iterations
    """
    print("=" * 80)
    print("🤖 ITERATIVE RESEARCH AGENT WITH NOVEUM TRACE")
    print("=" * 80)

    # Setup Noveum Trace
    setup_noveum_trace()

    # Create callback handler
    handler = NoveumTraceCallbackHandler()
    print("✅ Noveum Trace callback handler created")

    # Create agent
    workflow = create_research_agent()
    app = workflow.compile()
    print("✅ Research agent compiled")

    # Initial state
    initial_state: ResearchState = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "current_query": question,  # Start with the question itself
        "search_results": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "information_quality": "insufficient",
        "final_answer": "",
    }

    print(f"\n📋 Research Question: {question}")
    print(f"📊 Max Iterations: {max_iterations}")
    print("\n" + "=" * 80)
    print("🚀 STARTING AGENT EXECUTION")
    print("=" * 80)

    # Execute with callbacks
    config = {
        "callbacks": [handler],
        "metadata": {
            "agent_type": "research",
            "question": question,
            "max_iterations": max_iterations,
        },
        "tags": ["research_agent", "langgraph", "noveum_trace"],
        "recursion_limit": 200,
    }

    try:
        # Run the agent
        final_state = app.invoke(initial_state, config=config)

        print("\n" + "=" * 80)
        print("✅ AGENT EXECUTION COMPLETED")
        print("=" * 80)

        # Print summary
        print("\n📊 EXECUTION SUMMARY:")
        print(f"   • Iterations performed: {final_state['iteration_count']}")
        print(f"   • Search results gathered: {len(final_state['search_results'])}")
        print(f"   • Final quality: {final_state['information_quality']}")
        print(f"   • Answer length: {len(final_state['final_answer'])} characters")

        print("\n📝 FINAL ANSWER:")
        print("─" * 80)
        print(final_state["final_answer"])
        print("─" * 80)

        return final_state

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Simple question (should need 1-2 iterations)
    print("\n\n" + "=" * 80)
    print("EXAMPLE 1: Simple Question")
    print("=" * 80)

    run_research_agent(
        question="Explain the theoretical foundations of attention mechanisms in transformer architectures, including the mathematical formulation of scaled dot-product attention, multi-head attention, and how positional encodings enable sequence-order awareness in a permutation-invariant architecture. Compare this with RNN-based sequential processing and analyze the computational complexity trade-offs.",
        max_iterations=5,
    )

    # Wait a bit between examples
    import time

    time.sleep(2)

    # Example 2: Complex question (should need multiple iterations)
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Complex Question")
    print("=" * 80)

    run_research_agent(
        question="Analyze the distributed consensus algorithms used in blockchain networks, specifically comparing Byzantine Fault Tolerance (BFT) variants like PBFT, Tendermint, and HotStuff with Nakamoto consensus. Discuss the CAP theorem implications, finality guarantees, performance characteristics under adversarial conditions, and how these relate to the scalability trilemma. Include mathematical proofs of safety and liveness properties.",
        max_iterations=10,
    )

    print("\n\n" + "=" * 80)
    print("🎉 ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nCheck your Noveum Trace dashboard to see:")
    print("  • Graph-level spans for each agent execution")
    print("  • Node-level spans (research, evaluate, refine, synthesize)")
    print("  • LLM call spans within nodes")
    print("  • Tool call spans (web_search, check_information_quality)")
    print("  • Self-loop iterations clearly visible in trace hierarchy")
    print("\nTrace structure:")
    print("  Trace: research_agent")
    print("  └── Span: graph execution")
    print("      ├── Span: research (iteration 1)")
    print("      │   └── Span: web_search tool")
    print("      ├── Span: evaluate")
    print("      │   └── Span: check_information_quality tool")
    print("      ├── Span: refine")
    print("      │   └── Span: llm.gpt-4o-mini")
    print("      ├── Span: research (iteration 2) ← SELF-LOOP!")
    print("      │   └── Span: web_search tool")
    print("      ├── Span: evaluate")
    print("      └── Span: synthesize")
    print("          └── Span: llm.gpt-4o-mini")
