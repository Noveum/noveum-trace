"""
LangGraph Nested Graphs Example with Noveum Trace

This example demonstrates how to nest complete LangGraph graphs as nodes
within a parent graph, all executed with a single async invocation.

Architecture:
  Parent Graph (Orchestrator)
    ├── Node 1: data_collection (executes Graph 1)
    │   └── Graph 1: Data Collection Pipeline (3 nodes)
    │       - fetch_data → validate_data → store_data
    └── Node 2: analysis (executes Graph 2)
        └── Graph 2: Analysis Pipeline (3 nodes)
            - load_data → analyze_data → generate_report

Key Features:
- Single ainvoke call executes the entire pipeline
- Two complete graphs wrapped as nodes in a parent graph
- Data flows from Graph 1 → Graph 2 through the parent state
- All execution nested under a single trace

IMPORTANT:
When using ainvoke (async execution), it is imperative to set:
    use_langchain_assigned_parent=True
when creating the NoveumTraceCallbackHandler.

LangChain-assigned parent relationships ensure proper trace hierarchy when
nodes are executed asynchronously.
"""

import asyncio
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from noveum_trace import init as noveum_init
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

# Load environment variables
load_dotenv()


# =============================================================================
# GRAPH 1: DATA COLLECTION PIPELINE
# =============================================================================


class DataCollectionState(TypedDict):
    """State for data collection pipeline."""

    query: str
    raw_data: str
    validated: bool
    stored: bool
    error: str


def fetch_data_node(state: DataCollectionState) -> DataCollectionState:
    """Node 1: Fetch data from a source."""
    print("\n🔍 [Graph 1] FETCH DATA NODE")
    print(f"   Query: {state['query']}")

    # Simulate data fetching with LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke(
        [HumanMessage(content=f"Generate sample weather data for: {state['query']}")]
    )

    state["raw_data"] = response.content
    print(f"   ✓ Fetched {len(response.content)} characters of data")
    return state


def validate_data_node(state: DataCollectionState) -> DataCollectionState:
    """Node 2: Validate fetched data."""
    print("\n✅ [Graph 1] VALIDATE DATA NODE")

    # Simple validation check
    if state["raw_data"] and len(state["raw_data"]) > 10:
        state["validated"] = True
        state["error"] = ""
        print("   ✓ Data validation passed")
    else:
        state["validated"] = False
        state["error"] = "Data too short or empty"
        print(f"   ✗ Data validation failed: {state['error']}")

    return state


def store_data_node(state: DataCollectionState) -> DataCollectionState:
    """Node 3: Store validated data."""
    print("\n💾 [Graph 1] STORE DATA NODE")

    if state["validated"]:
        state["stored"] = True
        print("   ✓ Data stored successfully")
    else:
        state["stored"] = False
        print(f"   ✗ Cannot store invalid data: {state['error']}")

    return state


def create_data_collection_graph() -> StateGraph:
    """Create Graph 1: Data Collection Pipeline."""
    workflow = StateGraph(DataCollectionState)

    # Add nodes
    workflow.add_node("fetch", fetch_data_node)
    workflow.add_node("validate", validate_data_node)
    workflow.add_node("store", store_data_node)

    # Set entry point and edges
    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "validate")
    workflow.add_edge("validate", "store")
    workflow.add_edge("store", END)

    return workflow


# =============================================================================
# GRAPH 2: ANALYSIS PIPELINE
# =============================================================================


class AnalysisState(TypedDict):
    """State for analysis pipeline."""

    data: str
    insights: list[str]
    report: str
    recommendations: list[str]


def load_data_node(state: AnalysisState) -> AnalysisState:
    """Node 1: Load data for analysis."""
    print("\n📥 [Graph 2] LOAD DATA NODE")
    print(f"   Loading {len(state['data'])} characters of data")
    print("   ✓ Data loaded successfully")
    return state


def analyze_data_node(state: AnalysisState) -> AnalysisState:
    """Node 2: Analyze the data."""
    print("\n🔬 [Graph 2] ANALYZE DATA NODE")

    # Use LLM to analyze data
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    response = llm.invoke(
        [
            HumanMessage(
                content=f"Analyze this data and provide 3 key insights:\n\n{state['data'][:500]}"
            )
        ]
    )

    # Extract insights (simplified)
    insights = [
        line.strip()
        for line in response.content.split("\n")
        if line.strip() and len(line.strip()) > 10
    ][:3]

    state["insights"] = insights
    print(f"   ✓ Generated {len(insights)} insights")

    return state


def generate_report_node(state: AnalysisState) -> AnalysisState:
    """Node 3: Generate final report."""
    print("\n📊 [Graph 2] GENERATE REPORT NODE")

    # Use LLM to create report
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    insights_text = "\n".join(f"- {insight}" for insight in state["insights"])

    response = llm.invoke(
        [
            HumanMessage(
                content=f"Create a brief report summarizing these insights and provide 2 recommendations:\n\n{insights_text}"
            )
        ]
    )

    state["report"] = response.content

    # Extract recommendations (simplified)
    recommendations = [
        line.strip()
        for line in response.content.split("\n")
        if "recommend" in line.lower()
    ][:2]

    state["recommendations"] = (
        recommendations if recommendations else ["No specific recommendations"]
    )

    print(f"   ✓ Report generated ({len(state['report'])} characters)")
    print(f"   ✓ Recommendations: {len(state['recommendations'])}")

    return state


def create_analysis_graph() -> StateGraph:
    """Create Graph 2: Analysis Pipeline."""
    workflow = StateGraph(AnalysisState)

    # Add nodes
    workflow.add_node("load", load_data_node)
    workflow.add_node("analyze", analyze_data_node)
    workflow.add_node("report", generate_report_node)

    # Set entry point and edges
    workflow.set_entry_point("load")
    workflow.add_edge("load", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)

    return workflow


# =============================================================================
# PARENT GRAPH: COMBINES BOTH PIPELINES
# =============================================================================


class CombinedState(TypedDict):
    """Combined state for both pipelines."""

    # Data Collection fields
    query: str
    raw_data: str
    validated: bool
    stored: bool
    error: str
    # Analysis fields
    insights: list[str]
    report: str
    recommendations: list[str]


def data_collection_node(state: CombinedState, config) -> CombinedState:
    """Node that executes the entire data collection graph."""
    print("\n🔄 [Parent] EXECUTING DATA COLLECTION GRAPH NODE")

    # Create and compile the data collection graph
    graph = create_data_collection_graph()
    app = graph.compile()

    # Prepare input state for data collection
    collection_state: DataCollectionState = {
        "query": state["query"],
        "raw_data": state.get("raw_data", ""),
        "validated": state.get("validated", False),
        "stored": state.get("stored", False),
        "error": state.get("error", ""),
    }

    # Configure with custom name for this sub-graph
    sub_config = {
        "callbacks": config.get("callbacks", []),
        "metadata": {
            "noveum": {
                "name": "data_collection_graph",
            }
        },
        "tags": ["graph1", "data_collection"],
    }

    # Execute with config to enable tracing
    result = app.invoke(collection_state, config=sub_config)

    # Update combined state with results
    state["raw_data"] = result["raw_data"]
    state["validated"] = result["validated"]
    state["stored"] = result["stored"]
    state["error"] = result["error"]

    return state


def analysis_node(state: CombinedState, config) -> CombinedState:
    """Node that executes the entire analysis graph."""
    print("\n🔄 [Parent] EXECUTING ANALYSIS GRAPH NODE")

    # Create and compile the analysis graph
    graph = create_analysis_graph()
    app = graph.compile()

    # Prepare input state for analysis
    analysis_state: AnalysisState = {
        "data": state["raw_data"],
        "insights": state.get("insights", []),
        "report": state.get("report", ""),
        "recommendations": state.get("recommendations", []),
    }

    # Configure with custom name for this sub-graph
    sub_config = {
        "callbacks": config.get("callbacks", []),
        "metadata": {
            "noveum": {
                "name": "analysis_graph",
            }
        },
        "tags": ["graph2", "analysis"],
    }

    # Execute with config to enable tracing
    result = app.invoke(analysis_state, config=sub_config)

    # Update combined state with results
    state["insights"] = result["insights"]
    state["report"] = result["report"]
    state["recommendations"] = result["recommendations"]

    return state


def create_parent_graph() -> StateGraph:
    """Create parent graph that orchestrates both pipelines."""
    workflow = StateGraph(CombinedState)

    # Add nodes for both graphs
    workflow.add_node("data_collection", data_collection_node)
    workflow.add_node("analysis", analysis_node)

    # Set entry point and edges
    workflow.set_entry_point("data_collection")
    workflow.add_edge("data_collection", "analysis")
    workflow.add_edge("analysis", END)

    return workflow


# =============================================================================
# MAIN EXECUTION
# =============================================================================


async def main():
    """
    Example showing async graph execution with nested graphs as nodes.
    """
    print("=" * 80)
    print("🎯 LANGGRAPH NESTED GRAPHS ASYNC EXAMPLE")
    print("=" * 80)

    # Initialize Noveum Trace
    noveum_init(
        project=os.getenv("NOVEUM_PROJECT", "test-project"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
    )
    print("✅ Noveum Trace initialized")

    # Create callback handler
    handler = NoveumTraceCallbackHandler(use_langchain_assigned_parent=True)
    print("✅ Callback handler created")

    # ==========================================================================
    # EXECUTE PARENT GRAPH WITH BOTH PIPELINES AS NODES
    # ==========================================================================

    print("\n" + "=" * 80)
    print("📊 EXECUTING PARENT GRAPH (Contains both pipelines as nodes)")
    print("=" * 80)

    # Create and compile parent graph
    parent_graph = create_parent_graph()
    parent_app = parent_graph.compile()

    # Configure parent graph
    config = {
        "callbacks": [handler],
        "metadata": {
            "noveum": {
                "name": "data_pipeline_orchestrator",
            }
        },
        "tags": ["parent_graph", "orchestrator"],
    }

    # Execute parent graph with single ainvoke
    initial_state: CombinedState = {
        "query": "Tokyo weather patterns",
        "raw_data": "",
        "validated": False,
        "stored": False,
        "error": "",
        "insights": [],
        "report": "",
        "recommendations": [],
    }

    result = await parent_app.ainvoke(initial_state, config=config)

    print("\n✅ Parent graph completed")
    print(f"   Data Collection - Validated: {result['validated']}")
    print(f"   Data Collection - Stored: {result['stored']}")
    print(f"   Analysis - Insights: {len(result['insights'])}")
    print(f"   Analysis - Report length: {len(result['report'])} characters")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    print("\n" + "=" * 80)
    print("✅ EXAMPLE COMPLETED")
    print("=" * 80)

    print("\n📊 Trace Structure:")
    print("  Trace: data_pipeline_orchestrator")
    print("  └── Span: data_pipeline_orchestrator (Parent Graph)")
    print("      ├── Span: data_collection (node - executes Graph 1)")
    print("      │   └── Span: StateGraph (Graph 1)")
    print("      │       ├── Span: fetch (node)")
    print("      │       │   └── Span: llm.gpt-4o-mini")
    print("      │       ├── Span: validate (node)")
    print("      │       └── Span: store (node)")
    print("      └── Span: analysis (node - executes Graph 2)")
    print("          └── Span: StateGraph (Graph 2)")
    print("              ├── Span: load (node)")
    print("              ├── Span: analyze (node)")
    print("              │   └── Span: llm.gpt-4o-mini")
    print("              └── Span: report (node)")
    print("                  └── Span: llm.gpt-4o-mini")

    print("\n💡 Key Points:")
    print("   1. Single parent graph orchestrates both pipelines")
    print("   2. Each pipeline is wrapped as a node in the parent graph")
    print("   3. Only ONE ainvoke call executes everything")
    print("   4. Data flows from data_collection node → analysis node")
    print("   5. Parent graph uses metadata.noveum.name='data_pipeline_orchestrator'")
    print("   6. All execution is nested under a single trace")
    print("   7. Traces are auto-managed (no manual start/end needed)")

    print("\n📋 Results:")
    print(f"   Data Collection - Stored: {result['stored']}")
    print(f"   Analysis - Insights: {result['insights']}")
    print(f"   Analysis - Recommendations: {result['recommendations']}")

    print(f"\n🔍 Handler state: {handler}")


if __name__ == "__main__":
    asyncio.run(main())
