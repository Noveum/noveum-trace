"""
LangGraph Custom Parent Span Example with Noveum Trace (Prioritized Mode)

This example demonstrates how to use metadata.noveum with prioritize_manually_assigned_parents=True to:
1. Assign custom names to graph execution spans
2. Explicitly set parent-child relationships between two separate graphs
3. PRIORITIZE manually assigned parents over LangChain's automatic parent detection

The example shows two independent LangGraph graphs where Graph 2
is explicitly made a child of Graph 1 using custom naming.

Graph 1: Data Collection Pipeline (3 nodes)
  - fetch_data → validate_data → store_data

Graph 2: Analysis Pipeline (3 nodes)
  - load_data → analyze_data → generate_report

Both graphs are executed separately but Graph 2 is explicitly
nested under Graph 1 in the trace hierarchy using parent_name.

IMPORTANT: With prioritize_manually_assigned_parents=True, ALL of Graph 2's nodes
(load, analyze, report) are placed directly under the Graph 1 parent span
(data_collection_graph), NOT under a separate Graph 2 span. This allows for
flat hierarchies where child graph nodes appear as siblings to parent graph nodes.
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from noveum_trace import init as noveum_init
from noveum_trace.integrations import NoveumTraceCallbackHandler

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
# MAIN EXECUTION
# =============================================================================


def main():
    """
    Example showing custom graph names and explicit parent relationships.
    """
    print("=" * 80)
    print("🎯 LANGGRAPH CUSTOM PARENT SPAN EXAMPLE (PRIORITIZED MODE)")
    print("=" * 80)

    # Initialize Noveum Trace
    noveum_init(
        project=os.getenv("NOVEUM_PROJECT", "test-project"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
    )
    print("✅ Noveum Trace initialized")

    # Create callback handler with prioritize_manually_assigned_parents=True
    handler = NoveumTraceCallbackHandler(
        use_langchain_assigned_parent=True, prioritize_manually_assigned_parents=True
    )
    print("✅ Callback handler created (prioritize_manually_assigned_parents=True)")

    # Manually start a trace
    # This prevents auto-finishing so both graphs stay in the same trace
    handler.start_trace("data_pipeline_with_analysis")
    print("✅ Trace started manually")

    # ==========================================================================
    # EXECUTE GRAPH 1: DATA COLLECTION
    # ==========================================================================

    print("\n" + "=" * 80)
    print("📊 EXECUTING GRAPH 1: DATA COLLECTION PIPELINE (Parent)")
    print("=" * 80)

    # Create and compile Graph 1
    graph1 = create_data_collection_graph()
    app1 = graph1.compile()

    # Configure with custom name
    config_graph1 = {
        "callbacks": [handler],
        "metadata": {
            "noveum": {
                "name": "data_collection_graph",  # Custom name for Graph 1
            }
        },
        "tags": ["graph1", "data_collection"],
    }

    # Execute Graph 1
    initial_state_1: DataCollectionState = {
        "query": "Tokyo weather patterns",
        "raw_data": "",
        "validated": False,
        "stored": False,
        "error": "",
    }

    result1 = app1.invoke(initial_state_1, config=config_graph1)

    print("\n✅ Graph 1 completed")
    print(f"   Validated: {result1['validated']}")
    print(f"   Stored: {result1['stored']}")

    # ==========================================================================
    # EXECUTE GRAPH 2: ANALYSIS
    # ==========================================================================

    print("\n" + "=" * 80)
    print("📊 EXECUTING GRAPH 2: ANALYSIS PIPELINE (Child)")
    print("=" * 80)

    # Create and compile Graph 2
    graph2 = create_analysis_graph()
    app2 = graph2.compile()

    # Configure with custom name AND parent reference
    config_graph2 = {
        "callbacks": [handler],
        "metadata": {
            "noveum": {
                # Custom name for Graph 2
                "parent_name": "data_collection_graph",  # Reference to Graph 1
            }
        },
        "tags": ["graph2", "analysis"],
    }

    # Execute Graph 2 using data from Graph 1
    initial_state_2: AnalysisState = {
        "data": result1["raw_data"],  # Pass data from Graph 1
        "insights": [],
        "report": "",
        "recommendations": [],
    }

    result2 = app2.invoke(initial_state_2, config=config_graph2)

    print("\n✅ Graph 2 completed")
    print(f"   Insights: {len(result2['insights'])}")
    print(f"   Report length: {len(result2['report'])} characters")

    # Manually end the trace
    handler.end_trace()
    print("\n✅ Trace ended manually")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    print("\n" + "=" * 80)
    print("✅ EXAMPLE COMPLETED")
    print("=" * 80)

    print("\n📊 Trace Structure (as shown in the trace visualization):")
    print("  Trace: data_pipeline_with_analysis")
    print("  └── Span: data_collection_graph (Graph 1)")
    print("      ├── Span: fetch (Graph 1 node)")
    print("      │   └── Span: llm.gpt-4o-mini")
    print("      ├── Span: validate (Graph 1 node)")
    print("      ├── Span: store (Graph 1 node)")
    print("      ├── Span: graph.node.load (Graph 2 node) ← Directly under Graph 1!")
    print("      ├── Span: graph.node.analyze (Graph 2 node) ← Directly under Graph 1!")
    print("      │   └── Span: llm.gpt-4o-mini")
    print("      └── Span: graph.node.report (Graph 2 node) ← Directly under Graph 1!")
    print("          └── Span: llm.gpt-4o-mini")
    print("\n  ⚠️  NOTE: Graph 2's nodes are DIRECT children of data_collection_graph,")
    print("      NOT nested under a separate Graph 2 span. This is the key difference")
    print("      when using prioritize_manually_assigned_parents=True.")

    print("\n💡 Key Points:")
    print("   1. handler.start_trace() creates a trace and disables auto-finishing")
    print("   2. Both graphs stay within the same manually controlled trace")
    print("   3. Graph 1 uses metadata.noveum.name='data_collection_graph'")
    print("   4. Graph 2 uses metadata.noveum.parent_name='data_collection_graph'")
    print(
        "   5. prioritize_manually_assigned_parents=True ensures manual parent_name takes precedence"
    )
    print("   6. ALL Graph 2 nodes become DIRECT children of 'data_collection_graph'")
    print("   7. Graph 2 span is created as a flat hierarchy")
    print("   8. handler.end_trace() manually finishes the trace")

    print("\n📋 Results:")
    print(f"   Graph 1 stored data: {result1['stored']}")
    print(f"   Graph 2 insights: {result2['insights']}")
    print(f"   Graph 2 recommendations: {result2['recommendations']}")

    print(f"\n🔍 Handler state: {handler}")


if __name__ == "__main__":
    main()
