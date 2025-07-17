"""
Agent Example for Noveum Trace SDK.

This example demonstrates how to trace complex agent interactions,
including multi-agent systems, agent workflows, and agent graphs.
"""

import os
import random
import time
from typing import Any

# Load environment variables (install python-dotenv if needed)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "python-dotenv not installed. Environment variables will be read from system only."
    )
    pass

import noveum_trace
from noveum_trace.agents import (
    create_agent,
    create_agent_graph,
    create_agent_workflow,
    trace_agent_operation,
)

# Initialize the SDK
noveum_trace.init(
    project="agent-example",
    api_key=os.getenv("NOVEUM_API_KEY"),
    environment="development",
)


# Mock agent functions for demonstration
def mock_planning(query: str) -> dict[str, Any]:
    """Simulate a planning agent."""
    time.sleep(0.5)  # Simulate processing
    return {
        "plan": [
            {"step": 1, "task": "Research information", "agent": "researcher"},
            {"step": 2, "task": "Analyze findings", "agent": "analyst"},
            {"step": 3, "task": "Generate report", "agent": "writer"},
        ],
        "query": query,
    }


def mock_research(topic: str) -> list[dict[str, str]]:
    """Simulate a research agent."""
    time.sleep(0.7)  # Simulate processing
    return [
        {"source": "Wikipedia", "content": f"Basic information about {topic}"},
        {"source": "Academic Journal", "content": f"Detailed analysis of {topic}"},
        {"source": "News Article", "content": f"Recent developments in {topic}"},
    ]


def mock_analysis(data: list[dict[str, str]]) -> dict[str, Any]:
    """Simulate an analysis agent."""
    time.sleep(0.6)  # Simulate processing
    return {
        "summary": f"Analysis of {len(data)} sources",
        "key_points": [
            "First important insight",
            "Second important insight",
            "Third important insight",
        ],
        "confidence": random.uniform(0.7, 0.95),
    }


def mock_writing(analysis: dict[str, Any], topic: str) -> str:
    """Simulate a writing agent."""
    time.sleep(0.8)  # Simulate processing
    return f"""
    # Report on {topic}

    ## Summary
    {analysis["summary"]}

    ## Key Points
    - {analysis["key_points"][0]}
    - {analysis["key_points"][1]}
    - {analysis["key_points"][2]}

    ## Conclusion
    This report was generated with {analysis["confidence"]:.2f} confidence.
    """


# Example 1: Basic agent tracing
def example_basic_agent():
    """Demonstrate basic agent tracing."""
    print("\n=== Example 1: Basic Agent Tracing ===")

    # Create a research agent
    researcher = create_agent(
        agent_id="researcher",
        agent_type="research_agent",
        capabilities=["web_search", "document_retrieval"],
    )

    # Use the agent to track operations
    with researcher:
        print("Researcher agent activated")

        # Record an interaction
        researcher.record_interaction(
            interaction_type="query_received", content="quantum computing"
        )

        # Trace a specific operation
        with trace_agent_operation(researcher, "web_search") as span:
            print("Performing research on quantum computing...")
            results = mock_research("quantum computing")

            # Add operation metrics
            span.set_attributes(
                {
                    "search.query": "quantum computing",
                    "search.results_count": len(results),
                }
            )

        # Record the results
        researcher.record_interaction(
            interaction_type="search_results", content=results
        )

        print(f"Research complete: {len(results)} sources found")

    print("\n")


# Example 2: Multi-agent graph
def example_agent_graph():
    """Demonstrate multi-agent graph tracing."""
    print("\n=== Example 2: Multi-Agent Graph ===")

    # Create an agent graph
    graph = create_agent_graph(name="Research Team")

    # Use the graph to track multi-agent interactions
    with graph:
        print("Agent graph activated")

        # Add agents to the graph
        graph.add_agent(
            agent_id="planner",
            agent_type="planning_agent",
            capabilities=["task_planning"],
        )

        graph.add_agent(
            agent_id="researcher",
            agent_type="research_agent",
            capabilities=["web_search"],
        )

        graph.add_agent(
            agent_id="analyst",
            agent_type="analysis_agent",
            capabilities=["data_analysis"],
        )

        # Add relationships between agents
        graph.add_edge(
            source_id="planner", target_id="researcher", edge_type="delegates"
        )

        graph.add_edge(
            source_id="researcher", target_id="analyst", edge_type="provides_data"
        )

        # Activate the planner agent
        planner = graph.get_agent("planner")
        with planner:
            print("Planner agent activated")

            # Create a plan
            with trace_agent_operation(planner, "planning") as span:
                print("Creating research plan...")
                plan = mock_planning("artificial intelligence")

                # Add operation metrics
                span.set_attributes(
                    {"plan.steps": len(plan["plan"]), "plan.query": plan["query"]}
                )

            # Record the plan
            planner.record_interaction(interaction_type="plan_created", content=plan)

            # Delegate to researcher
            graph.record_interaction(
                source_id="planner",
                interaction_type="task_assignment",
                target_id="researcher",
                content="Research artificial intelligence",
            )

        # Activate the researcher agent
        researcher = graph.get_agent("researcher")
        with researcher:
            print("Researcher agent activated")

            # Perform research
            with trace_agent_operation(researcher, "research") as span:
                print("Researching artificial intelligence...")
                results = mock_research("artificial intelligence")

                # Add operation metrics
                span.set_attributes(
                    {
                        "research.topic": "artificial intelligence",
                        "research.sources": len(results),
                    }
                )

            # Record the results
            researcher.record_interaction(
                interaction_type="research_completed", content=results
            )

            # Send to analyst
            graph.record_interaction(
                source_id="researcher",
                interaction_type="data_transfer",
                target_id="analyst",
                content=results,
            )

        # Activate the analyst agent
        analyst = graph.get_agent("analyst")
        with analyst:
            print("Analyst agent activated")

            # Analyze the data
            with trace_agent_operation(analyst, "analysis") as span:
                print("Analyzing research data...")
                analysis = mock_analysis(results)

                # Add operation metrics
                span.set_attributes(
                    {
                        "analysis.confidence": analysis["confidence"],
                        "analysis.key_points": len(analysis["key_points"]),
                    }
                )

            # Record the analysis
            analyst.record_interaction(
                interaction_type="analysis_completed", content=analysis
            )

        print("Multi-agent process complete")

    print("\n")


# Example 3: Agent workflow
def example_agent_workflow():
    """Demonstrate agent workflow tracing."""
    print("\n=== Example 3: Agent Workflow ===")

    # Create an agent workflow
    workflow = create_agent_workflow(name="Research Project")

    # Use the workflow to track task execution
    with workflow:
        print("Workflow activated")

        # Add tasks to the workflow
        planning_task = workflow.add_task(
            task_name="Create research plan", agent_id="planner"
        )

        research_task = workflow.add_task(
            task_name="Gather information",
            agent_id="researcher",
            dependencies=[planning_task["id"]],
        )

        analysis_task = workflow.add_task(
            task_name="Analyze findings",
            agent_id="analyst",
            dependencies=[research_task["id"]],
        )

        report_task = workflow.add_task(
            task_name="Generate report",
            agent_id="writer",
            dependencies=[analysis_task["id"]],
        )

        # Execute the planning task
        print("Executing planning task...")
        plan = mock_planning("climate change")
        workflow.update_task_status(
            task_id=planning_task["id"], status="completed", result=plan
        )

        # Get next tasks to execute
        next_tasks = workflow.get_next_tasks()
        print(f"Next tasks: {[task['name'] for task in next_tasks]}")

        # Execute the research task
        print("Executing research task...")
        research_results = mock_research("climate change")
        workflow.update_task_status(
            task_id=research_task["id"], status="completed", result=research_results
        )

        # Get next tasks to execute
        next_tasks = workflow.get_next_tasks()
        print(f"Next tasks: {[task['name'] for task in next_tasks]}")

        # Execute the analysis task
        print("Executing analysis task...")
        analysis_results = mock_analysis(research_results)
        workflow.update_task_status(
            task_id=analysis_task["id"], status="completed", result=analysis_results
        )

        # Get next tasks to execute
        next_tasks = workflow.get_next_tasks()
        print(f"Next tasks: {[task['name'] for task in next_tasks]}")

        # Execute the report task
        print("Executing report task...")
        report = mock_writing(analysis_results, "climate change")
        workflow.update_task_status(
            task_id=report_task["id"], status="completed", result=report
        )

        # Check workflow completion
        completed_tasks = workflow.get_tasks(status="completed")
        print(
            f"Workflow complete: {len(completed_tasks)}/{len(workflow.tasks)} tasks completed"
        )

    print("\n")


# Example 4: Complex agent system
def example_complex_agent_system():
    """Demonstrate a complex agent system with graph and workflow."""
    print("\n=== Example 4: Complex Agent System ===")

    # Create an agent graph
    graph = create_agent_graph(name="Research System")

    # Create an agent workflow
    workflow = create_agent_workflow(name="Climate Research")

    # Use both to track a complex system
    with graph, workflow:
        print("Complex agent system activated")

        # Add agents to the graph
        graph.add_agent(
            agent_id="orchestrator",
            agent_type="orchestrator_agent",
            capabilities=["workflow_management"],
        )

        graph.add_agent(
            agent_id="planner",
            agent_type="planning_agent",
            capabilities=["task_planning"],
        )

        graph.add_agent(
            agent_id="researcher",
            agent_type="research_agent",
            capabilities=["web_search"],
        )

        graph.add_agent(
            agent_id="analyst",
            agent_type="analysis_agent",
            capabilities=["data_analysis"],
        )

        graph.add_agent(
            agent_id="writer",
            agent_type="writing_agent",
            capabilities=["content_generation"],
        )

        # Add relationships
        graph.add_edge(
            source_id="orchestrator", target_id="planner", edge_type="manages"
        )

        graph.add_edge(
            source_id="orchestrator", target_id="researcher", edge_type="manages"
        )

        graph.add_edge(
            source_id="orchestrator", target_id="analyst", edge_type="manages"
        )

        graph.add_edge(
            source_id="orchestrator", target_id="writer", edge_type="manages"
        )

        # Add workflow tasks
        planning_task = workflow.add_task(
            task_name="Create research plan", agent_id="planner"
        )

        research_task = workflow.add_task(
            task_name="Gather information",
            agent_id="researcher",
            dependencies=[planning_task["id"]],
        )

        analysis_task = workflow.add_task(
            task_name="Analyze findings",
            agent_id="analyst",
            dependencies=[research_task["id"]],
        )

        report_task = workflow.add_task(
            task_name="Generate report",
            agent_id="writer",
            dependencies=[analysis_task["id"]],
        )

        # Activate the orchestrator
        orchestrator = graph.get_agent("orchestrator")
        with orchestrator:
            print("Orchestrator activated")

            # Start the workflow
            orchestrator.record_interaction(
                interaction_type="workflow_started",
                content="Starting climate change research workflow",
            )

            # Activate the planner
            planner = graph.get_agent("planner")
            with planner:
                print("Planner activated")

                # Execute planning task
                with trace_agent_operation(planner, "planning") as span:
                    print("Creating research plan...")
                    plan = mock_planning("climate change")

                    # Add operation metrics
                    span.set_attributes(
                        {"plan.steps": len(plan["plan"]), "plan.query": plan["query"]}
                    )

                # Update workflow
                workflow.update_task_status(
                    task_id=planning_task["id"], status="completed", result=plan
                )

                # Record interaction
                graph.record_interaction(
                    source_id="planner",
                    interaction_type="plan_completed",
                    target_id="orchestrator",
                    content=plan,
                )

            # Activate the researcher
            researcher = graph.get_agent("researcher")
            with researcher:
                print("Researcher activated")

                # Execute research task
                with trace_agent_operation(researcher, "research") as span:
                    print("Researching climate change...")
                    research_results = mock_research("climate change")

                    # Add operation metrics
                    span.set_attributes(
                        {
                            "research.topic": "climate change",
                            "research.sources": len(research_results),
                        }
                    )

                # Update workflow
                workflow.update_task_status(
                    task_id=research_task["id"],
                    status="completed",
                    result=research_results,
                )

                # Record interaction
                graph.record_interaction(
                    source_id="researcher",
                    interaction_type="research_completed",
                    target_id="orchestrator",
                    content=research_results,
                )

            # Activate the analyst
            analyst = graph.get_agent("analyst")
            with analyst:
                print("Analyst activated")

                # Execute analysis task
                with trace_agent_operation(analyst, "analysis") as span:
                    print("Analyzing research data...")
                    analysis_results = mock_analysis(research_results)

                    # Add operation metrics
                    span.set_attributes(
                        {
                            "analysis.confidence": analysis_results["confidence"],
                            "analysis.key_points": len(analysis_results["key_points"]),
                        }
                    )

                # Update workflow
                workflow.update_task_status(
                    task_id=analysis_task["id"],
                    status="completed",
                    result=analysis_results,
                )

                # Record interaction
                graph.record_interaction(
                    source_id="analyst",
                    interaction_type="analysis_completed",
                    target_id="orchestrator",
                    content=analysis_results,
                )

            # Activate the writer
            writer = graph.get_agent("writer")
            with writer:
                print("Writer activated")

                # Execute report task
                with trace_agent_operation(writer, "writing") as span:
                    print("Generating report...")
                    report = mock_writing(analysis_results, "climate change")

                    # Add operation metrics
                    span.set_attributes(
                        {
                            "writing.topic": "climate change",
                            "writing.length": len(report),
                        }
                    )

                # Update workflow
                workflow.update_task_status(
                    task_id=report_task["id"], status="completed", result=report
                )

                # Record interaction
                graph.record_interaction(
                    source_id="writer",
                    interaction_type="report_completed",
                    target_id="orchestrator",
                    content=report,
                )

            # Complete the workflow
            orchestrator.record_interaction(
                interaction_type="workflow_completed",
                content="Climate change research workflow completed",
            )

        # Check workflow completion
        completed_tasks = workflow.get_tasks(status="completed")
        print(
            f"Complex system complete: {len(completed_tasks)}/{len(workflow.tasks)} tasks completed"
        )

    print("\n")


if __name__ == "__main__":
    # Run all examples
    example_basic_agent()
    example_agent_graph()
    example_agent_workflow()
    example_complex_agent_system()

    # Flush traces before exiting
    noveum_trace.flush()
