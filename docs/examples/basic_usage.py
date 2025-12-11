#!/usr/bin/env python3
"""
Basic usage example for Noveum Trace SDK.

This example demonstrates the core functionality of the SDK
including initialization and context manager-based tracing.
"""

import os
import time
from typing import Any, cast

# Load environment variables (install python-dotenv if needed)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "python-dotenv not installed. Environment variables will be read from system only."
    )
    pass

# Import Noveum Trace SDK
import noveum_trace
from noveum_trace import trace_llm_call, trace_operation, trace_agent_operation


def main() -> None:
    """Main example function."""

    # Initialize the SDK
    api_key = os.getenv("NOVEUM_API_KEY")
    if not api_key:
        raise ValueError(
            "NOVEUM_API_KEY environment variable is required. "
            "Please set it before running this example."
        )

    noveum_trace.init(
        api_key=api_key,
        project="example-project-testing-basic-usage",
        environment="dev-aman",
    )

    print("Noveum Trace SDK initialized!")

    # Example 1: Basic function tracing
    result1 = process_data("Hello, World!")
    print(f"Result 1: {result1}")

    # Example 2: LLM call tracing
    result2 = call_llm("What is the capital of France?")
    print(f"Result 2: {result2}")

    # Example 3: Agent operation tracing
    result3 = research_agent("AI trends 2024")
    print(f"Result 3: {result3}")

    # Example 4: Tool usage tracing
    result4 = search_web("Python best practices")
    print(f"Result 4: {result4}")

    # Example 5: Multi-agent workflow
    workflow_result = run_multi_agent_workflow("Analyze market trends")
    print(f"Workflow result: {workflow_result}")

    # Flush any pending traces
    noveum_trace.flush()
    print("All traces sent!")


def process_data(data: str) -> dict[str, Any]:
    """
    Example function with basic tracing using context managers.

    Args:
        data: Input data to process

    Returns:
        Processed data dictionary
    """
    with trace_operation("process_data") as span:
        # Simulate some processing
        time.sleep(0.1)

        result = {
            "original": data,
            "processed": data.upper(),
            "length": len(data),
        }

        span.set_attribute("data.length", len(data))
        span.set_attribute("data.processed", result["processed"])

        return result


def call_llm(prompt: str) -> str:
    """
    Example LLM call with context manager-based tracing.

    Args:
        prompt: Input prompt for the LLM

    Returns:
        LLM response
    """
    with trace_llm_call(model="gpt-4", provider="openai") as span:
        # Simulate LLM call
        time.sleep(0.5)

        # In a real implementation, this would call an actual LLM
        response = f"Mock LLM response to: {prompt}"

        span.set_attribute("llm.prompt", prompt)
        span.set_attribute("llm.response", response)

        return response


def research_agent(query: str) -> dict[str, Any]:
    """
    Example agent operation with context manager-based tracing.

    Args:
        query: Research query

    Returns:
        Research results
    """
    with trace_agent_operation(
        agent_type="researcher", operation="information_gathering"
    ) as span:
        # Simulate research process
        time.sleep(0.3)

        # Use a tool within the agent
        search_results = search_web(f"research: {query}")

        result = {
            "query": query,
            "findings": f"Research findings for: {query}",
            "sources": ["source1.com", "source2.com"],
            "confidence": 0.85,
            "search_results": search_results,
        }

        span.set_attribute("agent.query", query)
        span.set_attribute("agent.confidence", 0.85)
        sources = cast(list[str], result["sources"])
        span.set_attribute("agent.sources_count", len(sources))

        return result


def search_web(query: str) -> list[dict[str, str]]:
    """
    Example tool usage with context manager-based tracing.

    Args:
        query: Search query

    Returns:
        Search results
    """
    with trace_operation("web_search") as span:
        # Simulate web search
        time.sleep(0.2)

        results = [
            {"title": f"Result 1 for {query}", "url": "https://example1.com"},
            {"title": f"Result 2 for {query}", "url": "https://example2.com"},
            {"title": f"Result 3 for {query}", "url": "https://example3.com"},
        ]

        span.set_attribute("tool.name", "web_search")
        span.set_attribute("tool.query", query)
        span.set_attribute("tool.results_count", len(results))

        return results


def run_multi_agent_workflow(task: str) -> dict[str, Any]:
    """
    Example multi-agent workflow with context manager-based tracing.

    Args:
        task: The task to be processed by agents

    Returns:
        Combined results from all agents
    """
    with trace_operation("multi_agent_workflow") as span:
        # Research phase
        research_data = research_agent(task)

        # Analysis phase
        analysis_result = analysis_agent(research_data)

        # Synthesis phase
        final_report = report_agent(analysis_result)

        result = {
            "task": task,
            "research": research_data,
            "analysis": analysis_result,
            "final_report": final_report,
            "workflow_status": "completed",
        }

        span.set_attribute("workflow.task", task)
        span.set_attribute("workflow.status", "completed")
        span.set_attribute("workflow.steps", 3)

        return result


def analysis_agent(data: dict[str, Any]) -> dict[str, Any]:
    """
    Analysis agent that processes research data.

    Args:
        data: Research data to analyze

    Returns:
        Analysis results
    """
    with trace_agent_operation(agent_type="analyst", operation="data_analysis") as span:
        time.sleep(0.4)

        result = {
            "trends": ["trend1", "trend2", "trend3"],
            "insights": f"Key insights from {data['query']}",
            "confidence": 0.92,
            "methodology": "statistical_analysis",
        }

        span.set_attribute("agent.confidence", result["confidence"])
        trends = cast(list[str], result["trends"])
        span.set_attribute("agent.trends_count", len(trends))

        return result


def report_agent(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Report agent that creates final reports.

    Args:
        analysis: Analysis results

    Returns:
        Final report
    """
    with trace_agent_operation(
        agent_type="reporter", operation="report_generation"
    ) as span:
        time.sleep(0.3)

        result = {
            "report_type": "market_analysis",
            "summary": "Executive summary of findings",
            "recommendations": ["rec1", "rec2", "rec3"],
            "charts": ["chart1.png", "chart2.png"],
            "confidence": analysis["confidence"],
        }

        span.set_attribute("agent.report_type", result["report_type"])
        span.set_attribute(
            "agent.recommendations_count", len(result["recommendations"])
        )

        return result


if __name__ == "__main__":
    main()
