#!/usr/bin/env python3
"""
Basic usage example for Noveum Trace SDK.

This example demonstrates the core functionality of the SDK
including initialization, decorators, and basic tracing.
"""

import os
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

# Import Noveum Trace SDK
import noveum_trace
from noveum_trace import trace, trace_agent, trace_llm, trace_tool


def main():
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
        project="example-project",
        environment="development",
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


@trace(capture_performance=True)
def process_data(data: str) -> dict[str, Any]:
    """
    Example function with basic tracing.

    Args:
        data: Input data to process

    Returns:
        Processed data dictionary
    """
    # Simulate some processing
    time.sleep(0.1)

    return {
        "original": data,
        "processed": data.upper(),
        "length": len(data),
    }


@trace_llm(capture_tokens=True, estimate_costs=True)
def call_llm(prompt: str) -> str:
    """
    Example LLM call with automatic tracing.

    Args:
        prompt: Input prompt for the LLM

    Returns:
        LLM response
    """
    # Simulate LLM call
    time.sleep(0.5)

    # In a real implementation, this would call an actual LLM
    return f"Mock LLM response to: {prompt}"


@trace_agent(
    agent_id="researcher",
    role="information_gatherer",
    capabilities=["web_search", "document_analysis"],
)
def research_agent(query: str) -> dict[str, Any]:
    """
    Example agent operation with tracing.

    Args:
        query: Research query

    Returns:
        Research results
    """
    # Simulate research process
    time.sleep(0.3)

    # Use a tool within the agent
    search_results = search_web(f"research: {query}")

    return {
        "query": query,
        "findings": f"Research findings for: {query}",
        "sources": ["source1.com", "source2.com"],
        "confidence": 0.85,
        "search_results": search_results,
    }


@trace_tool(tool_name="web_search", tool_type="api")
def search_web(query: str) -> list[dict[str, str]]:
    """
    Example tool usage with tracing.

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Simulate web search
    time.sleep(0.2)

    return [
        {"title": f"Result 1 for {query}", "url": "https://example1.com"},
        {"title": f"Result 2 for {query}", "url": "https://example2.com"},
        {"title": f"Result 3 for {query}", "url": "https://example3.com"},
    ]


@trace(name="multi_agent_workflow")
def run_multi_agent_workflow(task: str) -> dict[str, Any]:
    """
    Example multi-agent workflow with tracing.

    Args:
        task: The task to be processed by agents

    Returns:
        Combined results from all agents
    """
    # Research phase
    research_data = research_agent(task)

    # Analysis phase
    analysis_result = analysis_agent(research_data)

    # Synthesis phase
    final_report = report_agent(analysis_result)

    return {
        "task": task,
        "research": research_data,
        "analysis": analysis_result,
        "final_report": final_report,
        "workflow_status": "completed",
    }


@trace_agent(
    agent_id="analyst",
    role="data_analyzer",
    capabilities=["statistical_analysis", "trend_detection"],
)
def analysis_agent(data: dict[str, Any]) -> dict[str, Any]:
    """
    Analysis agent that processes research data.

    Args:
        data: Research data to analyze

    Returns:
        Analysis results
    """
    time.sleep(0.4)

    return {
        "trends": ["trend1", "trend2", "trend3"],
        "insights": f"Key insights from {data['query']}",
        "confidence": 0.92,
        "methodology": "statistical_analysis",
    }


@trace_agent(
    agent_id="reporter",
    role="report_generator",
    capabilities=["document_generation", "visualization"],
)
def report_agent(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Report agent that creates final reports.

    Args:
        analysis: Analysis results

    Returns:
        Final report
    """
    time.sleep(0.3)

    return {
        "report_type": "market_analysis",
        "summary": "Executive summary of findings",
        "recommendations": ["rec1", "rec2", "rec3"],
        "charts": ["chart1.png", "chart2.png"],
        "confidence": analysis["confidence"],
    }


if __name__ == "__main__":
    main()
