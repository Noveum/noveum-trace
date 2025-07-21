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

# Initialize the SDK
noveum_trace.init(
    project="agent-example",
    api_key=os.getenv("NOVEUM_API_KEY"),
    environment="development",
)


# Data structures for agent communication
class AgentMessage:
    sender: str
    recipient: str
    content: str
    message_type: str = "text"
    metadata: dict[str, Any] = None


class ResearchAgent:
    """Agent that performs research tasks."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.knowledge_base = []

    @noveum_trace.trace_agent(agent_id="researcher_001")
    def research_topic(self, topic: str) -> dict[str, Any]:
        """Research a given topic and return findings."""

        # Simulate research process
        with noveum_trace.trace_context(name="web_search"):
            search_results = self._search_web(topic)

        with noveum_trace.trace_context(name="analyze_sources"):
            analysis = self._analyze_sources(search_results)

        # Store in knowledge base
        research_result = {
            "topic": topic,
            "findings": analysis,
            "sources": search_results,
            "confidence": random.uniform(0.7, 0.95),
            "timestamp": time.time(),
            "agent_id": self.agent_id,
        }

        self.knowledge_base.append(research_result)
        return research_result

    @noveum_trace.trace_tool(tool_name="web_search", tool_type="api")
    def _search_web(self, query: str) -> list[dict[str, str]]:
        """Search the web for information."""
        time.sleep(0.3)  # Simulate search time

        # Mock search results
        results = [
            {
                "title": f"Research on {query} - Academic Paper",
                "url": f"https://example.com/paper/{query.replace(' ', '-')}",
                "snippet": f"Comprehensive analysis of {query} with detailed findings...",
                "relevance": random.uniform(0.8, 1.0),
            },
            {
                "title": f"{query} - Industry Report",
                "url": f"https://industry.com/report/{query.replace(' ', '-')}",
                "snippet": f"Industry insights and trends related to {query}...",
                "relevance": random.uniform(0.7, 0.9),
            },
        ]

        return results

    @noveum_trace.trace_tool(tool_name="source_analyzer", tool_type="analysis")
    def _analyze_sources(self, sources: list[dict[str, str]]) -> str:
        """Analyze research sources and extract key insights."""
        time.sleep(0.2)  # Simulate analysis time

        # Mock analysis
        key_points = [
            "Primary finding indicates significant correlation",
            "Secondary analysis reveals emerging trends",
            "Expert consensus supports main hypothesis",
        ]

        return "; ".join(key_points)


class WriterAgent:
    """Agent that creates written content."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.writing_style = "professional"

    @noveum_trace.trace_agent(agent_id="writer_001")
    def write_report(self, research_data: dict[str, Any]) -> str:
        """Write a report based on research data."""

        with noveum_trace.trace_context(name="outline_creation"):
            outline = self._create_outline(research_data)

        with noveum_trace.trace_context(name="content_generation"):
            content = self._generate_content(outline, research_data)

        with noveum_trace.trace_context(name="review_and_edit"):
            final_report = self._review_and_edit(content)

        return final_report

    @noveum_trace.trace_tool(tool_name="outline_generator", tool_type="writing")
    def _create_outline(self, research_data: dict[str, Any]) -> list[str]:
        """Create an outline for the report."""
        time.sleep(0.1)

        outline = [
            "Executive Summary",
            f"Introduction to {research_data['topic']}",
            "Key Findings",
            "Analysis and Implications",
            "Conclusions and Recommendations",
        ]

        return outline

    @noveum_trace.trace_tool(tool_name="content_generator", tool_type="writing")
    def _generate_content(
        self, outline: list[str], research_data: dict[str, Any]
    ) -> str:
        """Generate content based on outline and research."""
        time.sleep(0.4)

        content_sections = []
        for section in outline:
            if "Executive Summary" in section:
                section_content = f"""
                ## {section}
                This report provides a comprehensive analysis of {research_data['topic']}.
                Key findings include: {research_data['findings'][:100]}...
                """
            elif "Introduction" in section:
                section_content = f"""
                ## {section}
                {research_data['topic']} represents a critical area of study.
                This analysis draws from {len(research_data['sources'])} sources.
                """
            else:
                section_content = f"""
                ## {section}
                [Content for {section} would be generated based on research findings]
                """

            content_sections.append(section_content)

        return "\n".join(content_sections)

    @noveum_trace.trace_tool(tool_name="content_editor", tool_type="writing")
    def _review_and_edit(self, content: str) -> str:
        """Review and edit the generated content."""
        time.sleep(0.2)

        # Mock editing process
        edited_content = content.replace("[Content for", "Detailed analysis for")
        edited_content += "\n\n## Conclusion\nThis analysis provides valuable insights into the topic."

        return edited_content


class CoordinatorAgent:
    """Agent that coordinates other agents."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    @noveum_trace.trace_agent(agent_id="coordinator_main")
    def coordinate_research_project(self, topic: str) -> dict[str, Any]:
        """Coordinate a multi-agent research project."""

        # Initialize sub-agents
        researcher = ResearchAgent("researcher_001")
        writer = WriterAgent("writer_001")

        # Step 1: Research phase
        with noveum_trace.trace_context(name="research_phase"):
            research_result = researcher.research_topic(topic)

            # Send message to writer (simulated inter-agent communication)
            message = AgentMessage(
                sender="coordinator",
                recipient="writer_001",
                content="Research completed, ready for writing phase",
                metadata={"research_id": research_result.get("timestamp")},
            )
            self._send_agent_message(message)

        # Step 2: Writing phase
        with noveum_trace.trace_context(name="writing_phase"):
            report = writer.write_report(research_result)

            # Send completion message
            completion_message = AgentMessage(
                sender="coordinator",
                recipient="system",
                content="Project completed successfully",
                metadata={"report_length": len(report)},
            )
            self._send_agent_message(completion_message)

        # Step 3: Quality assessment
        with noveum_trace.trace_context(name="quality_assessment"):
            quality_score = self._assess_quality(research_result, report)

        project_result = {
            "topic": topic,
            "research_data": research_result,
            "report": report,
            "quality_score": quality_score,
            "completion_time": time.time(),
            "agents_involved": ["coordinator_main", "researcher_001", "writer_001"],
        }

        return project_result

    @noveum_trace.trace_tool(tool_name="message_sender", tool_type="communication")
    def _send_agent_message(self, message: AgentMessage) -> bool:
        """Send a message between agents."""
        time.sleep(0.05)  # Simulate message passing delay

        # Log the message
        print(f"Message: {message.sender} -> {message.recipient}: {message.content}")

        return True

    @noveum_trace.trace_tool(tool_name="quality_assessor", tool_type="analysis")
    def _assess_quality(self, research_data: dict[str, Any], report: str) -> float:
        """Assess the quality of the completed work."""
        time.sleep(0.1)

        # Mock quality assessment
        research_quality = research_data.get("confidence", 0.8)
        report_quality = min(len(report) / 1000, 1.0)  # Simple length-based metric

        overall_quality = (research_quality + report_quality) / 2
        return round(overall_quality, 2)


def main():
    """Main function demonstrating the agent workflow."""

    # Initialize Noveum Trace SDK
    api_key = os.getenv("NOVEUM_API_KEY")
    if not api_key:
        raise ValueError(
            "NOVEUM_API_KEY environment variable is required. "
            "Please set it before running this example."
        )

    noveum_trace.init(
        api_key=api_key,
        project="agent_workflow_demo",
        environment="development",
    )

    print("🤖 Starting Multi-Agent Research Project Demo")
    print("=" * 50)

    # Create coordinator agent
    coordinator = CoordinatorAgent("main_coordinator")

    # Execute a research project
    topic = "Artificial Intelligence in Healthcare"

    with noveum_trace.trace_context(name="complete_research_project"):
        result = coordinator.coordinate_research_project(topic)

    # Display results
    print("\n📊 Project Results:")
    print(f"Topic: {result['topic']}")
    print(f"Quality Score: {result['quality_score']}")
    print(f"Agents Involved: {', '.join(result['agents_involved'])}")
    print(f"Report Length: {len(result['report'])} characters")

    print("\n📝 Generated Report Preview:")
    print("-" * 30)
    print(
        result["report"][:300] + "..."
        if len(result["report"]) > 300
        else result["report"]
    )

    # Flush traces
    noveum_trace.flush()

    print("\n✅ Demo completed! Traces have been sent to Noveum platform.")
    print("🔍 Check your Noveum dashboard to view the complete agent workflow trace.")


if __name__ == "__main__":
    main()
