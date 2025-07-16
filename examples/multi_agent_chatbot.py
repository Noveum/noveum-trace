"""
Multi-Agent Chatbot Example

This example demonstrates how to use the Noveum Trace SDK to build and monitor
a multi-agent chatbot system with different specialized agents.

The system includes:
- Router Agent: Routes user queries to appropriate specialists
- Knowledge Agent: Handles factual questions and information retrieval
- Creative Agent: Handles creative writing and brainstorming
- Code Agent: Handles programming and technical questions
- Coordinator Agent: Manages complex multi-step conversations
"""

import asyncio
import time
from typing import Dict, List

# Import Noveum Trace SDK with multi-agent support
import noveum_trace
from noveum_trace import (
    AgentConfig,
    AgentContext,
    get_agent_registry,
    llm_trace,
    observe,
    trace,
    update_current_span,
)
from noveum_trace.sinks.console import ConsoleSink, ConsoleSinkConfig
from noveum_trace.sinks.file import FileSink, FileSinkConfig
from noveum_trace.types import CustomHeaders


# Mock LLM client for demonstration
class MockLLMClient:
    """Mock LLM client that simulates different AI models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def chat_completion(self, messages: List[Dict], temperature: float = 0.7):
        """Simulate LLM chat completion."""
        # Simulate API call delay
        await asyncio.sleep(0.1)

        # Generate mock response based on model and input
        last_message = messages[-1]["content"] if messages else ""

        if "router" in self.model_name:
            # Router responses
            if "code" in last_message.lower() or "programming" in last_message.lower():
                response = "code"
            elif "creative" in last_message.lower() or "story" in last_message.lower():
                response = "creative"
            elif (
                "complex" in last_message.lower()
                or "multi-step" in last_message.lower()
            ):
                response = "coordinator"
            else:
                response = "knowledge"
        elif "knowledge" in self.model_name:
            response = f"Knowledge response: Here's what I know about '{last_message}'"
        elif "creative" in self.model_name:
            response = f"Creative response: Let me create something interesting about '{last_message}'"
        elif "code" in self.model_name:
            response = f"Code response: Here's a solution for '{last_message}'"
        elif "coordinator" in self.model_name:
            response = (
                f"Coordinator response: I'll break down '{last_message}' into steps"
            )
        else:
            response = f"Generic response to: {last_message}"

        return MockResponse(response, self.model_name)


class MockResponse:
    """Mock LLM response object."""

    def __init__(self, content: str, model: str):
        self.choices = [MockChoice(content)]
        self.usage = MockUsage()
        self.model = model
        self.id = f"mock-{int(time.time())}"


class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)
        self.finish_reason = "stop"


class MockMessage:
    def __init__(self, content: str):
        self.content = content


class MockUsage:
    def __init__(self):
        self.prompt_tokens = 50
        self.completion_tokens = 100
        self.total_tokens = 150


class MultiAgentChatbot:
    """Multi-agent chatbot system with specialized agents."""

    def __init__(self):
        """Initialize the multi-agent chatbot system."""
        self.registry = get_agent_registry()
        self.agents = {}
        self.conversation_history = []

        # Initialize agents
        self._setup_agents()

    def _setup_agents(self):
        """Set up all specialized agents."""

        # Router Agent - Routes queries to appropriate specialists
        router_config = AgentConfig(
            name="router",
            agent_type="router",
            description="Routes user queries to appropriate specialist agents",
            capabilities={"query_routing", "intent_classification"},
            tags={"core", "routing"},
            custom_headers=CustomHeaders(additional_headers={"agent_role": "router"}),
        )

        # Knowledge Agent - Handles factual questions
        knowledge_config = AgentConfig(
            name="knowledge",
            agent_type="knowledge_specialist",
            description="Handles factual questions and information retrieval",
            capabilities={"fact_retrieval", "information_synthesis"},
            tags={"specialist", "knowledge"},
            parent_agent="router",
            custom_headers=CustomHeaders(
                additional_headers={"agent_role": "knowledge_specialist"}
            ),
        )

        # Creative Agent - Handles creative tasks
        creative_config = AgentConfig(
            name="creative",
            agent_type="creative_specialist",
            description="Handles creative writing and brainstorming",
            capabilities={"creative_writing", "brainstorming", "storytelling"},
            tags={"specialist", "creative"},
            parent_agent="router",
            custom_headers=CustomHeaders(
                additional_headers={"agent_role": "creative_specialist"}
            ),
        )

        # Code Agent - Handles programming questions
        code_config = AgentConfig(
            name="code",
            agent_type="code_specialist",
            description="Handles programming and technical questions",
            capabilities={"code_generation", "debugging", "technical_explanation"},
            tags={"specialist", "technical"},
            parent_agent="router",
            custom_headers=CustomHeaders(
                additional_headers={"agent_role": "code_specialist"}
            ),
        )

        # Coordinator Agent - Manages complex multi-step tasks
        coordinator_config = AgentConfig(
            name="coordinator",
            agent_type="coordinator",
            description="Manages complex multi-step conversations",
            capabilities={"task_coordination", "multi_step_planning"},
            tags={"coordinator", "complex_tasks"},
            parent_agent="router",
            custom_headers=CustomHeaders(
                additional_headers={"agent_role": "coordinator"}
            ),
        )

        # Register all agents
        self.agents["router"] = self.registry.register_agent(router_config)
        self.agents["knowledge"] = self.registry.register_agent(knowledge_config)
        self.agents["creative"] = self.registry.register_agent(creative_config)
        self.agents["code"] = self.registry.register_agent(code_config)
        self.agents["coordinator"] = self.registry.register_agent(coordinator_config)

        # Create LLM clients for each agent
        self.llm_clients = {
            "router": MockLLMClient("router-model"),
            "knowledge": MockLLMClient("knowledge-model"),
            "creative": MockLLMClient("creative-model"),
            "code": MockLLMClient("code-model"),
            "coordinator": MockLLMClient("coordinator-model"),
        }

    @trace(name="process_user_query")
    async def process_query(self, user_query: str) -> str:
        """Process a user query through the multi-agent system."""
        update_current_span(
            input=user_query,
            metadata={
                "query_length": len(user_query),
                "conversation_turn": len(self.conversation_history) + 1,
            },
        )

        # Step 1: Route the query
        specialist_type = await self._route_query(user_query)

        # Step 2: Process with appropriate specialist
        response = await self._process_with_specialist(user_query, specialist_type)

        # Step 3: Store conversation history
        self.conversation_history.append(
            {
                "user_query": user_query,
                "specialist_used": specialist_type,
                "response": response,
                "timestamp": time.time(),
            }
        )

        update_current_span(
            output=response,
            metadata={
                "specialist_used": specialist_type,
                "response_length": len(response),
                "conversation_length": len(self.conversation_history),
            },
        )

        return response

    @observe(
        name="query_routing",
        metrics=["routing_accuracy", "routing_latency"],
        capture_input=True,
        capture_output=True,
    )
    async def _route_query(self, query: str) -> str:
        """Route query to appropriate specialist agent."""
        with AgentContext(self.agents["router"]):
            update_current_span(
                metadata={
                    "routing_algorithm": "llm_based",
                    "available_specialists": list(self.agents.keys()),
                }
            )

            # Use router LLM to classify intent
            routing_response = await self._call_router_llm(query)
            specialist_type = routing_response.choices[0].message.content.strip()

            # Validate specialist type
            if specialist_type not in ["knowledge", "creative", "code", "coordinator"]:
                specialist_type = "knowledge"  # Default fallback

            update_current_span(
                metadata={
                    "routing_decision": specialist_type,
                    "routing_confidence": 0.85,  # Mock confidence score
                }
            )

            return specialist_type

    @llm_trace(model="router-gpt-4", operation="chat", ai_system="openai")
    async def _call_router_llm(self, query: str):
        """Call LLM for query routing."""
        messages = [
            {
                "role": "system",
                "content": "You are a query router. Respond with only one word: 'knowledge', 'creative', 'code', or 'coordinator'",
            },
            {"role": "user", "content": query},
        ]

        return await self.llm_clients["router"].chat_completion(messages)

    @observe(
        name="specialist_processing",
        metrics=["processing_quality", "response_time"],
        capture_input=True,
        capture_output=True,
    )
    async def _process_with_specialist(self, query: str, specialist_type: str) -> str:
        """Process query with the appropriate specialist agent."""
        specialist_agent = self.agents[specialist_type]

        with AgentContext(specialist_agent):
            update_current_span(
                metadata={
                    "specialist_type": specialist_type,
                    "specialist_capabilities": list(
                        specialist_agent.config.capabilities
                    ),
                    "processing_mode": "single_turn",
                }
            )

            # Call appropriate specialist method
            if specialist_type == "knowledge":
                response = await self._process_knowledge_query(query)
            elif specialist_type == "creative":
                response = await self._process_creative_query(query)
            elif specialist_type == "code":
                response = await self._process_code_query(query)
            elif specialist_type == "coordinator":
                response = await self._process_coordinator_query(query)
            else:
                response = "I'm not sure how to handle that query."

            update_current_span(
                metadata={
                    "processing_completed": True,
                    "response_type": specialist_type,
                }
            )

            return response

    @llm_trace(model="knowledge-gpt-4", operation="chat", ai_system="openai")
    async def _process_knowledge_query(self, query: str) -> str:
        """Process knowledge-based queries."""
        update_current_span(
            metadata={"query_type": "knowledge", "knowledge_domain": "general"}
        )

        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable assistant. Provide accurate, factual information.",
            },
            {"role": "user", "content": query},
        ]

        response = await self.llm_clients["knowledge"].chat_completion(messages)
        return response.choices[0].message.content

    @llm_trace(model="creative-gpt-4", operation="chat", ai_system="openai")
    async def _process_creative_query(self, query: str) -> str:
        """Process creative queries."""
        update_current_span(
            metadata={"query_type": "creative", "creativity_level": "high"}
        )

        messages = [
            {
                "role": "system",
                "content": "You are a creative assistant. Be imaginative and original in your responses.",
            },
            {"role": "user", "content": query},
        ]

        response = await self.llm_clients["creative"].chat_completion(
            messages, temperature=0.9
        )
        return response.choices[0].message.content

    @llm_trace(model="code-gpt-4", operation="chat", ai_system="openai")
    async def _process_code_query(self, query: str) -> str:
        """Process code-related queries."""
        update_current_span(
            metadata={"query_type": "code", "programming_context": "general"}
        )

        messages = [
            {
                "role": "system",
                "content": "You are a programming assistant. Provide clear, working code solutions.",
            },
            {"role": "user", "content": query},
        ]

        response = await self.llm_clients["code"].chat_completion(
            messages, temperature=0.3
        )
        return response.choices[0].message.content

    @llm_trace(model="coordinator-gpt-4", operation="chat", ai_system="openai")
    async def _process_coordinator_query(self, query: str) -> str:
        """Process complex queries requiring coordination."""
        update_current_span(
            metadata={"query_type": "coordinator", "complexity_level": "high"}
        )

        # Coordinator might delegate to multiple specialists
        messages = [
            {
                "role": "system",
                "content": "You are a coordinator. Break down complex tasks into manageable steps.",
            },
            {"role": "user", "content": query},
        ]

        response = await self.llm_clients["coordinator"].chat_completion(messages)
        return response.choices[0].message.content

    @trace(name="get_conversation_stats")
    def get_conversation_stats(self) -> Dict:
        """Get statistics about the conversation."""
        if not self.conversation_history:
            return {"total_queries": 0}

        specialist_usage = {}
        total_response_length = 0

        for entry in self.conversation_history:
            specialist = entry["specialist_used"]
            specialist_usage[specialist] = specialist_usage.get(specialist, 0) + 1
            total_response_length += len(entry["response"])

        stats = {
            "total_queries": len(self.conversation_history),
            "specialist_usage": specialist_usage,
            "average_response_length": total_response_length
            / len(self.conversation_history),
            "conversation_duration": time.time()
            - self.conversation_history[0]["timestamp"],
        }

        update_current_span(output=stats, metadata={"stats_generated": True})

        return stats


async def main():
    """Main function demonstrating the multi-agent chatbot."""

    # Initialize Noveum Trace SDK
    noveum_trace.init(
        project_name="multi-agent-chatbot",
        custom_headers=CustomHeaders(
            project_id="chatbot-demo", org_id="noveum-examples"
        ),
        sinks=[
            ConsoleSink(ConsoleSinkConfig(format_json=True, include_timestamp=True)),
            FileSink(
                FileSinkConfig(file_path="chatbot_traces.jsonl", max_file_size_mb=10)
            ),
        ],
    )

    print("🤖 Multi-Agent Chatbot Demo")
    print("=" * 50)

    # Create chatbot system
    chatbot = MultiAgentChatbot()

    # Display agent information
    registry = get_agent_registry()
    print(f"\n📊 Registered Agents: {len(registry)}")
    for agent in registry.list_agents():
        print(f"  • {agent.name} ({agent.agent_type}): {agent.config.description}")

    # Sample queries to demonstrate different agents
    sample_queries = [
        "What is the capital of France?",  # Knowledge agent
        "Write a short story about a robot",  # Creative agent
        "How do I implement a binary search in Python?",  # Code agent
        "Help me plan a complex multi-step project",  # Coordinator agent
        "Tell me about machine learning",  # Knowledge agent
    ]

    print(f"\n💬 Processing {len(sample_queries)} sample queries...")
    print("-" * 50)

    # Process each query
    for i, query in enumerate(sample_queries, 1):
        print(f"\n[Query {i}] User: {query}")

        try:
            response = await chatbot.process_query(query)
            print(f"[Response {i}] Bot: {response}")

            # Show which agent handled the query
            if chatbot.conversation_history:
                last_entry = chatbot.conversation_history[-1]
                specialist = last_entry["specialist_used"]
                print(f"[Agent] Handled by: {specialist}")

        except Exception as e:
            print(f"[Error] Failed to process query: {e}")

        # Small delay between queries
        await asyncio.sleep(0.5)

    # Display conversation statistics
    print("\n📈 Conversation Statistics")
    print("-" * 30)
    stats = chatbot.get_conversation_stats()

    print(f"Total Queries: {stats['total_queries']}")
    print(f"Average Response Length: {stats['average_response_length']:.1f} characters")
    print(f"Conversation Duration: {stats['conversation_duration']:.2f} seconds")

    print("\nSpecialist Usage:")
    for specialist, count in stats["specialist_usage"].items():
        percentage = (count / stats["total_queries"]) * 100
        print(f"  • {specialist}: {count} queries ({percentage:.1f}%)")

    # Display registry statistics
    print("\n🔍 Agent Registry Statistics")
    print("-" * 30)
    registry_stats = registry.get_registry_stats()

    print(f"Total Agents: {registry_stats['total_agents']}")
    print(f"Active Agents: {registry_stats['active_agents']}")
    print(f"Total Traces: {registry_stats['total_traces']}")
    print(f"Active Traces: {registry_stats['active_traces']}")

    print("\nAgent Types:")
    for agent_type, count in registry_stats["agent_type_counts"].items():
        print(f"  • {agent_type}: {count}")

    # Shutdown
    print("\n🔄 Shutting down...")
    registry.shutdown()
    noveum_trace.shutdown()

    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
