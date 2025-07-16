#!/usr/bin/env python3
"""
Test Tools and Agents integration with Noveum Trace.
"""
import os
import sys
from typing import Any, Dict

# Load environment variables
from dotenv import load_dotenv

import noveum_trace

load_dotenv()


def test_noveum_agent_system():
    """Test the built-in Noveum Agent system."""
    print("🤖 Testing Noveum Agent System...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        # Test creating agents
        agent_registry = noveum_trace.get_agent_registry()

        # Create a simple agent
        agent_config = noveum_trace.AgentConfig(
            name="TestAgent",
            description="A test agent for demonstration",
            version="1.0.0",
            metadata={"test": True},
        )

        agent = agent_registry.register_agent(agent_config)

        print(f"✅ Agent created: {agent.name}")

        # Test agent context
        from noveum_trace.agents.context import AgentContext

        with AgentContext(agent):
            print(f"✅ Agent context active: {noveum_trace.get_current_agent().name}")

            # Test trace decorator
            @noveum_trace.trace(name="test_task")
            def test_task(input_data: str) -> str:
                return f"Processed: {input_data}"

            result = test_task("Hello from agent!")
            print(f"✅ Agent task result: {result}")

        # Test agent retrieval
        retrieved_agent = agent_registry.get_agent("TestAgent")
        print(f"✅ Retrieved agent: {retrieved_agent.name}")

        return True

    except Exception as e:
        print(f"❌ Noveum Agent system test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_openai_function_calling():
    """Test OpenAI function calling with tracing."""
    print("\n🔧 Testing OpenAI Function Calling...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        import openai

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Define function schemas
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_tip",
                    "description": "Calculate tip amount",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bill_amount": {"type": "number"},
                            "tip_percentage": {"type": "number"},
                        },
                        "required": ["bill_amount", "tip_percentage"],
                    },
                },
            },
        ]

        # Test function calling
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in New York and calculate a 20% tip on a $50 bill?",
                }
            ],
            tools=functions,
            tool_choice="auto",
        )

        print(f"✅ OpenAI Function Calling Response: {response.choices[0].message}")

        if response.choices[0].message.tool_calls:
            print(f"✅ Tool calls made: {len(response.choices[0].message.tool_calls)}")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  - {tool_call.function.name}: {tool_call.function.arguments}")

        return True

    except Exception as e:
        print(f"❌ OpenAI Function Calling test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_anthropic_tool_use():
    """Test Anthropic tool use with tracing."""
    print("\n🤖 Testing Anthropic Tool Use...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Define tools
        tools = [
            {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given symbol",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol, e.g. AAPL",
                        }
                    },
                    "required": ["symbol"],
                },
            }
        ]

        # Test tool use
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            tools=tools,
            messages=[{"role": "user", "content": "What's the stock price of Apple?"}],
        )

        print(f"✅ Anthropic Tool Use Response: {response.content}")

        # Check if tool use was triggered
        tool_use_blocks = [
            block for block in response.content if block.type == "tool_use"
        ]
        if tool_use_blocks:
            print(f"✅ Tool use blocks: {len(tool_use_blocks)}")
            for block in tool_use_blocks:
                print(f"  - {block.name}: {block.input}")

        return True

    except Exception as e:
        print(f"❌ Anthropic Tool Use test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_multi_step_agent_workflow():
    """Test a multi-step agent workflow."""
    print("\n🔄 Testing Multi-Step Agent Workflow...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        # Create agent
        agent_config = noveum_trace.AgentConfig(
            name="WorkflowAgent",
            description="Agent that executes multi-step workflows",
            version="1.0.0",
        )

        agent = noveum_trace.Agent(agent_config)

        from noveum_trace.agents.context import AgentContext

        with AgentContext(agent):

            @noveum_trace.trace(name="step_1_analyze")
            def analyze_request(request: str) -> Dict[str, Any]:
                """Analyze the user request."""
                return {
                    "intent": "information_request",
                    "entities": ["weather", "location"],
                    "confidence": 0.95,
                }

            @noveum_trace.trace(name="step_2_gather_info")
            def gather_information(analysis: Dict[str, Any]) -> Dict[str, Any]:
                """Gather information based on analysis."""
                return {
                    "weather_data": {
                        "location": "San Francisco",
                        "temperature": 72,
                        "condition": "sunny",
                    },
                    "sources": ["weather_api"],
                }

            @noveum_trace.trace(name="step_3_generate_response")
            def generate_response(info: Dict[str, Any]) -> str:
                """Generate final response."""
                weather = info["weather_data"]
                return f"The weather in {weather['location']} is {weather['condition']} with a temperature of {weather['temperature']}°F."

            # Execute workflow
            request = "What's the weather in San Francisco?"

            print(f"✅ Processing request: {request}")

            analysis = analyze_request(request)
            print(f"✅ Analysis: {analysis}")

            info = gather_information(analysis)
            print(f"✅ Information gathered: {info}")

            response = generate_response(info)
            print(f"✅ Final response: {response}")

        return True

    except Exception as e:
        print(f"❌ Multi-step workflow test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_agent_with_llm_calls():
    """Test agent that makes LLM calls."""
    print("\n🧠 Testing Agent with LLM Calls...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        import openai

        # Create agent
        agent_config = noveum_trace.AgentConfig(
            name="LLMAgent",
            description="Agent that uses LLM for processing",
            version="1.0.0",
        )

        agent = noveum_trace.Agent(agent_config)

        from noveum_trace.agents.context import AgentContext

        with AgentContext(agent):

            @noveum_trace.trace(name="llm_processing")
            def process_with_llm(query: str) -> str:
                """Process query using LLM."""
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that provides concise answers.",
                        },
                        {"role": "user", "content": query},
                    ],
                    max_tokens=100,
                )

                return response.choices[0].message.content

            # Test LLM processing within agent context
            query = "What is the capital of France?"
            result = process_with_llm(query)
            print(f"✅ LLM Agent result: {result}")

        return True

    except Exception as e:
        print(f"❌ Agent with LLM calls test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_nested_agent_calls():
    """Test nested agent calls."""
    print("\n🪆 Testing Nested Agent Calls...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        # Create parent agent
        parent_config = noveum_trace.AgentConfig(
            name="ParentAgent",
            description="Parent agent that coordinates child agents",
            version="1.0.0",
        )

        parent_agent = noveum_trace.Agent(parent_config)

        # Create child agent
        child_config = noveum_trace.AgentConfig(
            name="ChildAgent",
            description="Child agent that performs specific tasks",
            version="1.0.0",
        )

        child_agent = noveum_trace.Agent(child_config)

        from noveum_trace.agents.context import AgentContext

        with AgentContext(parent_agent):

            @noveum_trace.trace(name="parent_task")
            def parent_task(data: str) -> str:
                """Parent task that calls child."""

                with AgentContext(child_agent):

                    @noveum_trace.trace(name="child_task")
                    def child_task(input_data: str) -> str:
                        """Child task."""
                        return f"Child processed: {input_data}"

                    child_result = child_task(data)
                    return f"Parent received: {child_result}"

            # Test nested execution
            result = parent_task("test data")
            print(f"✅ Nested agent result: {result}")

        return True

    except Exception as e:
        print(f"❌ Nested agent calls test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


if __name__ == "__main__":
    print("🚀 Starting Tools and Agents Tests...")

    results = []
    results.append(test_noveum_agent_system())
    results.append(test_openai_function_calling())
    results.append(test_anthropic_tool_use())
    results.append(test_multi_step_agent_workflow())
    results.append(test_agent_with_llm_calls())
    results.append(test_nested_agent_calls())

    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")
    if all(results):
        print("🎉 All Tools and Agents tests passed!")
    else:
        print("❌ Some Tools and Agents tests failed")
        sys.exit(1)
