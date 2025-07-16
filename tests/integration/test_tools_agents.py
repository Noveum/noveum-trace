#!/usr/bin/env python3
"""
Test Tools and Agents integration with Noveum Trace.
"""
import logging
import os
from typing import Any, Dict

import pytest

# Load environment variables
from dotenv import load_dotenv

import noveum_trace

load_dotenv()

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def tracer_setup():
    """Setup tracer for tests."""
    logger.info("Setting up tracer for test")
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )
    yield
    logger.info("Shutting down tracer after test")
    noveum_trace.shutdown()


@pytest.fixture
def agent_registry(tracer_setup):
    """Get agent registry for tests."""
    return noveum_trace.get_agent_registry()


@pytest.fixture
def test_agent(agent_registry):
    """Create a test agent for use in tests."""
    agent_config = noveum_trace.AgentConfig(
        name="TestAgent",
        description="A test agent for demonstration",
        version="1.0.0",
        metadata={"test": True},
    )
    return agent_registry.register_agent(agent_config)


@pytest.fixture
def openai_client():
    """Create OpenAI client if API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OpenAI API key not available")

    # Validate API key format
    if not api_key.startswith("sk-"):
        pytest.skip("Invalid OpenAI API key format - must start with 'sk-'")

    # Validate minimum length
    if len(api_key) < 20:
        pytest.skip("Invalid OpenAI API key - too short")

    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        # Test the client with a simple request to validate the key
        # We'll just create the client and let individual tests handle API calls
        logger.info("OpenAI client created successfully")
        return client
    except ImportError:
        pytest.skip("OpenAI library not available")
    except Exception as e:
        pytest.skip(f"Failed to create OpenAI client: {e}")


@pytest.fixture
def anthropic_client():
    """Create Anthropic client if API key is available."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("Anthropic API key not available")

    # Validate API key format
    if not api_key.startswith("sk-ant-"):
        pytest.skip("Invalid Anthropic API key format - must start with 'sk-ant-'")

    # Validate minimum length
    if len(api_key) < 30:
        pytest.skip("Invalid Anthropic API key - too short")

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client created successfully")
        return client
    except ImportError:
        pytest.skip("Anthropic library not available")
    except Exception as e:
        pytest.skip(f"Failed to create Anthropic client: {e}")


@pytest.mark.integration
def test_noveum_agent_system(tracer_setup, agent_registry, test_agent):
    """Test the built-in Noveum Agent system."""
    logger.info("Testing Noveum Agent System")

    # Test agent was created successfully
    assert test_agent.name == "TestAgent"
    assert test_agent.config.description == "A test agent for demonstration"
    logger.info(f"Agent created: {test_agent.name}")

    # Test agent context
    from noveum_trace.agents.context import AgentContext

    with AgentContext(test_agent):
        current_agent = noveum_trace.get_current_agent()
        assert current_agent.name == "TestAgent"
        logger.info(f"Agent context active: {current_agent.name}")

        # Test trace decorator
        @noveum_trace.trace(name="test_task")
        def test_task(input_data: str) -> str:
            return f"Processed: {input_data}"

        result = test_task("Hello from agent!")
        assert result == "Processed: Hello from agent!"
        logger.info(f"Agent task result: {result}")

    # Test agent retrieval
    retrieved_agent = agent_registry.get_agent("TestAgent")
    assert retrieved_agent.name == "TestAgent"
    logger.info(f"Retrieved agent: {retrieved_agent.name}")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_function_calling(tracer_setup, openai_client):
    """Test OpenAI function calling with tracing."""
    logger.info("Testing OpenAI Function Calling")

    # Additional runtime validation for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OpenAI API key not available")

    # Validate API key format (basic check)
    if not api_key.startswith("sk-"):
        pytest.skip("Invalid OpenAI API key format")

    logger.info("OpenAI API key validated successfully")

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
    try:
        response = openai_client.chat.completions.create(
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
    except Exception as e:
        pytest.fail(f"OpenAI API call failed: {e}")

    assert response.choices[0].message is not None
    logger.info(f"OpenAI Function Calling Response: {response.choices[0].message}")

    if response.choices[0].message.tool_calls:
        tool_calls = response.choices[0].message.tool_calls
        assert len(tool_calls) > 0
        logger.info(f"Tool calls made: {len(tool_calls)}")
        for tool_call in tool_calls:
            logger.info(
                f"  - {tool_call.function.name}: {tool_call.function.arguments}"
            )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_anthropic_tool_use(tracer_setup, anthropic_client):
    """Test Anthropic tool use with tracing."""
    logger.info("Testing Anthropic Tool Use")

    # Additional runtime validation for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("Anthropic API key not available")

    # Validate API key format (basic check)
    if not api_key.startswith("sk-ant-"):
        pytest.skip("Invalid Anthropic API key format")

    logger.info("Anthropic API key validated successfully")

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
    try:
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            tools=tools,
            messages=[{"role": "user", "content": "What's the stock price of Apple?"}],
        )
    except Exception as e:
        pytest.fail(f"Anthropic API call failed: {e}")

    assert response.content is not None
    logger.info(f"Anthropic Tool Use Response: {response.content}")

    # Check if tool use was triggered
    tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
    if tool_use_blocks:
        logger.info(f"Tool use blocks: {len(tool_use_blocks)}")
        for block in tool_use_blocks:
            logger.info(f"  - {block.name}: {block.input}")


@pytest.mark.integration
def test_multi_step_agent_workflow(tracer_setup):
    """Test a multi-step agent workflow."""
    logger.info("Testing Multi-Step Agent Workflow")

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
        logger.info(f"Processing request: {request}")

        analysis = analyze_request(request)
        assert analysis["intent"] == "information_request"
        assert analysis["confidence"] == 0.95
        logger.info(f"Analysis: {analysis}")

        info = gather_information(analysis)
        assert info["weather_data"]["location"] == "San Francisco"
        assert info["weather_data"]["temperature"] == 72
        logger.info(f"Information gathered: {info}")

        response = generate_response(info)
        expected_response = (
            "The weather in San Francisco is sunny with a temperature of 72°F."
        )
        assert response == expected_response
        logger.info(f"Final response: {response}")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_agent_with_llm_calls(tracer_setup, openai_client):
    """Test agent that makes LLM calls."""
    logger.info("Testing Agent with LLM Calls")

    # Additional runtime validation for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OpenAI API key not available")

    # Validate API key format (basic check)
    if not api_key.startswith("sk-"):
        pytest.skip("Invalid OpenAI API key format")

    logger.info("OpenAI API key validated successfully")

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
            try:
                response = openai_client.chat.completions.create(
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
            except Exception as e:
                logger.error(f"OpenAI API call failed in LLM processing: {e}")
                raise

        # Test LLM processing within agent context
        query = "What is the capital of France?"
        try:
            result = process_with_llm(query)
            assert result is not None
            assert len(result) > 0
            logger.info(f"LLM Agent result: {result}")
        except Exception as e:
            pytest.fail(f"LLM processing failed: {e}")


@pytest.mark.integration
def test_nested_agent_calls(tracer_setup):
    """Test nested agent calls."""
    logger.info("Testing Nested Agent Calls")

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
        expected_result = "Parent received: Child processed: test data"
        assert result == expected_result
        logger.info(f"Nested agent result: {result}")


if __name__ == "__main__":
    logger.info("Starting Tools and Agents Tests")
    pytest.main([__file__, "-v"])
