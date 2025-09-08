"""
LangChain Integration Example for Noveum Trace SDK.

This example demonstrates how to use the NoveumTraceCallbackHandler to automatically
trace LangChain operations including LLM calls, chains, agents, tools, and retrieval.

Prerequisites:
    pip install noveum-trace[langchain]
    pip install langchain langchain-openai langchain-community

Environment Variables:
    NOVEUM_API_KEY: Your Noveum API key
    OPENAI_API_KEY: Your OpenAI API key (for LLM examples)
"""

import os

import noveum_trace
from noveum_trace import NoveumTraceCallbackHandler
from dotenv import load_dotenv

load_dotenv()


def example_basic_llm_tracing():
    """Example: Basic LLM call tracing."""
    print("=== Basic LLM Tracing ===")

    try:
        from langchain_openai import ChatOpenAI

        # Initialize Noveum Trace with batch size 1
        noveum_trace.init(
            project="langchain-integration-demo",
            environment="development",
            endpoint="https://noveum.free.beeceptor.com",
            transport_config={"batch_size": 1, "batch_timeout": 5.0},
        )

        # Create callback handler
        callback_handler = NoveumTraceCallbackHandler()

        # Create LLM with callback
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.7, callbacks=[callback_handler]
        )

        # Make LLM call - this will be automatically traced
        response = llm.invoke("What is the capital of France?")
        print(f"Response: {response.content}")

    except ImportError:
        print("Skipping LLM example - langchain-openai not installed")
    except Exception as e:
        print(f"Error in LLM example: {e}")


def example_chain_tracing():
    """Example: Chain tracing with multiple steps."""
    print("\n=== Chain Tracing ===")

    try:
        from langchain_openai import ChatOpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        # Create callback handler
        callback_handler = NoveumTraceCallbackHandler()

        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["topic"], template="Write a brief summary about {topic}:"
        )

        # Create LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.5, callbacks=[callback_handler]
        )

        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt, callbacks=[callback_handler])

        # Run chain - this will create a trace with nested spans
        result = chain.run(topic="artificial intelligence")
        print(f"Chain result: {result[:100]}...")

    except ImportError:
        print("Skipping chain example - required packages not installed")
    except Exception as e:
        print(f"Error in chain example: {e}")


def example_tool_usage():
    """Example: Tool usage tracing."""
    print("\n=== Tool Usage Tracing ===")

    try:
        from langchain.tools import Tool
        from langchain.agents import initialize_agent, AgentType
        from langchain_openai import ChatOpenAI

        # Create callback handler
        callback_handler = NoveumTraceCallbackHandler()

        # Define custom tools
        def calculator(expression: str) -> str:
            """Simple calculator tool."""
            try:
                result = eval(expression)
                return f"The result is: {result}"
            except Exception as e:
                return f"Error: {str(e)}"

        # Create tools
        tools = [
            Tool(
                name="Calculator",
                func=calculator,
                description="Use this to perform mathematical calculations",
            )
        ]

        # Create LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0, callbacks=[callback_handler]
        )

        # Create agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            callbacks=[callback_handler],
            verbose=True,
        )

        # Use agent with tools
        result = agent.run("Calculate 15 * 23")
        print(f"Agent result: {result}")

    except ImportError:
        print("Skipping tool example - required packages not installed")
    except Exception as e:
        print(f"Error in tool example: {e}")


def example_error_handling():
    """Example: Error handling in tracing."""
    print("\n=== Error Handling ===")

    try:
        from langchain_openai import ChatOpenAI

        # Create callback handler
        callback_handler = NoveumTraceCallbackHandler()

        # Create LLM with invalid API key to trigger error
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key="invalid-key",
            callbacks=[callback_handler],
        )

        try:
            # This should fail and be traced as an error
            llm.invoke("This will fail")
        except Exception as e:
            print(f"Expected error occurred: {type(e).__name__}")
            print("Error was traced and recorded in span")

    except ImportError:
        print("Skipping error handling example - langchain-openai not installed")
    except Exception as e:
        print(f"Error in error handling example: {e}")


def main():
    """Run all examples."""
    print("Noveum Trace - LangChain Integration Examples")
    print("=" * 50)

    # Check if API keys are set
    if not os.getenv("NOVEUM_API_KEY"):
        print("Warning: NOVEUM_API_KEY not set. Using mock mode.")

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may fail.")

    print()

    # Run examples
    example_basic_llm_tracing()
    example_chain_tracing()
    example_tool_usage()
    example_error_handling()

    print("\n=== Examples Complete ===")
    print("Check your Noveum dashboard to see the traced operations!")

    # Flush any pending traces
    noveum_trace.flush()


if __name__ == "__main__":
    main()
