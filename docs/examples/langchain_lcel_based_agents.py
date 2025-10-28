"""
LangChain LCEL-Based Agents Example

This example demonstrates how to use Noveum Trace with LangChain's LCEL-based agents:
- create_react_agent
- create_openai_functions_agent
- create_tool_calling_agent
- create_structured_chat_agent
- create_json_chat_agent

All of these agents use RunnableSequence internally and work seamlessly with
Noveum Trace callback handlers for comprehensive observability.
"""

import os

from dotenv import load_dotenv

import noveum_trace
from noveum_trace.integrations import NoveumTraceCallbackHandler

load_dotenv()

noveum_trace.init(
    project=os.getenv("NOVEUM_PROJECT", "find-lcel-agents"),
    api_key=os.getenv("NOVEUM_API_KEY"),
    environment="development",
)


def example_create_react_agent():
    """Example: Using create_react_agent with Noveum Trace."""
    print("\n=== Example: LangChain create_react_agent (LCEL) ===")

    try:
        from langchain import hub
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI

        callback_handler = NoveumTraceCallbackHandler()

        def tool_func(x: str) -> str:
            return f"Result: {x}"

        tool = Tool(name="TestTool", func=tool_func, description="Test tool")
        llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, callbacks=[callback_handler]
        )

        # Get ReAct prompt from hub
        prompt = hub.pull("hwchase17/react")

        # Create LCEL-based agent
        agent = create_react_agent(llm, [tool], prompt)

        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            callbacks=[callback_handler],
            verbose=True,
        )

        agent_executor.invoke({"input": "Use TestTool with 'test data'"})

        print("✓ Example completed successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


def example_create_openai_functions_agent():
    """Example: Using create_openai_functions_agent with Noveum Trace."""
    print("\n=== Example: LangChain create_openai_functions_agent (LCEL) ===")

    try:
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI

        callback_handler = NoveumTraceCallbackHandler()

        def tool_func(x: str) -> str:
            return f"Result: {x}"

        tool = Tool(name="TestTool", func=tool_func, description="Test tool")
        llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, callbacks=[callback_handler]
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Create LCEL-based OpenAI Functions agent
        agent = create_openai_functions_agent(llm, [tool], prompt)

        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            callbacks=[callback_handler],
            verbose=True,
        )

        agent_executor.invoke({"input": "Use TestTool with 'test data'"})

        print("✓ Example completed successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


def example_create_tool_calling_agent():
    """Example: Using create_tool_calling_agent with Noveum Trace."""
    print("\n=== Example: LangChain create_tool_calling_agent (LCEL) ===")

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI

        callback_handler = NoveumTraceCallbackHandler()

        def tool_func(x: str) -> str:
            return f"Result: {x}"

        tool = Tool(name="TestTool", func=tool_func, description="Test tool")
        llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, callbacks=[callback_handler]
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Create LCEL-based tool calling agent
        agent = create_tool_calling_agent(llm, [tool], prompt)

        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            callbacks=[callback_handler],
            verbose=True,
        )

        agent_executor.invoke({"input": "Use TestTool with 'test data'"})

        print("✓ Example completed successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


def example_create_structured_chat_agent():
    """Example: Using create_structured_chat_agent with Noveum Trace."""
    print("\n=== Example: LangChain create_structured_chat_agent (LCEL) ===")

    try:
        from langchain.agents import AgentExecutor, create_structured_chat_agent
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI

        callback_handler = NoveumTraceCallbackHandler()

        def tool_func(x: str) -> str:
            return f"Result: {x}"

        tool = Tool(name="TestTool", func=tool_func, description="Test tool")
        llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, callbacks=[callback_handler]
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Create LCEL-based structured chat agent
        agent = create_structured_chat_agent(llm, [tool], prompt)

        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            callbacks=[callback_handler],
            verbose=True,
        )

        agent_executor.invoke({"input": "Use TestTool with 'test data'"})

        print("✓ Example completed successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


def example_create_json_chat_agent():
    """Example: Using create_json_chat_agent with Noveum Trace."""
    print("\n=== Example: LangChain create_json_chat_agent (LCEL) ===")

    try:
        from langchain.agents import AgentExecutor, create_json_chat_agent
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI

        callback_handler = NoveumTraceCallbackHandler()

        def tool_func(x: str) -> str:
            return f"Result: {x}"

        tool = Tool(name="TestTool", func=tool_func, description="Test tool")
        llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, callbacks=[callback_handler]
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Create LCEL-based JSON chat agent
        agent = create_json_chat_agent(llm, [tool], prompt)

        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            callbacks=[callback_handler],
            verbose=True,
        )

        agent_executor.invoke({"input": "Use TestTool with 'test data'"})

        print("✓ Example completed successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


def main():
    print("=" * 80)
    print("LANGCHAIN LCEL-BASED AGENTS EXAMPLES")
    print("=" * 80)
    print("\nThis demonstrates various LangChain LCEL-based agents integrated")
    print("with Noveum Trace for comprehensive observability.\n")

    examples = [
        example_create_react_agent,
        example_create_openai_functions_agent,
        example_create_tool_calling_agent,
        example_create_structured_chat_agent,
        example_create_json_chat_agent,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nExample error: {e}")

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nView traces in your Noveum dashboard to see the detailed")
    print("execution flow and performance metrics.")

    noveum_trace.flush()


if __name__ == "__main__":
    main()
