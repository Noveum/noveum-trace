"""
Simple LangChain Agent Example

This example shows how to create a LangChain agent exactly like the pattern
shown in the docs/examples/langchain_integration_example.py file.
"""

from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler
import os
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# Import our callback handler
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculator(expression: str) -> str:
    """Simple calculator tool."""
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def next_turn(message: str = "") -> str:
    """Tool to pass control to user."""
    return f"Control passed to user. Message: {message}"


def escalate_to_human(reason: str = "Complex query") -> str:
    """Tool to escalate to human support."""
    return f"ESCALATED TO HUMAN SUPPORT. Reason: {reason}. Please wait for a human customer support representative to get back to you."


def end_conversation(summary: str = "Query resolved") -> str:
    """Tool to end the conversation."""
    return f"CONVERSATION ENDED. Summary: {summary}"


def main():
    """Run the simple agent example."""
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()

    # Create tools exactly like the example
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Use this to perform mathematical calculations",
        ),
        Tool(
            name="next_turn",
            func=next_turn,
            description="Use this when you need the user to provide more information or respond to a question",
        ),
        Tool(
            name="escalate_to_human",
            func=escalate_to_human,
            description="Use this when the query is too complex or you cannot resolve the user's issue",
        ),
        Tool(
            name="end_conversation",
            func=end_conversation,
            description="Use this when the user's query has been fully resolved and the conversation can be ended",
        )
    ]

    # Create LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        callbacks=[callback_handler]
    )

    # Create agent exactly like the example
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[callback_handler],
        verbose=True,
        handle_parsing_errors=True
    )

    # Run the agent
    print("🤖 Simple LangChain Agent Example")
    print("=" * 40)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye! 👋")
            break

        if not user_input:
            continue

        try:
            response = agent.run(input=user_input)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
