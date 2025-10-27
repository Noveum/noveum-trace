"""
LangChain Agent with Chains and Tools Example

This example demonstrates a complete LangChain agent system that uses:
1. Chains - for structured LLM processing
2. Tools - for executing specific operations
3. Agent - for orchestrating tool selection and execution

Use Case: A research assistant that can search, summarize, and analyze information.

Prerequisites:
    pip install noveum-trace[langchain]
    pip install langchain langchain-openai langchain-community

Environment Variables:
    NOVEUM_API_KEY: Your Noveum API key
    OPENAI_API_KEY: Your OpenAI API key
"""

import os
import time
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import noveum_trace
from noveum_trace.integrations import NoveumTraceCallbackHandler

load_dotenv()

# =============================================================================
# TOOLS DEFINITION
# =============================================================================


def web_search_tool(query: str) -> str:
    """
    Simulate a web search tool.
    In production, this would call a real search API (Tavily, SerpAPI, etc.)
    """
    print(f"🔍 Searching for: {query}")
    time.sleep(0.5)  # Simulate API call
    
    # Simulated search results based on query
    search_results = {
        "langchain": """
        LangChain is a framework for developing applications powered by language models.
        Key components: Chains, Agents, Memory, Prompts, and Tools.
        Used for building chatbots, question-answering systems, and AI agents.
        """,
        "transformers": """
        Transformers are a type of neural network architecture introduced in 2017.
        Based on attention mechanisms, they revolutionized NLP.
        Popular models: BERT, GPT, T5, and their variants.
        """,
        "python": """
        Python is a high-level, interpreted programming language.
        Known for its simplicity and extensive ecosystem.
        Popular in data science, web development, and AI.
        """,
    }
    
    # Find matching result or return default
    for key, content in search_results.items():
        if key.lower() in query.lower():
            return f"Search Results for '{query}':\n{content}"
    
    return f"Search Results for '{query}':\nGeneral information available."


def calculator_tool(expression: str) -> str:
    """
    Calculator tool for mathematical operations.
    """
    print(f"🧮 Calculating: {expression}")
    try:
        # Safety: Only allow basic math operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def text_analyzer_tool(text: str) -> dict[str, Any]:
    """
    Analyze text and return statistics.
    This tool demonstrates returning structured data.
    """
    print(f"📊 Analyzing text ({len(text)} chars)...")
    time.sleep(0.3)
    
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": sentences if sentences > 0 else 1,
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
    }


# =============================================================================
# CHAINS DEFINITION
# =============================================================================


def create_summarization_chain(llm: ChatOpenAI, callback_handler: NoveumTraceCallbackHandler) -> LLMChain:
    """
    Create a chain for summarizing text.
    This demonstrates how chains are used within an agent system.
    """
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Summarize the following text in 2-3 sentences:
        
        Text: {text}
        
        Summary:
        """
    )
    
    return LLMChain(
        llm=llm,
        prompt=prompt,
        callbacks=[callback_handler],
        verbose=True
    )


def create_analysis_chain(llm: ChatOpenAI, callback_handler: NoveumTraceCallbackHandler) -> LLMChain:
    """
    Create a chain for analyzing information quality.
    """
    prompt = PromptTemplate(
        input_variables=["information"],
        template="""
        Analyze the following information and provide:
        1. Main topics covered
        2. Quality assessment (high/medium/low)
        3. Key insights
        
        Information: {information}
        
        Analysis:
        """
    )
    
    return LLMChain(
        llm=llm,
        prompt=prompt,
        callbacks=[callback_handler],
        verbose=True
    )


# =============================================================================
# AGENT SETUP
# =============================================================================


def create_research_agent(callback_handler: NoveumTraceCallbackHandler) -> AgentExecutor:
    """
    Create a ReAct agent with multiple tools and chains.
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        callbacks=[callback_handler]
    )
    
    # Create chains
    summarization_chain = create_summarization_chain(llm, callback_handler)
    analysis_chain = create_analysis_chain(llm, callback_handler)
    
    # Create chain-based tools
    def summarize_tool_func(text: str) -> str:
        """Use the summarization chain to summarize text."""
        return summarization_chain.run(text=text)
    
    def analyze_tool_func(info: str) -> str:
        """Use the analysis chain to analyze information."""
        return analysis_chain.run(information=info)
    
    # Define all tools
    tools = [
        Tool(
            name="WebSearch",
            func=web_search_tool,
            description="Search the web for information about a topic. Input should be a search query string."
        ),
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Perform mathematical calculations. Input should be a mathematical expression like '25 * 4 + 10'."
        ),
        Tool(
            name="TextAnalyzer",
            func=text_analyzer_tool,
            description="Analyze text and get statistics (word count, sentence count, etc.). Input should be the text to analyze."
        ),
        Tool(
            name="Summarizer",
            func=summarize_tool_func,
            description="Summarize long text into 2-3 sentences. Input should be the text to summarize."
        ),
        Tool(
            name="InformationAnalyzer",
            func=analyze_tool_func,
            description="Analyze information quality and extract key insights. Input should be the information to analyze."
        ),
    ]
    
    # Create ReAct agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful research assistant. You have access to several tools to help answer questions.

Available tools:
{tools}

Tool names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""),
    ])
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[callback_handler],
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )
    
    return agent_executor


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


def example_simple_query():
    """
    Example 1: Simple query that uses web search and summarization.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Query with Search and Summarization")
    print("=" * 80 + "\n")
    
    # Initialize Noveum Trace
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "langchain-example"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment="development",
    )
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create agent
    agent = create_research_agent(callback_handler)
    
    # Run query
    query = "Search for information about LangChain and summarize it."
    print(f"Query: {query}\n")
    
    try:
        result = agent.invoke({"input": query})
        print("\n" + "=" * 80)
        print(f"Final Answer: {result['output']}")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"Error: {e}")


def example_complex_query():
    """
    Example 2: Complex query that uses multiple tools and chains.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Complex Query with Multiple Tools")
    print("=" * 80 + "\n")
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create agent
    agent = create_research_agent(callback_handler)
    
    # Run complex query
    query = """
    I need you to:
    1. Search for information about transformers in AI
    2. Analyze the search results
    3. Calculate how many words are in the search results (multiply by 2 for some reason)
    4. Give me a final summary of what you found
    """
    print(f"Query: {query}\n")
    
    try:
        result = agent.invoke({"input": query})
        print("\n" + "=" * 80)
        print(f"Final Answer: {result['output']}")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"Error: {e}")


def example_math_and_analysis():
    """
    Example 3: Query combining calculation with text analysis.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Math and Text Analysis Combination")
    print("=" * 80 + "\n")
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create agent
    agent = create_research_agent(callback_handler)
    
    # Run query
    query = """
    Calculate the result of (150 * 3) + (200 / 4), then search for information 
    about Python programming language, analyze the text statistics, and tell me 
    if the number of words in the search result is greater than the calculation result.
    """
    print(f"Query: {query}\n")
    
    try:
        result = agent.invoke({"input": query})
        print("\n" + "=" * 80)
        print(f"Final Answer: {result['output']}")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 80)
    print("LangChain Agent with Chains and Tools - Noveum Trace Integration")
    print("=" * 80)
    
    # Check environment variables
    if not os.getenv("NOVEUM_API_KEY"):
        print("⚠️  Warning: NOVEUM_API_KEY not set")
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
    
    # Run examples
    try:
        example_simple_query()
        time.sleep(2)
        
        example_complex_query()
        time.sleep(2)
        
        example_math_and_analysis()
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
    
    # Flush traces
    print("\n" + "=" * 80)
    print("Flushing traces...")
    noveum_trace.flush()
    
    print("\n✅ All examples completed!")
    print("\nCheck your Noveum Trace dashboard to see:")
    print("  • Agent spans showing the reasoning process")
    print("  • Tool spans for each tool invocation (WebSearch, Calculator, etc.)")
    print("  • Chain spans for summarization and analysis")
    print("  • LLM spans for each model call")
    print("  • Complete trace hierarchy showing agent decision flow")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

