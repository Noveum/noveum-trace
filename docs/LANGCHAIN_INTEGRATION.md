# LangChain Integration Guide

This guide shows how to integrate Noveum Trace with your LangChain applications to automatically track LLM calls, chains, agents, and tools.

## Quick Start

### 1. Install Dependencies

```bash
pip install noveum-trace[langchain]
```

### 2. Initialize Noveum Trace

```python
import noveum_trace

# Initialize with your project details
noveum_trace.init(
    project="my-langchain-app",
    api_key="your-noveum-api-key"  # or set NOVEUM_API_KEY env var
)
```

### 3. Add Callback Handler

The callback handler is what makes tracing work. It listens to LangChain events and automatically creates traces. You need to add it to each LangChain component you want to trace.

```python
from noveum_trace import NoveumTraceCallbackHandler
from langchain_openai import ChatOpenAI

# Create callback handler
callback_handler = NoveumTraceCallbackHandler()

# Add to your LangChain components
llm = ChatOpenAI(callbacks=[callback_handler])
```

## Basic Examples

### LLM Tracing

Start with simple LLM calls. The callback handler will automatically capture the model used, your prompt, the response, and token usage.

```python
from langchain_openai import ChatOpenAI
from noveum_trace import NoveumTraceCallbackHandler

callback_handler = NoveumTraceCallbackHandler()
llm = ChatOpenAI(callbacks=[callback_handler])

# This call is automatically traced
response = llm.invoke("What is the capital of France?")
```

### Chain Tracing

Chains combine multiple steps. The tracer will show you the complete flow from input to output, including any nested LLM calls within the chain.

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Add callback to chain
chain = LLMChain(
    llm=llm, 
    prompt=PromptTemplate(input_variables=["topic"], template="Tell me about {topic}"),
    callbacks=[callback_handler]
)

# Chain execution is traced with nested spans
result = chain.run(topic="artificial intelligence")
```

### Agent with Tools

Agents can use tools to perform actions. The tracer will show you the agent's decision-making process, which tools it calls, and the results of each tool execution.

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

def calculator(expression: str) -> str:
    return str(eval(expression))

tools = [Tool(name="Calculator", func=calculator, description="Calculate math expressions")]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[callback_handler]
)

# Agent execution and tool usage are fully traced
result = agent.run("Calculate 15 * 23")
```

## What Gets Traced

The integration automatically captures:

- **LLM Calls**: Model, prompts, responses, token usage
- **Chains**: Input/output flow, execution steps
- **Agents**: Decision-making, tool usage, reasoning
- **Tools**: Function calls, inputs, outputs
- **Retrievers**: Queries, document results

## Configuration

You can configure Noveum Trace using environment variables or programmatically in your code.

### Environment Variables

Set these in your environment or `.env` file for automatic configuration:

```bash
export NOVEUM_API_KEY="your-api-key"
export NOVEUM_PROJECT="my-project"
```

### Programmatic Configuration

For more control, configure directly in your code:

```python
noveum_trace.init(
    project="my-app",
    api_key="your-key",
    environment="production",
    transport_config={
        "batch_size": 10,
        "batch_timeout": 5.0
    }
)
```

## Error Handling

The integration includes robust error handling:

- **Automatic error recording** in traces
- **No impact** on your LangChain application performance

## Viewing Traces

Once your application is running with tracing enabled, you can view the captured data in the Noveum dashboard:

1. Go to your [Noveum Dashboard](https://app.noveum.ai)
2. Navigate to your project
3. View traces, spans, and detailed metadata
4. Analyze performance and debug issues

## Next Steps

- Explore the [examples directory](examples/) for more complex use cases
- Check out [langchain_integration_example.py](examples/langchain_integration_example.py) for complete working examples
