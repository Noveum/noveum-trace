# LangChain Integration Guide

This guide shows how to integrate Noveum Trace with your LangChain/Langgraph applications to automatically track LLM calls, chains, agents, tools, as well as routing decisions.

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

# Create callback handler with default settings
callback_handler = NoveumTraceCallbackHandler()

# Or with LangChain parent ID resolution for complex workflows
callback_handler = NoveumTraceCallbackHandler(use_langchain_assigned_parent=True)

# Add to your LangChain components
llm = ChatOpenAI(callbacks=[callback_handler])
```

#### Callback Handler Options

- `use_langchain_assigned_parent=False`: Use context-based parent assignment (default)
- `use_langchain_assigned_parent=True`: Use LangChain's parent_run_id for parent relationships (recommended for LangGraph)

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

## Advanced Features

### Manual Trace Control

For advanced use cases, you can manually control trace lifecycle:

```python
from noveum_trace import NoveumTraceCallbackHandler

# Create callback handler
handler = NoveumTraceCallbackHandler()

# Manually start a trace
handler.start_trace("my-custom-trace")

# Your LangChain operations here
llm = ChatOpenAI(callbacks=[handler])
response = llm.invoke("Hello world")

# Manually end the trace
handler.end_trace()
```

### Prioritizing Custom Parent Relationships

SDK supports fine-grained control over how parent spans are assigned. When initializing the callback handler, set `prioritize_manually_assigned_parents=True` and `use_langchain_assigned_parent=True` to prioritize manually assigned `parent_name` metadata over LangChain's parent_run_id. This is helpful in advanced graph and agent flows where parent-child relationships need to be explicitly managed:

```python
handler = NoveumTraceCallbackHandler(
    use_langchain_assigned_parent=True,
    prioritize_manually_assigned_parents=True
)
```

See [`examples/langgraph_custom_parent_prioritzed_example.py`](examples/langgraph_custom_parent_prioritzed_example.py) and [`examples/langgraph_custom_parent_example.py`](examples/langgraph_custom_parent_example.py) for working examples showing both parent ID strategies.

### Asynchronous LLM Invocation

SDK supports asynchronous tracing using `await llm.ainvoke(...)` or similar async calls within traced LangChain components. This is fully supported in both callback handler and graph tracing modes.

See [`examples/langgraph_asynchronous_execution.py`](examples/langgraph_asynchronous_execution.py) for end-to-end async examples.

### Custom Parent Span Relationships

You can explicitly set parent-child relationships between spans using custom names:

```python
# Create a parent span with custom name
llm = ChatOpenAI(
    callbacks=[handler],
    metadata={"noveum": {"name": "parent_llm"}}
)

# Create child spans that reference the parent
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    callbacks=[handler],
    metadata={"noveum": {"parent_name": "parent_llm"}}
)
```

#### Metadata Structure

The `metadata` parameter supports a `noveum` configuration object:

```python
metadata = {
    "noveum": {
        "name": "custom_span_name",        # Custom name for this span
        "parent_name": "parent_span_name"  # Name of parent span to attach to
    }
}
```

**Note**: When using custom parent relationships, you must manually control trace lifecycle with `start_trace()` and `end_trace()`.

### LangChain Parent ID Support

For LangGraph and complex workflows, you can use LangChain's built-in parent run IDs:

```python
# Enable LangChain parent ID resolution
handler = NoveumTraceCallbackHandler(use_langchain_assigned_parent=True)

# LangChain will automatically resolve parent relationships
# based on parent_run_id in the callback events
```

### LangGraph Routing Decision Tracking

Track routing decisions in LangGraph workflows as separate spans:

```python
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

def route_function(state, config):
    """Routing function that emits routing events."""
    
    # Make routing decision
    decision = "next_node" if state["count"] < 5 else "finish"
    
    # Emit routing event (if callbacks available)
    if config and config.get("callbacks"):
        callbacks = config["callbacks"]
        
        # Normalize callbacks into an iterable
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]
        
        # Iterate over each callback handler
        for handler in callbacks:
            if hasattr(handler, 'on_custom_event'):
                handler.on_custom_event(
                    "langgraph.routing_decision",
                    {
                        "source_node": "current_node",
                        "target_node": decision,
                        "decision": decision,
                        "reason": f"Count {state['count']} {'< 5' if state['count'] < 5 else '>= 5'}",
                        "confidence": 0.9,
                        "state_snapshot": state,
                    }
                )
    
    return decision

# Create graph with routing
workflow = StateGraph(State)
workflow.add_node("process", process_node)
workflow.add_node("finish", finish_node)
workflow.add_conditional_edges(
    "process",
    route_function,
    {"next_node": "process", "finish": "finish"}
)

# Run with callback handler
app = workflow.compile()
result = app.invoke(
    {"count": 0},
    config={"callbacks": [handler]}
)
```

## What Gets Traced

The integration automatically captures:

- **LLM Calls**: Model, prompts, responses, token usage
- **Chains**: Input/output flow, execution steps
- **Agents**: Decision-making, tool usage, reasoning
- **Tools**: Function calls, inputs, outputs
- **Retrievers**: Queries, document results
- **LangGraph Nodes**: Graph execution, node transitions
- **Routing Decisions**: Conditional routing logic and decisions

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
- **Graceful fallbacks** for parent span resolution
- **Warning logs** for missing parent spans in custom relationships

### Troubleshooting

#### Custom Parent Relationships Not Working

If custom parent relationships aren't working:

1. Ensure you're using manual trace control:
   ```python
   handler.start_trace("my-trace")
   # ... your operations ...
   handler.end_trace()
   ```

2. Check that parent spans are created before child spans:
   ```python
   # Parent must be created first
   parent_llm = ChatOpenAI(
       callbacks=[handler],
       metadata={"noveum": {"name": "parent"}}
   )
   
   # Then child can reference it
   child_chain = LLMChain(
       llm=parent_llm,
       callbacks=[handler],
       metadata={"noveum": {"parent_name": "parent"}}
   )
   ```

3. Verify span names match exactly (case-sensitive)

#### LangGraph Routing Not Appearing

If routing decisions aren't being tracked:

1. Ensure you're using the correct event name: `"langgraph.routing_decision"`
2. Check that callbacks are properly passed to the graph:
   ```python
   result = app.invoke(
       state,
       config={"callbacks": [handler]}  # Make sure callbacks are in config
   )
   ```
3. Verify the routing function has access to the config parameter

## Viewing Traces

Once your application is running with tracing enabled, you can view the captured data in the Noveum dashboard:

1. Go to your [Noveum Dashboard](https://app.noveum.ai)
2. Navigate to your project
3. View traces, spans, and detailed metadata
4. Analyze performance and debug issues

## LangGraph Integration

### Complete LangGraph Example

For a complete working example of LangGraph routing decision tracking, see:

```python
# See docs/examples/langgraph_routing_example.py for full implementation
from docs.examples.langgraph_routing_example import run_counter_example

# Run the example
result = run_counter_example()
```

### Routing Decision Attributes

When you emit routing decisions, the following attributes are automatically captured:

- `routing.source_node`: The node making the routing decision
- `routing.target_node`: The destination node
- `routing.decision`: The routing decision value
- `routing.reason`: Human-readable reason for the decision
- `routing.confidence`: Confidence score (0.0 to 1.0)
- `routing.state_snapshot`: State at the time of routing
- `routing.alternatives`: Other possible routing options
- `routing.tool_scores`: Tool selection scores (if applicable)

## Next Steps

- Explore the [examples directory](examples/) for more complex use cases
- Check out [langchain_integration_example.py](examples/langchain_integration_example.py) for complete working examples
- See [langgraph_routing_example.py](examples/langgraph_routing_example.py) for LangGraph routing tracking
