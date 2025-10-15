# Noveum Trace SDK

[![CI](https://github.com/Noveum/noveum-trace/actions/workflows/ci.yml/badge.svg)](https://github.com/Noveum/noveum-trace/actions/workflows/ci.yml)
[![Release](https://github.com/Noveum/noveum-trace/actions/workflows/release.yml/badge.svg)](https://github.com/Noveum/noveum-trace/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/Noveum/noveum-trace/branch/main/graph/badge.svg)](https://codecov.io/gh/Noveum/noveum-trace)
[![PyPI version](https://badge.fury.io/py/noveum-trace.svg)](https://badge.fury.io/py/noveum-trace)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Simple, decorator-based tracing SDK for LLM applications and multi-agent systems.**

Noveum Trace provides an easy way to add observability to your LLM applications. With simple decorators, you can trace function calls, LLM interactions, agent workflows, and multi-agent coordination patterns.

## ✨ Key Features

- **🎯 Decorator-First API** - Add tracing with a single `@trace` decorator
- **🤖 Multi-Agent Support** - Built for multi-agent systems and workflows
- **☁️ Cloud Integration** - Send traces to Noveum platform or custom endpoints
- **🔌 Framework Agnostic** - Works with any Python LLM framework
- **🚀 Zero Configuration** - Works out of the box with sensible defaults
- **📊 Comprehensive Tracing** - Capture function calls, LLM interactions, and agent workflows
- **🔄 Flexible Approaches** - Decorators, and context managers

## 🚀 Quick Start

### Installation

```bash
pip install noveum-trace
```

### Basic Usage

```python
import noveum_trace

# Initialize the SDK
noveum_trace.init(
    api_key="your-api-key",
    project="my-llm-app"
)

# Trace any function
@noveum_trace.trace
def process_document(document_id: str) -> dict:
    # Your function logic here
    return {"status": "processed", "id": document_id}

# Trace LLM calls with automatic metadata capture
@noveum_trace.trace_llm
def call_openai(prompt: str) -> str:
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Trace agent workflows
@noveum_trace.trace_agent(agent_id="researcher")
def research_task(query: str) -> dict:
    # Agent logic here
    return {"findings": "...", "confidence": 0.95}
```

### Multi-Agent Example

```python
import noveum_trace

noveum_trace.init(
    api_key="your-api-key",
    project="multi-agent-system"
)

@noveum_trace.trace_agent(agent_id="orchestrator")
def orchestrate_workflow(task: str) -> dict:
    # Coordinate multiple agents
    research_result = research_agent(task)
    analysis_result = analysis_agent(research_result)
    return synthesis_agent(research_result, analysis_result)

@noveum_trace.trace_agent(agent_id="researcher")
def research_agent(task: str) -> dict:
    # Research implementation
    return {"data": "...", "sources": [...]}

@noveum_trace.trace_agent(agent_id="analyst")
def analysis_agent(data: dict) -> dict:
    # Analysis implementation
    return {"insights": "...", "metrics": {...}}
```

## 🏗️ Architecture

```
noveum_trace/
├── core/           # Core tracing primitives (Trace, Span, Context)
├── decorators/     # Decorator-based API (@trace, @trace_llm, etc.)
├── context_managers/ # Context managers for inline tracing
├── transport/      # HTTP transport and batch processing
├── agents/         # Multi-agent system support
├── streaming/      # Streaming LLM support
├── threads/        # Conversation thread management
└── utils/          # Utilities (exceptions, serialization, etc.)
```

## 🔧 Configuration

### Environment Variables

```bash
export NOVEUM_API_KEY="your-api-key"
export NOVEUM_PROJECT="your-project-name"
export NOVEUM_ENVIRONMENT="production"
```

### Programmatic Configuration

```python
import noveum_trace

# Basic configuration
noveum_trace.init(
    api_key="your-api-key",
    project="my-project",
    environment="production"
)

# Advanced configuration with transport settings
noveum_trace.init(
    api_key="your-api-key",
    project="my-project",
    environment="production",
    transport_config={
        "batch_size": 50,
        "batch_timeout": 2.0,
        "retry_attempts": 3,
        "timeout": 30
    },
    tracing_config={
        "sample_rate": 1.0,
        "capture_errors": True,
        "capture_stack_traces": False
    }
)
```

## 🔄 Context Managers - Inline Tracing

For scenarios where you need granular control or can't modify function signatures:

```python
import noveum_trace

def process_user_query(user_input: str) -> str:
    # Pre-processing (not traced)
    cleaned_input = user_input.strip().lower()

    # Trace just the LLM call
    with noveum_trace.trace_llm_call(model="gpt-4", provider="openai") as span:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": cleaned_input}]
        )

        # Add custom attributes
        span.set_attributes({
            "llm.input_tokens": response.usage.prompt_tokens,
            "llm.output_tokens": response.usage.completion_tokens
        })

    # Post-processing (not traced)
    return format_response(response.choices[0].message.content)

def multi_step_workflow(task: str) -> dict:
    results = {}

    # Trace agent operation
    with noveum_trace.trace_agent_operation(
        agent_type="planner",
        operation="task_planning"
    ) as span:
        plan = create_task_plan(task)
        span.set_attribute("plan.steps", len(plan.steps))
        results["plan"] = plan

    # Trace tool usage
    with noveum_trace.trace_operation("database_query") as span:
        data = query_database(plan.query)
        span.set_attributes({
            "query.results_count": len(data),
            "query.table": "tasks"
        })
        results["data"] = data

    return results
```

## 🧵 Thread Management

Track conversation threads and multi-turn interactions:

```python
from noveum_trace import ThreadContext

# Create and manage conversation threads
with ThreadContext(name="customer_support") as thread:
    thread.add_message("user", "Hello, I need help with my order")

    # LLM response within thread context
    with noveum_trace.trace_llm_call(model="gpt-4") as span:
        response = llm_client.chat.completions.create(...)
        thread.add_message("assistant", response.choices[0].message.content)
```

## 🌊 Streaming Support

Trace streaming LLM responses with real-time metrics:

```python
from noveum_trace import trace_streaming

def stream_openai_response(prompt: str):
    with trace_streaming(model="gpt-4", provider="openai") as manager:
        stream = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                manager.add_token(content)
                yield content

        # Streaming metrics are automatically captured
```

## 🔗 LangChain Integration

Noveum Trace provides seamless integration with LangChain and LangGraph applications through a simple callback handler.

### Quick Setup

```python
from noveum_trace.integrations import NoveumTraceCallbackHandler
from langchain_openai import ChatOpenAI

# Initialize Noveum Trace
import noveum_trace
noveum_trace.init(project="my-langchain-app", api_key="your-api-key")

# Create callback handler
handler = NoveumTraceCallbackHandler()

# Add to your LangChain components
llm = ChatOpenAI(callbacks=[handler])
response = llm.invoke("What is the capital of France?")
```

### Advanced Features

**Manual Trace Control**: For complex workflows, manually control trace lifecycle:

```python
# Start trace manually
handler.start_trace("my-workflow")

# Your LangChain operations
llm = ChatOpenAI(callbacks=[handler])
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])

# End trace manually
handler.end_trace()
```

**Custom Parent Relationships**: Use `parent_name` to create explicit span hierarchies:

```python
# Parent span with custom name
llm = ChatOpenAI(
    callbacks=[handler],
    metadata={"noveum": {"name": "parent_llm"}}
)

# Child span that references parent
chain = LLMChain(
    llm=llm,
    callbacks=[handler],
    metadata={"noveum": {"parent_name": "parent_llm"}}
)
```

**LangGraph Routing Tracking**: Track routing decisions in LangGraph workflows:

```python
def route_function(state, config):
    decision = "next_node" if state["count"] < 5 else "finish"
    
    # Emit routing event
    if config and config.get("callbacks"):
        callbacks = config["callbacks"]
        callbacks.on_custom_event("langgraph.routing_decision", {
            "source_node": "current_node",
            "target_node": decision,
            "reason": f"Count {state['count']} {'< 5' if state['count'] < 5 else '>= 5'}"
        })
    
    return decision
```

### What Gets Traced

- **LLM Calls**: Model, prompts, responses, token usage
- **Chains**: Input/output flow, execution steps  
- **Agents**: Decision-making, tool usage, reasoning
- **Tools**: Function calls, inputs, outputs
- **LangGraph Nodes**: Graph execution, node transitions
- **Routing Decisions**: Conditional routing logic and decisions

For complete integration details, see the [LangChain Integration Guide](docs/LANGCHAIN_INTEGRATION.md).

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=noveum_trace --cov-report=html

# Run specific test categories
pytest -m llm
pytest -m agent
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Noveum/noveum-trace.git
cd noveum-trace

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run examples
python docs/examples/basic_usage.py
```

## 📖 Examples

Check out the [examples](docs/examples/) directory for complete working examples:

- [Basic Usage](docs/examples/basic_usage.py) - Simple function tracing
- [Agent Workflow](docs/examples/agent_workflow_example.py) - Multi-agent coordination
- [Flexible Tracing](docs/examples/flexible_tracing_example.py) - Context managers and inline tracing
- [Streaming Example](docs/examples/streaming_example.py) - Real-time streaming support
- [Multimodal Examples](docs/examples/multimodal_examples.py) - Image, audio, and video tracing
- [LangGraph Routing](docs/examples/langgraph_routing_example.py) - LangGraph routing decision tracking

## 🚀 Advanced Usage

### Manual Trace Creation

```python
# Create traces manually for full control
client = noveum_trace.get_client()

with client.create_contextual_trace("custom_workflow") as trace:
    with client.create_contextual_span("step_1") as span1:
        # Step 1 implementation
        span1.set_attributes({"step": 1, "status": "completed"})

    with client.create_contextual_span("step_2") as span2:
        # Step 2 implementation
        span2.set_attributes({"step": 2, "status": "completed"})
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

- [GitHub Issues](https://github.com/Noveum/noveum-trace/issues)
- [Documentation](https://github.com/Noveum/noveum-trace/tree/main/docs)
- [Examples](https://github.com/Noveum/noveum-trace/tree/main/examples)

---

**Built by the Noveum Team**
