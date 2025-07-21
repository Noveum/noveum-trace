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
- **🔄 Flexible Approaches** - Decorators, context managers, and auto-instrumentation

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
├── integrations/   # Framework integrations (OpenAI, etc.)
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

## 🎯 Available Decorators

### @trace - General Purpose Tracing

```python
@noveum_trace.trace
def my_function(arg1: str, arg2: int) -> dict:
    return {"result": f"{arg1}_{arg2}"}

# With options
@noveum_trace.trace(capture_performance=True, capture_args=True)
def expensive_function(data: list) -> dict:
    # Function implementation
    return {"processed": len(data)}
```

### @trace_llm - LLM Call Tracing

```python
@noveum_trace.trace_llm
def call_llm(prompt: str) -> str:
    # LLM call implementation
    return response

# With provider specification
@noveum_trace.trace_llm(provider="openai", capture_tokens=True)
def call_openai(prompt: str) -> str:
    # OpenAI specific implementation
    return response
```

### @trace_agent - Agent Workflow Tracing

```python
# Required: agent_id parameter
@noveum_trace.trace_agent(agent_id="my_agent")
def agent_function(task: str) -> dict:
    # Agent implementation
    return result

# With full configuration
@noveum_trace.trace_agent(
    agent_id="researcher",
    role="information_gatherer",
    capabilities=["web_search", "document_analysis"]
)
def research_agent(query: str) -> dict:
    # Research implementation
    return {"findings": "...", "sources": [...]}
```

### @trace_tool - Tool Usage Tracing

```python
@noveum_trace.trace_tool
def search_web(query: str) -> list:
    # Tool implementation
    return results

# With tool specification
@noveum_trace.trace_tool(tool_name="web_search", tool_type="api")
def search_api(query: str) -> list:
    # API search implementation
    return search_results
```

### @trace_retrieval - Retrieval Operation Tracing

```python
@noveum_trace.trace_retrieval
def retrieve_documents(query: str) -> list:
    # Retrieval implementation
    return documents

# With retrieval configuration
@noveum_trace.trace_retrieval(
    retrieval_type="vector_search",
    index_name="documents",
    capture_scores=True
)
def vector_search(query: str, top_k: int = 5) -> list:
    # Vector search implementation
    return results
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

## 🤖 Auto-Instrumentation

Automatically trace existing code without modifications:

```python
import noveum_trace

# Initialize with auto-instrumentation
noveum_trace.init(
    api_key="your-api-key",
    project="my-project",
    auto_instrument=["openai", "anthropic", "langchain"]
)

# Now all OpenAI calls are automatically traced
import openai
client = openai.OpenAI()

# This call is automatically traced
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

## 🔌 Framework Integrations

### OpenAI Integration

```python
import noveum_trace
import openai

# Initialize tracing
noveum_trace.init(api_key="your-key", project="openai-app")

@noveum_trace.trace_llm
def chat_with_openai(message: str) -> str:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

# Or use context managers for existing code
def existing_openai_function(prompt: str) -> str:
    with noveum_trace.trace_llm_call(model="gpt-4", provider="openai") as span:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        # Capture usage metrics
        if response.usage:
            span.set_attributes({
                "llm.input_tokens": response.usage.prompt_tokens,
                "llm.output_tokens": response.usage.completion_tokens,
                "llm.total_tokens": response.usage.total_tokens
            })

        return response.choices[0].message.content
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

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=noveum_trace --cov-report=html

# Run integration tests
pytest tests/integration/

# Run specific test categories
pytest -m integration
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
python docs/examples/agent_workflow_example.py
```

## 📖 Examples

Check out the [examples](docs/examples/) directory for complete working examples:

- [Basic Usage](docs/examples/basic_usage.py) - Simple function tracing
- [Agent Workflow](docs/examples/agent_workflow_example.py) - Multi-agent coordination
- [Langchain Integration](docs/examples/langchain_integration_example.py) - Framework integration patterns
- [Flexible Tracing](docs/examples/flexible_tracing_example.py) - Context managers and inline tracing
- [Streaming Example](docs/examples/streaming_example.py) - Real-time streaming support
- [Multimodal Examples](docs/examples/multimodal_examples.py) - Image, audio, and video tracing

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

### Plugin System

```python
# Register custom plugins
noveum_trace.register_plugin("my_plugin", MyCustomPlugin())

# List available plugins
plugins = noveum_trace.list_plugins()
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

- [GitHub Issues](https://github.com/Noveum/noveum-trace/issues)
- [Documentation](https://github.com/Noveum/noveum-trace/tree/main/docs)
- [Examples](https://github.com/Noveum/noveum-trace/tree/main/examples)

---

**Built by the Noveum Team**
