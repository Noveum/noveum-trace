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
├── transport/      # HTTP transport and batch processing
├── integrations/   # Framework integrations (OpenAI, etc.)
├── utils/          # Utilities (exceptions, serialization, etc.)
└── examples/       # Usage examples
```

## 🔧 Configuration

### Environment Variables

```bash
export NOVEUM_API_KEY="your-api-key"
export NOVEUM_PROJECT="your-project-name"
```

### Programmatic Configuration

```python
import noveum_trace
from noveum_trace.core.config import Config

# Basic configuration
noveum_trace.init(
    api_key="your-api-key",
    project="my-project",
    endpoint="https://api.noveum.ai"
)

# Advanced configuration
config = Config(
    api_key="your-api-key",
    project="my-project",
    endpoint="https://api.noveum.ai"
)
config.transport.batch_size = 10
config.transport.batch_timeout = 5.0

noveum_trace.configure(config)
```

## 🎯 Available Decorators

### @trace - General Purpose Tracing

```python
@noveum_trace.trace
def my_function(arg1: str, arg2: int) -> dict:
    return {"result": f"{arg1}_{arg2}"}
```

### @trace_llm - LLM Call Tracing

```python
@noveum_trace.trace_llm
def call_llm(prompt: str) -> str:
    # LLM call implementation
    return response
```

### @trace_agent - Agent Workflow Tracing

```python
@noveum_trace.trace_agent(agent_id="my_agent")
def agent_function(task: str) -> dict:
    # Agent implementation
    return result
```

### @trace_tool - Tool Usage Tracing

```python
@noveum_trace.trace_tool
def search_web(query: str) -> list:
    # Tool implementation
    return results
```

### @trace_retrieval - Retrieval Operation Tracing

```python
@noveum_trace.trace_retrieval
def retrieve_documents(query: str) -> list:
    # Retrieval implementation
    return documents
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
python examples/basic_usage.py
python examples/agent_workflow_example.py
```

## 📖 Examples

Check out the [examples](examples/) directory for complete working examples:

- [Basic Usage](examples/basic_usage.py) - Simple function tracing
- [Agent Workflow](examples/agent_workflow_example.py) - Multi-agent coordination
- [Langchain Integration](examples/langchain_integration_example.py) - Framework integration patterns

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

- [GitHub Issues](https://github.com/Noveum/noveum-trace/issues)
- [Documentation](https://github.com/Noveum/noveum-trace/tree/main/docs)
- [Examples](https://github.com/Noveum/noveum-trace/tree/main/examples)

---

**Built by the Noveum Team**
