# Noveum Trace SDK

**Production-ready Python SDK for tracing LLM applications, multi-agent systems, and tool calls with OpenTelemetry compliance.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```python
import noveum_trace

# Initialize with your project ID
noveum_trace.init(project_id="my-ai-project")

# Use the simple @trace decorator
@noveum_trace.trace
def process_data(data):
    return f"Processed: {data}"

@noveum_trace.trace(type="llm", model="gpt-4")
def call_llm(prompt):
    # Your LLM call here
    return "AI response"

@noveum_trace.trace(type="component", agent="data-processor")
def agent_task(task):
    return f"Agent completed: {task}"

# Your functions are now automatically traced!
result = process_data("user input")
```

## ✨ Key Features

### 🎯 **Simplified Decorator Approach**
- **Single `@trace` decorator** with parameters instead of multiple decorators
- **Backward compatibility** with `@observe` and `@llm_trace` aliases
- **Parameter-based specialization** for different operation types

### 🤖 **Multi-Agent Support**
- **Agent registry and management** with hierarchical relationships
- **Cross-agent correlation** and trace propagation
- **Agent-aware context management** for complex workflows
- **Thread-safe operations** for concurrent agent execution

### 🔍 **LLM & Tool Call Tracing**
- **Auto-instrumentation** for OpenAI and Anthropic SDKs
- **Comprehensive LLM metrics** (tokens, latency, model info)
- **Tool call tracking** with arguments and results
- **OpenTelemetry semantic conventions** compliance

### 🏗️ **Project-Based Organization**
- **Required project ID** for proper trace organization
- **Custom headers support** (projectId, orgId, additional headers)
- **Environment-aware** (development, staging, production)
- **Proper trace ID generation** with UUID-based identifiers

## 📦 Installation

```bash
pip install noveum-trace
```

## 🔧 Configuration

### Basic Initialization

```python
import noveum_trace

# Minimal setup (project_id is required)
tracer = noveum_trace.init(project_id="my-project")

# Full configuration
tracer = noveum_trace.init(
    project_id="my-project",
    project_name="My AI Application",
    org_id="org-123",
    user_id="user-456",
    session_id="session-789",
    environment="production",
    api_key="your-noveum-api-key",  # For Noveum.ai platform
    file_logging=True,
    log_directory="./traces",
    auto_instrument=True,
    capture_content=True,
    custom_headers={"X-Custom-Header": "value"}
)
```

### Environment Variables

```bash
export NOVEUM_PROJECT_ID="my-project"
export NOVEUM_API_KEY="your-api-key"
export NOVEUM_ORG_ID="org-123"
export NOVEUM_USER_ID="user-456"
export NOVEUM_SESSION_ID="session-789"
export NOVEUM_ENVIRONMENT="production"
```

## 🎨 Usage Examples

### Simple Function Tracing

```python
@noveum_trace.trace
def data_processing(data):
    # Your processing logic
    return processed_data

@noveum_trace.trace(name="custom-operation")
def custom_function():
    return "result"
```

### LLM Tracing

```python
@noveum_trace.trace(type="llm", model="gpt-4", operation="chat")
def chat_completion(messages):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response

@noveum_trace.trace(type="llm", model="claude-3", operation="completion")
def text_completion(prompt):
    response = anthropic.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": prompt}]
    )
    return response
```

### Multi-Agent Workflows

```python
from noveum_trace import Agent, AgentConfig, AgentContext, trace

# Define agents
coordinator = Agent(AgentConfig(
    name="coordinator",
    agent_type="orchestrator",
    id="coord-001"
))

worker = Agent(AgentConfig(
    name="data-worker",
    agent_type="processor",
    id="worker-001"
))

# Use agent context
with AgentContext(coordinator):
    @trace
    def plan_task(task):
        # Coordinator planning
        return task_plan

    plan = plan_task("analyze data")

    with AgentContext(worker):
        @trace
        def execute_task(plan):
            # Worker execution
            return results

        results = execute_task(plan)
```

### Tool Call Tracing

```python
@noveum_trace.trace(type="tool", tool_name="web_search")
def search_web(query):
    # Tool implementation
    return search_results

@noveum_trace.trace(type="tool", tool_name="calculator")
def calculate(expression):
    # Calculator implementation
    return result
```

### Dynamic Span Updates

```python
from noveum_trace import trace, update_current_span

@trace
def long_running_task():
    update_current_span(
        metadata={"step": "initialization"},
        progress=10
    )

    # Do some work
    initialize()

    update_current_span(
        metadata={"step": "processing"},
        progress=50
    )

    # More work
    process_data()

    update_current_span(
        metadata={"step": "completion"},
        progress=100
    )

    return "completed"
```

## 🔌 Auto-Instrumentation

The SDK automatically instruments popular LLM libraries:

```python
# Auto-instrumentation is enabled by default
noveum_trace.init(project_id="my-project", auto_instrument=True)

# Now all OpenAI and Anthropic calls are automatically traced
import openai
response = openai.chat.completions.create(...)  # Automatically traced!

import anthropic
response = anthropic.messages.create(...)  # Automatically traced!
```

## 📊 Trace Data Structure

Each trace contains:

```json
{
  "trace_id": "uuid-v4",
  "span_id": "uuid-v4",
  "parent_span_id": "uuid-v4",
  "name": "operation-name",
  "kind": "internal|client|server",
  "status": "ok|error",
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-01T00:00:01Z",
  "duration_ms": 1000,
  "project_id": "my-project",
  "project_name": "My AI Application",
  "org_id": "org-123",
  "user_id": "user-456",
  "session_id": "session-789",
  "environment": "production",
  "attributes": {
    "llm.model": "gpt-4",
    "llm.operation": "chat",
    "gen_ai.system": "openai",
    "gen_ai.usage.input_tokens": 100,
    "gen_ai.usage.output_tokens": 50
  },
  "llm_request": {
    "model": "gpt-4",
    "messages": [...],
    "temperature": 0.7
  },
  "llm_response": {
    "id": "response-id",
    "model": "gpt-4",
    "choices": [...],
    "usage": {
      "prompt_tokens": 100,
      "completion_tokens": 50,
      "total_tokens": 150
    }
  },
  "agent": {
    "name": "data-processor",
    "type": "worker",
    "id": "agent-123"
  },
  "tool_calls": [
    {
      "id": "call-123",
      "name": "web_search",
      "arguments": {"query": "AI news"},
      "result": "search results",
      "duration_ms": 500
    }
  ]
}
```

## 🏆 Competitive Advantages

| Feature | Noveum Trace | DeepEval | Phoenix | Braintrust |
|---------|--------------|----------|---------|------------|
| Multi-agent support | ✅ | ❌ | ⚠️ | ⚠️ |
| Simplified decorators | ✅ | ✅ | ✅ | ✅ |
| Auto agent resolution | ✅ | ❌ | ❌ | ❌ |
| OpenTelemetry compliant | ✅ | ❌ | ✅ | ❌ |
| Project-based organization | ✅ | ❌ | ❌ | ❌ |
| Custom headers | ✅ | ❌ | ❌ | ❌ |
| Trace ID management | ✅ | ⚠️ | ✅ | ⚠️ |
| Tool call tracing | ✅ | ❌ | ⚠️ | ⚠️ |

## 📁 Project Structure

```
noveum-trace/
├── src/noveum_trace/
│   ├── __init__.py              # Main exports
│   ├── init.py                  # Simplified initialization
│   ├── types.py                 # Type definitions
│   ├── core/
│   │   ├── tracer.py           # Main tracer implementation
│   │   ├── span.py             # Span implementation
│   │   └── context.py          # Context management
│   ├── agents/
│   │   ├── agent.py            # Agent classes
│   │   ├── registry.py         # Agent registry
│   │   ├── context.py          # Agent context
│   │   └── decorators.py       # Simplified decorators
│   ├── sinks/
│   │   ├── base.py             # Base sink interface
│   │   ├── file.py             # File sink
│   │   ├── console.py          # Console sink
│   │   ├── noveum.py           # Noveum.ai sink
│   │   └── elasticsearch.py    # Elasticsearch sink
│   ├── instrumentation/
│   │   ├── openai.py           # OpenAI auto-instrumentation
│   │   └── anthropic.py        # Anthropic auto-instrumentation
│   └── utils/
│       └── exceptions.py       # Custom exceptions
├── examples/                    # Usage examples
├── tests/                      # Test suite
├── docs/                       # Documentation
└── README.md                   # This file
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest tests/ --cov=noveum_trace
```

## 📚 Documentation

- [Getting Started Guide](docs/getting-started/)
- [Configuration Guide](docs/guides/configuration.md)
- [Multi-Agent Tracing](docs/guides/multi-agent-tracing.md)
- [LLM Tracing Guide](docs/guides/llm-tracing.md)
- [API Reference](docs/api/)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/noveum/noveum-trace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/noveum/noveum-trace/discussions)

## 🚀 Roadmap

- [ ] **JavaScript/TypeScript SDK** - Cross-platform support
- [ ] **Real-time evaluation** - Integration with NovaEval
- [ ] **Advanced analytics** - Performance insights and recommendations
- [ ] **Custom metrics** - User-defined metrics and alerts
- [ ] **Distributed tracing** - Cross-service trace correlation
- [ ] **Visual trace explorer** - Interactive trace visualization

---

**Built with ❤️ by the Noveum team**
