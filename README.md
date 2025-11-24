# Noveum Trace SDK

[![CI](https://github.com/Noveum/noveum-trace/actions/workflows/ci.yml/badge.svg)](https://github.com/Noveum/noveum-trace/actions/workflows/ci.yml)
[![Release](https://github.com/Noveum/noveum-trace/actions/workflows/release.yml/badge.svg)](https://github.com/Noveum/noveum-trace/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/Noveum/noveum-trace/branch/main/graph/badge.svg)](https://codecov.io/gh/Noveum/noveum-trace)
[![PyPI version](https://badge.fury.io/py/noveum-trace.svg)](https://badge.fury.io/py/noveum-trace)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Simple, intuitive tracing SDK for LLM applications and multi-agent systems.**

Noveum Trace provides an easy way to add observability to your LLM applications. With intuitive context managers, you can trace function calls, LLM interactions, agent workflows, and multi-agent coordination patterns.

## ✨ Key Features

- **🎯 Simple Context Manager API** - Add tracing with intuitive `with` statements
- **🤖 Multi-Agent Support** - Built for multi-agent systems and workflows
- **☁️ Cloud Integration** - Send traces to Noveum platform or custom endpoints
- **🔌 Framework Agnostic** - Works with any Python LLM framework
- **🚀 Zero Configuration** - Works out of the box with sensible defaults
- **📊 Comprehensive Tracing** - Capture function calls, LLM interactions, and agent workflows
- **🔄 Flexible Integration** - Context managers for granular control

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

# Trace any operation using context managers
def process_document(document_id: str) -> dict:
    with noveum_trace.trace_operation("process_document") as span:
        # Your function logic here
        span.set_attribute("document_id", document_id)
        return {"status": "processed", "id": document_id}

# Trace LLM calls with automatic metadata capture
def call_openai(prompt: str) -> str:
    import openai
    client = openai.OpenAI()
    
    with noveum_trace.trace_llm_call(model="gpt-4", provider="openai") as span:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        span.set_attributes({
            "llm.input_tokens": response.usage.prompt_tokens,
            "llm.output_tokens": response.usage.completion_tokens
        })
        return response.choices[0].message.content
```

## ⚙️ Setup

### Core Configuration

The SDK requires a few core environment variables to function:

```bash
# Required: Your Noveum API key
export NOVEUM_API_KEY="your-api-key"

# Required: Project name for organizing traces
export NOVEUM_PROJECT="your-project-name"

# Optional: Environment name (defaults to "development")
export NOVEUM_ENVIRONMENT="production"

# Optional: Custom API endpoint (defaults to https://api.noveum.ai/api)
export NOVEUM_ENDPOINT="https://api.noveum.ai/api"
```

### Additional Environment Variables

For a complete list of all available environment variables including debug settings, logging configuration, and agent registry limits, see [`.env.example`](.env.example) in the repository root.


## 🏗️ Architecture

```
noveum_trace/
├── core/              # Core tracing primitives (Trace, Span, Context)
├── context_managers/  # Context managers for inline tracing
├── transport/         # HTTP transport and batch processing
├── integrations/      # Framework integrations (LangChain, etc.)
├── streaming/         # Streaming LLM support
├── threads/           # Conversation thread management
└── utils/             # Utilities (exceptions, serialization, etc.)
```

## 🔧 Configuration

### Environment Variables

The SDK can be configured via environment variables. The core configuration variables are:

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

## 🔄 Context Manager Usage

For scenarios with granular control:

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

## 🔗 LangChain Integration

Noveum Trace provides seamless integration with LangChain and LangGraph applications through a simple callback handler.

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

### What Gets Traced

- **LLM Calls**: Model, prompts, responses, token usage
- **Chains**: Input/output flow, execution steps  
- **Agents**: Decision-making, tool usage, reasoning
- **Tools**: Function calls, inputs, outputs
- **LangGraph Nodes**: Graph execution, node transitions
- **Routing Decisions**: Conditional routing logic and decisions

### Advanced Features

The integration also supports:
- **Manual Trace Control** for complex workflows
- **Custom Parent Relationships** for explicit span hierarchies
- **LangGraph Routing Tracking** for routing decisions

For complete details and examples, see the [LangChain Integration Guide](docs/LANGCHAIN_INTEGRATION.md).

## 🎤 LiveKit Integration

Automatically trace LiveKit agent sessions with complete observability:

```python
from livekit.agents import Agent, AgentSession, JobContext
from livekit.plugins import deepgram, cartesia
from noveum_trace.integrations.livekit_session import setup_livekit_tracing
from noveum_trace.integrations.livekit import LiveKitSTTWrapper, LiveKitTTSWrapper

# Initialize noveum-trace
noveum_trace.init(project="livekit-agent")

async def agent_entrypoint(ctx: JobContext):
    # Wrap STT/TTS providers for detailed audio tracking
    traced_stt = LiveKitSTTWrapper(
        stt=deepgram.STT(model="nova-2"),
        session_id=ctx.job.id,
        job_context={"job_id": ctx.job.id, "room": ctx.room.name}
    )
    
    traced_tts = LiveKitTTSWrapper(
        tts=cartesia.TTS(model="sonic-english"),
        session_id=ctx.job.id,
        job_context={"job_id": ctx.job.id}
    )
    
    # Create session with traced providers
    session = AgentSession(stt=traced_stt, tts=traced_tts)
    
    # Enable session tracing for automatic event tracking
    # This creates the trace automatically - no need for start_trace()
    setup_livekit_tracing(session)
    
    agent = Agent(instructions="You are a helpful assistant.")
    await ctx.connect()
    await session.start(agent)  # Complete tracing active!
```

### What Gets Traced

**Session Events** (automatic):
- **AgentSession Events**: State changes, transcriptions, function calls, errors, metrics
- **RealtimeSession Events**: Speech detection, transcriptions, generations (when using RealtimeModel)
- **Automatic Trace Creation**: Trace is created when `session.start()` is called

**STT/TTS Operations** (via wrappers):
- **STT Operations**: Transcripts, confidence scores, audio files, durations
- **TTS Operations**: Synthesized text, audio files, durations
- **Job Context**: Room info, participant details, agent metadata
- **Audio Capture**: Automatic saving of audio files for debugging

### Key Features

- ✅ **Complete Observability**: Session events + detailed STT/TTS tracking
- ✅ **Zero Configuration**: Session tracing creates trace automatically
- ✅ Works with **any** LiveKit STT/TTS provider
- ✅ Supports streaming and batch modes
- ✅ Automatic audio file capture and storage
- ✅ Rich metadata in span attributes
- ✅ Graceful degradation (no disruption if tracing fails)

For step-by-step setup instructions, see the [LiveKit Integration Guide](docs/LIVEKIT_INTEGRATION_GUIDE.md).

For detailed API documentation, see the [LiveKit Integration Docs](docs/LIVEKIT_INTEGRATION.md).

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
