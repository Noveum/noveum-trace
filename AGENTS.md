# AGENTS.md — Noveum Trace SDK reference for AI agents and IDEs

This file is a compact machine-readable reference. Do not invent APIs not listed here.

## Installation

```bash
# Core SDK (Python 3.9+)
pip install noveum-trace

# With framework extras
pip install "noveum-trace[langchain]"    # LangChain / LangGraph callback handler
pip install "noveum-trace[livekit]"     # LiveKit STT/TTS wrappers + session tracing
pip install "noveum-trace[pipecat]"     # Pipecat observer
pip install "noveum-trace[crewai]"      # CrewAI listener (Python 3.10+ required)
pip install "noveum-trace[openai]"      # OpenAI extras
pip install "noveum-trace[anthropic]"   # Anthropic extras
```

## Initialization

```python
import noveum_trace

noveum_trace.init(
    api_key="your-api-key",          # or NOVEUM_API_KEY env var
    project="my-app",                # or NOVEUM_PROJECT env var
    environment="production",        # optional, default "development"
)
# init() is idempotent — calling it again has no effect
```

## Core root exports (`from noveum_trace import ...`)

| Symbol | Type | Notes |
|--------|------|-------|
| `init` | function | Initialize SDK; idempotent |
| `shutdown` | function | Flush + teardown |
| `flush` | function | Send pending traces |
| `get_client` | function | Returns `NoveumClient`; raises `InitializationError` if not init'd |
| `get_config` | function | Returns current config |
| `is_initialized` | function | Returns bool |
| `configure` | function | Low-level config update |
| `trace_llm_call` | context manager factory | Trace an LLM call |
| `trace_agent_operation` | context manager factory | Trace an agent operation |
| `trace_operation` | context manager factory | Trace any generic operation |
| `trace_batch_operation` | context manager factory | Trace batch operations |
| `trace_pipeline_stage` | context manager factory | Trace a pipeline stage |
| `create_child_span` | context manager factory | Create a child span |
| `trace_context` | context manager | Manual trace context |
| `start_trace` | function | Manually start a trace |
| `start_span` | function | Manually start a span |
| `NoveumClient` | class | Direct client access |
| `Trace`, `Span` | classes | Trace/span primitives |
| `create_agent_graph` | function | Create an agent graph |
| `create_agent`, `get_agent` | functions | Agent registry |
| `create_agent_workflow`, `get_agent_workflow` | functions | Workflow management |
| `trace_streaming`, `streaming_llm` | functions | Streaming LLM support |
| `create_thread`, `trace_thread_llm` | functions | Thread/conversation management |
| `NoveumTraceCallbackHandler` | class | Available when `[langchain]` extra installed |
| `NoveumCrewAIListener`, `setup_crewai_tracing` | class/function | Available when `[crewai]` extra installed (Python 3.10+) |

## Context manager patterns

```python
# LLM tracing
with noveum_trace.trace_llm_call(model="gpt-4", provider="openai") as span:
    response = openai_client.chat.completions.create(...)
    span.set_attributes({
        "llm.input_tokens": response.usage.prompt_tokens,
        "llm.output_tokens": response.usage.completion_tokens,
    })
    # Or use capture_response for automatic extraction:
    span.capture_response(response)

# Agent operation tracing
with noveum_trace.trace_agent_operation(
    agent_type="planner", operation="task_planning"
) as span:
    plan = agent.plan(task)
    span.set_attribute("plan.steps", len(plan.steps))

# Generic operation tracing
with noveum_trace.trace_operation("database_query") as span:
    result = db.query(sql)
    span.set_attribute("query.rows", len(result))
```

## Integration import paths

```python
# LangChain (requires [langchain])
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler
# Also available via:
from noveum_trace import NoveumTraceCallbackHandler  # when langchain installed
from noveum_trace.integrations import NoveumTraceCallbackHandler  # when langchain installed

# LiveKit (requires [livekit])
from noveum_trace.integrations.livekit import (
    setup_livekit_tracing,       # setup_livekit_tracing(session, *, enabled, trace_name_prefix, record, cleanup_audio_files)
    LiveKitSTTWrapper,
    LiveKitTTSWrapper,
    LiveKitLLMWrapper,
    extract_job_context,
)

# Pipecat (requires [pipecat])
from noveum_trace.integrations.pipecat import NoveumTraceObserver, setup_pipecat_tracing
# setup_pipecat_tracing(**kwargs) -> NoveumTraceObserver
# obs.attach_to_task(task) must be called before runner.run()

# CrewAI (requires [crewai], Python 3.10+)
from noveum_trace.integrations.crewai import NoveumCrewAIListener, setup_crewai_tracing
# Also: from noveum_trace import NoveumCrewAIListener, setup_crewai_tracing
# Also: from noveum_trace.integrations import NoveumCrewAIListener, setup_crewai_tracing
```

## LiveKit signature (IMPORTANT)

```python
setup_livekit_tracing(
    session,                    # AgentSession instance — required positional
    *,                          # keyword-only after this
    enabled: bool = True,
    trace_name_prefix: str | None = None,
    record: bool = True,
    cleanup_audio_files: bool = True,
)
# NO metadata= parameter — do not pass metadata=job_metadata
```

## APIs that do NOT exist — never invent these

- `from noveum_trace import trace_llm` — use `trace_llm_call` instead
- `trace_llm(..., input=..., output=...)` — invalid constructor pattern
- `from noveum_trace.fastapi import NoveumMiddleware` — does not exist
- `configure_sampling(...)` — does not exist as a top-level function
- `trace_function_calls` — does not exist
- `from noveum_trace import NovaEval` — does not exist
- `setup_livekit_tracing(session, metadata=job_metadata)` — no metadata param
- `register_plugin` / `list_plugins` — exist in `__all__` but raise `NotImplementedError`

## Examples

See [`docs/examples/`](docs/examples/) for runnable examples:
- `basic_usage.py` — core context managers
- `langchain_integration_example.py` — LangChain callback handler
- `livekit_integration_example.py` — LiveKit session + STT/TTS wrappers
- `crewai_e2e_test.py` — CrewAI crew with full tracing

## Source of truth

`src/noveum_trace/__init__.py` `__all__` is the definitive public API list.
`pyproject.toml` `[project.optional-dependencies]` is the definitive extras list.
