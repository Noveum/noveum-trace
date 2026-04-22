# CrewAI Integration Guide

This guide shows how to add Noveum Trace to your [CrewAI](https://www.crewai.com/) crews and flows so runs are exported to your Noveum project.

## Prerequisites

- **Python 3.10 or newer** (the `noveum-trace[crewai]` extra installs CrewAI versions that require Python ≥ 3.10.)
- A **CrewAI** application (crews, tasks, agents, and optionally Flows, memory, A2A, or MCP).
- A **Noveum API key** ([noveum.ai](https://noveum.ai)).

## Installation

```bash
pip install "noveum-trace[crewai]"
```

Optional, depending on your app:

```bash
# Remote A2A delegation (when you use A2A client agents)
pip install "crewai[a2a]"

# Memory embedders (when you use unified memory with local embeddings)
pip install sentence-transformers
```

## Quick start

### 1. Configure Noveum

Set credentials (environment or `.env`):

```bash
export NOVEUM_API_KEY="your-api-key"
export NOVEUM_PROJECT="my-crewai-app"
# Optional
export NOVEUM_ENVIRONMENT="production"
```

### 2. Initialize the SDK

Call once at process startup, before creating the listener:

```python
import noveum_trace

noveum_trace.init(
    project="my-crewai-app",
    api_key="your-noveum-api-key",  # or rely on NOVEUM_API_KEY
    environment="production",
)
```

### 3. Start CrewAI tracing

```python
from noveum_trace.integrations.crewai import setup_crewai_tracing

listener = setup_crewai_tracing()
```

`setup_crewai_tracing()` must run **after** `noveum_trace.init()`. It returns a `NoveumCrewAIListener` you should keep alive for the whole run.

### 4. Run your crew (or flow)

Use your existing `Crew` / `kickoff()` (or `Flow` / `kickoff()`) code as usual. If your CrewAI version expects an explicit callback on the crew, attach the listener before kickoff:

```python
from crewai import Crew

crew = Crew(agents=[...], tasks=[...], ...)
crew.callback_function = listener
result = crew.kickoff(inputs={"topic": "example"})
```

When you are finished tracing (e.g. before shutdown in tests or short-lived scripts), call:

```python
listener.shutdown()
```

## Manual construction

If you already hold a Noveum client:

```python
from noveum_trace import get_client
from noveum_trace.integrations.crewai import NoveumCrewAIListener

client = get_client()
listener = NoveumCrewAIListener(client, capture_memory=True, verbose=False)
crew.callback_function = listener
crew.kickoff()
```

## Listener options

Pass any of these keyword arguments to `setup_crewai_tracing()` or `NoveumCrewAIListener(...)`:

| Option | Default | Purpose |
|--------|---------|---------|
| `capture_inputs` | `True` | Task prompts, tool arguments, crew inputs |
| `capture_outputs` | `True` | LLM answers, tool outputs, task and crew results |
| `capture_llm_messages` | `True` | Serialized LLM message payloads |
| `capture_tool_schemas` | `True` | Tool definitions / available-tool metadata |
| `capture_agent_snapshot` | `True` | Agent goal, backstory, and profile fields |
| `capture_crew_snapshot` | `True` | Crew-level task and agent snapshots at kickoff |
| `capture_memory` | `True` | Memory query / save / retrieval spans |
| `capture_knowledge` | `True` | Knowledge-related events |
| `capture_a2a` | `True` | A2A delegation and conversation spans |
| `capture_mcp` | `True` | MCP connection and tool spans |
| `capture_flow` | `True` | Flow and flow-method spans |
| `capture_reasoning` | `True` | Reasoning-step events |
| `capture_guardrails` | `True` | Guardrail events |
| `capture_streaming` | `True` | Join streaming LLM chunks into span attributes |
| `capture_thinking` | `True` | Extended-thinking / reasoning token buffers |
| `trace_name_prefix` | `"crewai"` | Prefix for exported trace names |
| `verbose` | `False` | Extra debug logging from the listener |

Turn off any flag you do not want recorded (for example `capture_llm_messages=False` in regulated environments).

## Programmatic SDK options

You can pass transport and other SDK settings to `noveum_trace.init()` the same way as other integrations, for example batching:

```python
noveum_trace.init(
    project="my-crewai-app",
    api_key="your-key",
    environment="production",
    transport_config={
        "batch_size": 10,
        "batch_timeout": 5.0,
    },
)
```

## Environment variables

Typical variables:

| Variable | Description |
|----------|-------------|
| `NOVEUM_API_KEY` | Required for export (unless you pass `api_key=` into `init()`) |
| `NOVEUM_PROJECT` | Project name in Noveum |
| `NOVEUM_ENVIRONMENT` | Logical environment (for example `production` / `staging`) |

CrewAI’s own telemetry is separate; if you need to disable it for local runs, see CrewAI’s docs for `CREWAI_DISABLE_TELEMETRY` and related settings.

## Verification

1. Run a short crew with tracing enabled and a valid `NOVEUM_API_KEY`.
2. Open the [Noveum dashboard](https://app.noveum.ai), select your project, and confirm new traces appear for your kickoff.
3. Optionally call `noveum_trace.flush()` before exit so batched spans are sent in short scripts.

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `RuntimeError: Noveum tracing not initialized` | Call `noveum_trace.init()` before `setup_crewai_tracing()`. |
| `pip` cannot install `noveum-trace[crewai]` on Python 3.9 | Use Python **3.10+** for the CrewAI extra (CrewAI’s supported Python range). |
| No traces in the UI | API key, project name, network, and that the process did not exit before flush. |
| Very large payloads | Disable or narrow `capture_llm_messages`, `capture_tool_schemas`, or `capture_inputs` / `capture_outputs`. |

## Example script

For a full runnable checklist (multi-agent crew, optional A2A URL, Flow, and shutdown behavior), see:

[`docs/examples/crewai_e2e_test.py`](examples/crewai_e2e_test.py)

Run (from the repo root):

```bash
python docs/examples/crewai_e2e_test.py
```

## Next steps

- Tune listener flags for your compliance and payload size needs.
- Use the Noveum UI to filter traces by project and time range and to inspect span attributes for debugging.
