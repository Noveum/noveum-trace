# OpenAI Agents SDK Integration Guide

Trace [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) runs
with Noveum. The integration registers a `TracingProcessor` that mirrors every
OpenAI Agents **trace** as a Noveum trace and every OpenAI Agents **span**
(agent runs, tool/function calls, LLM generations, handoffs, guardrails, MCP tool
listings) as a Noveum span — preserving the parent-child hierarchy and attaching
`gen_ai`-compatible `llm.*` / `tool.*` / `agent.*` attributes.

## Prerequisites

- Python 3.10+ (required by `openai-agents`)
- A Noveum project + API key
- `openai-agents >= 0.19.2`

## Installation

```bash
pip install "noveum-trace[openai-agents]"
```

## Quick start

### 1. Initialize the SDK

```python
import noveum_trace

noveum_trace.init(project="my-project", api_key="...")
```

### 2. Register the trace processor

```python
from agents import add_trace_processor
from noveum_trace.integrations.openai_agents import NoveumTraceProcessor

add_trace_processor(NoveumTraceProcessor())
```

### 3. Run your agents

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are helpful.")
result = await Runner.run(agent, "Hello!")

noveum_trace.flush()  # export buffered traces before exit
```

## Convenience factory

`setup_openai_agents_tracing()` creates the processor **and** registers it in one
call. It requires `noveum_trace.init(...)` to have run first:

```python
from noveum_trace.integrations.openai_agents import setup_openai_agents_tracing

noveum_trace.init(project="my-project", api_key="...")
setup_openai_agents_tracing()  # add alongside OpenAI's default exporter
```

Pass `replace_processors=True` to make Noveum the **only** processor (disables
OpenAI's own trace upload):

```python
setup_openai_agents_tracing(replace_processors=True)
```

## Processor options and privacy

All options are accepted by `NoveumTraceProcessor(...)` and
`setup_openai_agents_tracing(...)`:

| Option | Default | Captures |
| --- | --- | --- |
| `capture_inputs` | `False` | Raw tool / function / custom-span inputs |
| `capture_outputs` | `False` | Raw tool / function / LLM outputs |
| `capture_llm_messages` | `False` | Full LLM prompt/response message arrays |
| `capture_tool_schemas` | `True` | Agent tool & handoff **names** (not argument values) |
| `capture_trace_metadata` | `True` | OpenAI trace `metadata` and `group_id` |
| `capture_cost` | `True` | Estimated LLM cost from model + token counts |
| `trace_name_prefix` | `"openai_agents"` | Prefix used when a workflow has no name |

**Privacy-safe by default.** Raw payloads that may contain sensitive data —
tool inputs/outputs and LLM message content — are **not** captured unless you opt
in via `capture_inputs`, `capture_outputs`, or `capture_llm_messages`. Structural
metadata that is always captured: span type, agent/tool/handoff names, model
name and provider, token usage, estimated cost, guardrail triggered flag, latency
(span start/end), and error type/message.

```python
# Capture full payloads (e.g. in a trusted dev environment)
NoveumTraceProcessor(
    capture_inputs=True,
    capture_outputs=True,
    capture_llm_messages=True,
)
```

## What gets traced

| OpenAI span type | Noveum span | Key attributes |
| --- | --- | --- |
| `agent` | `openai_agents.agent` | `agent.name`, `agent.tools`, `agent.handoffs` |
| `generation` | `openai_agents.generation` | `llm.model`, `llm.input_tokens`, `llm.output_tokens`, `llm.total_tokens`, `llm.cost.*` |
| `response` | `openai_agents.response` | `llm.model`, `llm.request_id`, token usage |
| `function` | `openai_agents.function` | `tool.name`, `tool.input`\*, `tool.output`\* |
| `handoff` | `openai_agents.handoff` | `handoff.from_agent`, `handoff.to_agent` |
| `guardrail` | `openai_agents.guardrail` | `guardrail.name`, `guardrail.triggered` |
| `mcp_tools` | `openai_agents.mcp_tools` | `mcp.server_name`, `mcp.tools` |
| `custom` | `openai_agents.custom` | `custom.name`, `custom.data`\* |

\* Captured only when the corresponding capture flag is enabled.

## Version compatibility

| Component | Supported |
| --- | --- |
| `openai-agents` | `>= 0.19.2` |
| Python | `>= 3.10` |

The processor reads span data defensively (`getattr`), so newer span types are
recorded generically rather than raising.

## Resilience

Failures inside the processor (a mapping error, an unreachable Noveum backend)
are logged at debug level and never propagate into your agent run. The OpenAI
Agents SDK additionally wraps every processor callback in its own try/except.

## Troubleshooting

- **No traces appear:** confirm `noveum_trace.init(...)` ran before the agent
  executed, and call `noveum_trace.flush()` (or rely on `shutdown`) before the
  process exits.
- **`ImportError` from `setup_openai_agents_tracing`:** the `openai-agents` extra
  is not installed — `pip install "noveum-trace[openai-agents]"`.
- **Tool arguments / LLM messages missing:** these are opt-in; enable
  `capture_inputs` / `capture_outputs` / `capture_llm_messages`.

## Example script

See [`docs/examples/openai_agents_integration_example.py`](examples/openai_agents_integration_example.py).

## Next steps

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents tracing docs](https://openai.github.io/openai-agents-python/tracing/)
