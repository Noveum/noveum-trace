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

noveum_trace.flush()     # export buffered traces
noveum_trace.shutdown()  # release SDK resources at application termination
```

## Convenience factory

`setup_openai_agents_tracing()` creates the processor **and** registers it in one
call. It requires `noveum_trace.init(...)` to have run first:

```python
import noveum_trace
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
| `capture_inputs` | `True` | Raw tool / function / custom-span inputs |
| `capture_outputs` | `True` | Raw tool / function outputs |
| `capture_llm_messages` | `True` | Full LLM prompt/response messages, system prompts and tool calls (generation & response spans) |
| `capture_tool_schemas` | `True` | Agent tool & handoff names, and the tool schemas offered to the model |
| `capture_trace_metadata` | `True` | OpenAI trace `metadata` and `group_id` (see privacy note) |
| `capture_cost` | `True` | Estimated LLM cost from model + token counts |
| `trace_name_prefix` | `"openai_agents"` | Prefix used when a workflow has no name |

**Everything is captured by default**, matching the other Noveum integrations —
a trace without prompts, tool arguments and results cannot be replayed, diffed
or turned into an evaluation dataset. Set any flag to `False` to reduce what is
sent:

```python
# Reduce what leaves the process
NoveumTraceProcessor(
    capture_inputs=False,
    capture_outputs=False,
    capture_llm_messages=False,
)
```

**Payloads are never truncated.** System prompts, tool results and message
arrays routinely run past any fixed character budget, and a clipped prompt is
useless for evaluation, so this integration records them in full.

**Note on trace metadata and errors.** With `capture_trace_metadata=True`
(default), the OpenAI trace's `metadata` and `group_id` are serialized as-is,
with no allowlist or redaction — if you place sensitive data there, it is sent to
Noveum. Set `capture_trace_metadata=False` to disable this. Error details
(message, and any structured `error.data` on a span) are also captured by default
for observability and may echo sensitive context; redact at the source if that is
a concern.

### The SDK has its own payload switch

The Agents SDK decides separately whether to record prompts and responses on its
span data at all. Running with `RunConfig(trace_include_sensitive_data=False)`,
or with `OPENAI_AGENTS_DONT_LOG_MODEL_DATA` set in the environment, leaves
`span_data.input` / `span_data.output` empty **before** any processor sees them —
no flag here can recover data the SDK never recorded. If prompts are missing
while `capture_llm_messages=True`, check that switch first.

## What gets traced

The integration is a mapping layer: the Agents SDK already records the run as a
trace/span tree, and the processor mirrors that tree into Noveum, translating
each `SpanData` into `llm.*` / `tool.*` / `agent.*` attributes that the rest of
the platform (and the `otel_compat` `gen_ai` crosswalk) understands. It does not
re-instrument the agent or intercept the model client, so it can only report what
the SDK puts on its span data — the table below is that complete surface.

| OpenAI span type | Noveum span | Key attributes |
| --- | --- | --- |
| `agent` | `openai_agents.agent` | `agent.name`, `agent.tools`, `agent.tool_count`, `agent.handoffs`, `agent.output_type`, `agent.metadata` |
| `generation` | `openai_agents.generation` | `llm.model`, `llm.provider`, `llm.system_prompt`, `llm.input`, `llm.input_text`, `llm.output`, `llm.output_text`, `llm.tool_calls`, `llm.tool_call_count`, `llm.input_tokens`, `llm.output_tokens`, `llm.total_tokens`, `llm.cached_input_tokens`, `llm.cache_write_input_tokens`, `llm.cache_hit`, `llm.reasoning_tokens`, `llm.temperature`, `llm.top_p`, `llm.max_tokens`, `llm.reasoning_effort`, `llm.cost.*` |
| `response` | `openai_agents.response` | everything above, plus `llm.request_id`, `llm.response_status`, `llm.available_tools`, `llm.available_tool_count`, `llm.reasoning` (reasoning summaries) |
| `function` | `openai_agents.function` | `tool.name`, `tool.input`, `tool.output`, `tool.is_mcp`, `tool.mcp_data` |
| `handoff` | `openai_agents.handoff` | `handoff.from_agent`, `handoff.to_agent` |
| `guardrail` | `openai_agents.guardrail` | `guardrail.name`, `guardrail.triggered` |
| `mcp_tools` | `openai_agents.mcp_tools` | `mcp.server_name`, `mcp.tools`, `mcp.tool_count` |
| `custom` | `openai_agents.custom` | `custom.name`, `custom.data` |
| `task` / `turn` | `openai_agents.task` / `.turn` | `task.name`, `turn.number`, `turn.agent_name`, aggregate token usage |

### Where to find each kind of data

The Agents SDK splits a run across span types, so some things are not where you
might first look:

- **Tool calls and their results** are on `function` spans (`tool.input` /
  `tool.output`), one per invocation — not on the `agent` span. The model's
  *decision* to call a tool is also recorded on the model span as
  `llm.tool_calls`.
- **The agent's input** is the input of its first model call
  (`llm.input` / `llm.input_text` on the child `generation` / `response` span).
  `AgentSpanData` carries only configuration — name, tool names, handoff names,
  output type, metadata.
- **Available tools**: names on `agent.tools`; full schemas (with descriptions and
  JSON-Schema parameters) on `llm.available_tools` for Responses-API calls.
  Chat-Completions `model_config` carries no tool list upstream, so
  `generation` spans fall back to the enclosing agent's `agent.tools`.
- **System prompt**: `llm.system_prompt` — from the `system`/`developer` messages
  on generation spans, and from `response.instructions` on response spans.
- **Reasoning / thinking**: `llm.reasoning_tokens` always; the reasoning text
  itself (`llm.reasoning`) only on response spans, and only when the model
  returns reasoning summaries.
- **Prompt caching**: `llm.cached_input_tokens`, `llm.cache_write_input_tokens`
  and the derived `llm.cache_hit` boolean.
- **Input images** are not extracted into their own attribute; they remain inside
  the serialized `llm.input` message array.

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
  executed. For a short-lived process, call `noveum_trace.flush()` then
  `noveum_trace.shutdown()` before it exits — `flush()` sends buffered traces and
  `shutdown()` releases SDK resources.
- **`ImportError` from `setup_openai_agents_tracing`:** the `openai-agents` extra
  is not installed — `pip install "noveum-trace[openai-agents]"`.
- **Tool arguments missing:** these are on by default; check that
  `capture_inputs` / `capture_outputs` were not disabled.
- **LLM prompts/responses missing:** `capture_llm_messages` is on by default, so
  check the SDK's own switch first — `RunConfig(trace_include_sensitive_data=False)`
  or `OPENAI_AGENTS_DONT_LOG_MODEL_DATA` blanks the payloads upstream.
- **Tool results not on the agent span:** they are on the child `function` spans;
  see [Where to find each kind of data](#where-to-find-each-kind-of-data).

## Example script

See [`docs/examples/openai_agents_integration_example.py`](examples/openai_agents_integration_example.py).

## Next steps

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents tracing docs](https://openai.github.io/openai-agents-python/tracing/)
