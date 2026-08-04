# Upstream Submission Reference

Engineering-facing reference for submitting Noveum Trace observability
integrations to upstream ecosystems (Pipecat, CrewAI, LiveKit, LangChain,
LangGraph). This is the single source of truth for what we can accurately claim
in an upstream docs or listing PR.

Every symbol, extra, and version below is taken from the current source
(`pyproject.toml`, `src/noveum_trace/`). Re-verify against source before opening
any PR — do not document an API, extra, or flag that does not exist in the
released package.

- Package: `noveum-trace` (PyPI), `requires-python = ">=3.9"`, Apache-2.0. See
  `pyproject.toml` for the current version (do not pin a version number here — it
  goes stale on every release).
- Install: `pip install "noveum-trace[<extra>]"`.

## 1. Support matrix

| Ecosystem | SDK module | Install extra | Canonical API | Upstream status |
|-----------|-----------|---------------|---------------|-----------------|
| LangChain | `noveum_trace.integrations.langchain` (also re-exported at package root) | `langchain` | `NoveumTraceCallbackHandler` | Community-maintained observability integration (callback handler). No upstream PR yet. |
| LangGraph | `noveum_trace.integrations.langchain` (same handler) | `langchain` (no separate `langgraph` extra) | `NoveumTraceCallbackHandler` (optionally `use_langchain_assigned_parent=True`) | Community-maintained observability integration (callback handler). No upstream PR yet. |
| LiveKit | `noveum_trace.integrations.livekit` | `livekit` | `setup_livekit_tracing(session)` | Community-maintained observability integration. Candidate for upstream docs/listing. |
| Pipecat | `noveum_trace.integrations.pipecat` | `pipecat` (add `pipecat-otel` for OTEL span export) | `NoveumPipecatTracer` (two-call: `observe_pipeline` + `register_task_handlers`) | Community-maintained observability integration (observer). Candidate for upstream docs/listing. |
| CrewAI | `noveum_trace.integrations.crewai` | `crewai` | `setup_crewai_tracing()` (or `NoveumCrewAIListener`) | Community-maintained observability integration (listener). Candidate for upstream docs/listing. |
| OpenTelemetry alignment | `noveum_trace.integrations.pipecat.custom_spans` | `pipecat-otel` | Plain OTEL spans folded into the Pipecat trace via `capture_custom_spans=True` (registers an OTEL `SpanProcessor`) | Bridge only — there is no standalone OTEL exporter. Do not describe a general-purpose OpenTelemetry backend. |

**Not yet supported — do not claim support in any listing:** OpenAI Agents SDK,
LlamaIndex, AutoGen, Vercel AI SDK, and LiteLLM have no dedicated integration
module in `src/noveum_trace/integrations/`. Direct OpenAI/Anthropic calls can be
traced with the core context managers (`trace_llm_call`), but that is not a
framework integration.

There is no plugin system: `noveum_trace.register_plugin` and
`noveum_trace.list_plugins` currently raise `NotImplementedError`. Never
describe Noveum Trace as a "plugin".

## 2. Version compatibility

Effective Python floor per ecosystem is the higher of the SDK core floor (3.9)
and the upstream framework's own floor.

| Ecosystem | Effective Python floor | Upstream dependency pin |
|-----------|------------------------|-------------------------|
| Core SDK / direct OpenAI + Anthropic | 3.9+ | `openai>=1.0.0`, `anthropic>=0.3.0` |
| LangChain | 3.10+ | `langchain-core>=0.1.0` (+ `Pillow>=9.0.0`) |
| LangGraph | 3.10+ | `langchain-core>=0.1.0` (shares the `langchain` extra) |
| LiveKit | 3.10+ | `livekit>=1.0.19,<2`, `livekit-agents>=1.0.0` |
| CrewAI | 3.10+ | `crewai>=0.177.0; python_version>='3.10'` |
| Pipecat | 3.11+ (required by `pipecat-ai`) | `pipecat-ai>=0.0.108`; `pipecat-otel` adds `opentelemetry-api>=1.0.0`, `opentelemetry-sdk>=1.0.0` |

Other extras: `bedrock` (`boto3>=1.34.0`), `pii_redaction` (`spacy>=3.7.0`).

## 3. Canonical quick-start API per ecosystem

`noveum_trace.init(api_key=..., project=..., environment=...)` configures the
global client and must run before any integration setup. Never pass `api_key`
or `project` into an integration constructor — they belong only in `init()`.

### Pipecat — `NoveumPipecatTracer` (two-call)

```python
import noveum_trace
from noveum_trace.integrations.pipecat import NoveumPipecatTracer

noveum_trace.init(api_key="...", project="my-voice-bot")

tracer = NoveumPipecatTracer(record_audio=True)
pipeline = tracer.observe_pipeline(pipeline)
task = await tracer.register_task_handlers(task, transport=transport)
```

`setup_pipecat_tracing` / `NoveumTraceObserver` is the older low-level path; keep
it only as an advanced option.

### CrewAI — `setup_crewai_tracing`

```python
import noveum_trace
from noveum_trace.integrations.crewai import setup_crewai_tracing

noveum_trace.init(api_key="...", project="my-crew")  # required first

listener = setup_crewai_tracing()  # registers with CrewAI's global event bus
try:
    crew.kickoff()
finally:
    listener.shutdown()   # detaches the listener
    noveum_trace.flush()  # shutdown() does not flush the SDK
```

`setup_crewai_tracing()` raises `RuntimeError` if `noveum_trace.init()` has not
been called. The listener registers with CrewAI's global event bus on
construction (`NoveumCrewAIListener` subclasses `crewai.events.BaseEventListener`),
so do **not** assign it to `crew.callback_function` — that field does not exist
on current `Crew` versions and raises `ValueError`. Wrap `crew.kickoff()` in
`try/finally`, call `listener.shutdown()` to detach, then `noveum_trace.flush()`
so buffered spans are delivered before a short-lived process exits.

### LiveKit — `setup_livekit_tracing`

```python
import noveum_trace
from noveum_trace.integrations.livekit import setup_livekit_tracing

noveum_trace.init(api_key="...", project="voice-agent")

setup_livekit_tracing(session)  # record=True captures full conversation audio
```

### LangChain / LangGraph — `NoveumTraceCallbackHandler`

```python
import noveum_trace
from noveum_trace import NoveumTraceCallbackHandler

noveum_trace.init(api_key="...", project="my-app")

handler = NoveumTraceCallbackHandler()  # create a fresh handler per request
result = chain.invoke({"text": "..."}, config={"callbacks": [handler]})
```

The handler keeps internal state, so instantiate a new one per concurrent
execution (`asyncio.gather`, thread pools). Sequential calls may reuse one
handler. For LangGraph, `NoveumTraceCallbackHandler(use_langchain_assigned_parent=True)`
resolves parent/child spans from LangChain's `parent_run_id`.

Short-lived scripts should call `noveum_trace.flush()` before the process exits
so buffered spans are delivered.

## 4. Capture / privacy flag reference

Noveum Trace can capture prompts, responses, tool inputs/outputs, tool schemas,
transcripts, conversation history, and audio depending on configuration. Every
flag below defaults to capturing (`True`) unless noted; document how to disable
capture whenever an upstream audience may handle sensitive data.

### CrewAI — `setup_crewai_tracing(**kwargs)` → `NoveumCrewAIListener`

All default `True`: `capture_inputs`, `capture_outputs`, `capture_llm_messages`,
`capture_tool_schemas`, `capture_agent_snapshot`, `capture_crew_snapshot`,
`capture_memory`, `capture_knowledge`, `capture_a2a`, `capture_mcp`,
`capture_flow`, `capture_reasoning`, `capture_guardrails`, `capture_streaming`,
`capture_thinking`. Non-capture defaults: `trace_name_prefix="crewai"`,
`verbose=False`.

### LiveKit — `setup_livekit_tracing(session, *, ...)`

`enabled=True`, `trace_name_prefix=None`, `record=True`,
`cleanup_audio_files=True`. (LiveKit uses `record`, not `record_audio`.)
`record=False` only stops the wrapper from forcing `session.start(record=True)`;
if the application starts its own `RecorderIO`, conversation audio can still be
uploaded. To disable the LiveKit integration entirely (no start-method wrapping,
no event handlers, no audio upload), pass `enabled=False`.

### Pipecat — `NoveumPipecatTracer(...)`

`record_audio=True`, `record_raw_input_audio=True`, `capture_custom_spans=True`,
`auto_enable_metrics=True`, `capture_errors=True`, `capture_system_logs=False`,
`capture_session_metadata=True`, plus any `NoveumTraceObserver` kwarg. The
observer also exposes `capture_text=True` (LLM/TTS text) and
`capture_function_calls=True` (tool calls). Set `record_audio` /
`record_raw_input_audio` / `capture_text` to `False` to reduce what is stored.

### LangChain / LangGraph — `NoveumTraceCallbackHandler`

No per-field capture toggles; the handler captures prompts, responses, and tool
results by default. Reduce exposure at the application layer (redaction, or not
attaching the handler to sensitive chains).

### Disabling capture (quick reference)

```python
# CrewAI: turn off every capture channel
listener = setup_crewai_tracing(
    capture_inputs=False,
    capture_outputs=False,
    capture_llm_messages=False,
    capture_tool_schemas=False,
    capture_agent_snapshot=False,
    capture_crew_snapshot=False,
    capture_memory=False,
    capture_knowledge=False,
    capture_a2a=False,
    capture_mcp=False,
    capture_flow=False,
    capture_reasoning=False,
    capture_guardrails=False,
    capture_streaming=False,
    capture_thinking=False,
)

# Pipecat: no audio, no LLM/TTS text, no tool-call capture
tracer = NoveumPipecatTracer(
    record_audio=False,
    record_raw_input_audio=False,
    capture_text=False,
    capture_function_calls=False,
)

# LiveKit: disable the integration entirely (record=False alone is not enough)
setup_livekit_tracing(session, enabled=False)

# LangChain / LangGraph: omit the handler on chains that handle sensitive data
result = sensitive_chain.invoke({"text": "..."})  # no callbacks=[handler]
```

## 5. Upstream positioning language

**Use:** observability integration, community-maintained, observer, callback
handler, listener, external tracing processor, trace processor.

**Avoid:** native, first-party, plugin, "officially supported by <upstream>", or
any wording implying the upstream project maintains or endorses the integration.

## 6. Per-PR acceptance checklist

Before opening an upstream docs/listing PR, confirm:

- [ ] Every snippet runs against the released `noveum-trace` on PyPI (no
      unreleased APIs).
- [ ] The install extra referenced actually exists in `pyproject.toml`.
- [ ] Every symbol name matches the package exports exactly.
- [ ] The text states the integration is community-maintained (not native /
      first-party / officially supported).
- [ ] Links to the Noveum docs, the `noveum-trace` repo, and the PyPI page are
      included.
- [ ] No API keys or secrets appear in any example.
- [ ] Privacy/payload-capture note and Python version-compatibility note are
      included.
- [ ] The PR is docs-only unless the upstream maintainers explicitly agree to
      accept integration code.
