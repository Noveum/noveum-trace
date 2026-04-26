# AGENTS.md — Noveum Trace SDK complete reference for AI agents and IDEs

This file is a **machine-readable, code-verified reference** derived from the source
(`src/noveum_trace/`) as of v1.5.11. Do **not** invent any API not listed here.

---

## Installation

```bash
# Core SDK — Python 3.9+
pip install noveum-trace

# Framework extras (install what you actually use)
pip install "noveum-trace[langchain]"    # LangChain / LangGraph callback handler
pip install "noveum-trace[livekit]"     # LiveKit STT/TTS/LLM wrappers + session tracing
pip install "noveum-trace[pipecat]"     # Pipecat pipeline observer
pip install "noveum-trace[crewai]"      # CrewAI listener — Python 3.10+ required
pip install "noveum-trace[openai]"      # OpenAI SDK extras
pip install "noveum-trace[anthropic]"   # Anthropic SDK extras
```

---

## Initialization

```python
import noveum_trace

noveum_trace.init(
    api_key="your-api-key",          # or NOVEUM_API_KEY env var
    project="my-app",                # or NOVEUM_PROJECT env var
    environment="production",        # optional — default "development"
    endpoint="https://api.noveum.ai/api",  # optional — override for self-hosted
    transport_config={               # optional — tune batching
        "batch_size": 50,
        "batch_timeout": 2.0,
        "retry_attempts": 3,
        "timeout": 30,
    },
    tracing_config={                 # optional — tune tracing
        "sample_rate": 1.0,
        "capture_errors": True,
        "capture_stack_traces": False,
    },
)
# init() is idempotent — calling it again when already initialized is a no-op.
# All configuration must be set on the first call.
```

---

## SpanStatus values (IMPORTANT)

`SpanStatus` enum has exactly **three** valid string values. Any other value
silently fails when passed to `set_status()`:

```python
"ok"     # Operation completed successfully
"error"  # Operation failed
"unset"  # Status not set (default)
```

Always use `"ok"` not `"success"`. Example:

```python
span.set_status("ok")
span.set_status("error", "Something went wrong")
```

---

## Core context manager reference

### `trace_llm_call` — LLM call tracing

Returns an `LLMContextManager`. The `with` block yields `self` (the context manager),
**not** a bare `Span`. Use span methods directly on it.

```python
from noveum_trace import trace_llm_call
from openai import OpenAI

client = OpenAI()

with trace_llm_call(
    model="gpt-4",
    provider="openai",
    operation="chat_completion",   # optional label
) as span:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )

    # Option A — automatic extraction (recommended)
    span.capture_response(response)  # sets tokens, costs, finish_reason automatically

    # Option B — manual
    span.set_attributes({
        "llm.input_tokens": response.usage.prompt_tokens,
        "llm.output_tokens": response.usage.completion_tokens,
        "llm.total_tokens": response.usage.total_tokens,
    })

    # Option C — convenience helper (also calculates cost automatically)
    span.set_usage_attributes(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )
```

**`LLMContextManager` span methods** (all valid on the `span` above):
- `span.set_input_attributes(**kwargs)` — stores as `llm.input.<key>`
- `span.set_output_attributes(**kwargs)` — stores as `llm.output.<key>`
- `span.set_usage_attributes(input_tokens, output_tokens, total_tokens, cost)` — sets tokens + calculates cost
- `span.capture_response(response)` — auto-extracts everything from OpenAI/Anthropic/Google response objects
- `span.set_attribute(key, value)` — single attribute
- `span.set_attributes({"key": value})` — batch attributes
- `span.add_event(name, attributes)` — point-in-time event
- `span.record_exception(exception)` — record an exception
- `span.set_status("ok" | "error" | "unset", message="")` — set span status

---

### `trace_agent_operation` — agent operation tracing

```python
from noveum_trace import trace_agent_operation

with trace_agent_operation(
    agent_type="researcher",     # type label, e.g. "planner", "writer"
    operation="document_search", # optional operation name
    capabilities=["web_search"],  # optional list
) as span:
    result = agent.search(query)
    span.set_attribute("result.count", len(result))
```

**Note:** The root export is `trace_agent_operation`. You can also use
`from noveum_trace.context_managers import trace_agent` (internal name).

---

### `trace_operation` — generic operation tracing

```python
from noveum_trace import trace_operation

with trace_operation(
    "database_query",
    attributes={"query.table": "users"},  # optional initial attributes
) as span:
    rows = db.query(sql)
    span.set_attribute("query.rows", len(rows))
```

---

### `trace_batch_operation` — batch processing

`batch_size` is **required** (positional int):

```python
from noveum_trace import trace_batch_operation

documents = load_documents()

with trace_batch_operation("process_documents", len(documents)) as span:
    results = [process(doc) for doc in documents]
    span.set_attributes({
        "batch.succeeded": sum(1 for r in results if r),
        "batch.failed": sum(1 for r in results if not r),
    })
```

---

### `trace_pipeline_stage` — pipeline stage tracing

Parameter is `stage_index` (not `stage_number`):

```python
from noveum_trace import trace_pipeline_stage

with trace_pipeline_stage("embed_query", pipeline_id="rag_v2", stage_index=0) as span:
    embedding = embed(query)
    span.set_attribute("embedding.dims", len(embedding))

with trace_pipeline_stage("vector_search", pipeline_id="rag_v2", stage_index=1) as span:
    results = vector_db.search(embedding, top_k=5)
    span.set_attribute("results.count", len(results))

with trace_pipeline_stage("generate_answer", pipeline_id="rag_v2", stage_index=2) as span:
    answer = llm.generate(query, results)
    span.set_attribute("answer.length", len(answer))
```

---

### `create_child_span` — explicit child span

```python
from noveum_trace import trace_operation, create_child_span

with trace_operation("parent_task") as parent_span:
    with create_child_span(parent_span, "sub_task") as child_span:
        result = do_sub_task()
        child_span.set_attribute("result", result)
```

---

### `streaming_llm` — streaming LLM context manager

The `with` block yields a `StreamingSpanManager`. Call `add_token(token)` per chunk:

```python
from noveum_trace import streaming_llm
from openai import OpenAI

client = OpenAI()

with streaming_llm(model="gpt-4", provider="openai") as stream_mgr:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            stream_mgr.add_token(token)
            full_response += token
```

---

### `trace_streaming` — wrap an existing stream iterator

```python
from noveum_trace import trace_streaming

stream = openai_client.chat.completions.create(..., stream=True)

with trace_streaming(stream, model="gpt-4", provider="openai") as traced:
    for chunk in traced:
        print(chunk.choices[0].delta.content or "", end="")
```

---

## Manual trace/span control

Use `client.create_contextual_trace` and `client.create_contextual_span` when you need
full lifecycle control. These are context managers:

```python
import noveum_trace

noveum_trace.init(...)
client = noveum_trace.get_client()  # raises InitializationError if not init'd

# Context manager style (preferred)
with client.create_contextual_trace("document_pipeline") as trace:
    trace.set_attribute("pipeline.version", "2.0")

    with client.create_contextual_span("extract_text") as span:
        text = extract(doc)
        span.set_attribute("text.length", len(text))

    with client.create_contextual_span("classify") as span:
        label = classify(text)
        span.set_attribute("classification.label", label)

noveum_trace.flush()
```

`start_trace(name)` is a convenience wrapper returning a `ContextualTrace` (same as
`client.create_contextual_trace(name)`). Use it as a context manager only:

```python
with noveum_trace.start_trace("my_trace") as trace:
    trace.set_attribute("key", "value")
```

---

## LangChain / LangGraph integration

```bash
pip install "noveum-trace[langchain]"
```

```python
import noveum_trace
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler
# Equivalent paths:
# from noveum_trace import NoveumTraceCallbackHandler          (when langchain installed)
# from noveum_trace.integrations import NoveumTraceCallbackHandler

noveum_trace.init(api_key="your-api-key", project="langchain-app")

handler = NoveumTraceCallbackHandler()
# handler = NoveumTraceCallbackHandler(
#     use_langchain_assigned_parent=True,       # default True — use LangChain parent_run_id
#     prioritize_manually_assigned_parents=False, # default False
# )

# Pass to LLM constructor (LLM-level callbacks)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[handler])
response = llm.invoke("What is AI?")

# Or pass via config at invocation (chain/graph-level callbacks, preferred for LCEL)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_template("{topic}") | ChatOpenAI() | StrOutputParser()
result = chain.invoke({"topic": "AI"}, config={"callbacks": [handler]})

# LangGraph — same pattern
from langgraph.graph import StateGraph, END

workflow = StateGraph(dict)
workflow.add_node("agent", agent_node)
app = workflow.compile()
result = app.invoke({"input": "Hello"}, config={"callbacks": [handler]})

# IMPORTANT: Do NOT share a handler instance across concurrent operations
# (threads, asyncio.gather). Create a new handler per concurrent call:
import asyncio
results = await asyncio.gather(
    chain.ainvoke({"topic": "AI"}, config={"callbacks": [NoveumTraceCallbackHandler()]}),
    chain.ainvoke({"topic": "ML"}, config={"callbacks": [NoveumTraceCallbackHandler()]}),
)

noveum_trace.flush()
```

**What gets traced automatically:** LLM calls (model, prompts, tokens, cost, latency),
chains (input/output flow), agents (decisions, tool usage), tools (function calls,
results), LangGraph nodes (transitions, routing), retrieval operations.

---

## LiveKit integration

```bash
pip install "noveum-trace[livekit]"
# also install your STT/TTS plugins:
pip install livekit-plugins-deepgram livekit-plugins-cartesia livekit-plugins-openai
```

### Recommended: `setup_livekit_tracing` (session-level, automatic)

Hooks into `AgentSession` events and creates a trace for the entire session. All
LLM calls, STT transcriptions, TTS synthesis, tool executions, and conversation
history are captured automatically when `session.start()` is called.

```python
import noveum_trace
from noveum_trace.integrations.livekit import setup_livekit_tracing
from livekit.agents import Agent, AgentSession, JobContext

noveum_trace.init(api_key="your-api-key", project="voice-agent")

async def entrypoint(ctx: JobContext):
    from livekit.plugins import openai

    session = AgentSession(llm=openai.LLM(model="gpt-4o-mini"))

    # Minimal setup — trace is created automatically at session.start()
    setup_livekit_tracing(session)

    await session.start(
        agent=Agent(instructions="You are a helpful voice assistant."),
        room=ctx.room,
    )
```

**Full signature:**
```python
setup_livekit_tracing(
    session,                         # AgentSession — required positional arg
    *,                               # all following are keyword-only
    enabled: bool = True,            # set False to disable tracing without removing code
    trace_name_prefix: str | None = None,   # default "livekit"
    record: bool = True,             # force RecorderIO recording for audio capture
    cleanup_audio_files: bool = True, # delete local audio files after upload
) -> _LiveKitTracingManager
# NO metadata= parameter
```

---

### STT wrapper: `LiveKitSTTWrapper`

Wraps any LiveKit-compatible STT provider. Captures audio files, transcripts,
confidence scores, and timing per utterance.

```python
import noveum_trace
from noveum_trace.integrations.livekit import (
    LiveKitSTTWrapper,
    LiveKitTTSWrapper,
    setup_livekit_tracing,
    extract_job_context,
)
from livekit.agents import Agent, AgentSession, JobContext
from livekit.plugins import deepgram, cartesia, openai

noveum_trace.init(api_key="your-api-key", project="voice-agent")

async def entrypoint(ctx: JobContext):
    session_id = ctx.job.id

    # Optional: extract LiveKit job context for enriching trace attributes
    job_ctx = await extract_job_context(ctx)
    # job_ctx is a dict like:
    # {
    #   "job_id": "...", "room_name": "...", "room_sid": "...",
    #   "job_room_name": "...", "job_room_sid": "..."
    # }

    # Wrap STT provider
    traced_stt = LiveKitSTTWrapper(
        stt=deepgram.STT(model="nova-2", language="en-US"),
        session_id=session_id,
        job_context=job_ctx,   # optional — enriches span attributes
    )

    # Wrap TTS provider
    traced_tts = LiveKitTTSWrapper(
        tts=cartesia.TTS(model="sonic-english"),
        session_id=session_id,
        job_context=job_ctx,   # optional
    )

    session = AgentSession(
        stt=traced_stt,
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=traced_tts,
    )

    # Wire session-level tracing on top of wrapper tracing
    setup_livekit_tracing(session)

    await session.start(
        agent=Agent(instructions="You are a helpful voice assistant."),
        room=ctx.room,
    )
```

**Constructor signatures:**
```python
LiveKitSTTWrapper(
    stt,             # any LiveKit STT provider (deepgram.STT, assemblyai.STT, etc.)
    session_id: str, # used to group audio files in storage
    job_context: dict | None = None,  # extra attributes to attach to each span
)

LiveKitTTSWrapper(
    tts,             # any LiveKit TTS provider (cartesia.TTS, elevenlabs.TTS, etc.)
    session_id: str,
    job_context: dict | None = None,
)
```

---

### LLM wrapper: `LiveKitLLMWrapper`

Wraps a LiveKit LLM provider for detailed per-completion tracing (input messages,
sampling params, tokens, TTFT, tool calls):

```python
from noveum_trace.integrations.livekit import LiveKitLLMWrapper
from livekit.plugins import openai

traced_llm = LiveKitLLMWrapper(
    llm=openai.LLM(model="gpt-4o-mini"),
    session_id=session_id,
    job_context=job_ctx,  # optional
)

session = AgentSession(
    stt=traced_stt,
    llm=traced_llm,   # use wrapped LLM
    tts=traced_tts,
)
setup_livekit_tracing(session)
```

---

### `extract_job_context` — extract LiveKit job metadata

```python
from noveum_trace.integrations.livekit import extract_job_context

# Must be awaited — async function
job_ctx = await extract_job_context(ctx)   # ctx is livekit.agents.JobContext
# Returns a plain dict suitable for job_context= params of wrappers
```

---

## Pipecat integration

```bash
pip install "noveum-trace[pipecat]"
```

The observer **always uses the globally initialised Noveum client** — call
`noveum_trace.init()` before creating the pipeline.

### Using `setup_pipecat_tracing` (recommended factory)

```python
import noveum_trace
from noveum_trace.integrations.pipecat import setup_pipecat_tracing
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

noveum_trace.init(api_key="your-api-key", project="pipecat-bot")

async def main():
    pipeline = Pipeline([...])  # your processors

    observer = setup_pipecat_tracing(
        trace_name_prefix="my-bot",   # default "pipecat"
        record_audio=True,            # default True
        capture_text=True,            # default True
        capture_function_calls=True,  # default True
    )

    task = PipelineTask(pipeline, observers=[observer])

    # IMPORTANT: always call attach_to_task before runner.run()
    await observer.attach_to_task(task)

    runner = PipelineRunner()
    await runner.run(task)
```

### Direct instantiation: `NoveumTraceObserver`

```python
from noveum_trace.integrations.pipecat import NoveumTraceObserver

obs = NoveumTraceObserver(
    trace_name_prefix="my-bot",       # default "pipecat"
    record_audio=True,                # buffer audio → upload WAV per span
    capture_text=True,                # accumulate LLM/TTS text buffers
    capture_function_calls=True,      # create child spans for tool calls
    turn_end_timeout_secs=3.0,        # seconds to wait after bot stops speaking
    # Optional — supply pre-constructed observers:
    turn_tracking_observer=None,      # Pipecat TurnTrackingObserver
    latency_observer=None,            # Pipecat UserBotLatencyObserver
)
task = PipelineTask(pipeline, observers=[obs])
await obs.attach_to_task(task)
```

### With audio capture (`AudioBufferProcessor`)

To capture the full stereo conversation WAV (user left, bot right), add
`AudioBufferProcessor` to your pipeline. `attach_to_task` auto-detects it:

```python
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

audio_buffer = AudioBufferProcessor(num_channels=2)
pipeline = Pipeline([..., audio_buffer, ...])

observer = setup_pipecat_tracing(record_audio=True)
task = PipelineTask(pipeline, observers=[observer])
await observer.attach_to_task(task)   # auto-wires audio_buffer
```

**Span hierarchy produced:**
```
Trace: pipecat.conversation
  Span: pipecat.turn           (one per conversation turn)
    Span: pipecat.stt
    Span: pipecat.llm
      attributes: llm.thoughts[], llm.function_calls[], llm.function_call_results[]
    Span: pipecat.tts
```

**Important:** `setup_pipecat_tracing()` raises `TypeError` if you pass `api_key=`
or `project=` — configure those with `noveum_trace.init()` instead.

---

## CrewAI integration

```bash
pip install "noveum-trace[crewai]"   # requires Python 3.10+
```

### Using `setup_crewai_tracing` (recommended factory)

```python
import noveum_trace
from noveum_trace.integrations.crewai import setup_crewai_tracing
# Also available at:
# from noveum_trace import setup_crewai_tracing           (when crewai installed)
# from noveum_trace.integrations import setup_crewai_tracing
from crewai import Agent, Task, Crew, Process

noveum_trace.init(api_key="your-api-key", project="crewai-app")

researcher = Agent(
    role="Research Analyst",
    goal="Research AI industry trends",
    backstory="Expert at finding and synthesizing research",
    verbose=True,
)
writer = Agent(
    role="Content Writer",
    goal="Write compelling content from research",
    backstory="Expert at turning research into narratives",
    verbose=True,
)

research_task = Task(
    description="Research the latest LLM developments",
    expected_output="Summary of 5 key developments",
    agent=researcher,
)
writing_task = Task(
    description="Write a blog post from the research",
    expected_output="500-word blog post",
    agent=writer,
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
)

# Attach tracing listener
listener = setup_crewai_tracing()          # uses global client from init()
crew.callback_function = listener
result = crew.kickoff()

noveum_trace.flush()
```

### Direct instantiation: `NoveumCrewAIListener`

Use this when you need per-crew configuration or want to pass the client explicitly:

```python
from noveum_trace import get_client
from noveum_trace.integrations.crewai import NoveumCrewAIListener

client = get_client()

listener = NoveumCrewAIListener(
    client,
    capture_inputs=True,            # default True — task prompts, tool args
    capture_outputs=True,           # default True — LLM responses, task results
    capture_llm_messages=True,      # default True — full message history
    capture_tool_schemas=True,      # default True — tool definitions
    capture_agent_snapshot=True,    # default True — agent goal/backstory at start
    capture_crew_snapshot=True,     # default True — crew agents/tasks at kickoff
    capture_memory=True,            # default True — memory query/save operations
    capture_knowledge=True,         # default True — knowledge integration events
    capture_a2a=True,               # default True — agent-to-agent delegation
    capture_mcp=True,               # default True — MCP server calls
    capture_flow=True,              # default True — CrewAI Flow events
    capture_reasoning=True,         # default True — reasoning/thinking steps
    capture_guardrails=True,        # default True — guardrail checks
    capture_streaming=True,         # default True — accumulate streaming chunks
    capture_thinking=True,          # default True — extended thinking tokens
    trace_name_prefix="crewai",     # default "crewai"
    verbose=False,                  # default False — debug logging
)

crew.callback_function = listener
result = crew.kickoff()
```

**What gets traced automatically:**

| Event | Captured data |
|---|---|
| Crew kickoff / completion | Inputs, final output, execution time |
| Agent execution | Role, goal, task assignment, iteration count |
| Task execution | Description, assigned agent, output, status |
| LLM calls | Model, tokens, cost, latency, full messages |
| Tool calls | Name, inputs, outputs, execution time |
| Agent-to-agent delegation | Source/target agent, task details |
| Memory operations | Query/save events |
| MCP server calls | Server name, tool called, result |
| Flow events | CrewAI Flow state transitions |
| Reasoning steps | Chain-of-thought tokens |
| Guardrail evaluations | Validation results, pass/fail |

---

## Streaming callbacks

```python
from noveum_trace import create_openai_streaming_callback, create_anthropic_streaming_callback
from openai import OpenAI

client = OpenAI()

# Create a reusable callback
callback = create_openai_streaming_callback("gpt-4")

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
traced_stream = callback(stream)
for chunk in traced_stream:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## Thread management (multi-turn conversations)

```python
import noveum_trace
from noveum_trace import create_thread, trace_thread_llm
from openai import OpenAI

client = OpenAI()
noveum_trace.init(...)

# Create a persistent conversation thread
thread = create_thread(
    thread_id="user-session-abc",   # optional — auto-generated if omitted
    name="customer_support_session",
    metadata={"customer_id": "cust_123", "channel": "chat"},
)

# Use as context manager and add messages
with thread:
    thread.add_message("user", "I need help with my order")

    with trace_thread_llm(thread, "gpt-4", "openai") as llm_span:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=thread.get_messages(),
        )
        answer = response.choices[0].message.content
        llm_span.set_attributes({
            "llm.input_tokens": response.usage.prompt_tokens,
            "llm.output_tokens": response.usage.completion_tokens,
        })

    thread.add_message("assistant", answer)

# Later: retrieve or delete
retrieved = noveum_trace.get_thread("user-session-abc")
all_threads = noveum_trace.list_threads()
noveum_trace.delete_thread("user-session-abc")
```

**`ThreadContext` methods:**
- `thread.add_message(role, content, metadata=None)` — add to thread history
- `thread.get_messages(limit=None, role_filter=None)` — retrieve messages
- `thread.turn_count` — number of user turns so far

---

## Agent graph / workflow (multi-agent registry)

For tracking complex multi-agent systems with explicit topology:

```python
import noveum_trace
from noveum_trace import (
    create_agent, create_agent_graph, create_agent_workflow,
    trace_agent_operation,
)

# Create agents
planner = create_agent(
    agent_id="planner",
    agent_type="planning_agent",
    capabilities=["task_decomposition"],
)
researcher = create_agent(
    agent_id="researcher",
    agent_type="research_agent",
    capabilities=["web_search", "analysis"],
)

# Use as context managers during execution
with planner:
    with trace_agent_operation(agent_type="planning_agent", operation="decompose") as span:
        plan = decompose_task(task)
        span.set_attribute("plan.steps", len(plan))

with researcher:
    with trace_agent_operation(agent_type="research_agent", operation="search") as span:
        results = search(plan)
        span.set_attribute("results.count", len(results))

# Alternatively, use trace_agent_op(agent_node, operation) which links
# the span to the agent registry node automatically:
# from noveum_trace import trace_agent_op
# with planner:
#     with trace_agent_op(planner, "decompose") as span:
#         ...

noveum_trace.flush()
```

---

## APIs that do NOT exist — never generate these

```python
# Wrong root import names
from noveum_trace import trace_llm          # use trace_llm_call
from noveum_trace import trace_tool         # use trace_operation
from noveum_trace import trace_function_calls  # does not exist
from noveum_trace import trace_retrieval    # does not exist
from noveum_trace import NovaEval           # does not exist

# Two different agent-tracing exports — do NOT confuse them:
# trace_agent_operation(agent_type="...", operation="...")  — string params, no AgentNode
# trace_agent_op(agent_node, operation="...")               — needs an AgentNode instance
trace_agent_op("researcher", "search")     # WRONG — first arg must be an AgentNode
trace_agent_operation(agent_node, "search") # WRONG — first arg must be a string

# Wrong init arguments
trace_llm_call(..., input="...", output="...")  # not constructor args
configure_sampling(...)                         # does not exist
noveum_trace.fastapi.NoveumMiddleware           # module does not exist

# Wrong LangChain imports (legacy, removed in v0.3+)
from langchain.tools import Tool               # use from langchain_core.tools import Tool
from langchain.schema import HumanMessage      # use from langchain_core.messages import HumanMessage
from langchain.tools import DuckDuckGoSearchRun # use from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

# Wrong LiveKit
setup_livekit_tracing(session, metadata=job_metadata)   # no metadata param
setup_livekit_tracing(session, metadata={"key": "val"}) # same — no metadata

# Wrong Pipecat
setup_pipecat_tracing(api_key="...", project="...")  # raises TypeError
NoveumTraceObserver(api_key="...")                   # raises TypeError

# Wrong span status
span.set_status("success")   # silently fails — use "ok"
span.set_status("SUCCESS")   # silently fails — use "ok"

# Wrong trace/span lifecycle (these methods don't exist on Trace/Span)
trace.start_span(name="...")   # wrong — use client.create_contextual_span()
span.end()                     # wrong — use client.finish_span(span) or context manager
trace.end()                    # wrong — use client.finish_trace(trace) or context manager

# Plugin system — exists but raises NotImplementedError
register_plugin(...)           # raises NotImplementedError
list_plugins()                 # raises NotImplementedError
```

---

## Utility functions

```python
noveum_trace.flush()          # block until all pending traces are sent
noveum_trace.shutdown()       # flush + teardown (releases resources)
noveum_trace.is_initialized() # True/False
noveum_trace.get_client()     # NoveumClient — raises InitializationError if not init'd
noveum_trace.get_config()     # current Config object
```

---

## Example files

The `docs/examples/` directory contains runnable examples from the test suite:

| File | What it demonstrates |
|---|---|
| `basic_usage.py` | Core context managers, nested spans |
| `flexible_tracing_example.py` | Mixed context manager patterns |
| `langchain_integration_example.py` | `NoveumTraceCallbackHandler` basics |
| `langgraph_agent_example.py` | LangGraph graph + callback handler |
| `langgraph_routing_example.py` | LangGraph routing tracking |
| `langchain_long_conversation_example.py` | Multi-turn LangChain tracing |
| `livekit_integration_example.py` | STT/TTS wrappers + `setup_livekit_tracing` |
| `pipecat_integration_example.py` | `NoveumTraceObserver` + `attach_to_task` |
| `crewai_e2e_test.py` | Full CrewAI crew with listener |
| `streaming_example.py` | `streaming_llm`, `trace_streaming`, callbacks |
| `thread_example.py` | `create_thread`, `trace_thread_llm` |
| `agent_example.py` | `create_agent`, agent graph patterns |
| `comprehensive_tool_calling_examples.py` | Tool call tracing |
| `image_example.py` | Multimodal / image tracing |

---

## Source of truth

- Public API: `src/noveum_trace/__init__.py` → `__all__`
- Optional extras: `pyproject.toml` → `[project.optional-dependencies]`
- Integration exports: `src/noveum_trace/integrations/__init__.py`
- `SpanStatus` values: `src/noveum_trace/core/span.py` → `class SpanStatus(Enum)`
