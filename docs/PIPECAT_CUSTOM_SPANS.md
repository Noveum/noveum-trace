# Custom Spans in Pipecat Pipelines

This guide covers how to fold your own spans into a Pipecat conversation trace —
alongside the spans that `NoveumPipecatTracer` produces automatically.

There are two ways to do it:

| Method | When to use | Import in your code |
|---|---|---|
| **OTEL spans** (`capture_custom_spans=True`) | You want zero coupling — emit plain OpenTelemetry spans from business logic | `opentelemetry` only |
| **Direct observer reference** (`tracer.observer._trace`) | You're already writing a custom `FrameProcessor` / subclassed service and want explicit control over span placement | `noveum_trace` |

Both share the same `trace_id` as the Pipecat spans. Pick whichever fits your
code; you can use both in the same pipeline.

---

## Method 1 — Plain OpenTelemetry spans (recommended)

You emit **plain OpenTelemetry spans** from anywhere in your code; the tracer
captures them and nests them under the active `pipecat.turn`. Your business logic
needs **no `noveum_trace` import** — it only depends on OpenTelemetry.

### Installation

OTEL-span capture uses an OTEL `SpanProcessor`, so install the `pipecat-otel`
extra:

```bash
pip install "noveum-trace[pipecat-otel]"
```

### Enable capture

Pass `capture_custom_spans=True` to the tracer. The OTEL `SpanProcessor` is
registered inside `observe_pipeline()`:

```python
import noveum_trace
from noveum_trace.integrations.pipecat import NoveumPipecatTracer

noveum_trace.init(api_key="...", project="my-voice-bot")

tracer = NoveumPipecatTracer(record_audio=True, capture_custom_spans=True)

pipeline = tracer.observe_pipeline(pipeline)   # registers the SpanProcessor
task = PipelineTask(pipeline, params=PipelineParams(...))
task = await tracer.register_task_handlers(task, transport=transport)
```

### Emit a span

Use the standard OpenTelemetry API anywhere — inside a function-call handler, a
custom `FrameProcessor`, a subclassed service, or plain business logic:

```python
from opentelemetry import trace as otel_trace

_tracer = otel_trace.get_tracer(__name__)


async def add_item_to_order(params):
    item = params.arguments.get("item")
    price = MENU[item]

    # Plain OTEL span — captured automatically and nested under the active
    # pipecat.turn. No noveum_trace import needed here.
    with _tracer.start_as_current_span("menu.price_lookup") as span:
        span.set_attribute("menu.item", item)
        span.set_attribute("menu.price", price)
        span.set_attribute("menu.quantity", params.arguments.get("quantity", 1))

    ...
```

That's all. The `NoveumCustomSpanProcessor` intercepts the span on export and
attaches it to the active Noveum conversation trace.

### Notes

- Spans are captured on export, so always close them — use the
  `with _tracer.start_as_current_span(...)` context manager (or call
  `span.end()`) so the span is flushed.
- Attribute values follow OTEL rules: strings, numbers, booleans, and sequences
  of those. Complex objects should be serialized to a string first.
- `capture_custom_spans` defaults to `False`. Enable it only when you actually
  emit custom spans; it adds an OTEL `SpanProcessor` to the process.
- If the `pipecat-otel` extra is not installed, the tracer logs a warning and
  custom-span capture stays inactive — the rest of tracing is unaffected.

---

## Method 2 — Direct observer reference (no OTEL)

If you'd rather not depend on OpenTelemetry, you can create Noveum spans directly
from inside a custom `FrameProcessor` (or a subclassed service) by holding a
reference to the tracer's observer and reading its `_trace` field.

### Why this needs special handling

Pipecat runs every processor in the pipeline as its own asyncio Task. Noveum's
`get_current_trace()` uses Python's `ContextVar` system, which is **task-local** —
each Task gets its own isolated copy of the context at the moment it is created.

This means:

```text
NoveumTraceObserver task  →  creates trace, calls set_current_trace(trace)
YourCustomProcessor task  →  calls get_current_trace() → returns None  ❌
```

The observer's write never crosses the Task boundary. The fix is to **not go
through `get_current_trace()` at all** — instead, hold a direct reference to the
observer and read its `_trace` field. A plain object reference has no notion of
task ownership, so both tasks see the same value.

### 1. Define your processor

```python
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from noveum_trace.integrations.pipecat import NoveumTraceObserver


class STTLogger(FrameProcessor):
    """
    Example: emit a custom Noveum span for every final STT transcription.
    Receives the observer by reference so it can access _trace directly,
    bypassing the ContextVar task-isolation issue.
    """

    def __init__(self, observer: NoveumTraceObserver):
        super().__init__()
        self._observer = observer

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            trace = self._observer._trace          # direct reference, not get_current_trace()
            if trace is not None:
                span = trace.create_span(
                    name="custom.stt.transcription",
                    attributes={
                        "stt.transcript": frame.text,
                        "stt.user_id": frame.user_id,
                        "stt.language": str(frame.language) if frame.language else None,
                        "stt.word_count": len(frame.text.split()),
                        "stt.finalized": frame.finalized,
                        # add any attributes relevant to your use case
                    },
                )
                span.finish()

        await self.push_frame(frame, direction)    # always forward the frame
```

Two things to notice:
- `self._observer._trace` instead of `noveum_trace.get_current_trace()` — this is the key difference.
- `await self.push_frame(frame, direction)` must always be called, otherwise the frame never reaches the next processor.

### 2. Wire it into the pipeline

The tracer owns the observer (`tracer.observer`). Create the tracer **before** the
processor, since the processor holds a reference to the observer.

```python
import noveum_trace
from noveum_trace.integrations.pipecat import NoveumPipecatTracer

noveum_trace.init(
    api_key="...",
    project="my-voice-bot",
)

# Create the tracer first, then hand its observer to your custom processor
tracer = NoveumPipecatTracer(record_audio=True)
stt_logger = STTLogger(tracer.observer)

pipeline = Pipeline([
    transport.input(),
    stt,
    stt_logger,          # sits right after STT
    user_aggregator,
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])

pipeline = tracer.observe_pipeline(pipeline)
task = PipelineTask(pipeline, params=PipelineParams(...))
task = await tracer.register_task_handlers(task, transport=transport)
```

### Where your spans appear in the trace

Custom spans created this way share the same `trace_id` as all the Pipecat spans.
They appear as top-level children of the trace (siblings of the turn spans),
because the observer does not push individual turn/LLM spans onto the ContextVar
stack.

```text
pipecat.conversation  (trace root)
├── pipecat.turn
│   ├── pipecat.stt
│   ├── pipecat.llm
│   └── pipecat.tts
├── custom.stt.transcription   ← your span, same trace_id
└── ...
```

If you need your span nested inside a specific turn span, pass the turn span
explicitly:

```python
span = trace.create_span(
    name="custom.stt.transcription",
    parent_span_id=self._observer._current_turn_span.span_id,
    attributes={...},
)
```

### If your custom logic lives inside a subclassed service

The same rule applies. A subclass of the STT service still runs in its own
pipeline Task, separate from the observer's Task. Pass the observer reference in
via `__init__` the same way:

```python
class TracingSTTService(DeepgramSTTService):

    def __init__(self, observer: NoveumTraceObserver, **kwargs):
        super().__init__(**kwargs)
        self._observer = observer

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            trace = self._observer._trace
            if trace is not None:
                span = trace.create_span(
                    name="custom.stt.transcription",
                    attributes={"stt.transcript": frame.text},
                )
                span.finish()
```

The inheritance hierarchy does not affect Task isolation — position in the
pipeline determines the Task boundary, not the class hierarchy.

### Summary

| Approach | Works? | Why |
|---|---|---|
| `noveum_trace.get_current_trace()` inside a pipeline processor | No | ContextVar is task-local; observer's write is invisible here |
| `tracer.observer._trace` inside a pipeline processor | Yes | Plain object reference; no task boundary |
| Passing the observer via `__init__` | Yes | Same as above; direct pointer to shared heap object |
