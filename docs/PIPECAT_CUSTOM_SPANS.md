# Custom Spans in Pipecat Pipelines

This guide covers how to emit your own Noveum spans from inside a Pipecat pipeline — alongside the spans that `NoveumTraceObserver` produces automatically.

---

## Why this needs special handling

Pipecat runs every processor in the pipeline as its own asyncio Task. Noveum's `get_current_trace()` uses Python's `ContextVar` system, which is **task-local** — each Task gets its own isolated copy of the context at the moment it is created.

This means:

```text
NoveumTraceObserver task  →  creates trace, calls set_current_trace(trace)
YourCustomProcessor task  →  calls get_current_trace() → returns None  ❌
```

The observer's write never crosses the Task boundary. The fix is to **not go through `get_current_trace()` at all** — instead, hold a direct reference to the observer and read its `_trace` field. A plain object reference has no notion of task ownership, so both tasks see the same value.

---

## Pattern: custom FrameProcessor with a Noveum span

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

---

### 2. Wire it into the pipeline

The observer must be created **before** the processor, since the processor holds a reference to it.

```python
import noveum_trace
from noveum_trace.integrations.pipecat import NoveumTraceObserver

noveum_trace.init(
    api_key="...",
    project="my-voice-bot",
)

# Create observer first
trace_obs = NoveumTraceObserver(record_audio=True)

# Pass it to your custom processor
stt_logger = STTLogger(trace_obs)

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

task = PipelineTask(pipeline, observers=[trace_obs])
await trace_obs.attach_to_task(task)
```

---

## Where your spans appear in the trace

Custom spans created this way share the same `trace_id` as all the Pipecat spans. They appear as top-level children of the trace (siblings of the turn spans), because the observer does not push individual turn/LLM spans onto the ContextVar stack.

```text
pipecat.conversation  (trace root)
├── pipecat.turn
│   ├── pipecat.stt
│   ├── pipecat.llm
│   └── pipecat.tts
├── custom.stt.transcription   ← your span, same trace_id
└── ...
```

If you need your span nested inside a specific turn span, pass the turn span explicitly:

```python
span = trace.create_span(
    name="custom.stt.transcription",
    parent_span_id=self._observer._current_turn_span.span_id,
    attributes={...},
)
```

---

## If your custom logic lives inside a subclassed STT service

The same rule applies. A subclass of the STT service still runs in its own pipeline Task, separate from the observer's Task. Pass the observer reference in via `__init__` the same way:

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

The inheritance hierarchy does not affect Task isolation — position in the pipeline determines the Task boundary, not the class hierarchy.

---

## Summary

| Approach | Works? | Why |
|---|---|---|
| `noveum_trace.get_current_trace()` inside a pipeline processor | No | ContextVar is task-local; observer's write is invisible here |
| `self._observer._trace` inside a pipeline processor | Yes | Plain object reference; no task boundary |
| Passing the observer via `__init__` | Yes | Same as above; direct pointer to shared heap object |
