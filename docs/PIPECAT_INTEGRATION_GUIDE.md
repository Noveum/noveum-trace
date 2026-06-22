# Pipecat Integration Guide

Add automatic tracing to your Pipecat voice pipeline in two calls. Every conversation is recorded as a structured trace with per-turn spans for STT, LLM, TTS; tool/function-call details are attached to the LLM span as attributes, along with latency and token usage.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [What Gets Traced](#what-gets-traced)
5. [Configuration Options](#configuration-options)
6. [Raw (pre-filter) audio capture](#raw-pre-filter-audio-capture)
7. [Custom spans](#custom-spans)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.11+
- A working Pipecat pipeline (`pipecat-ai`)
- A Noveum API key (get one at [noveum.ai](https://noveum.ai))

---

## Installation

```bash
pip install "noveum-trace[pipecat]"
```

---

## Quick Start

Integration is two calls on the `NoveumPipecatTracer`. Your transport, pipeline,
and `PipelineTask` stay stock Pipecat — no wrappers, no class-swaps.

```python
import noveum_trace
from noveum_trace.integrations.pipecat import NoveumPipecatTracer

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# 1. Initialize noveum-trace once at startup
noveum_trace.init(
    api_key="your-noveum-api-key",
    project="my-voice-bot",
)

# 2. Create a tracer
tracer = NoveumPipecatTracer(record_audio=True)


async def main():
    # --- your existing pipeline setup (stock Pipecat) ---
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    # 3. Wrap the pipeline — auto-inserts an AudioBufferProcessor when needed.
    #    You MUST use the return value.
    pipeline = tracer.observe_pipeline(pipeline)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    # 4. Register handlers — adds the observer, wires turn tracking, taps the
    #    transport for raw audio, and stamps session metadata. You MUST use the
    #    return value.
    task = await tracer.register_task_handlers(task, transport=transport)

    runner = PipelineRunner()
    await runner.run(task)


import asyncio
asyncio.run(main())
```

That's it. Traces are flushed automatically when the pipeline ends
(`EndFrame` / `CancelFrame`).

> **Two rules:** always use the return value of both `observe_pipeline()` and
> `register_task_handlers()` — each may return a new/modified object.

### Even shorter

When you don't need to set anything on the `PipelineTask` between wrapping and
registration, collapse both calls into one:

```python
task = await tracer.observe_and_create_task(
    pipeline,
    transport=transport,
    params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
)
```

---

## What Gets Traced

Each pipeline session produces one **conversation trace** containing a **turn span** per conversational exchange. Each turn contains child spans for STT, LLM, and TTS.

```
Trace: pipecat.conversation
├── pipeline.allow_interruptions: true
├── pipeline.sample_rate: 16000
├── Event: client.connected
├── Event: bot.connected
│
├── Span: pipecat.turn  (one per user→bot exchange)
│   ├── turn.number: 1
│   ├── turn.user_input: "What's the weather in Paris?"
│   ├── turn.user_bot_latency_seconds: 0.42
│   ├── turn.duration_seconds: 9.5
│   ├── turn.was_interrupted: false
│   │
│   ├── Span: pipecat.stt
│   │   ├── stt.text: "What's the weather in Paris?"
│   │   ├── stt.model: "nova-2"
│   │   ├── stt.audio_uuid: "…"        (post-filter audio, when record_audio=True)
│   │   └── stt.raw_audio_uuid: "…"   (pre-filter audio, when raw capture enabled)
│   │
│   ├── Span: pipecat.llm
│   │   ├── llm.model: "gpt-4o-mini"
│   │   ├── llm.input: "[{...full message history...}]"
│   │   ├── llm.output: "Today in Paris it's 12°C..."
│   │   ├── llm.input_tokens: 145
│   │   ├── llm.output_tokens: 62
│   │   ├── llm.time_to_first_token_ms: 180
│   │   └── llm.cost.total: 0.00031
│   │
│   └── Span: pipecat.tts
│       ├── tts.input_text: "Today in Paris it's 12°C..."
│       └── tts.time_to_first_byte_ms: 95
│
├── Span: pipecat.turn  (turn 2, turn 3, ...)
│   └── ...
│
└── conversation.turn_count: 3
    conversation.total_input_tokens: 890
    conversation.total_output_tokens: 310
    conversation.total_cost: 0.0021
```

## Configuration Options

All options are passed to the `NoveumPipecatTracer` constructor:

```python
tracer = NoveumPipecatTracer(
    # Buffer and upload STT/TTS/conversation audio as WAV files per span
    # (default: True). Adds stt.audio_uuid / tts.audio_uuid attributes.
    # When True, observe_pipeline() auto-inserts an AudioBufferProcessor
    # at the pipeline tail if one isn't already present.
    record_audio=True,

    # Also capture pre-filter mic bytes by tapping the transport
    # (default: True). Adds stt.raw_audio_uuid on the STT span.
    # See "Raw (pre-filter) audio capture" below.
    record_raw_input_audio=True,

    # Fold plain-OTEL spans emitted anywhere in your code into the active
    # conversation trace, nested under the active turn (default: False).
    # Requires: pip install "noveum-trace[pipecat-otel]". See "Custom spans".
    capture_custom_spans=False,

    # Force PipelineParams.enable_metrics / enable_usage_metrics to True on
    # the task so MetricsFrame (tokens, TTFB, latency) is always emitted,
    # even if you forgot to set them (default: True).
    auto_enable_metrics=True,

    # Record ErrorFrame / FatalErrorFrame as span errors + trace events
    # (default: True).
    capture_errors=True,

    # Record SystemLogFrame entries (warning/error/critical) as span events
    # (default: False — opt-in, volume can be high).
    capture_system_logs=False,

    # Stamp transport type, room URL, and runner idle timeout onto the root
    # trace at connection time (default: True). Pass runner_args= to
    # register_task_handlers to enrich this further.
    capture_session_metadata=True,

    # Any NoveumTraceObserver kwarg is also accepted and forwarded verbatim,
    # e.g.:
    trace_name_prefix="pipecat",      # trace named "pipecat.conversation"
    capture_text=True,                # capture LLM/TTS text in spans
    capture_function_calls=True,      # record tool calls on the pipecat.llm span
)
```

The underlying observer is available as `tracer.observer` for advanced use.

---

## Raw (pre-filter) audio capture

By default, `stt.audio_uuid` contains audio **after** Pipecat's `audio_in_filter`
(Krisp, Koala, RNNoise, …). To also capture **pre-filter** mic bytes — useful for
STT efficacy or filter A/B analysis — leave `record_raw_input_audio=True` (the
default) and pass your transport to `register_task_handlers`:

```python
tracer = NoveumPipecatTracer(record_audio=True, record_raw_input_audio=True)

pipeline = tracer.observe_pipeline(pipeline)
task = PipelineTask(pipeline, params=PipelineParams(...))
task = await tracer.register_task_handlers(task, transport=transport)
```

`register_task_handlers` taps `transport.input().push_audio_frame` — the single
hook that runs before any filter mutates the audio. This works with **any**
transport, stock or custom; no wrapper class is needed.

- **`stt.audio_uuid`** — post-filter audio the STT path consumed.
- **`stt.raw_audio_uuid`** — pre-filter snapshot (set when raw capture is enabled
  and the transport tap succeeds).
- **`record_raw_input_audio`** defaults to `True`; set `False` to opt out.
- Upload status is all-or-nothing: both enabled uploads must succeed for
  `pipecat_span_status="ok"` on the STT span.
- "Raw" means pre-Pipecat-filter, not pre-SDK or hardware processing.
- Requires an STT processor in the pipeline; otherwise raw buffering is a no-op.
  The tap is idempotent (safe to re-run `register_task_handlers`), and the buffer
  is capped (oldest frames dropped on overflow).

---

## Custom spans

To fold your own instrumentation into the conversation trace, emit **plain OpenTelemetry
spans** anywhere in your code and enable `capture_custom_spans=True`. They are
captured automatically and nested under the active `pipecat.turn` — no
`noveum_trace` import needed in your business logic.

```bash
pip install "noveum-trace[pipecat-otel]"
```

```python
tracer = NoveumPipecatTracer(record_audio=True, capture_custom_spans=True)
pipeline = tracer.observe_pipeline(pipeline)   # registers the OTEL SpanProcessor
```

```python
from opentelemetry import trace as otel_trace

_tracer = otel_trace.get_tracer(__name__)

async def add_item_to_order(params):
    with _tracer.start_as_current_span("menu.price_lookup") as span:
        span.set_attribute("menu.item", item)
        span.set_attribute("menu.price", price)
    ...
```

See [PIPECAT_CUSTOM_SPANS.md](./PIPECAT_CUSTOM_SPANS.md) for details and trace
placement.

---

## Troubleshooting

**No traces appearing**

- Verify `noveum_trace.init()` is called before the pipeline starts.
- Make sure you assigned the return values: `pipeline = tracer.observe_pipeline(pipeline)`
  and `task = await tracer.register_task_handlers(task, ...)`. Discarding either
  return value means the wiring is lost.
- Confirm your API key is correct and the project name matches what you set up in the Noveum dashboard.

**Turn spans missing or not splitting correctly**

- Turn tracking is enabled by default in recent Pipecat versions
  (`task.turn_tracking_observer`). `register_task_handlers` also installs a
  fallback `TurnTrackingObserver` if you explicitly built the task with
  `enable_turn_tracking=False`.

**LLM token counts not appearing**

- Token counts come from Pipecat's `MetricsFrame`. With `auto_enable_metrics=True`
  (the default), `register_task_handlers` forces `enable_metrics` /
  `enable_usage_metrics` on the task, so this is handled for you. Confirm your LLM
  processor emits metrics (most standard Pipecat LLM services do).

**Function call spans missing**

- `capture_function_calls=True` (the default) records tool calls on the
  `pipecat.llm` span.
- Confirm your LLM processor emits `FunctionCallInProgressFrame` /
  `FunctionCallResultFrame`.

**Raw audio (`stt.raw_audio_uuid`) missing**

- Pass `transport=` to `register_task_handlers` so the input transport can be
  tapped.
- Keep `record_audio=True` and `record_raw_input_audio=True` (both defaults).

---
