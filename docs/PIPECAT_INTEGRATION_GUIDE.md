# Pipecat Integration Guide

Add automatic tracing to your Pipecat voice pipeline in minutes. Every conversation is recorded as a structured trace with per-turn spans for STT, LLM, TTS; tool/function-call details are attached to the LLM span as attributes, along with latency and token usage.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [What Gets Traced](#what-gets-traced)
5. [Configuration Options](#configuration-options)
6. [Capturing pre-filter (raw) audio](#capturing-pre-filter-raw-audio)
7. [Troubleshooting](#troubleshooting)

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

Three changes to your existing pipeline code:

```python
import asyncio

import noveum_trace
from noveum_trace.integrations.pipecat import NoveumTraceObserver

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner

# 1. Initialize noveum-trace once at startup
noveum_trace.init(
    api_key="your-noveum-api-key",
    project="my-voice-bot",
)

# --- your existing pipeline setup ---
pipeline = Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])

async def main():
    # 2. Add NoveumTraceObserver to your PipelineTask
    trace_obs = NoveumTraceObserver()

    task = PipelineTask(
        pipeline,
        observers=[trace_obs],
    )

    # 3. Wire turn tracking — this is required
    await trace_obs.attach_to_task(task)

    runner = PipelineRunner()
    await runner.run(task)


asyncio.run(main())
```

That's it. Traces are flushed automatically when the pipeline ends (`EndFrame` / `CancelFrame`).

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

```python
trace_obs = NoveumTraceObserver(
    # Prefix for the conversation trace name (default: "pipecat")
    # Produces trace named "pipecat.conversation"
    trace_name_prefix="pipecat",

    # Capture LLM input/output text and TTS text in spans (default: True)
    capture_text=True,

    # Record tool/function calls on the existing `pipecat.llm` span as
    # `llm.function_calls` / `llm.function_call_results` lists (default: True)
    capture_function_calls=True,

    # Buffer and upload STT/TTS audio as WAV files per span (default: True)
    # Adds stt.audio_uuid / tts.audio_uuid attributes when enabled
    record_audio=True,

    # Also capture pre-filter mic bytes via Noveum*Transport wrappers (default: True)
    # Adds stt.raw_audio_uuid on the STT span (~2x audio storage per turn)
    record_raw_input_audio=True,
)
```

---

## Capturing pre-filter (raw) audio

By default, `stt.audio_uuid` contains audio **after** Pipecat's `audio_in_filter`
(Krisp, Koala, etc.). To also capture **pre-filter** bytes for STT efficacy or
filter A/B analysis, swap your transport for a `Noveum*Transport` wrapper and pass
the observer:

```python
from noveum_trace.integrations.pipecat import NoveumTraceObserver, NoveumDailyTransport

trace_obs = NoveumTraceObserver(record_audio=True, record_raw_input_audio=True)

transport = NoveumDailyTransport(
    room_url,
    token,
    "Bot",
    DailyParams(...),
    noveum_observer=trace_obs,
)

task = PipelineTask(pipeline, observers=[trace_obs])
await trace_obs.attach_to_task(task)
```

- **`stt.audio_uuid`** — unchanged; post-filter audio the STT path consumed.
- **`stt.raw_audio_uuid`** — pre-filter snapshot (requires `Noveum*Transport` or a
  custom mixin; see [PIPECAT_CUSTOM_TRANSPORTS.md](./PIPECAT_CUSTOM_TRANSPORTS.md)).
- **`record_raw_input_audio`** defaults to `True`; set `False` to opt out.
- Upload status is all-or-nothing: both enabled uploads must succeed for
  `pipecat_span_status="ok"` on the STT span.
- “Raw” means pre-pipecat-filter, not pre-SDK or hardware processing.

Available wrappers: `NoveumDailyTransport`, `NoveumLiveKitTransport`,
`NoveumSmallWebRTCTransport`, `NoveumFastAPIWebsocketTransport`,
`NoveumWebsocketServerTransport`, `NoveumWebsocketClientTransport`,
`NoveumLocalAudioTransport`, `NoveumTkTransport`, `NoveumTavusTransport`,
`NoveumHeyGenTransport`, `NoveumLemonSliceTransport`.

---

## Troubleshooting

**No traces appearing**

- Verify `noveum_trace.init()` is called before the pipeline starts.
- Check that `await trace_obs.attach_to_task(task)` is called (from async code) after `PipelineTask` is constructed and before the runner starts. Missing this call means turn spans won't have accurate boundaries.
- Confirm your API key is correct and the project name matches what you set up in the Noveum dashboard.

**Turn spans missing or not splitting correctly**

- `await trace_obs.attach_to_task(task)` is required for accurate turn tracking. Make sure it is called before `runner.run(task)`.
- Ensure your `PipelineTask` has turn tracking enabled (so `task.turn_tracking_observer` is present for `trace_obs.attach_to_task(task)` to wire external turn boundaries; it is on by default in recent Pipecat versions).

**LLM token counts not appearing**

- Token counts come from Pipecat's `MetricsFrame`. Confirm your LLM processor emits metrics (most standard Pipecat LLM services do).

**Function call spans missing**

- Set `capture_function_calls=True` (the default) on the observer.
- Confirm your LLM processor emits `FunctionCallInProgressFrame` / `FunctionCallResultFrame`.

---
