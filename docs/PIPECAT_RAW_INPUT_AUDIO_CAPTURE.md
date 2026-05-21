# Capturing pre-filter (raw mic) audio in pipecat pipelines

> **Status:** implemented. Use `Noveum*Transport` wrappers and `stt.raw_audio_uuid`.

## Problem

The `NoveumTraceObserver` captures input audio for the STT span by appending
`UserAudioRawFrame` / `InputAudioRawFrame` instances to `_stt_audio_buffer`
when they reach the observer's frame dispatch table
(`_handlers_stt.py::_handle_user_audio`). The audio is **not raw**.

Pipecat applies `audio_in_filter` in place, then calls `push_frame`:

```python
# pipecat/transports/base_input.py (audio task handler)
if self._params.audio_in_filter:
    frame.audio = await self._params.audio_in_filter.filter(frame.audio)
...
await self.push_frame(frame)   # observer only sees frames after this
```

Downstream analysis using `stt.audio_uuid` therefore scores transcripts against
**post-filter** audio. For filter efficacy work, you need bytes from before the
filter runs.

## Implemented approach: transport subclass + mixin

We override `push_audio_frame` on input transports — the single hook on
`BaseInputTransport` that runs before any filter or queue.

### Public API

- **`NoveumRawAudioTapMixin`** — apply to custom `BaseInputTransport` subclasses.
- **11 `Noveum*Transport` composites** — drop-in replacements for stock pipecat
  transports (Daily, LiveKit, SmallWebRTC, FastAPI/WebSocket × 3, LocalAudio, Tk,
  Tavus, HeyGen, LemonSlice). Bare `_Noveum*InputTransport` classes are private.

### Observer changes

- `_stt_raw_audio_buffer` parallels `_stt_audio_buffer`.
- `capture_raw_input_audio(frame)` snapshots `bytes(frame.audio)` into a fresh
  `InputAudioRawFrame` so in-place filter mutation does not overwrite raw bytes.
- On `TranscriptionFrame`, post-filter and raw WAVs upload concurrently;
  attributes `stt.audio_uuid` and `stt.raw_audio_uuid` are set on success.
- `record_raw_input_audio=True` by default (opt-out).
- `pipecat_span_status` is **all-or-nothing** across enabled uploads.
- Both buffers share `MAX_STT_AUDIO_FRAMES` (oldest frames dropped on overflow).

### Customer wiring

```python
from noveum_trace.integrations.pipecat import NoveumDailyTransport, setup_pipecat_tracing

observer = setup_pipecat_tracing(record_audio=True, record_raw_input_audio=True)
transport = NoveumDailyTransport(..., noveum_observer=observer)
```

Custom transports: see [PIPECAT_CUSTOM_TRANSPORTS.md](./PIPECAT_CUSTOM_TRANSPORTS.md).

Pipelines that keep stock `DailyTransport` (etc.) behave exactly as before —
post-filter `stt.audio_uuid` only.

## Alternatives considered

### Filter wrapper (Option A1)

Wrap `audio_in_filter` in a delegating `BaseAudioFilter` that tees bytes before
calling the inner filter. Useful for filter-specific A/B work; **deferred** as a
complement, not a replacement (misses transports without `audio_in_filter`).

### Monkey-patch `BaseInputTransport.push_audio_frame`

Zero customer wiring but global mutation, import-order fragility, and silent
breakage on pipecat refactors. **Rejected** except as a documented last resort.

### Per-transport SDK callbacks

Bespoke overrides per transport with no benefit over `push_audio_frame`. **Rejected.**

## Out of scope

- Conversation-level `AudioBufferProcessor` stereo WAV (unchanged).
- Public exports of bare `_Noveum*InputTransport` classes (add later if needed).
