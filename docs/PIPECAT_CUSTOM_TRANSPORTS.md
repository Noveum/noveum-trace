# Custom Pipecat transports — raw input audio capture

Use this guide when your pipeline uses a **proprietary or in-house input transport**
that is not one of the stock `Noveum*Transport` wrappers shipped in
`noveum_trace.integrations.pipecat`:

- `NoveumDailyTransport`
- `NoveumLiveKitTransport`
- `NoveumSmallWebRTCTransport`
- `NoveumFastAPIWebsocketTransport`
- `NoveumWebsocketServerTransport`
- `NoveumWebsocketClientTransport`
- `NoveumLocalAudioTransport` (requires pyaudio)
- `NoveumTkTransport` (requires tkinter)
- `NoveumTavusTransport`
- `NoveumHeyGenTransport`
- `NoveumLemonSliceTransport`

For custom input transports, use `NoveumRawAudioTapMixin` instead of a wrapper above.

For background on why pre-filter capture exists and how it differs from the
existing post-filter `stt.audio_uuid` path, see
[PIPECAT_RAW_INPUT_AUDIO_CAPTURE.md](./PIPECAT_RAW_INPUT_AUDIO_CAPTURE.md).

## Why you need the mixin

Noveum buffers post-filter STT audio from frames that reach the observer after
`audio_in_filter` runs. Raw (pre-filter) bytes must be tapped earlier, at
`BaseInputTransport.push_audio_frame`, before the filter mutates `frame.audio`
in place.

Stock transports are covered by the wrappers listed above. Custom transports
require a one-line subclass.

## Subclass recipe

```python
from noveum_trace.integrations.pipecat import NoveumRawAudioTapMixin
from my_company.transports import AcmeInputTransport

class TappedAcmeInputTransport(NoveumRawAudioTapMixin, AcmeInputTransport):
    pass
```

**MRO rule:** the mixin must come **first** (`NoveumRawAudioTapMixin, AcmeInputTransport`).
Misordering raises `TypeError` at class definition time.

## Wiring the observer

Pass the active `NoveumTraceObserver` into the input transport constructor:

```python
import noveum_trace
from noveum_trace.integrations.pipecat import setup_pipecat_tracing

noveum_trace.init(api_key="...", project="my-bot")
observer = setup_pipecat_tracing(record_audio=True, record_raw_input_audio=True)

input_transport = TappedAcmeInputTransport(
    ...,
    noveum_observer=observer,
)
```

If you also have a composite transport class with an `.input()` factory, override
`.input()` to construct your tapped input class and pass `noveum_observer=`.

## What “raw” means

- **Included:** bytes as Pipecat sees them at `push_audio_frame`, before
  `audio_in_filter` (Krisp, Koala, RNNoise, AIC, …).
- **Not included:** transport-SDK noise suppression, hardware AEC, or other
  processing that runs before Pipecat builds `InputAudioRawFrame`.

## Constraints

- Use **one** input transport instance per observer for raw capture (one raw buffer).
- Call `await observer.attach_to_task(task)` so the observer can detect an STT
  processor in the pipeline; without STT, raw buffering is a no-op.
- `record_raw_input_audio` defaults to `True`; set `False` on the observer to
  disable raw capture while keeping post-filter `stt.audio_uuid`.
