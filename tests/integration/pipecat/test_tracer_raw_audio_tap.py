"""
Tier 2 (C3 / C6) — transport ``push_audio_frame`` tap for raw, pre-filter audio.

Spec mapping (``.cursor/plans/pipecat-plan-tier-2-transport-tap.md``,
``Noveum_Pipecat_SDK_Integration_Spec.md`` C3/C6):

* **C3** — capture STT input audio **without** the ``Noveum*Transport`` class
  swap, so the integration survives the customer's own transport wrappers.
* **C6** — capture user audio at the transport boundary *before* any custom
  processor can swallow ``InputAudioRawFrame``.

``register_task_handlers`` monkeypatches the cached ``transport.input()``
instance's ``push_audio_frame`` so each raw frame is tee'd into
``observer.capture_raw_input_audio`` and then forwarded to the original method.

Plan §3 + spec non-intrusive guarantee require the tap to be **side-band**: a
capture failure must never propagate onto the audio path.  The current
implementation drops the per-frame guard the old mixin had — that gap is encoded
as an ``xfail(strict=True)`` below so it flips green the moment it is fixed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


def _tracer(**kwargs: Any) -> Any:
    pytest.importorskip("pipecat.frames.frames")
    from noveum_trace.integrations.pipecat import NoveumPipecatTracer

    return NoveumPipecatTracer(**kwargs)


async def _make_tapped_task(tracer: Any, transport: Any) -> Any:
    """observe + a real PipelineTask + register_task_handlers(transport=…)."""
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.frame_processor import FrameProcessor

    class _P(FrameProcessor):
        pass

    pipeline = tracer.observe_pipeline(Pipeline([_P(), _P()]))
    task = PipelineTask(pipeline)
    return await tracer.register_task_handlers(task, transport=transport)


# --------------------------------------------------------------------------- #
# Tap installed — capture then original, return value preserved               #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_tap_calls_capture_then_forwards(ff: Any, fake_transport: Any) -> None:
    tracer = _tracer(record_audio=True, record_raw_input_audio=True)
    transport = fake_transport()

    order: list[str] = []
    real_capture = tracer.observer.capture_raw_input_audio

    def _spy(frame: Any) -> None:
        order.append("capture")
        real_capture(frame)

    tracer.observer.capture_raw_input_audio = _spy  # type: ignore[method-assign]

    await _make_tapped_task(tracer, transport)

    inp = transport.input()
    frame = ff.InputAudioRawFrame(audio=b"\x01\x02", sample_rate=16000, num_channels=1)
    ret = await inp.push_audio_frame(frame)

    # capture ran, then the original forwarded the frame…
    assert order == ["capture"]
    assert inp.forwarded == [frame]
    # …and the original return value is preserved (side-band, non-intrusive).
    assert ret == "original-return-value"


@pytest.mark.asyncio
async def test_tap_buffers_raw_audio_end_to_end(ff: Any, fake_transport: Any) -> None:
    """Full C3 path: a frame pushed through the tapped transport lands in the
    observer's raw-audio buffer (when an STT service is present)."""
    tracer = _tracer(record_audio=True, record_raw_input_audio=True)
    transport = fake_transport()
    await _make_tapped_task(tracer, transport)

    # capture_raw_input_audio only buffers when the pipeline has an STT service;
    # our trivial pipeline has none, so set the precondition the same way the
    # observer would after detecting one.
    tracer.observer._pipeline_has_stt = True

    inp = transport.input()
    frame = ff.InputAudioRawFrame(
        audio=b"\x01\x02\x03", sample_rate=16000, num_channels=1
    )
    await inp.push_audio_frame(frame)

    assert len(tracer.observer._stt_raw_audio_buffer) == 1
    assert tracer.observer._stt_raw_audio_buffer[0].audio == b"\x01\x02\x03"


@pytest.mark.asyncio
async def test_customer_push_override_survives_tap(ff: Any) -> None:
    """C3 durability: a customer who overrides ``push_audio_frame`` on their own
    transport still has that override run after the tap (we wrap the bound
    method, we don't replace the class)."""

    calls: list[str] = []

    class CustomInput:
        async def push_audio_frame(self, frame: Any, *a: Any, **k: Any) -> str:
            calls.append("customer-override")
            return "custom-return"

    class CustomTransport:
        def __init__(self) -> None:
            self._inp = CustomInput()

        def input(self) -> CustomInput:
            return self._inp

    tracer = _tracer(record_audio=True, record_raw_input_audio=True)
    transport = CustomTransport()
    await _make_tapped_task(tracer, transport)

    frame = ff.InputAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)
    ret = await transport.input().push_audio_frame(frame)

    assert calls == ["customer-override"]  # override still runs
    assert ret == "custom-return"


@pytest.mark.asyncio
async def test_tap_is_idempotent(ff: Any, fake_transport: Any) -> None:
    # Regression: the inline tap guards the input transport with
    # `_noveum_tap_applied` (Tier-2 plan §3), so re-registering on the same
    # transport must NOT stack wrappers and double-capture each frame.
    tracer = _tracer(record_audio=True, record_raw_input_audio=True)
    transport = fake_transport()

    # Wire the tap twice (e.g. a retry, or observer reuse across attach calls).
    await _make_tapped_task(tracer, transport)
    await _make_tapped_task(tracer, transport)

    tracer.observer.capture_raw_input_audio = MagicMock()  # type: ignore[method-assign]
    frame = ff.InputAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)
    await transport.input().push_audio_frame(frame)

    # One frame in → exactly one capture out.
    assert tracer.observer.capture_raw_input_audio.call_count == 1


# --------------------------------------------------------------------------- #
# Tap skipped — flags off / no transport                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_tap_skipped_when_raw_audio_disabled(
    ff: Any, fake_transport: Any
) -> None:
    tracer = _tracer(record_audio=True, record_raw_input_audio=False)
    transport = fake_transport()
    tracer.observer.capture_raw_input_audio = MagicMock()

    await _make_tapped_task(tracer, transport)

    frame = ff.InputAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)
    await transport.input().push_audio_frame(frame)

    tracer.observer.capture_raw_input_audio.assert_not_called()


@pytest.mark.asyncio
async def test_tap_skipped_when_record_audio_disabled(
    ff: Any, fake_transport: Any
) -> None:
    tracer = _tracer(record_audio=False, record_raw_input_audio=True)
    transport = fake_transport()
    tracer.observer.capture_raw_input_audio = MagicMock()

    await _make_tapped_task(tracer, transport)

    frame = ff.InputAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)
    await transport.input().push_audio_frame(frame)

    tracer.observer.capture_raw_input_audio.assert_not_called()


@pytest.mark.asyncio
async def test_no_transport_does_not_crash(make_pipeline: Any) -> None:
    """No transport handle → no tap, but the rest of the wiring still completes."""
    from pipecat.pipeline.task import PipelineTask

    tracer = _tracer(record_audio=True, record_raw_input_audio=True)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline)

    returned = await tracer.register_task_handlers(task, transport=None)
    assert returned is task
    assert tracer.observer in task._observer._observers


@pytest.mark.asyncio
async def test_transport_input_raises_skips_tap_gracefully(
    make_pipeline: Any, fake_transport: Any
) -> None:
    """If ``transport.input()`` raises, the tap is skipped and the session
    continues (the tap must never take down handler registration)."""
    from pipecat.pipeline.task import PipelineTask

    tracer = _tracer(record_audio=True, record_raw_input_audio=True)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline)
    transport = fake_transport(input_raises=True)

    returned = await tracer.register_task_handlers(task, transport=transport)
    assert returned is task
    assert tracer.observer in task._observer._observers


# --------------------------------------------------------------------------- #
# Side-band guarantee — a capture failure must not reach the audio path        #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_capture_exception_must_not_reach_audio_path(
    ff: Any, fake_transport: Any
) -> None:
    # Regression: the tap wraps capture_raw_input_audio in try/except so a
    # capture failure is side-band and never propagates onto the audio path
    # (Tier-2 plan §3 / spec non-intrusive guarantee).
    tracer = _tracer(record_audio=True, record_raw_input_audio=True)
    transport = fake_transport()
    await _make_tapped_task(tracer, transport)

    # Simulate a capture-side failure (e.g. transient buffer error).
    tracer.observer.capture_raw_input_audio = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("capture blew up")
    )

    inp = transport.input()
    frame = ff.InputAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)

    # The audio path must keep flowing despite the capture failure.
    ret = await inp.push_audio_frame(frame)
    assert ret == "original-return-value"
    assert inp.forwarded == [frame]
