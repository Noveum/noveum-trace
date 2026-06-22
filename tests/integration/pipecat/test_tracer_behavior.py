"""
Tracer two-call API (§G, TR-1..12) — value-asserting regression tests.

Subsystem: ``NoveumPipecatTracer`` (``integrations/pipecat/tracer.py``).  The
tracer emits **no spans of its own** — it is pure wiring: pipeline rebuild
(auto-insert ``AudioBufferProcessor``), placing the observer in the real
``TaskObserver`` proxy list, auto-enabling metrics flags, tapping the transport
for pre-filter raw audio, and stamping session metadata onto the root trace.

These tests drive REAL Pipecat objects (``Pipeline`` / ``PipelineTask`` /
``AudioBufferProcessor``) plus a fake transport, then assert concrete values:
pipeline identity & processor ordering, observer membership, the exact
``session.*`` metadata keys, the tap-install marker, call ordering, and the
``*args``/``**kwargs`` passthrough of the tap — per PIPECAT_TEST_PLAN.md §G.
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("pipecat.frames.frames")

pytestmark = pytest.mark.integration


def _tracer(**kwargs: Any) -> Any:
    pytest.importorskip("pipecat.observers.base_observer")
    from noveum_trace.integrations.pipecat import NoveumPipecatTracer

    return NoveumPipecatTracer(**kwargs)


def _is_abp(proc: Any) -> bool:
    return any(b.__name__ == "AudioBufferProcessor" for b in type(proc).__mro__)


# --------------------------------------------------------------------------- #
# Local fakes (conftest's FakeInputTransport records only the frame; TR-12     #
# needs args/kwargs, and TR-3 needs a __setattr__ hook — define them here).    #
# --------------------------------------------------------------------------- #
class RecordingInput:
    """Input transport whose ``push_audio_frame`` records ``(frame, args, kwargs)``
    and returns a sentinel so the tap's passthrough can be proven."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    async def push_audio_frame(self, frame: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((frame, args, kwargs))
        return "original-return-value"


class RecordingTransport:
    def __init__(self, *, room_url: str | None = None) -> None:
        self._input = RecordingInput()
        if room_url is not None:
            self.room_url = room_url

    def input(self) -> RecordingInput:
        return self._input


# --------------------------------------------------------------------------- #
# TR-1 — observe_pipeline gate reads the OBSERVER's _record_audio              #
# --------------------------------------------------------------------------- #
def test_observe_pipeline_gate_is_observer_record_audio(make_pipeline: Any) -> None:
    # Guards: a refactor that adds a tracer _record_audio copy and reads the wrong one.
    tracer = _tracer(record_audio=True)
    tracer.observer._record_audio = False  # mutate the OBSERVER's flag only
    pipeline = make_pipeline(2)

    result = tracer.observe_pipeline(pipeline)

    assert result is pipeline  # no rebuild — the observer's flag gates it
    assert not any(_is_abp(p) for p in result._processors)


# --------------------------------------------------------------------------- #
# TR-2 — nested pipeline: ABP appended at OUTER level; entries reused by id    #
# --------------------------------------------------------------------------- #
def test_observe_pipeline_nested_appends_abp_at_outer_level() -> None:
    # Guards: a rebuild that flattens/re-wraps nested pipelines (breaks RTVI layouts).
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.frame_processor import FrameProcessor

    class _P(FrameProcessor):
        pass

    inner_pipe = Pipeline([_P(), _P()])
    outer_proc = _P()
    pipeline = Pipeline([inner_pipe, outer_proc])

    tracer = _tracer(record_audio=True)
    result = tracer.observe_pipeline(pipeline)

    assert result is not pipeline  # rebuilt
    inner = result._processors[1:-1]  # strip auto source/sink
    # Customer's outer entries are the SAME instances, in order (no descent/clone)
    assert inner[0] is inner_pipe
    assert inner[1] is outer_proc
    # ABP appended at the tail of the OUTER level only
    assert _is_abp(inner[-1])
    assert sum(1 for p in inner if _is_abp(p)) == 1


# --------------------------------------------------------------------------- #
# TR-3 — register_task_handlers ordering: store_transport -> tap -> attach     #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_register_task_handlers_ordering(make_pipeline: Any) -> None:
    # Guards: a reorder where first frames bypass capture or metadata isn't
    # available at connection time.
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    order: list[str] = []

    class _OrderInput:
        def __setattr__(self, name: str, value: Any) -> None:
            # The tap is the inline assignment `input.push_audio_frame = _patched`.
            if name == "push_audio_frame":
                order.append("tap")
            object.__setattr__(self, name, value)

        async def push_audio_frame(self, frame: Any, *a: Any, **k: Any) -> Any:
            return "orig"

    class _OrderTransport:
        def __init__(self) -> None:
            self.room_url = "https://room"
            self._input = _OrderInput()

        def input(self) -> Any:
            return self._input

    tracer = _tracer(record_audio=True, record_raw_input_audio=True)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline, params=PipelineParams())

    orig_store = tracer.observer._store_transport
    orig_attach = tracer.observer.attach_to_task

    def _wrapped_store(*a: Any, **k: Any) -> Any:
        order.append("store_transport")
        return orig_store(*a, **k)

    async def _wrapped_attach(*a: Any, **k: Any) -> Any:
        order.append("attach_to_task")
        return await orig_attach(*a, **k)

    tracer.observer._store_transport = _wrapped_store  # type: ignore[method-assign]
    tracer.observer.attach_to_task = _wrapped_attach  # type: ignore[method-assign]

    await tracer.register_task_handlers(task, transport=_OrderTransport())

    assert order == ["store_transport", "tap", "attach_to_task"]


# --------------------------------------------------------------------------- #
# TR-4 — _store_transport populates real session.* keys from transport+runner  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_store_transport_populates_session_metadata(make_pipeline: Any) -> None:
    # Guards: dropping the runner_args= forward (untested at tracer level).
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    tracer = _tracer(record_audio=False)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline, params=PipelineParams())
    runner_args = types.SimpleNamespace(pipeline_idle_timeout_secs=30, bot_name="b")

    await tracer.register_task_handlers(
        task,
        transport=RecordingTransport(room_url="https://room"),
        runner_args=runner_args,
    )

    meta = tracer.observer._session_metadata
    assert meta["session.transport_type"] == "RecordingTransport"
    assert meta["session.room_url"] == "https://room"
    assert meta["session.idle_timeout_secs"] == 30
    assert meta["session.bot_name"] == "b"


# --------------------------------------------------------------------------- #
# TR-5 — no session metadata when capture_session_metadata=False               #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_session_metadata_opt_out(make_pipeline: Any) -> None:
    # Guards: the tracer-level opt-out gate.
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    tracer = _tracer(record_audio=False, capture_session_metadata=False)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline, params=PipelineParams())

    await tracer.register_task_handlers(
        task, transport=RecordingTransport(room_url="x")
    )

    meta = tracer.observer._session_metadata
    assert meta == {}
    assert not any(k.startswith("session.") for k in meta)


# --------------------------------------------------------------------------- #
# TR-6 — raw-audio tap requires BOTH _record_audio AND _record_raw_input_audio #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record_audio,record_raw_input_audio,expect_tap",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
async def test_raw_audio_tap_requires_both_flags(
    make_pipeline: Any,
    record_audio: bool,
    record_raw_input_audio: bool,
    expect_tap: bool,
) -> None:
    # Guards: AND->OR or a wrong-flag read in the tap-install gate.
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    tracer = _tracer(
        record_audio=record_audio, record_raw_input_audio=record_raw_input_audio
    )
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline, params=PipelineParams())
    transport = RecordingTransport(room_url="https://room")

    await tracer.register_task_handlers(task, transport=transport)

    inp = transport.input()
    assert getattr(inp, "_noveum_tap_applied", False) is expect_tap
    if expect_tap:
        # Positive case: the wrapped push_audio_frame still forwards & returns.
        from pipecat.frames.frames import InputAudioRawFrame

        frame = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
        ret = await inp.push_audio_frame(frame)
        assert ret == "original-return-value"
        assert inp.calls[-1][0] is frame


# --------------------------------------------------------------------------- #
# TR-7 — metrics auto-enable degrades: no inject when attr absent; no crash    #
#        on read-only params                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_metrics_autoenable_skips_when_attr_absent() -> None:
    # Guards: a refactor to getattr(...,False) that spuriously injects attrs.
    class _NoMetricsParams:
        pass  # no enable_metrics / enable_usage_metrics attributes at all

    class _Task:
        turn_tracking_observer = None

        def __init__(self, params: Any) -> None:
            self._params = params

        def add_observer(self, o: Any) -> None:
            pass

    params = _NoMetricsParams()
    task = _Task(params)
    tracer = _tracer(
        record_audio=False,
        record_raw_input_audio=False,
        capture_session_metadata=False,
        auto_enable_metrics=True,
    )

    returned = await tracer.register_task_handlers(task)  # type: ignore[arg-type]

    assert returned is task
    # The metrics block reads getattr(params,"enable_metrics",True): the default
    # True means `not True` is False, so nothing is set — no spurious attribute.
    assert not hasattr(params, "enable_metrics")
    assert not hasattr(params, "enable_usage_metrics")


@pytest.mark.asyncio
async def test_metrics_autoenable_swallows_readonly_params() -> None:
    # Guards: removal of the try/except — a crash on slots/read-only params.
    class _ReadOnlyParams:
        @property
        def enable_metrics(self) -> bool:
            return False

        @enable_metrics.setter
        def enable_metrics(self, value: bool) -> None:
            raise AttributeError("read-only")

        @property
        def enable_usage_metrics(self) -> bool:
            return False

        @enable_usage_metrics.setter
        def enable_usage_metrics(self, value: bool) -> None:
            raise AttributeError("read-only")

    class _Task:
        turn_tracking_observer = None

        def __init__(self, params: Any) -> None:
            self._params = params

        def add_observer(self, o: Any) -> None:
            pass

    task = _Task(_ReadOnlyParams())
    tracer = _tracer(
        record_audio=False,
        record_raw_input_audio=False,
        capture_session_metadata=False,
        auto_enable_metrics=True,
    )

    returned = await tracer.register_task_handlers(task)  # type: ignore[arg-type]

    assert returned is task  # swallowed, no crash
    assert task._params.enable_metrics is False  # still read-only False


# --------------------------------------------------------------------------- #
# TR-8 — observe_and_create_task forwards transport AND runner_args            #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_observe_and_create_task_forwards_transport_and_runner_args(
    make_pipeline: Any,
) -> None:
    # Guards: dropping transport=/runner_args= in the convenience wrapper.
    from pipecat.pipeline.task import PipelineParams

    tracer = _tracer(record_audio=True, record_raw_input_audio=False)
    pipeline = make_pipeline(2)
    runner_args = types.SimpleNamespace(pipeline_idle_timeout_secs=12)
    transport = RecordingTransport(room_url="r")

    task = await tracer.observe_and_create_task(
        pipeline,
        transport=transport,
        runner_args=runner_args,
        params=PipelineParams(),
    )

    # Observer wired into the real proxy list.
    assert tracer.observer in task._observer._observers
    # Metrics auto-enabled.
    assert task._params.enable_metrics is True
    assert task._params.enable_usage_metrics is True
    # ABP auto-inserted AND detected.
    assert tracer.observer._audio_buffer_processor is not None
    # Session metadata flowed through both transport AND runner_args.
    meta = tracer.observer._session_metadata
    assert meta["session.room_url"] == "r"
    assert meta["session.idle_timeout_secs"] == 12


# --------------------------------------------------------------------------- #
# TR-9 — capture_custom_spans registers a processor; failure leaves wrapping   #
# --------------------------------------------------------------------------- #
def test_capture_custom_spans_registers_processor(make_pipeline: Any) -> None:
    # Guards: a custom-span registration error breaking pipeline wrapping.
    tracer = _tracer(record_audio=False, capture_custom_spans=True)
    sentinel = object()
    with patch(
        "noveum_trace.integrations.pipecat.custom_spans.register_custom_span_processor",
        return_value=sentinel,
    ):
        result = tracer.observe_pipeline(make_pipeline(2))

    assert tracer._span_processor is sentinel
    assert result is not None  # pipeline still returned


def test_capture_custom_spans_registration_failure_keeps_pipeline(
    make_pipeline: Any,
) -> None:
    # Guards: a registration exception must NOT break observe_pipeline wrapping.
    tracer = _tracer(record_audio=False, capture_custom_spans=True)
    pipeline = make_pipeline(2)
    with patch(
        "noveum_trace.integrations.pipecat.custom_spans.register_custom_span_processor",
        side_effect=RuntimeError("registration boom"),
    ):
        result = tracer.observe_pipeline(pipeline)

    # record_audio=False → no rebuild → original pipeline returned unchanged.
    assert result is pipeline
    assert tracer._span_processor is None  # stays None when registration raised


# --------------------------------------------------------------------------- #
# TR-10 — turn-tracking fallback engages external tracking (observe-then-pin)  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_turn_tracking_fallback_engages_external_tracking(
    make_pipeline: Any,
) -> None:
    # Guards: a silent change to whether the fallback engages external tracking
    # (which gates the orphan-span behavior). Observe-then-pin, not assert-the-fix.
    from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
    from pipecat.pipeline.task import PipelineTask

    tracer = _tracer(record_audio=False)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline, enable_turn_tracking=False)
    assert task.turn_tracking_observer is None  # precondition

    await tracer.register_task_handlers(task)

    # A real fallback TurnTrackingObserver was created and attached to the observer.
    fallback = getattr(task, "_turn_tracking_observer", None)
    assert isinstance(fallback, TurnTrackingObserver)
    assert tracer.observer._turn_tracker is fallback
    # PIN current design: attaching the fallback turn tracker engages external
    # turn tracking (gates orphan-span fallback behavior).
    assert tracer.observer._using_external_turn_tracking is True


# --------------------------------------------------------------------------- #
# TR-11 — degrades but still attaches when transport.input() returns None      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_register_task_handlers_no_input_transport(make_pipeline: Any) -> None:
    # Guards: an AttributeError on None when a transport has no input side.
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    class _NoInputTransport:
        room_url = "https://room"

        def input(self) -> Any:
            return None

    tracer = _tracer(record_audio=True, record_raw_input_audio=True)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline, params=PipelineParams())

    returned = await tracer.register_task_handlers(task, transport=_NoInputTransport())

    assert returned is task  # no crash, still attaches
    assert tracer.observer in task._observer._observers
    # No tap marker anywhere — there was no input transport to tap.
    assert task in tracer.observer._registered_pipeline_tasks


# --------------------------------------------------------------------------- #
# TR-12 — tap preserves *args/**kwargs passthrough to push_audio_frame         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_raw_audio_tap_preserves_args_kwargs(make_pipeline: Any) -> None:
    # Guards: a tap that drops extra args (some transports pass additional args)
    # while the single-arg test still passes.
    from pipecat.frames.frames import InputAudioRawFrame
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    tracer = _tracer(record_audio=True, record_raw_input_audio=True)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline, params=PipelineParams())
    transport = RecordingTransport(room_url="https://room")

    await tracer.register_task_handlers(task, transport=transport)

    inp = transport.input()
    assert inp._noveum_tap_applied is True  # tap installed
    frame = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)

    ret = await inp.push_audio_frame(frame, "extra", key="v")

    recorded_frame, recorded_args, recorded_kwargs = inp.calls[-1]
    assert recorded_frame is frame
    assert recorded_args == ("extra",)
    assert recorded_kwargs == {"key": "v"}
    assert ret == "original-return-value"  # sentinel propagated back
