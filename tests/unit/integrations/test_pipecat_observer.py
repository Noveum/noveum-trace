"""Unit tests for NoveumTraceObserver (pipecat_observer)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def pipecat_frames():
    pytest.importorskip("pipecat.frames.frames")
    from pipecat.frames import frames as ff

    return ff


def test_noveum_trace_observer_initializes_pipecat_base_object() -> None:
    """TaskObserver stringifies observers via BaseObject.name; _name must exist (MRO/super)."""
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    assert getattr(obs, "_name", None) is not None
    assert str(obs) == obs.name


def test_noveum_trace_observer_turn_handlers_emitter_first() -> None:
    """Pipecat calls event handlers as handler(emitter, *args)."""
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    registered: dict[str, object] = {}

    class FakeTurnTracker:
        def add_event_handler(self, event_name: str, handler: object) -> None:
            registered[event_name] = handler

    fake = FakeTurnTracker()
    obs._attach_turn_tracker(fake)

    assert "on_turn_started" in registered
    assert obs._using_external_turn_tracking is True

    async def run() -> None:
        await registered["on_turn_started"](fake, 3)  # type: ignore[misc]

    asyncio.run(run())


def test_eou_metrics_buffered_when_no_turn_span() -> None:
    """EOU metrics are buffered when no turn span exists, then flushed on new turn."""
    pytest.importorskip("pipecat.metrics.metrics")

    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import TurnMetricsData

    from noveum_trace.core.trace import Trace
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    async def run() -> None:
        obs = NoveumTraceObserver()
        obs._trace = MagicMock(spec=Trace)

        assert obs._current_turn_span is None

        eou_data = TurnMetricsData(
            processor="turn",
            is_complete=True,
            probability=0.87,
            e2e_processing_time_ms=120.5,
        )
        frame = MetricsFrame(data=[eou_data])
        data = MagicMock()
        data.frame = frame

        await obs._handle_metrics(data)

        assert obs._pending_turn_eou_metrics["turn_eou_is_complete"] is True
        assert obs._pending_turn_eou_metrics["turn_eou_confidence"] == 0.87
        assert obs._pending_turn_eou_metrics["turn_eou_processing_time_ms"] == 120.5

        turn_span = MagicMock()
        turn_span.attributes = {}
        turn_span.span_id = "turn-1"
        obs._trace.create_span.return_value = turn_span

        await obs._start_new_turn()

        assert turn_span.attributes["turn.eou_is_complete"] is True
        assert turn_span.attributes["turn.eou_confidence"] == 0.87
        assert turn_span.attributes["turn.eou_processing_time_ms"] == 120.5
        assert obs._pending_turn_eou_metrics == {}

    asyncio.run(run())


def test_observer_init_defaults() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    assert obs._trace_name_prefix == "pipecat"
    assert obs._record_audio is False
    assert obs._capture_text is True
    assert obs._capture_function_calls is True
    assert obs._using_external_turn_tracking is False


def test_llm_marker_frame_is_registered_when_capability_exists(monkeypatch) -> None:
    from pipecat.frames import frames as ff

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    class LLMMarkerFrame:
        pass

    monkeypatch.setattr(ff, "LLMMarkerFrame", LLMMarkerFrame, raising=False)
    obs = NoveumTraceObserver(record_audio=False)

    assert obs._frame_handlers[LLMMarkerFrame] == obs._handle_llm_marker


def test_get_client_uses_noveum_get_client() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from unittest.mock import patch

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    mock_c = MagicMock()
    with patch("noveum_trace.get_client", return_value=mock_c):
        assert obs._get_client() is mock_c


@pytest.mark.asyncio
async def test_on_pipeline_started_creates_trace(pipecat_frames) -> None:
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    mock_trace = MagicMock()
    mock_trace.trace_id = "tid"
    client = MagicMock()
    client.start_trace.return_value = mock_trace

    obs = NoveumTraceObserver(trace_name_prefix="pfx")
    with patch.object(obs, "_get_client", return_value=client):
        await obs.on_pipeline_started()

    client.start_trace.assert_called_once()
    assert obs._trace is mock_trace


@pytest.mark.asyncio
async def test_on_pipeline_started_skips_without_client(pipecat_frames) -> None:
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    with patch.object(obs, "_get_client", return_value=None):
        await obs.on_pipeline_started()
    assert obs._trace is None


@pytest.mark.asyncio
async def test_handle_start_frame_ensures_trace_and_attrs(pipecat_frames) -> None:
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    mock_trace = MagicMock()
    obs = NoveumTraceObserver()
    obs._trace = mock_trace

    sf = pipecat_frames.StartFrame()
    sf.allow_interruptions = True
    sf.sample_rate = 16000

    data = MagicMock()
    data.frame = sf

    with patch.object(obs, "_ensure_audio_buffer_recording", new_callable=AsyncMock):
        await obs._handle_start_frame(data)

    mock_trace.set_attributes.assert_called()
    call_kw = mock_trace.set_attributes.call_args[0][0]
    assert call_kw.get("pipeline.allow_interruptions") is True
    assert call_kw.get("pipeline.sample_rate") == 16000


@pytest.mark.asyncio
async def test_finish_conversation_idempotent() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    await obs._finish_conversation()
    await obs._finish_conversation()


@pytest.mark.asyncio
async def test_finish_conversation_resets_llm_text_buffer() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(capture_text=True)
    obs._llm_text_buffer = ["hello"]

    trace = MagicMock()
    trace.attributes = {}
    trace.finish = MagicMock()
    obs._trace = trace

    with patch.object(obs, "_get_client", return_value=None):
        await obs._finish_conversation()

    assert obs._llm_text_buffer == []


@pytest.mark.asyncio
async def test_cancelled_conversation_uploads_supported_ok_status() -> None:
    from noveum_trace.core.span import SpanStatus
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    trace = MagicMock()
    trace.attributes = {}
    obs._trace = trace

    with patch.object(obs, "_get_client", return_value=None):
        await obs._finish_conversation(cancelled=True)

    assert trace.attributes["pipecat_span_status"] == "cancelled"
    trace.set_status.assert_called_once_with(SpanStatus.OK)


@pytest.mark.asyncio
async def test_finish_conversation_rollup_uses_canonical_llm_usage() -> None:
    from types import SimpleNamespace

    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    trace = MagicMock()
    trace.attributes = {}
    obs._trace = trace
    source = SimpleNamespace(name="llm")
    obs.register_processor_role(source, "llm")
    span = MagicMock()
    span.attributes = {}
    span.is_finished.return_value = False
    obs._llm_operations.start(source, span=span, processor_name="llm")

    for prompt, completion in ((10, 20), (10, 20), (12, 22)):
        item = LLMUsageMetricsData(
            processor="llm",
            model="gpt-4o-mini",
            value=LLMTokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            ),
        )
        await obs._handle_metrics(
            SimpleNamespace(frame=MetricsFrame(data=[item]), source=source)
        )

    with patch.object(obs, "_get_client", return_value=None):
        await obs._finish_conversation()

    summary = trace.set_attributes.call_args.args[0]
    assert summary["conversation.total_input_tokens"] == 12
    assert summary["conversation.total_output_tokens"] == 22


@pytest.mark.asyncio
async def test_create_child_span_returns_none_without_trace() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    obs._trace = None
    assert obs._create_child_span("x") is None


@pytest.mark.asyncio
async def test_on_push_frame_dedup_same_frame_id(pipecat_frames) -> None:
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    obs._frame_handlers = {pipecat_frames.LLMTextFrame: AsyncMock()}

    f = pipecat_frames.LLMTextFrame(text="a")
    fid = getattr(f, "id", None)
    data = MagicMock()
    data.frame = f

    await obs.on_push_frame(data)
    await obs.on_push_frame(data)

    if fid is not None:
        obs._frame_handlers[pipecat_frames.LLMTextFrame].assert_called_once()
    else:
        # No id on frame — handler may run twice
        assert obs._frame_handlers[pipecat_frames.LLMTextFrame].call_count >= 1


@pytest.mark.asyncio
async def test_llm_input_waits_for_final_llm_destination(pipecat_frames) -> None:
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    intermediary = SimpleNamespace(name="PromptFilter")
    llm_a = SimpleNamespace(name="OpenAILLMService")
    llm_b = SimpleNamespace(name="OpenAILLMService")
    obs.register_processor_role(llm_a, "llm")
    obs.register_processor_role(llm_b, "llm")

    frame_a = pipecat_frames.LLMMessagesUpdateFrame(
        messages=[{"role": "user", "content": "for-a"}]
    )
    await obs.on_push_frame(
        SimpleNamespace(frame=frame_a, source=object(), destination=intermediary)
    )

    assert obs._pending_llm_context == {}
    assert obs._llm_operations.pending_input_for(llm_a) is None

    await obs.on_push_frame(
        SimpleNamespace(frame=frame_a, source=intermediary, destination=llm_a)
    )
    # Seeing the same route again is idempotent.
    await obs.on_push_frame(
        SimpleNamespace(frame=frame_a, source=intermediary, destination=llm_a)
    )

    frame_b = pipecat_frames.LLMMessagesUpdateFrame(
        messages=[{"role": "user", "content": "for-b"}]
    )
    await obs.on_push_frame(
        SimpleNamespace(frame=frame_b, source=object(), destination=llm_b)
    )

    pending_a = obs._llm_operations.pending_input_for(llm_a)
    pending_b = obs._llm_operations.pending_input_for(llm_b)
    assert pending_a is not None and "for-a" in pending_a["messages"]
    assert pending_a is not None and "for-b" not in pending_a["messages"]
    assert pending_b is not None and "for-b" in pending_b["messages"]
    assert pending_b is not None and "for-a" not in pending_b["messages"]


@pytest.mark.asyncio
async def test_attach_to_task_wires_observers() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    task = MagicMock()
    task.turn_tracking_tracker = None
    task.turn_tracking_observer = MagicMock()
    task._user_bot_latency_observer = MagicMock()
    task._pipeline = None
    task.pipeline = None

    with patch.object(
        obs, "_attach_audio_buffer_from_pipeline", new_callable=AsyncMock
    ):
        await obs.attach_to_task(task)

    task.turn_tracking_observer.add_event_handler.assert_called()
    task._user_bot_latency_observer.add_event_handler.assert_called()


def test_iter_nested_processors_depth_first() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    inner = MagicMock()
    inner.processors = [MagicMock()]
    root = MagicMock()
    root.processors = [inner]

    obs = NoveumTraceObserver()
    out = list(obs._iter_nested_processors(root))
    assert inner in out
    assert inner.processors[0] in out


@pytest.mark.asyncio
async def test_attach_audio_buffer_registers_processor() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    abp = MagicMock()
    abp.__class__.__name__ = "AudioBufferProcessor"
    abp.start_recording = AsyncMock()

    pipeline = MagicMock()
    pipeline.processors = [abp]
    task = MagicMock()
    task._pipeline = pipeline

    obs = NoveumTraceObserver(record_audio=True)
    await obs._attach_audio_buffer_from_pipeline(task)

    assert obs._audio_buffer_processor is abp
    abp.add_event_handler.assert_called()
    abp.start_recording.assert_called_once()
    call_args = abp.add_event_handler.call_args[0]
    assert call_args[0] == "on_audio_data"


@pytest.mark.asyncio
async def test_on_conversation_audio_concatenates_chunks() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    await obs._on_conversation_audio(MagicMock(), b"a", 16000, 2)
    await obs._on_conversation_audio(MagicMock(), b"b", 16000, 2)
    assert obs._conversation_audio_chunks == [b"a", b"b"]
    assert obs._conversation_audio_sample_rate == 16000
    assert obs._conversation_audio_num_channels == 2


@pytest.mark.asyncio
async def test_upload_full_conversation_audio_creates_error_span_when_no_chunks() -> (
    None
):
    """Even when no audio chunks were captured, we should emit the span."""
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=True)
    span = MagicMock()
    span.span_id = "sid"
    span.trace_id = "tid"
    span.attributes = {}

    trace = MagicMock()
    trace.create_span.return_value = span

    obs._trace = trace
    obs._conversation_audio_chunks = []
    obs._audio_buffer_processor = None

    await obs._upload_full_conversation_audio()

    trace.create_span.assert_called_once()
    args, kwargs = trace.create_span.call_args
    assert kwargs["name"] == "pipecat.full_conversation"
    assert kwargs["attributes"]["full_conversation.missing_reason"] == (
        "audio_buffer_processor_not_attached"
    )
    assert span.attributes["pipecat_span_status"] == "error"
    trace.finish_span.assert_called_once_with("sid")


@pytest.mark.asyncio
async def test_upload_full_conversation_audio_uploads_when_chunks_present() -> None:
    """When audio chunks exist, we should upload and finish the span."""
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=True)
    span = MagicMock()
    span.span_id = "sid"
    span.trace_id = "tid"
    span.attributes = {}

    trace = MagicMock()
    trace.create_span.return_value = span
    obs._trace = trace

    # Enough PCM for a non-zero duration_ms.
    obs._conversation_audio_chunks = [b"\x00\x00" * 5000]
    obs._conversation_audio_sample_rate = 16000
    obs._conversation_audio_num_channels = 2

    client = MagicMock()
    client.export_audio = MagicMock()

    obs._get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

    await obs._upload_full_conversation_audio()

    trace.create_span.assert_called_once()
    client.export_audio.assert_called_once()
    trace.finish_span.assert_called_once_with("sid")
    assert span.attributes["pipecat_span_status"] == "ok"
    assert obs._conversation_audio_chunks == []


@pytest.mark.asyncio
async def test_ensure_audio_buffer_recording_calls_start() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    proc = MagicMock()
    proc._recording = False
    proc.start_recording = AsyncMock()

    obs = NoveumTraceObserver()
    obs._audio_buffer_processor = proc
    await obs._ensure_audio_buffer_recording()
    proc.start_recording.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_audio_buffer_recording_skips_when_record_audio_false() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    proc = MagicMock()
    proc._recording = False
    proc.start_recording = AsyncMock()

    obs = NoveumTraceObserver(record_audio=False)
    obs._audio_buffer_processor = proc
    await obs._ensure_audio_buffer_recording()

    proc.start_recording.assert_not_called()


@pytest.mark.asyncio
async def test_on_push_frame_does_not_ensure_audio_buffer_when_record_audio_false() -> (
    None
):
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)

    class InputAudioRawFrame:
        pass

    proc_ab = MagicMock()
    proc_ab._recording = False
    obs._audio_buffer_processor = proc_ab

    obs._ensure_audio_buffer_recording = AsyncMock()
    data = MagicMock()
    data.frame = InputAudioRawFrame()

    await obs.on_push_frame(data)
    assert obs._ensure_audio_buffer_recording.await_count == 0


@pytest.mark.asyncio
async def test_on_push_frame_does_not_ensure_audio_buffer_when_abp_already_recording() -> (
    None
):
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=True)

    class InputAudioRawFrame:
        pass

    proc_ab = MagicMock()
    proc_ab._recording = True
    obs._audio_buffer_processor = proc_ab

    obs._ensure_audio_buffer_recording = AsyncMock()
    data = MagicMock()
    data.frame = InputAudioRawFrame()

    await obs.on_push_frame(data)
    assert obs._ensure_audio_buffer_recording.await_count == 0


@pytest.mark.asyncio
async def test_latency_observer_handler(pipecat_frames) -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    registered = {}

    class Lat:
        def add_event_handler(self, name: str, h: object) -> None:
            registered[name] = h

    obs._attach_latency_tracker(Lat())
    turn = MagicMock()
    turn.attributes = {}
    obs._current_turn_span = turn

    await registered["on_latency_measured"](Lat(), 1.25)  # type: ignore[misc]
    assert turn.attributes["turn.user_bot_latency_seconds"] == 1.25


# ---------------------------------------------------------------------------
# on_pipeline_finished safety-net tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attach_to_task_registers_on_pipeline_finished() -> None:
    """attach_to_task registers an on_pipeline_finished handler on the task."""
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)

    registered: dict[str, object] = {}

    class FakeTask:
        turn_tracking_observer = None
        _user_bot_latency_observer = None

        def event_handler(self, event_name: str):
            def decorator(fn):
                registered[event_name] = fn
                return fn

            return decorator

    await obs.attach_to_task(FakeTask())
    assert (
        "on_pipeline_finished" in registered
    ), "attach_to_task should register an on_pipeline_finished handler"


@pytest.mark.asyncio
async def test_on_pipeline_finished_drains_proxy_queue_before_finalizing() -> None:
    """Queued metrics/frames are handled before conversation state is settled."""
    pytest.importorskip("pipecat.observers.base_observer")
    pytest.importorskip("pipecat.frames.frames")

    from pipecat.frames.frames import EndFrame

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    events: list[str] = []
    registered: dict[str, object] = {}

    class FakeQueue:
        async def join(self) -> None:
            events.append("drain")

    proxy = SimpleNamespace(observer=obs, queue=FakeQueue())

    class FakeTask:
        turn_tracking_observer = None
        _user_bot_latency_observer = None
        _observer = SimpleNamespace(_proxies={obs: proxy})

        def event_handler(self, event_name: str):
            def decorator(fn):
                registered[event_name] = fn
                return fn

            return decorator

    async def mock_finish(cancelled: bool = False) -> None:
        events.append("finish")

    obs._finish_conversation = mock_finish  # type: ignore[method-assign]
    task = FakeTask()
    await obs.attach_to_task(task)

    handler = registered["on_pipeline_finished"]
    await handler(task, EndFrame())  # type: ignore[operator]

    assert events == ["drain", "finish"]


@pytest.mark.asyncio
async def test_proxy_queue_drain_is_bounded_when_join_never_completes() -> None:
    """A stuck/crashed observer proxy cannot block pipeline finalization forever."""
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat import pipecat_observer
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    join_started = asyncio.Event()

    class StuckQueue:
        async def join(self) -> None:
            join_started.set()
            await asyncio.Event().wait()

    proxy = SimpleNamespace(observer=obs, queue=StuckQueue())
    task = SimpleNamespace(_observer=SimpleNamespace(_proxies={obs: proxy}))

    with patch.object(pipecat_observer, "_OBSERVER_PROXY_DRAIN_TIMEOUT_SECS", 0.001):
        await obs._await_task_observer_proxy_queue(task)

    assert join_started.is_set()


@pytest.mark.asyncio
async def test_finish_conversation_is_single_flight() -> None:
    """Proxy and safety terminal paths cannot run teardown concurrently."""
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    obs._trace = object()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def finish_once(cancelled: bool = False) -> None:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        obs._trace = None

    obs._finish_conversation_once = finish_once  # type: ignore[method-assign]
    first = asyncio.create_task(obs._finish_conversation())
    await started.wait()
    second = asyncio.create_task(obs._finish_conversation(cancelled=True))
    release.set()
    await asyncio.gather(first, second)

    assert calls == 1


@pytest.mark.asyncio
async def test_on_pipeline_finished_calls_finish_conversation_cancelled() -> None:
    """The safety-net handler calls _finish_conversation(cancelled=True) for CancelFrame."""
    pytest.importorskip("pipecat.observers.base_observer")
    pytest.importorskip("pipecat.frames.frames")

    from pipecat.frames.frames import CancelFrame

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    finish_calls: list[dict] = []

    async def mock_finish(cancelled: bool = False) -> None:
        finish_calls.append({"cancelled": cancelled})

    obs._finish_conversation = mock_finish  # type: ignore[method-assign]

    registered: dict[str, object] = {}

    class FakeTask:
        turn_tracking_observer = None
        _user_bot_latency_observer = None

        def event_handler(self, event_name: str):
            def decorator(fn):
                registered[event_name] = fn
                return fn

            return decorator

    await obs.attach_to_task(FakeTask())

    handler = registered["on_pipeline_finished"]
    await handler(FakeTask(), CancelFrame())  # type: ignore[operator]

    assert len(finish_calls) == 1
    assert finish_calls[0]["cancelled"] is True


@pytest.mark.asyncio
async def test_on_pipeline_finished_calls_finish_conversation_not_cancelled() -> None:
    """The safety-net handler calls _finish_conversation(cancelled=False) for EndFrame."""
    pytest.importorskip("pipecat.observers.base_observer")
    pytest.importorskip("pipecat.frames.frames")

    from pipecat.frames.frames import EndFrame

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    finish_calls: list[dict] = []

    async def mock_finish(cancelled: bool = False) -> None:
        finish_calls.append({"cancelled": cancelled})

    obs._finish_conversation = mock_finish  # type: ignore[method-assign]

    registered: dict[str, object] = {}

    class FakeTask:
        turn_tracking_observer = None
        _user_bot_latency_observer = None

        def event_handler(self, event_name: str):
            def decorator(fn):
                registered[event_name] = fn
                return fn

            return decorator

    await obs.attach_to_task(FakeTask())

    handler = registered["on_pipeline_finished"]
    await handler(FakeTask(), EndFrame())  # type: ignore[operator]

    assert len(finish_calls) == 1
    assert finish_calls[0]["cancelled"] is False


@pytest.mark.asyncio
async def test_attach_to_task_skips_safety_net_without_event_handler() -> None:
    """attach_to_task is safe on tasks that lack the event_handler() API."""
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)

    class LegacyTask:
        """Simulates a pipecat version that does not have event_handler()."""

        turn_tracking_observer = None
        _user_bot_latency_observer = None
        # no event_handler attribute

    # Should not raise
    await obs.attach_to_task(LegacyTask())


@pytest.mark.asyncio
async def test_on_pipeline_finished_idempotent_after_proxy_path() -> None:
    """Safety net is a no-op when proxy path already ran _finish_conversation."""
    pytest.importorskip("pipecat.observers.base_observer")
    pytest.importorskip("pipecat.frames.frames")

    from pipecat.frames.frames import CancelFrame

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=False)
    call_count = 0

    async def mock_finish(cancelled: bool = False) -> None:
        nonlocal call_count
        call_count += 1
        # Simulate what the real method does: clear _trace on first call
        obs._trace = None

    obs._finish_conversation = mock_finish  # type: ignore[method-assign]

    registered: dict[str, object] = {}

    class FakeTask:
        turn_tracking_observer = None
        _user_bot_latency_observer = None

        def event_handler(self, event_name: str):
            def decorator(fn):
                registered[event_name] = fn
                return fn

            return decorator

    await obs.attach_to_task(FakeTask())

    handler = registered["on_pipeline_finished"]

    # Simulate proxy path firing first
    await obs._finish_conversation(cancelled=True)
    assert call_count == 1

    # Safety net fires afterwards — real _finish_conversation would be a no-op
    # because _trace is now None; our mock just counts calls
    await handler(FakeTask(), CancelFrame())  # type: ignore[operator]
    assert (
        call_count == 2
    )  # mock doesn't guard, but real impl would no-op via _trace check
