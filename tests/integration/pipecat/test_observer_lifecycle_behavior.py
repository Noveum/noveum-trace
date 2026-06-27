"""
Value-asserting regression tests for the NoveumTraceObserver lifecycle
(subsystem A, OBS-1..11 in PIPECAT_TEST_PLAN.md).

These exercise ``NoveumTraceObserver`` against a *real* Noveum ``Trace`` and
*real* pipecat 1.3.0 frames — never MagicMock spans — so they pin actual span
names, attribute values, ``parent_span_id`` parenting, the custom
``pipecat_span_status`` string, SpanEvent names, and the teardown/reset block.

Covered: trace creation, ``on_push_frame`` dedup + terminal-frame ABP-source
gating, ``_create_child_span``, full-conversation WAV upload, ``attach_to_task``
wiring, idempotent finish + reset, and session-metadata flush.

Harness per the verified pattern: real ``Trace`` on ``obs._trace``; finish tests
run under one asyncio loop (``_finish_lock`` binds to the running loop on first
use). ``pipecat.full_conversation`` / ``pipecat.turn`` are trace-root **by
design** (intentional roots, Issues 3 & 4) — ``parent_span_id is None`` is the
EXPECTED contract here, not an xfail.
"""

from __future__ import annotations

import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pipecat.frames.frames")
pytest.importorskip("pipecat.observers.base_observer")

from pipecat.frames import frames as ff  # noqa: E402

from noveum_trace.core.trace import Trace  # noqa: E402
from noveum_trace.integrations.pipecat.pipecat_observer import (  # noqa: E402
    NoveumTraceObserver,
)


# --------------------------------------------------------------------------- #
# OBS-1 — real 1.3.0 StartFrame carries no pipeline.* attrs                     #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_handle_start_frame_no_pipeline_attrs_on_1x() -> None:
    # Guards: the belief that pipeline.* is populated on 1.x (real StartFrame
    # has no allow_interruptions/sample_rate/audio_sample_rate).
    # Version gate: on 0.0.x the real StartFrame DOES carry allow_interruptions,
    # so the integration captures pipeline.allow_interruptions there (the VC-2
    # old-only leg). This is the 1.x leg — skip cleanly on 0.0.x.
    if hasattr(ff.StartFrame(), "allow_interruptions"):
        pytest.skip(
            "0.0.x StartFrame carries allow_interruptions (captured as "
            "pipeline.allow_interruptions); 1.x-only assertion — see VC-2."
        )
    obs = NoveumTraceObserver(record_audio=False)
    trace = Trace(name="pipecat.conversation")
    obs._trace = trace

    sf = ff.StartFrame()  # NO attribute injection — real 1.3.0 frame
    data = types.SimpleNamespace(frame=sf, source=None)

    with patch.object(obs, "_ensure_audio_buffer_recording", new_callable=AsyncMock):
        await obs._handle_start_frame(data)

    assert obs._trace is trace
    assert not [k for k in trace.attributes if k.startswith("pipeline.")]


# --------------------------------------------------------------------------- #
# OBS-2 — EndFrame teardown deferred until the ABP is the source               #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_on_push_frame_defers_endframe_until_abp_source() -> None:
    # Guards: audio-flush ordering — finishing on the wrong hop runs the WAV
    # upload before on_audio_data populated chunks (empty WAV).
    obs = NoveumTraceObserver(record_audio=True)
    obs._trace = Trace(name="pipecat.conversation")

    class FakeABP:
        _recording = True

    abp = FakeABP()
    obs._audio_buffer_processor = abp

    calls: list[bool] = []

    async def spy(cancelled: bool = False) -> None:
        calls.append(cancelled)

    obs._finish_conversation = spy  # type: ignore[method-assign]

    # EndFrame from a non-ABP source → early return, no teardown.
    other_source = object()
    await obs.on_push_frame(
        types.SimpleNamespace(frame=ff.EndFrame(), source=other_source)
    )
    assert calls == []

    # EndFrame from the ABP source → teardown fires exactly once.
    await obs.on_push_frame(types.SimpleNamespace(frame=ff.EndFrame(), source=abp))
    assert calls == [False]


# --------------------------------------------------------------------------- #
# OBS-3 — dedup of a frame whose id is the falsy value 0                        #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_on_push_frame_dedups_falsy_frame_id_zero() -> None:
    # Guards: the `fid is not None` check against a refactor to truthy `if fid:`,
    # which would double-process the first frame (id==0 in pipecat 1.3.0).
    obs = NoveumTraceObserver(record_audio=False)

    count = {"n": 0}

    async def handler(_data: object) -> None:
        count["n"] += 1

    frame = ff.StartFrame()
    # Pin the id to the genuinely-falsy-but-valid 0 (frame ids come from a
    # process-global counter, so a natural StartFrame is past 0 by test time).
    frame.id = 0
    obs._frame_handlers = {type(frame): handler}

    data = types.SimpleNamespace(frame=frame, source=None)
    await obs.on_push_frame(data)
    await obs.on_push_frame(data)

    assert frame.id == 0
    assert count["n"] == 1


# --------------------------------------------------------------------------- #
# OBS-4 — _create_child_span(None) yields a trace-root span (orphan mechanism)  #
# --------------------------------------------------------------------------- #
def test_create_child_span_none_parent_is_root_real_parent_is_child() -> None:
    # Guards: the None->root mechanism underlying orphan Issues 1 & 2 (without
    # asserting the unimplemented fix).
    obs = NoveumTraceObserver(record_audio=False)
    trace = Trace(name="pipecat.conversation")
    obs._trace = trace

    rootish = obs._create_child_span(
        "pipecat.stt", parent_span=None, attributes={"k": "v"}
    )
    assert rootish in trace.spans
    assert rootish.parent_span_id is None
    assert rootish.attributes["k"] == "v"

    parent = trace.create_span(name="pipecat.turn")
    child = obs._create_child_span("pipecat.stt", parent_span=parent, attributes={})
    assert child.parent_span_id == parent.span_id


# --------------------------------------------------------------------------- #
# OBS-5 — _finish_conversation writes summary, status ok, resets for reuse      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_finish_conversation_writes_summary_and_resets_for_reuse() -> None:
    # Guards: the large untested teardown+reset block (summary attrs, ok status,
    # full_conversation root span, full state reset, observer reuse).
    obs = NoveumTraceObserver(record_audio=True, capture_text=True)
    trace = Trace(name="pipecat.conversation")
    obs._trace = trace
    obs._metrics_accumulator = {
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "total_cost": 0.003,
        "turn_count": 2,
    }
    obs._transcription_buffer = ["hello", "world"]
    obs._current_turn_number = 5  # nonzero so the reset assertion is non-vacuous
    obs._processed_frame_ids = {1, 2, 3}
    obs._session_metadata = {"session.room_url": "https://room"}

    client = MagicMock()
    with patch.object(obs, "_get_client", return_value=client):
        await obs._finish_conversation()

    # Summary written onto the real trace, with ok status.
    assert trace.attributes["conversation.total_input_tokens"] == 100
    assert trace.attributes["conversation.total_output_tokens"] == 50
    assert trace.attributes["conversation.total_cost"] == 0.003
    assert trace.attributes["conversation.turn_count"] == 2
    assert trace.attributes["conversation.last_user_input"] == "hello world"
    assert trace.attributes["pipecat_span_status"] == "ok"

    # full_conversation span exists and is an intentional trace root (Issue 3).
    fc = [s for s in trace.spans if s.name == "pipecat.full_conversation"]
    assert len(fc) == 1
    assert fc[0].parent_span_id is None

    assert client.finish_trace.call_count == 1

    # State fully reset for observer reuse.
    assert obs._trace is None
    assert obs._metrics_accumulator == {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "turn_count": 0,
    }
    assert obs._current_turn_number == 0
    assert obs._processed_frame_ids == set()
    assert obs._audio_buffer_processor is None
    assert obs._session_metadata == {}

    # Second conversation on the SAME observer proves reuse.
    trace2 = Trace(name="pipecat.conversation")
    obs._trace = trace2
    with patch.object(obs, "_get_client", return_value=client):
        await obs._finish_conversation()
    assert client.finish_trace.call_count == 2


# --------------------------------------------------------------------------- #
# OBS-6 — cancelled finish marks spans/trace cancelled but preserves error      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_finish_conversation_cancelled_preserves_prior_error() -> None:
    # Guards: never-overwrite-error + cancelled-vs-ok status logic on teardown.
    obs = NoveumTraceObserver(record_audio=False, capture_text=True)
    trace = Trace(name="pipecat.conversation")
    obs._trace = trace

    llm = trace.create_span(
        name="pipecat.llm", attributes={"pipecat_span_status": "error"}
    )
    tts = trace.create_span(name="pipecat.tts")
    obs._active_llm_span = llm
    obs._active_tts_span = tts

    client = MagicMock()
    with patch.object(obs, "_get_client", return_value=client):
        await obs._finish_conversation(cancelled=True)

    assert tts.attributes["pipecat_span_status"] == "cancelled"
    assert tts.is_finished()
    assert llm.attributes["pipecat_span_status"] == "error"  # preserved
    assert llm.is_finished()
    assert trace.attributes["pipecat_span_status"] == "cancelled"


# --------------------------------------------------------------------------- #
# OBS-7 — full_conversation happy path pins WAV-derived values + root parenting #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_upload_full_conversation_audio_pins_wav_values_and_root() -> None:
    # Guards: WAV-derived attribute values + intentional trace-root parenting +
    # export_audio wiring (audio_uuid matches the span attribute).
    obs = NoveumTraceObserver(record_audio=True)
    trace = Trace(name="pipecat.conversation")
    obs._trace = trace

    pcm = b"\x00\x01" * 8000
    sr, ch = 16000, 2
    obs._conversation_audio_chunks = [pcm]
    obs._conversation_audio_sample_rate = sr
    obs._conversation_audio_num_channels = ch

    client = MagicMock()
    with patch.object(obs, "_get_client", return_value=client):
        await obs._upload_full_conversation_audio()

    fc = next(s for s in trace.spans if s.name == "pipecat.full_conversation")
    assert fc.parent_span_id is None  # intentional root (Issue 3)
    assert fc.attributes["full_conversation.audio_channels"] == "stereo"
    assert fc.attributes["full_conversation.audio_channel_left"] == "user"
    assert fc.attributes["full_conversation.audio_channel_right"] == "bot"
    assert fc.attributes["full_conversation.sample_rate"] == sr
    assert fc.attributes["full_conversation.duration_ms"] == int(
        len(pcm) / (sr * ch * 2) * 1000
    )
    assert fc.attributes["full_conversation.audio_format"] == "wav"
    assert fc.attributes["pipecat_span_status"] == "ok"

    audio_uuid = fc.attributes["full_conversation.audio_uuid"]
    # uuid attr is a valid UUID string echoed to export_audio.
    assert str(uuid.UUID(audio_uuid)) == audio_uuid
    assert client.export_audio.call_args.kwargs["audio_uuid"] == audio_uuid


# --------------------------------------------------------------------------- #
# OBS-8 — attach_to_task wires real TurnTracking/Latency observer event names   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_attach_to_task_wires_real_observer_event_names() -> None:
    # Guards: the exact pipecat event names the observer subscribes to
    # (on_turn_started/on_turn_ended/on_latency_measured) + external flag.
    pytest.importorskip("pipecat.observers.turn_tracking_observer")
    from pipecat.observers.turn_tracking_observer import TurnTrackingObserver

    obs = NoveumTraceObserver(record_audio=False)
    tto = TurnTrackingObserver()

    latency_names: list[str] = []

    class FakeLatencyObserver:
        def add_event_handler(self, name: str, _handler: object) -> None:
            latency_names.append(name)

    class FakeTask:
        turn_tracking_observer = tto
        _user_bot_latency_observer = FakeLatencyObserver()

        def event_handler(self, _name: str):
            def deco(fn):
                return fn

            return deco

    with patch.object(
        obs, "_attach_audio_buffer_from_pipeline", new_callable=AsyncMock
    ):
        await obs.attach_to_task(FakeTask())

    # Real TurnTrackingObserver records subscriptions in _event_handlers.
    assert "on_turn_started" in tto._event_handlers
    assert "on_turn_ended" in tto._event_handlers
    assert latency_names == ["on_latency_measured"]
    assert obs._using_external_turn_tracking is True


# --------------------------------------------------------------------------- #
# OBS-9 — attach_to_task is idempotent for the on_pipeline_finished safety net  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_attach_to_task_idempotent_pipeline_finished_registration() -> None:
    # Guards: the _registered_pipeline_tasks dedup; a regression would
    # double-register the safety net and double-fire teardown.
    obs = NoveumTraceObserver(record_audio=False)

    registrations = {"n": 0}

    class FakeTask:
        turn_tracking_observer = None
        _user_bot_latency_observer = None

        def event_handler(self, name: str):
            def deco(fn):
                if name == "on_pipeline_finished":
                    registrations["n"] += 1
                return fn

            return deco

    task = FakeTask()
    with patch.object(
        obs, "_attach_audio_buffer_from_pipeline", new_callable=AsyncMock
    ):
        await obs.attach_to_task(task)
        await obs.attach_to_task(task)

    assert registrations["n"] == 1
    assert task in obs._registered_pipeline_tasks


# --------------------------------------------------------------------------- #
# OBS-10 — ClientConnectedFrame e2e flushes buffered session metadata          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_client_connected_frame_flushes_session_metadata_e2e() -> None:
    # Guards: the production frame-handler path (on_push_frame -> dispatch ->
    # _handle_client_connected -> _flush_session_metadata), not a direct call.
    # Trace now exposes `.events`, so we also assert the `client.connected`
    # SpanEvent that `_handle_client_connected` appends to the trace.
    obs = NoveumTraceObserver(record_audio=False)
    trace = Trace(name="pipecat.conversation")
    obs._trace = trace

    class FakeTransport:
        room_url = "https://room.example"

    class RunnerArgs:
        pipeline_idle_timeout_secs = 30

    obs._store_transport(FakeTransport(), RunnerArgs())
    assert obs._session_metadata  # buffered before connection

    data = types.SimpleNamespace(frame=ff.ClientConnectedFrame(), source=None)
    await obs.on_push_frame(data)

    assert trace.attributes["session.room_url"] == "https://room.example"
    assert trace.attributes["session.transport_type"] == "FakeTransport"
    assert trace.attributes["session.idle_timeout_secs"] == 30
    assert obs._session_metadata == {}  # cleared after flush
    assert any(e.name == "client.connected" for e in trace.events)


# --------------------------------------------------------------------------- #
# OBS-11 — _on_conversation_audio ignores audio from a stale processor          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_on_conversation_audio_drops_stale_processor() -> None:
    # Guards: cross-session audio contamination after attach_to_task swaps the
    # ABP between tasks (only the active processor's audio is kept).
    obs = NoveumTraceObserver(record_audio=True)
    abp_a = object()
    abp_b = object()
    obs._audio_buffer_processor = abp_a

    await obs._on_conversation_audio(abp_b, b"x", 16000, 2)  # stale → dropped
    await obs._on_conversation_audio(abp_a, b"y", 16000, 2)  # active → kept

    assert obs._conversation_audio_chunks == [b"y"]
