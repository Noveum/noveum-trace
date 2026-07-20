"""
Tests for the per-call-client / deferred-export observer features used by host
integrations (e.g. dograh):

- ``client=`` injection: ``_get_client()`` resolves injected → global → None;
  injected clients start traces with ``set_as_current=False``.
- ``DeferredTransport`` + ``transport=`` client injection: the trace is built
  through the normal client flow but captured in memory instead of sent; no
  batch thread, no atexit hook.
- ``deferred=True``: the constructor guard (deferred + record_audio requires an
  audio_sink) and ``build_payload_snapshot()`` reading the captured trace.
- ``audio_sink=``: recorded segments handed to the sink (WAV bytes) instead of
  uploaded, span uuid attributes stamped identically.
- ``register_finish_safety_net=False``: attach_to_task skips only the
  on_pipeline_finished handler.
- Direct sends: ``NoveumClient.send_trace_dict`` / ``send_audio_sync`` are
  synchronous, delivery-observed, and raise ``NoveumTraceError``.
- ``Span`` exception info survives the to_dict/from_dict round trip.

Same conventions as the sibling behavior tests: real ``NoveumTraceObserver``,
real ``Trace`` objects, client faked at the ``_get_client`` seam where a full
client is not needed.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pipecat.frames.frames")

# disable_transport_mocking: the suite-wide autouse mock_client_creation fixture
# (tests/conftest.py) replaces NoveumClient.__init__ and force-assigns a Mock
# transport, which would swallow the transport_instance= injection these tests
# exist to exercise.
pytestmark = [pytest.mark.asyncio, pytest.mark.disable_transport_mocking]

_UPLOAD = "noveum_trace.integrations.pipecat._handlers_tts.upload_audio_frames"


def _make_obs(**kwargs):
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    return NoveumTraceObserver(**kwargs)


def _deferred_client():
    """Real NoveumClient wired to a capture-only transport: no HttpTransport,
    no batch thread, no atexit registration, no global-config mutation."""
    from noveum_trace.core.client import NoveumClient
    from noveum_trace.core.config import Config
    from noveum_trace.transport.deferred_transport import DeferredTransport

    config = Config.create(project="deferred-test", api_key="deferred")
    return NoveumClient(config=config, transport_instance=DeferredTransport())


def _bare_client():
    """NoveumClient with a fully mocked transport (for direct-send tests)."""
    from noveum_trace.core.client import NoveumClient

    client = NoveumClient.__new__(NoveumClient)
    client._shutdown = False
    client.transport = MagicMock()
    return client


def _tts_source():
    return types.SimpleNamespace(
        _settings=types.SimpleNamespace(voice="nova", model="tts-1", language=None)
    )


def _audio_frame(ff):
    return ff.TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)


async def _drive_tts_utterance(obs, ff):
    """TTSStarted → one audio frame → TTSStopped; returns the closed span."""
    src = _tts_source()
    await obs._handle_tts_started(
        types.SimpleNamespace(frame=ff.TTSStartedFrame(), source=src)
    )
    span = obs._active_tts_span
    await obs._handle_tts_audio(
        types.SimpleNamespace(frame=_audio_frame(ff), source=src)
    )
    await obs._handle_tts_stopped(types.SimpleNamespace())
    return span


# --------------------------------------------------------------------------- #
# Client injection resolution
# --------------------------------------------------------------------------- #
async def test_injected_client_wins_over_global():
    client = MagicMock()
    obs = _make_obs(client=client)
    with patch("noveum_trace.get_client", return_value=MagicMock()):
        assert obs._get_client() is client


async def test_without_injection_falls_back_to_global():
    global_client = MagicMock()
    obs = _make_obs()
    with patch("noveum_trace.get_client", return_value=global_client):
        assert obs._get_client() is global_client


async def test_uninitialized_global_resolves_to_none():
    from noveum_trace.utils.exceptions import InitializationError

    obs = _make_obs()
    with patch("noveum_trace.get_client", side_effect=InitializationError("no init")):
        assert obs._get_client() is None


async def test_injected_client_starts_trace_without_touching_context():
    client = MagicMock()
    obs = _make_obs(client=client)

    await obs.on_pipeline_started()

    client.start_trace.assert_called_once()
    assert client.start_trace.call_args.kwargs["set_as_current"] is False


async def test_global_client_keeps_set_as_current():
    global_client = MagicMock()
    obs = _make_obs()

    with patch("noveum_trace.get_client", return_value=global_client):
        await obs.on_pipeline_started()

    assert global_client.start_trace.call_args.kwargs["set_as_current"] is True


# --------------------------------------------------------------------------- #
# DeferredTransport + transport-injected client
# --------------------------------------------------------------------------- #
async def test_transport_injected_client_skips_atexit_and_http_transport():
    from noveum_trace.transport.deferred_transport import DeferredTransport

    with (
        patch("noveum_trace.core.client.atexit.register") as register_mock,
        patch("noveum_trace.core.client.HttpTransport") as http_mock,
    ):
        client = _deferred_client()

    register_mock.assert_not_called()
    http_mock.assert_not_called()
    assert isinstance(client.transport, DeferredTransport)


async def test_deferred_flow_captures_finished_trace(ff):
    client = _deferred_client()
    obs = _make_obs(client=client, deferred=True, record_audio=False)

    await obs.on_pipeline_started()
    trace = obs._trace
    await _drive_tts_utterance(obs, ff)
    await obs._finish_conversation()

    assert obs._trace is None
    assert client.transport.captured_trace is trace
    assert trace._finished
    # Client-path parity: normal start_trace stamped the noveum.* attributes.
    assert trace.attributes.get("noveum.project") == "deferred-test"


async def test_snapshot_reads_captured_trace(ff):
    client = _deferred_client()
    obs = _make_obs(client=client, deferred=True, record_audio=False)
    await obs.on_pipeline_started()
    trace = obs._trace
    await _drive_tts_utterance(obs, ff)
    await obs._finish_conversation()

    snapshot = obs.build_payload_snapshot()

    assert snapshot is not None
    assert snapshot["trace_id"] == trace.trace_id
    assert snapshot["end_time"] is not None
    assert any(s["name"] == "pipecat.tts" for s in snapshot["spans"])


async def test_snapshot_falls_back_to_in_progress_trace():
    client = _deferred_client()
    obs = _make_obs(client=client, deferred=True, record_audio=False)
    await obs.on_pipeline_started()

    snapshot = obs.build_payload_snapshot()

    assert snapshot is not None
    assert snapshot["trace_id"] == obs._trace.trace_id
    assert snapshot["end_time"] is None


async def test_snapshot_none_when_no_trace():
    obs = _make_obs(client=_deferred_client(), deferred=True, record_audio=False)
    assert obs.build_payload_snapshot() is None


async def test_snapshot_roundtrips_through_from_dict(ff):
    from noveum_trace.core.trace import Trace

    client = _deferred_client()
    obs = _make_obs(client=client, deferred=True, record_audio=False)
    await obs.on_pipeline_started()
    await _drive_tts_utterance(obs, ff)
    await obs._finish_conversation()

    snapshot = obs.build_payload_snapshot()
    restored = Trace.from_dict(snapshot)
    assert restored.trace_id == snapshot["trace_id"]
    assert [s.name for s in restored.spans] == [s["name"] for s in snapshot["spans"]]


async def test_deferred_transport_drops_audio_without_crashing():
    from noveum_trace.transport.deferred_transport import DeferredTransport

    transport = DeferredTransport()
    transport.export_audio(
        audio_data=b"x", trace_id="t", span_id="s", audio_uuid="u", metadata={}
    )
    transport.flush()
    transport.shutdown()
    assert transport.captured_trace is None


# --------------------------------------------------------------------------- #
# Constructor guard: deferred + record_audio requires a sink
# --------------------------------------------------------------------------- #
async def test_deferred_record_audio_without_sink_raises():
    with pytest.raises(ValueError, match="audio_sink"):
        _make_obs(client=_deferred_client(), deferred=True, record_audio=True)


async def test_deferred_guard_satisfied_by_sink_or_no_audio():
    _make_obs(
        client=_deferred_client(),
        deferred=True,
        record_audio=True,
        audio_sink=AsyncMock(return_value=True),
    )
    _make_obs(client=_deferred_client(), deferred=True, record_audio=False)


# --------------------------------------------------------------------------- #
# Audio sink — segment routing, uuid stamping, failure semantics
# --------------------------------------------------------------------------- #
async def test_tts_audio_routed_to_sink_not_upload(ff, real_trace_with_turn):
    trace, turn = real_trace_with_turn
    sink = AsyncMock(return_value=True)
    obs = _make_obs(record_audio=True, audio_sink=sink)
    obs._trace = trace
    obs._current_turn_span = turn

    with patch(_UPLOAD) as upload_mock:
        span = await _drive_tts_utterance(obs, ff)

    upload_mock.assert_not_called()
    sink.assert_awaited_once()
    kwargs = sink.await_args.kwargs
    assert kwargs["kind"] == "tts"
    assert isinstance(kwargs["wav_bytes"], bytes) and kwargs["wav_bytes"]
    assert kwargs["trace_id"] == span.trace_id
    assert kwargs["span_id"] == span.span_id
    assert kwargs["metadata"]["format"] == "wav"
    assert kwargs["metadata"]["duration_ms"] > 0
    assert span.attributes["tts.audio_uuid"] == kwargs["audio_uuid"]
    assert span.attributes["pipecat_span_status"] == "ok"
    assert obs._tts_audio_buffer == []  # cleared on success


async def test_tts_sink_failure_marks_upload_failed_and_retains_buffer(
    ff, real_trace_with_turn
):
    trace, turn = real_trace_with_turn
    sink = AsyncMock(return_value=False)
    obs = _make_obs(record_audio=True, audio_sink=sink)
    obs._trace = trace
    obs._current_turn_span = turn

    span = await _drive_tts_utterance(obs, ff)

    assert span.attributes["pipecat_span_status"] == "upload_failed"
    assert "tts.audio_uuid" not in span.attributes
    assert obs._tts_audio_buffer  # retained for retry/inspection (clear-on-success)


async def test_sink_exception_is_contained(ff, real_trace_with_turn):
    trace, turn = real_trace_with_turn
    sink = AsyncMock(side_effect=RuntimeError("sink boom"))
    obs = _make_obs(record_audio=True, audio_sink=sink)
    obs._trace = trace
    obs._current_turn_span = turn

    span = await _drive_tts_utterance(obs, ff)

    assert span.attributes["pipecat_span_status"] == "upload_failed"
    assert "tts.audio_uuid" not in span.attributes


async def test_sink_receives_copy_not_live_buffer(ff, real_trace_with_turn):
    """The frames handed to the encode/sink must be a snapshot: clearing the
    instance buffer (as conversation teardown does) after the handler grabbed
    it must not affect what the sink received."""
    trace, turn = real_trace_with_turn
    received: dict = {}

    async def sink(**kwargs):
        received.update(kwargs)
        return True

    obs = _make_obs(record_audio=True, audio_sink=sink)
    obs._trace = trace
    obs._current_turn_span = turn

    await _drive_tts_utterance(obs, ff)

    # One 320-byte mono 16kHz frame → WAV header (44 bytes) + 320 PCM bytes.
    assert len(received["wav_bytes"]) == 44 + 320


async def test_no_sink_keeps_default_upload_path(ff, real_trace_with_turn):
    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=True)
    obs._trace = trace
    obs._current_turn_span = turn

    with patch(_UPLOAD, return_value=True) as upload_mock:
        span = await _drive_tts_utterance(obs, ff)

    upload_mock.assert_called_once()
    assert "tts.audio_uuid" in span.attributes


async def test_full_conversation_audio_routed_to_sink():
    sink = AsyncMock(return_value=True)
    obs = _make_obs(
        client=_deferred_client(), deferred=True, record_audio=True, audio_sink=sink
    )
    await obs.on_pipeline_started()
    obs._audio_buffer_processor = object()  # pretend ABP was attached
    obs._conversation_audio_chunks = [b"\x00" * 3200]
    obs._conversation_audio_sample_rate = 16000
    obs._conversation_audio_num_channels = 2

    await obs._upload_full_conversation_audio()

    sink.assert_awaited_once()
    kwargs = sink.await_args.kwargs
    assert kwargs["kind"] == "conversation"
    assert isinstance(kwargs["wav_bytes"], bytes) and kwargs["wav_bytes"]
    span = next(s for s in obs._trace.spans if s.name == "pipecat.full_conversation")
    assert span.attributes["full_conversation.audio_uuid"] == kwargs["audio_uuid"]
    assert span.attributes["pipecat_span_status"] == "ok"


# --------------------------------------------------------------------------- #
# register_finish_safety_net flag
# --------------------------------------------------------------------------- #
class _FakeTask:
    """Minimal PipelineTask stand-in for attach_to_task."""

    def __init__(self) -> None:
        self.turn_tracking_observer = None
        self.registered_events: list = []

    def event_handler(self, name):
        self.registered_events.append(name)

        def _decorator(fn):
            return fn

        return _decorator


async def test_safety_net_registered_by_default():
    obs = _make_obs(record_audio=False)
    task = _FakeTask()

    await obs.attach_to_task(task)

    assert "on_pipeline_finished" in task.registered_events
    assert task in obs._registered_pipeline_tasks


async def test_safety_net_skipped_when_host_opts_out():
    obs = _make_obs(record_audio=False, register_finish_safety_net=False)
    task = _FakeTask()

    await obs.attach_to_task(task)

    assert task.registered_events == []
    assert task not in obs._registered_pipeline_tasks


# --------------------------------------------------------------------------- #
# Direct sends: send_trace_dict / send_audio_sync
# --------------------------------------------------------------------------- #
async def test_send_trace_dict_sends_synchronously():
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    trace.finish()
    client = _bare_client()

    client.send_trace_dict(trace.to_dict())

    client.transport.send_trace_now.assert_called_once()
    sent = client.transport.send_trace_now.call_args.args[0]
    assert sent.trace_id == trace.trace_id


async def test_send_trace_dict_wraps_transport_errors():
    from noveum_trace.core.trace import Trace
    from noveum_trace.utils.exceptions import NoveumTraceError, TransportError

    trace = Trace(name="t")
    trace.finish()
    client = _bare_client()
    client.transport.send_trace_now.side_effect = TransportError("401")

    with pytest.raises(NoveumTraceError):
        client.send_trace_dict(trace.to_dict())


async def test_send_trace_dict_raises_after_shutdown():
    from noveum_trace.core.trace import Trace
    from noveum_trace.utils.exceptions import NoveumTraceError

    trace = Trace(name="t")
    trace.finish()
    client = _bare_client()
    client._shutdown = True

    with pytest.raises(NoveumTraceError):
        client.send_trace_dict(trace.to_dict())


async def test_send_trace_dict_rejects_invalid_snapshot():
    from noveum_trace.utils.exceptions import NoveumTraceError

    client = _bare_client()
    with pytest.raises(NoveumTraceError):
        client.send_trace_dict({"bogus": True})
    client.transport.send_trace_now.assert_not_called()


async def test_send_audio_sync_delivers_and_raises_on_failure():
    from noveum_trace.utils.exceptions import NoveumTraceError, TransportError

    client = _bare_client()
    client.send_audio_sync(
        audio_data=b"RIFF",
        trace_id="t1",
        span_id="s1",
        audio_uuid="u1",
        metadata={"format": "wav"},
    )
    kwargs = client.transport.send_audio_now.call_args.kwargs
    assert kwargs["audio_uuid"] == "u1"
    assert kwargs["audio_data"] == b"RIFF"

    client.transport.send_audio_now.side_effect = TransportError("500")
    with pytest.raises(NoveumTraceError):
        client.send_audio_sync(
            audio_data=b"RIFF", trace_id="t1", span_id="s1", audio_uuid="u2"
        )


async def test_send_audio_sync_raises_after_shutdown():
    from noveum_trace.utils.exceptions import NoveumTraceError

    client = _bare_client()
    client._shutdown = True
    with pytest.raises(NoveumTraceError):
        client.send_audio_sync(
            audio_data=b"x", trace_id="t", span_id="s", audio_uuid="u"
        )


# --------------------------------------------------------------------------- #
# Span exception info survives the snapshot round trip
# --------------------------------------------------------------------------- #
async def test_span_exception_block_roundtrips():
    from noveum_trace.core.span import Span
    from noveum_trace.core.trace import Trace

    trace = Trace(name="t")
    span = trace.create_span(name="op")
    span.record_exception(ValueError("boom"))
    span.finish()
    trace.finish()

    once = trace.to_dict()
    restored = Trace.from_dict(once)
    twice = restored.to_dict()

    original = next(s for s in once["spans"] if s["name"] == "op")["exception"]
    roundtripped = next(s for s in twice["spans"] if s["name"] == "op")["exception"]
    assert original == roundtripped
    assert roundtripped["type"] == "ValueError"
    assert roundtripped["message"] == "boom"
    # Sanity: Span.from_dict alone preserves the block too.
    span_dict = next(s for s in once["spans"] if s["name"] == "op")
    assert Span.from_dict(span_dict).to_dict()["exception"] == original


# --------------------------------------------------------------------------- #
# Observer reuse: no stale capture served for the next conversation
# --------------------------------------------------------------------------- #
async def test_reused_observer_does_not_serve_previous_calls_capture(ff):
    client = _deferred_client()
    obs = _make_obs(
        client=client,
        deferred=True,
        record_audio=False,
        register_finish_safety_net=False,
    )

    # Conversation 1: full lifecycle → captured.
    await obs.on_pipeline_started()
    call1_trace_id = obs._trace.trace_id
    await _drive_tts_utterance(obs, ff)
    await obs._finish_conversation()
    assert obs.build_payload_snapshot()["trace_id"] == call1_trace_id

    # Conversation 2 dies BEFORE StartFrame: attach happened, nothing else.
    await obs.attach_to_task(_FakeTask())
    assert obs.build_payload_snapshot() is None  # never call 1's trace

    # Conversation 2 retried and started: snapshot is the NEW trace.
    await obs.on_pipeline_started()
    snapshot = obs.build_payload_snapshot()
    assert snapshot is not None
    assert snapshot["trace_id"] != call1_trace_id
