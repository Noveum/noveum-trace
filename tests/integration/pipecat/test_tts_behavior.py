"""
Value-asserting regression tests for the Pipecat TTS subsystem
(``_handlers_tts._TTSHandlersMixin``), per ``PIPECAT_TEST_PLAN.md`` §C (TTS-1..9).

These drive the real ``NoveumTraceObserver`` TTS handlers against a real
``Trace`` (no MagicMock spans) and assert the emitted contract: the
``pipecat.tts`` span name, turn parenting (``parent_span_id``),
``tts.voice``/``tts.model``/``tts.input_text``/``tts.audio_uuid`` values, the
custom ``pipecat_span_status`` string, audio-buffer source pinning, the
``_last_tts_span`` late-metrics backref, and dispatch-table routing through
``on_push_frame``.
"""

from __future__ import annotations

import types
import uuid
from unittest.mock import patch

import pytest

pytest.importorskip("pipecat.frames.frames")
pytest.importorskip("pipecat.metrics.metrics")

# Make the async intent explicit (the repo's asyncio_mode="auto" already
# auto-discovers these, but this keeps them portable across pytest configs).
pytestmark = pytest.mark.asyncio

_UPLOAD = "noveum_trace.integrations.pipecat._handlers_tts.upload_audio_frames"


def _make_obs(*, capture_text: bool = True, record_audio: bool = True):
    """Fresh real observer (no mocked spans) with the given opt-in flags."""
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    return NoveumTraceObserver(capture_text=capture_text, record_audio=record_audio)


def _source(*, voice=None, model=None, has_settings=True):
    """Fake TTS source processor exposing ``._settings.voice`` / ``.model``."""
    if not has_settings:
        return types.SimpleNamespace(_settings=None)
    return types.SimpleNamespace(
        _settings=types.SimpleNamespace(voice=voice, model=model)
    )


def _audio_frame(ff):
    return ff.TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)


# --------------------------------------------------------------------------- #
# TTS-1 — started span is a real child of the turn with voice/model from source
# --------------------------------------------------------------------------- #
async def test_tts_started_span_is_turn_child_with_voice_model(
    ff, real_trace_with_turn
):
    # Guards: turn parenting + voice/model read from source._settings (WEAK-REPLACE).
    trace, turn = real_trace_with_turn
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn
    src = _source(voice="nova", model="tts-1")

    await obs._handle_tts_started(
        types.SimpleNamespace(frame=ff.TTSStartedFrame(), source=src)
    )

    span = obs._active_tts_span
    assert span in trace.spans
    assert span.name == "pipecat.tts"
    assert span.parent_span_id == turn.span_id
    assert span.attributes["tts.voice"] == "nova"
    assert span.attributes["tts.model"] == "tts-1"
    assert obs._tts_source_processor is src
    assert obs._last_tts_span is None  # cleared on every new TTS start


# --------------------------------------------------------------------------- #
# TTS-2 — no current turn: span is created and named, but orphaned (Issue 2)
# --------------------------------------------------------------------------- #
async def test_tts_started_no_turn_still_creates_named_span(ff, real_trace_with_turn):
    # Guards: a teardown/post-turn TTS still produces a named pipecat.tts span
    # (the orphan-parenting itself is the XFAIL below).
    trace, _turn = real_trace_with_turn
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = None  # post-turn / end_call teardown TTS

    await obs._handle_tts_started(
        types.SimpleNamespace(
            frame=ff.TTSStartedFrame(), source=_source(has_settings=False)
        )
    )

    span = obs._active_tts_span
    assert span in trace.spans
    assert span.name == "pipecat.tts"
    # No teardown/context tagging exists today (no fallback handling).
    assert "tts.context" not in span.attributes
    assert "session_teardown" not in span.attributes


@pytest.mark.xfail(
    strict=True,
    reason="Issue 2 orphan: TTS span created with no open turn parents to trace "
    "root (parent_span_id is None); flips to xpass when the _last_turn_span "
    "fallback lands — see PIPECAT_SPAN_HIERARCHY_ISSUES.md",
)
async def test_tts_started_no_turn_should_be_parented(ff, real_trace_with_turn):
    # Guards: the documented orphan — TTS started with no current turn SHOULD
    # still be parented under the last turn; currently it is a trace-root orphan.
    trace, _turn = real_trace_with_turn
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = None

    await obs._handle_tts_started(
        types.SimpleNamespace(
            frame=ff.TTSStartedFrame(), source=_source(has_settings=False)
        )
    )

    span = obs._active_tts_span
    assert span.parent_span_id is not None  # SHOULD be under a turn; currently None


# --------------------------------------------------------------------------- #
# TTS-3 — multiple TTSTextFrame chunks join (empty separator) into tts.input_text
# --------------------------------------------------------------------------- #
async def test_tts_text_chunks_join_into_input_text(ff, real_trace_with_turn):
    # Guards: ''.join order of TTSTextFrame chunks + finish + buffer reset + backref.
    trace, turn = real_trace_with_turn
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn

    await obs._handle_tts_started(
        types.SimpleNamespace(
            frame=ff.TTSStartedFrame(), source=_source(has_settings=False)
        )
    )
    span = obs._active_tts_span
    for chunk in ("Hello, ", "world", "!"):
        await obs._handle_tts_text(
            types.SimpleNamespace(
                frame=ff.TTSTextFrame(text=chunk, aggregated_by="word")
            )
        )

    await obs._handle_tts_stopped(types.SimpleNamespace())

    assert span.attributes["tts.input_text"] == "Hello, world!"
    assert span.is_finished()
    assert obs._tts_text_buffer == []
    assert obs._last_tts_span is span
    assert span.attributes["pipecat_span_status"] == "ok"


# --------------------------------------------------------------------------- #
# TTS-4 — capture_text=False: TTSTextFrame ignored, no tts.input_text
# --------------------------------------------------------------------------- #
async def test_tts_text_ignored_when_capture_text_false(ff, real_trace_with_turn):
    # Guards: the `if not self._capture_text: return` privacy gate.
    trace, turn = real_trace_with_turn
    obs = _make_obs(capture_text=False, record_audio=False)
    obs._trace = trace
    obs._current_turn_span = turn

    await obs._handle_tts_started(
        types.SimpleNamespace(
            frame=ff.TTSStartedFrame(), source=_source(has_settings=False)
        )
    )
    span = obs._active_tts_span
    await obs._handle_tts_text(
        types.SimpleNamespace(
            frame=ff.TTSTextFrame(text="secret", aggregated_by="word")
        )
    )
    assert obs._tts_text_buffer == []

    await obs._handle_tts_stopped(types.SimpleNamespace())

    assert "tts.input_text" not in span.attributes
    assert span.attributes["pipecat_span_status"] == "ok"


# --------------------------------------------------------------------------- #
# TTS-5 — audio upload success: tts.audio_uuid set, buffer cleared, args correct
# --------------------------------------------------------------------------- #
async def test_tts_audio_upload_success(ff, real_trace_with_turn):
    # Guards: the only per-span audio attr (tts.audio_uuid) + buffer-clear-on-success
    # + exact upload wiring; no duration/sample_rate/format leak onto the span.
    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=True)
    obs._trace = trace
    obs._current_turn_span = turn
    src = _source(has_settings=False)

    await obs._handle_tts_started(
        types.SimpleNamespace(frame=ff.TTSStartedFrame(), source=src)
    )
    span = obs._active_tts_span
    for _ in range(2):
        await obs._handle_tts_audio(
            types.SimpleNamespace(frame=_audio_frame(ff), source=src)
        )
    assert len(obs._tts_audio_buffer) == 2

    with patch(_UPLOAD, return_value=True) as up:
        await obs._handle_tts_stopped(types.SimpleNamespace())

    audio_uuid = span.attributes["tts.audio_uuid"]
    assert str(uuid.UUID(audio_uuid)) == audio_uuid  # valid uuid str
    assert span.attributes["pipecat_span_status"] == "ok"
    assert obs._tts_audio_buffer == []  # cleared on success

    assert up.call_count == 1
    args, kwargs = up.call_args
    # upload_audio_frames(buffer, audio_uuid, "tts", trace_id, span_id, client=...)
    assert args[1] == audio_uuid
    assert args[2] == "tts"
    assert args[3] == span.trace_id
    assert args[4] == span.span_id

    # These belong to upload metadata, not the span itself.
    for absent in ("tts.duration_ms", "tts.sample_rate", "tts.format"):
        assert absent not in span.attributes


# --------------------------------------------------------------------------- #
# TTS-6 — upload False / raises: upload_failed, no uuid, buffer retained, finishes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "upload_kwargs",
    [
        {"return_value": False},
        {"side_effect": RuntimeError("upload boom")},
    ],
    ids=["returns_false", "raises"],
)
async def test_tts_audio_upload_failure(ff, real_trace_with_turn, upload_kwargs):
    # Guards: upload_failed status + buffer-retention + finally-finish + broad-except.
    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=True)
    obs._trace = trace
    obs._current_turn_span = turn
    src = _source(has_settings=False)

    await obs._handle_tts_started(
        types.SimpleNamespace(frame=ff.TTSStartedFrame(), source=src)
    )
    span = obs._active_tts_span
    audio = _audio_frame(ff)
    await obs._handle_tts_audio(types.SimpleNamespace(frame=audio, source=src))

    with patch(_UPLOAD, **upload_kwargs):
        await obs._handle_tts_stopped(types.SimpleNamespace())

    assert span.attributes["pipecat_span_status"] == "upload_failed"
    assert "tts.audio_uuid" not in span.attributes
    assert obs._tts_audio_buffer == [audio]  # retained for retry/inspection
    assert span.is_finished()  # finish() runs in finally


# --------------------------------------------------------------------------- #
# TTS-7 — audio buffering: matching source / pinned-None bypass / record_audio off
# --------------------------------------------------------------------------- #
async def test_tts_audio_buffering_matching_source(ff):
    # Guards: a matching pinned source buffers the frame.
    obs = _make_obs(record_audio=True)
    src = types.SimpleNamespace()
    obs._tts_source_processor = src
    await obs._handle_tts_audio(
        types.SimpleNamespace(frame=_audio_frame(ff), source=src)
    )
    assert len(obs._tts_audio_buffer) == 1


async def test_tts_audio_buffering_pinned_none_bypasses_filter(ff):
    # Guards: pinned source None disables the source filter — buffers regardless.
    obs = _make_obs(record_audio=True)
    obs._tts_source_processor = None
    await obs._handle_tts_audio(
        types.SimpleNamespace(frame=_audio_frame(ff), source=types.SimpleNamespace())
    )
    assert len(obs._tts_audio_buffer) == 1


async def test_tts_audio_dropped_when_record_audio_false(ff):
    # Guards: record_audio=False drops audio even from the correct source.
    obs = _make_obs(record_audio=False)
    src = types.SimpleNamespace()
    obs._tts_source_processor = src
    await obs._handle_tts_audio(
        types.SimpleNamespace(frame=_audio_frame(ff), source=src)
    )
    assert obs._tts_audio_buffer == []


# --------------------------------------------------------------------------- #
# TTS-8 — late TTSUsageMetricsData lands on closed span via _last_tts_span;
#         a new TTS start clears the backref so later metrics never leak back.
# --------------------------------------------------------------------------- #
async def test_late_tts_metrics_backref_and_clear_on_new_start(
    ff, real_trace_with_turn
):
    # Guards: the _last_tts_span backref handshake + start-time clear preventing
    # stale leakage of a later metric onto an already-closed span.
    from pipecat.metrics.metrics import TTSUsageMetricsData

    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=False)
    obs._trace = trace
    obs._current_turn_span = turn
    src = _source(has_settings=False)

    # Span 1: open then close.
    await obs._handle_tts_started(
        types.SimpleNamespace(frame=ff.TTSStartedFrame(), source=src)
    )
    span1 = obs._active_tts_span
    await obs._handle_tts_stopped(types.SimpleNamespace())
    assert obs._last_tts_span is span1
    assert span1.is_finished()

    # Late metric lands on the finished span1 via the backref.
    frame = ff.MetricsFrame(data=[TTSUsageMetricsData(processor="tts", value=42)])
    await obs._handle_metrics(types.SimpleNamespace(frame=frame))
    assert span1.attributes["tts.characters"] == 42

    # New TTS start clears the backref.
    await obs._handle_tts_started(
        types.SimpleNamespace(frame=ff.TTSStartedFrame(), source=src)
    )
    span2 = obs._active_tts_span
    assert obs._last_tts_span is None

    # A later metric now targets the active span2, leaving span1 untouched.
    frame2 = ff.MetricsFrame(data=[TTSUsageMetricsData(processor="tts", value=99)])
    await obs._handle_metrics(types.SimpleNamespace(frame=frame2))
    assert span2.attributes["tts.characters"] == 99
    assert span1.attributes["tts.characters"] == 42  # unchanged: no stale leak


# --------------------------------------------------------------------------- #
# TTS-9 — on_push_frame routes real TTS frames through the dispatch table
# --------------------------------------------------------------------------- #
async def test_tts_dispatch_through_on_push_frame(ff, real_trace_with_turn):
    # Guards: dispatch-table registration (exact-type miss / handler typo) that
    # direct handler calls cannot catch — full started→text→stopped cycle.
    trace, turn = real_trace_with_turn
    obs = _make_obs(capture_text=True, record_audio=False)
    obs._trace = trace
    obs._current_turn_span = turn
    src = _source(has_settings=False)

    started = ff.TTSStartedFrame()
    text = ff.TTSTextFrame(text="Hi", aggregated_by="word")
    stopped = ff.TTSStoppedFrame()
    # Distinct frame ids so on_push_frame's frame.id dedup does not drop them.
    assert started.id != text.id != stopped.id

    await obs.on_push_frame(types.SimpleNamespace(frame=started, source=src))
    await obs.on_push_frame(types.SimpleNamespace(frame=text, source=src))
    await obs.on_push_frame(types.SimpleNamespace(frame=stopped, source=src))

    tts_spans = [s for s in trace.spans if s.name == "pipecat.tts"]
    assert len(tts_spans) == 1
    span = tts_spans[0]
    assert span.parent_span_id == turn.span_id
    assert span.attributes["tts.input_text"] == "Hi"
    assert span.attributes["pipecat_span_status"] == "ok"
