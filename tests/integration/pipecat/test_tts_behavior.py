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
from datetime import datetime
from unittest.mock import patch

import pytest

pytest.importorskip("pipecat.frames.frames")
pytest.importorskip("pipecat.metrics.metrics")

# Make the async intent explicit (the repo's asyncio_mode="auto" already
# auto-discovers these, but this keeps them portable across pytest configs).
pytestmark = pytest.mark.asyncio

_UPLOAD = "noveum_trace.integrations.pipecat._handlers_tts.upload_audio_frames"
_TO_THREAD = "noveum_trace.integrations.pipecat._handlers_tts.asyncio.to_thread"


def _make_obs(*, capture_text: bool = True, record_audio: bool = True):
    """Fresh real observer (no mocked spans) with the given opt-in flags."""
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    return NoveumTraceObserver(capture_text=capture_text, record_audio=record_audio)


def _source(*, voice=None, model=None, language=None, has_settings=True):
    """Fake TTS source processor exposing ``._settings.voice`` / ``.model`` /
    ``.language``."""
    if not has_settings:
        return types.SimpleNamespace(_settings=None)
    return types.SimpleNamespace(
        _settings=types.SimpleNamespace(voice=voice, model=model, language=language)
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


async def test_tts_request_opens_span_and_prestart_metrics_attach(
    ff, real_trace_with_turn
):
    """Request-stage metrics belong to the span before TTSStarted arrives."""
    from pipecat.metrics.metrics import (
        ProcessingMetricsData,
        TextAggregationMetricsData,
        TTSUsageMetricsData,
    )

    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=False)
    obs._trace = trace
    obs._current_turn_span = turn
    src = _source(voice="nova")
    obs.register_processor_role(src, "tts")
    context_id = "ctx-request-first"
    processor_name = obs._processor_registry.get(src).name

    token = ff.LLMTextFrame("Hello")
    await obs.on_push_frame(
        types.SimpleNamespace(frame=token, source=object(), destination=src)
    )
    span = obs._active_tts_span
    assert span.attributes["tts.aggregation_started_at"] == span.start_time.isoformat()
    assert "tts.requested_at" not in span.attributes

    # Pipecat stops the first text-aggregation metric immediately before it
    # emits the first context-bearing AggregatedTextFrame.
    pre_request_metrics = ff.MetricsFrame(
        data=[TextAggregationMetricsData(processor=processor_name, value=0.01)]
    )
    await obs.on_push_frame(
        types.SimpleNamespace(frame=pre_request_metrics, source=src)
    )

    first = ff.AggregatedTextFrame("Hello", "sentence", context_id=context_id)
    second = ff.AggregatedTextFrame("world", "sentence", context_id=context_id)
    # The upstream copy must not consume the frame ID before the TTS service
    # itself emits the context-bearing request boundary.
    await obs.on_push_frame(
        types.SimpleNamespace(frame=first, source=object(), destination=src)
    )
    assert obs._active_tts_span is span
    await obs.on_push_frame(types.SimpleNamespace(frame=first, source=src))
    await obs.on_push_frame(types.SimpleNamespace(frame=second, source=src))

    assert span is obs._active_tts_span
    assert span.parent_span_id == turn.span_id
    assert span.attributes["tts.context_id"] == context_id
    assert (
        datetime.fromisoformat(span.attributes["tts.requested_at"]) >= span.start_time
    )
    assert "tts.started_at" not in span.attributes

    metrics = ff.MetricsFrame(
        data=[
            TTSUsageMetricsData(processor=processor_name, value=5),
            TTSUsageMetricsData(processor=processor_name, value=5),
            ProcessingMetricsData(processor=processor_name, value=0.1),
            ProcessingMetricsData(processor=processor_name, value=0.2),
            TextAggregationMetricsData(processor=processor_name, value=0.02),
        ]
    )
    await obs.on_push_frame(types.SimpleNamespace(frame=metrics, source=src))

    started = ff.TTSStartedFrame(context_id=context_id)
    await obs.on_push_frame(types.SimpleNamespace(frame=started, source=src))
    assert span is obs._active_tts_span
    assert "tts.started_at" in span.attributes

    await obs.on_push_frame(
        types.SimpleNamespace(
            frame=ff.TTSStoppedFrame(context_id=context_id), source=src
        )
    )
    # A late provider delta updates the native total without replacing the
    # canonical complete-input count derived at finalization.
    await obs.on_push_frame(
        types.SimpleNamespace(
            frame=ff.MetricsFrame(
                data=[TTSUsageMetricsData(processor=processor_name, value=7)]
            ),
            source=src,
        )
    )

    assert len([item for item in trace.spans if item.name == "pipecat.tts"]) == 1
    assert not [item for item in trace.spans if item.name.startswith("pipecat.metric")]
    assert span.attributes["tts.input_text"] == "Hello world"
    assert span.attributes["tts.input_characters"] == 11
    assert span.attributes["tts.characters"] == 11
    assert span.attributes["tts.provider_reported_characters"] == 17
    assert span.attributes["tts.processing_observations_ms"] == pytest.approx(
        [100, 200]
    )
    assert span.attributes["tts.processing_total_ms"] == pytest.approx(300)
    assert span.attributes["tts.text_aggregation_observations_ms"] == pytest.approx(
        [10, 20]
    )
    assert span.attributes["tts.text_aggregation_ms"] == pytest.approx(30)


async def test_skip_tts_text_does_not_open_tts_operation(ff, real_trace_with_turn):
    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=False)
    obs._trace = trace
    obs._current_turn_span = turn
    src = _source(voice="nova")
    obs.register_processor_role(src, "tts")
    frame = ff.LLMTextFrame("This text must not be synthesized")
    frame.skip_tts = True

    await obs.on_push_frame(
        types.SimpleNamespace(frame=frame, source=object(), destination=src)
    )

    assert obs._active_tts_span is None
    assert not [span for span in trace.spans if span.name == "pipecat.tts"]


async def test_tts_request_without_started_preserves_input_on_interruption(
    ff, real_trace_with_turn
):
    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=False)
    obs._trace = trace
    obs._current_turn_span = turn
    src = _source(voice="nova")
    obs.register_processor_role(src, "tts")
    request = ff.AggregatedTextFrame(
        "Prepared but interrupted", "sentence", context_id="ctx-no-start"
    )

    await obs.on_push_frame(types.SimpleNamespace(frame=request, source=src))
    span = obs._active_tts_span
    await obs._handle_interruption_internal(interrupted_by_user=True)

    assert span.is_finished()
    assert span.attributes["tts.input_text"] == "Prepared but interrupted"
    assert span.attributes["tts.input_characters"] == len("Prepared but interrupted")
    assert span.attributes["tts.output.complete"] is False
    assert span.attributes["tts.termination_reason"] == "user_interruption"
    assert "tts.started_at" not in span.attributes


# --------------------------------------------------------------------------- #
# D6 — tts.language is captured from source settings (was dropped)             #
# --------------------------------------------------------------------------- #
async def test_tts_started_captures_language(ff, real_trace_with_turn):
    # Guards D6: extract_service_settings resolves language; _handle_tts_started
    # now copies it (it previously copied only voice/model).
    trace, turn = real_trace_with_turn
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn
    src = _source(voice="nova", model="tts-1", language="en-US")

    await obs._handle_tts_started(
        types.SimpleNamespace(frame=ff.TTSStartedFrame(), source=src)
    )

    assert obs._active_tts_span.attributes["tts.language"] == "en-US"


# --------------------------------------------------------------------------- #
# D5 — tts.provider derived from the source service class name                 #
# --------------------------------------------------------------------------- #
async def test_tts_started_captures_provider(ff, real_trace_with_turn):
    trace, turn = real_trace_with_turn
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn

    class ElevenLabsTTSService:
        _settings = types.SimpleNamespace(voice="rachel", model="eleven_turbo_v2")

    await obs._handle_tts_started(
        types.SimpleNamespace(frame=ff.TTSStartedFrame(), source=ElevenLabsTTSService())
    )

    assert obs._active_tts_span.attributes["tts.provider"] == "elevenlabs"


# --------------------------------------------------------------------------- #
# D8 — buffered TTS text is flushed on the finalizer force-close path          #
# --------------------------------------------------------------------------- #
async def test_tts_text_flushed_on_finalizer_force_close(ff, real_trace_with_turn):
    # Guards D8: an abnormal close (no TTSStoppedFrame) must still write the
    # buffered TTS text — the finalizer force-close loop flushes it.
    from unittest.mock import patch

    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=False)
    obs._trace = trace
    obs._current_turn_span = turn

    await obs._handle_tts_started(
        types.SimpleNamespace(frame=ff.TTSStartedFrame(), source=_source(voice="v"))
    )
    tts_span = obs._active_tts_span
    # Sentence-aggregated text buffered, but NO TTSStoppedFrame arrives.
    obs._tts_text_buffer.extend([("Hello,", False), ("world.", False)])

    with patch.object(obs, "_get_client", return_value=None):
        await obs._finish_conversation(cancelled=True)

    assert tts_span.attributes["tts.input_text"] == "Hello, world."
    assert tts_span.is_finished()


async def test_tts_interruption_preserves_partial_text_and_is_idempotent(
    ff, real_trace_with_turn
):
    """An interruption finalizes partial synthesis without exporting cancelled."""
    from noveum_trace.core.span import SpanStatus

    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=False)
    obs._trace = trace
    obs._current_turn_span = turn
    source = _source(voice="v")

    await obs._handle_tts_started(
        types.SimpleNamespace(frame=ff.TTSStartedFrame(), source=source)
    )
    span = obs._active_tts_span
    await obs._handle_tts_text(
        types.SimpleNamespace(
            frame=ff.TTSTextFrame(text="Partial response", aggregated_by="sentence"),
            source=source,
        )
    )

    await obs._handle_interruption_internal(interrupted_by_user=True)
    first_end_time = span.end_time
    await obs._handle_interruption_internal(interrupted_by_user=True)

    assert span.attributes["tts.input_text"] == "Partial response"
    assert span.attributes["tts.output.complete"] is False
    assert span.attributes["tts.termination_reason"] == "user_interruption"
    assert span.attributes["pipecat_span_status"] == "cancelled"
    assert span.status is SpanStatus.OK
    assert span.end_time == first_end_time


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
# TTS-3 — word/token chunks (no inter-frame spaces) are RESPACED into the        #
#         interim attribute; tts.input_text stays absent (sentence-only)         #
# --------------------------------------------------------------------------- #
async def test_tts_word_chunks_respace_into_interim(ff, real_trace_with_turn):
    # Guards: de-spacing fix — word frames carry includes_inter_frame_spaces=False,
    # so they must be rejoined WITH spaces (not mashed together) and land in the
    # interim attribute; tts.input_text is sentence-only and stays absent here.
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
    for chunk in ("Hello", "world"):  # default includes_inter_frame_spaces=False
        await obs._handle_tts_text(
            types.SimpleNamespace(
                frame=ff.TTSTextFrame(text=chunk, aggregated_by="word")
            )
        )

    await obs._handle_tts_stopped(types.SimpleNamespace())

    assert span.attributes["tts.input_text_interim"] == "Hello world"  # respaced
    assert "tts.input_text" not in span.attributes  # sentence-only; none emitted
    assert span.is_finished()
    assert obs._tts_text_buffer == []
    assert obs._tts_text_interim_buffer == []
    assert obs._last_tts_span is span
    assert span.attributes["pipecat_span_status"] == "ok"


# --------------------------------------------------------------------------- #
# TTS-3a — frames that already include inter-frame spaces are NOT double-spaced  #
# --------------------------------------------------------------------------- #
async def test_tts_chunks_with_spaces_not_double_spaced(ff, real_trace_with_turn):
    # Guards: includes_inter_frame_spaces=True path — append as-is, no injected space.
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
    for chunk in ("Hello", " world"):  # second carries its own leading space
        frame = ff.TTSTextFrame(text=chunk, aggregated_by="word")
        frame.includes_inter_frame_spaces = True
        await obs._handle_tts_text(types.SimpleNamespace(frame=frame))
    await obs._handle_tts_stopped(types.SimpleNamespace())
    assert span.attributes["tts.input_text_interim"] == "Hello world"


# --------------------------------------------------------------------------- #
# TTS-3b — interim (word/token) vs final (sentence) kept in SEPARATE attrs       #
# --------------------------------------------------------------------------- #
async def test_tts_text_interim_and_final_split(ff, real_trace_with_turn):
    # Guards: sentence-aggregated -> tts.input_text; word/token -> interim; the
    # same utterance is not double-counted, and each is independently respaced.
    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=False)
    obs._trace = trace
    obs._current_turn_span = turn

    await obs._handle_tts_started(
        types.SimpleNamespace(
            frame=ff.TTSStartedFrame(), source=_source(has_settings=False)
        )
    )
    span = obs._active_tts_span
    for w in ("Hello", "world"):  # interim word stream (no inter-frame spaces)
        await obs._handle_tts_text(
            types.SimpleNamespace(frame=ff.TTSTextFrame(text=w, aggregated_by="word"))
        )
    await obs._handle_tts_text(  # final sentence aggregation
        types.SimpleNamespace(
            frame=ff.TTSTextFrame(text="Hello world!", aggregated_by="sentence")
        )
    )
    await obs._handle_tts_stopped(types.SimpleNamespace())

    assert span.attributes["tts.input_text"] == "Hello world!"  # final (sentence)
    assert span.attributes["tts.input_text_interim"] == "Hello world"  # words respaced
    assert obs._tts_text_buffer == []
    assert obs._tts_text_interim_buffer == []


# --------------------------------------------------------------------------- #
# TTS-3c — word-only: interim populated, tts.input_text absent (no fallback)     #
# --------------------------------------------------------------------------- #
async def test_tts_word_only_leaves_input_text_absent(ff, real_trace_with_turn):
    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=False)
    obs._trace = trace
    obs._current_turn_span = turn
    await obs._handle_tts_started(
        types.SimpleNamespace(
            frame=ff.TTSStartedFrame(), source=_source(has_settings=False)
        )
    )
    span = obs._active_tts_span
    for w in ("Got", "it"):
        await obs._handle_tts_text(
            types.SimpleNamespace(frame=ff.TTSTextFrame(text=w, aggregated_by="word"))
        )
    await obs._handle_tts_stopped(types.SimpleNamespace())
    assert span.attributes["tts.input_text_interim"] == "Got it"
    assert "tts.input_text" not in span.attributes


# --------------------------------------------------------------------------- #
# TTS-3e — start/stop timestamps + audio_duration_ms are standalone attributes  #
#          and do NOT overwrite the span's wall-clock start_time / duration_ms   #
# --------------------------------------------------------------------------- #
async def test_tts_timing_attributes(ff, real_trace_with_turn):
    # Guards: tts.started_at (from TTSStarted), tts.stopped_at (from TTSStopped),
    # tts.audio_duration_ms (summed from buffered PCM), all separate from span
    # wall-clock (which stays true elapsed, not the audio length).
    trace, turn = real_trace_with_turn
    obs = _make_obs(record_audio=True)
    obs._trace = trace
    obs._current_turn_span = turn

    await obs._handle_tts_started(
        types.SimpleNamespace(
            frame=ff.TTSStartedFrame(), source=_source(has_settings=False)
        )
    )
    span = obs._active_tts_span
    # started_at is set on TTSStarted and equals the span's own start.
    assert span.attributes["tts.started_at"] == span.start_time.isoformat()

    # 100 frames x 10ms (320 bytes @ 16k mono 16-bit) = 1000ms of audio.
    with (
        patch(_UPLOAD, return_value=True),
        patch(_TO_THREAD, side_effect=lambda fn, *args: fn(*args)),
        patch.object(obs, "_get_client", return_value=None),
    ):
        for _ in range(100):
            await obs._handle_tts_audio(
                types.SimpleNamespace(
                    frame=_audio_frame(ff), source=obs._tts_source_processor
                )
            )
        await obs._handle_tts_stopped(types.SimpleNamespace())

    assert span.attributes["tts.audio_duration_ms"] == pytest.approx(1000.0, rel=1e-6)
    assert span.attributes["tts.stopped_at"] == span.end_time.isoformat()
    # span wall-clock is the real elapsed (near-zero in a test), NOT the audio length.
    assert span.duration_ms < 500.0


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

    with (
        patch(_UPLOAD, return_value=True) as up,
        patch(_TO_THREAD, side_effect=lambda fn, *args: fn(*args)),
        patch.object(obs, "_get_client", return_value=None),
    ):
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

    with (
        patch(_UPLOAD, **upload_kwargs),
        patch(_TO_THREAD, side_effect=lambda fn, *args: fn(*args)),
        patch.object(obs, "_get_client", return_value=None),
    ):
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
    assert span.attributes["tts.input_text_interim"] == "Hi"  # word frame -> interim
    assert span.attributes["pipecat_span_status"] == "ok"
