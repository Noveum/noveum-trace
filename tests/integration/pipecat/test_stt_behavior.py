"""Value-asserting regression tests for the pipecat STT subsystem (§B, STT-1..11).

Covers ``_handlers_stt._STTHandlersMixin``: the VAD-gated ``pipecat.stt`` span
lifecycle, final-transcript / interim attribution, source-pinned audio buffering,
and the documented trace-root orphan bug under external turn tracking.

Per PIPECAT_TEST_PLAN.md these are written against a REAL ``Trace`` + a real
``pipecat.turn`` span and REAL pipecat frames — they assert span *names*,
attribute *values*, real *parenting* (``parent_span_id``), ``SpanEvent`` names,
and the custom ``pipecat_span_status`` string — never "a mock was called".

XFAIL policy (STT-1/STT-2): the orphan bug (Issues 1 & 2 in
PIPECAT_SPAN_HIERARCHY_ISSUES.md) means a span created under external turn
tracking with no open turn currently parents to None. We assert the DESIRED
behavior (``parent_span_id is not None``) under ``xfail(strict=True)`` so the
test flips to xpass the moment the ``_last_turn_span`` fallback lands. The
genuinely-correct current values (input_type/is_final/status) are pinned in a
SEPARATE non-xfail test so they act as real green regression guards.
"""

from __future__ import annotations

import json
import types

import pytest

pytest.importorskip("pipecat.frames.frames")


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _make_obs(*, capture_text: bool = True, record_audio: bool = False):
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    return NoveumTraceObserver(capture_text=capture_text, record_audio=record_audio)


def _data(frame, *, source=None, direction=None):
    """Build the ``FramePushed``-shaped payload handlers read (.frame/.source/.direction)."""
    return types.SimpleNamespace(frame=frame, source=source, direction=direction)


def _stt_spans(trace):
    return [s for s in trace.spans if s.name == "pipecat.stt"]


@pytest.fixture
def _direction():
    """Real pipecat ``FrameDirection`` enum (skip the file's VAD tests if absent)."""
    mod = pytest.importorskip("pipecat.processors.frame_processor")
    return mod.FrameDirection


# --------------------------------------------------------------------------- #
# STT-1 — XFAIL: final TranscriptionFrame orphans under external tracking      #
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="Issue 1: final transcript orphans to trace-root under external "
    "tracking with no open turn; flips to xpass when the _last_turn_span "
    "fallback lands — see PIPECAT_SPAN_HIERARCHY_ISSUES.md",
)
@pytest.mark.asyncio
async def test_stt1_transcription_orphan_should_be_parented(ff) -> None:
    # Guards: silent change to the documented orphan (Issue 1); desired = parented.
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    obs = _make_obs()
    obs._trace = trace
    obs._using_external_turn_tracking = True
    obs._current_turn_span = None

    frame = ff.TranscriptionFrame(text="orphaned", user_id="u", timestamp="ts")
    await obs._handle_transcription(_data(frame, source=None))

    span = _stt_spans(trace)[0]
    # DESIRED behavior: the span should live under a turn, not at the trace root.
    assert span.parent_span_id is not None


# --------------------------------------------------------------------------- #
# STT-2 — XFAIL orphan (typed input) + SOLID value guards (split)             #
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="Issue 2: typed InputTextRawFrame orphans to trace-root under "
    "external tracking with no open turn; flips to xpass when the "
    "_last_turn_span fallback lands — see PIPECAT_SPAN_HIERARCHY_ISSUES.md",
)
@pytest.mark.asyncio
async def test_stt2_typed_input_orphan_should_be_parented(ff) -> None:
    # Guards: the typed-text face of the orphan bug (Issue 2); desired = parented.
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    obs = _make_obs()
    obs._trace = trace
    obs._using_external_turn_tracking = True
    obs._current_turn_span = None

    await obs._handle_input_text(_data(ff.InputTextRawFrame(text="typed"), source=None))

    span = _stt_spans(trace)[0]
    assert span.parent_span_id is not None


@pytest.mark.asyncio
async def test_stt2_typed_input_pins_input_type_and_status(ff) -> None:
    # Guards: real stt.input_type/is_final/status values on the typed-text point span.
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    obs = _make_obs()
    obs._trace = trace
    obs._using_external_turn_tracking = True
    obs._current_turn_span = None

    await obs._handle_input_text(_data(ff.InputTextRawFrame(text="typed"), source=None))

    span = _stt_spans(trace)[0]
    assert span.attributes["stt.text"] == "typed"
    assert span.attributes["stt.input_type"] == "text"
    assert span.attributes["stt.is_final"] is True
    assert span.attributes["pipecat_span_status"] == "ok"
    assert span.is_finished()


# --------------------------------------------------------------------------- #
# STT-3 — VAD start opens ONE span, final transcript REUSES it under the turn  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stt3_vad_start_opens_span_final_reuses_it(ff, _direction) -> None:
    # Guards: span reuse + turn parenting — a regression that mints a duplicate
    # point span per utterance instead of reusing the VAD-opened span.
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn
    obs._vad_present = True

    vad = ff.VADUserStartedSpeakingFrame()
    await obs._handle_vad_stt_start(
        _data(vad, source=None, direction=_direction.UPSTREAM)
    )
    opened = obs._active_stt_span
    assert opened is not None
    assert opened.name == "pipecat.stt"
    assert opened.parent_span_id == turn.span_id

    final = ff.TranscriptionFrame(text="hello world", user_id="u", timestamp="ts")
    await obs._handle_transcription(_data(final, source=None))

    spans = _stt_spans(trace)
    assert len(spans) == 1  # the VAD span was reused, not duplicated
    assert spans[0] is opened
    assert spans[0].parent_span_id == turn.span_id
    assert spans[0].attributes["stt.is_final"] is True
    assert spans[0].attributes["pipecat_span_status"] == "ok"
    assert spans[0].is_finished()
    # State reset so the next utterance re-opens / re-pins cleanly.
    assert obs._active_stt_span is None
    assert obs._stt_source_processor is None


# --------------------------------------------------------------------------- #
# STT-4 — final transcript pins confidence/language/user_id/timing/interim     #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stt4_final_transcript_pins_all_real_values(ff, _direction) -> None:
    # Guards: attribute extraction/coercion — confidence from frame.result,
    # str-coerced language/user_id, monotonic vad_to_final_ms, interim JSON list.
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn
    obs._vad_present = True

    await obs._handle_vad_stt_start(
        _data(
            ff.VADUserStartedSpeakingFrame(), source=None, direction=_direction.UPSTREAM
        )
    )
    interim = ff.InterimTranscriptionFrame(
        text="que",
        user_id="u",
        timestamp="ts",
        result=types.SimpleNamespace(confidence=0.8),
    )
    await obs._handle_interim_transcription(_data(interim, source=None))

    final = ff.TranscriptionFrame(
        text="question final",
        user_id="u",
        timestamp="ts",
        language="en",
        result=types.SimpleNamespace(confidence=0.91),
    )
    await obs._handle_transcription(_data(final, source=None))

    span = _stt_spans(trace)[0]
    assert span.attributes["stt.text"] == "question final"
    assert span.attributes["stt.is_final"] is True
    assert span.attributes["stt.language"] == "en"
    assert span.attributes["stt.user_id"] == "u"
    assert span.attributes["stt.confidence"] == 0.91

    vad_to_final = span.attributes["stt.vad_to_final_ms"]
    assert isinstance(vad_to_final, float) and vad_to_final > 0

    interim_results = json.loads(span.attributes["stt.interim_results"])
    assert interim_results == [{"text": "que", "confidence": 0.8}]


# --------------------------------------------------------------------------- #
# STT-5 — interim emits a stt.interim_transcription SpanEvent + once-only       #
#         first_text_latency_ms                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stt5_interim_event_and_once_only_latency(ff, _direction) -> None:
    # Guards: SpanEvent name + text/confidence attrs, and that the second interim
    # does NOT overwrite stt.first_text_latency_ms (recorded exactly once).
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn
    obs._vad_present = True

    await obs._handle_vad_stt_start(
        _data(
            ff.VADUserStartedSpeakingFrame(), source=None, direction=_direction.UPSTREAM
        )
    )
    span = obs._active_stt_span

    first = ff.InterimTranscriptionFrame(
        text="que",
        user_id="u",
        timestamp="ts",
        result=types.SimpleNamespace(confidence=0.8),
    )
    await obs._handle_interim_transcription(_data(first, source=None))
    latency_after_first = span.attributes["stt.first_text_latency_ms"]
    assert isinstance(latency_after_first, float) and latency_after_first > 0

    events = [e for e in span.events if e.name == "stt.interim_transcription"]
    assert len(events) == 1
    assert events[0].attributes["text"] == "que"
    assert events[0].attributes["confidence"] == 0.8

    second = ff.InterimTranscriptionFrame(
        text="question",
        user_id="u",
        timestamp="ts",
        result=types.SimpleNamespace(confidence=0.85),
    )
    await obs._handle_interim_transcription(_data(second, source=None))

    # Latency written exactly once — unchanged by the second interim.
    assert span.attributes["stt.first_text_latency_ms"] == latency_after_first
    assert obs._stt_interim_results == [
        {"text": "que", "confidence": 0.8},
        {"text": "question", "confidence": 0.85},
    ]


# --------------------------------------------------------------------------- #
# STT-6 — interim with NO active STT span falls back to the turn span          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stt6_interim_falls_back_to_turn_no_latency(ff) -> None:
    # Guards: target_span = _active_stt_span or _current_turn_span fallback, and
    # that first_text_latency_ms is NOT recorded when there is no active STT span.
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn
    obs._active_stt_span = None

    interim = ff.InterimTranscriptionFrame(
        text="hi",
        user_id="u",
        timestamp="ts",
        result=types.SimpleNamespace(confidence=0.7),
    )
    await obs._handle_interim_transcription(_data(interim, source=None))

    events = [e for e in turn.events if e.name == "stt.interim_transcription"]
    assert len(events) == 1
    assert events[0].attributes["text"] == "hi"
    assert events[0].attributes["confidence"] == 0.7
    assert "stt.first_text_latency_ms" not in turn.attributes
    assert obs._stt_interim_results == [{"text": "hi", "confidence": 0.7}]


# --------------------------------------------------------------------------- #
# STT-7 — downstream VAD-start duplicate does not open a second span           #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stt7_downstream_vad_duplicate_no_second_span(ff, _direction) -> None:
    # Guards: the broadcast-duplicate guard (upstream then downstream) against
    # opening two pipecat.stt spans for one utterance.
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn
    obs._vad_present = True

    await obs._handle_vad_stt_start(
        _data(
            ff.VADUserStartedSpeakingFrame(), source=None, direction=_direction.UPSTREAM
        )
    )
    first = obs._active_stt_span
    await obs._handle_vad_stt_start(
        _data(
            ff.VADUserStartedSpeakingFrame(),
            source=None,
            direction=_direction.DOWNSTREAM,
        )
    )

    assert len(_stt_spans(trace)) == 1
    assert obs._active_stt_span is first


# --------------------------------------------------------------------------- #
# STT-8 — a new VAD utterance cancels a still-open prior STT span              #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stt8_new_utterance_cancels_prior_open_span(ff, _direction) -> None:
    # Guards: orphaned-span cleanup — when an utterance never gets a transcript,
    # the next VAD start cancels it ('cancelled') and opens a fresh span.
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn
    obs._vad_present = True

    await obs._handle_vad_stt_start(
        _data(
            ff.VADUserStartedSpeakingFrame(), source=None, direction=_direction.UPSTREAM
        )
    )
    first = obs._active_stt_span
    await obs._handle_vad_stt_start(
        _data(
            ff.VADUserStartedSpeakingFrame(), source=None, direction=_direction.UPSTREAM
        )
    )
    second = obs._active_stt_span

    assert first.attributes["pipecat_span_status"] == "cancelled"
    assert first.is_finished()
    assert second is not first
    assert len(_stt_spans(trace)) == 2


# --------------------------------------------------------------------------- #
# STT-9 — audio source pinning DROPS frames from a foreign source              #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stt9_audio_source_pinning_drops_foreign(ff) -> None:
    # Guards: de-dup of downstream re-emitted audio — only the pinned source's
    # frames are buffered; a foreign source is silently dropped.
    from noveum_trace.core.trace import Trace

    obs = _make_obs(record_audio=True)
    obs._trace = Trace(name="pipecat.conversation")
    source_a = types.SimpleNamespace(name="A")
    source_b = types.SimpleNamespace(name="B")
    frame = ff.UserAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)

    await obs._handle_user_audio(_data(frame, source=source_a))  # pins A
    assert obs._stt_source_processor is source_a
    assert len(obs._stt_audio_buffer) == 1

    await obs._handle_user_audio(_data(frame, source=source_b))  # foreign -> dropped
    assert len(obs._stt_audio_buffer) == 1

    await obs._handle_user_audio(_data(frame, source=source_a))  # pinned -> buffered
    assert len(obs._stt_audio_buffer) == 2


# --------------------------------------------------------------------------- #
# STT-10 — opt-out gates: record_audio=False and capture_text=False            #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stt10_optout_gates(ff) -> None:
    # Guards: the two privacy opt-outs — audio is not buffered when
    # record_audio=False; interims are not recorded when capture_text=False.
    from noveum_trace.core.trace import Trace

    no_audio = _make_obs(record_audio=False)
    no_audio._trace = Trace(name="pipecat.conversation")
    frame = ff.UserAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)
    await no_audio._handle_user_audio(_data(frame, source=types.SimpleNamespace()))
    assert no_audio._stt_audio_buffer == []
    assert no_audio._stt_source_processor is None

    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    no_text = _make_obs(capture_text=False)
    no_text._trace = trace
    no_text._current_turn_span = turn
    no_text._active_stt_span = None
    interim = ff.InterimTranscriptionFrame(text="hi", user_id="u", timestamp="ts")
    await no_text._handle_interim_transcription(_data(interim, source=None))
    assert no_text._stt_interim_results == []
    assert [e for e in turn.events if e.name == "stt.interim_transcription"] == []


# --------------------------------------------------------------------------- #
# STT-11 — stt.model pulled from STT processor settings on final transcript    #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stt11_model_from_processor_settings(ff, _direction) -> None:
    # Guards: model attribution via extract_service_settings — present when the
    # source processor exposes _settings.model, absent when source is None.
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    obs = _make_obs()
    obs._trace = trace
    obs._current_turn_span = turn
    obs._vad_present = True

    await obs._handle_vad_stt_start(
        _data(
            ff.VADUserStartedSpeakingFrame(), source=None, direction=_direction.UPSTREAM
        )
    )
    with_model_span = obs._active_stt_span
    source = types.SimpleNamespace(_settings=types.SimpleNamespace(model="nova-2"))
    final = ff.TranscriptionFrame(text="hi", user_id="u", timestamp="ts")
    await obs._handle_transcription(_data(final, source=source))
    assert with_model_span.attributes["stt.model"] == "nova-2"

    # Second utterance with source=None: stt.model must be absent.
    await obs._handle_vad_stt_start(
        _data(
            ff.VADUserStartedSpeakingFrame(), source=None, direction=_direction.UPSTREAM
        )
    )
    no_model_span = obs._active_stt_span
    await obs._handle_transcription(
        _data(
            ff.TranscriptionFrame(text="bye", user_id="u", timestamp="ts"), source=None
        )
    )
    assert "stt.model" not in no_model_span.attributes
