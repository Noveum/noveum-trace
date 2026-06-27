"""Value-asserting regression tests for the Pipecat turn-manager subsystem.

Subsystem: ``_turn_manager._TurnManagerMixin`` — turn-span lifecycle
(``_start_new_turn`` / ``_end_current_turn``), external-vs-standalone turn
detection, EOU/latency landing, and interruption teardown.

These tests follow the ``§F`` (TM-1..13) section of ``PIPECAT_TEST_PLAN.md``.
They drive the real handlers against a real ``Trace`` (no MagicMock spans) and
assert span *names*, attribute *values*, ``parent_span_id`` parenting, the
custom ``pipecat_span_status`` string, and ``SpanEvent`` names — never merely
that a mock was called or a key is present.

The external-wiring tests use a **real** ``TurnTrackingObserver`` and drain its
``_event_tasks`` before asserting, because Pipecat dispatches async event
handlers via ``asyncio.create_task`` (not awaited inline).
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("pipecat.frames.frames")

from noveum_trace.core.trace import Trace  # noqa: E402
from noveum_trace.integrations.pipecat.pipecat_constants import (  # noqa: E402
    SPAN_STT,
    SPAN_TURN,
)
from noveum_trace.integrations.pipecat.pipecat_observer import (  # noqa: E402
    NoveumTraceObserver,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _obs_with_trace(**kwargs):
    """An observer backed by a real, empty conversation ``Trace``."""
    obs = NoveumTraceObserver(turn_end_timeout_secs=0.01, **kwargs)
    obs._trace = Trace(name="pipecat.conversation")
    return obs


def _dummy_data():
    """A ``data`` payload with ``.frame``/``.source`` like Pipecat passes."""
    return types.SimpleNamespace(frame=object(), source=object())


async def _drain_event_tasks(tracker) -> None:
    """Await the async handler tasks a ``TurnTrackingObserver`` scheduled.

    ``_event_tasks`` is a set of ``(event_name, asyncio.Task)`` tuples; the
    handlers run via ``asyncio.create_task`` and are NOT awaited inline, so
    asserting without draining is a guaranteed false-pass.
    """
    for _name, task in list(tracker._event_tasks):
        await task


# --------------------------------------------------------------------------- #
# TM-1 — _start_new_turn span name / root parenting / turn.number              #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_start_new_turn_creates_root_turn_span() -> None:
    # Guards: SPAN_TURN name, intentional trace-root parenting (Issue 4), and
    # the turn.number / turn_count the dashboard groups on.
    obs = _obs_with_trace()

    await obs._start_new_turn()

    assert len(obs._trace.spans) == 1
    span = obs._trace.spans[0]
    assert span.name == SPAN_TURN == "pipecat.turn"
    assert span.parent_span_id is None  # intentional root (Issue 4), NOT an orphan
    assert span.attributes["turn.number"] == 1
    assert obs._current_turn_number == 1
    assert obs._metrics_accumulator["turn_count"] == 1


# --------------------------------------------------------------------------- #
# TM-2 — standalone increment vs absolute external numbering + auto-close      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_turn_numbering_increment_absolute_and_autoclose() -> None:
    # Guards: the absolute-vs-increment branch + auto-close of the prior turn.
    obs = _obs_with_trace()

    await obs._start_new_turn()  # -> 1
    await obs._start_new_turn()  # -> 2
    await obs._start_new_turn(turn_number=7)  # absolute -> 7
    await obs._start_new_turn()  # -> 8

    spans = obs._trace.spans
    assert [s.attributes["turn.number"] for s in spans] == [1, 2, 7, 8]
    assert obs._current_turn_number == 8
    # Each new start auto-closes the previous turn; only the last stays open.
    assert [s.is_finished() for s in spans] == [True, True, True, False]


# --------------------------------------------------------------------------- #
# TM-3 — _end_current_turn duration / joined user_input / status               #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_end_current_turn_writes_values_and_finishes() -> None:
    # Guards: joined-transcript value, default was_interrupted, explicit
    # duration, and the 'ok' status write on a clean turn end.
    obs = _obs_with_trace()
    await obs._start_new_turn()
    obs._transcription_buffer = ["hello", "world"]

    await obs._end_current_turn(was_interrupted=False, duration=2.5)

    span = obs._trace.spans[0]
    assert span.attributes["turn.user_input"] == "hello world"
    assert span.attributes["turn.was_interrupted"] is False
    assert span.attributes["turn.duration_seconds"] == 2.5
    assert span.attributes["pipecat_span_status"] == "ok"
    assert obs._current_turn_span is None
    assert span.is_finished()


# --------------------------------------------------------------------------- #
# TM-4 — turn boundary preserves in-flight STT span + audio buffer             #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_end_current_turn_preserves_inflight_stt() -> None:
    # Guards: the always-buffer invariant — a turn boundary must never cancel an
    # in-flight STT span or clear its audio (transcript may still be arriving).
    obs = _obs_with_trace()
    await obs._start_new_turn()
    turn = obs._current_turn_span

    stt_span = obs._trace.create_span(name=SPAN_STT, parent_span_id=turn.span_id)
    obs._active_stt_span = stt_span
    audio_frame = object()
    obs._stt_audio_buffer = [audio_frame]

    await obs._end_current_turn()

    assert obs._active_stt_span is stt_span
    assert stt_span.is_finished() is False
    assert obs._stt_audio_buffer == [audio_frame]
    assert turn.is_finished()


# --------------------------------------------------------------------------- #
# TM-5 — interruption cancels llm/tts, sets backrefs, clears buffers           #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_interruption_internal_cancels_and_clears() -> None:
    # Guards: cancelled-status on llm/tts + late-billing backrefs + the full
    # buffer cleanup + turn.was_interrupted on interruption.
    obs = _obs_with_trace()
    await obs._start_new_turn()
    turn = obs._current_turn_span

    llm = obs._trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)
    tts = obs._trace.create_span(name="pipecat.tts", parent_span_id=turn.span_id)
    obs._active_llm_span = llm
    obs._active_tts_span = tts
    obs._tts_source_processor = object()
    obs._llm_text_buffer = ["x"]
    obs._tts_text_buffer = ["y"]
    obs._tts_audio_buffer = [object()]
    obs._pending_function_calls = {"a": {}}
    obs._function_call_results = [{}]

    await obs._handle_interruption_internal(interrupted_by_user=True)

    assert llm.attributes["pipecat_span_status"] == "cancelled"
    assert tts.attributes["pipecat_span_status"] == "cancelled"
    assert llm.is_finished()
    assert tts.is_finished()
    assert obs._active_llm_span is None
    assert obs._active_tts_span is None
    # Backrefs let late MetricsFrames bill the interrupted call.
    assert obs._last_llm_span is llm
    assert obs._last_tts_span is tts
    assert obs._tts_source_processor is None
    assert turn.attributes["turn.was_interrupted"] is True
    assert obs._llm_text_buffer == []
    assert obs._tts_text_buffer == []
    assert obs._tts_audio_buffer == []
    assert obs._pending_function_calls == {}
    assert obs._function_call_results == []


# --------------------------------------------------------------------------- #
# TM-6 — real TurnTrackingObserver drives start/end via (emitter, *args)       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_external_turn_tracking_observer_drives_lifecycle() -> None:
    # Guards: the emitter-first handler signature + add_event_handler
    # subscription wiring against pipecat API drift. (Drain or false-pass.)
    pytest.importorskip("pipecat.observers.turn_tracking_observer")
    from pipecat.observers.turn_tracking_observer import TurnTrackingObserver

    obs = _obs_with_trace()
    tto = TurnTrackingObserver()
    obs.attach_turn_tracking_observer(tto)

    assert obs._using_external_turn_tracking is True
    assert obs._turn_tracker is tto

    await tto._call_event_handler("on_turn_started", 5)
    await _drain_event_tasks(tto)

    assert len(obs._trace.spans) == 1
    span = obs._trace.spans[0]
    assert span.attributes["turn.number"] == 5
    assert span.parent_span_id is None  # intentional root
    assert obs._current_turn_number == 5

    await tto._call_event_handler("on_turn_ended", 5, 3.2, False)
    await _drain_event_tasks(tto)

    assert span.attributes["turn.duration_seconds"] == 3.2  # verbatim from observer
    assert span.attributes["turn.was_interrupted"] is False
    assert span.attributes["pipecat_span_status"] == "ok"
    assert obs._current_turn_span is None


# --------------------------------------------------------------------------- #
# TM-7 — external mode suppresses standalone VAD/user/bot turn handlers        #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_external_mode_suppresses_standalone_handlers() -> None:
    # Guards: external/standalone mutual-exclusion — a regression would
    # double-track turns or double-write latency.
    obs = _obs_with_trace()
    obs._using_external_turn_tracking = True
    data = _dummy_data()

    await obs._handle_vad_user_started(data)
    await obs._handle_vad_user_stopped(data)
    await obs._handle_user_started_speaking(data)
    await obs._handle_user_stopped_speaking(data)

    assert obs._trace.spans == []  # no standalone turn was opened
    assert obs._current_turn_span is None
    assert obs._user_stopped_speaking_time is None  # vad_stopped is a no-op

    # Bot speaking flags still flip (those are NOT mode-gated), but latency is
    # owned by UserBotLatencyObserver in external mode → none written here.
    await obs._handle_bot_started_speaking(data)
    assert obs._is_bot_speaking is True
    assert obs._bot_has_spoken_in_turn is True


# --------------------------------------------------------------------------- #
# TM-8 — _handle_latency_measured lands on turn; no-op without a turn          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_latency_measured_lands_on_turn_and_no_turn_guard() -> None:
    # Guards: the latency-landing path + the no-turn guard.
    obs = _obs_with_trace()
    await obs._start_new_turn()
    turn = obs._current_turn_span

    await obs._handle_latency_measured(0.42)
    assert turn.attributes["turn.user_bot_latency_seconds"] == 0.42

    obs._current_turn_span = None
    await obs._handle_latency_measured(0.9)  # no current turn → must not raise
    # The prior turn's value is left untouched (0.9 was dropped, not written).
    assert turn.attributes["turn.user_bot_latency_seconds"] == 0.42


# --------------------------------------------------------------------------- #
# TM-9 — buffered EOU metrics flush (non-None only) onto the next turn         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_pending_eou_metrics_flush_on_new_turn() -> None:
    # Guards: buffer→turn remap, the non-None filter, and post-flush clear.
    obs = _obs_with_trace()
    obs._pending_turn_eou_metrics = {
        "turn_eou_is_complete": True,
        "turn_eou_confidence": 0.87,
        "turn_eou_inference_ms": 12.0,
        "turn_eou_processing_time_ms": None,
    }

    await obs._start_new_turn()

    span = obs._current_turn_span
    assert span.attributes["turn.eou_is_complete"] is True
    assert span.attributes["turn.eou_confidence"] == 0.87
    assert span.attributes["turn.eou_inference_ms"] == 12.0
    # None value is filtered out, not written.
    assert "turn.eou_processing_time_ms" not in span.attributes
    assert obs._pending_turn_eou_metrics == {}


# --------------------------------------------------------------------------- #
# TM-10 — _deferred_turn_end closes only when the bot is no longer speaking    #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_deferred_turn_end_respects_bot_speaking() -> None:
    # Guards: the `not _is_bot_speaking` guard preventing a mid-speech close.
    # Case A: bot finished → turn closes.
    obs_a = _obs_with_trace()
    await obs_a._start_new_turn()
    span_a = obs_a._current_turn_span
    obs_a._is_bot_speaking = False
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await obs_a._deferred_turn_end()
    assert span_a.is_finished()
    assert obs_a._current_turn_span is None

    # Case B: bot still speaking → turn stays open.
    obs_b = _obs_with_trace()
    await obs_b._start_new_turn()
    span_b = obs_b._current_turn_span
    obs_b._is_bot_speaking = True
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await obs_b._deferred_turn_end()
    assert span_b.is_finished() is False
    assert obs_b._current_turn_span is span_b


# --------------------------------------------------------------------------- #
# TM-11 — _attach_turn_tracker is idempotent, None-safe, sets external flag    #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_attach_turn_tracker_idempotent_and_none_safe() -> None:
    # Guards: identity-based idempotency + None guard so repeated attach calls
    # don't double-register handlers (which would double-fire turn events).
    pytest.importorskip("pipecat.observers.turn_tracking_observer")
    from pipecat.observers.turn_tracking_observer import TurnTrackingObserver

    obs = _obs_with_trace()
    tto = TurnTrackingObserver()

    obs.attach_turn_tracking_observer(tto)
    obs.attach_turn_tracking_observer(tto)  # second identical attach is a no-op

    assert obs._using_external_turn_tracking is True
    assert obs._turn_tracker is tto
    # Exactly one handler per event despite two attach calls.
    assert len(tto._event_handlers["on_turn_started"].handlers) == 1
    assert len(tto._event_handlers["on_turn_ended"].handlers) == 1

    obs.attach_turn_tracking_observer(None)  # None must not raise or reset
    assert obs._turn_tracker is tto


# --------------------------------------------------------------------------- #
# TM-12 — mute SpanEvents carry exact names on the turn span                   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_mute_events_carry_exact_names() -> None:
    # Guards: the observable user.muted / user.unmuted event names + empty attrs.
    obs = _obs_with_trace()
    await obs._start_new_turn()
    turn = obs._current_turn_span

    await obs._handle_user_mute_started(_dummy_data())
    await obs._handle_user_mute_stopped(_dummy_data())

    assert [e.name for e in turn.events] == ["user.muted", "user.unmuted"]
    assert all(e.attributes == {} for e in turn.events)


# --------------------------------------------------------------------------- #
# TM-13 — current pin (green): after turn end, no fallback ref exists          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_after_turn_end_no_current_turn_or_fallback_ref() -> None:
    # Guards (observe-then-pin, current design): after _end_current_turn there
    # is no _current_turn_span and NO _last_turn_span fallback attribute — this
    # is the precondition that makes the next child span orphan (see XFAIL).
    obs = _obs_with_trace()
    await obs._start_new_turn()
    await obs._end_current_turn()

    assert obs._current_turn_span is None
    assert getattr(obs, "_last_turn_span", None) is None


# --------------------------------------------------------------------------- #
# TM-13 — XFAIL: child span created after turn end SHOULD reparent under a turn #
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="Issues 1&2 (PIPECAT_SPAN_HIERARCHY_ISSUES.md): after a turn ends "
    "there is no _last_turn_span fallback, so STT/TTS/LLM child spans created "
    "before the next turn orphan to the trace root. Flips to xpass when the "
    "_last_turn_span fallback fix lands.",
)
@pytest.mark.asyncio
async def test_child_span_after_turn_end_should_not_orphan() -> None:
    # Guards: the root-cause precondition for orphan spans. Asserts the DESIRED
    # behavior (parented, not root); currently fails because the fix is absent.
    obs = _obs_with_trace()
    await obs._start_new_turn()
    await obs._end_current_turn()

    # Exactly how the STT/TTS handlers create their spans — parent_span is the
    # (now None) current turn span.
    child = obs._create_child_span(SPAN_STT, parent_span=obs._current_turn_span)

    assert child.name == SPAN_STT
    # DESIRED: the child should fall back to the just-ended turn, not orphan.
    assert child.parent_span_id is not None
