"""
Version-compatibility / cross-version regression tests (§K, VC-1..6).

Subsystem: the pipecat integration absorbs version differences almost entirely
by class-name string dispatch (``getattr(pipecat.frames.frames, name, None)``)
and ``try/except ImportError``. These value-asserting regression tests (per
PIPECAT_TEST_PLAN.md §K) pin that robustness: the dispatch table builds cleanly
when version-specific frames are absent, the genuine StartFrame / message-frame /
interruption-frame forks behave correctly, the dead-on-both handlers are never
wired through a real frame, and audio→WAV reconstruction is byte-identical.

This file is designed to be GREEN on BOTH supported pipecat lines:
  - 1.3.0 (the working venv): version-specific frames are absent → "absent" legs run.
  - 0.0.108 (the declared floor): those frames are present → "present" legs run.
Each fork test self-gates with ``hasattr`` / ``skipif`` so the leg that does not
apply to the installed line SKIPS cleanly rather than failing.

Probe-confirmed on pipecat 1.3.0:
  - LLMMessagesFrame / LLMUsageMetricsFrame / SystemLogFrame / StartInterruptionFrame
    are all ``None`` (absent); their handlers are NOT registered.
  - StartFrame has none of allow_interruptions / sample_rate / audio_sample_rate
    → _handle_start_frame captures no ``pipeline.*`` key.
  - The replace path still works via LLMMessagesUpdateFrame (1.x).
  - InterruptionFrame drives _handle_interruption (cancel + turn was_interrupted).
"""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

pytest.importorskip("pipecat.frames.frames")

import pipecat.frames.frames as ff  # noqa: E402

from noveum_trace.core.trace import Trace  # noqa: E402
from noveum_trace.integrations.pipecat.pipecat_observer import (  # noqa: E402
    NoveumTraceObserver,
)
from noveum_trace.integrations.pipecat.pipecat_utils import (  # noqa: E402
    _frames_to_wav_bytes,
    calculate_audio_duration_ms,
)

# Frames that the integration registers via getattr-string dispatch but that are
# absent on at least one supported line. The handler-method names they map to.
_VERSION_SPECIFIC = {
    "LLMMessagesFrame": "_handle_llm_messages_replace",
    "LLMUsageMetricsFrame": "_handle_llm_usage_metrics",
    "SystemLogFrame": "_handle_system_log",
    "StartInterruptionFrame": "_handle_interruption",
}

# Handlers that are DEAD on both supported lines (the frames they would dispatch
# from are absent in both 0.0.108 and 1.3.0). These must never be reachable
# through a real frame's exact type.
_DEAD_ON_BOTH = {
    "LLMUsageMetricsFrame": "_handle_llm_usage_metrics",
    "SystemLogFrame": "_handle_system_log",
}


def _registered_handler_names(obs: NoveumTraceObserver) -> set[str]:
    """The set of bound handler method names in the observer's dispatch map."""
    names: set[str] = set()
    for handler in obs._frame_handlers.values():
        name = getattr(handler, "__name__", None)
        if name:
            names.add(name)
    return names


def _registered_frame_type_names(obs: NoveumTraceObserver) -> set[str]:
    """The set of frame-class names that have an entry in the dispatch map."""
    return {frame_type.__name__ for frame_type in obs._frame_handlers}


# --------------------------------------------------------------------------- #
# VC-1 — import-robustness: dispatch table builds with absent frames           #
# --------------------------------------------------------------------------- #
def test_vc1_dispatch_table_builds_with_absent_version_frames() -> None:
    # Guards: the getattr/try-import robustness the whole compat strategy rests on
    # — a regression to a hard top-level frame import would crash import on 1.3.0.
    obs = NoveumTraceObserver(capture_text=True, record_audio=True)

    # The dispatch map built without raising and registered a meaningful set.
    assert isinstance(obs._frame_handlers, dict)
    assert len(obs._frame_handlers) > 0
    # Core frames present on every line must always be registered by exact type.
    frame_type_names = _registered_frame_type_names(obs)
    for always_present in (
        "StartFrame",
        "EndFrame",
        "TranscriptionFrame",
        "LLMFullResponseStartFrame",
        "TTSStartedFrame",
        "MetricsFrame",
    ):
        assert getattr(ff, always_present, None) is not None
        assert always_present in frame_type_names

    # For any version-specific frame that is ABSENT on this line, the integration
    # must neither expose the class nor register a dispatch entry of that type.
    frame_handler_keys = set(obs._frame_handlers)
    for frame_name in _VERSION_SPECIFIC:
        cls = getattr(ff, frame_name, None)
        if cls is None:
            assert frame_name not in frame_type_names
            assert cls not in frame_handler_keys


# --------------------------------------------------------------------------- #
# VC-2 — StartFrame config fork                                                #
# --------------------------------------------------------------------------- #
def _start_frame_has_allow_interruptions() -> bool:
    sf = ff.StartFrame()
    return hasattr(sf, "allow_interruptions")


@pytest.mark.skipif(
    not _start_frame_has_allow_interruptions(),
    reason="0.0.x-only: StartFrame.allow_interruptions removed on 1.x",
)
async def test_vc2_start_frame_captures_pipeline_allow_interruptions_old() -> None:
    # Guards: the StartFrame config fork (old leg) — pipeline.allow_interruptions
    # IS captured onto the trace on 0.0.x where StartFrame carries the field.
    obs = NoveumTraceObserver(record_audio=False)
    trace = Trace(name="pipecat.conversation")
    obs._trace = trace

    data = types.SimpleNamespace(frame=ff.StartFrame(), source=None)
    await obs._handle_start_frame(data)

    assert "pipeline.allow_interruptions" in trace.attributes
    assert obs._trace is trace


@pytest.mark.skipif(
    _start_frame_has_allow_interruptions(),
    reason="1.x-only: StartFrame still carries allow_interruptions on 0.0.x",
)
async def test_vc2_start_frame_captures_no_pipeline_attrs_new() -> None:
    # Guards: the belief that pipeline.* is populated on 1.x. Real 1.3.0 StartFrame
    # exposes audio_in_sample_rate/audio_out_sample_rate, NOT the probed
    # allow_interruptions / sample_rate / audio_sample_rate, so NO pipeline.* key
    # is written; the trace still exists. (Pairs with OBS-1.)
    obs = NoveumTraceObserver(record_audio=False)
    trace = Trace(name="pipecat.conversation")
    obs._trace = trace

    data = types.SimpleNamespace(frame=ff.StartFrame(), source=None)
    await obs._handle_start_frame(data)

    pipeline_keys = [k for k in trace.attributes if k.startswith("pipeline.")]
    assert pipeline_keys == []
    assert obs._trace is trace


# --------------------------------------------------------------------------- #
# VC-3 — LLMMessagesFrame replace path is 0.0.x-only                            #
# --------------------------------------------------------------------------- #
def _obs_with_open_llm_context() -> tuple[NoveumTraceObserver, Trace, Any]:
    obs = NoveumTraceObserver(record_audio=False)
    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    obs._trace = trace
    obs._current_turn_span = turn
    # external mode so _handle_llm_response_start won't auto-open a new turn
    obs._using_external_turn_tracking = True
    return obs, trace, turn


@pytest.mark.skipif(
    getattr(ff, "LLMMessagesFrame", None) is None,
    reason="0.0.x-only: LLMMessagesFrame removed on 1.x",
)
async def test_vc3_llm_messages_frame_replace_path_old() -> None:
    # Guards: the removed-frame replace path — on 0.0.x LLMMessagesFrame routes to
    # _handle_llm_messages_replace and its messages flush to llm.input.
    obs, _trace, _turn = _obs_with_open_llm_context()

    msgs = [{"role": "user", "content": "hi"}]
    frame = ff.LLMMessagesFrame(messages=msgs)  # type: ignore[attr-defined]
    await obs._handle_llm_messages_replace(types.SimpleNamespace(frame=frame))
    await obs._handle_llm_response_start(types.SimpleNamespace(frame=None, source=None))

    span = obs._active_llm_span
    assert json.loads(span.attributes["llm.input"]) == msgs


@pytest.mark.skipif(
    getattr(ff, "LLMMessagesFrame", None) is not None,
    reason="1.x-only: LLMMessagesFrame still present on 0.0.x",
)
async def test_vc3_replace_path_via_update_frame_new() -> None:
    # Guards: that 1.x retains a working replace path via LLMMessagesUpdateFrame
    # despite LLMMessagesFrame being removed — the message stash flushes to
    # llm.input exactly as the legacy frame did.
    obs, _trace, _turn = _obs_with_open_llm_context()

    # The removed frame must not be present nor dispatchable as its own type.
    assert getattr(ff, "LLMMessagesFrame", None) is None

    msgs = [{"role": "user", "content": "hi"}]
    update_frame = ff.LLMMessagesUpdateFrame(messages=msgs)
    await obs._handle_llm_messages_replace(types.SimpleNamespace(frame=update_frame))
    assert obs._pending_llm_context.get("messages") == json.dumps(msgs)

    await obs._handle_llm_response_start(types.SimpleNamespace(frame=None, source=None))
    span = obs._active_llm_span
    assert json.loads(span.attributes["llm.input"]) == msgs


# --------------------------------------------------------------------------- #
# VC-4 — StartInterruptionFrame registration is 0.0.x-only                      #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    getattr(ff, "StartInterruptionFrame", None) is None,
    reason="0.0.x-only: StartInterruptionFrame removed on 1.x",
)
def test_vc4_start_interruption_frame_registered_old() -> None:
    # Guards: the removed-subclass registration — on 0.0.x the StartInterruptionFrame
    # subclass is explicitly wired to _handle_interruption (exact-type dispatch
    # would otherwise miss it).
    obs = NoveumTraceObserver(record_audio=False)
    start_interruption = ff.StartInterruptionFrame  # type: ignore[attr-defined]
    assert start_interruption in obs._frame_handlers
    assert obs._frame_handlers[start_interruption].__name__ == "_handle_interruption"


@pytest.mark.skipif(
    getattr(ff, "StartInterruptionFrame", None) is not None,
    reason="1.x-only: StartInterruptionFrame still present on 0.0.x",
)
async def test_vc4_interruption_via_interruption_frame_new() -> None:
    # Guards: that interruption handling does not silently vanish on 1.x where
    # StartInterruptionFrame is removed — a real InterruptionFrame still drives
    # _handle_interruption, cancelling active llm/tts spans and marking the turn.
    assert getattr(ff, "StartInterruptionFrame", None) is None
    assert (
        ff.InterruptionFrame in NoveumTraceObserver(record_audio=False)._frame_handlers
    )

    obs = NoveumTraceObserver(record_audio=False)
    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    obs._trace = trace
    obs._current_turn_span = turn
    llm = trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)
    tts = trace.create_span(name="pipecat.tts", parent_span_id=turn.span_id)
    obs._active_llm_span = llm
    obs._active_tts_span = tts

    await obs._handle_interruption(types.SimpleNamespace(frame=ff.InterruptionFrame()))

    assert turn.attributes["turn.was_interrupted"] is True
    assert llm.attributes["pipecat_span_status"] == "cancelled" and llm.is_finished()
    assert tts.attributes["pipecat_span_status"] == "cancelled" and tts.is_finished()
    assert obs._active_llm_span is None
    assert obs._active_tts_span is None


# --------------------------------------------------------------------------- #
# VC-5 — dead-on-both handlers are NOT wired through a real frame               #
# --------------------------------------------------------------------------- #
def test_vc5_dead_on_both_frames_absent_and_not_registered() -> None:
    # Guards: an accidental hard-import regression that would make the absent-frame
    # handlers crash dispatch; documents that the live token path is
    # MetricsFrame→LLMUsageMetricsData (LLM-9 / MET-1/2), not the standalone frame.
    obs = NoveumTraceObserver(record_audio=False)

    registered_methods = _registered_handler_names(obs)
    registered_type_names = _registered_frame_type_names(obs)

    for frame_name, handler_name in _DEAD_ON_BOTH.items():
        assert getattr(ff, frame_name, None) is None
        assert frame_name not in registered_type_names
        assert handler_name not in registered_methods


# --------------------------------------------------------------------------- #
# VC-6 — audio-frame WAV parity                                                 #
# --------------------------------------------------------------------------- #
def test_vc6_input_audio_frame_wav_parity() -> None:
    # Guards: a reconstruction-by-kwarg regression on either line — the same PCM
    # through InputAudioRawFrame yields a valid RIFF/WAV with a fixed duration, and
    # UserAudioRawFrame (user_id additive) reconstructs byte-identically.
    pcm = b"\x01\x02" * 160  # 160 samples mono 16-bit LE = 320 bytes
    input_frame = ff.InputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)

    assert input_frame.audio == pcm
    assert input_frame.sample_rate == 16000
    assert input_frame.num_channels == 1

    wav = _frames_to_wav_bytes([input_frame])
    assert wav[:4] == b"RIFF"
    # 160 samples / 16000 Hz * 1000 = 10.0 ms
    assert calculate_audio_duration_ms([input_frame]) == 10.0

    user_frame = ff.UserAudioRawFrame(
        audio=pcm, sample_rate=16000, num_channels=1, user_id="u1"
    )
    # user_id is additive metadata; the WAV body and duration are identical.
    assert _frames_to_wav_bytes([user_frame]) == wav
    assert calculate_audio_duration_ms([user_frame]) == 10.0
