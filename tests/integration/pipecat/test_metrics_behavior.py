"""
Value-asserting regression tests for the Pipecat Metrics subsystem (§E, MET-1..7).

Covers ``_handlers_metrics._MetricsHandlerMixin._handle_metrics`` (routing / merge /
EOU buffering) and ``pipecat_utils.extract_metrics_data`` (per-type parsing).

This subsystem **never creates spans** — it writes attributes onto pre-existing
LLM / TTS / turn spans (so it cannot itself produce orphans).  These tests drive
real ``Trace`` objects and real ``pipecat.metrics.metrics`` payloads and assert the
emitted attribute *names*, *values*, coercions, cost computation, the EOU
buffer→drain contract, and the TTFB routing precedence — per PIPECAT_TEST_PLAN.md.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("pipecat.frames.frames")
pytest.importorskip("pipecat.metrics.metrics")

from pipecat.frames.frames import MetricsFrame  # noqa: E402
from pipecat.metrics.metrics import (  # noqa: E402
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    SmartTurnMetricsData,
    TextAggregationMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)

from noveum_trace.core.trace import Trace  # noqa: E402
from noveum_trace.integrations.pipecat.pipecat_observer import (  # noqa: E402
    NoveumTraceObserver,
)
from noveum_trace.integrations.pipecat.pipecat_utils import (  # noqa: E402
    extract_metrics_data,
)
from noveum_trace.utils.llm_utils import estimate_cost  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _new_obs() -> NoveumTraceObserver:
    """A fresh observer with a real Trace attached (no live spans)."""
    obs = NoveumTraceObserver(capture_text=True, record_audio=True)
    obs._trace = Trace(name="pipecat.conversation")
    return obs


def _metrics_data(*items):
    """Wrap metric payload items in the ``data=...``/``frame`` shape handlers expect."""
    return types.SimpleNamespace(frame=MetricsFrame(data=list(items)))


# --------------------------------------------------------------------------- #
# MET-1 — metrics land on a FINISHED span via direct attributes[] write       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_metrics_write_to_finished_span_via_direct_attributes() -> None:
    # Guards: a refactor to span.set_attribute() (gated by the finished guard) would
    # silently drop every metric — the COMMON case, since pipecat emits MetricsFrame
    # after the LLM span has already finished.
    obs = _new_obs()
    span = obs._trace.create_span(name="pipecat.llm")
    span.finish()
    assert span.is_finished()
    obs._last_llm_span = span  # most-recently-closed LLM span is the target

    usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    item = LLMUsageMetricsData(processor="llm", model="gpt-4o-mini", value=usage)
    await obs._handle_metrics(_metrics_data(item))

    # Despite the span being finished, the metrics were written directly.
    assert span.attributes["llm.input_tokens"] == 10
    assert span.attributes["llm.output_tokens"] == 20
    assert span.attributes["llm.total_tokens"] == 30
    assert span.attributes["llm.model"] == "gpt-4o-mini"
    assert span.attributes["llm.cost.total"] > 0

    # Control: the public set_attribute() path IS gated by the finished guard and
    # would have dropped the write — proving the handler bypasses it deliberately.
    span.set_attribute("x_control", 1)
    assert "x_control" not in span.attributes


# --------------------------------------------------------------------------- #
# MET-2 — full token + cache + cost + accumulator merge (exact values)        #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_token_usage_writes_full_attr_set_cost_and_accumulator() -> None:
    # Guards: full token+cache+cost+accumulator merge + cache_read_input_tokens→
    # cache_read_tokens rename. Replaces the sloppy >= accumulator assertion.
    obs = _new_obs()
    llm = obs._trace.create_span(name="pipecat.llm")
    obs._active_llm_span = llm

    usage = LLMTokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        cache_read_input_tokens=4,
        reasoning_tokens=3,
    )
    item = LLMUsageMetricsData(processor="llm", model="gpt-4o-mini", value=usage)
    await obs._handle_metrics(_metrics_data(item))

    expected = estimate_cost("gpt-4o-mini", input_tokens=10, output_tokens=20)

    assert llm.attributes["llm.input_tokens"] == 10
    assert llm.attributes["llm.output_tokens"] == 20
    assert llm.attributes["llm.total_tokens"] == 30
    assert llm.attributes["llm.cache_read_tokens"] == 4
    assert llm.attributes["llm.reasoning_tokens"] == 3
    assert llm.attributes["llm.model"] == "gpt-4o-mini"
    assert llm.attributes["llm.cost.input"] == pytest.approx(expected["input_cost"])
    assert llm.attributes["llm.cost.output"] == pytest.approx(expected["output_cost"])
    assert llm.attributes["llm.cost.total"] == pytest.approx(expected["total_cost"])
    assert llm.attributes["llm.cost.currency"] == "USD"

    # Conversation-level accumulator increments by exactly these values.
    assert obs._metrics_accumulator["total_input_tokens"] == 10
    assert obs._metrics_accumulator["total_output_tokens"] == 20
    assert obs._metrics_accumulator["total_cost"] == pytest.approx(
        expected["total_cost"]
    )


# --------------------------------------------------------------------------- #
# MET-3 — EOU metrics buffer with no turn, then drain onto next turn span      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_eou_metrics_buffer_when_no_turn_then_drain_onto_next_turn() -> None:
    # Guards: the DESIGNED buffering-when-no-turn-span contract (not the orphan bug)
    # across _handlers_metrics + _turn_manager._start_new_turn.
    obs = _new_obs()
    obs._current_turn_span = None

    st = SmartTurnMetricsData(
        processor="turn",
        is_complete=True,
        probability=0.9,
        e2e_processing_time_ms=120,
        inference_time_ms=30,
        server_total_time_ms=50,
    )
    await obs._handle_metrics(_metrics_data(st))

    # Buffered, not written to any span yet.
    assert obs._pending_turn_eou_metrics == {
        "turn_eou_is_complete": True,
        "turn_eou_confidence": 0.9,
        "turn_eou_processing_time_ms": 120.0,
        "turn_eou_inference_ms": 30.0,
        "turn_eou_server_total_ms": 50.0,
    }
    assert obs._trace.spans == []

    await obs._start_new_turn()
    turn = obs._current_turn_span

    assert turn.attributes["turn.eou_is_complete"] is True
    assert turn.attributes["turn.eou_confidence"] == 0.9
    assert turn.attributes["turn.eou_processing_time_ms"] == 120.0
    assert turn.attributes["turn.eou_inference_ms"] == 30.0
    assert turn.attributes["turn.eou_server_total_ms"] == 50.0
    # Buffer drained.
    assert obs._pending_turn_eou_metrics == {}


# --------------------------------------------------------------------------- #
# MET-4 — EOU live path writes to current turn + clears prior buffer           #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_eou_live_path_writes_to_turn_and_clears_buffer() -> None:
    # Guards: live-path write + buffer-clear + the Turn-vs-SmartTurn field distinction
    # (plain TurnMetricsData has no inference/server_total fields in 1.3.0).
    obs = _new_obs()
    turn = obs._trace.create_span(name="pipecat.turn")
    obs._current_turn_span = turn
    obs._pending_turn_eou_metrics["stale"] = "leftover"

    tm = TurnMetricsData(
        processor="turn",
        is_complete=False,
        probability=0.4,
        e2e_processing_time_ms=70,
    )
    await obs._handle_metrics(_metrics_data(tm))

    assert turn.attributes["turn.eou_is_complete"] is False
    assert turn.attributes["turn.eou_confidence"] == 0.4
    assert turn.attributes["turn.eou_processing_time_ms"] == 70.0
    # Plain TurnMetricsData carries no SmartTurn-only fields.
    assert "turn.eou_inference_ms" not in turn.attributes
    assert "turn.eou_server_total_ms" not in turn.attributes
    # Stale buffer cleared once the live write lands.
    assert obs._pending_turn_eou_metrics == {}


# --------------------------------------------------------------------------- #
# MET-5 — parser maps each metric type to the right flat key + coercion        #
# --------------------------------------------------------------------------- #
def test_extract_metrics_data_ttfb_processing_tts_and_textagg() -> None:
    # Guards: float/int coercion + the per-type key map for TTFB/Processing/
    # TTSUsage/TextAggregation.
    ttfb = extract_metrics_data(
        MetricsFrame(data=[TTFBMetricsData(processor="OpenAILLMService", value=0.1)])
    )
    assert ttfb["ttfb_seconds"] == 0.1
    assert isinstance(ttfb["ttfb_seconds"], float)
    assert ttfb["ttfb_processor"] == "OpenAILLMService"

    proc = extract_metrics_data(
        MetricsFrame(data=[ProcessingMetricsData(processor="p", value=0.25)])
    )
    assert proc["processing_seconds"] == 0.25
    assert isinstance(proc["processing_seconds"], float)

    tts = extract_metrics_data(
        MetricsFrame(data=[TTSUsageMetricsData(processor="tts", value=42)])
    )
    assert tts["tts_characters"] == 42
    assert isinstance(tts["tts_characters"], int)

    tag = extract_metrics_data(
        MetricsFrame(data=[TextAggregationMetricsData(processor="tts", value=0.05)])
    )
    assert tag["text_aggregation_seconds"] == 0.05
    assert isinstance(tag["text_aggregation_seconds"], float)


def test_extract_metrics_data_llm_usage_and_turn_types() -> None:
    # Guards: token int coercion + llm_model key, plus the SmartTurn-before-Turn
    # isinstance ordering (subclass before parent) and bool/float EOU coercion.
    usage = extract_metrics_data(
        MetricsFrame(
            data=[
                LLMUsageMetricsData(
                    processor="llm",
                    model="gpt-4o",
                    value=LLMTokenUsage(
                        prompt_tokens=5, completion_tokens=7, total_tokens=12
                    ),
                )
            ]
        )
    )
    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 12
    assert all(
        isinstance(usage[k], int)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    assert usage["llm_model"] == "gpt-4o"

    turn = extract_metrics_data(
        MetricsFrame(
            data=[
                TurnMetricsData(
                    processor="t",
                    is_complete=True,
                    probability=0.5,
                    e2e_processing_time_ms=80,
                )
            ]
        )
    )
    assert turn["turn_eou_is_complete"] is True
    assert turn["turn_eou_confidence"] == 0.5
    assert turn["turn_eou_processing_time_ms"] == 80.0
    # Plain Turn carries none of the SmartTurn-only keys.
    assert "turn_eou_inference_ms" not in turn
    assert "turn_eou_server_total_ms" not in turn

    smart = extract_metrics_data(
        MetricsFrame(
            data=[
                SmartTurnMetricsData(
                    processor="t",
                    is_complete=True,
                    probability=0.6,
                    e2e_processing_time_ms=90,
                    inference_time_ms=10,
                    server_total_time_ms=20,
                )
            ]
        )
    )
    # SmartTurn must be matched by its own branch (it subclasses TurnMetricsData),
    # so the inference/server fields appear — proving subclass-before-parent ordering.
    assert smart["turn_eou_is_complete"] is True
    assert smart["turn_eou_confidence"] == 0.6
    assert smart["turn_eou_processing_time_ms"] == 90.0
    assert smart["turn_eou_inference_ms"] == 10.0
    assert smart["turn_eou_server_total_ms"] == 20.0


# --------------------------------------------------------------------------- #
# MET-6 — TTFB routing precedence: LLM wins; TTS-only fallback                  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_ttfb_routes_to_llm_when_both_spans_present() -> None:
    # Guards: case A — with both an LLM and a TTS target, an LLM-named processor
    # routes TTFB to the LLM span and leaves the TTS span untouched.
    obs = _new_obs()
    llm = obs._trace.create_span(name="pipecat.llm")
    tts = obs._trace.create_span(name="pipecat.tts")
    obs._active_llm_span = llm
    obs._active_tts_span = tts

    await obs._handle_metrics(
        _metrics_data(TTFBMetricsData(processor="OpenAILLMService", value=0.1))
    )

    assert llm.attributes["llm.time_to_first_token_ms"] == pytest.approx(100.0)
    assert "tts.time_to_first_byte_ms" not in tts.attributes


@pytest.mark.asyncio
async def test_ttfb_falls_back_to_tts_when_only_tts_present() -> None:
    # Guards: case B — with only a TTS target, even an LLM-named processor falls
    # back onto the TTS span (elif tts_target branch).
    obs = _new_obs()
    tts = obs._trace.create_span(name="pipecat.tts")
    obs._active_tts_span = tts
    obs._active_llm_span = None
    obs._last_llm_span = None

    await obs._handle_metrics(
        _metrics_data(TTFBMetricsData(processor="OpenAILLMService", value=0.2))
    )

    assert tts.attributes["tts.time_to_first_byte_ms"] == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_ttfb_processor_with_both_tts_and_llm_routes_to_llm() -> None:
    # Guards: case C — tts_only requires "tts" in proc AND "llm" not in proc; a
    # processor name containing both must route to the LLM span, not TTS.
    obs = _new_obs()
    llm = obs._trace.create_span(name="pipecat.llm")
    tts = obs._trace.create_span(name="pipecat.tts")
    obs._active_llm_span = llm
    obs._active_tts_span = tts

    await obs._handle_metrics(
        _metrics_data(TTFBMetricsData(processor="tts_and_llm_combo", value=0.3))
    )

    assert llm.attributes["llm.time_to_first_token_ms"] == pytest.approx(300.0)
    assert "tts.time_to_first_byte_ms" not in tts.attributes


# --------------------------------------------------------------------------- #
# MET-7 — TTFB value of exactly 0.0 drops ttfb_seconds (observe-then-pin bug)  #
# --------------------------------------------------------------------------- #
def test_ttfb_zero_value_keeps_processor_but_drops_seconds_pinned() -> None:
    # PIN CURRENT (latent inconsistency, NOT an orphan bug): extract uses
    # `getattr(value) or getattr(ttfb)`, so a falsy 0.0 falls through and no
    # ttfb_seconds key is emitted — though ttfb_processor still is. Every other
    # metric uses `is not None` and keeps 0.0. Pins the falsy-or behavior so a
    # future fix to `is not None` is test-visible.
    result = extract_metrics_data(
        MetricsFrame(data=[TTFBMetricsData(processor="OpenAILLMService", value=0.0)])
    )
    assert "ttfb_seconds" not in result
    assert result["ttfb_processor"] == "OpenAILLMService"


@pytest.mark.asyncio
async def test_ttfb_zero_value_writes_no_latency_to_span_pinned() -> None:
    # PIN CURRENT: because the parser drops ttfb_seconds for a 0.0 value (above),
    # the handler writes no llm.time_to_first_token_ms onto the LLM span.
    obs = _new_obs()
    llm = obs._trace.create_span(name="pipecat.llm")
    obs._active_llm_span = llm

    await obs._handle_metrics(
        _metrics_data(TTFBMetricsData(processor="OpenAILLMService", value=0.0))
    )

    assert "llm.time_to_first_token_ms" not in llm.attributes
