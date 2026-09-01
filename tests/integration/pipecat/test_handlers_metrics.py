"""Behavior tests for typed Pipecat metric attribution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from noveum_trace.integrations.pipecat._processor_registry import (
    PROCESSOR_ROLE_LLM,
    PROCESSOR_ROLE_STT,
    PROCESSOR_ROLE_TTS,
)


@pytest.fixture
def metric_types():
    pytest.importorskip("pipecat.metrics.metrics")
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import (
        LLMTokenUsage,
        LLMUsageMetricsData,
        ProcessingMetricsData,
        TTFBMetricsData,
    )

    return SimpleNamespace(
        MetricsFrame=MetricsFrame,
        LLMTokenUsage=LLMTokenUsage,
        LLMUsageMetricsData=LLMUsageMetricsData,
        ProcessingMetricsData=ProcessingMetricsData,
        TTFBMetricsData=TTFBMetricsData,
    )


def _observer():
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    return NoveumTraceObserver(record_audio=False)


def _span(name: str):
    span = MagicMock()
    span.span_id = name
    span.attributes = {}
    span.is_finished.return_value = False
    return span


def _processor(obs, name: str, role: str):
    source = SimpleNamespace(name=name)
    obs.register_processor_role(source, role)
    return source


def _start_llm(obs, source, span):
    return obs._llm_operations.start(
        source, span=span, processor_name=source.name, model=None
    )


@pytest.mark.asyncio
async def test_multi_item_frame_routes_stt_llm_and_tts_processing(metric_types) -> None:
    obs = _observer()
    stt_source = _processor(obs, "speech", PROCESSOR_ROLE_STT)
    llm_source = _processor(obs, "reasoner", PROCESSOR_ROLE_LLM)
    tts_source = _processor(obs, "voice", PROCESSOR_ROLE_TTS)
    stt_span, llm_span, tts_span = _span("stt"), _span("llm"), _span("tts")
    obs._active_stt_span = stt_span
    obs._stt_metric_processor = stt_source
    _start_llm(obs, llm_source, llm_span)
    obs._active_tts_span = tts_span
    obs._tts_source_processor = tts_source

    frame = metric_types.MetricsFrame(
        data=[
            metric_types.ProcessingMetricsData(processor="speech", value=0.1),
            metric_types.ProcessingMetricsData(processor="reasoner", value=0.2),
            metric_types.ProcessingMetricsData(processor="voice", value=0.3),
        ]
    )
    await obs._handle_metrics(SimpleNamespace(frame=frame, source=None))

    assert stt_span.attributes["stt.processing_ms"] == pytest.approx(100)
    assert llm_span.attributes["llm.processing_ms"] == pytest.approx(200)
    assert tts_span.attributes["tts.processing_ms"] == pytest.approx(300)


@pytest.mark.asyncio
async def test_stt_ttfb_never_routes_to_active_llm(metric_types) -> None:
    obs = _observer()
    stt_source = _processor(obs, "custom-speech", PROCESSOR_ROLE_STT)
    llm_source = _processor(obs, "custom-brain", PROCESSOR_ROLE_LLM)
    stt_span, llm_span = _span("stt"), _span("llm")
    obs._active_stt_span = stt_span
    obs._stt_metric_processor = stt_source
    _start_llm(obs, llm_source, llm_span)

    item = metric_types.TTFBMetricsData(
        processor="custom-speech", model=None, value=0.125
    )
    await obs._handle_metrics(
        SimpleNamespace(frame=metric_types.MetricsFrame(data=[item]), source=stt_source)
    )

    assert stt_span.attributes["stt.ttfb_ms"] == pytest.approx(125)
    assert "llm.ttfb_ms" not in llm_span.attributes


@pytest.mark.asyncio
async def test_late_stt_metric_routes_to_exact_recent_processor(metric_types) -> None:
    obs = _observer()
    source = _processor(obs, "custom-speech", PROCESSOR_ROLE_STT)
    span = _span("stt")
    obs._last_stt_span = span
    obs._last_stt_metric_processor = source

    item = metric_types.TTFBMetricsData(
        processor="custom-speech", model=None, value=0.125
    )
    await obs._handle_metrics(
        SimpleNamespace(frame=metric_types.MetricsFrame(data=[item]), source=source)
    )

    assert span.attributes["stt.ttfb_ms"] == pytest.approx(125)


def test_ttfa_and_incremental_stt_usage_keep_native_breakdowns() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import NormalizedMetricData

    obs = _observer()
    tts_source = _processor(obs, "voice", PROCESSOR_ROLE_TTS)
    stt_source = _processor(obs, "speech", PROCESSOR_ROLE_STT)
    tts_span, stt_span = _span("tts"), _span("stt")
    obs._active_tts_span = tts_span
    obs._tts_source_processor = tts_source
    obs._active_stt_span = stt_span
    obs._stt_metric_processor = stt_source

    ttfa = NormalizedMetricData(
        "TTFAMetricsData",
        "ttfa",
        {"ttfa": 0.4, "ttfb": 0.25, "leading_silence": 0.15},
        "seconds",
        "voice",
        None,
    )
    obs._route_metric(
        SimpleNamespace(source=tts_source), ttfa, "ttfa-1", channel="metrics_frame"
    )
    assert tts_span.attributes["tts.ttfa_ms"] == pytest.approx(400)
    assert tts_span.attributes["tts.ttfb_ms"] == pytest.approx(250)
    assert tts_span.attributes["tts.leading_silence_ms"] == pytest.approx(150)

    usage = NormalizedMetricData(
        "STTUsageMetricsData",
        "stt_usage",
        {"audio_seconds": 1.5},
        "seconds",
        "speech",
        None,
    )
    data = SimpleNamespace(source=stt_source)
    obs._route_metric(data, usage, "usage-1", channel="metrics_frame")
    obs._route_metric(data, usage, "usage-2", channel="metrics_frame")
    assert stt_span.attributes["stt.audio_seconds"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_zero_metric_is_preserved(metric_types) -> None:
    obs = _observer()
    source = _processor(obs, "zero-llm", PROCESSOR_ROLE_LLM)
    span = _span("llm")
    _start_llm(obs, source, span)
    item = metric_types.TTFBMetricsData(processor="zero-llm", model=None, value=0.0)
    await obs._handle_metrics(
        SimpleNamespace(frame=metric_types.MetricsFrame(data=[item]), source=source)
    )
    assert span.attributes["llm.ttfb_ms"] == 0.0


@pytest.mark.asyncio
async def test_duplicate_usage_is_idempotent_and_revision_replaces(
    metric_types,
) -> None:
    obs = _observer()
    source = _processor(obs, "usage-llm", PROCESSOR_ROLE_LLM)
    span = _span("llm")
    operation = _start_llm(obs, source, span)

    async def emit(prompt: int, completion: int) -> None:
        usage = metric_types.LLMTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
        item = metric_types.LLMUsageMetricsData(
            processor="usage-llm", model="gpt-4o-mini", value=usage
        )
        await obs._handle_metrics(
            SimpleNamespace(frame=metric_types.MetricsFrame(data=[item]), source=source)
        )

    await emit(10, 20)
    await emit(10, 20)
    await emit(12, 22)
    assert len(operation.raw_metrics) == 2
    assert span.attributes["llm.input_tokens"] == 12
    assert obs._metrics_accumulator["total_input_tokens"] == 0

    obs._llm_operations.complete(source)
    obs._llm_operations.settle(source)
    totals = obs._llm_operations.sum_metric_fields(
        "llm_usage", ("prompt_tokens", "completion_tokens")
    )
    assert totals == {"prompt_tokens": 12.0, "completion_tokens": 22.0}


@pytest.mark.asyncio
async def test_standalone_usage_frame_uses_same_operation_rollup(metric_types) -> None:
    obs = _observer()
    source = _processor(obs, "usage-llm", PROCESSOR_ROLE_LLM)
    span = _span("llm")
    _start_llm(obs, source, span)
    usage = metric_types.LLMTokenUsage(
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    )
    item = metric_types.LLMUsageMetricsData(
        processor="usage-llm", model="gpt-4o-mini", value=usage
    )
    await obs._handle_metrics(
        SimpleNamespace(frame=metric_types.MetricsFrame(data=[item]), source=source)
    )

    standalone = SimpleNamespace(
        id=987,
        tokens=usage,
        model="gpt-4o-mini",
        processor="usage-llm",
    )
    await obs._handle_llm_usage_metrics(
        SimpleNamespace(frame=standalone, source=source)
    )

    obs._llm_operations.complete(source)
    obs._llm_operations.settle(source)
    totals = obs._llm_operations.sum_metric_fields(
        "llm_usage", ("prompt_tokens", "completion_tokens")
    )
    assert totals == {"prompt_tokens": 10.0, "completion_tokens": 20.0}
    assert span.attributes["llm.input_tokens"] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("with_matches", [False, True])
async def test_unmatched_or_ambiguous_metric_gets_diagnostic_span(
    metric_types, with_matches
) -> None:
    obs = _observer()
    obs._trace = MagicMock()
    if with_matches:
        first = _processor(obs, "same-name", PROCESSOR_ROLE_LLM)
        second = _processor(obs, "same-name", PROCESSOR_ROLE_LLM)
        _start_llm(obs, first, _span("a"))
        _start_llm(obs, second, _span("b"))
        name = "same-name"
        expected = "multiple_matches"
        expected_count = 2
    else:
        name = "missing"
        expected = "zero_matches"
        expected_count = 0

    diagnostic = _span("diagnostic")
    obs._create_child_span = MagicMock(return_value=diagnostic)
    item = metric_types.ProcessingMetricsData(processor=name, value=0.1)
    await obs._handle_metrics(
        SimpleNamespace(
            frame=metric_types.MetricsFrame(data=[item]),
            source=None,
            destination=None,
            direction="downstream",
            timestamp=456,
        )
    )

    attrs = obs._create_child_span.call_args.kwargs["attributes"]
    assert attrs["metric.attribution_result"] == expected
    assert attrs["metric.operation_candidate_count"] == expected_count
    assert attrs["metric.rollup_eligible"] is False
    diagnostic.finish.assert_called_once()
