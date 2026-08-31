"""Unit tests for typed Pipecat metric routing."""

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
        TextAggregationMetricsData,
        TTFBMetricsData,
        TTSUsageMetricsData,
    )

    return SimpleNamespace(
        MetricsFrame=MetricsFrame,
        LLMTokenUsage=LLMTokenUsage,
        LLMUsageMetricsData=LLMUsageMetricsData,
        ProcessingMetricsData=ProcessingMetricsData,
        TextAggregationMetricsData=TextAggregationMetricsData,
        TTFBMetricsData=TTFBMetricsData,
        TTSUsageMetricsData=TTSUsageMetricsData,
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
            metric_types.ProcessingMetricsData(
                processor="speech", model=None, value=0.1
            ),
            metric_types.ProcessingMetricsData(
                processor="reasoner", model=None, value=0.2
            ),
            metric_types.ProcessingMetricsData(
                processor="voice", model=None, value=0.3
            ),
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
async def test_post_end_usage_enriches_without_refinishing_span(metric_types) -> None:
    obs = _observer()
    source = _processor(obs, "anthropic", PROCESSOR_ROLE_LLM)
    span = _span("llm")
    _start_llm(obs, source, span)

    await obs._handle_llm_response_end(SimpleNamespace(source=source))
    usage = metric_types.LLMTokenUsage(
        prompt_tokens=7, completion_tokens=9, total_tokens=16
    )
    item = metric_types.LLMUsageMetricsData(
        processor="anthropic", model="claude", value=usage
    )
    await obs._handle_metrics(
        SimpleNamespace(frame=metric_types.MetricsFrame(data=[item]), source=source)
    )

    span.finish.assert_called_once()
    assert span.attributes["llm.input_tokens"] == 7
    assert obs._llm_operations.metrics_pending_operations


@pytest.mark.asyncio
async def test_pre_start_late_metric_does_not_attach_to_next_call(metric_types) -> None:
    obs = _observer()
    obs._trace = MagicMock()
    source = _processor(obs, "sequential", PROCESSOR_ROLE_LLM)
    first, second = _span("first"), _span("second")
    obs._llm_operations.start(
        source,
        span=first,
        processor_name="sequential",
        start_frame_id=100,
    )
    obs._llm_operations.complete(source)
    obs._llm_operations.start(
        source,
        span=second,
        processor_name="sequential",
        start_frame_id=200,
    )
    diagnostic = _span("diagnostic")
    obs._create_child_span = MagicMock(return_value=diagnostic)
    item = metric_types.TTFBMetricsData(processor="sequential", model=None, value=0.4)
    frame = metric_types.MetricsFrame(data=[item])
    frame.id = 150

    await obs._handle_metrics(SimpleNamespace(frame=frame, source=source))

    assert "llm.ttfb_ms" not in second.attributes
    attrs = obs._create_child_span.call_args.kwargs["attributes"]
    assert attrs["metric.attribution_result"] == "zero_matches"


@pytest.mark.asyncio
async def test_ambiguous_same_name_metric_gets_diagnostic_span(metric_types) -> None:
    obs = _observer()
    obs._trace = MagicMock()
    source_a = _processor(obs, "same-name", PROCESSOR_ROLE_LLM)
    source_b = _processor(obs, "same-name", PROCESSOR_ROLE_LLM)
    _start_llm(obs, source_a, _span("a"))
    _start_llm(obs, source_b, _span("b"))
    diagnostic = _span("diagnostic")
    obs._create_child_span = MagicMock(return_value=diagnostic)
    item = metric_types.TTFBMetricsData(processor="same-name", model=None, value=0.5)

    await obs._handle_metrics(
        SimpleNamespace(
            frame=metric_types.MetricsFrame(data=[item]),
            source=None,
            destination=None,
            direction="downstream",
            timestamp=123,
        )
    )

    attrs = obs._create_child_span.call_args.kwargs["attributes"]
    assert attrs["metric.attribution_result"] == "multiple_matches"
    assert attrs["metric.candidate_count"] == 2
    assert attrs["metric.rollup_eligible"] is False
    diagnostic.finish.assert_called_once()


@pytest.mark.asyncio
async def test_zero_match_metric_gets_diagnostic_span(metric_types) -> None:
    obs = _observer()
    obs._trace = MagicMock()
    diagnostic = _span("diagnostic")
    obs._create_child_span = MagicMock(return_value=diagnostic)
    item = metric_types.ProcessingMetricsData(
        processor="missing", model=None, value=0.1
    )

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
    assert attrs["metric.attribution_result"] == "zero_matches"
    assert attrs["metric.producer_timestamp"] == 456
    assert attrs["metric.processor_candidate_count"] == 0
    assert attrs["metric.operation_candidate_count"] == 0


@pytest.mark.asyncio
async def test_combined_llm_tts_processor_routes_specific_and_diagnoses_generic(
    metric_types,
) -> None:
    obs = _observer()
    obs._trace = MagicMock()
    source = SimpleNamespace(name="realtime")
    obs.register_processor_role(source, PROCESSOR_ROLE_LLM)
    obs.register_processor_role(source, PROCESSOR_ROLE_TTS)
    llm_span, tts_span = _span("llm"), _span("tts")
    _start_llm(obs, source, llm_span)
    obs._active_tts_span = tts_span
    obs._tts_source_processor = source

    usage = metric_types.TTSUsageMetricsData(processor="realtime", model=None, value=42)
    await obs._handle_metrics(
        SimpleNamespace(frame=metric_types.MetricsFrame(data=[usage]), source=source)
    )
    assert tts_span.attributes["tts.characters"] == 42
    assert "tts.characters" not in llm_span.attributes

    diagnostic = _span("diagnostic")
    obs._create_child_span = MagicMock(return_value=diagnostic)
    generic = metric_types.ProcessingMetricsData(
        processor="realtime", model=None, value=0.25
    )
    await obs._handle_metrics(
        SimpleNamespace(frame=metric_types.MetricsFrame(data=[generic]), source=source)
    )
    attrs = obs._create_child_span.call_args.kwargs["attributes"]
    assert attrs["metric.attribution_result"] == "multiple_matches"
    assert attrs["metric.operation_candidate_count"] == 2


@pytest.mark.asyncio
async def test_equal_processing_observations_from_distinct_frames_both_survive(
    metric_types,
) -> None:
    obs = _observer()
    source = _processor(obs, "llm", PROCESSOR_ROLE_LLM)
    span = _span("llm")
    operation = _start_llm(obs, source, span)

    for _ in range(2):
        item = metric_types.ProcessingMetricsData(
            processor="llm", model=None, value=0.1
        )
        await obs._handle_metrics(
            SimpleNamespace(frame=metric_types.MetricsFrame(data=[item]), source=source)
        )

    assert span.attributes["llm.processing_observations_ms"] == [100.0, 100.0]
    assert len(operation.raw_metrics) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "active_attr", "source_attr", "start_attr"),
    [
        (
            PROCESSOR_ROLE_STT,
            "_active_stt_span",
            "_stt_metric_processor",
            "_stt_start_frame_id",
        ),
        (
            PROCESSOR_ROLE_TTS,
            "_active_tts_span",
            "_tts_source_processor",
            "_tts_start_frame_id",
        ),
    ],
)
async def test_pre_start_stt_tts_metric_is_not_assigned_to_next_operation(
    metric_types, role, active_attr, source_attr, start_attr
) -> None:
    obs = _observer()
    obs._trace = MagicMock()
    source = _processor(obs, role, role)
    active = _span(role)
    setattr(obs, active_attr, active)
    setattr(obs, source_attr, source)
    setattr(obs, start_attr, 200)
    diagnostic = _span("diagnostic")
    obs._create_child_span = MagicMock(return_value=diagnostic)
    metric = metric_types.TTFBMetricsData(processor=role, model=None, value=0.1)
    frame = metric_types.MetricsFrame(data=[metric])
    frame.id = 100

    await obs._handle_metrics(SimpleNamespace(frame=frame, source=source))

    assert f"{role}.ttfb_ms" not in active.attributes
    attrs = obs._create_child_span.call_args.kwargs["attributes"]
    assert attrs["metric.attribution_result"] == "zero_matches"


@pytest.mark.asyncio
async def test_pre_start_tts_aggregation_does_not_attach_to_previous_span(
    metric_types,
) -> None:
    obs = _observer()
    obs._trace = MagicMock()
    source = _processor(obs, "tts", PROCESSOR_ROLE_TTS)
    previous = _span("previous")
    obs._last_tts_span = previous
    obs._last_tts_source_processor = source
    diagnostic = _span("diagnostic")
    obs._create_child_span = MagicMock(return_value=diagnostic)
    metric = metric_types.TextAggregationMetricsData(
        processor="tts", model=None, value=0.2
    )

    await obs._handle_metrics(
        SimpleNamespace(frame=metric_types.MetricsFrame(data=[metric]), source=source)
    )

    assert "tts.text_aggregation_ms" not in previous.attributes
    attrs = obs._create_child_span.call_args.kwargs["attributes"]
    assert attrs["metric.attribution_result"] == "zero_matches"
