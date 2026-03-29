"""Unit tests for Pipecat metrics handler mixin (_handlers_metrics)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def metrics_mod():
    return pytest.importorskip("pipecat.metrics.metrics")


@pytest.mark.asyncio
async def test_metrics_ttfb_routes_to_llm_span(metrics_mod) -> None:
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import TTFBMetricsData

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    llm = MagicMock()
    llm.attributes = {}
    obs._active_llm_span = llm
    obs._last_llm_span = None
    obs._active_tts_span = None
    obs._last_tts_span = None
    obs._current_turn_span = None

    item = TTFBMetricsData(processor="OpenAILLMService", model=None, value=0.1)
    data = MagicMock(frame=MetricsFrame(data=[item]))
    await obs._handle_metrics(data)

    assert llm.attributes.get("llm.time_to_first_token_ms") == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_metrics_ttfb_routes_to_tts_when_processor_tts(metrics_mod) -> None:
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import TTFBMetricsData

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    tts = MagicMock()
    tts.attributes = {}
    obs._active_llm_span = None
    obs._active_tts_span = tts

    item = TTFBMetricsData(processor="CartesiaTTSService", model=None, value=0.2)
    data = MagicMock(frame=MetricsFrame(data=[item]))
    await obs._handle_metrics(data)

    assert tts.attributes.get("tts.time_to_first_byte_ms") == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_metrics_token_usage_updates_accumulator(metrics_mod) -> None:
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver()
    llm = MagicMock()
    llm.attributes = {}
    obs._active_llm_span = llm

    usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    item = LLMUsageMetricsData(processor="llm", model="gpt-4o-mini", value=usage)
    data = MagicMock(frame=MetricsFrame(data=[item]))
    await obs._handle_metrics(data)

    assert obs._metrics_accumulator["total_input_tokens"] >= 10
    assert obs._metrics_accumulator["total_output_tokens"] >= 20
    assert llm.attributes.get("llm.input_tokens") == 10
