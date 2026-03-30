"""Tests for noveum_trace.integrations.pipecat public API."""

from __future__ import annotations

import pytest


def test_setup_pipecat_tracing_returns_observer() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat import (
        NoveumTraceObserver,
        setup_pipecat_tracing,
    )

    obs = setup_pipecat_tracing(record_audio=True)
    assert isinstance(obs, NoveumTraceObserver)
    assert obs._record_audio is True


def test_pipecat_all_exports() -> None:
    import noveum_trace.integrations.pipecat as m

    assert set(m.__all__) == {"NoveumTraceObserver", "setup_pipecat_tracing"}


def test_setup_pipecat_tracing_forwards_kwargs_to_observer() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat import setup_pipecat_tracing

    obs = setup_pipecat_tracing(record_audio=False, trace_name_prefix="custom")
    assert obs._record_audio is False
    assert obs._trace_name_prefix == "custom"


def test_setup_pipecat_tracing_rejects_api_key() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat import setup_pipecat_tracing

    with pytest.raises(TypeError, match="api_key"):
        setup_pipecat_tracing(api_key="x")  # type: ignore[call-arg]
