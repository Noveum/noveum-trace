"""
Regression / parity — the new ``NoveumPipecatTracer`` must not break the
existing observer-based integration, and the two wiring paths must be equivalent.

Covers the user's explicit requirement: *tests that make sure the existing
functionality is not broken.*  Specifically:

1. The **old documented path** (``setup_pipecat_tracing`` + ``observers=[obs]`` +
   ``await obs.attach_to_task(task)``) still attaches and detects audio.
2. The **non-intrusive guarantee** — ``observe_pipeline`` never reorders the
   customer's processors (it only appends the ABP).
3. **Parity** — a tracer-wired observer and a manually-wired observer end up in
   the same delivery list and detect the same full-conversation audio processor
   (plan tier-0 §6).
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _is_abp(proc: Any) -> bool:
    return any(b.__name__ == "AudioBufferProcessor" for b in type(proc).__mro__)


def _pipeline_with_manual_abp(passthrough_processors: Any) -> Any:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

    return Pipeline([*passthrough_processors(2), AudioBufferProcessor(num_channels=2)])


# --------------------------------------------------------------------------- #
# Old documented path still works                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_legacy_observers_kwarg_path_still_attaches(
    passthrough_processors: Any,
) -> None:
    """The pre-tracer integration — ``observers=[obs]`` + ``attach_to_task`` —
    must remain fully functional for customers who have not migrated."""
    pytest.importorskip("pipecat.pipeline.task")
    from pipecat.pipeline.task import PipelineTask

    from noveum_trace.integrations.pipecat import setup_pipecat_tracing

    obs = setup_pipecat_tracing(record_audio=True)
    pipeline = _pipeline_with_manual_abp(passthrough_processors)
    task = PipelineTask(pipeline, observers=[obs])

    await obs.attach_to_task(task)

    # Observer delivered via the classic kwarg, and the manual ABP was detected.
    assert obs in task._observer._observers
    assert obs._audio_buffer_processor is not None
    assert _is_abp(obs._audio_buffer_processor)


def test_setup_pipecat_tracing_alias_unchanged() -> None:
    """The legacy factory still returns a configured observer (back-compat API)."""
    pytest.importorskip("pipecat.observers.base_observer")
    from noveum_trace.integrations.pipecat import (
        NoveumTraceObserver,
        setup_pipecat_tracing,
    )

    obs = setup_pipecat_tracing(record_audio=True, capture_text=False)
    assert isinstance(obs, NoveumTraceObserver)
    assert obs._record_audio is True
    assert obs._capture_text is False


# --------------------------------------------------------------------------- #
# Non-intrusive guarantee — no reordering                                     #
# --------------------------------------------------------------------------- #
def test_observe_pipeline_preserves_distinct_processor_order() -> None:
    """Give each processor a distinct identity and assert the customer order is
    untouched after the ABP is appended (only an append, never a reorder)."""
    pytest.importorskip("pipecat.pipeline.pipeline")
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.frame_processor import FrameProcessor

    from noveum_trace.integrations.pipecat import NoveumPipecatTracer

    class Tagged(FrameProcessor):
        def __init__(self, tag: str) -> None:
            super().__init__()
            self.tag = tag

    a, b, c = Tagged("a"), Tagged("b"), Tagged("c")
    tracer = NoveumPipecatTracer(record_audio=True)

    result = tracer.observe_pipeline(Pipeline([a, b, c]))

    inner = result._processors[1:-1]
    customer = [p for p in inner if not _is_abp(p)]
    assert [p.tag for p in customer] == ["a", "b", "c"]
    assert customer == [a, b, c]  # same instances, same order


# --------------------------------------------------------------------------- #
# Parity — tracer path vs manual path                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_tracer_and_manual_paths_have_equivalent_wiring(
    passthrough_processors: Any,
) -> None:
    """Both wiring styles place the observer in the proxy delivery list and
    detect the full-conversation ABP — the tracer adds no behavioural regression
    versus the documented manual flow."""
    from pipecat.pipeline.task import PipelineTask

    from noveum_trace.integrations.pipecat import (
        NoveumPipecatTracer,
        setup_pipecat_tracing,
    )

    # --- manual path (with a manually-added ABP) ---
    manual_obs = setup_pipecat_tracing(record_audio=True, record_raw_input_audio=False)
    manual_task = PipelineTask(
        _pipeline_with_manual_abp(passthrough_processors), observers=[manual_obs]
    )
    await manual_obs.attach_to_task(manual_task)

    # --- tracer path (ABP auto-inserted, no observers=[...]) ---
    tracer = NoveumPipecatTracer(record_audio=True, record_raw_input_audio=False)
    from pipecat.pipeline.pipeline import Pipeline

    tracer_pipeline = tracer.observe_pipeline(Pipeline(passthrough_processors(2)))
    tracer_task = PipelineTask(tracer_pipeline)
    await tracer.register_task_handlers(tracer_task)

    # Equivalent outcomes:
    assert manual_obs in manual_task._observer._observers
    assert tracer.observer in tracer_task._observer._observers
    assert _is_abp(manual_obs._audio_buffer_processor)
    assert _is_abp(tracer.observer._audio_buffer_processor)
