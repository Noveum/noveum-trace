"""
Tier 1 (C1 / S2) — ``observe_pipeline`` auto-inserts the ``AudioBufferProcessor``.

Spec mapping (``.cursor/plans/pipecat-plan-tier-1-abp.md``,
``Noveum_Pipecat_SDK_Integration_Spec.md`` C1/S2):

* **C1** — full-conversation audio works with **zero** customer pipeline edits;
  the customer no longer has to add ``AudioBufferProcessor(num_channels=2)``.
* **S2** — stereo (``num_channels=2``) so user/bot channels separate without
  manual channel config.

The hard constraint (plan §1): you cannot append to an already-built ``Pipeline``
(linking is frozen at construction), so ``observe_pipeline`` **rebuilds and
returns** a new pipeline.  The non-negotiable invariant (plan §1, spec
"non-intrusive guarantee"): it **only appends** — the customer's processors keep
their original instances and order, nothing is reordered or wrapped.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _tracer(**kwargs: Any) -> Any:
    pytest.importorskip("pipecat.processors.audio.audio_buffer_processor")
    from noveum_trace.integrations.pipecat import NoveumPipecatTracer

    return NoveumPipecatTracer(**kwargs)


def _is_abp(proc: Any) -> bool:
    return any(b.__name__ == "AudioBufferProcessor" for b in type(proc).__mro__)


# --------------------------------------------------------------------------- #
# Append at the tail, preserve customer order/instances                       #
# --------------------------------------------------------------------------- #
def test_abp_appended_at_tail_when_absent(
    make_pipeline: Any, passthrough_processors: Any
) -> None:
    from pipecat.pipeline.pipeline import Pipeline

    procs = passthrough_processors(3)
    pipeline = Pipeline(procs)
    tracer = _tracer(record_audio=True)

    result = tracer.observe_pipeline(pipeline)

    # Rebuilt → a new Pipeline object (the old one is discarded).
    assert result is not pipeline

    inner = result._processors[1:-1]  # strip Pipecat's auto Source/Sink
    # Exactly one ABP, and it is LAST among the customer's processors (tail).
    abp_indices = [i for i, p in enumerate(inner) if _is_abp(p)]
    assert len(abp_indices) == 1
    assert abp_indices[0] == len(inner) - 1

    # The customer's original processors are the SAME instances, in the SAME
    # order — nothing was reordered or wrapped (non-intrusive guarantee).
    customer_procs = inner[:-1]
    assert customer_procs == procs
    for original, kept in zip(procs, customer_procs):
        assert original is kept


def test_abp_is_stereo(make_pipeline: Any) -> None:
    """S2: the inserted ABP is dual-channel so user/bot audio separate with no
    manual channel config."""
    tracer = _tracer(record_audio=True)
    result = tracer.observe_pipeline(make_pipeline(2))

    abp = next(p for p in result._processors if _is_abp(p))
    # Pipecat stores it as `_num_channels`; assert the stereo contract.
    assert getattr(abp, "_num_channels", None) == 2


# --------------------------------------------------------------------------- #
# Idempotent — never add a second ABP                                         #
# --------------------------------------------------------------------------- #
def test_abp_not_duplicated_when_already_present(passthrough_processors: Any) -> None:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

    customer_abp = AudioBufferProcessor(num_channels=2)
    procs = [*passthrough_processors(2), customer_abp]
    pipeline = Pipeline(procs)
    tracer = _tracer(record_audio=True)

    result = tracer.observe_pipeline(pipeline)

    # The customer already supplied an ABP → passthrough, no rebuild, none added.
    assert result is pipeline
    inner = result._processors[1:-1]
    assert sum(1 for p in inner if _is_abp(p)) == 1
    assert customer_abp in inner


def test_abp_detects_subclassed_audio_buffer_processor(
    passthrough_processors: Any,
) -> None:
    """Idempotency uses the MRO name check, so a customer's *subclass* of
    AudioBufferProcessor also counts — we don't add a redundant one."""
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

    class MyABP(AudioBufferProcessor):
        pass

    procs = [*passthrough_processors(1), MyABP(num_channels=2)]
    pipeline = Pipeline(procs)
    tracer = _tracer(record_audio=True)

    result = tracer.observe_pipeline(pipeline)
    assert result is pipeline  # subclass detected → passthrough


# --------------------------------------------------------------------------- #
# record_audio=False — never insert                                           #
# --------------------------------------------------------------------------- #
def test_no_abp_when_record_audio_false(make_pipeline: Any) -> None:
    tracer = _tracer(record_audio=False)
    pipeline = make_pipeline(2)
    result = tracer.observe_pipeline(pipeline)

    assert result is pipeline  # passthrough
    assert not any(_is_abp(p) for p in result._processors)


# --------------------------------------------------------------------------- #
# Rebuilt pipeline re-links and the observer wires recording end-to-end       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inserted_abp_is_detected_and_recording_started(
    make_pipeline: Any,
) -> None:
    """After ``observe_pipeline`` + ``register_task_handlers``, the observer must
    discover the auto-inserted ABP and call ``start_recording()`` — the full C1
    chain with no manual ABP."""
    from pipecat.pipeline.task import PipelineTask

    tracer = _tracer(record_audio=True, record_raw_input_audio=False)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline)

    await tracer.register_task_handlers(task)

    abp = tracer.observer._audio_buffer_processor
    assert abp is not None and _is_abp(abp)
    # The auto-inserted ABP is the one the observer attached to.
    assert tracer.observer._abp_is_recording is True


def test_rebuilt_pipeline_reuses_same_processor_instances(
    passthrough_processors: Any,
) -> None:
    """Rebuild must reuse the customer's processor instances (not clones) so any
    state/handlers the customer already wired survive.

    Force the rebuild path first (``result is not pipeline``) so this does not
    silently pass on a reverted/passthrough implementation, then assert the
    customer's processors are the SAME objects at the SAME positions.
    """
    from pipecat.pipeline.pipeline import Pipeline

    procs = passthrough_processors(2)
    pipeline = Pipeline(procs)
    tracer = _tracer(record_audio=True)

    result = tracer.observe_pipeline(pipeline)

    # The ABP-insert path must have actually rebuilt (not returned the original).
    assert result is not pipeline
    customer = result._processors[1:-1][:-1]  # drop auto source/sink + appended ABP
    assert len(customer) == len(procs)
    for original, kept in zip(procs, customer):
        assert original is kept  # identity, positionally
