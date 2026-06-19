"""
Tier 0 (C7) + Tier 4a (C4) + S1 — ``NoveumPipecatTracer`` wiring.

Spec mapping (``Noveum_Pipecat_SDK_Integration_Spec.md`` /
``.cursor/plans/pipecat-plan-tier-0-wiring.md``):

* **C7** — fragile ``attach_to_task`` ordering folded into two stable calls.
* **C4** — ``enable_metrics`` / ``enable_usage_metrics`` auto-set so
  ``MetricsFrame`` (token counts, cost, TTFB) is never silently missing.
* **S1** — pipeline-wrapping single call (``observe_and_create_task``).

The load-bearing Tier-0 risk (plan §7 O1) is *frame delivery*: a post-construction
``add_observer`` must place the observer in the list Pipecat's ``TaskObserver``
proxy actually iterates.  We assert membership in the **real** proxy list rather
than a mock ``add_observer`` call, since the latter would give false confidence on
the one thing Tier 0 could get wrong.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _tracer(**kwargs: Any) -> Any:
    pytest.importorskip("pipecat.observers.base_observer")
    from noveum_trace.integrations.pipecat import NoveumPipecatTracer

    return NoveumPipecatTracer(**kwargs)


# --------------------------------------------------------------------------- #
# Constructor — flags forwarded to the underlying observer                     #
# --------------------------------------------------------------------------- #
def test_constructor_forwards_flags_to_observer() -> None:
    tracer = _tracer(
        record_audio=False,
        record_raw_input_audio=False,
        capture_custom_spans=True,
        auto_enable_metrics=False,
        capture_errors=False,
        capture_system_logs=True,
        capture_session_metadata=False,
    )
    obs = tracer.observer
    assert obs._record_audio is False
    assert obs._record_raw_input_audio is False
    assert obs._auto_enable_metrics is False
    assert obs._capture_errors is False
    assert obs._capture_system_logs is True
    assert obs._capture_session_metadata is False
    # tracer-level flags
    assert tracer._capture_custom_spans is True
    assert tracer._auto_enable_metrics is False
    assert tracer._capture_session_metadata is False


def test_constructor_forwards_extra_observer_kwargs() -> None:
    tracer = _tracer(record_audio=True, trace_name_prefix="custombot")
    assert tracer.observer._trace_name_prefix == "custombot"


def test_constructor_defaults_match_spec() -> None:
    """Spec/plan defaults: audio + raw audio + metrics + session metadata ON,
    custom spans OFF (opt-in OTEL extra)."""
    tracer = _tracer()
    obs = tracer.observer
    assert obs._record_audio is True
    assert obs._record_raw_input_audio is True
    assert obs._auto_enable_metrics is True
    assert obs._capture_session_metadata is True
    assert tracer._capture_custom_spans is False


# --------------------------------------------------------------------------- #
# observe_pipeline — passthrough when audio recording is off                   #
# --------------------------------------------------------------------------- #
def test_observe_pipeline_passthrough_when_record_audio_false(
    make_pipeline: Any,
) -> None:
    """With ``record_audio=False`` and no custom spans, ``observe_pipeline`` is a
    pure passthrough — it must not rebuild or reorder the customer's pipeline."""
    tracer = _tracer(record_audio=False)
    pipeline = make_pipeline(3)
    result = tracer.observe_pipeline(pipeline)
    assert result is pipeline


# --------------------------------------------------------------------------- #
# register_task_handlers — REAL frame-delivery wiring (plan §7 O1)             #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_register_task_handlers_adds_observer_to_real_proxy(
    make_pipeline: Any,
) -> None:
    from pipecat.pipeline.task import PipelineTask

    tracer = _tracer(record_audio=False)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline)

    returned = await tracer.register_task_handlers(task)

    assert returned is task
    # The observer must land in the list the TaskObserver proxy iterates per
    # frame.  This is what guarantees frames reach the observer without the
    # customer writing observers=[...].
    assert tracer.observer in task._observer._observers
    # C7 is two halves: add_observer (above) AND attach_to_task ordering folded
    # in.  Pin that attach actually ran on the two-call path (the surface
    # customers use directly): attach_to_task registers the on_pipeline_finished
    # safety net and records the task here.  Goes False if the `await
    # attach_to_task` line in register_task_handlers is dropped.
    assert task in tracer.observer._registered_pipeline_tasks


@pytest.mark.asyncio
async def test_register_task_handlers_warns_without_add_observer() -> None:
    """If a future PipelineTask lacks ``add_observer``, we warn (and tell the
    customer to fall back to ``observers=[tracer.observer]``) rather than crash.

    The ``noveum_trace`` logger sets ``propagate=False``, so ``caplog`` (which
    captures on the root logger) never sees the record — patch the tracer
    module's logger directly instead.
    """
    from unittest.mock import patch

    import noveum_trace.integrations.pipecat.tracer as tracer_mod

    class _TaskNoAddObserver:
        # No add_observer(); no _params; no event_handler. attach_to_task reads
        # everything via getattr and returns cleanly on this shape.
        turn_tracking_observer = None

    tracer = _tracer(
        record_audio=False,
        record_raw_input_audio=False,
        capture_session_metadata=False,
    )
    task = _TaskNoAddObserver()

    with patch.object(tracer_mod, "logger") as mock_logger:
        returned = await tracer.register_task_handlers(task)  # type: ignore[arg-type]

    assert returned is task  # no crash — degrades gracefully
    warn_msgs = [str(c.args[0]) for c in mock_logger.warning.call_args_list]
    assert any("add_observer" in m for m in warn_msgs)


# --------------------------------------------------------------------------- #
# Turn-tracking fallback (plan §3) — enable_turn_tracking=False                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_turn_tracking_fallback_when_disabled(make_pipeline: Any) -> None:
    """When the customer builds the task with ``enable_turn_tracking=False``,
    ``register_task_handlers`` creates a fallback ``TurnTrackingObserver`` so
    turn spans still match Pipecat's boundaries."""
    from pipecat.pipeline.task import PipelineTask

    tracer = _tracer(record_audio=False)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline, enable_turn_tracking=False)
    assert task.turn_tracking_observer is None  # precondition

    await tracer.register_task_handlers(task)

    # A fallback TurnTrackingObserver was created and exposed for attach_to_task.
    assert getattr(task, "_turn_tracking_observer", None) is not None
    from pipecat.observers.turn_tracking_observer import TurnTrackingObserver

    assert isinstance(task._turn_tracking_observer, TurnTrackingObserver)
    assert task._turn_tracking_observer in task._observer._observers


@pytest.mark.asyncio
async def test_turn_tracking_default_not_clobbered(make_pipeline: Any) -> None:
    """With Pipecat's default ``enable_turn_tracking=True``, the existing
    observer is reused — the fallback must not replace it."""
    from pipecat.pipeline.task import PipelineTask

    tracer = _tracer(record_audio=False)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline)
    original_tto = task.turn_tracking_observer
    assert original_tto is not None

    await tracer.register_task_handlers(task)

    assert task.turn_tracking_observer is original_tto


# --------------------------------------------------------------------------- #
# C4 — auto-enable PipelineParams metrics flags                               #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_metrics_flags_auto_enabled_when_unset(make_pipeline: Any) -> None:
    """C4: a customer who never set the metrics flags still gets MetricsFrame —
    token usage / cost / TTFB are no longer silently missing."""
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    tracer = _tracer(record_audio=False, auto_enable_metrics=True)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    params = PipelineParams()
    assert params.enable_metrics is False  # precondition (Pipecat default)
    assert params.enable_usage_metrics is False
    task = PipelineTask(pipeline, params=params)

    await tracer.register_task_handlers(task)

    assert task._params.enable_metrics is True
    assert task._params.enable_usage_metrics is True


@pytest.mark.asyncio
async def test_metrics_flags_not_touched_when_auto_disabled(make_pipeline: Any) -> None:
    """``auto_enable_metrics=False`` leaves the customer's params exactly as-is."""
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    tracer = _tracer(record_audio=False, auto_enable_metrics=False)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline, params=PipelineParams())

    await tracer.register_task_handlers(task)

    assert task._params.enable_metrics is False
    assert task._params.enable_usage_metrics is False


@pytest.mark.asyncio
async def test_metrics_flags_already_enabled_stay_enabled(make_pipeline: Any) -> None:
    """Explicit ``True`` is a no-op — we never flip a customer's enabled flag."""
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    tracer = _tracer(record_audio=False, auto_enable_metrics=True)
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    await tracer.register_task_handlers(task)

    assert task._params.enable_metrics is True
    assert task._params.enable_usage_metrics is True


# --------------------------------------------------------------------------- #
# S1 — pipeline-wrapping single call                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_observe_and_create_task_single_call(make_pipeline: Any) -> None:
    """``observe_and_create_task`` collapses observe_pipeline + PipelineTask +
    register_task_handlers into one call and returns a fully-wired task."""
    from pipecat.pipeline.task import PipelineParams

    tracer = _tracer(record_audio=True, record_raw_input_audio=False)
    pipeline = make_pipeline(2)

    task = await tracer.observe_and_create_task(pipeline, params=PipelineParams())

    # Observer attached to the proxy list…
    assert tracer.observer in task._observer._observers
    # …metrics auto-enabled…
    assert task._params.enable_metrics is True
    assert task._params.enable_usage_metrics is True
    # …and the ABP was auto-inserted (record_audio=True) AND detected by the
    # observer, which is the real end-to-end outcome: full-conversation audio
    # is wired with zero customer pipeline edits.
    abp = tracer.observer._audio_buffer_processor
    assert abp is not None
    assert any(b.__name__ == "AudioBufferProcessor" for b in type(abp).__mro__)
