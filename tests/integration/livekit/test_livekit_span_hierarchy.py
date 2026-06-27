"""
Span hierarchy / parent resolution + generation finalize
(Sections E and G of LIVEKIT_TEST_PLAN.md).

This is the most regression-prone logic in the integration. All tests are
SYNCHRONOUS and call ``create_event_span`` directly against a real trace, then
assert the resulting span tree (names + parent_span_id + status). We never fire
a handler and ``sleep`` — ``asyncio.create_task`` copies contextvars, so context
set in one task is invisible elsewhere; the integration tracks
``manager._last_agent_state_changed_span_id`` precisely for that reason, and so
do we. The ``speech_created`` cases (which DO spawn a background task) are async
and patch the system-prompt updater to keep them deterministic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("livekit.agents")

from noveum_trace.core.context import (  # noqa: E402
    get_current_span,
    set_current_span,
    set_current_trace,
)
from noveum_trace.core.span import SpanStatus  # noqa: E402
from noveum_trace.integrations.livekit.livekit_session import (  # noqa: E402
    _LiveKitTracingManager,
)
from noveum_trace.integrations.livekit.livekit_utils import (  # noqa: E402
    create_event_span,
)

from ._fakes import one_span  # noqa: E402

_SYS_PROMPT_PATCH = (
    "noveum_trace.integrations.livekit.livekit_utils._update_span_with_system_prompt"
)


def _manager():
    return _LiveKitTracingManager(session=SimpleNamespace())


# --------------------------------------------------------------------------- #
# E1/E2 — basic creation + no-trace guard
# --------------------------------------------------------------------------- #
def test_basic_event_creates_finished_span(lk_trace):
    span = create_event_span(
        "user_state_changed",
        SimpleNamespace(old_state="listening", new_state="speaking"),
        manager=_manager(),
    )
    assert span is not None
    captured = one_span(lk_trace, "livekit.user_state_changed")
    assert captured.attributes["event.type"] == "user_state_changed"
    assert "metadata" in captured.attributes
    # the serialized event payload must actually land on the span (not just the
    # post-hoc event.type/metadata keys)
    assert captured.attributes["user_state_changed.new_state"] == "speaking"
    assert captured.attributes["user_state_changed.old_state"] == "listening"
    assert captured.is_finished()


def test_no_active_trace_returns_none_without_creating_span(lk_client):
    """Guards: the no-trace early-out happens BEFORE span creation — not a span
    that gets created then swallowed by the broad except."""
    set_current_trace(None)
    with patch.object(lk_client, "start_span") as start_span:
        result = create_event_span(
            "user_state_changed", SimpleNamespace(), manager=_manager()
        )
    assert result is None
    start_span.assert_not_called()


# --------------------------------------------------------------------------- #
# E3/E4/E5 — metrics_collected parent resolution
# --------------------------------------------------------------------------- #
def test_metrics_collected_parented_to_agent_state(lk_trace):
    """Guards: metrics_collected re-parents under the latest agent_state_changed
    span and is never left as the current span."""
    mgr = _manager()
    create_event_span(
        "agent_state_changed", SimpleNamespace(new_state="thinking"), manager=mgr
    )
    create_event_span("metrics_collected", SimpleNamespace(value=1), manager=mgr)

    agent_state = one_span(lk_trace, "livekit.agent_state_changed")
    metrics = one_span(lk_trace, "livekit.metrics_collected")
    assert metrics.parent_span_id == agent_state.span_id
    current = get_current_span()
    assert current is None or current.name != "livekit.metrics_collected"


def test_metrics_collected_without_agent_state_is_trace_child(lk_trace):
    """Guards: with no agent_state yet, metrics_collected is a direct child of the
    trace (parent_span_id is None), not parented to some stray current span."""
    create_event_span("metrics_collected", SimpleNamespace(value=1), manager=_manager())
    metrics = one_span(lk_trace, "livekit.metrics_collected")
    assert metrics.parent_span_id is None


def test_agent_state_span_id_is_tracked(lk_trace):
    mgr = _manager()
    create_event_span(
        "agent_state_changed", SimpleNamespace(new_state="x"), manager=mgr
    )
    agent_state = one_span(lk_trace, "livekit.agent_state_changed")
    assert mgr._last_agent_state_changed_span_id == agent_state.span_id


# --------------------------------------------------------------------------- #
# E8 — a non-metrics event re-parents away from a current metrics span
# --------------------------------------------------------------------------- #
def test_other_event_reparented_when_current_is_metrics(lk_trace):
    mgr = _manager()
    create_event_span(
        "agent_state_changed", SimpleNamespace(new_state="x"), manager=mgr
    )
    agent_state = one_span(lk_trace, "livekit.agent_state_changed")
    metrics = create_event_span("metrics_collected", SimpleNamespace(), manager=mgr)
    set_current_span(metrics)  # simulate metrics span being current

    create_event_span(
        "user_input_transcribed", SimpleNamespace(transcript="hi"), manager=mgr
    )
    uit = one_span(lk_trace, "livekit.user_input_transcribed")
    assert uit.parent_span_id == agent_state.span_id


# --------------------------------------------------------------------------- #
# E9/E10 — error status
# --------------------------------------------------------------------------- #
def test_error_event_type_sets_error_status(lk_trace):
    create_event_span("error", SimpleNamespace(error="boom"), manager=_manager())
    assert one_span(lk_trace, "livekit.error").status == SpanStatus.ERROR


def test_event_with_truthy_error_attribute_sets_error_status(lk_trace):
    create_event_span("close", SimpleNamespace(error="kaboom"), manager=_manager())
    assert one_span(lk_trace, "livekit.close").status == SpanStatus.ERROR


# --------------------------------------------------------------------------- #
# E6/E7 — speech_created parent resolution (background task patched out)
# --------------------------------------------------------------------------- #
async def test_speech_created_uses_current_span_as_parent(lk_trace, lk_client):
    mgr = _manager()
    parent = lk_client.start_span("parent_span")  # becomes current span
    with patch(_SYS_PROMPT_PATCH, new_callable=AsyncMock):
        create_event_span("speech_created", SimpleNamespace(), manager=mgr)
    sc = one_span(lk_trace, "livekit.speech_created")
    assert sc.parent_span_id == parent.span_id


async def test_speech_created_reparented_when_current_is_metrics(lk_trace):
    mgr = _manager()
    create_event_span(
        "agent_state_changed", SimpleNamespace(new_state="x"), manager=mgr
    )
    agent_state = one_span(lk_trace, "livekit.agent_state_changed")
    metrics = create_event_span("metrics_collected", SimpleNamespace(), manager=mgr)
    set_current_span(metrics)
    with patch(_SYS_PROMPT_PATCH, new_callable=AsyncMock):
        create_event_span("speech_created", SimpleNamespace(), manager=mgr)
    sc = one_span(lk_trace, "livekit.speech_created")
    assert sc.parent_span_id == agent_state.span_id


# --------------------------------------------------------------------------- #
# G1/G2/G3 — generation span finalize with function calls/outputs
# --------------------------------------------------------------------------- #
def test_finalize_generation_span_with_functions(lk_trace, lk_client):
    """Guards: pending generation span absorbs serialized function calls + outputs
    then is finished OK and cleared."""
    mgr = _manager()
    span = lk_client.start_span("livekit.realtime.generation_created", attributes={})
    mgr._pending_generation_span = span

    ev = SimpleNamespace(
        function_calls=[
            SimpleNamespace(name="get_weather", arguments='{"city":"SF"}', call_id="c1")
        ],
        function_call_outputs=[
            SimpleNamespace(name="get_weather", output="sunny", is_error=False)
        ],
    )
    mgr._finalize_generation_span_with_functions(ev)

    assert span.attributes["llm.function_calls.count"] == 1
    assert "get_weather" in span.attributes["llm.function_calls"]
    assert span.attributes["llm.function_outputs.count"] == 1
    outputs = json.loads(span.attributes["llm.function_outputs"])
    assert outputs[0]["output"] == "sunny"
    assert outputs[0]["is_error"] is False
    assert span.status == SpanStatus.OK
    assert span.is_finished()
    assert mgr._pending_generation_span is None


def test_finalize_without_pending_span_is_noop():
    mgr = _manager()
    mgr._pending_generation_span = None
    mgr._finalize_generation_span_with_functions(None)  # must not raise
    assert mgr._pending_generation_span is None


def test_finalize_without_client_clears_pending(lk_trace, lk_client):
    mgr = _manager()
    span = lk_client.start_span("livekit.realtime.generation_created", attributes={})
    mgr._pending_generation_span = span
    with patch("noveum_trace.get_client", return_value=None):
        mgr._finalize_generation_span_with_functions(None)
    assert mgr._pending_generation_span is None
