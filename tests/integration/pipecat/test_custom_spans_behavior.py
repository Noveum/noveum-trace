"""
Custom spans subsystem (§I, CS-1..8) — value-asserting regression tests per
``PIPECAT_TEST_PLAN.md``.

Subsystem under test: ``NoveumCustomSpanProcessor`` (an OTEL ``SpanProcessor``)
folds customer plain-OTEL spans into the active Noveum ``pipecat.conversation``
trace, nested under the current ``pipecat.turn``.  These tests drive the REAL
processor (registered on a real owned ``TracerProvider``) with REAL OTEL spans
created via ``get_tracer(...)``, and assert real ``trace.spans`` values: span
names, ``parent_span_id`` parenting, attribute values, ``SpanStatus`` and the
serialised ``to_dict()['status']`` — never "a mock was called".

Filter rule: spans whose instrumentation scope is ``pipecat`` / ``pipecat.turn``
are dropped (the observer emits those natively, so folding would double-count).

Every test runs under :func:`reset_otel_provider` (autouse below) because
``opentelemetry.trace.set_tracer_provider`` is set-once-per-process; without the
reset the "no provider → create & own" branch would only fire in the first test.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _isolate_otel(reset_otel_provider: Any) -> Any:
    """Apply OTEL global-provider isolation to every test in this module."""
    yield reset_otel_provider


# --------------------------------------------------------------------------- #
# Module-local helpers (mirrors test_custom_spans.py; conftest is off-limits) #
# --------------------------------------------------------------------------- #
def _observer(**kwargs: Any) -> Any:
    pytest.importorskip("pipecat.observers.base_observer")
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    return NoveumTraceObserver(record_audio=False, **kwargs)


def _register(observer: Any) -> Any:
    from noveum_trace.integrations.pipecat.custom_spans import (
        register_custom_span_processor,
    )

    return register_custom_span_processor(observer)


def _tracer(name: str = "customer.module") -> Any:
    from opentelemetry import trace as otel_trace

    return otel_trace.get_tracer(name)


# --------------------------------------------------------------------------- #
# CS-1 — filter proven against REAL pipecat-scoped OTEL spans (on_start+on_end)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scope", ["pipecat", "pipecat.turn"])
def test_real_pipecat_scoped_span_filtered_through_on_start_and_on_end(
    real_trace_with_turn: Any, scope: str
) -> None:
    # Guards: a _PIPECAT_SCOPES / scope-comparison change that silently double-
    # counts pipecat-native spans (filter must hold on on_start AND on_end).
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    proc = _register(obs)

    before = len(trace.spans)
    tr = _tracer(scope)
    # start_as_current_span drives the registered processor's on_start AND
    # on_end via a REAL pipecat-scoped tracer (scope name == `scope`).
    with tr.start_as_current_span("pipecat.native") as span:
        otel_span_id = span.get_span_context().span_id
        # mid-span: on_start already ran and must have dropped this scope.
        assert otel_span_id not in proc._map

    # No Noveum span produced for the filtered scope…
    assert len(trace.spans) == before
    # …and on_end did not leave the id lingering in the map either.
    assert otel_span_id not in proc._map
    assert proc._map == {}


# --------------------------------------------------------------------------- #
# CS-2 — customer span whose OTEL parent is a SKIPPED pipecat span folds under
#         the active turn (fallback fires because parent not in _map)         #
# --------------------------------------------------------------------------- #
def test_custom_span_under_skipped_pipecat_parent_folds_to_turn(
    real_trace_with_turn: Any,
) -> None:
    # Guards: _resolve_parent's fallback-when-parent-not-in-map path (the OTEL
    # parent was a filtered pipecat span, so it never entered _map).
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    _register(obs)

    pipecat_tr = _tracer("pipecat")
    customer_tr = _tracer("customer.module")
    with pipecat_tr.start_as_current_span("pipecat.llm"):
        with customer_tr.start_as_current_span("customer.work"):
            pass

    by_name = {s.name: s for s in trace.spans}
    # The pipecat-scoped parent is filtered — no Noveum span for it.
    assert "pipecat.llm" not in by_name
    # The customer child folds under the active turn via the fallback.
    customer = by_name["customer.work"]
    assert customer.parent_span_id == turn.span_id


# --------------------------------------------------------------------------- #
# CS-3 — non-error customer span keeps OK status (UNSET/OK NOT flipped to ERROR)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("set_explicit_ok", [False, True])
def test_non_error_span_keeps_ok_status_and_serialises(
    real_trace_with_turn: Any, set_explicit_ok: bool
) -> None:
    # Guards: an on_end refactor that mis-maps UNSET/OK to ERROR (false errors
    # on every custom span). The existing error test only covers the ERROR dir.
    from opentelemetry.trace import Status, StatusCode

    from noveum_trace.core.span import SpanStatus

    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    _register(obs)

    name = "ok_span" if set_explicit_ok else "unset_span"
    tr = _tracer()
    with tr.start_as_current_span(name) as span:
        if set_explicit_ok:
            span.set_status(Status(StatusCode.OK))

    nov = next(s for s in trace.spans if s.name == name)
    # Span.finish() promotes UNSET→OK; the ERROR branch must NOT have fired.
    assert nov.status == SpanStatus.OK
    assert nov.to_dict()["status"] == SpanStatus.OK.value  # serialises as "ok"


# --------------------------------------------------------------------------- #
# CS-4 — span with attributes == None is captured without raising             #
# --------------------------------------------------------------------------- #
def test_none_attributes_span_is_captured_and_finished(
    real_trace_with_turn: Any,
) -> None:
    # Guards: dropping the `or {}` / `if span.attributes` guards so a None-attrs
    # span raises inside the swallowed callback (silent loss of the span).
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    proc = _register(obs)

    # A live OTEL span always coerces attributes to a dict; the None case can
    # only arrive via a ReadableSpan-shaped object, so drive the handler direct.
    fake = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="customer.module"),
        name="none_attrs_op",
        attributes=None,
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        parent=None,
        status=None,
        context=SimpleNamespace(span_id=4040),
    )
    proc.on_start(fake)  # must not raise on None attributes
    proc.on_end(fake)

    nov = next(s for s in trace.spans if s.name == "none_attrs_op")
    assert nov.parent_span_id == turn.span_id
    assert nov.is_finished()


# --------------------------------------------------------------------------- #
# CS-5 — on_end for an unknown span_id is a clean no-op                        #
# --------------------------------------------------------------------------- #
def test_on_end_unknown_span_id_is_noop(real_trace_with_turn: Any) -> None:
    # Guards: an on_end that assumes the span is always in _map
    # (KeyError/AttributeError on a None pop result).
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    proc = _register(obs)

    spans_before = list(trace.spans)
    map_before = dict(proc._map)

    fake = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="customer.module"),
        name="never_started",
        attributes={"x": 1},
        start_time=1_000,
        end_time=2_000,
        parent=None,
        status=None,
        context=SimpleNamespace(span_id=987654),  # never on_start'd
    )
    proc.on_end(fake)  # must not raise

    assert trace.spans == spans_before  # nothing added or finished
    assert proc._map == map_before  # map untouched


# --------------------------------------------------------------------------- #
# CS-6 — custom span emitted while _current_turn_span is None folds at root    #
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="Issue 1/2 orphan family: a custom span emitted between turns "
    "(_current_turn_span is None) parents at trace root; it SHOULD fall back "
    "to the last turn. Flips to xpass when the _resolve_parent _last_turn_span "
    "fallback lands — see PIPECAT_SPAN_HIERARCHY_ISSUES.md.",
)
def test_custom_span_between_turns_should_fall_back_to_a_turn(
    real_trace_with_turn: Any,
) -> None:
    # Guards: the documented orphan edge for custom spans — desired behavior is
    # that the span is parented (not a trace-root orphan). Currently orphaned.
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace = trace
    obs._current_turn_span = None  # between-turns gap
    _register(obs)

    tr = _tracer()
    with tr.start_as_current_span("between_turns_op"):
        pass

    nov = next(s for s in trace.spans if s.name == "between_turns_op")
    # DESIRED: parented under a turn. Currently None (orphan) → xfail now.
    assert nov.parent_span_id is not None


def test_custom_span_between_turns_is_still_captured(
    real_trace_with_turn: Any,
) -> None:
    # Guards (non-xfail companion to CS-6): even with no active turn the custom
    # span is still created, named, attribute-captured and finished. Asserts the
    # genuinely-correct current facts WITHOUT touching parenting.
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace = trace
    obs._current_turn_span = None
    _register(obs)

    tr = _tracer()
    with tr.start_as_current_span("between_turns_op2") as span:
        span.set_attribute("phase", "teardown")

    nov = next(s for s in trace.spans if s.name == "between_turns_op2")
    assert nov.attributes.get("phase") == "teardown"
    assert nov.is_finished()


# --------------------------------------------------------------------------- #
# CS-7 — create_span returning None does not poison _map (no on_end KeyError)  #
# --------------------------------------------------------------------------- #
def test_create_span_returning_none_does_not_poison_map(
    real_trace_with_turn: Any,
) -> None:
    # Guards: removing the `if nov is not None` guard so a None Noveum span is
    # stored in _map and crashes on_end. (The error test patches create_span to
    # RAISE — a distinct branch.)
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    proc = _register(obs)

    fake = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="customer.module"),
        name="will_be_none",
        attributes={"a": 1},
        start_time=1_000,
        end_time=2_000,
        parent=None,
        status=None,
        context=SimpleNamespace(span_id=5550),
    )
    with patch.object(trace, "create_span", return_value=None):
        proc.on_start(fake)
        # None result must NOT be stored under the span_id.
        assert 5550 not in proc._map
        proc.on_end(fake)  # must no-op cleanly, not KeyError

    assert proc._map == {}


# --------------------------------------------------------------------------- #
# CS-8 — late attributes survive even when on_start saw none (None→populated)  #
# --------------------------------------------------------------------------- #
def test_late_attributes_from_zero_are_all_flushed(
    real_trace_with_turn: Any,
) -> None:
    # Guards: an on_end change that only flushes when on_start already captured
    # attributes — the span here starts with ZERO attrs, all are set after.
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    _register(obs)

    tr = _tracer()
    span = tr.start_span("late_only")  # on_start: zero attributes
    span.set_attribute("a", 1)
    span.set_attribute("b", "two")
    span.set_attribute("c", True)
    span.end()  # on_end: all late attrs must flush

    nov = next(s for s in trace.spans if s.name == "late_only")
    assert nov.attributes.get("a") == 1
    assert nov.attributes.get("b") == "two"
    assert nov.attributes.get("c") is True
