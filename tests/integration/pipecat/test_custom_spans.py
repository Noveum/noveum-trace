"""
Tier 3 (C2) — OTEL custom-span capture (``NoveumCustomSpanProcessor``).

Spec mapping (``.cursor/plans/pipecat-plan-tier-3-otel-custom-spans.md``,
``Noveum_Pipecat_SDK_Integration_Spec.md`` C2):

The customer writes **plain OTEL** anywhere in their bot; the processor folds
those spans into the active Noveum conversation trace, nested under the current
``pipecat.turn``, replaying the OTEL span's real start/end times and attributes.
No Noveum span API in customer code, no manual ``STTCustomSpanProcessor``.

Filter rule (plan §2): spans whose instrumentation scope is ``pipecat`` /
``pipecat.turn`` are dropped — the observer already emits those natively, so
keeping them would double-count.

Every test runs under :func:`reset_otel_provider` (autouse below) because
``opentelemetry.trace.set_tracer_provider`` is set-once-per-process; without the
reset the provider-mode tests would be order-dependent.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _isolate_otel(reset_otel_provider: Any) -> Any:
    """Apply OTEL global-provider isolation to every test in this module."""
    yield reset_otel_provider


def _observer(**kwargs: Any) -> Any:
    pytest.importorskip("pipecat.observers.base_observer")
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    return NoveumTraceObserver(record_audio=False, **kwargs)


def _register(observer: Any) -> Any:
    from noveum_trace.integrations.pipecat.custom_spans import (
        register_custom_span_processor,
    )

    return register_custom_span_processor(observer)


def _customer_tracer() -> Any:
    from opentelemetry import trace as otel_trace

    return otel_trace.get_tracer("customer.module")


# --------------------------------------------------------------------------- #
# Capture + nesting under the active turn (replayed timing/attrs)             #
# --------------------------------------------------------------------------- #
def test_custom_span_folds_under_active_turn(real_trace_with_turn: Any) -> None:
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace = trace
    obs._current_turn_span = turn
    _register(obs)

    tr = _customer_tracer()
    with tr.start_as_current_span("lookup_order") as span:
        span.set_attribute("order.id", "abc123")

    by_name = {s.name: s for s in trace.spans}
    assert "lookup_order" in by_name
    nov = by_name["lookup_order"]
    # Nested under the active turn…
    assert nov.parent_span_id == turn.span_id
    # …attributes captured…
    assert nov.attributes.get("order.id") == "abc123"
    # …and the span is finished with a real (replayed) end time.
    assert nov.is_finished()


def test_custom_span_replays_real_timing(real_trace_with_turn: Any) -> None:
    """The whole point of Tier 3 over the old workaround: the Noveum span carries
    the OTEL span's REAL start/end timing, not a 'now' stamp. Pin the exact
    replayed timestamps so a 'stamp now' implementation would fail."""
    from noveum_trace.integrations.pipecat.custom_spans import _ns_to_dt

    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    _register(obs)

    tr = _customer_tracer()
    span = tr.start_span("work")  # on_start → Noveum span created
    otel_start_ns = span.start_time
    span.end()  # on_end → Noveum span finished
    otel_end_ns = span.end_time

    nov = next(s for s in trace.spans if s.name == "work")
    # Exact replay of the OTEL span's nanosecond timestamps (not "now").
    assert nov.start_time == _ns_to_dt(otel_start_ns)
    assert nov.end_time == _ns_to_dt(otel_end_ns)


def test_on_end_flushes_final_attributes(real_trace_with_turn: Any) -> None:
    """OTEL may add attributes between on_start and on_end (plan §1 'final
    attrs'); on_end must flush them onto the Noveum span."""
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    _register(obs)

    tr = _customer_tracer()
    span = tr.start_span("op")
    span.set_attribute("early", 1)  # present at on_start
    span.set_attribute("late", 2)  # added after on_start, before on_end
    span.end()

    nov = next(s for s in trace.spans if s.name == "op")
    assert nov.attributes.get("early") == 1
    assert nov.attributes.get("late") == 2  # flushed at on_end


def test_nested_custom_spans_mirror_their_own_hierarchy(
    real_trace_with_turn: Any,
) -> None:
    """A customer's *own* nesting (validate → db_lookup) is preserved via the
    ``_map``, not flattened under the turn."""
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    _register(obs)

    tr = _customer_tracer()
    with tr.start_as_current_span("validate_order"):
        with tr.start_as_current_span("db_lookup"):
            pass

    by_name = {s.name: s for s in trace.spans}
    validate, db = by_name["validate_order"], by_name["db_lookup"]
    assert validate.parent_span_id == turn.span_id  # top under turn
    assert db.parent_span_id == validate.span_id  # child under its own parent


# --------------------------------------------------------------------------- #
# Filter — pipecat / pipecat.turn scopes are dropped                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scope", ["pipecat", "pipecat.turn"])
def test_pipecat_scoped_spans_are_ignored(
    real_trace_with_turn: Any, scope: str
) -> None:
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    proc = _register(obs)

    before = len(trace.spans)
    fake = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name=scope),
        name="pipecat.llm",
        attributes={},
        start_time=1_000,
        parent=None,
        context=SimpleNamespace(span_id=42),
    )
    proc.on_start(fake)

    assert len(trace.spans) == before  # nothing added
    assert 42 not in proc._map


# --------------------------------------------------------------------------- #
# No active conversation — clean no-op                                        #
# --------------------------------------------------------------------------- #
def test_on_start_noop_when_no_active_trace() -> None:
    obs = _observer()
    obs._trace = None  # no conversation yet
    proc = _register(obs)

    fake = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="customer"),
        name="orphan",
        attributes={},
        start_time=1_000,
        parent=None,
        context=SimpleNamespace(span_id=7),
    )
    proc.on_start(fake)
    proc.on_end(fake)  # must also no-op cleanly
    assert proc._map == {}


def test_on_start_swallows_internal_errors() -> None:
    """A malformed span must never raise out of the OTEL callback."""
    obs = _observer()
    obs._trace = MagicMock()
    obs._trace.create_span.side_effect = RuntimeError("boom")
    proc = _register(obs)

    fake = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="customer"),
        name="x",
        attributes={},
        start_time=1_000,
        parent=None,
        context=SimpleNamespace(span_id=1),
    )
    proc.on_start(fake)  # must not raise
    assert proc._map == {}


# --------------------------------------------------------------------------- #
# Provider modes — create+own vs add-to-existing                              #
# --------------------------------------------------------------------------- #
def test_creates_and_owns_provider_when_none_set(reset_otel_provider: Any) -> None:
    from opentelemetry.sdk.trace import TracerProvider

    obs = _observer()
    proc = _register(obs)

    assert proc._owns_provider is True
    assert isinstance(reset_otel_provider.get_tracer_provider(), TracerProvider)
    assert obs._custom_span_processor is proc  # backref wired for teardown


def test_adds_to_existing_provider_without_owning(reset_otel_provider: Any) -> None:
    from opentelemetry.sdk.trace import TracerProvider

    preset = TracerProvider()
    reset_otel_provider.set_tracer_provider(preset)

    obs = _observer()
    proc = _register(obs)

    assert proc._owns_provider is False  # we did not create it
    assert proc._provider is preset  # added to the customer's provider
    assert reset_otel_provider.get_tracer_provider() is preset  # not replaced


def test_register_raises_without_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear actionable error when the ``pipecat-otel`` extra is missing."""
    import builtins

    from noveum_trace.integrations.pipecat.custom_spans import (
        register_custom_span_processor,
    )

    real_import = builtins.__import__

    def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("opentelemetry"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    with pytest.raises(RuntimeError, match="pipecat-otel"):
        register_custom_span_processor(_observer())


# --------------------------------------------------------------------------- #
# Drain at conversation end + provider shutdown only if owned                 #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_open_span_drained_at_conversation_end(real_trace_with_turn: Any) -> None:
    """A custom span still open when the conversation ends is finished during
    ``_finish_conversation`` (no leak), and ``_map`` is emptied."""
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    proc = _register(obs)

    tr = _customer_tracer()
    leaked = tr.start_span("long_running")  # started, never ended
    assert len(proc._map) == 1

    client = MagicMock()
    with patch.object(obs, "_get_client", return_value=client):
        await obs._finish_conversation()

    assert proc._map == {}  # drained
    matches = [s for s in trace.spans if s.name == "long_running"]
    assert len(matches) == 1 and matches[0].is_finished()
    client.finish_trace.assert_called_once()
    leaked.end()  # tidy up the OTEL side


@pytest.mark.asyncio
async def test_owned_provider_is_shut_down_on_finish(real_trace_with_turn: Any) -> None:
    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    proc = _register(obs)
    assert proc._owns_provider is True
    proc._provider = MagicMock()  # spy shutdown

    with patch.object(obs, "_get_client", return_value=MagicMock()):
        await obs._finish_conversation()

    proc._provider.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_unowned_provider_not_shut_down_on_finish(
    real_trace_with_turn: Any, reset_otel_provider: Any
) -> None:
    from opentelemetry.sdk.trace import TracerProvider

    preset = TracerProvider()
    reset_otel_provider.set_tracer_provider(preset)
    preset.shutdown = MagicMock()  # spy

    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    proc = _register(obs)
    assert proc._owns_provider is False

    with patch.object(obs, "_get_client", return_value=MagicMock()):
        await obs._finish_conversation()

    preset.shutdown.assert_not_called()  # we never created it → never shut it down


# --------------------------------------------------------------------------- #
# Error-status propagation (SOURCE BUG — see reason)                          #
# --------------------------------------------------------------------------- #
def test_errored_custom_span_is_recorded_and_serialisable(
    real_trace_with_turn: Any,
) -> None:
    # Regression: an errored customer OTEL span is recorded as an errored Noveum
    # span via SpanStatus.ERROR, so the trace still serialises on export
    # (custom_spans must not pass a bare "error" string to Span.set_status).
    from opentelemetry.trace import Status, StatusCode

    from noveum_trace.core.span import SpanStatus

    trace, turn = real_trace_with_turn
    obs = _observer()
    obs._trace, obs._current_turn_span = trace, turn
    _register(obs)

    tr = _customer_tracer()
    with tr.start_as_current_span("risky") as span:
        span.set_status(Status(StatusCode.ERROR, "boom"))

    nov = next(s for s in trace.spans if s.name == "risky")
    assert nov.status == SpanStatus.ERROR  # recorded as a real error status
    # The whole trace gets serialised on export; an errored customer span must
    # not break that (Span.to_dict() does status.value — requires the enum).
    payload = nov.to_dict()
    assert payload["status"] == SpanStatus.ERROR.value
