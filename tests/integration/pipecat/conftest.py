"""
Shared fixtures for the NoveumPipecatTracer / custom-span test suite.

These tests verify the four-tier Pipecat integration overhaul described in
``Noveum_Pipecat_SDK_Integration_Spec.md`` and ``.cursor/plans/pipecat-plan-*``:

    Tier 0 — NoveumPipecatTracer wiring (C7)        → test_tracer_wiring.py
    Tier 1 — auto-insert AudioBufferProcessor (C1)  → test_tracer_abp_insertion.py
    Tier 2 — transport raw-audio tap (C3/C6)        → test_tracer_raw_audio_tap.py
    Tier 3 — OTEL custom-span capture (C2)          → test_custom_spans.py
    Tier 4 — metrics defaults (C4), session metadata→ test_tracer_wiring.py / test_observer_session_metadata.py

The fixtures deliberately use *real* Pipecat objects (``Pipeline``,
``PipelineTask``, ``AudioBufferProcessor``) and a *real* Noveum ``Trace`` rather
than mocks wherever the assertion is about structure/wiring, so the tests prove
behaviour rather than mirror an implementation detail.
"""

from __future__ import annotations

from typing import Any

import pytest


# --------------------------------------------------------------------------- #
# Pipecat module access (skip cleanly if pipecat-ai is not installed)          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def ff() -> Any:
    """The ``pipecat.frames.frames`` module (skips if pipecat-ai missing)."""
    pytest.importorskip("pipecat.frames.frames")
    from pipecat.frames import frames as _ff

    return _ff


# --------------------------------------------------------------------------- #
# Real Pipecat pipeline / task builders                                        #
# --------------------------------------------------------------------------- #
@pytest.fixture
def passthrough_processors() -> Any:
    """Factory: build ``n`` trivial real ``FrameProcessor`` instances.

    Used to construct genuine ``Pipeline`` objects so ``observe_pipeline``'s
    strip-rebuild path (``_processors[1:-1]``) is exercised against the same
    auto source/sink wrapping Pipecat applies in production.
    """
    pytest.importorskip("pipecat.processors.frame_processor")
    from pipecat.processors.frame_processor import FrameProcessor

    class _Passthrough(FrameProcessor):
        """Minimal real processor — Pipecat links it like any other."""

    def _make(n: int = 2) -> list[Any]:
        return [_Passthrough() for _ in range(n)]

    return _make


@pytest.fixture
def make_pipeline(passthrough_processors: Any) -> Any:
    """Factory: a real ``Pipeline`` of ``n`` passthrough processors."""
    pytest.importorskip("pipecat.pipeline.pipeline")
    from pipecat.pipeline.pipeline import Pipeline

    def _make(n: int = 2) -> Any:
        return Pipeline(passthrough_processors(n))

    return _make


# --------------------------------------------------------------------------- #
# Fake transport whose ``input()`` exposes an async ``push_audio_frame``        #
# --------------------------------------------------------------------------- #
class FakeInputTransport:
    """Stand-in for a Pipecat ``BaseInputTransport`` instance.

    Records every frame forwarded to the *original* ``push_audio_frame`` and
    returns a sentinel so tests can prove the tap preserves the return value.
    """

    def __init__(self, *, raise_on_push: bool = False) -> None:
        self.forwarded: list[Any] = []
        self._raise_on_push = raise_on_push

    async def push_audio_frame(self, frame: Any, *args: Any, **kwargs: Any) -> Any:
        if self._raise_on_push:
            raise RuntimeError("downstream push failure")
        self.forwarded.append(frame)
        return "original-return-value"


class FakeTransport:
    """Minimal transport with a cached ``input()`` (mirrors Pipecat's lazy-init).

    Optional ``room_url`` / ``room_name`` attributes feed the session-metadata
    extraction path in ``_store_transport``.
    """

    def __init__(
        self,
        *,
        room_url: str | None = None,
        room_name: str | None = None,
        no_input: bool = False,
        input_raises: bool = False,
    ) -> None:
        self._input: FakeInputTransport | None = None
        self._no_input = no_input
        self._input_raises = input_raises
        if room_url is not None:
            self.room_url = room_url
        if room_name is not None:
            self.room_name = room_name

    def input(self) -> FakeInputTransport:
        if self._input_raises:
            raise RuntimeError("transport.input() unavailable")
        if self._no_input:
            return None  # type: ignore[return-value]
        if self._input is None:
            self._input = FakeInputTransport()
        return self._input


@pytest.fixture
def fake_transport() -> Any:
    """Factory for :class:`FakeTransport`."""
    return FakeTransport


# --------------------------------------------------------------------------- #
# Real Noveum Trace + active turn span                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def real_trace_with_turn() -> Any:
    """A real ``Trace`` plus an open ``pipecat.turn`` span (the fold target).

    Returns ``(trace, turn_span)``.  Custom-span tests attach this to an
    observer's ``_trace`` / ``_current_turn_span`` so the processor folds
    customer spans under a genuine turn span and we can assert real parenting
    and serialisation.
    """
    from noveum_trace.core.trace import Trace

    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn")
    return trace, turn


# --------------------------------------------------------------------------- #
# OTEL global-provider isolation                                               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def reset_otel_provider() -> Any:
    """Save/restore the process-global OTEL ``TracerProvider``.

    ``opentelemetry.trace.set_tracer_provider`` is *set-once-per-process*: the
    second call no-ops with a warning, gated by a ``Once`` flag.  Without this
    reset, ``register_custom_span_processor``'s "no provider → create & own"
    branch would only ever fire in the first test that ran, making the
    provider-mode tests order-dependent and flaky.  This fixture restores both
    the provider and the ``Once`` flag so every test starts from a clean slate.
    """
    pytest.importorskip("opentelemetry.sdk.trace")
    import opentelemetry.trace as ot
    from opentelemetry.util._once import Once

    saved_provider = ot._TRACER_PROVIDER
    saved_once = ot._TRACER_PROVIDER_SET_ONCE
    ot._TRACER_PROVIDER = None
    ot._TRACER_PROVIDER_SET_ONCE = Once()
    try:
        yield ot
    finally:
        ot._TRACER_PROVIDER = saved_provider
        ot._TRACER_PROVIDER_SET_ONCE = saved_once
