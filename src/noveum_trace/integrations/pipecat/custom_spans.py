"""
NoveumCustomSpanProcessor — OTEL SpanProcessor that folds customer plain-OTEL
spans into the active Noveum pipecat conversation trace.

Activated by ``NoveumPipecatTracer(capture_custom_spans=True)`` via
``register_custom_span_processor``.  Do NOT import this module at the top level
of any integration file; it is always imported lazily inside function bodies so
that ``opentelemetry-sdk`` remains an optional dependency.

Span hierarchy produced when active::

    Trace: pipecat.conversation
      Span: pipecat.turn
        Span: <customer OTEL span name>   ← folded in here
          Span: <child OTEL span>         ← parent resolved via _map lookup
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OTEL base class — graceful degradation when opentelemetry-sdk is
# not installed (same stub pattern as BaseObserver in pipecat_observer.py).
# ---------------------------------------------------------------------------
try:
    from opentelemetry.sdk.trace import SpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

    class SpanProcessor:  # type: ignore[no-redef]
        """Fallback stub when opentelemetry-sdk is not installed."""

        def on_start(self, span: Any, parent_context: Any = None) -> None:
            pass

        def on_end(self, span: Any) -> None:
            pass

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    logger.debug(
        "opentelemetry-sdk is not importable. "
        "Install with: pip install 'noveum-trace[pipecat-otel]'"
    )

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pipecat's own instrumentation scopes — these are already handled natively by
# NoveumTraceObserver.  Filter them out to prevent double-counting.
_PIPECAT_SCOPES: frozenset[str] = frozenset({"pipecat", "pipecat.turn"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns_to_dt(ns: int) -> Any:
    """Convert an OTEL nanosecond timestamp to a timezone-aware datetime."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class NoveumCustomSpanProcessor(SpanProcessor):
    """
    OTEL ``SpanProcessor`` that re-parents customer spans under the active
    Noveum conversation turn.

    Registered on the global OTEL ``TracerProvider`` by
    ``register_custom_span_processor``.  The processor is ADD-only — it never
    replaces existing processors, so customers can still export to their own
    OTEL backend in parallel.

    Thread-safety: v1 assumes custom spans are created and finished on the
    asyncio event loop thread (same as all Pipecat frame handlers).  No lock
    is used; ``_map`` operations are therefore safe.
    """

    def __init__(self, observer: Any) -> None:
        self._obs = observer
        # Maps OTEL span_id (int) → Noveum span object for all in-flight spans.
        self._map: dict[int, Any] = {}
        # Set by register_custom_span_processor after construction.
        self._owns_provider: bool = False
        self._provider: Any = None

    # ---------------------------------------------------------------------- #
    # SpanProcessor interface                                                 #
    # ---------------------------------------------------------------------- #

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            scope_name = getattr(
                getattr(span, "instrumentation_scope", None), "name", ""
            )
            if scope_name in _PIPECAT_SCOPES:
                return

            if self._obs._trace is None:
                return

            parent_nov = self._resolve_parent(span)
            parent_span_id = parent_nov.span_id if parent_nov is not None else None

            attrs = dict(span.attributes or {})
            start_time = _ns_to_dt(span.start_time) if span.start_time else None

            nov = self._obs._trace.create_span(
                name=span.name,
                parent_span_id=parent_span_id,
                attributes=attrs,
                start_time=start_time,
            )
            if nov is not None:
                self._map[span.context.span_id] = nov
        except Exception:
            logger.debug(
                "NoveumCustomSpanProcessor.on_start failed for span %r",
                getattr(span, "name", "<unknown>"),
                exc_info=True,
            )

    def on_end(self, span: Any) -> None:
        try:
            scope_name = getattr(
                getattr(span, "instrumentation_scope", None), "name", ""
            )
            if scope_name in _PIPECAT_SCOPES:
                return

            span_id = span.context.span_id
            nov = self._map.pop(span_id, None)
            if nov is None or self._obs._trace is None:
                return

            # Flush final attributes (OTEL may add attrs between on_start and on_end)
            if span.attributes:
                nov.set_attributes(dict(span.attributes))

            # Propagate error status
            status = getattr(span, "status", None)
            if status is not None:
                status_code = getattr(status, "status_code", None)
                if (
                    status_code is not None
                    and getattr(status_code, "name", "") == "ERROR"
                ):
                    nov.set_status("error", status.description or "")

            end_time = _ns_to_dt(span.end_time) if span.end_time else None
            self._obs._trace.finish_span(nov.span_id, end_time=end_time)
        except Exception:
            logger.debug(
                "NoveumCustomSpanProcessor.on_end failed for span %r",
                getattr(span, "name", "<unknown>"),
                exc_info=True,
            )

    def shutdown(self) -> None:
        # Provider lifecycle is managed by register_custom_span_processor /
        # _finish_conversation_impl; nothing to do here.
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _resolve_parent(self, span: Any) -> Any:
        """
        Resolve the Noveum parent span for an OTEL span.

        Resolution order:
        1. If the OTEL span has a parent whose span_id is already in ``_map``,
           use that Noveum span (preserves nested OTEL hierarchies).
        2. Fall back to the active ``_current_turn_span`` so all custom spans
           are nested under the current pipecat.turn.
        """
        parent_ctx = span.parent if span.parent is not None else None
        if parent_ctx is not None:
            pctx_id = getattr(parent_ctx, "span_id", None)
            if pctx_id is not None and pctx_id in self._map:
                return self._map[pctx_id]
        return self._obs._current_turn_span


# ---------------------------------------------------------------------------
# Factory / registration
# ---------------------------------------------------------------------------


def register_custom_span_processor(observer: Any) -> NoveumCustomSpanProcessor:
    """
    Create a ``NoveumCustomSpanProcessor``, register it on the global OTEL
    ``TracerProvider``, and wire the backref onto *observer*.

    If no SDK provider is active, a new ``TracerProvider`` is created and set
    as the global provider (``owns_provider=True``).  This is tracked on the
    processor so ``_finish_conversation_impl`` can call ``provider.shutdown()``
    at session end.

    Raises:
        RuntimeError: If ``opentelemetry-sdk`` is not installed.
    """
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError as exc:
        raise RuntimeError(
            "capture_custom_spans=True requires the 'pipecat-otel' extra "
            "(opentelemetry-sdk). Install it with: "
            "pip install 'noveum-trace[pipecat-otel]'"
        ) from exc

    provider = otel_trace.get_tracer_provider()
    owns_provider = False

    if not isinstance(provider, TracerProvider):
        # The global is still the default no-op proxy — create and set a real one.
        provider = TracerProvider()
        otel_trace.set_tracer_provider(provider)
        owns_provider = True

    proc = NoveumCustomSpanProcessor(observer)
    # ADD, never replace — the customer may export to their own backend too.
    provider.add_span_processor(proc)

    proc._owns_provider = owns_provider
    proc._provider = provider

    # Backref so _finish_conversation_impl can drain and shut down.
    observer._custom_span_processor = proc

    logger.debug(
        "NoveumPipecatTracer: registered NoveumCustomSpanProcessor on OTEL provider "
        "(owns_provider=%s)",
        owns_provider,
    )
    return proc
