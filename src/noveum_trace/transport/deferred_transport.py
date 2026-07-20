"""
DeferredTransport — a transport that captures the finished trace instead of
sending it.

Used by host integrations (e.g. dograh) whose live call path must never touch
the network: the observer runs its normal client flow (``start_trace`` …
``finish_trace``), but the export lands here as an in-memory capture. The host
snapshots the captured trace (``Trace.to_dict()``), persists it, and exports it
later from another process via ``NoveumClient.send_trace_dict`` with real
credentials.

Unlike ``HttpTransport``, constructing this class has **no side effects**: no
HTTP session, no ``BatchProcessor`` background thread, nothing registered with
``atexit`` (``NoveumClient`` skips its atexit hook for injected transports).
That makes it safe to build one per call in a many-calls-per-process host.

Contract: ``captured_trace`` holds the most recently exported ``Trace`` (the
conversation trace, after ``finish_trace``). ``NoveumTraceObserver.
build_payload_snapshot()`` reads it by attribute lookup.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DeferredTransport:
    """Capture-only transport: exports become in-memory retention, everything
    else is a no-op. Implements the subset of the transport interface that
    ``NoveumClient`` and the pipecat observer touch."""

    def __init__(self) -> None:
        #: Most recently exported trace (the contract attribute — see module
        #: docstring). Overwritten if the same client finishes another trace.
        self.captured_trace: Any = None

    def export_trace(self, trace: Any) -> None:
        """Capture the finished trace instead of sending it."""
        if getattr(trace, "_noop", False):
            return
        self.captured_trace = trace
        logger.debug(
            "DeferredTransport captured trace %s",
            getattr(trace, "trace_id", "<unknown>"),
        )

    def export_audio(self, *args: Any, **kwargs: Any) -> None:
        """
        Drop audio, loudly. Deferred hosts route segment audio through the
        observer's ``audio_sink`` (the observer's constructor guard enforces
        this when ``record_audio=True``), so reaching here means audio was
        produced with nowhere durable to go.
        """
        logger.warning(
            "DeferredTransport received audio %s but has nowhere to store it — "
            "configure an audio_sink on the observer. Audio dropped.",
            kwargs.get("audio_uuid", "<unknown>"),
        )

    def export_image(self, *args: Any, **kwargs: Any) -> None:
        """Drop images (no deferred image path is defined yet)."""
        logger.warning("DeferredTransport dropping image export")

    def flush(self, timeout: Optional[float] = None) -> None:
        """No queue to drain."""

    def shutdown(self) -> None:
        """No thread or session to stop."""
