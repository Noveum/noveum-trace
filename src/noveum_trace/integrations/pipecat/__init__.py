"""
Pipecat integration for Noveum Trace SDK.

Provides automatic tracing for Pipecat pipelines via a BaseObserver subclass
that maps frame-based pipeline events into the Noveum trace/span hierarchy.

The observer always uses the **globally** initialised Noveum client (see
``noveum_trace.init`` or ``noveum_trace.get_client``). It does not create or
hold a per-observer client.

Usage::

    import noveum_trace
    from noveum_trace.integrations.pipecat import NoveumTraceObserver

    noveum_trace.init(api_key="...", project="my-bot")

    obs = NoveumTraceObserver()
    task = PipelineTask(pipeline, observers=[obs])
    obs.attach_to_task(task)

``attach_to_task`` wires ``TurnTrackingObserver`` / ``UserBotLatencyObserver``
from the task so turn spans match Pipecat's turn boundaries.  It also
auto-detects an ``AudioBufferProcessor`` in the pipeline; if one is present
the full stereo conversation WAV (user left, bot right) is captured and
uploaded when the session ends.  If ``record_audio=True`` but no
``AudioBufferProcessor`` is found, a warning is logged.

To enable full-conversation audio, add ``AudioBufferProcessor`` to your
pipeline::

    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

    audio_buffer = AudioBufferProcessor(num_channels=2)
    pipeline = Pipeline([..., audio_buffer, ...])
    task = PipelineTask(pipeline, observers=[obs])
    obs.attach_to_task(task)   # auto-detects audio_buffer

Or use the convenience helper::

    from noveum_trace.integrations.pipecat import setup_pipecat_tracing

    task = PipelineTask(
        pipeline,
        observers=[setup_pipecat_tracing(record_audio=True)],
    )
"""

from typing import Any

from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver


def setup_pipecat_tracing(**kwargs: Any) -> NoveumTraceObserver:
    """
    Convenience factory for NoveumTraceObserver.

    Returns a fully-configured observer ready to be added to PipelineTask.
    Tracing uses the global Noveum client only — call ``noveum_trace.init()``
    (or equivalent) before running the pipeline.

    Args:
        **kwargs: Forwarded to ``NoveumTraceObserver`` (e.g. ``record_audio``,
            ``trace_name_prefix``, ``name=`` for Pipecat's ``BaseObject``).

    Raises:
        TypeError: If ``api_key`` or ``project`` are passed — use
            ``noveum_trace.init()`` instead; the observer uses only
            ``get_client()``.
    """
    for key in ("api_key", "project"):
        if key in kwargs:
            raise TypeError(
                f"setup_pipecat_tracing() does not accept {key!r}. "
                "Use noveum_trace.init(api_key=..., project=...) to configure "
                "the global client; the Pipecat observer always uses get_client()."
            )
    return NoveumTraceObserver(**kwargs)


__all__ = ["NoveumTraceObserver", "setup_pipecat_tracing"]
