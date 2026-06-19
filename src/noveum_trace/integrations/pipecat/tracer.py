"""
NoveumPipecatTracer — cekura-shaped two-call wrapper around NoveumTraceObserver.

``observe_pipeline`` may auto-insert ``AudioBufferProcessor`` and register an OTEL
``SpanProcessor`` for custom spans.  ``register_task_handlers`` adds the observer
to the task, awaits ``attach_to_task``, taps transport for pre-filter raw audio,
auto-enables metrics flags, and stamps session metadata onto the root trace.

Usage::

    import noveum_trace
    from noveum_trace.integrations.pipecat import NoveumPipecatTracer

    noveum_trace.init(api_key="...", project="my-bot")

    tracer = NoveumPipecatTracer(
        record_audio=True,
        record_raw_input_audio=True,
        capture_custom_spans=False,
        auto_enable_metrics=True,
        capture_errors=True,
        capture_system_logs=False,
        capture_session_metadata=True,
    )

    pipeline = tracer.observe_pipeline(pipeline)
    task = PipelineTask(pipeline, params=PipelineParams(...))
    task = await tracer.register_task_handlers(task, transport=transport)
    await runner.run(task)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class NoveumPipecatTracer:
    """
    Pipecat tracer object for Noveum Trace.

    Replaces the old seven-step manual wiring (transport class-swap,
    ``observers=[obs]``, ``attach_to_task``, manual ``AudioBufferProcessor``,
    manual ``STTCustomSpanProcessor``) with two stable calls::

        pipeline = tracer.observe_pipeline(pipeline)
        task = await tracer.register_task_handlers(task, transport=transport)

    The two call sites stay stable; additional capability is added inside these
    two methods.

    The underlying ``NoveumTraceObserver`` is accessible as ``tracer.observer``
    for advanced use.
    """

    def __init__(
        self,
        *,
        record_audio: bool = True,
        record_raw_input_audio: bool = True,
        capture_custom_spans: bool = False,
        auto_enable_metrics: bool = True,
        capture_errors: bool = True,
        capture_system_logs: bool = False,
        capture_session_metadata: bool = True,
        **observer_kwargs: Any,
    ) -> None:
        """
        Args:
            record_audio: Buffer STT/TTS/conversation audio and upload a WAV
                file per span to ``/v1/audio``.
            record_raw_input_audio: Tap the transport's
                ``push_audio_frame`` for pre-filter raw input audio, uploading
                it as ``stt.raw_audio_uuid``.  The flag is forwarded to
                ``NoveumTraceObserver`` so it is available when the transport
                tap is wired.
            capture_custom_spans: Register an OTEL ``SpanProcessor``
                in ``observe_pipeline`` that folds customer plain-OTEL spans
                into the active Noveum conversation trace, nested under the
                active turn.  Requires the ``pipecat-otel`` extra.
            auto_enable_metrics: Automatically set
                ``PipelineParams.enable_metrics=True`` and
                ``enable_usage_metrics=True`` on the task in
                ``register_task_handlers`` if they are not already enabled.
                Ensures ``MetricsFrame`` is emitted without requiring the caller
                to remember to set these flags.  Default ``True``.
            capture_errors: Record ``ErrorFrame`` / ``FatalErrorFrame``
                as span errors and trace events.  Default ``True``.
            capture_system_logs: Record ``SystemLogFrame`` entries at
                warning/error/critical level as span events.  Default ``False``
                (opt-in — volume can be high).
            capture_session_metadata: Stamp transport and runner
                metadata (room URL, transport type, idle timeout, etc.) onto
                the root conversation trace at connection time.  Default ``True``.
            **observer_kwargs: Forwarded verbatim to ``NoveumTraceObserver``
                (e.g. ``trace_name_prefix``, ``capture_function_calls``,
                ``turn_end_timeout_secs``, ``name=``).
        """
        from noveum_trace.integrations.pipecat.pipecat_observer import (
            NoveumTraceObserver,
        )

        self._capture_custom_spans = capture_custom_spans
        self._auto_enable_metrics = auto_enable_metrics
        self._capture_session_metadata = capture_session_metadata
        self._span_processor: Any = None

        self.observer: NoveumTraceObserver = NoveumTraceObserver(
            record_audio=record_audio,
            record_raw_input_audio=record_raw_input_audio,
            auto_enable_metrics=auto_enable_metrics,
            capture_errors=capture_errors,
            capture_system_logs=capture_system_logs,
            capture_session_metadata=capture_session_metadata,
            **observer_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def observe_pipeline(self, pipeline: Any) -> Any:
        """
        Instrument a Pipecat ``Pipeline`` for Noveum tracing.

        If ``record_audio=True`` and no ``AudioBufferProcessor`` is already
        present, appends one at the tail and returns a new rebuilt
        ``Pipeline``.  The original pipeline is discarded — callers **must**
        use the return value.

        Callers **must** use the return value, not the original reference,
        because this method may return a new ``Pipeline`` instance.

        When ``capture_custom_spans`` is ``True``, also registers an OTEL
        ``SpanProcessor`` here.

        Args:
            pipeline: A constructed ``pipecat.pipeline.pipeline.Pipeline``.

        Returns:
            The (possibly modified) pipeline.
        """
        # --- auto-insert AudioBufferProcessor at pipeline tail ---
        if self.observer._record_audio:
            # pipeline._processors includes Pipecat's auto-prepended PipelineSource
            # at index 0 and auto-appended PipelineSink at index -1.  Strip them so
            # we only operate on the customer's processors; Pipeline(inner) re-adds them.
            inner = list(pipeline._processors[1:-1])

            # Same MRO check used by _attach_audio_buffer_from_pipeline in pipecat_observer.py
            has_abp = any(
                any(base.__name__ == "AudioBufferProcessor" for base in type(p).__mro__)
                for p in inner
            )

            if not has_abp:
                try:
                    from pipecat.processors.audio.audio_buffer_processor import (
                        AudioBufferProcessor,
                    )
                except ImportError:
                    logger.warning(
                        "Could not import AudioBufferProcessor — full-conversation audio "
                        "will not be recorded. Ensure pipecat-ai is installed."
                    )
                else:
                    inner.append(AudioBufferProcessor(num_channels=2))

                    try:
                        from pipecat.pipeline.pipeline import Pipeline as _Pipeline
                    except ImportError as exc:
                        raise ImportError(
                            "pipecat-ai is required for NoveumPipecatTracer. "
                            "Install with: pip install 'noveum-trace[pipecat]'"
                        ) from exc

                    logger.debug(
                        "NoveumPipecatTracer: auto-inserted AudioBufferProcessor at pipeline tail"
                    )
                    pipeline = _Pipeline(inner)

        # --- register OTEL SpanProcessor for custom spans ---
        if self._capture_custom_spans:
            try:
                from noveum_trace.integrations.pipecat.custom_spans import (
                    register_custom_span_processor,
                )

                self._span_processor = register_custom_span_processor(self.observer)
            except Exception:
                logger.warning(
                    "NoveumPipecatTracer: failed to register custom span processor — "
                    "capture_custom_spans will be inactive for this session",
                    exc_info=True,
                )

        return pipeline

    async def register_task_handlers(
        self,
        task: Any,
        *,
        transport: Optional[Any] = None,
        runner_args: Optional[Any] = None,
    ) -> Any:
        """
        Add the Noveum observer to *task* and wire all lifecycle hooks.

        Calls ``task.add_observer(self.observer)`` — removing the need for
        the customer to pass ``observers=[...]`` to ``PipelineTask``.

        ``enable_turn_tracking`` defaults to ``True`` in Pipecat
        (``task.py:233``), so ``task.turn_tracking_observer`` is always
        populated.  As a safety net this method creates and attaches a
        ``TurnTrackingObserver`` itself if the attribute is absent (i.e. the
        customer explicitly passed ``enable_turn_tracking=False``).

        Then awaits ``observer.attach_to_task(task)`` which wires the
        ``TurnTrackingObserver`` / ``UserBotLatencyObserver`` from the task,
        auto-detects an ``AudioBufferProcessor`` for full-conversation audio,
        and registers the ``on_pipeline_finished`` safety-net handler.

        Monkeypatches ``transport.input().push_audio_frame`` for pre-filter
        raw audio capture, eliminating the transport class-swap entirely.

        Patches ``task._params.enable_metrics`` and ``enable_usage_metrics`` to
        ``True`` when they are not already set, ensuring ``MetricsFrame`` is
        always emitted.

        Passes the transport and runner_args to the observer so that session
        metadata (room URL, transport type, idle timeout, …) can be stamped
        onto the root trace at first client connection.

        Args:
            task: A ``PipelineTask`` instance constructed by the caller.
            transport: The Pipecat transport for this session.  Used for raw
                audio tapping and session metadata extraction.
            runner_args: Optional ``pipecat.runner.types.RunnerArguments``.
                When provided, room URL and pipeline idle timeout are extracted
                and stamped onto the root trace.

        Returns:
            The same *task* (enables ``task = await tracer.register_task_handlers(...)``).
        """
        # task.add_observer() exists at pipecat task.py:510 and delegates to
        # the WorkerObserver proxy.  Adding post-construction (before
        # runner.run()) is safe — the proxy iterates its list dynamically per
        # frame.  This replaces the customer's observers=[self.observer] kwarg.
        if hasattr(task, "add_observer"):
            task.add_observer(self.observer)
        else:
            logger.warning(
                "PipelineTask does not expose add_observer(); "
                "the Noveum observer may not receive frames. "
                "As a fallback, pass observers=[tracer.observer] to PipelineTask."
            )

        # enable_turn_tracking defaults to True in Pipecat (task.py:233), so
        # task.turn_tracking_observer is normally already set by the time we
        # get here.  This block is the edge-case fallback for a customer who
        # explicitly passes enable_turn_tracking=False.
        if getattr(task, "turn_tracking_observer", None) is None:
            try:
                from pipecat.observers.turn_tracking_observer import (
                    TurnTrackingObserver,
                )

                tto = TurnTrackingObserver()
                task.add_observer(tto)
                # Set the attribute so attach_to_task can find it via
                # getattr(task, "turn_tracking_observer", None).
                task._turn_tracking_observer = tto
                logger.debug(
                    "NoveumPipecatTracer: created fallback TurnTrackingObserver "
                    "(task was constructed with enable_turn_tracking=False)"
                )
            except Exception:
                logger.debug(
                    "Could not create fallback TurnTrackingObserver; "
                    "falling back to standalone turn detection (turn_end_timeout_secs)",
                    exc_info=True,
                )

        # --- auto-enable PipelineParams metrics flags ---
        # Ensures MetricsFrame (token usage, TTFB, processing latency) is emitted
        # even if the caller forgot to set enable_metrics / enable_usage_metrics on
        # PipelineParams.  Silently skipped if params are unavailable or read-only.
        if self._auto_enable_metrics:
            try:
                params = getattr(task, "_params", None)
                if params is not None:
                    patched: list[str] = []
                    if not getattr(params, "enable_metrics", True):
                        params.enable_metrics = True
                        patched.append("enable_metrics")
                    if not getattr(params, "enable_usage_metrics", True):
                        params.enable_usage_metrics = True
                        patched.append("enable_usage_metrics")
                    if patched:
                        logger.debug(
                            "NoveumPipecatTracer: auto-enabled PipelineParams: %s",
                            patched,
                        )
            except Exception:
                logger.debug(
                    "NoveumPipecatTracer: could not patch PipelineParams for metrics "
                    "— MetricsFrame may not be emitted if flags were not set",
                    exc_info=True,
                )

        # --- store transport + runner_args for session metadata ---
        if self._capture_session_metadata and (
            transport is not None or runner_args is not None
        ):
            try:
                self.observer._store_transport(transport, runner_args=runner_args)
            except Exception:
                logger.debug(
                    "NoveumPipecatTracer: failed to store transport metadata — "
                    "session attributes will not be stamped on the trace",
                    exc_info=True,
                )

        # Tap transport.input().push_audio_frame for pre-filter raw audio.
        if (
            transport is not None
            and self.observer._record_raw_input_audio
            and self.observer._record_audio
        ):
            try:
                try:
                    input_transport = transport.input()
                except Exception:
                    logger.debug(
                        "NoveumPipecatTracer: transport.input() unavailable — "
                        "raw audio tap skipped",
                        exc_info=True,
                    )
                    input_transport = None

                if input_transport is not None and getattr(
                    input_transport, "_noveum_tap_applied", False
                ):
                    # Idempotency guard: never wrap an already-tapped instance.
                    # Re-tapping (observer reuse, retry, a second
                    # register_task_handlers) would stack wrappers and capture
                    # each frame once per layer.
                    logger.debug(
                        "NoveumPipecatTracer: raw audio tap already applied on "
                        "input transport — skipping (idempotent)"
                    )
                elif input_transport is not None:
                    _original = getattr(input_transport, "push_audio_frame", None)
                    if _original is None:
                        logger.debug(
                            "NoveumPipecatTracer: push_audio_frame not found on "
                            "input transport — raw audio tap skipped"
                        )
                    else:
                        _observer_ref = self.observer

                        async def _patched(
                            frame: Any, *args: Any, **kwargs: Any
                        ) -> Any:
                            # Side-band: a capture failure must NEVER propagate
                            # onto the audio path (non-intrusive guarantee).
                            try:
                                _observer_ref.capture_raw_input_audio(frame)
                            except Exception:
                                logger.debug(
                                    "NoveumPipecatTracer: raw audio capture failed; "
                                    "continuing without it",
                                    exc_info=True,
                                )
                            return await _original(frame, *args, **kwargs)

                        input_transport.push_audio_frame = _patched
                        input_transport._noveum_tap_applied = True
                        logger.debug(
                            "NoveumPipecatTracer: monkeypatched push_audio_frame "
                            "on input transport for raw audio capture"
                        )
            except Exception:
                logger.warning(
                    "NoveumPipecatTracer: failed to tap transport for raw audio — "
                    "session will continue without it",
                    exc_info=True,
                )

        # attach_to_task wires TurnTrackingObserver / UserBotLatencyObserver
        # from the task, scans the pipeline for AudioBufferProcessor
        # (full-conversation stereo audio), and registers the
        # on_pipeline_finished safety-net handler.
        await self.observer.attach_to_task(task)

        return task

    async def observe_and_create_task(
        self,
        pipeline: Any,
        *,
        transport: Optional[Any] = None,
        runner_args: Optional[Any] = None,
        params: Optional[Any] = None,
        **task_kwargs: Any,
    ) -> Any:
        """
        Convenience wrapper: ``observe_pipeline`` + ``PipelineTask`` +
        ``register_task_handlers`` in one call.

        Use this when no ``PipelineTask`` kwargs need to be set between
        pipeline wrapping and handler registration.  For full control use
        the two-call form::

            pipeline = tracer.observe_pipeline(pipeline)
            task = PipelineTask(pipeline, params=..., idle_timeout_secs=30)
            task = await tracer.register_task_handlers(
                task, transport=transport, runner_args=runner_args
            )

        Args:
            pipeline: A constructed ``Pipeline`` instance.
            transport: Forwarded to ``register_task_handlers``.
            runner_args: Optional ``RunnerArguments``; forwarded to
                ``register_task_handlers`` for session metadata extraction.
            params: ``PipelineParams`` passed to ``PipelineTask``.
            **task_kwargs: Additional keyword arguments forwarded to
                ``PipelineTask`` (e.g. ``idle_timeout_secs``).

        Returns:
            A fully-wired ``PipelineTask``.
        """
        try:
            from pipecat.pipeline.task import PipelineTask
        except ImportError as exc:
            raise ImportError(
                "pipecat-ai is required for NoveumPipecatTracer. "
                "Install with: pip install 'noveum-trace[pipecat]'"
            ) from exc

        pipeline = self.observe_pipeline(pipeline)
        task = PipelineTask(pipeline, params=params, **task_kwargs)
        return await self.register_task_handlers(
            task, transport=transport, runner_args=runner_args
        )
