#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

r"""Reproduce missed Noveum trace flush when using ``task.cancel()`` (hang-up teardown).

Production pattern (e.g. Exotel WebSocket): ``on_client_disconnected`` calls
``await task.cancel()``. Pipecat queues a ``CancelFrame`` and eventually tears
down the task. In parallel, ``TaskObserver`` puts every ``FramePushed`` event on
per-observer asyncio queues. The pipeline task considers the run finished once
``CancelFrame`` reaches the sink and sets ``_pipeline_end_event``, *before* those
queues are necessarily drained. ``PipelineTask``\ ’s ``finally`` block then calls
``TaskObserver.stop()``, which cancels the proxy tasks — so ``NoveumTraceObserver``
may never run ``on_push_frame`` for the terminal hop and never calls
``_finish_conversation()`` (no ``finish_trace`` / ``flush``).

This script mirrors ``pipecat_integration_example.py`` only in how Noveum and the
observer are initialised; the pipeline is intentionally minimal (a chain of
identity filters) plus a large burst of ``TextFrame``\ s to deepen the observer
queues, then ``task.cancel()`` simulating hang-up.

Run from anywhere after adding ``NOVEUM_API_KEY`` to ``.env`` at the repository
root (same as other examples)::

    python docs/examples/pipecat_task_cancel_repro_example.py

If the race triggers, logs report that ``NoveumTraceObserver`` still holds an
open trace when the runner exits.

By default the script subclasses the observer with a tiny ``on_push_frame`` delay
(``NOVEUM_REPRO_OBSERVER_DELAY``, default ``8e-5``) so the backlog in Pipecat's
``TaskObserver`` reliably outruns the proxy task. Set ``NOVEUM_REPRO_OBSERVER_DELAY=0``
to use an unmodified consumer (race may not show on a fast machine — then raise
``TEXT_BURST`` / ``NUM_IDENTITY_STAGES`` instead).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.filters.identity_filter import IdentityFilter

import noveum_trace
from noveum_trace.integrations.pipecat import NoveumTraceObserver

# Resolve repo-root `.env` so the key loads when cwd is not the project root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env", override=True)

# Deeper pipelines and larger bursts increase FramePushed backlog in TaskObserver queues.
NUM_IDENTITY_STAGES = 12
TEXT_BURST = 4_000
# TaskObserver delivers FramePushed via asyncio queues; if the consumer is slower than
# the pipeline, ``CancelFrame`` can reach the sink (run completes) before the
# Noveum observer handles the terminal notifications. A tiny per-hook delay makes the
# repro reliable without changing production code.
OBSERVER_PUSH_DELAY_SEC = float(os.getenv("NOVEUM_REPRO_OBSERVER_DELAY", "8e-5"))


class SlowNoveumTraceObserver(NoveumTraceObserver):
    """NoveumTraceObserver with an optional delay in ``on_push_frame`` for repro only."""

    def __init__(
        self, *args: object, push_delay_sec: float = 0.0, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self._push_delay_sec = push_delay_sec

    async def on_push_frame(self, data: object) -> None:  # FramePushed at runtime
        if self._push_delay_sec > 0:
            await asyncio.sleep(self._push_delay_sec)
        await super().on_push_frame(data)


async def main() -> None:
    api_key = os.getenv("NOVEUM_API_KEY")
    if not api_key:
        logger.warning(
            "NOVEUM_API_KEY not set (add it to .env at %s) — trace export is skipped but cancellation race still runs.",
            _REPO_ROOT,
        )

    noveum_trace.init(
        api_key=api_key,
        project=os.getenv("NOVEUM_PROJECT", "pipecat-cancel-repro"),
    )

    processors = [
        IdentityFilter(name=f"identity-{i}") for i in range(NUM_IDENTITY_STAGES)
    ]
    pipeline = Pipeline(processors)

    trace_obs = SlowNoveumTraceObserver(
        trace_name_prefix=os.getenv("NOVEUM_TRACE_PREFIX", "pipecat-cancel-repro"),
        capture_text=True,
        capture_function_calls=True,
        record_audio=False,
        push_delay_sec=OBSERVER_PUSH_DELAY_SEC,
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=False,
            enable_usage_metrics=False,
        ),
        observers=[trace_obs],
    )
    await trace_obs.attach_to_task(task)

    async def simulated_client_hangup() -> None:
        # Wait for StartFrame to finish traversing the pipeline.
        await asyncio.sleep(0.75)
        logger.info(
            "Simulated hang-up: queuing {} TextFrames then await task.cancel() "
            "(same idea as on_client_disconnected).",
            TEXT_BURST,
        )
        for i in range(TEXT_BURST):
            await task.queue_frame(TextFrame(text=f"burst-{i}"))
        await task.cancel()

    hangup_task = asyncio.create_task(simulated_client_hangup())

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        hangup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hangup_task

    # ``_finish_conversation`` sets ``_trace`` to None after finish_trace + flush.
    if trace_obs._trace is not None:
        logger.error(
            "Repro: NoveumTraceObserver still has an open conversation trace after runner.run() "
            "— CancelFrame / teardown likely did not drain to ``_finish_conversation``."
        )
    else:
        logger.warning(
            "Observer cleared the trace on teardown. Raise NOVEUM_REPRO_OBSERVER_DELAY (e.g. 2e-4), "
            "TEXT_BURST, or NUM_IDENTITY_STAGES to reproduce on a fast machine."
        )


if __name__ == "__main__":
    asyncio.run(main())
