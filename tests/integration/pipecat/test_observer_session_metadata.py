"""
Tier 4 — session-metadata capture (``capture_session_metadata``).

Spec mapping: S6/S7-adjacent operational metadata + the action-doc goal of
correlating Noveum traces with the customer's own session context (room URL,
transport type, idle timeout, bot name).

``register_task_handlers`` calls ``observer._store_transport(transport,
runner_args=…)`` *before* ``attach_to_task``.  The trace may not exist yet at
that point, so metadata is buffered and flushed onto the root conversation trace
at first client/bot connection (``_flush_session_metadata``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _observer(**kwargs: Any) -> Any:
    pytest.importorskip("pipecat.observers.base_observer")
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    return NoveumTraceObserver(record_audio=False, **kwargs)


# --------------------------------------------------------------------------- #
# _store_transport — extraction from transport + runner_args                  #
# --------------------------------------------------------------------------- #
def test_store_transport_extracts_transport_type_and_room(fake_transport: Any) -> None:
    obs = _observer(capture_session_metadata=True)
    transport = fake_transport(room_url="https://example.daily.co/room1")

    obs._store_transport(transport)

    meta = obs._session_metadata
    assert meta["session.transport_type"] == "FakeTransport"
    assert meta["session.room_url"] == "https://example.daily.co/room1"
    assert obs._transport is transport


def test_store_transport_reads_room_name_when_no_room_url(fake_transport: Any) -> None:
    obs = _observer(capture_session_metadata=True)
    transport = fake_transport(room_name="lobby")

    obs._store_transport(transport)

    assert obs._session_metadata["session.room_url"] == "lobby"


def test_store_transport_extracts_runner_args() -> None:
    obs = _observer(capture_session_metadata=True)
    transport = SimpleNamespace()  # no room attrs
    runner_args = SimpleNamespace(
        room_url="https://daily.co/authoritative",
        pipeline_idle_timeout_secs=30,
        bot_name="haptik-bot",
    )

    obs._store_transport(transport, runner_args=runner_args)

    meta = obs._session_metadata
    assert meta["session.room_url"] == "https://daily.co/authoritative"
    assert meta["session.idle_timeout_secs"] == 30
    assert meta["session.bot_name"] == "haptik-bot"


def test_store_transport_captures_zero_idle_timeout() -> None:
    """``idle_timeout_secs`` uses an ``is not None`` check (not truthiness) so a
    deliberate ``0`` is recorded, not dropped as falsy."""
    obs = _observer(capture_session_metadata=True)
    runner_args = SimpleNamespace(pipeline_idle_timeout_secs=0)

    obs._store_transport(SimpleNamespace(), runner_args=runner_args)

    assert obs._session_metadata["session.idle_timeout_secs"] == 0


def test_store_transport_noop_when_capture_disabled(fake_transport: Any) -> None:
    obs = _observer(capture_session_metadata=False)
    obs._store_transport(fake_transport(room_url="x"))
    assert obs._session_metadata == {}
    assert obs._transport is None


def test_store_transport_degrades_on_weird_transport() -> None:
    """Extraction is all ``getattr`` — a transport with no expected attrs must
    not raise; we still record at least the class name."""
    obs = _observer(capture_session_metadata=True)

    class Weird:
        pass

    obs._store_transport(Weird())
    assert obs._session_metadata["session.transport_type"] == "Weird"


# --------------------------------------------------------------------------- #
# _flush_session_metadata — write once onto the trace, then clear             #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_flush_writes_metadata_to_trace_then_clears(fake_transport: Any) -> None:
    from noveum_trace.core.trace import Trace

    obs = _observer(capture_session_metadata=True)
    obs._store_transport(fake_transport(room_url="https://daily.co/r"))
    trace = Trace(name="pipecat.conversation")
    obs._trace = trace

    await obs._flush_session_metadata()

    assert trace.attributes["session.room_url"] == "https://daily.co/r"
    assert trace.attributes["session.transport_type"] == "FakeTransport"
    # Buffer cleared so a second flush is a no-op.
    assert obs._session_metadata == {}

    trace.attributes.pop("session.room_url")
    await obs._flush_session_metadata()
    assert "session.room_url" not in trace.attributes


@pytest.mark.asyncio
async def test_flush_noop_when_trace_absent(fake_transport: Any) -> None:
    obs = _observer(capture_session_metadata=True)
    obs._store_transport(fake_transport(room_url="x"))
    obs._trace = None

    await obs._flush_session_metadata()

    # Nothing written, buffer retained for a later flush once the trace exists.
    assert obs._session_metadata.get("session.room_url") == "x"


# --------------------------------------------------------------------------- #
# Wiring — register_task_handlers buffers transport metadata                  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_register_task_handlers_stores_session_metadata(
    make_pipeline: Any, fake_transport: Any
) -> None:
    from pipecat.pipeline.task import PipelineTask

    from noveum_trace.integrations.pipecat import NoveumPipecatTracer

    tracer = NoveumPipecatTracer(
        record_audio=False,
        record_raw_input_audio=False,
        capture_session_metadata=True,
    )
    pipeline = tracer.observe_pipeline(make_pipeline(2))
    task = PipelineTask(pipeline)
    transport = fake_transport(room_url="https://daily.co/session")
    runner_args = SimpleNamespace(pipeline_idle_timeout_secs=45)

    await tracer.register_task_handlers(
        task, transport=transport, runner_args=runner_args
    )

    meta = tracer.observer._session_metadata
    assert meta["session.room_url"] == "https://daily.co/session"
    assert meta["session.idle_timeout_secs"] == 45
