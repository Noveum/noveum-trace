"""
Session lifecycle: start wrapping + close handling + setup/cleanup
(Section H of LIVEKIT_TEST_PLAN.md).

``_on_close`` is exercised SYNCHRONOUSLY: with no running event loop it runs its
handler to completion via ``run_until_complete`` (the integration's documented
fallback), which makes the assertions deterministic without sleeps.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents.llm import ChatContext  # noqa: E402
from livekit.agents.voice.events import CloseReason  # noqa: E402

from noveum_trace.core.context import set_current_trace  # noqa: E402
from noveum_trace.core.span import SpanStatus  # noqa: E402
from noveum_trace.integrations.livekit import setup_livekit_tracing  # noqa: E402
from noveum_trace.integrations.livekit.livekit_session import (  # noqa: E402
    _LiveKitTracingManager,
)


class FakeSession:
    """Minimal AgentSession: real-ish ``start``/``on``/``off``, no agent_activity
    or _recorder_io (so realtime/recorder setup short-circuit cleanly)."""

    def __init__(self, *, start_error=None, history=None):
        self._handlers: dict = {}
        self._start_error = start_error
        self.history = history
        self.started_with = None

    async def start(self, agent, **kwargs):
        self.started_with = kwargs
        if self._start_error:
            raise self._start_error
        return "ok"

    def on(self, event, handler):
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event, handler):
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)


def _agent(*, instructions="You are helpful.", tools=None):
    return SimpleNamespace(
        label="Assistant",
        instructions=instructions,
        tools=tools if tools is not None else [],
    )


def _ctx_with_two_messages():
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="Hi")
    ctx.add_message(role="assistant", content="Hello!")
    return ctx


# --------------------------------------------------------------------------- #
# H1 — wrapped start creates a trace with the expected attributes
# --------------------------------------------------------------------------- #
async def test_wrapped_start_creates_trace_with_attributes(lk_client):
    session = FakeSession()
    agent = _agent(
        tools=[SimpleNamespace(name="get_weather", description="Get weather")]
    )
    mgr = setup_livekit_tracing(session, record=True)

    result = await session.start(agent)

    assert result == "ok"
    trace = mgr._trace
    assert trace is not None
    assert trace.name == "livekit.agent_session"  # default name (no job context)
    a = trace.attributes
    assert a["livekit.session_type"] == "agent_session"
    assert a["livekit.agent.label"] == "Assistant"
    assert a["llm.system_prompt"] == "You are helpful."
    assert a["llm.available_tools.count"] == 1
    assert a["llm.available_tools.names"] == ["get_weather"]
    # record=True is forced into the underlying start call
    assert session.started_with.get("record") is True


# --------------------------------------------------------------------------- #
# H2 — original start raises -> trace marked ERROR and exception propagates
# --------------------------------------------------------------------------- #
async def test_wrapped_start_marks_error_and_reraises(lk_client):
    session = FakeSession(start_error=RuntimeError("boom"))
    mgr = setup_livekit_tracing(session)

    captured = {}
    original_finish = lk_client.finish_trace

    def spy(trace, end_time=None):
        captured["trace"] = trace
        return original_finish(trace, end_time)

    with patch.object(lk_client, "finish_trace", side_effect=spy):
        with pytest.raises(RuntimeError):
            await session.start(_agent())

    assert captured["trace"].status == SpanStatus.ERROR
    assert mgr._trace is None


# --------------------------------------------------------------------------- #
# H3 — trace creation failure -> fall back to original start (no crash)
# --------------------------------------------------------------------------- #
async def test_wrapped_start_falls_back_when_trace_creation_fails(lk_client):
    session = FakeSession()
    mgr = setup_livekit_tracing(session)

    with patch.object(lk_client, "start_trace", side_effect=RuntimeError("no trace")):
        result = await session.start(_agent())

    assert result == "ok"  # original start still ran
    assert session.started_with is not None
    assert mgr._trace is None


# --------------------------------------------------------------------------- #
# H4 — close handler finishes the trace with history + status by reason
# --------------------------------------------------------------------------- #
def test_on_close_finishes_trace_with_history_and_ok_status(lk_client):
    session = FakeSession(history=_ctx_with_two_messages())
    mgr = _LiveKitTracingManager(session=session)
    trace = lk_client.start_trace("livekit.close_test")
    set_current_trace(trace)
    mgr._trace = trace
    mgr._available_tools = [{"name": "t", "description": "d"}]
    # Pre-mark ERROR so the OK transition must come from the close handler's
    # explicit set_status(OK) — finish()'s UNSET->OK auto-promotion would NOT
    # overwrite an explicit ERROR, so this distinguishes real behavior.
    trace.set_status(SpanStatus.ERROR, "pre-existing")

    mgr._on_close(SimpleNamespace(reason=CloseReason.JOB_SHUTDOWN))  # no running loop

    assert trace.is_finished()
    assert trace.status == SpanStatus.OK
    assert trace.attributes["conversation.history.message_count"] == 2
    # the serialized history must actually contain the two messages
    history_json = trace.attributes["conversation.history"]
    assert "Hi" in history_json and "Hello!" in history_json
    assert trace.attributes["agent.available_tools.count"] == 1
    assert trace.attributes["agent.available_tools.names"] == ["t"]
    assert trace.attributes["agent.available_tools.descriptions"] == ["d"]
    assert mgr._trace is None


def test_on_close_error_reason_sets_error_status(lk_client):
    session = FakeSession(history=None)
    mgr = _LiveKitTracingManager(session=session)
    trace = lk_client.start_trace("livekit.close_err")
    set_current_trace(trace)
    mgr._trace = trace

    mgr._on_close(SimpleNamespace(reason=CloseReason.ERROR, error="kaboom"))

    assert trace.is_finished()
    assert trace.status == SpanStatus.ERROR


# --------------------------------------------------------------------------- #
# H5 — disabled setup neither wraps start nor registers handlers
# --------------------------------------------------------------------------- #
def test_setup_disabled_does_not_wrap_or_register():
    session = FakeSession()
    mgr = setup_livekit_tracing(session, enabled=False)

    assert mgr.enabled is False
    assert mgr._original_start is None
    assert session._handlers == {}


# --------------------------------------------------------------------------- #
# H6 — cleanup restores start and removes every registered handler
# --------------------------------------------------------------------------- #
def test_cleanup_restores_start_and_removes_handlers(lk_client):
    session = FakeSession()
    mgr = setup_livekit_tracing(session)

    assert mgr._original_start is not None
    original = mgr._original_start
    # exact set of AgentSession events hooked (not just the count)
    assert set(session._handlers) == {
        "user_state_changed",
        "agent_state_changed",
        "user_input_transcribed",
        "conversation_item_added",
        "agent_false_interruption",
        "function_tools_executed",
        "metrics_collected",
        "speech_created",
        "error",
        "close",
    }

    mgr.cleanup()

    assert session.start == original
    for handlers in session._handlers.values():
        assert handlers == []
