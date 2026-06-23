"""Unit tests for PolicyPoller.

Scenarios:
  - force_refresh() immediately polls all registered policies
  - Policies with poll_interval=None are skipped
  - start() triggers an initial poll before the background thread launches
  - stop() terminates the background thread cleanly
  - Policies attached mid-run are polled on next force_refresh
"""

from __future__ import annotations

import time
from typing import Optional

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.policies.base import AbstractPolicy
from noveum_trace.guard.poller import PolicyPoller
from noveum_trace.guard.types import PolicyDeps

# ---------------------------------------------------------------------------
# Helpers — stub policy that records poll() calls
# ---------------------------------------------------------------------------


class _PollSpy(AbstractPolicy):
    name = "poll_spy"
    poll_interval: Optional[float] = 30.0

    def __init__(self, *, poll_interval: Optional[float] = 30.0) -> None:
        super().__init__()
        self.poll_interval = poll_interval
        self.poll_calls: list = []

    def poll(self, deps: PolicyDeps) -> None:
        self.poll_calls.append(time.monotonic())


def _engine_with(*spies: _PollSpy) -> PolicyEngine:
    api = GuardAPIClient()
    engine = PolicyEngine(api_client=api)
    for spy in spies:
        engine.attach(spy)
    return engine


# ---------------------------------------------------------------------------
# force_refresh
# ---------------------------------------------------------------------------


class TestForceRefresh:
    def test_polls_all_registered_policies(self):
        spy1 = _PollSpy()
        spy2 = _PollSpy()
        engine = _engine_with(spy1, spy2)
        poller = PolicyPoller(engine)

        poller.force_refresh()

        assert len(spy1.poll_calls) == 1
        assert len(spy2.poll_calls) == 1

    def test_skips_policies_with_no_poll_interval(self):
        active = _PollSpy(poll_interval=60.0)
        inactive = _PollSpy(poll_interval=None)
        engine = _engine_with(active, inactive)
        poller = PolicyPoller(engine)

        poller.force_refresh()

        assert len(active.poll_calls) == 1
        assert len(inactive.poll_calls) == 0

    def test_multiple_calls_accumulate(self):
        spy = _PollSpy()
        engine = _engine_with(spy)
        poller = PolicyPoller(engine)

        poller.force_refresh()
        poller.force_refresh()
        poller.force_refresh()

        assert len(spy.poll_calls) == 3

    def test_poll_exception_does_not_propagate(self):
        """A crashing policy poll must not kill the poller."""

        class CrashingPolicy(_PollSpy):
            name = "crasher"

            def poll(self, deps: PolicyDeps) -> None:
                raise RuntimeError("poll failure")

        engine = _engine_with(CrashingPolicy())
        poller = PolicyPoller(engine)

        # Should not raise
        poller.force_refresh()


# ---------------------------------------------------------------------------
# start() — initial poll
# ---------------------------------------------------------------------------


class TestStart:
    def test_start_triggers_immediate_poll(self):
        spy = _PollSpy(
            poll_interval=3600.0
        )  # long interval so background loop won't fire
        engine = _engine_with(spy)
        poller = PolicyPoller(engine)

        poller.start()
        poller.stop()

        # The immediate first poll from start() must have fired
        assert len(spy.poll_calls) >= 1

    def test_start_twice_does_not_double_start(self):
        """Calling start() a second time while the thread is alive is a no-op."""
        spy = _PollSpy(poll_interval=3600.0)
        engine = _engine_with(spy)
        poller = PolicyPoller(engine)

        poller.start()
        thread_id_first = poller._thread.ident if poller._thread else None
        poller.start()
        thread_id_second = poller._thread.ident if poller._thread else None
        poller.stop()

        assert thread_id_first == thread_id_second


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_terminates_background_thread(self):
        spy = _PollSpy(poll_interval=3600.0)
        engine = _engine_with(spy)
        poller = PolicyPoller(engine)

        poller.start()
        assert poller._thread is not None and poller._thread.is_alive()

        poller.stop()

        # Give the thread a moment to finish
        if poller._thread:
            poller._thread.join(timeout=2.0)
        assert poller._thread is None or not poller._thread.is_alive()

    def test_stop_before_start_does_not_raise(self):
        engine = _engine_with()
        poller = PolicyPoller(engine)
        # Should not raise even though thread was never started
        poller.stop()


# ---------------------------------------------------------------------------
# Attach mid-run
# ---------------------------------------------------------------------------


class TestAttachMidRun:
    def test_newly_attached_policy_is_polled_on_force_refresh(self):
        spy1 = _PollSpy()
        engine = _engine_with(spy1)
        poller = PolicyPoller(engine)

        # Attach spy2 after poller creation
        spy2 = _PollSpy()
        spy2.name = "late_spy"
        engine.attach(spy2)

        poller.force_refresh()

        assert len(spy1.poll_calls) == 1
        assert len(spy2.poll_calls) == 1

    def test_detached_policy_not_polled(self):
        spy = _PollSpy()
        engine = _engine_with(spy)
        poller = PolicyPoller(engine)

        engine.detach(spy.name)
        poller.force_refresh()

        assert len(spy.poll_calls) == 0


# ---------------------------------------------------------------------------
# Interval-based polling (patching time.monotonic)
# ---------------------------------------------------------------------------


class TestIntervalPolling:
    def test_policy_fires_when_interval_elapsed(self):
        """Simulate enough monotonic time passing to trigger interval-based polling."""
        import unittest.mock

        spy = _PollSpy(poll_interval=10.0)
        engine = _engine_with(spy)
        poller = PolicyPoller(engine, tick=0.01)

        # Patch monotonic so the poller thinks 15 seconds have passed after a few ticks.
        base = time.monotonic()
        call_count = [0]
        original_monotonic = time.monotonic

        def fake_monotonic():
            call_count[0] += 1
            # First few calls: return real time for startup; thereafter fast-forward
            if call_count[0] > 3:
                return base + 15.0
            return original_monotonic()

        # Patch inside the poller module so the background thread sees the fake clock.
        with unittest.mock.patch(
            "noveum_trace.guard.poller.time.monotonic", fake_monotonic
        ):
            poller.start()
            # Give the background thread a moment to tick at least once past startup.
            time.sleep(0.1)
            poller.stop()

        # start() fires an immediate force_refresh AND the interval logic should
        # have fired at least one additional poll once fake time jumped 15 s.
        assert (
            len(spy.poll_calls) >= 2
        ), f"Expected at least 2 poll calls (initial + interval), got {len(spy.poll_calls)}"
