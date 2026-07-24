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

from noveum_trace.guard import _state as guard_state
from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.exceptions import GuardBackendUnavailable
from noveum_trace.guard.policies.base import AbstractPolicy

# Imported for its register_policy_type("rate_limit", ...) side effect, which the
# scope-binding tests rely on to instantiate a policy from a backend config row.
from noveum_trace.guard.policies.rate_limit import RateLimitPolicy  # noqa: F401
from noveum_trace.guard.poller import PolicyPoller
from noveum_trace.guard.types import ParsedRequest, PolicyContext, PolicyDeps

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
# Backend-unavailable handling — fail closed, loudly (P1b)
# ---------------------------------------------------------------------------


class _FakeAPIClient:
    """Stand-in for GuardAPIClient exposing fetch_remote_policies() + get_state()."""

    def __init__(self, *, raise_exc=None, policies=None, state=None):
        self._raise_exc = raise_exc
        self._policies = policies or []
        self._state = state or {}
        self.calls = 0
        self.state_calls: list = []

    def fetch_remote_policies(self, project_id):
        self.calls += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._policies

    def get_state(self, project_id, window=None):
        self.state_calls.append(project_id)
        return dict(self._state)


class TestBackendUnavailableHandling:
    def test_backend_unavailable_sets_engine_degraded(self):
        api = _FakeAPIClient(raise_exc=GuardBackendUnavailable("down"))
        engine = PolicyEngine(api_client=api)
        poller = PolicyPoller(engine, project_id="proj")

        poller._fetch_backend_policies()

        assert engine.is_backend_unavailable() is True

    def test_successful_fetch_clears_degraded_state(self):
        api = _FakeAPIClient(policies=[])
        engine = PolicyEngine(api_client=api)
        engine.set_backend_unavailable(True)
        poller = PolicyPoller(engine, project_id="proj")

        poller._fetch_backend_policies()

        assert engine.is_backend_unavailable() is False

    def test_no_project_id_does_not_change_degraded_state(self):
        """No project configured — matches today's silent early-return, unchanged."""
        api = _FakeAPIClient(raise_exc=GuardBackendUnavailable("down"))
        engine = PolicyEngine(api_client=api)
        poller = PolicyPoller(engine)  # no project_id, no ambient context

        poller._fetch_backend_policies()

        assert engine.is_backend_unavailable() is False
        assert api.calls == 0

    def test_unexpected_exception_does_not_set_degraded_state(self):
        """A bug in fetch_remote_policies itself must not crash the poller
        thread or falsely flip the engine into fail-closed mode."""
        api = _FakeAPIClient(raise_exc=RuntimeError("unexpected bug"))
        engine = PolicyEngine(api_client=api)
        poller = PolicyPoller(engine, project_id="proj")

        poller._fetch_backend_policies()  # must not raise

        assert engine.is_backend_unavailable() is False


# ---------------------------------------------------------------------------
# Scope binding for backend-fetched policies
# ---------------------------------------------------------------------------

_RATE_POLICY_ROW = {
    "type": "rate_limit",
    "name": "backend-rate",
    "fail_closed": True,
    "windows": [{"period": "1h", "maxRequests": 10}],
}
_BACKEND_RATE = {"requests_1h": 99, "tokens_1h": 990}


def _ctx(project_id: str) -> PolicyContext:
    return PolicyContext(
        project_id=project_id,
        organization_id=None,
        environment="test",
        trace_id=None,
        span_id=None,
        call_id="c",
    )


def _parsed_request() -> ParsedRequest:
    return ParsedRequest(
        provider="openai",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        max_tokens=16,
        estimated_input_tokens=10,
        raw_body=b"{}",
    )


class TestBackendPolicyScopeBinding:
    """A policy instantiated from the backend must learn its project scope.

    Without it ``poll()`` returns on its first line and the policy never reads
    ``get_state()`` — enforcing only against calls made by this process, which
    silently defeats cross-process rate limits and cost caps.
    """

    @staticmethod
    def _fetch_and_poll(api, **poller_kwargs):
        engine = PolicyEngine(api_client=api)
        poller = PolicyPoller(engine, **poller_kwargs)
        poller._fetch_backend_policies()
        poller._poll_all_now()
        policy = next(p for p in engine.policies if p.name == "backend-rate")
        return policy

    def test_explicit_context_scopes_policy(self):
        api = _FakeAPIClient(policies=[_RATE_POLICY_ROW], state={"rate": _BACKEND_RATE})

        policy = self._fetch_and_poll(api, project_id="proj", context=_ctx("proj"))

        assert policy._stored_scope_id() == "proj"
        assert policy.data_map == _BACKEND_RATE
        assert api.state_calls == ["proj"]

    def test_project_id_alone_scopes_policy(self):
        """No context anywhere — the poller's own project_id must still bind."""
        api = _FakeAPIClient(policies=[_RATE_POLICY_ROW], state={"rate": _BACKEND_RATE})

        policy = self._fetch_and_poll(api, project_id="proj")

        assert policy._stored_scope_id() == "proj"
        assert policy.data_map == _BACKEND_RATE

    def test_ambient_state_context_scopes_policy(self):
        api = _FakeAPIClient(policies=[_RATE_POLICY_ROW], state={"rate": _BACKEND_RATE})
        engine = PolicyEngine(api_client=api)
        poller = PolicyPoller(engine)  # no project_id, no explicit context
        guard_state.set_guard(engine, _ctx("ambient-proj"), poller)
        try:
            poller._fetch_backend_policies()
            poller._poll_all_now()
        finally:
            guard_state.clear()

        policy = next(p for p in engine.policies if p.name == "backend-rate")
        assert policy._stored_scope_id() == "ambient-proj"
        assert policy.data_map == _BACKEND_RATE

    def test_explicit_context_wins_over_ambient(self):
        api = _FakeAPIClient(policies=[_RATE_POLICY_ROW], state={"rate": _BACKEND_RATE})
        engine = PolicyEngine(api_client=api)
        poller = PolicyPoller(engine, project_id="explicit", context=_ctx("explicit"))
        guard_state.set_guard(engine, _ctx("ambient"), poller)
        try:
            poller._fetch_backend_policies()
        finally:
            guard_state.clear()

        policy = next(p for p in engine.policies if p.name == "backend-rate")
        assert policy._stored_scope_id() == "explicit"

    def test_binding_context_is_none_when_nothing_available(self):
        poller = PolicyPoller(PolicyEngine(api_client=_FakeAPIClient()))

        assert poller._binding_context() is None

    def test_first_call_blocked_by_counts_from_another_process(self):
        """The behaviour the binding exists for: a freshly started process must
        inherit usage it did not make. The backend already reports 12 requests
        against a cap of 10, so call number one is blocked even though this
        process has made no calls at all.
        """
        api = _FakeAPIClient(
            policies=[_RATE_POLICY_ROW],
            state={"rate": {"requests_1h": 12, "tokens_1h": 120}},
        )
        policy = self._fetch_and_poll(api, project_id="proj", context=_ctx("proj"))

        decision = policy.pre(_parsed_request(), _ctx("proj"), PolicyDeps(api=api))

        assert decision.is_blocking
        assert "12/10 requests per 1h" in decision.reason


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
