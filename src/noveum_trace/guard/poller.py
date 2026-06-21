from __future__ import annotations

import random
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noveum_trace.guard.engine import PolicyEngine
    from noveum_trace.guard.policies.base import AbstractPolicy


_JITTER_MAX = (
    2.0  # seconds; prevents thundering herd when many policies share an interval
)


class PolicyPoller:
    """Daemon thread that calls poll() on each policy at its own interval.

    Each policy declares poll_interval (seconds); None means skip.
    A small random jitter is added so co-registered policies don't all fire
    at the same wall-clock tick.

    Usage:
        poller = PolicyPoller(engine)
        poller.start()   # triggers an immediate first poll, then runs on schedule
        ...
        poller.stop()    # signals the thread to exit; joins with timeout
    """

    def __init__(self, engine: PolicyEngine, tick: float = 1.0) -> None:
        self._engine = engine
        self._tick = tick  # inner sleep granularity; must be < smallest poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_poll: dict[str, float] = {}  # policy.name → monotonic timestamp

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_all_now()  # immediate first poll before background loop starts
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="noveum-guard-poller"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._tick * 10)

    def force_refresh(self) -> None:
        """Synchronously poll all policies right now (useful after attach())."""
        self._poll_all_now()

    # Internal

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self._tick):
            now = time.monotonic()
            for policy in self._engine.policies:
                interval = policy.poll_interval
                if interval is None:
                    continue
                last = self._last_poll.get(policy.name, 0.0)
                jitter = random.uniform(0.0, _JITTER_MAX)
                if now - last >= interval + jitter:
                    self._poll_one(policy)
                    self._last_poll[policy.name] = now

    def _poll_all_now(self) -> None:
        now = time.monotonic()
        for policy in self._engine.policies:
            if policy.poll_interval is not None:
                self._poll_one(policy)
                self._last_poll[policy.name] = now

    def _poll_one(self, policy: AbstractPolicy) -> None:
        from noveum_trace.guard.types import PolicyDeps

        deps = PolicyDeps(api=self._engine._api_client)
        try:
            policy.poll(deps)
        except Exception:
            pass
