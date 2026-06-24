from __future__ import annotations

import logging
import random
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from noveum_trace.guard.engine import PolicyEngine
    from noveum_trace.guard.policies.base import AbstractPolicy

_log = logging.getLogger(__name__)

_JITTER_MAX = (
    2.0  # seconds; prevents thundering herd when many policies share an interval
)

# How often (seconds) the poller fetches the full policy list from the backend.
# This is independent of per-policy poll_interval (which refreshes local spend state).
_BACKEND_FETCH_INTERVAL = 60.0

# Registry mapping backend ``type`` strings → concrete policy classes.
# New policy types must be registered here so the poller can instantiate them.
_POLICY_TYPE_REGISTRY: dict[str, type] = {}


def register_policy_type(type_name: str, cls: type) -> None:
    """Register a policy class for a given backend ``type`` string.

    Called at import time by each policy module so the poller can instantiate
    policies returned by the backend without a hard import dependency here.

    Example::

        register_policy_type("cost_cap", CostCapPolicy)
    """
    _POLICY_TYPE_REGISTRY[type_name] = cls


class PolicyPoller:
    """Daemon thread that:

    1. Calls ``poll()`` on each registered policy at its own ``poll_interval``
       (refreshes local spend state / data_map from the backend).

    2. Periodically fetches the full policy list from the backend via
       ``api_client.fetch_remote_policies(project_id)`` and **upserts** each
       entry into the engine:

       - If the engine already has a policy with that name → call
         ``policy.update_params(config)`` so parameters stay in sync without
         replacing the object (and losing any in-flight reservation state).
       - If the engine does **not** have that policy → instantiate it from the
         ``_POLICY_TYPE_REGISTRY`` and ``engine.attach()`` it so the next call
         is protected immediately.

    This makes the backend the **single source of truth** for which policies are
    active and what their parameters are.  The ``policies=[]`` argument to
    ``noveum_trace.init()`` is retained for backwards-compatibility and for local
    development / testing, but should **not** be used alongside backend polling
    (doing so may result in conflicting parameters for the same policy name).

    Each policy declares ``poll_interval`` (seconds); ``None`` means skip step 1.
    A small random jitter is added so co-registered policies don't all fire at
    the same wall-clock tick.

    Usage::

        poller = PolicyPoller(engine, project_id="my-project")
        poller.start()   # immediate first poll + backend fetch, then runs on schedule
        ...
        poller.stop()    # signals the thread to exit; joins with timeout
    """

    def __init__(
        self,
        engine: PolicyEngine,
        tick: float = 1.0,
        project_id: Optional[str] = None,
        backend_fetch_interval: float = _BACKEND_FETCH_INTERVAL,
    ) -> None:
        self._engine = engine
        self._tick = tick  # inner sleep granularity; must be < smallest poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_poll: dict[str, float] = {}  # policy.name → monotonic timestamp
        # Per-policy jitter fixed at first-seen time so the offset stays constant
        # across ticks — re-randomising on every tick collapses all policies to the
        # same wall-clock second whenever a tiny value is drawn.
        self._policy_jitter: dict[str, float] = {}  # policy.name → fixed jitter (s)
        # project_id for backend policy fetch; resolved lazily from _state if None
        self._project_id = project_id
        self._backend_fetch_interval = backend_fetch_interval
        self._last_backend_fetch: float = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._fetch_backend_policies()  # immediate first backend fetch
        self._poll_all_now()  # immediate first poll before background loop starts
        self._last_backend_fetch = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="noveum-guard-poller"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._tick * 10)

    def force_refresh(self) -> None:
        """Synchronously poll all policies right now (useful after attach()).

        Also triggers a backend policy fetch so any new policies are picked up
        immediately rather than waiting for the next scheduled fetch.
        """
        self._fetch_backend_policies()
        self._poll_all_now()
        self._last_backend_fetch = time.monotonic()

    # Internal

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self._tick):
            now = time.monotonic()
            for policy in self._engine.policies:
                interval = policy.poll_interval
                if interval is None:
                    continue
                # Fix jitter once per policy so the offset is stable across ticks.
                # Re-randomising every tick means a near-zero draw can fire all
                # policies simultaneously, defeating thundering-herd prevention.
                if policy.name not in self._policy_jitter:
                    self._policy_jitter[policy.name] = random.uniform(0.0, _JITTER_MAX)
                jitter = self._policy_jitter[policy.name]
                last = self._last_poll.get(policy.name, 0.0)
                if now - last >= interval + jitter:
                    self._poll_one(policy)
                    self._last_poll[policy.name] = now

            # Backend policy-definition fetch (independent cadence)
            if now - self._last_backend_fetch >= self._backend_fetch_interval:
                self._fetch_backend_policies()
                self._last_backend_fetch = now

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

    def _resolve_project_id(self) -> Optional[str]:
        """Return the project_id for the backend fetch.

        Uses the explicitly supplied ``project_id`` when available; otherwise
        falls back to the ambient ``PolicyContext`` stored in ``_state``.
        """
        if self._project_id:
            return self._project_id
        try:
            from noveum_trace.guard import _state

            ctx = _state.get_context()
            return ctx.project_id if ctx else None
        except Exception:
            return None

    def _fetch_backend_policies(self) -> None:
        """Fetch policy definitions from the backend and upsert into the engine.

        Upsert semantics:
        - Policy already in engine  → call ``update_params(config)``
        - Policy not in engine      → instantiate from registry and ``attach()``

        Unknown policy types are skipped with a debug log so a backend
        configuration mistake never crashes the poller thread.
        """
        project_id = self._resolve_project_id()
        if not project_id:
            return

        api = self._engine._api_client
        try:
            remote_policies: list[dict[str, Any]] = api.fetch_remote_policies(
                project_id
            )
        except Exception as exc:
            _log.debug("PolicyPoller: fetch_remote_policies failed — %s", exc)
            return

        if not remote_policies:
            return

        # Build a name-keyed index of currently registered policies for O(1) lookup.
        existing: dict[str, AbstractPolicy] = {p.name: p for p in self._engine.policies}

        for config in remote_policies:
            policy_type = config.get("type", "")
            # Prefer an explicit ``name`` from the backend; fall back to ``type``
            # so simple configs don't have to repeat themselves.
            policy_name = config.get("name", policy_type)

            if policy_name in existing:
                # Update params on the live policy object (thread-safe via its lock).
                try:
                    existing[policy_name].update_params(config)
                    _log.debug(
                        "PolicyPoller: updated params for policy %r", policy_name
                    )
                except Exception as exc:
                    _log.debug(
                        "PolicyPoller: update_params failed for %r — %s",
                        policy_name,
                        exc,
                    )
            else:
                # Instantiate and attach a new policy from the registry.
                cls = _POLICY_TYPE_REGISTRY.get(policy_type)
                if cls is None:
                    _log.debug(
                        "PolicyPoller: unknown policy type %r — skipping", policy_type
                    )
                    continue
                try:
                    new_policy = _instantiate_policy(cls, policy_name, config)
                    # Bind the ambient context so poll() can scope its get_state() call.
                    try:
                        from noveum_trace.guard import _state

                        ctx = _state.get_context()
                        if ctx is not None:
                            new_policy.bind_context(ctx)
                    except Exception:
                        pass
                    self._engine.attach(new_policy)
                    _log.debug(
                        "PolicyPoller: attached new policy %r (type=%r)",
                        policy_name,
                        policy_type,
                    )
                except Exception as exc:
                    _log.debug(
                        "PolicyPoller: failed to instantiate policy %r — %s",
                        policy_type,
                        exc,
                    )


def _instantiate_policy(cls: type, name: str, config: dict[str, Any]) -> AbstractPolicy:
    """Construct a policy instance from a backend config dict.

    The ``type`` and ``name`` keys are stripped before passing ``config`` to
    the constructor so each policy class only sees its own parameters.
    Positional construction is attempted first using the class's ``__init__``
    signature; unknown extra keys are silently dropped.
    """
    import inspect

    params = {k: v for k, v in config.items() if k not in ("type", "name")}
    sig = inspect.signature(cls)
    accepted = set(sig.parameters)
    filtered = {k: v for k, v in params.items() if k in accepted}
    policy = cls(**filtered)
    # Override name if the backend supplied an explicit one.
    policy.name = name
    return policy
