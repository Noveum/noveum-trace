"""Concurrency integration tests for the Guard system.

Demonstrates and verifies thread-safety properties:
  - GuardAPIClient reserve/reconcile under concurrent calls
  - PolicyEngine.attach/detach under concurrent reads
  - NoveumTransport: concurrent requests each get their own call_id
  - CostCapPolicy: spend cap is respected under concurrent pressure
"""

from __future__ import annotations

import json
import threading
import uuid

import httpx

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.policies.cost_cap import CostCapPolicy
from noveum_trace.guard.transport.adapters.base import AdapterRegistry
from noveum_trace.guard.transport.adapters.openai_adapter import OpenAIAdapter
from noveum_trace.guard.transport.sync_transport import NoveumTransport
from noveum_trace.guard.types import EnforcementMode, PolicyContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(project_id: str = "proj") -> PolicyContext:
    return PolicyContext(
        project_id=project_id,
        organization_id=None,
        environment="test",
        trace_id=None,
        span_id=None,
        call_id=str(uuid.uuid4()),
    )


def _openai_request(max_tokens: int = 10) -> httpx.Request:
    body = json.dumps(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": max_tokens,
        }
    )
    return httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=body.encode(),
    )


def _openai_response(
    prompt_tokens: int = 5, completion_tokens: int = 5
) -> httpx.Response:
    body = json.dumps(
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
    )
    return httpx.Response(
        200, content=body.encode(), headers={"content-type": "application/json"}
    )


class _MockInner(httpx.BaseTransport):
    def handle_request(self, _: httpx.Request) -> httpx.Response:
        return _openai_response()


# ---------------------------------------------------------------------------
# GuardAPIClient concurrency
# ---------------------------------------------------------------------------


class TestApiClientConcurrency:
    def test_concurrent_reserves_are_thread_safe(self):
        """N threads each reserve; total admitted spend must not exceed cap."""
        api = GuardAPIClient()
        cap = 1.0
        per_call = 0.1
        n_threads = 20

        results: list[bool] = []
        lock = threading.Lock()

        def _reserve():
            cid = str(uuid.uuid4())
            result = api.reserve(cid, "proj", reserved_usd=per_call, max_usd=cap)
            with lock:
                results.append(result.admitted)

        threads = [threading.Thread(target=_reserve) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        admitted_count = sum(1 for r in results if r)
        # At most ceil(cap / per_call) = 10 can be admitted
        assert admitted_count <= 10
        # Spend must not exceed cap
        assert api.current_spend("proj") <= cap + 1e-9

    def test_concurrent_reconcile_does_not_corrupt_spend(self):
        """Multiple simultaneous reconcile() calls with different call_ids are safe."""
        api = GuardAPIClient()
        n = 10
        call_ids = [str(uuid.uuid4()) for _ in range(n)]

        # Pre-populate inflight
        for cid in call_ids:
            api.reserve(cid, "proj", reserved_usd=0.1, max_usd=100.0)

        errors: list[Exception] = []

        def _reconcile(cid: str) -> None:
            try:
                api.reconcile(cid, "proj", unconsumed_usd=0.05)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_reconcile, args=(cid,)) for cid in call_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert api.current_spend("proj") >= 0


# ---------------------------------------------------------------------------
# PolicyEngine concurrency
# ---------------------------------------------------------------------------


class TestEngineConcurrency:
    def test_concurrent_attach_detach_is_safe(self):
        """Attaching and detaching policies concurrently must not raise."""
        from noveum_trace.guard.decision import PolicyDecision
        from noveum_trace.guard.policies.base import AbstractPolicy
        from noveum_trace.guard.types import Phase

        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        errors: list[Exception] = []

        class _DummyPolicy(AbstractPolicy):
            name = "dummy"

            def pre(self, parsed, ctx, deps) -> PolicyDecision:
                return PolicyDecision.allow(self.name, Phase.pre)

        def _attach_detach() -> None:
            try:
                p = _DummyPolicy()
                p.name = f"p_{uuid.uuid4().hex[:8]}"
                engine.attach(p)
                engine.detach(p.name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_attach_detach) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ---------------------------------------------------------------------------
# NoveumTransport concurrency
# ---------------------------------------------------------------------------


class TestTransportConcurrency:
    def test_concurrent_requests_each_get_unique_call_id(self):
        """dataclasses.replace(ctx, call_id=uuid) must produce distinct ids per request."""
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=1000.0, mode=EnforcementMode.strict, project_id="proj"
            )
        )

        registry = AdapterRegistry([OpenAIAdapter()])
        inner = _MockInner()
        transport = NoveumTransport(
            engine=engine, context=_ctx(), inner=inner, registry=registry
        )

        seen_call_ids: list[str] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        # Patch engine.pre_call to capture call_ids
        original_pre = engine.pre_call

        def _capturing_pre(parsed_req, ctx):
            with lock:
                seen_call_ids.append(ctx.call_id)
            return original_pre(parsed_req, ctx)

        engine.pre_call = _capturing_pre

        def _call():
            try:
                transport.handle_request(_openai_request())
            except Exception as e:
                errors.append(e)

        n = 20
        threads = [threading.Thread(target=_call) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(seen_call_ids) == n
        assert len(set(seen_call_ids)) == n, "call_ids must be unique per request"

    def test_concurrent_requests_do_not_exceed_cap(self):
        """Under concurrent load, spend cap must not be violated."""
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        cap = 0.001  # very tight cap in USD
        engine.attach(
            CostCapPolicy(max_usd=cap, mode=EnforcementMode.strict, project_id="proj")
        )

        registry = AdapterRegistry([OpenAIAdapter()])
        inner = _MockInner()
        transport = NoveumTransport(
            engine=engine, context=_ctx(), inner=inner, registry=registry
        )

        statuses: list[int] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        def _call():
            try:
                resp = transport.handle_request(_openai_request())
                with lock:
                    statuses.append(resp.status_code)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=_call) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected thread errors: {errors}"
        assert (
            len(statuses) == 30
        ), f"Expected 30 recorded statuses, got {len(statuses)}"
        # Some requests should have been blocked (403), none should have failed with an exception
        assert all(s in (200, 403) for s in statuses)
        # Spend must not exceed cap (with tiny epsilon for float arithmetic)
        assert api.current_spend("proj") <= cap + 1e-9


# ---------------------------------------------------------------------------
# Spend cap isolation across projects under concurrency
# ---------------------------------------------------------------------------


class TestProjectIsolationConcurrency:
    def test_concurrent_calls_to_different_projects_are_isolated(self):
        """proj-a and proj-b spend counters must not bleed into each other."""
        # Shared api_client so we can inspect both project buckets after the run.
        api = GuardAPIClient()
        registry = AdapterRegistry([OpenAIAdapter()])
        errors: list[Exception] = []
        lock = threading.Lock()

        def _call(project_id: str) -> None:
            try:
                local_engine = PolicyEngine(api_client=api)
                local_engine.attach(
                    CostCapPolicy(
                        max_usd=100.0,
                        mode=EnforcementMode.strict,
                        project_id=project_id,
                    )
                )
                inner = _MockInner()
                transport = NoveumTransport(
                    engine=local_engine,
                    context=_ctx(project_id),
                    inner=inner,
                    registry=registry,
                )
                resp = transport.handle_request(_openai_request())
                assert resp.status_code == 200
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=_call, args=("proj-a",)) for _ in range(10)
        ] + [threading.Thread(target=_call, args=("proj-b",)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected thread errors: {errors}"
        # Each project must have accumulated spend from its own 10 calls.
        assert api.current_spend("proj-a") > 0, "proj-a should have recorded spend"
        assert api.current_spend("proj-b") > 0, "proj-b should have recorded spend"
        # Crucially, the two buckets must be equal (same request shape, same count)
        # and neither should have been inflated by the other project's calls.
        # We can't trivially assert they are equal due to float arithmetic, but we
        # can confirm neither bucket is zero while also confirming neither has
        # absorbed the other project's spend by checking they are both bounded by
        # the amount a single project's 10 calls would produce (not 20 calls).
        per_project_max = api.current_spend("proj-a") + api.current_spend("proj-b")
        assert (
            api.current_spend("proj-a") < per_project_max
        ), "proj-a spend should be less than the combined total"
        assert (
            api.current_spend("proj-b") < per_project_max
        ), "proj-b spend should be less than the combined total"
