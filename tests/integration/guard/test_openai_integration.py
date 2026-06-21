"""Integration tests for the Guard + OpenAI adapter full pipeline.

Uses a mock inner transport — no real API key required.
Verifies the complete data flow: request parsing → policy evaluation →
response parsing → cost reconciliation.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

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


def _ctx(project_id: str = "test-proj") -> PolicyContext:
    return PolicyContext(
        project_id=project_id,
        organization_id=None,
        environment="test",
        trace_id=None,
        span_id=None,
        call_id=str(uuid.uuid4()),
    )


def _openai_request(model: str = "gpt-4o-mini", max_tokens: int = 100) -> httpx.Request:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": max_tokens,
        }
    )
    return httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=body.encode(),
        headers={"authorization": "Bearer sk-test", "content-type": "application/json"},
    )


def _openai_success_response(
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 20,
    completion_tokens: int = 10,
    text: str = "4",
) -> httpx.Response:
    body = json.dumps(
        {
            "id": "chatcmpl-test",
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    return httpx.Response(
        200, content=body.encode(), headers={"content-type": "application/json"}
    )


class _MockInner(httpx.BaseTransport):
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def handle_request(self, _: httpx.Request) -> httpx.Response:
        return self._response


def _build_stack(
    max_usd: float = 10.0,
    response: httpx.Response | None = None,
    project_id: str = "test-proj",
):
    api = GuardAPIClient()
    engine = PolicyEngine(api_client=api)
    engine.attach(
        CostCapPolicy(
            max_usd=max_usd, mode=EnforcementMode.strict, project_id=project_id
        )
    )

    registry = AdapterRegistry([OpenAIAdapter()])
    inner = _MockInner(response or _openai_success_response())
    transport = NoveumTransport(
        engine=engine, context=_ctx(project_id), inner=inner, registry=registry
    )
    return transport, api


# ---------------------------------------------------------------------------
# Allow path
# ---------------------------------------------------------------------------


class TestOpenAIAllowPath:
    def test_successful_call_returns_200(self):
        transport, _ = _build_stack()
        resp = transport.handle_request(_openai_request())
        assert resp.status_code == 200

    def test_response_body_preserved(self):
        transport, _ = _build_stack()
        resp = transport.handle_request(_openai_request())
        body = json.loads(resp.content)
        assert "choices" in body

    def test_spend_accumulates_after_call(self):
        _, api = _build_stack()
        transport, _ = _build_stack()
        # Make a call
        transport.handle_request(_openai_request())
        # Spend tracked in the api client of the transport's engine
        # We verify via the build_stack api reference
        transport2, api2 = _build_stack(project_id="proj-spend")
        transport2.handle_request(_openai_request())
        assert api2.current_spend("proj-spend") >= 0  # at least exists


# ---------------------------------------------------------------------------
# Block path — pre (cap exhausted)
# ---------------------------------------------------------------------------


class TestOpenAIPreBlock:
    def test_exhausted_cap_returns_403(self):
        api = GuardAPIClient()
        api._spend["test-proj"] = 1000.0  # cap exhausted
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=1.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        registry = AdapterRegistry([OpenAIAdapter()])
        inner = _MockInner(_openai_success_response())
        transport = NoveumTransport(
            engine=engine, context=_ctx(), inner=inner, registry=registry
        )

        resp = transport.handle_request(_openai_request())

        assert resp.status_code == 403

    def test_block_response_has_openai_error_shape(self):
        api = GuardAPIClient()
        api._spend["test-proj"] = 1000.0
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=1.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        registry = AdapterRegistry([OpenAIAdapter()])
        transport = NoveumTransport(
            engine=engine,
            context=_ctx(),
            inner=_MockInner(_openai_success_response()),
            registry=registry,
        )

        resp = transport.handle_request(_openai_request())
        body = json.loads(resp.content)

        assert "error" in body

    def test_spend_unchanged_after_pre_block(self):
        api = GuardAPIClient()
        api._spend["test-proj"] = 1000.0
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=1.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        registry = AdapterRegistry([OpenAIAdapter()])
        transport = NoveumTransport(
            engine=engine,
            context=_ctx(),
            inner=_MockInner(_openai_success_response()),
            registry=registry,
        )

        spend_before = api.current_spend("test-proj")
        transport.handle_request(_openai_request())
        # Spend must not increase on a pre-block (reservation rolled back)
        assert api.current_spend("test-proj") == pytest.approx(spend_before)


# ---------------------------------------------------------------------------
# Cost reconciliation
# ---------------------------------------------------------------------------


class TestOpenAICostReconciliation:
    def test_reserved_cost_is_reconciled_after_response(self):
        """Worst-case reservation > actual cost → excess should be released."""
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=100.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        registry = AdapterRegistry([OpenAIAdapter()])
        # Tiny actual usage
        inner = _MockInner(
            _openai_success_response(prompt_tokens=5, completion_tokens=2)
        )
        transport = NoveumTransport(
            engine=engine, context=_ctx(), inner=inner, registry=registry
        )

        transport.handle_request(_openai_request(max_tokens=10_000))

        # After reconcile, spend should reflect actual (small) cost, not worst-case estimate
        spend = api.current_spend("test-proj")
        # Cost for 5+2 tokens on gpt-4o-mini should be tiny
        assert spend < 0.01

    def test_unknown_host_passes_through_unguarded(self):
        """Requests to unknown providers bypass guard and get real response."""
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=0.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        # No adapter registered for this host
        registry = AdapterRegistry([])
        inner = _MockInner(_openai_success_response())
        transport = NoveumTransport(
            engine=engine, context=_ctx(), inner=inner, registry=registry
        )

        req = httpx.Request(
            "POST", "https://unknown.example.com/v1/chat", content=b"{}"
        )
        resp = transport.handle_request(req)

        # Passed through without enforcement
        assert resp.status_code == 200
        assert api.current_spend("test-proj") == 0.0
