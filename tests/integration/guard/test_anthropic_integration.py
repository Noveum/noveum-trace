"""Integration tests for the Guard + Anthropic adapter full pipeline.

Uses a mock inner transport — no real API key required.
"""

from __future__ import annotations

import json
import uuid

import httpx

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.policies.cost_cap import CostCapPolicy
from noveum_trace.guard.transport.adapters.anthropic_adapter import AnthropicAdapter
from noveum_trace.guard.transport.adapters.base import AdapterRegistry
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


def _anthropic_request(
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 100,
    system: str | None = None,
) -> httpx.Request:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": max_tokens,
    }
    if system:
        payload["system"] = system
    return httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        content=json.dumps(payload).encode(),
        headers={"x-api-key": "sk-test", "content-type": "application/json"},
    )


def _anthropic_success_response(
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 20,
    output_tokens: int = 10,
    text: str = "4",
) -> httpx.Response:
    body = json.dumps(
        {
            "id": "msg_test",
            "type": "message",
            "model": model,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
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

    registry = AdapterRegistry([AnthropicAdapter()])
    inner = _MockInner(response or _anthropic_success_response())
    transport = NoveumTransport(
        engine=engine, context=_ctx(project_id), inner=inner, registry=registry
    )
    return transport, api


# ---------------------------------------------------------------------------
# Allow path
# ---------------------------------------------------------------------------


class TestAnthropicAllowPath:
    def test_successful_call_returns_200(self):
        transport, _ = _build_stack()
        resp = transport.handle_request(_anthropic_request())
        assert resp.status_code == 200

    def test_response_content_preserved(self):
        transport, _ = _build_stack()
        resp = transport.handle_request(_anthropic_request())
        body = json.loads(resp.content)
        assert "content" in body

    def test_system_prompt_is_parsed_correctly(self):
        """System prompt is prepended by AnthropicAdapter — guard should still allow."""
        transport, _ = _build_stack()
        req = _anthropic_request(system="You are a math tutor.")
        resp = transport.handle_request(req)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Block path
# ---------------------------------------------------------------------------


class TestAnthropicPreBlock:
    def test_exhausted_cap_returns_403(self):
        api = GuardAPIClient()
        api._spend["test-proj"] = 1000.0
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=1.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        registry = AdapterRegistry([AnthropicAdapter()])
        transport = NoveumTransport(
            engine=engine,
            context=_ctx(),
            inner=_MockInner(_anthropic_success_response()),
            registry=registry,
        )

        resp = transport.handle_request(_anthropic_request())
        assert resp.status_code == 403

    def test_block_response_has_anthropic_error_shape(self):
        api = GuardAPIClient()
        api._spend["test-proj"] = 1000.0
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=1.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        registry = AdapterRegistry([AnthropicAdapter()])
        transport = NoveumTransport(
            engine=engine,
            context=_ctx(),
            inner=_MockInner(_anthropic_success_response()),
            registry=registry,
        )

        resp = transport.handle_request(_anthropic_request())
        body = json.loads(resp.content)

        assert body.get("type") == "error"
        assert "error" in body


# ---------------------------------------------------------------------------
# Mixed registry — OpenAI and Anthropic requests routed correctly
# ---------------------------------------------------------------------------


class TestMixedRegistry:
    def test_openai_and_anthropic_in_same_registry(self):
        from noveum_trace.guard.transport.adapters.openai_adapter import OpenAIAdapter

        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=100.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        registry = AdapterRegistry([OpenAIAdapter(), AnthropicAdapter()])

        # Anthropic call
        inner_a = _MockInner(_anthropic_success_response())
        transport_a = NoveumTransport(
            engine=engine, context=_ctx(), inner=inner_a, registry=registry
        )
        resp_a = transport_a.handle_request(_anthropic_request())
        assert resp_a.status_code == 200

    def test_spend_tracked_across_providers(self):
        from noveum_trace.guard.transport.adapters.openai_adapter import OpenAIAdapter

        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=100.0, mode=EnforcementMode.strict, project_id="multi-proj"
            )
        )

        registry = AdapterRegistry([OpenAIAdapter(), AnthropicAdapter()])

        # Anthropic call
        inner = _MockInner(
            _anthropic_success_response(input_tokens=100, output_tokens=50)
        )
        transport = NoveumTransport(
            engine=engine, context=_ctx("multi-proj"), inner=inner, registry=registry
        )
        transport.handle_request(_anthropic_request(max_tokens=100))

        spend = api.current_spend("multi-proj")
        assert spend > 0
