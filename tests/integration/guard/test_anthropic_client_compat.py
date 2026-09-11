"""Exercise SDK-native clients against Guard without credentials or network I/O."""

import json

import httpx
import pytest

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.policies.cost_cap import CostCapPolicy
from noveum_trace.guard.policies.rate_limit import RateLimitPolicy
from noveum_trace.guard.types import PolicyContext

anthropic = pytest.importorskip("anthropic")
from tests.integration.guard.anthropic_client import (  # noqa: E402
    async_http_client,
    http_client,
)

MODEL = "claude-haiku-4-5-20251001"
ARGS = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16,
}


@pytest.fixture
def stack():
    api = GuardAPIClient()
    engine = PolicyEngine(api_client=api)
    engine.attach(CostCapPolicy(max_usd=1.0, project_id="compat"))
    engine.attach(
        RateLimitPolicy(
            windows=[{"period": "1m", "maxRequests": 1}], project_id="compat"
        )
    )
    context = PolicyContext(
        project_id="compat",
        organization_id=None,
        environment="test",
        trace_id=None,
        span_id=None,
        call_id="compat",
    )
    calls = []

    def respond(request):
        calls.append(request)
        message = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
        if not json.loads(request.content).get("stream"):
            return httpx.Response(200, json=message)
        events = [
            {
                "type": "message_start",
                "message": {
                    **message,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 2, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "done"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        body = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        )
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    return engine, context, api, httpx.MockTransport(respond), calls


def _assert_guard(api, calls):
    assert len(calls) == 1  # The second request was blocked before the provider.
    assert api.current_rate("compat")["requests_1m"] == 1
    assert 0 < api.current_spend("compat") < 1
    assert api.inflight_count() == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_sync_native_client_preserves_guard(stack, streaming):
    engine, context, api, inner, calls = stack
    with anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=http_client(engine, context, inner=inner),
    ) as client:
        if streaming:
            with client.messages.stream(**ARGS) as stream:
                assert "".join(stream.text_stream) == "done"
        else:
            assert client.messages.create(**ARGS).content[0].text == "done"
        with pytest.raises(anthropic.PermissionDeniedError):
            client.messages.create(**ARGS)
    _assert_guard(api, calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_native_client_preserves_guard(stack, streaming):
    engine, context, api, inner, calls = stack
    async with anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=async_http_client(engine, context, inner=inner),
    ) as client:
        if streaming:
            async with client.messages.stream(**ARGS) as stream:
                assert "".join([text async for text in stream.text_stream]) == "done"
        else:
            assert (await client.messages.create(**ARGS)).content[0].text == "done"
        with pytest.raises(anthropic.PermissionDeniedError):
            await client.messages.create(**ARGS)
    _assert_guard(api, calls)
