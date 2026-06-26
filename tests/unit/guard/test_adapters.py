"""Unit tests for ProviderAdapters and AdapterRegistry.

Covers:
  - OpenAIAdapter: parse_request, parse_response, synthetic_block_response
  - AnthropicAdapter: same, plus system-prompt prepend
  - AdapterRegistry: host match, body-shape specificity fallback, no match
"""

from __future__ import annotations

import json

import httpx

from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.transport.adapters.anthropic_adapter import AnthropicAdapter
from noveum_trace.guard.transport.adapters.base import AdapterRegistry
from noveum_trace.guard.transport.adapters.openai_adapter import OpenAIAdapter
from noveum_trace.guard.types import Phase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openai_request(body: dict | None = None) -> httpx.Request:
    payload = body or {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 256,
    }
    return httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )


def _anthropic_request(body: dict | None = None) -> httpx.Request:
    payload = body or {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 256,
    }
    return httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )


def _openai_response_body(**overrides) -> bytes:
    body = {
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        **overrides,
    }
    return json.dumps(body).encode()


def _anthropic_response_body(**overrides) -> bytes:
    body = {
        "model": "claude-sonnet-4-6",
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "content": [{"type": "text", "text": "hi there"}],
        **overrides,
    }
    return json.dumps(body).encode()


def _block_decision(reason: str = "blocked") -> PolicyDecision:
    return PolicyDecision.block("test_policy", Phase.pre, reason=reason)


# ---------------------------------------------------------------------------
# OpenAIAdapter — parse_request
# ---------------------------------------------------------------------------


class TestOpenAIAdapterParseRequest:
    def test_extracts_model(self):
        adapter = OpenAIAdapter()
        req = _openai_request({"model": "gpt-4o-mini", "messages": []})
        parsed = adapter.parse_request(req)
        assert parsed.model == "gpt-4o-mini"

    def test_extracts_messages(self):
        adapter = OpenAIAdapter()
        messages = [{"role": "user", "content": "test"}]
        req = _openai_request({"model": "gpt-4o", "messages": messages})
        parsed = adapter.parse_request(req)
        assert parsed.messages == messages

    def test_extracts_max_tokens(self):
        adapter = OpenAIAdapter()
        req = _openai_request({"model": "gpt-4o", "messages": [], "max_tokens": 512})
        parsed = adapter.parse_request(req)
        assert parsed.max_tokens == 512

    def test_max_tokens_none_when_absent(self):
        adapter = OpenAIAdapter()
        req = _openai_request({"model": "gpt-4o", "messages": []})
        parsed = adapter.parse_request(req)
        assert parsed.max_tokens is None

    def test_provider_is_openai(self):
        adapter = OpenAIAdapter()
        parsed = adapter.parse_request(_openai_request())
        assert parsed.provider == "openai"

    def test_estimated_input_tokens_is_positive(self):
        adapter = OpenAIAdapter()
        req = _openai_request(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello world"}],
            }
        )
        parsed = adapter.parse_request(req)
        assert parsed.estimated_input_tokens > 0

    def test_stream_defaults_false(self):
        adapter = OpenAIAdapter()
        req = _openai_request({"model": "gpt-4o", "messages": []})
        parsed = adapter.parse_request(req)
        assert parsed.stream is False

    def test_stream_true_when_set(self):
        adapter = OpenAIAdapter()
        req = _openai_request({"model": "gpt-4o", "messages": [], "stream": True})
        parsed = adapter.parse_request(req)
        assert parsed.stream is True


# ---------------------------------------------------------------------------
# OpenAIAdapter — parse_response
# ---------------------------------------------------------------------------


class TestOpenAIAdapterParseResponse:
    def test_extracts_model(self):
        adapter = OpenAIAdapter()
        resp = httpx.Response(200, content=_openai_response_body(model="gpt-4o-mini"))
        parsed = adapter.parse_response(_openai_request(), resp)
        assert parsed.model == "gpt-4o-mini"

    def test_extracts_input_tokens_from_prompt_tokens(self):
        adapter = OpenAIAdapter()
        body = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 15, "completion_tokens": 5},
        }
        resp = httpx.Response(200, content=json.dumps(body).encode())
        parsed = adapter.parse_response(_openai_request(), resp)
        assert parsed.input_tokens == 15

    def test_extracts_output_tokens_from_completion_tokens(self):
        adapter = OpenAIAdapter()
        body = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 15, "completion_tokens": 5},
        }
        resp = httpx.Response(200, content=json.dumps(body).encode())
        parsed = adapter.parse_response(_openai_request(), resp)
        assert parsed.output_tokens == 5

    def test_cost_usd_is_positive_for_known_model(self):
        adapter = OpenAIAdapter()
        body = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }
        resp = httpx.Response(200, content=json.dumps(body).encode())
        parsed = adapter.parse_response(_openai_request(), resp)
        assert parsed.cost_usd > 0

    def test_extracts_text_from_choices(self):
        adapter = OpenAIAdapter()
        body = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"message": {"content": "hello!"}, "finish_reason": "stop"}],
        }
        resp = httpx.Response(200, content=json.dumps(body).encode())
        parsed = adapter.parse_response(_openai_request(), resp)
        assert parsed.text == "hello!"

    def test_text_is_none_when_no_choices(self):
        adapter = OpenAIAdapter()
        body = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
            "choices": [],
        }
        resp = httpx.Response(200, content=json.dumps(body).encode())
        parsed = adapter.parse_response(_openai_request(), resp)
        assert parsed.text is None


# ---------------------------------------------------------------------------
# OpenAIAdapter — synthetic_block_response
# ---------------------------------------------------------------------------


class TestOpenAIAdapterSyntheticBlockResponse:
    def test_returns_403(self):
        adapter = OpenAIAdapter()
        resp = adapter.synthetic_block_response(_openai_request(), _block_decision())
        assert resp.status_code == 403

    def test_body_contains_error_key(self):
        adapter = OpenAIAdapter()
        resp = adapter.synthetic_block_response(_openai_request(), _block_decision())
        body = json.loads(resp.content)
        assert "error" in body

    def test_block_reason_in_message(self):
        adapter = OpenAIAdapter()
        decision = _block_decision(reason="cap exceeded")
        resp = adapter.synthetic_block_response(_openai_request(), decision)
        body = json.loads(resp.content)
        assert "cap exceeded" in body["error"]["message"]


# ---------------------------------------------------------------------------
# AnthropicAdapter — parse_request
# ---------------------------------------------------------------------------


class TestAnthropicAdapterParseRequest:
    def test_extracts_model(self):
        adapter = AnthropicAdapter()
        req = _anthropic_request(
            {"model": "claude-haiku-4-5", "messages": [], "max_tokens": 100}
        )
        parsed = adapter.parse_request(req)
        assert parsed.model == "claude-haiku-4-5"

    def test_prepends_system_prompt_to_messages(self):
        adapter = AnthropicAdapter()
        req = _anthropic_request(
            {
                "model": "claude-sonnet-4-6",
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            }
        )
        parsed = adapter.parse_request(req)
        assert parsed.messages[0] == {"role": "system", "content": "You are helpful."}
        assert len(parsed.messages) == 2

    def test_no_system_prompt_leaves_messages_unchanged(self):
        adapter = AnthropicAdapter()
        messages = [{"role": "user", "content": "hello"}]
        req = _anthropic_request(
            {"model": "claude-sonnet-4-6", "messages": messages, "max_tokens": 100}
        )
        parsed = adapter.parse_request(req)
        assert parsed.messages == messages

    def test_provider_is_anthropic(self):
        adapter = AnthropicAdapter()
        parsed = adapter.parse_request(_anthropic_request())
        assert parsed.provider == "anthropic"


# ---------------------------------------------------------------------------
# AnthropicAdapter — parse_response
# ---------------------------------------------------------------------------


class TestAnthropicAdapterParseResponse:
    def test_extracts_input_tokens(self):
        adapter = AnthropicAdapter()
        body = {
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 8, "output_tokens": 4},
            "content": [],
        }
        resp = httpx.Response(200, content=json.dumps(body).encode())
        parsed = adapter.parse_response(_anthropic_request(), resp)
        assert parsed.input_tokens == 8

    def test_extracts_output_tokens(self):
        adapter = AnthropicAdapter()
        body = {
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 8, "output_tokens": 4},
            "content": [],
        }
        resp = httpx.Response(200, content=json.dumps(body).encode())
        parsed = adapter.parse_response(_anthropic_request(), resp)
        assert parsed.output_tokens == 4

    def test_extracts_text_content(self):
        adapter = AnthropicAdapter()
        body = {
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 8, "output_tokens": 4},
            "content": [{"type": "text", "text": "Hello there!"}],
        }
        resp = httpx.Response(200, content=json.dumps(body).encode())
        parsed = adapter.parse_response(_anthropic_request(), resp)
        assert parsed.text == "Hello there!"

    def test_text_none_when_no_text_block(self):
        adapter = AnthropicAdapter()
        body = {
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 5, "output_tokens": 2},
            "content": [{"type": "tool_use", "id": "x"}],
        }
        resp = httpx.Response(200, content=json.dumps(body).encode())
        parsed = adapter.parse_response(_anthropic_request(), resp)
        assert parsed.text is None


# ---------------------------------------------------------------------------
# AnthropicAdapter — synthetic_block_response
# ---------------------------------------------------------------------------


class TestAnthropicAdapterSyntheticBlockResponse:
    def test_returns_403(self):
        adapter = AnthropicAdapter()
        resp = adapter.synthetic_block_response(_anthropic_request(), _block_decision())
        assert resp.status_code == 403

    def test_body_has_anthropic_error_shape(self):
        adapter = AnthropicAdapter()
        resp = adapter.synthetic_block_response(_anthropic_request(), _block_decision())
        body = json.loads(resp.content)
        assert body.get("type") == "error"
        assert "error" in body

    def test_block_reason_in_message(self):
        adapter = AnthropicAdapter()
        decision = _block_decision(reason="budget exceeded")
        resp = adapter.synthetic_block_response(_anthropic_request(), decision)
        body = json.loads(resp.content)
        assert "budget exceeded" in body["error"]["message"]


# ---------------------------------------------------------------------------
# AdapterRegistry
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    def test_matches_openai_by_host(self):
        registry = AdapterRegistry([OpenAIAdapter(), AnthropicAdapter()])
        req = _openai_request()
        adapter = registry.for_request(req)
        assert isinstance(adapter, OpenAIAdapter)

    def test_matches_anthropic_by_host(self):
        registry = AdapterRegistry([OpenAIAdapter(), AnthropicAdapter()])
        req = _anthropic_request()
        adapter = registry.for_request(req)
        assert isinstance(adapter, AnthropicAdapter)

    def test_returns_none_for_unknown_host(self):
        registry = AdapterRegistry([OpenAIAdapter(), AnthropicAdapter()])
        req = httpx.Request("POST", "https://unknown-provider.example.com/v1/chat")
        adapter = registry.for_request(req)
        assert adapter is None

    def test_body_shape_fallback_openai(self):
        """A custom base_url with OpenAI-shaped body should match OpenAI."""
        registry = AdapterRegistry([OpenAIAdapter(), AnthropicAdapter()])
        body = json.dumps({"model": "gpt-4o", "messages": []}).encode()
        req = httpx.Request(
            "POST", "https://my-proxy.example.com/v1/chat", content=body
        )
        adapter = registry.for_request(req)
        assert isinstance(adapter, OpenAIAdapter)

    def test_body_shape_fallback_anthropic_requires_header(self):
        """A custom Anthropic endpoint matches only when the anthropic-version
        header is present (the Anthropic SDK always sends it)."""
        registry = AdapterRegistry([OpenAIAdapter(), AnthropicAdapter()])
        body = json.dumps(
            {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            }
        ).encode()
        req = httpx.Request(
            "POST",
            "https://my-proxy.example.com/v1/messages",
            content=body,
            headers={"anthropic-version": "2023-06-01"},
        )
        adapter = registry.for_request(req)
        assert isinstance(adapter, AnthropicAdapter)

    def test_body_shape_fallback_openai_with_max_tokens_not_misrouted(self):
        """An OpenAI body carrying max_tokens (a subset of Anthropic's signature)
        must not be misclassified as Anthropic without the anthropic-version header."""
        registry = AdapterRegistry([OpenAIAdapter(), AnthropicAdapter()])
        body = json.dumps(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            }
        ).encode()
        req = httpx.Request(
            "POST", "https://my-proxy.example.com/v1/chat", content=body
        )
        adapter = registry.for_request(req)
        assert isinstance(adapter, OpenAIAdapter)

    def test_groq_matches_openai_by_host_pattern(self):
        registry = AdapterRegistry([OpenAIAdapter(), AnthropicAdapter()])
        body = json.dumps({"model": "llama3-8b-8192", "messages": []}).encode()
        req = httpx.Request(
            "POST", "https://api.groq.com/openai/v1/chat/completions", content=body
        )
        adapter = registry.for_request(req)
        assert isinstance(adapter, OpenAIAdapter)

    def test_register_adds_adapter(self):
        from unittest.mock import MagicMock

        registry = AdapterRegistry()
        adapter = MagicMock()
        adapter.host_patterns = ["custom.example.com"]
        registry.register(adapter)
        req = httpx.Request("POST", "https://custom.example.com/v1/chat", content=b"{}")
        assert registry.for_request(req) is adapter
