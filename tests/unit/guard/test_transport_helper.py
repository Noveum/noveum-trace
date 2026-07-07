"""Unit tests for noveum_trace.guard.transport.helper.

Covers:
  - _parse_sse_event_data: usage + assembled text extraction (OpenAI, Anthropic)
  - reconcile_stream: lazy-streaming reconciliation, including gzip bodies
  - build_generic_block_response
"""

from __future__ import annotations

import gzip
import json
import uuid

import httpx

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.policies.base import AbstractPolicy
from noveum_trace.guard.transport.adapters.openai_adapter import OpenAIAdapter
from noveum_trace.guard.transport.helper import (
    _parse_sse_event_data,
    build_buffered_stream_response,
    build_generic_block_response,
    reconcile_stream,
)
from noveum_trace.guard.types import ParsedRequest, Phase, PolicyContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> PolicyContext:
    return PolicyContext(
        project_id="proj",
        organization_id=None,
        environment="test",
        trace_id=None,
        span_id=None,
        call_id=str(uuid.uuid4()),
    )


def _streaming_req(provider: str = "openai", model: str = "gpt-4o") -> ParsedRequest:
    return ParsedRequest(
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        max_tokens=100,
        estimated_input_tokens=10,
        raw_body=b"{}",
    )


def _openai_sse(*, include_usage: bool = True) -> bytes:
    events = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]
    if include_usage:
        events.append(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _anthropic_sse() -> bytes:
    events = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 8}},
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hi"},
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": " there"},
        },
        {
            "type": "message_delta",
            "usage": {"output_tokens": 4},
        },
    ]
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    return "".join(lines).encode()


# ---------------------------------------------------------------------------
# _parse_sse_event_data
# ---------------------------------------------------------------------------


class TestParseSSEEventData:
    def test_openai_assembles_text_across_chunks(self):
        result = _parse_sse_event_data(_openai_sse(), "openai")
        assert result.text == "Hello world"

    def test_openai_extracts_usage(self):
        result = _parse_sse_event_data(_openai_sse(), "openai")
        assert result.has_usage is True
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_openai_no_usage_event_has_usage_false(self):
        result = _parse_sse_event_data(_openai_sse(include_usage=False), "openai")
        assert result.has_usage is False
        # Text is still assembled even without a usage event.
        assert result.text == "Hello world"

    def test_anthropic_assembles_text_across_content_block_deltas(self):
        result = _parse_sse_event_data(_anthropic_sse(), "anthropic")
        assert result.text == "Hi there"

    def test_anthropic_extracts_usage_from_start_and_delta_events(self):
        result = _parse_sse_event_data(_anthropic_sse(), "anthropic")
        assert result.has_usage is True
        assert result.input_tokens == 8
        assert result.output_tokens == 4

    def test_empty_body_has_no_usage_and_no_text(self):
        result = _parse_sse_event_data(b"", "openai")
        assert result.has_usage is False
        assert result.text is None


# ---------------------------------------------------------------------------
# reconcile_stream
# ---------------------------------------------------------------------------


class _AllowAllPolicy(AbstractPolicy):
    name = "allow_all"

    def pre(self, parsed, ctx, deps):
        from noveum_trace.guard.decision import PolicyDecision

        return PolicyDecision.allow(self.name, Phase.pre)


class TestReconcileStream:
    def _engine_and_ran(self):
        engine = PolicyEngine(api_client=GuardAPIClient())
        policy = _AllowAllPolicy()
        engine.attach(policy)
        ctx = _ctx()
        _, ran = engine.pre_call(_streaming_req(), ctx)
        return engine, ctx, ran

    def test_reconcile_with_usage_calls_post_call(self):
        engine, ctx, ran = self._engine_and_ran()
        chunks = [_openai_sse()]

        # Should not raise; post_call() runs to completion with real usage.
        reconcile_stream(chunks, engine, ctx, ran, _streaming_req())

    def test_reconcile_without_usage_releases_all(self):
        engine, ctx, ran = self._engine_and_ran()
        chunks = [_openai_sse(include_usage=False)]

        # No usage event → falls back to release_all(), must not raise.
        reconcile_stream(chunks, engine, ctx, ran, _streaming_req())

    def test_reconcile_handles_gzip_compressed_body(self):
        engine, ctx, ran = self._engine_and_ran()
        compressed = gzip.compress(_openai_sse())

        # Must decompress before SSE parsing; must not raise.
        reconcile_stream([compressed], engine, ctx, ran, _streaming_req())

    def test_reconcile_swallows_unexpected_exceptions_via_release_all(self):
        engine, ctx, ran = self._engine_and_ran()
        # Malformed bytes that will fail to decode/parse meaningfully.
        chunks = [b"\xff\xfe not valid sse at all"]

        # Must not raise even on garbage input.
        reconcile_stream(chunks, engine, ctx, ran, _streaming_req())


# ---------------------------------------------------------------------------
# build_generic_block_response
# ---------------------------------------------------------------------------


class TestBuildGenericBlockResponse:
    def test_returns_403(self):
        resp = build_generic_block_response("no adapter matched")
        assert resp.status_code == 403

    def test_body_contains_reason(self):
        resp = build_generic_block_response("no adapter matched")
        body = json.loads(resp.content)
        assert body["error"]["message"] == "no adapter matched"
        assert body["error"]["type"] == "policy_blocked"


# ---------------------------------------------------------------------------
# build_buffered_stream_response
# ---------------------------------------------------------------------------


class _BlockInPostPolicy(AbstractPolicy):
    name = "block_post"
    can_block_post = True

    def post(self, resp, ctx, decision, deps):
        return PolicyDecision.block(self.name, Phase.post, reason="blocked in post")


def _read_inner_response(sse_bytes: bytes, *, gzip_encoded: bool = False):
    """Build a response as the inner transport would return it, then read()
    it — mirroring what NoveumTransport does before build_buffered_stream_response.

    read() applies content-decoding, so response.content ends up decoded while
    the original Content-Encoding/Content-Length headers still describe the
    compressed body.
    """
    headers = {"content-type": "text/event-stream"}
    if gzip_encoded:
        body = gzip.compress(sse_bytes)
        headers["content-encoding"] = "gzip"
    else:
        body = sse_bytes
    headers["content-length"] = str(len(body))
    resp = httpx.Response(200, headers=headers, content=body)
    resp.read()
    return resp


class TestBuildBufferedStreamResponse:
    def _engine_ctx_ran(self, policy: AbstractPolicy):
        engine = PolicyEngine(api_client=GuardAPIClient())
        engine.attach(policy)
        ctx = _ctx()
        _, ran = engine.pre_call(_streaming_req(), ctx)
        return engine, ctx, ran

    def test_allow_path_replays_decoded_sse(self):
        engine, ctx, ran = self._engine_ctx_ran(_AllowAllPolicy())
        sse = _openai_sse()
        inner = _read_inner_response(sse)
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

        out = build_buffered_stream_response(
            inner, request, OpenAIAdapter(), engine, ctx, ran, _streaming_req()
        )
        out.read()
        assert out.status_code == 200
        assert out.content == sse

    def test_gzip_body_replays_without_double_decode(self):
        """Regression (Finding 2): response.content is already decoded, so the
        replay must drop Content-Encoding — otherwise the caller's httpx read
        attempts a second gunzip and raises DecodingError.
        """
        engine, ctx, ran = self._engine_ctx_ran(_AllowAllPolicy())
        sse = _openai_sse()
        inner = _read_inner_response(sse, gzip_encoded=True)
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

        out = build_buffered_stream_response(
            inner, request, OpenAIAdapter(), engine, ctx, ran, _streaming_req()
        )
        # Must not raise DecodingError, and must yield the decoded SSE bytes.
        out.read()
        assert out.content == sse
        # The stale gzip encoding header must be gone.
        assert "content-encoding" not in out.headers
        # httpx regenerates Content-Length from the (decoded) replay body, so
        # it now matches the real content instead of the compressed length.
        assert out.headers["content-length"] == str(len(sse))

    def test_post_block_returns_synthetic_block(self):
        engine, ctx, ran = self._engine_ctx_ran(_BlockInPostPolicy())
        inner = _read_inner_response(_openai_sse())
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

        out = build_buffered_stream_response(
            inner, request, OpenAIAdapter(), engine, ctx, ran, _streaming_req()
        )
        assert out.status_code == 403
        body = json.loads(out.content)
        assert body["error"]["type"] == "policy_blocked"
