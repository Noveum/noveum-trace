"""Unit tests for NoveumTransport and NoveumAsyncTransport.

Scenarios covered:
  - pre-block  → synthetic 403 returned, inner transport never called
  - allow      → inner transport called, real response returned
  - post-block → inner transport called, synthetic 403 returned after response
  - error      → inner raises, release_all() called, exception re-raised
  - no adapter → request passed through to inner unchanged
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import httpx
import pytest

from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.transport.async_transport import NoveumAsyncTransport
from noveum_trace.guard.transport.sync_transport import NoveumTransport
from noveum_trace.guard.types import ParsedRequest, ParsedResponse, Phase, PolicyContext

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


def _parsed_req() -> ParsedRequest:
    return ParsedRequest(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        max_tokens=100,
        estimated_input_tokens=10,
        raw_body=b"{}",
    )


def _parsed_resp() -> ParsedResponse:
    return ParsedResponse(
        model="gpt-4o",
        text="hello",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.0001,
    )


def _real_request() -> httpx.Request:
    body = json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    )
    return httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=body.encode(),
    )


def _real_response() -> httpx.Response:
    body = json.dumps(
        {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        }
    )
    return httpx.Response(
        200, content=body.encode(), headers={"content-type": "application/json"}
    )


def _block_response() -> httpx.Response:
    return httpx.Response(403, content=b'{"error":"blocked"}')


def _unread_response() -> httpx.Response:
    """A response backed by an unread byte stream, as a real inner transport returns.

    Accessing .content on this before .read()/.aread() raises httpx.ResponseNotRead —
    this is the shape that pre-materialized `content=` mocks fail to exercise.
    """
    body = json.dumps(
        {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        }
    ).encode()
    return httpx.Response(
        200, stream=httpx.ByteStream(body), headers={"content-type": "application/json"}
    )


def _streaming_parsed_req() -> ParsedRequest:
    return ParsedRequest(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        max_tokens=100,
        estimated_input_tokens=10,
        raw_body=b"{}",
    )


def _content_reading_adapter():
    """Stub adapter whose parse_response actually touches resp.content.

    Reproduces C1: if the transport hasn't read the stream, this raises
    httpx.ResponseNotRead instead of returning a parsed response.
    """
    adapter = MagicMock()
    adapter.parse_request.return_value = _parsed_req()

    def _parse_response(req, resp):
        json.loads(resp.content)  # raises ResponseNotRead if stream not materialized
        return _parsed_resp()

    adapter.parse_response.side_effect = _parse_response
    adapter.synthetic_block_response.return_value = _block_response()
    return adapter


class _MockInner(httpx.BaseTransport):
    def __init__(
        self, response: httpx.Response | None = None, raise_exc: Exception | None = None
    ) -> None:
        self._response = response
        self._raise = raise_exc
        self.call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        if self._raise:
            raise self._raise
        assert self._response is not None
        return self._response


class _MockAsyncInner(httpx.AsyncBaseTransport):
    def __init__(
        self, response: httpx.Response | None = None, raise_exc: Exception | None = None
    ) -> None:
        self._response = response
        self._raise = raise_exc
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        if self._raise:
            raise self._raise
        assert self._response is not None
        return self._response


def _make_transport(engine, inner, adapter=None):
    registry = MagicMock()
    if adapter is None:
        registry.for_request.return_value = None
    else:
        registry.for_request.return_value = adapter
    return (
        NoveumTransport(engine=engine, context=_ctx(), inner=inner, registry=registry),
        registry,
    )


def _make_async_transport(engine, inner, adapter=None):
    registry = MagicMock()
    if adapter is None:
        registry.for_request.return_value = None
    else:
        registry.for_request.return_value = adapter
    return (
        NoveumAsyncTransport(
            engine=engine, context=_ctx(), inner=inner, registry=registry
        ),
        registry,
    )


def _stub_adapter(block_resp=None):
    adapter = MagicMock()
    adapter.parse_request.return_value = _parsed_req()
    adapter.parse_response.return_value = _parsed_resp()
    adapter.synthetic_block_response.return_value = block_resp or _block_response()
    return adapter


# ---------------------------------------------------------------------------
# Sync transport — pre-block
# ---------------------------------------------------------------------------


class TestPreBlock:
    def test_returns_403_without_calling_inner(self):
        engine = MagicMock()
        block = PolicyDecision.block("cost_cap", Phase.pre, reason="over budget")
        engine.pre_call.return_value = (block, [])

        adapter = _stub_adapter(_block_response())
        inner = _MockInner(_real_response())
        transport, _ = _make_transport(engine, inner, adapter)

        resp = transport.handle_request(_real_request())

        assert resp.status_code == 403
        assert inner.call_count == 0

    def test_engine_release_all_not_called_on_pre_block(self):
        """Engine already handled rollback internally before returning the block decision."""
        engine = MagicMock()
        block = PolicyDecision.block("cost_cap", Phase.pre, reason="over budget")
        engine.pre_call.return_value = (block, [])

        adapter = _stub_adapter()
        inner = _MockInner(_real_response())
        transport, _ = _make_transport(engine, inner, adapter)

        transport.handle_request(_real_request())

        engine.release_all.assert_not_called()

    def test_block_response_uses_decision_mode(self):
        from noveum_trace.guard.types import BlockResponseMode

        engine = MagicMock()
        block = PolicyDecision.block(
            "cost_cap",
            Phase.pre,
            reason="over budget",
            mode=BlockResponseMode.synthetic_success,
        )
        engine.pre_call.return_value = (block, [])

        adapter = _stub_adapter()
        inner = _MockInner()
        transport, _ = _make_transport(engine, inner, adapter)

        transport.handle_request(_real_request())

        _, call_kwargs = adapter.synthetic_block_response.call_args
        positional = adapter.synthetic_block_response.call_args.args
        # mode is the third positional arg
        assert positional[2] is BlockResponseMode.synthetic_success


# ---------------------------------------------------------------------------
# Sync transport — allow
# ---------------------------------------------------------------------------


class TestAllow:
    def test_inner_transport_is_called(self):
        engine = MagicMock()
        engine.pre_call.return_value = (None, [])
        engine.post_call.return_value = None

        adapter = _stub_adapter()
        real_resp = _real_response()
        inner = _MockInner(real_resp)
        transport, _ = _make_transport(engine, inner, adapter)

        transport.handle_request(_real_request())

        assert inner.call_count == 1

    def test_real_response_returned_to_caller(self):
        engine = MagicMock()
        engine.pre_call.return_value = (None, [])
        engine.post_call.return_value = None

        adapter = _stub_adapter()
        real_resp = _real_response()
        inner = _MockInner(real_resp)
        transport, _ = _make_transport(engine, inner, adapter)

        result = transport.handle_request(_real_request())

        assert result.status_code == 200

    def test_post_call_invoked_with_parsed_response(self):
        engine = MagicMock()
        ran = [MagicMock()]
        engine.pre_call.return_value = (None, ran)
        engine.post_call.return_value = None

        adapter = _stub_adapter()
        inner = _MockInner(_real_response())
        transport, _ = _make_transport(engine, inner, adapter)

        transport.handle_request(_real_request())

        engine.post_call.assert_called_once()
        args = engine.post_call.call_args.args
        assert args[2] is ran  # ran list threaded through


# ---------------------------------------------------------------------------
# Sync transport — post-block
# ---------------------------------------------------------------------------


class TestPostBlock:
    def test_post_block_returns_403(self):
        engine = MagicMock()
        post_block = PolicyDecision.block(
            "spend_limit", Phase.post, reason="spend exceeded"
        )
        engine.pre_call.return_value = (None, [])
        engine.post_call.return_value = post_block

        adapter = _stub_adapter(_block_response())
        inner = _MockInner(_real_response())
        transport, _ = _make_transport(engine, inner, adapter)

        result = transport.handle_request(_real_request())

        assert result.status_code == 403

    def test_inner_transport_was_still_called(self):
        """Inner transport always runs on post-block — block decision comes after the LLM responds."""
        engine = MagicMock()
        post_block = PolicyDecision.block(
            "spend_limit", Phase.post, reason="spend exceeded"
        )
        engine.pre_call.return_value = (None, [])
        engine.post_call.return_value = post_block

        adapter = _stub_adapter()
        inner = _MockInner(_real_response())
        transport, _ = _make_transport(engine, inner, adapter)

        transport.handle_request(_real_request())

        assert inner.call_count == 1


# ---------------------------------------------------------------------------
# Sync transport — error during forward
# ---------------------------------------------------------------------------


class TestErrorDuringForward:
    def test_release_all_called_on_inner_exception(self):
        engine = MagicMock()
        ran = [MagicMock()]
        engine.pre_call.return_value = (None, ran)

        adapter = _stub_adapter()
        inner = _MockInner(raise_exc=ConnectionError("timeout"))
        transport, _ = _make_transport(engine, inner, adapter)

        with pytest.raises(ConnectionError):
            transport.handle_request(_real_request())

        engine.release_all.assert_called_once()
        assert engine.release_all.call_args.args[1] is ran

    def test_exception_is_reraised(self):
        engine = MagicMock()
        engine.pre_call.return_value = (None, [])

        adapter = _stub_adapter()
        inner = _MockInner(raise_exc=RuntimeError("network error"))
        transport, _ = _make_transport(engine, inner, adapter)

        with pytest.raises(RuntimeError, match="network error"):
            transport.handle_request(_real_request())


# ---------------------------------------------------------------------------
# Sync transport — no adapter (unknown provider)
# ---------------------------------------------------------------------------


class TestNoAdapter:
    def test_unknown_request_passes_through(self):
        engine = MagicMock()
        real_resp = _real_response()
        inner = _MockInner(real_resp)
        transport, _ = _make_transport(engine, inner, adapter=None)

        result = transport.handle_request(_real_request())

        assert result.status_code == 200
        assert inner.call_count == 1
        engine.pre_call.assert_not_called()

    def test_engine_never_invoked_for_unknown_request(self):
        engine = MagicMock()
        inner = _MockInner(_real_response())
        transport, _ = _make_transport(engine, inner, adapter=None)

        transport.handle_request(_real_request())

        engine.pre_call.assert_not_called()
        engine.post_call.assert_not_called()


# ---------------------------------------------------------------------------
# Sync transport — unread stream body (C1) and streaming pass-through (C2)
# ---------------------------------------------------------------------------


class TestResponseReading:
    def test_unread_stream_body_is_materialized_before_parse(self):
        """Real inner transports return an unread stream; the transport must
        read() it before parse_response touches .content."""
        engine = MagicMock()
        engine.pre_call.return_value = (None, [])
        engine.post_call.return_value = None

        adapter = _content_reading_adapter()
        inner = _MockInner(_unread_response())
        transport, _ = _make_transport(engine, inner, adapter)

        # Must not raise httpx.ResponseNotRead.
        result = transport.handle_request(_real_request())

        assert result.status_code == 200
        # Caller can still read the body (read() caches it).
        assert json.loads(result.content)["choices"][0]["message"]["content"] == "hello"

    def test_streaming_request_passes_response_through_untouched(self):
        """stream=True → response returned without reading/parsing/post_call so
        the caller's stream is preserved."""
        engine = MagicMock()
        engine.pre_call.return_value = (None, [])
        engine.post_call.return_value = None

        adapter = _content_reading_adapter()
        adapter.parse_request.return_value = _streaming_parsed_req()
        inner = _MockInner(_unread_response())
        transport, _ = _make_transport(engine, inner, adapter)

        result = transport.handle_request(_real_request())

        assert inner.call_count == 1
        adapter.parse_response.assert_not_called()
        engine.post_call.assert_not_called()
        # The unread stream is intact for the caller to consume.
        assert json.loads(result.read())["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Async transport
# ---------------------------------------------------------------------------


class TestAsyncTransport:
    @pytest.mark.asyncio
    async def test_pre_block_returns_403_without_calling_inner(self):
        engine = MagicMock()
        block = PolicyDecision.block("cost_cap", Phase.pre, reason="over budget")
        engine.pre_call.return_value = (block, [])

        adapter = _stub_adapter(_block_response())
        inner = _MockAsyncInner(_real_response())
        transport, _ = _make_async_transport(engine, inner, adapter)

        resp = await transport.handle_async_request(_real_request())

        assert resp.status_code == 403
        assert inner.call_count == 0

    @pytest.mark.asyncio
    async def test_allow_calls_inner_and_returns_real_response(self):
        engine = MagicMock()
        engine.pre_call.return_value = (None, [])
        engine.post_call.return_value = None

        adapter = _stub_adapter()
        inner = _MockAsyncInner(_real_response())
        transport, _ = _make_async_transport(engine, inner, adapter)

        result = await transport.handle_async_request(_real_request())

        assert result.status_code == 200
        assert inner.call_count == 1

    @pytest.mark.asyncio
    async def test_release_all_called_on_inner_exception(self):
        engine = MagicMock()
        ran = [MagicMock()]
        engine.pre_call.return_value = (None, ran)

        adapter = _stub_adapter()
        inner = _MockAsyncInner(raise_exc=ConnectionError("timeout"))
        transport, _ = _make_async_transport(engine, inner, adapter)

        with pytest.raises(ConnectionError):
            await transport.handle_async_request(_real_request())

        engine.release_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_adapter_passes_through(self):
        engine = MagicMock()
        inner = _MockAsyncInner(_real_response())
        transport, _ = _make_async_transport(engine, inner, adapter=None)

        result = await transport.handle_async_request(_real_request())

        assert result.status_code == 200
        engine.pre_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_unread_stream_body_is_materialized_before_parse(self):
        engine = MagicMock()
        engine.pre_call.return_value = (None, [])
        engine.post_call.return_value = None

        adapter = _content_reading_adapter()
        # parse_response touches resp.content; the transport must aread() first.
        inner = _MockAsyncInner(_unread_response())
        transport, _ = _make_async_transport(engine, inner, adapter)

        result = await transport.handle_async_request(_real_request())

        assert result.status_code == 200
        assert json.loads(result.content)["choices"][0]["message"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_streaming_request_passes_response_through_untouched(self):
        engine = MagicMock()
        engine.pre_call.return_value = (None, [])
        engine.post_call.return_value = None

        adapter = _content_reading_adapter()
        adapter.parse_request.return_value = _streaming_parsed_req()
        inner = _MockAsyncInner(_unread_response())
        transport, _ = _make_async_transport(engine, inner, adapter)

        result = await transport.handle_async_request(_real_request())

        assert inner.call_count == 1
        adapter.parse_response.assert_not_called()
        engine.post_call.assert_not_called()
        assert json.loads(await result.aread())["model"] == "gpt-4o"
