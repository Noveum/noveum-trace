from __future__ import annotations

import dataclasses
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Optional

import httpx

from noveum_trace.guard.transport.adapters.base import AdapterRegistry, default_registry
from noveum_trace.guard.types import ParsedRequest, PolicyContext

if TYPE_CHECKING:
    from noveum_trace.guard.decision import PolicyDecision
    from noveum_trace.guard.engine import PolicyEngine
    from noveum_trace.guard.policies.base import AbstractPolicy


class _AsyncStreamReconciler(httpx.AsyncByteStream):
    """Async counterpart of _SyncStreamReconciler.

    Buffers every chunk from the async stream; on exhaustion or aclose()
    parses the buffered SSE body for actual token usage and calls post_call(),
    or falls back to release_all() if usage data is absent.
    """

    def __init__(
        self,
        inner: httpx.AsyncByteStream,
        engine: PolicyEngine,
        ctx: PolicyContext,
        ran: list[tuple[AbstractPolicy, PolicyDecision]],
        parsed_req: ParsedRequest,
    ) -> None:
        self._inner = inner
        self._engine = engine
        self._ctx = ctx
        self._ran = ran
        self._parsed_req = parsed_req
        self._chunks: list[bytes] = []
        self._reconciled = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner:
                self._chunks.append(chunk)
                yield chunk
        finally:
            self._reconcile()

    async def aclose(self) -> None:
        self._reconcile()
        await self._inner.aclose()

    def _reconcile(self) -> None:
        if self._reconciled:
            return
        self._reconciled = True

        from noveum_trace.guard.transport.helper import reconcile_stream

        reconcile_stream(
            self._chunks, self._engine, self._ctx, self._ran, self._parsed_req
        )


class NoveumAsyncTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that enforces Guard policies around every LLM call.

    Same data flow as NoveumTransport; all I/O awaited.
    """

    def __init__(
        self,
        engine: PolicyEngine,
        context: PolicyContext,
        inner: Optional[httpx.AsyncBaseTransport] = None,
        registry: Optional[AdapterRegistry] = None,
    ) -> None:
        self._engine = engine
        self._context = context
        self._inner = inner or httpx.AsyncHTTPTransport()
        self._registry = registry or default_registry()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        adapter = self._registry.for_request(request)
        if adapter is None:
            return await self._inner.handle_async_request(request)

        parsed_req = adapter.parse_request(request)
        ctx = dataclasses.replace(self._context, call_id=str(uuid.uuid4()))

        block, ran = self._engine.pre_call(parsed_req, ctx)
        if block is not None:
            return adapter.synthetic_block_response(
                request, block, block.block_response_mode
            )

        try:
            response = await self._inner.handle_async_request(request)
        except Exception:
            self._engine.release_all(ctx, ran)
            raise

        # Wrap streaming responses so the reservation is reconciled when the
        # caller finishes consuming the stream (instead of staying inflight forever).
        if parsed_req.stream:
            reconciler = _AsyncStreamReconciler(
                response.stream, self._engine, ctx, ran, parsed_req
            )
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=reconciler,
                request=request,
            )

        # Materialize the body before parsing — a real inner transport returns
        # an unread stream and .content would raise httpx.ResponseNotRead.
        await response.aread()
        parsed_resp = adapter.parse_response(request, response)
        post_block = self._engine.post_call(parsed_resp, ctx, ran)
        if post_block is not None:
            return adapter.synthetic_block_response(
                request, post_block, post_block.block_response_mode
            )

        return response
