from __future__ import annotations

import dataclasses
import logging
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional

import httpx

from noveum_trace.guard.transport.adapters.base import AdapterRegistry, default_registry
from noveum_trace.guard.types import ParsedRequest, PolicyContext

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from noveum_trace.guard.decision import PolicyDecision
    from noveum_trace.guard.engine import PolicyEngine
    from noveum_trace.guard.policies.base import AbstractPolicy


class _SyncStreamReconciler(httpx.SyncByteStream):
    """Wraps a streaming response body; reconciles Guard reservation when fully consumed.

    Buffers every chunk as the caller reads the stream, then on exhaustion
    (or early close) parses the buffered SSE body to extract actual token usage
    and calls engine.post_call() with a real ParsedResponse.  Falls back to
    engine.release_all() if usage data is not present in the stream.
    """

    def __init__(
        self,
        inner: httpx.SyncByteStream,
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

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._inner:
                self._chunks.append(chunk)
                yield chunk
        finally:
            self._reconcile()

    def close(self) -> None:
        self._reconcile()
        self._inner.close()

    def _reconcile(self) -> None:
        if self._reconciled:
            return
        self._reconciled = True

        from noveum_trace.guard.transport.helper import reconcile_stream

        reconcile_stream(
            self._chunks, self._engine, self._ctx, self._ran, self._parsed_req
        )


class NoveumTransport(httpx.BaseTransport):
    """Sync httpx transport that enforces Guard policies around every LLM call.

    Data flow:
      1. Detect provider; skip if unknown.
      2. Parse request body → ParsedRequest.
      3. Mint call_id; build PolicyContext with it.
      4. engine.pre_call() → block decision or ran list.
         Block → return synthetic 403 (engine already rolled back).
      5. Forward to inner transport.
      6. Streaming responses are wrapped in _SyncStreamReconciler, which buffers
         every chunk and reconciles the reservation when the stream is exhausted
         or closed. Actual token usage is parsed from SSE events; falls back to
         release_all() if usage data is absent.
      7. Non-streaming: read + parse response → ParsedResponse.
      8. engine.post_call() → optional block.
         Block → return synthetic 403.
      9. Exception after forwarding → engine.release_all(); re-raise.
    """

    def __init__(
        self,
        engine: PolicyEngine,
        context: PolicyContext,
        inner: Optional[httpx.BaseTransport] = None,
        registry: Optional[AdapterRegistry] = None,
    ) -> None:
        self._engine = engine
        self._context = context
        self._inner = inner or httpx.HTTPTransport()
        self._registry = registry or default_registry()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        adapter = self._registry.for_request(request)
        if adapter is None:
            _log.error(
                "NovaGuard: no adapter found for request %s %s — passing through unguarded",
                request.method,
                request.url,
            )
            return self._inner.handle_request(request)

        parsed_req = adapter.parse_request(request)
        ctx = dataclasses.replace(self._context, call_id=str(uuid.uuid4()))

        block, ran = self._engine.pre_call(parsed_req, ctx)
        if block is not None:
            return adapter.synthetic_block_response(
                request, block, block.block_response_mode
            )

        try:
            response = self._inner.handle_request(request)
        except Exception:
            self._engine.release_all(ctx, ran)
            raise

        # Wrap streaming responses so the reservation is reconciled when the
        # caller finishes consuming the stream (instead of staying inflight forever).
        if parsed_req.stream:
            reconciler = _SyncStreamReconciler(
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
        try:
            response.read()
            parsed_resp = adapter.parse_response(request, response)
            post_block = self._engine.post_call(parsed_resp, ctx, ran)
            if post_block is not None:
                return adapter.synthetic_block_response(
                    request, post_block, post_block.block_response_mode
                )
        except Exception:
            self._engine.release_all(ctx, ran)
            raise

        return response
