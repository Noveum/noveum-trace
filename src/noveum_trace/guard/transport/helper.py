from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import httpx

from noveum_trace.guard.transport.async_transport import NoveumAsyncTransport
from noveum_trace.guard.transport.sync_transport import NoveumTransport
from noveum_trace.guard.types import ParsedRequest, PolicyContext
from noveum_trace.utils.logging import get_sdk_logger

if TYPE_CHECKING:
    from noveum_trace.guard.decision import PolicyDecision
    from noveum_trace.guard.engine import PolicyEngine
    from noveum_trace.guard.policies.base import AbstractPolicy

_logger = get_sdk_logger("guard.transport")


@dataclass
class SSEEventData:
    input_tokens: int
    output_tokens: int
    text: Optional[str]
    has_usage: bool  # whether any usage event was found (cost is only trustworthy then)


def _parse_sse_event_data(body: bytes, provider: str) -> SSEEventData:
    """Extract usage and assembled completion text from a fully-buffered SSE body.

    ``has_usage`` is False when no usage event was found (e.g. OpenAI without
    stream_options.include_usage=True) — callers that need cost accounting
    should treat that as "unknown" rather than "zero". ``text`` is best-effort
    assembled from delta events regardless of usage presence, since a
    post-phase content policy may need it even when cost can't be computed.

    Anthropic always includes usage in message_start / message_delta events.
    OpenAI only includes usage in the final chunk when the caller opts in.
    """
    decoded = body.decode("utf-8", errors="ignore")
    input_tokens = 0
    output_tokens = 0
    has_usage = False
    text_parts: list[str] = []

    for line in decoded.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        if provider == "anthropic":
            event_type = event.get("type", "")
            if event_type == "message_start":
                usage = event.get("message", {}).get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                has_usage = True
            elif event_type == "message_delta":
                usage = event.get("usage", {})
                output_tokens = usage.get("output_tokens", 0)
                has_usage = True
            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_parts.append(delta.get("text", ""))
        else:
            # OpenAI-compat: usage only present when stream_options.include_usage=True
            usage = event.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                has_usage = True
            choices = event.get("choices") or []
            if choices:
                delta_content = choices[0].get("delta", {}).get("content")
                if delta_content:
                    text_parts.append(delta_content)

    return SSEEventData(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        text="".join(text_parts) if text_parts else None,
        has_usage=has_usage,
    )


def reconcile_stream(
    chunks: list[bytes],
    engine: PolicyEngine,
    ctx: PolicyContext,
    ran: list[tuple[AbstractPolicy, PolicyDecision]],
    parsed_req: ParsedRequest,
) -> None:
    """Reconcile a Guard reservation from a fully-buffered SSE stream body.

    Shared by the sync and async stream reconcilers. Decompresses gzip if needed,
    parses actual token usage from the buffered SSE events, and calls
    engine.post_call() with a real ParsedResponse. Falls back to release_all()
    when usage data is absent or anything goes wrong.

    A post-phase block cannot be surfaced on a stream (bytes are already flushed
    to the caller), so a blocking post decision is logged rather than enforced.
    """
    # Local import keeps the engine→helper dependency one-directional at import time.
    from noveum_trace.guard.types import ParsedResponse
    from noveum_trace.utils.llm_utils import estimate_cost

    try:
        body = b"".join(chunks)
        # Raw stream bytes may be gzip-compressed (Content-Encoding: gzip).
        # Decompress before SSE parsing so _parse_sse_event_data sees plain text.
        if body[:2] == b"\x1f\x8b":
            try:
                body = gzip.decompress(body)
            except Exception:
                pass
        event_data = _parse_sse_event_data(body, parsed_req.provider)
        if not event_data.has_usage:
            # Stream has no usage events; release worst-case reservation.
            engine.release_all(ctx, ran)
            return
        costs = estimate_cost(
            parsed_req.model, event_data.input_tokens, event_data.output_tokens
        )
        parsed_resp = ParsedResponse(
            model=parsed_req.model,
            text=event_data.text,
            input_tokens=event_data.input_tokens,
            output_tokens=event_data.output_tokens,
            cost_usd=costs["total_cost"],
        )
        post_block = engine.post_call(parsed_resp, ctx, ran)
        if post_block is not None:
            # A policy with can_block_post=True attached to a truly-streaming
            # call (has_post_blocking_policies() was False when this call was
            # dispatched, or the policy was attached mid-stream) — an avoidable
            # enforcement gap, not an inert code path, hence error not warning.
            _logger.error(
                "Guard policy %r returned a post-phase block on a streaming "
                "response; cannot be enforced after bytes are streamed (reason: %s)",
                post_block.policy_name,
                post_block.reason,
            )
    except Exception:
        engine.release_all(ctx, ran)


def build_buffered_stream_response(
    response: httpx.Response,
    request: httpx.Request,
    adapter: Any,
    engine: PolicyEngine,
    ctx: PolicyContext,
    ran: list[tuple[AbstractPolicy, PolicyDecision]],
    parsed_req: ParsedRequest,
) -> httpx.Response:
    """Enforce post-phase policies on a streaming response before any bytes
    reach the caller, then either replay it or return a synthetic block.

    Used only when engine.has_post_blocking_policies() is True — otherwise
    the transport takes the lazy _SyncStreamReconciler/_AsyncStreamReconciler
    path (true streaming, no buffering). The caller must have already
    materialized ``response.content`` (response.read() / await
    response.aread()) before calling this.
    """
    from noveum_trace.guard.types import ParsedResponse
    from noveum_trace.utils.llm_utils import estimate_cost

    event_data = _parse_sse_event_data(response.content, parsed_req.provider)
    if event_data.has_usage:
        cost_usd = estimate_cost(
            parsed_req.model, event_data.input_tokens, event_data.output_tokens
        )["total_cost"]
    else:
        cost_usd = 0.0
    parsed_resp = ParsedResponse(
        model=parsed_req.model,
        text=event_data.text,
        input_tokens=event_data.input_tokens,
        output_tokens=event_data.output_tokens,
        cost_usd=cost_usd,
    )
    post_block = engine.post_call(parsed_resp, ctx, ran)
    if post_block is not None:
        return adapter.synthetic_block_response(
            request, post_block, post_block.block_response_mode
        )
    # Allowed: replay the exact original SSE bytes as a single materialized
    # body. httpx/provider SDKs parse SSE the same way whether delivered
    # incrementally or as one blob — the tradeoff is full-latency-before-
    # first-byte instead of true incremental streaming, paid only here.
    #
    # response.content is already content-decoded (response.read() applied any
    # Content-Encoding), so we must drop the original Content-Encoding and
    # Content-Length: replaying decoded bytes under a "gzip" header makes the
    # provider SDK's httpx read attempt a second gunzip (DecodingError), and
    # the original Content-Length describes the compressed body, not this one.
    replay_headers = [
        (name, value)
        for name, value in response.headers.items()
        if name.lower() not in ("content-encoding", "content-length")
    ]
    return httpx.Response(
        status_code=response.status_code,
        headers=replay_headers,
        content=response.content,
        request=request,
    )


def build_generic_block_response(reason: str) -> httpx.Response:
    """403 response for a request that matched no ProviderAdapter.

    Used by on_unmatched_request="block" — there is no adapter instance to
    call synthetic_block_response() on, so this builds a provider-agnostic
    equivalent (same shape as OpenAIAdapter's).
    """
    body = json.dumps({"error": {"type": "policy_blocked", "message": reason}}).encode()
    return httpx.Response(
        status_code=403, content=body, headers={"content-type": "application/json"}
    )


def _resolve_or_none(
    engine: Optional[PolicyEngine],
    context: Optional[PolicyContext],
) -> Optional[tuple[PolicyEngine, PolicyContext]]:
    """Like _resolve(), but returns None instead of raising when guard isn't
    configured and neither engine nor context was explicitly supplied.

    Used by call sites that run repeatedly for the life of the process (e.g.
    the Bedrock global _make_api_call patch's per-call hot path) rather than
    once at an explicit setup call site — raising there when guard hasn't
    been initialized yet would break every intercepted call in an
    unconfigured app; passing the call through unguarded is the safe failure
    mode instead. Partial provision (exactly one of engine/context) is still
    a caller bug and still raises, same as _resolve().
    """
    # Reject partial provision early: both must be supplied together or both omitted.
    if (engine is None) != (context is None):
        raise ValueError(
            "engine and context must be provided together or both omitted; "
            "supplying only one leads to mismatched policy binding."
        )
    if engine is not None and context is not None:
        return engine, context
    from noveum_trace.guard import _state

    resolved_engine = _state.get_engine()
    resolved_context = _state.get_context()
    if resolved_engine is None or resolved_context is None:
        return None
    return resolved_engine, resolved_context


def _resolve(
    engine: Optional[PolicyEngine],
    context: Optional[PolicyContext],
) -> tuple[PolicyEngine, PolicyContext]:
    resolved = _resolve_or_none(engine, context)
    if resolved is None:
        raise RuntimeError(
            "NovaGuard not initialized. Call "
            'noveum_trace.init(api_key="...", project="...", policies=[...]) first, '
            "or pass engine and context explicitly."
        )
    return resolved


def http_client(
    engine: Optional[PolicyEngine] = None,
    context: Optional[PolicyContext] = None,
    *,
    inner: Optional[httpx.BaseTransport] = None,
    on_unmatched_request: str = "passthrough",
    **kwargs: Any,
) -> httpx.Client:
    """Return a sync httpx.Client wired through the Guard transport.

    Zero-arg form (after noveum_trace.init(api_key="...", project="...", policies=[...])):
        openai.OpenAI(http_client=noveum_trace.guard.http_client())

    Explicit form:
        openai.OpenAI(http_client=noveum_trace.guard.http_client(engine, ctx))

    on_unmatched_request: "passthrough" (default, forwards requests that match
    no ProviderAdapter unguarded — today's behavior) or "block" (return a
    synthetic 403 instead, for defense-in-depth deployments that want to
    guarantee nothing bypasses Guard coverage).
    """
    resolved_engine, resolved_context = _resolve(engine, context)
    transport = NoveumTransport(
        engine=resolved_engine,
        context=resolved_context,
        inner=inner,
        on_unmatched_request=on_unmatched_request,
    )
    return httpx.Client(transport=transport, **kwargs)


def async_http_client(
    engine: Optional[PolicyEngine] = None,
    context: Optional[PolicyContext] = None,
    *,
    inner: Optional[httpx.AsyncBaseTransport] = None,
    on_unmatched_request: str = "passthrough",
    **kwargs: Any,
) -> httpx.AsyncClient:
    """Return an async httpx.AsyncClient wired through the Guard transport.

    Zero-arg form (after noveum_trace.init(api_key="...", project="...", policies=[...])):
        anthropic.AsyncAnthropic(http_client=noveum_trace.guard.async_http_client())

    Explicit form:
        anthropic.AsyncAnthropic(http_client=noveum_trace.guard.async_http_client(engine, ctx))

    See http_client() for on_unmatched_request.
    """
    resolved_engine, resolved_context = _resolve(engine, context)
    transport = NoveumAsyncTransport(
        engine=resolved_engine,
        context=resolved_context,
        inner=inner,
        on_unmatched_request=on_unmatched_request,
    )
    return httpx.AsyncClient(transport=transport, **kwargs)
