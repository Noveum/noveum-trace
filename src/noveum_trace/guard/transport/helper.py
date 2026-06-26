from __future__ import annotations

import gzip
import json
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


def _parse_sse_usage(body: bytes, provider: str) -> tuple[int, int] | None:
    """Extract (input_tokens, output_tokens) from a fully-buffered SSE stream body.

    Returns None when usage data is absent (e.g. OpenAI without
    stream_options.include_usage=True).

    Anthropic always includes usage in message_start / message_delta events.
    OpenAI only includes usage in the final chunk when the caller opts in.
    """
    text = body.decode("utf-8", errors="ignore")
    input_tokens = 0
    output_tokens = 0
    found = False

    for line in text.splitlines():
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
                found = True
            elif event_type == "message_delta":
                usage = event.get("usage", {})
                output_tokens = usage.get("output_tokens", 0)
                found = True
        else:
            # OpenAI-compat: usage only present when stream_options.include_usage=True
            usage = event.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                found = True

    return (input_tokens, output_tokens) if found else None


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
        # Decompress before SSE parsing so _parse_sse_usage sees plain text.
        if body[:2] == b"\x1f\x8b":
            try:
                body = gzip.decompress(body)
            except Exception:
                pass
        usage = _parse_sse_usage(body, parsed_req.provider)
        if usage is None:
            # Stream has no usage events; release worst-case reservation.
            engine.release_all(ctx, ran)
            return
        input_tokens, output_tokens = usage
        costs = estimate_cost(parsed_req.model, input_tokens, output_tokens)
        parsed_resp = ParsedResponse(
            model=parsed_req.model,
            text=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=costs["total_cost"],
        )
        post_block = engine.post_call(parsed_resp, ctx, ran)
        if post_block is not None:
            _logger.warning(
                "Guard policy %r returned a post-phase block on a streaming "
                "response; cannot be enforced after bytes are streamed (reason: %s)",
                post_block.policy_name,
                post_block.reason,
            )
    except Exception:
        engine.release_all(ctx, ran)


def _resolve(
    engine: Optional[PolicyEngine],
    context: Optional[PolicyContext],
) -> tuple[PolicyEngine, PolicyContext]:
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
        raise RuntimeError(
            "NovaGuard not initialized. Call "
            'noveum_trace.init(api_key="...", project="...", policies=[...]) first, '
            "or pass engine and context explicitly."
        )
    return resolved_engine, resolved_context


def http_client(
    engine: Optional[PolicyEngine] = None,
    context: Optional[PolicyContext] = None,
    *,
    inner: Optional[httpx.BaseTransport] = None,
    **kwargs: Any,
) -> httpx.Client:
    """Return a sync httpx.Client wired through the Guard transport.

    Zero-arg form (after noveum_trace.init(api_key="...", project="...", policies=[...])):
        openai.OpenAI(http_client=noveum_trace.guard.http_client())

    Explicit form:
        openai.OpenAI(http_client=noveum_trace.guard.http_client(engine, ctx))
    """
    resolved_engine, resolved_context = _resolve(engine, context)
    transport = NoveumTransport(
        engine=resolved_engine, context=resolved_context, inner=inner
    )
    return httpx.Client(transport=transport, **kwargs)


def async_http_client(
    engine: Optional[PolicyEngine] = None,
    context: Optional[PolicyContext] = None,
    *,
    inner: Optional[httpx.AsyncBaseTransport] = None,
    **kwargs: Any,
) -> httpx.AsyncClient:
    """Return an async httpx.AsyncClient wired through the Guard transport.

    Zero-arg form (after noveum_trace.init(api_key="...", project="...", policies=[...])):
        anthropic.AsyncAnthropic(http_client=noveum_trace.guard.async_http_client())

    Explicit form:
        anthropic.AsyncAnthropic(http_client=noveum_trace.guard.async_http_client(engine, ctx))
    """
    resolved_engine, resolved_context = _resolve(engine, context)
    transport = NoveumAsyncTransport(
        engine=resolved_engine, context=resolved_context, inner=inner
    )
    return httpx.AsyncClient(transport=transport, **kwargs)
