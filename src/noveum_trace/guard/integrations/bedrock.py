from __future__ import annotations

import dataclasses
import io
import json
import logging
import threading
import uuid
import weakref
from typing import TYPE_CHECKING, Any, Callable, Optional

from noveum_trace.guard.exceptions import NoveumGuardBlocked
from noveum_trace.guard.types import ParsedRequest, ParsedResponse, PolicyContext
from noveum_trace.utils.llm_utils import estimate_cost, estimate_token_count

if TYPE_CHECKING:
    from noveum_trace.guard.decision import PolicyDecision
    from noveum_trace.guard.engine import PolicyEngine
    from noveum_trace.guard.policies.base import AbstractPolicy

logger = logging.getLogger(__name__)

_CONVERSE_OPERATIONS = {"Converse", "ConverseStream"}
_STREAMING_OPERATIONS = {"InvokeModelWithResponseStream", "ConverseStream"}
_KNOWN_OPERATIONS = _CONVERSE_OPERATIONS | {
    "InvokeModel",
    "InvokeModelWithResponseStream",
}


def _model_family(model_id: str) -> str:
    for prefix in ("anthropic.", "amazon.titan", "meta.llama", "cohere.", "mistral."):
        if model_id.startswith(prefix):
            return prefix
    return ""


def _parse_invoke_model_body(
    model_id: str, body: dict[str, Any]
) -> tuple[Any, Optional[int]]:
    """Return (prompt-shaped content for token estimation, max_tokens).

    Bedrock's raw InvokeModel body shape differs per model family; unrecognized
    families fall back to the whole body as text so pre-call blocking still
    has something to estimate against.
    """
    family = _model_family(model_id)
    if family == "anthropic.":
        return body.get("messages", []), body.get("max_tokens")
    if family == "amazon.titan":
        max_tokens = (body.get("textGenerationConfig") or {}).get("maxTokenCount")
        return body.get("inputText", ""), max_tokens
    if family == "meta.llama":
        return body.get("prompt", ""), body.get("max_gen_len")
    if family == "cohere.":
        return body.get("message") or body.get("prompt", ""), body.get("max_tokens")
    if family == "mistral.":
        return body.get("prompt", ""), body.get("max_tokens")
    return json.dumps(body), None


def _parse_invoke_model_usage(
    model_id: str, response_body: dict[str, Any]
) -> tuple[Optional[int], Optional[int]]:
    """Return (input_tokens, output_tokens), or (None, None) if unrecognized.

    Callers must treat (None, None) as "unknown" — release the reservation
    rather than guess a cost, per the same convention used for streaming SSE
    reconciliation elsewhere in the guard transport.
    """
    family = _model_family(model_id)
    if family == "anthropic.":
        usage = response_body.get("usage") or {}
        if "input_tokens" in usage and "output_tokens" in usage:
            return usage["input_tokens"], usage["output_tokens"]
    elif family == "amazon.titan":
        if "inputTextTokenCount" in response_body:
            results = response_body.get("results") or [{}]
            return response_body["inputTextTokenCount"], results[0].get("tokenCount")
    elif family == "meta.llama":
        if "prompt_token_count" in response_body:
            return (
                response_body["prompt_token_count"],
                response_body.get("generation_token_count"),
            )
    elif family == "cohere.":
        billed = (response_body.get("meta") or {}).get("billed_units") or {}
        if "input_tokens" in billed:
            return billed.get("input_tokens"), billed.get("output_tokens")
    return None, None


def _is_embeddings_model(model_id: str) -> bool:
    """True for Bedrock embedding models (Titan Embed, Cohere Embed).

    Embedding InvokeModel bodies/responses differ from text-generation ones and
    never carry output tokens, so they need distinct request/usage parsing plus
    kind="embeddings" — otherwise CostCapPolicy reserves a chat-style output
    allowance (up to 4096 tokens) for a call that produces none, over-reserving
    and potentially blocking spuriously at pre().
    """
    return "embed" in model_id.lower()


def _parse_embeddings_input(model_id: str, body: dict[str, Any]) -> Any:
    """Return prompt-shaped content for input-token estimation of an embed call."""
    if _model_family(model_id) == "cohere.":
        texts = body.get("texts") or []
        return " ".join(t for t in texts if isinstance(t, str))
    # Titan Embed (text & image) uses a single inputText field.
    return body.get("inputText", "")


def _parse_embeddings_usage(
    model_id: str, response_body: dict[str, Any]
) -> Optional[int]:
    """Return exact input_tokens if the embed response reports them, else None.

    Titan embed responses include ``inputTextTokenCount``; Cohere embed responses
    carry no token count (the caller falls back to the request-time estimate).
    Output tokens are always 0 for embeddings.
    """
    if _model_family(model_id) == "amazon.titan":
        count = response_body.get("inputTextTokenCount")
        if isinstance(count, int):
            return count
    return None


def _extract_event_usage(
    event: dict[str, Any], operation_name: str
) -> Optional[tuple[Optional[int], Optional[int]]]:
    """Return (input_tokens, output_tokens) if this event carries usage, else None.

    Shared by the lazy tee reconciler and the buffer-before-yield enforcement
    path so both read usage from the stream identically.
    """
    if operation_name == "ConverseStream":
        usage = (event.get("metadata") or {}).get("usage")
        if usage:
            return usage.get("inputTokens"), usage.get("outputTokens")
        return None

    # InvokeModelWithResponseStream: AWS injects a trailing
    # amazon-bedrock-invocationMetrics field into the final chunk for every
    # model family, regardless of the underlying model's own response shape.
    chunk = (event.get("chunk") or {}).get("bytes")
    if not chunk:
        return None
    try:
        payload = json.loads(chunk)
    except (TypeError, ValueError):
        return None
    metrics = payload.get("amazon-bedrock-invocationMetrics")
    if metrics:
        return metrics.get("inputTokenCount"), metrics.get("outputTokenCount")
    return None


def _extract_event_text(event: dict[str, Any], operation_name: str) -> Optional[str]:
    """Best-effort incremental completion text from one stream event.

    Only needed for the buffered enforcement path, where a post-phase content
    policy may inspect the assembled output. Token accounting does not depend on
    it — an unrecognized shape simply contributes no text.
    """
    if operation_name == "ConverseStream":
        delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
        text = delta.get("text")
        return text if isinstance(text, str) else None

    chunk = (event.get("chunk") or {}).get("bytes")
    if not chunk:
        return None
    try:
        payload = json.loads(chunk)
    except (TypeError, ValueError):
        return None
    # Per-family InvokeModel chunk shapes: anthropic delta.text, titan
    # outputText, llama generation, cohere/mistral text, mistral outputs[].text.
    delta = payload.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("text"), str):
        return delta["text"]
    for key in ("outputText", "generation", "text", "completion"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    outputs = payload.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        text = outputs[0].get("text")
        return text if isinstance(text, str) else None
    return None


def _rebuffer_streaming_body(raw_bytes: bytes) -> Any:
    """Replace a single-read StreamingBody with a fresh one over the same bytes.

    Local import: botocore is only needed here (and only ever reached when the
    caller already holds a live boto3 client), so the module itself stays
    importable without boto3/botocore installed.
    """
    import botocore.response

    return botocore.response.StreamingBody(io.BytesIO(raw_bytes), len(raw_bytes))


class _BedrockStreamReconciler:
    """Tees a Bedrock EventStream; reconciles the Guard reservation on exhaustion.

    Mirrors ``_SyncStreamReconciler`` in ``guard.transport.sync_transport``,
    adapted to botocore's dict-yielding EventStream instead of a raw byte
    stream. Pre-call blocking already happened before this object exists —
    this only affects post-call cost reconciliation accuracy.
    """

    def __init__(
        self,
        inner: Any,
        engine: PolicyEngine,
        ctx: PolicyContext,
        ran: list[tuple[AbstractPolicy, PolicyDecision]],
        parsed_req: ParsedRequest,
        operation_name: str,
    ) -> None:
        self._inner = inner
        self._engine = engine
        self._ctx = ctx
        self._ran = ran
        self._parsed_req = parsed_req
        self._operation_name = operation_name
        self._reconciled = False
        self._input_tokens: Optional[int] = None
        self._output_tokens: Optional[int] = None

    def __iter__(self) -> Any:
        try:
            for event in self._inner:
                self._observe(event)
                yield event
        finally:
            self._reconcile()

    def _observe(self, event: dict[str, Any]) -> None:
        usage = _extract_event_usage(event, self._operation_name)
        if usage is not None:
            self._input_tokens, self._output_tokens = usage

    def _reconcile(self) -> None:
        if self._reconciled:
            return
        self._reconciled = True

        if self._input_tokens is None or self._output_tokens is None:
            logger.warning(
                "NovaGuard: no usage metrics found in Bedrock %s stream for "
                "model %r; releasing reservation without exact cost reconciliation",
                self._operation_name,
                self._parsed_req.model,
            )
            self._engine.release_all(self._ctx, self._ran)
            return

        cost_usd = estimate_cost(
            self._parsed_req.model, self._input_tokens, self._output_tokens
        )["total_cost"]
        resp = ParsedResponse(
            model=self._parsed_req.model,
            text=None,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=cost_usd,
        )
        post_block = self._engine.post_call(resp, self._ctx, self._ran)
        if post_block is not None:
            # Policy attached mid-stream after dispatch; bytes already yielded, can only log.
            logger.error(
                "NovaGuard: policy %r returned a post-phase block on a Bedrock "
                "%s stream; cannot be enforced after events are yielded (reason: %s)",
                post_block.policy_name,
                self._operation_name,
                post_block.reason,
            )


# Parsing / reconciliation helpers — module-level since one _guarded_make_api_call
# call spans pre-call -> real call -> post-call, with no cross-callback state to hand off.


def _parse_request(operation_name: str, params: dict[str, Any]) -> ParsedRequest:
    model_id = params.get("modelId", "")
    stream = operation_name in _STREAMING_OPERATIONS
    kind = "chat"

    if operation_name in _CONVERSE_OPERATIONS:
        messages = params.get("messages", [])
        max_tokens = (params.get("inferenceConfig") or {}).get("maxTokens")
        estimated = estimate_token_count(messages, model=model_id, provider="bedrock")
    else:
        body = params.get("body")
        if isinstance(body, (bytes, str)):
            try:
                body_dict = json.loads(body)
            except (TypeError, ValueError):
                body_dict = {}
        elif isinstance(body, dict):
            body_dict = body
        else:
            body_dict = {}
        if _is_embeddings_model(model_id):
            kind = "embeddings"
            messages = []
            max_tokens = None
            content = _parse_embeddings_input(model_id, body_dict)
            estimated = estimate_token_count(
                content, model=model_id, provider="bedrock"
            )
        else:
            content, max_tokens = _parse_invoke_model_body(model_id, body_dict)
            messages = content if isinstance(content, list) else []
            estimated = estimate_token_count(
                content, model=model_id, provider="bedrock"
            )

    return ParsedRequest(
        provider="bedrock",
        model=model_id,
        messages=messages,
        stream=stream,
        max_tokens=max_tokens,
        estimated_input_tokens=estimated,
        raw_body=b"",
        kind=kind,
    )


def _reconcile_non_streaming(
    operation_name: str,
    parsed_req: ParsedRequest,
    parsed: dict[str, Any],
    engine: PolicyEngine,
    ctx: PolicyContext,
    ran: list[tuple[AbstractPolicy, PolicyDecision]],
) -> None:
    model_id = parsed_req.model

    if operation_name in _CONVERSE_OPERATIONS:
        usage = parsed.get("usage") or {}
        input_tokens = usage.get("inputTokens")
        output_tokens = usage.get("outputTokens")
    else:
        # InvokeModel: parsed["body"] is a single-read StreamingBody. Read
        # it once for usage extraction, then substitute a fresh, fully
        # buffered one so the caller's own response["body"].read() still works.
        raw_bytes = parsed["body"].read()
        parsed["body"] = _rebuffer_streaming_body(raw_bytes)
        try:
            body_dict = json.loads(raw_bytes)
        except (TypeError, ValueError):
            body_dict = {}
        if parsed_req.kind == "embeddings":
            # Embeddings never produce output tokens. Prefer the exact input
            # count from the response (Titan's inputTextTokenCount); Cohere embed
            # carries none, so fall back to the request-time estimate rather than
            # releasing — that estimate is what the cap was reserved against.
            exact = _parse_embeddings_usage(model_id, body_dict)
            input_tokens = (
                exact if exact is not None else parsed_req.estimated_input_tokens
            )
            output_tokens = 0
        else:
            input_tokens, output_tokens = _parse_invoke_model_usage(model_id, body_dict)

    if input_tokens is None or output_tokens is None:
        logger.warning(
            "NovaGuard: no usage found in Bedrock %s response for model %r; "
            "releasing reservation without exact cost reconciliation",
            operation_name,
            model_id,
        )
        engine.release_all(ctx, ran)
        return

    cost_usd = estimate_cost(model_id, input_tokens, output_tokens)["total_cost"]
    resp = ParsedResponse(
        model=model_id,
        text=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )
    post_block = engine.post_call(resp, ctx, ran)
    if post_block is not None:
        raise NoveumGuardBlocked(post_block.policy_name, post_block.reason, post_block)


def _buffer_and_enforce_stream(
    parsed: dict[str, Any],
    engine: PolicyEngine,
    ctx: PolicyContext,
    ran: list[tuple[AbstractPolicy, PolicyDecision]],
    parsed_req: ParsedRequest,
    operation_name: str,
    key: str,
) -> dict[str, Any]:
    """Buffer the whole event stream, run post_call BEFORE any event is yielded.

    Mirrors ``build_buffered_stream_response`` on the httpx transport path: used
    only when a post-blocking policy is attached
    (``engine.has_post_blocking_policies()``). A post-phase block can then be
    enforced by raising ``NoveumGuardBlocked`` before the caller sees a single
    event — instead of only being logged after the bytes are already yielded, as
    the lazy ``_BedrockStreamReconciler`` path does. The tradeoff (paid only
    here) is full-latency-before-first-event instead of true streaming.
    """
    events = list(parsed[key])
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    text_parts: list[str] = []
    for event in events:
        usage = _extract_event_usage(event, operation_name)
        if usage is not None:
            input_tokens, output_tokens = usage
        chunk_text = _extract_event_text(event, operation_name)
        if chunk_text:
            text_parts.append(chunk_text)

    # Absent usage → cost is unknown (0.0); a content policy can still inspect
    # the assembled text. Matches the httpx buffered path's has_usage handling.
    if input_tokens is not None and output_tokens is not None:
        cost_usd = estimate_cost(parsed_req.model, input_tokens, output_tokens)[
            "total_cost"
        ]
    else:
        cost_usd = 0.0
    resp = ParsedResponse(
        model=parsed_req.model,
        text="".join(text_parts) if text_parts else None,
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        cost_usd=cost_usd,
    )
    post_block = engine.post_call(resp, ctx, ran)
    if post_block is not None:
        raise NoveumGuardBlocked(post_block.policy_name, post_block.reason, post_block)
    # Allowed: replay the buffered events so the caller iterates normally.
    parsed[key] = events
    return parsed


def _wrap_streaming_result(
    parsed: dict[str, Any],
    engine: PolicyEngine,
    ctx: PolicyContext,
    ran: list[tuple[AbstractPolicy, PolicyDecision]],
    parsed_req: ParsedRequest,
    operation_name: str,
) -> dict[str, Any]:
    key = "stream" if operation_name == "ConverseStream" else "body"
    if parsed is None or parsed.get(key) is None:
        # Nothing to tee — release rather than leave the reservation inflight.
        engine.release_all(ctx, ran)
        return parsed
    if engine.has_post_blocking_policies():
        # A policy that can block post() is attached: a block can't be enforced
        # on events already yielded, so buffer the whole stream and enforce
        # before releasing anything to the caller.
        try:
            return _buffer_and_enforce_stream(
                parsed, engine, ctx, ran, parsed_req, operation_name, key
            )
        except NoveumGuardBlocked:
            # Intended post-phase enforcement — post_call already reconciled the
            # reservation, so this must NOT release (that would double-refund).
            raise
        except Exception:
            # Draining the stream failed before we could reconcile — release the
            # reservation instead of leaking it inflight (mirrors the httpx
            # buffered path's release-on-error in sync_transport).
            engine.release_all(ctx, ran)
            raise
    parsed[key] = _BedrockStreamReconciler(
        parsed[key], engine, ctx, ran, parsed_req, operation_name
    )
    return parsed


# Patching BaseClient._make_api_call (not a subclass method) covers every
# bedrock-runtime client from any session, past or future — same technique
# OpenTelemetry/Sentry/Datadog use for botocore.

_patch_lock = threading.Lock()
_patched = False
_original_make_api_call: Optional[Callable[..., Any]] = None

# Per-client advanced override: instrument_bedrock(client, engine=..., context=...).
# Keyed by weak reference so instrumented clients can still be garbage collected.
_client_overrides: weakref.WeakKeyDictionary[
    Any, tuple[PolicyEngine, PolicyContext]
] = weakref.WeakKeyDictionary()
# Process-wide default set via instrument_bedrock(engine=..., context=...) with no
# client — takes precedence over noveum_trace.init() so callers don't need init() first.
_global_override: Optional[tuple[PolicyEngine, PolicyContext]] = None


def _resolve_for_client(client: Any) -> Optional[tuple[PolicyEngine, PolicyContext]]:
    """Resolution order: per-client override > global override > live guard._state.

    Returns None when nothing is configured anywhere — the wrapper must treat
    that as "pass through unguarded", not raise, since it runs on every
    Bedrock call for the life of the process, not once at an explicit setup
    call site.
    """
    override = _client_overrides.get(client)
    if override is not None:
        return override
    if _global_override is not None:
        return _global_override
    from noveum_trace.guard.transport.helper import _resolve_or_none

    return _resolve_or_none(None, None)


def _guarded_make_api_call(
    self: Any, operation_name: str, api_params: dict[str, Any]
) -> Any:
    assert _original_make_api_call is not None
    service_model = getattr(self.meta, "service_model", None)
    if service_model is None or service_model.service_name != "bedrock-runtime":
        return _original_make_api_call(self, operation_name, api_params)
    if operation_name not in _KNOWN_OPERATIONS:
        return _original_make_api_call(self, operation_name, api_params)

    resolved = _resolve_for_client(self)
    if resolved is None:
        return _original_make_api_call(self, operation_name, api_params)
    engine, base_context = resolved

    ctx = dataclasses.replace(base_context, call_id=str(uuid.uuid4()))
    parsed_req = _parse_request(operation_name, api_params)

    block, ran = engine.pre_call(parsed_req, ctx)
    if block is not None:
        raise NoveumGuardBlocked(block.policy_name, block.reason, block)

    try:
        parsed = _original_make_api_call(self, operation_name, api_params)
    except Exception:
        engine.release_all(ctx, ran)
        raise

    if operation_name in _STREAMING_OPERATIONS:
        return _wrap_streaming_result(
            parsed, engine, ctx, ran, parsed_req, operation_name
        )

    _reconcile_non_streaming(operation_name, parsed_req, parsed, engine, ctx, ran)
    return parsed


def _install_global_patch() -> None:
    global _patched, _original_make_api_call
    if _patched:
        return
    with _patch_lock:
        if _patched:
            return
        import botocore.client

        _original_make_api_call = botocore.client.BaseClient._make_api_call
        botocore.client.BaseClient._make_api_call = _guarded_make_api_call
        _patched = True


def _uninstrument_bedrock_for_tests() -> None:
    """Restore the unpatched botocore method and clear all override state.

    Not part of the public API — imported directly by test teardown fixtures
    only, since this is a genuine process-wide mutation of third-party class
    state that must never leak across test modules.
    """
    global _patched, _original_make_api_call, _global_override
    with _patch_lock:
        if _patched and _original_make_api_call is not None:
            import botocore.client

            botocore.client.BaseClient._make_api_call = _original_make_api_call
        _patched = False
        _original_make_api_call = None
    _client_overrides.clear()
    _global_override = None


def instrument_bedrock(
    client: Optional[Any] = None,
    *,
    engine: Optional[PolicyEngine] = None,
    context: Optional[PolicyContext] = None,
) -> Optional[Any]:
    """Guard AWS Bedrock (boto3 bedrock-runtime) calls, process-wide by default.

    Global form — call once after ``noveum_trace.init()``; every bedrock-runtime
    client built afterward, from any session, is guarded.

    Advanced form — ``instrument_bedrock(client, engine=..., context=...)`` binds
    one client to its own engine/context instead of the process-wide default
    (still installs the global patch). engine and context must be supplied together.
    """
    if (engine is None) != (context is None):
        raise ValueError(
            "engine and context must be provided together or both omitted; "
            "supplying only one leads to mismatched policy binding."
        )

    _install_global_patch()

    if client is not None:
        if engine is not None and context is not None:
            _client_overrides[client] = (engine, context)
        return client

    if engine is not None and context is not None:
        global _global_override
        _global_override = (engine, context)
    return None
