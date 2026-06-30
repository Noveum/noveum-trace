"""
OpenTelemetry compatibility helpers for Noveum Trace SDK.

These are pure, side-effect-free functions used at *export time* to derive
OpenTelemetry-aligned fields (W3C-format IDs, ``gen_ai.*`` semantic-convention
attributes, span kind, status object, resource, instrumentation scope) from data
already present on a span/trace.

The additions are strictly additive: every existing field is left untouched and the
OTel fields are emitted alongside, tagged with ``SCHEMA_VERSION`` so the ingestion
pipeline can route between the legacy and OTel-aware schema. Nothing here mutates the
inputs; callers merge the returned values into the outgoing payload.
"""

from typing import Any, Optional

# Bumped whenever the additive OTel field set changes. Absent on legacy payloads.
SCHEMA_VERSION = "2.0"

# Default instrumentation scope name when a span is not attributed to an integration.
DEFAULT_SCOPE_NAME = "noveum-trace-python"

# Span kinds (OTel). We only ever emit INTERNAL or CLIENT today; SERVER/PRODUCER/
# CONSUMER are reserved for future inbound/messaging spans.
KIND_INTERNAL = "INTERNAL"
KIND_CLIENT = "CLIENT"

# Status codes (OTel).
STATUS_UNSET = "UNSET"
STATUS_OK = "OK"
STATUS_ERROR = "ERROR"

# Legacy span-status string -> OTel status code. ``timeout``/``cancelled`` have no OTel
# equivalent and collapse to ERROR (the original value is preserved as an attribute).
_STATUS_MAP = {
    "unset": STATUS_UNSET,
    "ok": STATUS_OK,
    "error": STATUS_ERROR,
    "timeout": STATUS_ERROR,
    "cancelled": STATUS_ERROR,
}

# Source keys whose value is one of these placeholders are skipped when mirroring to
# ``gen_ai.*`` (empty-dict params, nulls, and stringified mocks from the transport).
_SKIP_VALUES: tuple[Any, ...] = (None, "<Mock object>")


def _is_meaningful(value: Any) -> bool:
    """True unless the value is a placeholder we should not mirror to gen_ai.*."""
    if value in _SKIP_VALUES:
        return False
    if isinstance(value, dict) and not value:  # empty-dict placeholder, e.g. llm.top_p
        return False
    return True


def _first_present(attrs: dict[str, Any], *keys: str) -> tuple[bool, Any]:
    """Return (found, value) for the first key present with a meaningful value."""
    for key in keys:
        if key in attrs and _is_meaningful(attrs[key]):
            return True, attrs[key]
    return False, None


def to_otel_trace_id(trace_id: Optional[str]) -> Optional[str]:
    """
    Convert a UUID trace id to a W3C 32-hex-char trace id.

    A UUIDv4 is already 128 bits, so this is lossless: strip dashes, lowercase.
    Already-hex ids pass through unchanged.
    """
    if not trace_id:
        return None
    return trace_id.replace("-", "").lower()


def to_otel_span_id(span_id: Optional[str]) -> Optional[str]:
    """
    Convert a UUID span id to a W3C 16-hex-char (64-bit) span id.

    OTel span ids are half the width of a UUID, so we take the first 16 hex chars.
    Deterministic, so parent/child references derive consistently. Already-short
    hex ids pass through unchanged.
    """
    if not span_id:
        return None
    return span_id.replace("-", "").lower()[:16]


# llm.* (kept) -> gen_ai.* (added). Ordered source keys: first meaningful one wins.
# Cost / ttft / tokens-per-second have no OTel standard and are intentionally omitted
# (they remain available under their original llm.* keys).
_GEN_AI_CROSSWALK: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gen_ai.request.model", ("llm.model",)),
    ("gen_ai.response.model", ("llm.response_model",)),
    ("gen_ai.provider.name", ("llm.provider",)),
    ("gen_ai.operation.name", ("llm.operation",)),
    (
        "gen_ai.usage.input_tokens",
        ("llm.input_tokens", "llm.prompt_tokens", "llm.usage.input_tokens"),
    ),
    (
        "gen_ai.usage.output_tokens",
        ("llm.output_tokens", "llm.completion_tokens", "llm.usage.output_tokens"),
    ),
    ("gen_ai.request.temperature", ("llm.temperature", "llm.input.temperature")),
    ("gen_ai.request.max_tokens", ("llm.max_tokens",)),
    ("gen_ai.request.top_p", ("llm.top_p",)),
    ("gen_ai.input.messages", ("llm.input.messages", "llm.chat_ctx", "llm.input")),
    (
        "gen_ai.output.messages",
        ("llm.output.response", "llm.response", "llm.output"),
    ),
    ("gen_ai.response.id", ("llm.request_id",)),
)


def derive_gen_ai_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    """
    Derive ``gen_ai.*`` attributes from existing ``llm.*`` attributes.

    Returns only the new keys to merge in; never overwrites a ``gen_ai.*`` key that is
    already present. Does not mutate ``attrs``.
    """
    result: dict[str, Any] = {}
    for target, sources in _GEN_AI_CROSSWALK:
        if target in attrs:  # respect an explicitly-set gen_ai.* value
            continue
        found, value = _first_present(attrs, *sources)
        if found:
            result[target] = value

    # finish_reasons is an array in OTel; wrap a scalar source value.
    if "gen_ai.response.finish_reasons" not in attrs:
        found, value = _first_present(attrs, "llm.finish_reason")
        if found:
            result["gen_ai.response.finish_reasons"] = (
                value if isinstance(value, list) else [value]
            )

    return result


def infer_span_kind(name: str, attrs: dict[str, Any]) -> str:
    """
    Infer the OTel span kind from the span name / attributes.

    CLIENT for outbound model/HTTP calls; INTERNAL otherwise.
    """
    lowered = name.lower() if name else ""
    if lowered.startswith(("llm.", "http.")):
        return KIND_CLIENT
    for key in attrs:
        if key.startswith(("llm.", "gen_ai.")):
            return KIND_CLIENT
    return KIND_INTERNAL


def to_otel_status(status: Optional[str], message: Optional[str]) -> dict[str, Any]:
    """
    Build an OTel ``{code, message}`` status object from the legacy status string.

    ``message`` is only included when the code is ERROR (per OTel convention).
    """
    code = _STATUS_MAP.get((status or "unset").lower(), STATUS_UNSET)
    result: dict[str, Any] = {"code": code}
    if code == STATUS_ERROR and message:
        result["message"] = message
    return result


def otel_span_name(name: str, attrs: dict[str, Any]) -> str:
    """
    Build an OTel ``"{operation} {model}"`` span name when model is known.

    Falls back to the existing span name otherwise.
    """
    _, model = _first_present(attrs, "gen_ai.request.model", "llm.model")
    if not model:
        return name
    _, operation = _first_present(attrs, "gen_ai.operation.name", "llm.operation")
    if not operation:
        # Derive a bare operation from the legacy span name (e.g. "llm.chat" -> "chat").
        operation = name.split(".")[-1] if name else "chat"
    return f"{operation} {model}"


def build_resource(
    *,
    service_name: Optional[str],
    sdk_version: str,
    environment: Optional[str],
    service_version: Optional[str] = None,
) -> dict[str, Any]:
    """Build an OTel-style Resource attribute map."""
    resource: dict[str, Any] = {
        "telemetry.sdk.name": DEFAULT_SCOPE_NAME,
        "telemetry.sdk.version": sdk_version,
        "telemetry.sdk.language": "python",
    }
    if service_name:
        resource["service.name"] = service_name
    if service_version:
        resource["service.version"] = service_version
    if environment:
        resource["deployment.environment"] = environment
    return resource
