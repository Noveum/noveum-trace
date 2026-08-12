"""
Shared helpers for framework integrations.

These helpers are framework-agnostic: span writes, defensive serialization,
timestamp coercion, provider derivation, token-usage normalisation, and cost
estimation. Integrations import from here instead of re-implementing the same
logic per framework.

Every public function absorbs its own exceptions and returns a sensible default,
so a failing helper can never break the host application.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "coerce_datetime",
    "derive_provider",
    "estimate_cost_safe",
    "extract_usage_tokens",
    "finish_span",
    "probe",
    "safe_serialize",
    "set_span_attributes",
    "stringify",
]


# ---------------------------------------------------------------------------
# Span writes
# ---------------------------------------------------------------------------


def set_span_attributes(span: Any, attributes: dict[str, Any]) -> None:
    """
    Write *attributes* onto *span*.

    Tries ``span.set_attributes`` first; on failure or absence, updates
    ``span.attributes`` when it supports ``.update`` (covers finished spans).
    """
    if not attributes or span is None:
        return
    try:
        setter = getattr(span, "set_attributes", None)
        if callable(setter):
            setter(attributes)
            return
    except Exception:
        pass
    try:
        attr_store = getattr(span, "attributes", None)
        if attr_store is not None and hasattr(attr_store, "update"):
            attr_store.update(attributes)
    except Exception as exc:
        logger.debug("set_span_attributes failed: %s", exc)


def finish_span(span: Any, end_time: Any = None) -> None:
    """Finish *span*, tolerating an already-finished span and never raising."""
    if span is None:
        return
    try:
        is_finished = getattr(span, "is_finished", None)
        if callable(is_finished) and is_finished():
            return
        if end_time is None:
            span.finish()
        else:
            span.finish(end_time)
    except Exception as exc:
        logger.debug("finish_span failed: %s", exc)


# ---------------------------------------------------------------------------
# Attribute probing / serialization
# ---------------------------------------------------------------------------


def probe(source: Any, *keys: str) -> Any:
    """Read *keys* from a dict or object, returning the first non-None value."""
    if source is None:
        return None
    for key in keys:
        try:
            value = (
                source.get(key)
                if isinstance(source, dict)
                else getattr(source, key, None)
            )
        except Exception:
            value = None
        if value is not None:
            return value
    return None


def stringify(value: Any) -> str:
    """
    Render *value* as a string without truncating it.

    Payloads such as system prompts and tool results routinely exceed any fixed
    character budget, so nothing is clipped here. Size limiting is the
    transport's responsibility, not the integration's.
    """
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception as exc:
        return f"<stringify_error:{type(value).__name__}:{exc}>"


def safe_serialize(value: Any, *, max_depth: int = 8) -> Any:
    """
    Recursively convert *value* into a JSON-serialisable structure.

    Primitives pass through; dict/list/tuple recurse; Pydantic v2/v1 objects use
    ``model_dump``/``dict``; objects exposing ``to_dict`` are converted;
    everything else falls back to ``str``. Depth is capped to guard against
    cyclic or deeply nested object graphs.
    """
    return _safe_serialize_inner(value, depth=0, max_depth=max_depth)


def _safe_serialize_inner(value: Any, depth: int, max_depth: int) -> Any:
    if depth >= max_depth:
        return f"<max_depth:{type(value).__name__}>"
    try:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {
                str(k): _safe_serialize_inner(v, depth + 1, max_depth)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [_safe_serialize_inner(item, depth + 1, max_depth) for item in value]
        for method in ("model_dump", "dict", "to_dict"):
            fn = getattr(value, method, None)
            if callable(fn):
                try:
                    return _safe_serialize_inner(fn(), depth + 1, max_depth)
                except Exception:
                    continue
        return str(value)
    except Exception as exc:
        return f"<serialization_error:{type(value).__name__}:{exc}>"


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def coerce_datetime(value: Any) -> Optional[datetime]:
    """
    Convert an ISO-8601 string (or epoch seconds) to a ``datetime``.

    Returns ``None`` when the value is missing or unparseable so callers can
    fall back to "now".
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except (OverflowError, OSError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Provider derivation
# ---------------------------------------------------------------------------

_OPENAI_MODEL_PREFIXES = (
    "gpt",
    "o1",
    "o3",
    "o4",
    "chatgpt",
    "text-",
    "ada",
    "babbage",
    "curie",
    "davinci",
)


def derive_provider(model: Optional[str]) -> Optional[str]:
    """
    Best-effort provider name from a model string.

    Recognises well-known prefixes, falls back to the left side of a LiteLLM
    ``"provider/model"`` string, and returns ``None`` when undeterminable.
    """
    if not model or not isinstance(model, str):
        return None
    lowered = model.lower()
    if any(lowered.startswith(prefix) for prefix in _OPENAI_MODEL_PREFIXES):
        return "openai"
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gemini") or lowered.startswith("models/gemini"):
        return "google"
    if lowered.startswith(("mistral", "mixtral")):
        return "mistral"
    if lowered.startswith(("llama", "meta-llama")):
        return "meta"
    if lowered.startswith("deepseek"):
        return "deepseek"
    if lowered.startswith(("command", "cohere")):
        return "cohere"
    if "/" in lowered:
        return lowered.split("/", 1)[0]
    return None


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*candidates: Any) -> Optional[int]:
    """Return the first candidate that coerces to an int, preserving zeros."""
    for candidate in candidates:
        coerced = _coerce_int(candidate)
        if coerced is not None:
            return coerced
    return None


USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "reasoning_tokens",
)


def extract_usage_tokens(usage: Any) -> dict[str, Optional[int]]:
    """
    Normalise a provider or framework usage payload to canonical token counts.

    Handles the flat OpenAI Agents shapes (``cached_input_tokens``,
    ``cache_write_input_tokens``), the nested OpenAI shapes
    (``input_tokens_details.cached_tokens``,
    ``output_tokens_details.reasoning_tokens``, and their
    ``prompt_tokens_details`` / ``completion_tokens_details`` aliases), and
    Chat-Completions naming (``prompt_tokens`` / ``completion_tokens``).

    ``total_tokens`` is computed from input + output when not supplied.
    """
    result: dict[str, Optional[int]] = dict.fromkeys(USAGE_KEYS)
    if usage is None:
        return result
    try:
        result["input_tokens"] = _coerce_int(
            probe(usage, "input_tokens", "prompt_tokens")
        )
        result["output_tokens"] = _coerce_int(
            probe(usage, "output_tokens", "completion_tokens")
        )
        result["total_tokens"] = _coerce_int(probe(usage, "total_tokens"))

        input_details = probe(usage, "input_tokens_details", "prompt_tokens_details")
        output_details = probe(
            usage, "output_tokens_details", "completion_tokens_details"
        )

        # ``or`` is wrong here: a genuine zero (no cache hit) must survive
        # rather than fall through to the nested lookup.
        result["cached_input_tokens"] = _first_int(
            probe(usage, "cached_input_tokens", "cached_tokens"),
            probe(input_details, "cached_tokens"),
        )
        result["cache_write_input_tokens"] = _first_int(
            probe(usage, "cache_write_input_tokens", "cache_creation_input_tokens"),
            probe(input_details, "cache_write_tokens"),
        )
        result["reasoning_tokens"] = _first_int(
            probe(usage, "reasoning_tokens"),
            probe(output_details, "reasoning_tokens"),
        )

        if (
            result["total_tokens"] is None
            and result["input_tokens"] is not None
            and result["output_tokens"] is not None
        ):
            result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    except Exception as exc:
        logger.debug("extract_usage_tokens failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def estimate_cost_safe(
    model: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> dict[str, Any]:
    """
    Estimate LLM cost via ``noveum_trace.utils.llm_utils.estimate_cost``.

    Returns a dict with ``input`` / ``output`` / ``total`` / ``currency`` keys,
    or an empty dict on any failure so callers can safely ``.get(...)``.
    """
    if not model:
        return {}
    if input_tokens is None and output_tokens is None:
        return {}
    try:
        from noveum_trace.utils.llm_utils import estimate_cost

        cost_info = estimate_cost(
            model,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
        )
        return {
            "input": cost_info.get("input_cost", 0.0),
            "output": cost_info.get("output_cost", 0.0),
            "total": cost_info.get("total_cost", 0.0),
            "currency": cost_info.get("currency", "USD"),
        }
    except Exception as exc:
        logger.debug("estimate_cost_safe failed for model=%s: %s", model, exc)
        return {}
