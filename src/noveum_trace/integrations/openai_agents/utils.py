"""
Utility helpers for the OpenAI Agents SDK integration.

All helpers are zero-impact: every public function absorbs exceptions internally
and returns a sensible default so a crashing utility can never break the host
application or the surrounding ``agents`` run.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestamp coercion
# ---------------------------------------------------------------------------


def coerce_iso_datetime(value: Any) -> Optional[datetime]:
    """
    Convert an OpenAI Agents ISO-8601 timestamp string to a ``datetime``.

    The Agents SDK stores ``started_at`` / ``ended_at`` as ISO-8601 strings
    (e.g. ``"2026-08-03T12:34:56.789012+00:00"``). Returns ``None`` when the
    value is missing or unparseable so callers fall back to "now".
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
    "davinci",
    "babbage",
)


def derive_provider(model: Optional[str]) -> Optional[str]:
    """
    Best-effort provider name from a model string.

    Returns ``"openai"`` / ``"anthropic"`` / ``"google"`` / ``"mistral"`` for
    well-known prefixes, the left side of a LiteLLM ``"provider/model"`` string,
    or ``None`` when the provider cannot be determined.
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
    if "/" in lowered:
        return lowered.split("/", 1)[0]
    return None


# ---------------------------------------------------------------------------
# Safe serialization
# ---------------------------------------------------------------------------


def to_serialisable(value: Any, *, max_depth: int = 6) -> Any:
    """
    Recursively convert *value* into a JSON-serialisable structure.

    Primitives pass through; dict/list/tuple recurse; Pydantic v2/v1 objects use
    ``model_dump``/``dict``; objects with ``to_dict`` are converted; everything
    else falls back to ``str``. Depth is capped to guard against deep graphs.
    """
    return _to_serialisable_inner(value, depth=0, max_depth=max_depth)


def _to_serialisable_inner(value: Any, depth: int, max_depth: int) -> Any:
    if depth >= max_depth:
        return f"<max_depth:{type(value).__name__}>"
    try:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {
                str(k): _to_serialisable_inner(v, depth + 1, max_depth)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                _to_serialisable_inner(item, depth + 1, max_depth) for item in value
            ]
        for method in ("model_dump", "dict"):
            fn = getattr(value, method, None)
            if callable(fn):
                try:
                    return _to_serialisable_inner(fn(), depth + 1, max_depth)
                except Exception:
                    pass
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return _to_serialisable_inner(to_dict(), depth + 1, max_depth)
            except Exception:
                pass
        return str(value)
    except Exception as exc:  # pragma: no cover - defensive
        return f"<serialization_error:{type(value).__name__}:{exc}>"


def truncate_text(text: Any, max_len: int = 8_192) -> str:
    """Stringify *text* and truncate to *max_len* characters with an ellipsis."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


# ---------------------------------------------------------------------------
# Token / cost extraction
# ---------------------------------------------------------------------------


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _probe(source: Any, *keys: str) -> Any:
    """Read *keys* from a dict or object, returning the first non-None value."""
    for key in keys:
        value: Any = None
        if isinstance(source, dict):
            value = source.get(key)
        else:
            value = getattr(source, key, None)
        if value is not None:
            return value
    return None


def extract_usage_tokens(usage: Any) -> dict[str, Optional[int]]:
    """
    Normalise an Agents ``usage`` payload to input/output/total token counts.

    Accepts either the ``dict`` the SDK sets on generation/response spans (keys
    ``input_tokens`` / ``output_tokens``) or a usage object exposing the same
    attributes. Computes ``total_tokens`` when it is not supplied directly.
    """
    result: dict[str, Optional[int]] = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    if usage is None:
        return result
    try:
        result["input_tokens"] = _coerce_int(
            _probe(usage, "input_tokens", "prompt_tokens")
        )
        result["output_tokens"] = _coerce_int(
            _probe(usage, "output_tokens", "completion_tokens")
        )
        result["total_tokens"] = _coerce_int(_probe(usage, "total_tokens"))
        if (
            result["total_tokens"] is None
            and result["input_tokens"] is not None
            and result["output_tokens"] is not None
        ):
            result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_usage_tokens failed: %s", exc)
    return result


def extract_model_config(model_config: Any) -> dict[str, Any]:
    """Pull common sampling parameters (temperature/top_p/max_tokens) from a config."""
    params: dict[str, Any] = {}
    if not model_config:
        return params
    try:
        temperature = _probe(model_config, "temperature")
        if temperature is not None:
            params["temperature"] = temperature
        top_p = _probe(model_config, "top_p")
        if top_p is not None:
            params["top_p"] = top_p
        max_tokens = _probe(model_config, "max_tokens", "max_output_tokens")
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_model_config failed: %s", exc)
    return params


def estimate_cost_safe(
    model: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> dict[str, Any]:
    """
    Estimate LLM cost via ``noveum_trace.utils.llm_utils.estimate_cost``.

    Returns a dict with ``input`` / ``output`` / ``total`` / ``currency`` keys, or
    an empty dict on any failure so callers can safely ``.get(...)``.
    """
    if not model:
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
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("estimate_cost_safe failed for model=%s: %s", model, exc)
        return {}
