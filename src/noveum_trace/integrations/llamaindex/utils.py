"""
Utility helpers for the LlamaIndex integration.

All helpers are zero-impact: every public function absorbs exceptions internally
and returns a sensible default so a crashing utility can never break the host
application or the LlamaIndex query pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from noveum_trace.integrations.llamaindex import constants as C

logger = logging.getLogger(__name__)

# ``id_`` values look like ``"RetrieverQueryEngine.query-<uuid4>"``; strip the
# trailing UUID to recover a readable operation name.
_UUID_SUFFIX = re.compile(
    r"-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}" r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

_OPENAI_MODEL_PREFIXES = (
    "gpt",
    "o1",
    "o3",
    "o4",
    "text-",
    "ada",
    "babbage",
    "davinci",
)


def operation_from_span_id(id_: str) -> str:
    """Recover the ``Class.method`` operation name from an instrumentation id."""
    if not isinstance(id_, str):
        return str(id_)
    return _UUID_SUFFIX.sub("", id_)


def classify_operation(operation: str) -> str:
    """
    Best-effort span-type from an operation name (``"other"`` when unknown).

    The method name (the part after the last ``.``) is the strongest signal, so
    it is matched first — ``"RetrieverQueryEngine.query"`` classifies as a query,
    not a retrieval, even though the class name contains "Retriever".
    """
    lowered = operation.lower()
    method = lowered.rsplit(".", 1)[-1]
    for needle, span_type in C.OPERATION_TYPE_HINTS:
        if needle in method:
            return span_type
    for needle, span_type in C.OPERATION_TYPE_HINTS:
        if needle in lowered:
            return span_type
    return C.SPAN_TYPE_OTHER


def derive_provider(model: Optional[str]) -> Optional[str]:
    """Best-effort provider name from a model string."""
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


def truncate_text(text: Any, max_len: int = 8_192) -> str:
    """Stringify *text* and truncate to *max_len* characters with an ellipsis."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _get(source: Any, *keys: str) -> Any:
    """Read *keys* from a dict or object, returning the first non-None value."""
    for key in keys:
        value = (
            source.get(key) if isinstance(source, dict) else getattr(source, key, None)
        )
        if value is not None:
            return value
    return None


def extract_model_name(model_dict: Any) -> Optional[str]:
    """Pull the model name out of an event's ``model_dict`` payload."""
    if not model_dict:
        return None
    value = _get(model_dict, "model", "model_name", "model_id")
    return str(value) if value is not None else None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_token_usage(response: Any) -> dict[str, Optional[int]]:
    """
    Extract token counts from a LlamaIndex ``ChatResponse`` / ``CompletionResponse``.

    Token usage is not a first-class field on LlamaIndex responses; it lives on
    ``response.raw`` (provider-native, e.g. OpenAI ``usage``) or
    ``response.additional_kwargs``. Probes both. Returns ``None`` values when a
    count is unavailable and computes ``total`` when possible.
    """
    result: dict[str, Optional[int]] = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    if response is None:
        return result
    try:
        sources = [
            getattr(response, "additional_kwargs", None),
            getattr(response, "raw", None),
            _get(getattr(response, "raw", None) or {}, "usage"),
        ]
        for source in sources:
            if not source:
                continue
            if result["input_tokens"] is None:
                result["input_tokens"] = _coerce_int(
                    _get(source, "prompt_tokens", "input_tokens")
                )
            if result["output_tokens"] is None:
                result["output_tokens"] = _coerce_int(
                    _get(source, "completion_tokens", "output_tokens")
                )
            if result["total_tokens"] is None:
                result["total_tokens"] = _coerce_int(_get(source, "total_tokens"))
        if (
            result["total_tokens"] is None
            and result["input_tokens"] is not None
            and result["output_tokens"] is not None
        ):
            result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_token_usage failed: %s", exc)
    return result


def extract_query_str(query: Any) -> Optional[str]:
    """Return the query text from a ``str`` or a ``QueryBundle``-like object."""
    if query is None:
        return None
    if isinstance(query, str):
        return query
    value = _get(query, "query_str")
    return str(value) if value is not None else str(query)


def serialize_messages(messages: Any) -> Optional[list[dict[str, Any]]]:
    """Serialise a list of ``ChatMessage`` objects to ``[{role, content}, ...]``."""
    if not messages:
        return None
    result: list[dict[str, Any]] = []
    try:
        for message in messages:
            role = _get(message, "role")
            content = _get(message, "content")
            result.append(
                {
                    "role": str(getattr(role, "value", role)) if role else None,
                    "content": None if content is None else str(content),
                }
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("serialize_messages failed: %s", exc)
        return None
    return result or None


def extract_response_text(response: Any) -> Optional[str]:
    """Extract the assistant text from a ``ChatResponse`` / ``CompletionResponse``."""
    if response is None:
        return None
    try:
        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is not None:
                return str(content)
        text = getattr(response, "text", None)
        if text is not None:
            return str(text)
        return str(response)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_response_text failed: %s", exc)
        return None


def extract_node_scores(nodes: Any) -> list[Optional[float]]:
    """Return the similarity scores of a list of ``NodeWithScore`` objects."""
    scores: list[Optional[float]] = []
    if not nodes:
        return scores
    try:
        for node in nodes:
            score = getattr(node, "score", None)
            scores.append(float(score) if score is not None else None)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_node_scores failed: %s", exc)
    return scores


def extract_node_contents(nodes: Any, max_len: int = 2_048) -> list[str]:
    """Return truncated text content of a list of ``NodeWithScore`` objects."""
    contents: list[str] = []
    if not nodes:
        return contents
    try:
        for node in nodes:
            inner = getattr(node, "node", node)
            getter = getattr(inner, "get_content", None)
            text = getter() if callable(getter) else getattr(inner, "text", None)
            if text is not None:
                contents.append(truncate_text(text, max_len))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_node_contents failed: %s", exc)
    return contents
