"""
Utility helpers for the LlamaIndex integration.

Framework-agnostic helpers (serialization, provider derivation, token usage,
cost) live in :mod:`noveum_trace.integrations._common` and are re-exported here.
What remains below is specific to LlamaIndex shapes: instrumentation span ids,
``ChatMessage`` arrays, ``NodeWithScore`` lists and ``ToolMetadata``.

All helpers are zero-impact: every public function absorbs exceptions internally
and returns a sensible default so a crashing utility can never break the host
application or the LlamaIndex query pipeline.

Nothing here truncates. Retrieved node text and prompts routinely exceed any
fixed budget, and a clipped node is not usable for evaluation or replay.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from noveum_trace.integrations._common import (
    derive_provider,
    estimate_cost_safe,
    extract_usage_tokens,
    probe,
    safe_serialize,
    stringify,
)
from noveum_trace.integrations.llamaindex import constants as C

logger = logging.getLogger(__name__)

__all__ = [
    "classify_operation",
    "derive_provider",
    "estimate_cost_safe",
    "extract_model_name",
    "extract_node_contents",
    "extract_node_scores",
    "extract_query_str",
    "extract_response_text",
    "extract_system_prompt",
    "extract_token_usage",
    "extract_tool_metadata",
    "operation_from_span_id",
    "safe_serialize",
    "serialize_messages",
    "serialize_nodes",
    "stringify",
    "vector_dimensions",
]

# ``id_`` values look like ``"RetrieverQueryEngine.query-<uuid4>"``; strip the
# trailing UUID to recover a readable operation name.
_UUID_SUFFIX = re.compile(
    r"-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}" r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

_SYSTEM_ROLES = frozenset({"system", "developer"})


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


def extract_model_name(model_dict: Any) -> Optional[str]:
    """Pull the model name out of an event's ``model_dict`` payload."""
    if not model_dict:
        return None
    value = probe(model_dict, "model", "model_name", "model_id")
    return str(value) if value is not None else None


def extract_token_usage(response: Any) -> dict[str, Optional[int]]:
    """
    Extract token counts from a LlamaIndex ``ChatResponse`` / ``CompletionResponse``.

    Token usage is not a first-class field on LlamaIndex responses; it lives on
    ``response.raw`` (provider-native, e.g. OpenAI ``usage``) or
    ``response.additional_kwargs``. Probes both, then normalises through the
    shared extractor so cached and reasoning token breakdowns come through too.
    """
    if response is None:
        return dict.fromkeys(
            (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_input_tokens",
                "cache_write_input_tokens",
                "reasoning_tokens",
            )
        )
    merged: dict[str, Optional[int]] = {}
    try:
        raw = probe(response, "raw")
        sources = [
            probe(response, "additional_kwargs"),
            raw,
            probe(raw, "usage") if raw is not None else None,
        ]
        for source in sources:
            if not source:
                continue
            for key, value in extract_usage_tokens(source).items():
                if merged.get(key) is None and value is not None:
                    merged[key] = value
        input_tokens = merged.get("input_tokens")
        output_tokens = merged.get("output_tokens")
        if (
            merged.get("total_tokens") is None
            and input_tokens is not None
            and output_tokens is not None
        ):
            merged["total_tokens"] = input_tokens + output_tokens
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_token_usage failed: %s", exc)
    return {
        key: merged.get(key)
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "reasoning_tokens",
        )
    }


def extract_query_str(query: Any) -> Optional[str]:
    """Return the query text from a ``str`` or a ``QueryBundle``-like object."""
    if query is None:
        return None
    if isinstance(query, str):
        return query
    value = probe(query, "query_str")
    return str(value) if value is not None else stringify(query)


def serialize_messages(messages: Any) -> Optional[list[dict[str, Any]]]:
    """Serialise a list of ``ChatMessage`` objects to ``[{role, content}, ...]``."""
    if not messages:
        return None
    result: list[dict[str, Any]] = []
    try:
        for message in messages:
            role = probe(message, "role")
            content = probe(message, "content")
            entry: dict[str, Any] = {
                "role": str(getattr(role, "value", role)) if role else None,
                "content": None if content is None else stringify(content),
            }
            tool_calls = probe(message, "additional_kwargs")
            calls = probe(tool_calls, "tool_calls") if tool_calls else None
            if calls:
                entry["tool_calls"] = safe_serialize(calls)
            result.append(entry)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("serialize_messages failed: %s", exc)
        return None
    return result or None


def extract_system_prompt(messages: Any) -> Optional[str]:
    """Join the content of every ``system`` / ``developer`` message."""
    if not messages:
        return None
    try:
        collected: list[str] = []
        for message in messages:
            role = probe(message, "role")
            role_name = str(getattr(role, "value", role) or "").lower()
            if role_name not in _SYSTEM_ROLES:
                continue
            content = probe(message, "content")
            if content is not None:
                collected.append(stringify(content))
        joined = "\n\n".join(part for part in collected if part)
        return joined or None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_system_prompt failed: %s", exc)
        return None


def extract_response_text(response: Any) -> Optional[str]:
    """Extract the assistant text from a ``ChatResponse`` / ``CompletionResponse``."""
    if response is None:
        return None
    try:
        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is not None:
                return stringify(content)
        text = getattr(response, "text", None)
        if text is not None:
            return stringify(text)
        return stringify(response)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_response_text failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _node_text(node: Any) -> Optional[str]:
    inner = getattr(node, "node", node)
    getter = getattr(inner, "get_content", None)
    try:
        text = getter() if callable(getter) else getattr(inner, "text", None)
    except Exception:  # pragma: no cover - defensive
        text = getattr(inner, "text", None)
    return None if text is None else stringify(text)


def extract_node_scores(nodes: Any) -> list[Optional[float]]:
    """
    Return the similarity scores of a list of ``NodeWithScore`` objects.

    These are the retriever's own top-k scores — cosine similarity for most
    vector stores, or the store's native distance metric. The list is ordered
    best-match first, matching the order the nodes were returned in.
    """
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


def serialize_nodes(nodes: Any) -> list[dict[str, Any]]:
    """
    Serialise ``NodeWithScore`` objects to ``{id, score, text, metadata}`` entries.

    ``text`` is the node's full chunk content — the actual passage the retriever
    returned and the synthesizer read, not a summary or a reference. Node
    ``metadata`` (source file, page, custom tags) is carried alongside so a
    retrieved chunk can be traced back to its document.
    """
    entries: list[dict[str, Any]] = []
    if not nodes:
        return entries
    try:
        for node in nodes:
            inner = getattr(node, "node", node)
            entry: dict[str, Any] = {}
            node_id = probe(inner, "node_id", "id_", "id")
            if node_id is not None:
                entry["id"] = stringify(node_id)
            score = getattr(node, "score", None)
            if score is not None:
                entry["score"] = float(score)
            text = _node_text(node)
            if text is not None:
                entry["text"] = text
            metadata = probe(inner, "metadata", "extra_info")
            if metadata:
                entry["metadata"] = safe_serialize(metadata)
            entries.append(entry)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("serialize_nodes failed: %s", exc)
    return entries


def extract_node_contents(nodes: Any) -> list[str]:
    """Return the full text content of a list of ``NodeWithScore`` objects."""
    contents: list[str] = []
    if not nodes:
        return contents
    try:
        for node in nodes:
            text = _node_text(node)
            if text is not None:
                contents.append(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_node_contents failed: %s", exc)
    return contents


# ---------------------------------------------------------------------------
# Tools / embeddings
# ---------------------------------------------------------------------------


def extract_tool_metadata(tool: Any) -> dict[str, Any]:
    """Normalise a LlamaIndex ``ToolMetadata`` to ``{name, description}``."""
    entry: dict[str, Any] = {}
    if tool is None:
        return entry
    try:
        name = probe(tool, "name")
        if name is not None:
            entry["name"] = stringify(name)
        description = probe(tool, "description")
        if description is not None:
            entry["description"] = stringify(description)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_tool_metadata failed: %s", exc)
    return entry


def vector_dimensions(embeddings: Any) -> Optional[int]:
    """
    Return the dimensionality of the first embedding vector, if determinable.

    Only the width is recorded — the vectors themselves are never attached to a
    span. They are large, and a float array is not something a trace viewer or
    an evaluation can use.
    """
    if not embeddings:
        return None
    try:
        first = embeddings[0]
        return len(first) if hasattr(first, "__len__") else None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("vector_dimensions failed: %s", exc)
        return None
