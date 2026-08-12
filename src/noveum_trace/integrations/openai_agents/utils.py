"""
Utility helpers for the OpenAI Agents SDK integration.

Framework-agnostic helpers (serialization, timestamps, provider derivation,
token usage, cost) live in :mod:`noveum_trace.integrations._common` and are
re-exported here. What remains below is specific to the shapes the Agents SDK
puts on its ``SpanData`` objects: Chat-Completions message arrays and
Responses-API output items.

All helpers are zero-impact: every public function absorbs exceptions internally
and returns a sensible default so a crashing utility can never break the host
application or the surrounding ``agents`` run.

Nothing here truncates. Payloads such as system prompts and tool results
routinely exceed any fixed budget, and a clipped payload is not usable for
evaluation or replay.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from noveum_trace.integrations._common import (
    coerce_datetime,
    derive_provider,
    estimate_cost_safe,
    extract_usage_tokens,
    probe,
    safe_serialize,
    stringify,
)

logger = logging.getLogger(__name__)

__all__ = [
    "coerce_datetime",
    "coerce_iso_datetime",
    "derive_provider",
    "estimate_cost_safe",
    "extract_message_text",
    "extract_model_config",
    "extract_response_output",
    "extract_system_prompt",
    "extract_tool_calls",
    "extract_tool_schemas",
    "extract_usage_tokens",
    "safe_serialize",
    "stringify",
    "to_serialisable",
]

# Backwards-compatible aliases for the names this module exported before the
# shared helpers were hoisted into ``integrations._common``.
coerce_iso_datetime = coerce_datetime
to_serialisable = safe_serialize

_SYSTEM_ROLES = frozenset({"system", "developer"})


# ---------------------------------------------------------------------------
# Message / content extraction
# ---------------------------------------------------------------------------


def _content_to_text(content: Any) -> str:
    """
    Flatten a message ``content`` value to plain text.

    Content is either a bare string or a list of typed parts
    (``{"type": "output_text", "text": ...}``, ``{"type": "input_text", ...}``,
    ``{"type": "refusal", "refusal": ...}``). Non-text parts (images, audio) are
    skipped — their payloads belong on dedicated attributes, not in the text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = probe(part, "text", "refusal")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(p for p in parts if p)
    text = probe(content, "text", "refusal")
    return text if isinstance(text, str) else stringify(content)


def _iter_messages(messages: Any) -> list[Any]:
    if messages is None:
        return []
    if isinstance(messages, (list, tuple)):
        return list(messages)
    return [messages]


def extract_system_prompt(messages: Any) -> Optional[str]:
    """
    Join the content of every ``system`` / ``developer`` message.

    Returns ``None`` when the message array carries no system instructions.
    """
    try:
        collected = [
            _content_to_text(probe(message, "content"))
            for message in _iter_messages(messages)
            if str(probe(message, "role") or "").lower() in _SYSTEM_ROLES
        ]
        joined = "\n\n".join(part for part in collected if part)
        return joined or None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_system_prompt failed: %s", exc)
        return None


def extract_message_text(messages: Any, *, skip_system: bool = True) -> Optional[str]:
    """
    Join the text content of a message array into a single string.

    System/developer turns are skipped by default because they are reported
    separately as ``llm.system_prompt``.
    """
    try:
        collected: list[str] = []
        for message in _iter_messages(messages):
            role = str(probe(message, "role") or "").lower()
            if skip_system and role in _SYSTEM_ROLES:
                continue
            text = _content_to_text(probe(message, "content"))
            if text:
                collected.append(f"{role}: {text}" if role else text)
        joined = "\n\n".join(collected)
        return joined or None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_message_text failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tool calls / tool schemas
# ---------------------------------------------------------------------------


def _tool_call_entry(raw: Any) -> Optional[dict[str, Any]]:
    """Normalise one Chat-Completions or Responses tool call to a flat dict."""
    function = probe(raw, "function")
    name = probe(function, "name") if function is not None else None
    name = name if name is not None else probe(raw, "name")
    arguments = probe(function, "arguments") if function is not None else None
    arguments = arguments if arguments is not None else probe(raw, "arguments")
    call_id = probe(raw, "call_id", "id")
    if name is None and arguments is None:
        return None
    entry: dict[str, Any] = {"name": stringify(name) if name is not None else None}
    if call_id is not None:
        entry["call_id"] = stringify(call_id)
    if arguments is not None:
        entry["arguments"] = (
            arguments if isinstance(arguments, str) else safe_serialize(arguments)
        )
    return entry


def extract_tool_calls(output: Any) -> list[dict[str, Any]]:
    """
    Collect tool calls from a Chat-Completions ``output`` message array.

    Handles both the ``message.tool_calls`` shape and Responses-style
    ``function_call`` items appearing directly in the array.
    """
    calls: list[dict[str, Any]] = []
    try:
        for item in _iter_messages(output):
            item_type = str(probe(item, "type") or "")
            if item_type in ("function_call", "custom_tool_call"):
                entry = _tool_call_entry(item)
                if entry is not None:
                    calls.append(entry)
                continue
            for raw in _iter_messages(probe(item, "tool_calls")):
                entry = _tool_call_entry(raw)
                if entry is not None:
                    calls.append(entry)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_tool_calls failed: %s", exc)
    return calls


def extract_tool_schemas(tools: Any) -> list[dict[str, Any]]:
    """
    Normalise a Responses ``response.tools`` list to ``{name, type, description}``.

    Falls back to a serialised form for tool kinds that expose no name (hosted
    tools such as ``web_search`` carry only a ``type``).
    """
    schemas: list[dict[str, Any]] = []
    try:
        for tool in _iter_messages(tools):
            if isinstance(tool, str):
                schemas.append({"name": tool})
                continue
            entry: dict[str, Any] = {}
            name = probe(tool, "name")
            if name is not None:
                entry["name"] = stringify(name)
            tool_type = probe(tool, "type")
            if tool_type is not None:
                entry["type"] = stringify(tool_type)
            description = probe(tool, "description")
            if description is not None:
                entry["description"] = stringify(description)
            parameters = probe(tool, "parameters")
            if parameters is not None:
                entry["parameters"] = safe_serialize(parameters)
            schemas.append(entry or {"tool": safe_serialize(tool)})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_tool_schemas failed: %s", exc)
    return schemas


# ---------------------------------------------------------------------------
# Responses-API output items
# ---------------------------------------------------------------------------


def extract_response_output(response: Any) -> dict[str, Any]:
    """
    Split a Responses ``response.output`` item list into its useful parts.

    Returns a dict with ``text`` (assistant message text), ``tool_calls``
    (function-call items) and ``reasoning`` (reasoning summaries). Missing
    pieces are omitted rather than reported as empty.
    """
    result: dict[str, Any] = {}
    if response is None:
        return result
    try:
        items = _iter_messages(probe(response, "output"))
        texts: list[str] = []
        reasoning: list[str] = []
        for item in items:
            item_type = str(probe(item, "type") or "")
            if item_type == "message":
                text = _content_to_text(probe(item, "content"))
                if text:
                    texts.append(text)
            elif item_type == "reasoning":
                for entry in _iter_messages(probe(item, "summary")):
                    text = probe(entry, "text")
                    if isinstance(text, str) and text:
                        reasoning.append(text)

        output_text = probe(response, "output_text")
        if isinstance(output_text, str) and output_text:
            result["text"] = output_text
        elif texts:
            result["text"] = "\n\n".join(texts)

        tool_calls = extract_tool_calls(items)
        if tool_calls:
            result["tool_calls"] = tool_calls
        if reasoning:
            result["reasoning"] = "\n\n".join(reasoning)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_response_output failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


def extract_model_config(model_config: Any) -> dict[str, Any]:
    """
    Pull sampling parameters and reasoning effort from a generation's config.

    ``model_config`` is ``ModelSettings.to_traceable_dict()`` plus ``base_url``;
    it carries no tool list, so available tools are read from the enclosing
    agent span (``agent.tools``) or from ``response.tools``.
    """
    params: dict[str, Any] = {}
    if not model_config:
        return params
    try:
        temperature = probe(model_config, "temperature")
        if temperature is not None:
            params["temperature"] = temperature
        top_p = probe(model_config, "top_p")
        if top_p is not None:
            params["top_p"] = top_p
        max_tokens = probe(model_config, "max_tokens", "max_output_tokens")
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        reasoning = probe(model_config, "reasoning")
        effort = probe(reasoning, "effort") if reasoning is not None else None
        if effort is not None:
            params["reasoning_effort"] = stringify(effort)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("extract_model_config failed: %s", exc)
    return params
