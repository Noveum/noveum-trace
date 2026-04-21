"""
Utility functions for CrewAI integration.

Safe serialization helpers, provider-agnostic
token extraction from LLM response objects, system-prompt extraction from a messages
list, tool-schema serialisation, duration helpers, a calculate_llm_cost wrapper,
:func:`resolve_agent_id`, :func:`set_span_attributes`, :func:`finish_span_common`
for handler mixins, and safe_getattr.

All helpers are designed to be zero-impact — every public function absorbs
exceptions internally and returns a sensible default so that a crashing utility
can never break the host application.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_STACKTRACE,
    ATTR_ERROR_TYPE,
    ATTR_STATUS_ERROR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# safe_getattr
# ---------------------------------------------------------------------------


def safe_getattr(obj: Any, *attrs: str, default: Any = None) -> Any:
    """
    Attribute probe that never raises.

    Supports dotted chains (each positional arg is one attribute hop) and
    also single dotted strings like ``"usage.input_tokens"``.

    Examples::

        safe_getattr(response, "usage", "input_tokens")
        safe_getattr(response, "usage.input_tokens")
        safe_getattr(event, "agent.role", default="unknown")
    """
    # Expand any dotted strings in the attrs tuple
    expanded: list[str] = []
    for attr in attrs:
        expanded.extend(attr.split("."))

    current = obj
    for attr in expanded:
        try:
            if current is None:
                return default
            if isinstance(current, dict):
                current = current.get(attr)
            else:
                current = getattr(current, attr, None)
        except Exception:
            return default
    return current if current is not None else default


# ---------------------------------------------------------------------------
# Safe serialization helpers
# ---------------------------------------------------------------------------


def safe_serialize(value: Any, *, max_depth: int = 8) -> Any:
    """
    Recursively convert *value* to a JSON-serialisable structure.

    Handles:
    - Primitives (int, float, bool, str, None) → returned as-is
    - dict / list / tuple → recursed
    - Objects with ``model_dump()`` or ``dict()`` (Pydantic v1/v2) → converted
    - Objects with ``to_dict()`` → converted
    - Objects with ``__dict__`` → public keys only, recursed
    - Anything else → ``str(value)``

    Depth is capped at *max_depth* to prevent stack overflows on deeply nested
    CrewAI task/agent graphs.
    """
    return _safe_serialize_inner(value, depth=0, max_depth=max_depth, _seen=set())


def _safe_serialize_inner(
    value: Any,
    depth: int,
    max_depth: int,
    _seen: set[int],
) -> Any:
    if depth >= max_depth:
        return f"<max_depth:{type(value).__name__}>"

    try:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        obj_id = id(value)

        if isinstance(value, dict):
            if obj_id in _seen:
                return "<circular_reference:dict>"
            _seen.add(obj_id)
            result = {
                str(k): _safe_serialize_inner(v, depth + 1, max_depth, _seen)
                for k, v in value.items()
            }
            _seen.discard(obj_id)
            return result

        if isinstance(value, (list, tuple)):
            if obj_id in _seen:
                return "<circular_reference:list>"
            _seen.add(obj_id)
            result_list = [
                _safe_serialize_inner(item, depth + 1, max_depth, _seen)
                for item in value
            ]
            _seen.discard(obj_id)
            return result_list

        # Pydantic v2 / v1
        for method in ("model_dump", "dict"):
            fn = getattr(value, method, None)
            if callable(fn):
                try:
                    return _safe_serialize_inner(fn(), depth + 1, max_depth, _seen)
                except Exception:
                    pass

        # Custom to_dict
        if callable(getattr(value, "to_dict", None)):
            try:
                return _safe_serialize_inner(
                    value.to_dict(), depth + 1, max_depth, _seen
                )
            except Exception:
                pass

        # Generic object — expose only public attributes
        if hasattr(value, "__dict__"):
            if obj_id in _seen:
                return f"<circular_reference:{type(value).__name__}>"
            _seen.add(obj_id)
            attrs = {
                k: _safe_serialize_inner(v, depth + 1, max_depth, _seen)
                for k, v in value.__dict__.items()
                if not k.startswith("_")
            }
            _seen.discard(obj_id)
            return attrs

        return str(value)

    except Exception as exc:
        return f"<serialization_error:{type(value).__name__}:{exc}>"


def safe_json_dumps(value: Any, *, fallback: str = "{}") -> str:
    """
    Serialise *value* to a JSON string.

    Uses :func:`safe_serialize` first so complex objects are handled
    before ``json.dumps`` is called. Returns *fallback* on any failure.
    """
    try:
        return json.dumps(safe_serialize(value), default=str)
    except Exception as exc:
        logger.debug("safe_json_dumps failed: %s", exc)
        return fallback


def truncate_str(text: str, max_len: int = 8192) -> str:
    """Return full string content"""
    if not isinstance(text, str):
        text = str(text)
    return text


# ---------------------------------------------------------------------------
# Token extraction — provider-agnostic
# ---------------------------------------------------------------------------

# Ordered candidate attribute paths for extracting token counts from whatever
# LLM response object CrewAI hands back (OpenAI, Anthropic, Google, Litellm …).
_INPUT_TOKEN_PATHS: tuple[tuple[str, ...], ...] = (
    ("usage", "input_tokens"),  # Anthropic
    ("usage", "prompt_tokens"),  # OpenAI / LiteLLM
    ("usage", "prompt_token_count"),  # Vertex AI
    ("usage", "inputTokenCount"),  # Bedrock
    ("usage", "input_token_count"),  # Watsonx
    ("usage_metadata", "prompt_token_count"),  # Google genai SDK
    ("prompt_token_count",),  # Gemini legacy
    ("input_tokens",),  # Anthropic top-level (some wrappers)
    ("prompt_tokens",),  # OpenAI top-level
)

_OUTPUT_TOKEN_PATHS: tuple[tuple[str, ...], ...] = (
    ("usage", "output_tokens"),  # Anthropic
    ("usage", "completion_tokens"),  # OpenAI / LiteLLM
    ("usage", "candidates_token_count"),  # Vertex AI
    ("usage", "outputTokenCount"),  # Bedrock
    ("usage", "generated_token_count"),  # Watsonx
    ("usage_metadata", "candidates_token_count"),  # Google genai SDK
    ("candidates_token_count",),  # Gemini legacy
    ("output_tokens",),  # Anthropic top-level
    ("completion_tokens",),  # OpenAI top-level
)

_TOTAL_TOKEN_PATHS: tuple[tuple[str, ...], ...] = (
    ("usage", "total_tokens"),
    ("usage", "total_token_count"),
    ("usage", "totalTokenCount"),
    ("usage_metadata", "total_token_count"),
    ("total_token_count",),
    ("total_tokens",),
)


def _probe_token(response: Any, paths: tuple[tuple[str, ...], ...]) -> Optional[int]:
    """Walk candidate attribute paths and return the first int-coercible value found."""
    for path in paths:
        val = safe_getattr(response, *path)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


def extract_token_usage(response: Any) -> dict[str, Optional[int]]:
    """
    Extract token counts from any provider's LLM response object.

    Returns a dict with keys ``input_tokens``, ``output_tokens``,
    ``total_tokens`` (values may be ``None`` when not available).

    Supports OpenAI, Anthropic, Google (Gemini / Vertex AI), AWS Bedrock,
    Watsonx, and any LiteLLM-wrapped response that normalises to the common
    ``usage`` namespace.
    """
    if response is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    try:
        input_t = _probe_token(response, _INPUT_TOKEN_PATHS)
        output_t = _probe_token(response, _OUTPUT_TOKEN_PATHS)
        total_t = _probe_token(response, _TOTAL_TOKEN_PATHS)

        # Compute total when not provided directly
        if total_t is None and input_t is not None and output_t is not None:
            total_t = input_t + output_t

        return {
            "input_tokens": input_t,
            "output_tokens": output_t,
            "total_tokens": total_t,
        }
    except Exception as exc:
        logger.debug("extract_token_usage failed: %s", exc)
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def extract_finish_reason(response: Any) -> Optional[str]:
    """
    Extract the finish/stop reason from a provider-agnostic response.

    Tries ``choices[0].finish_reason`` (OpenAI), ``stop_reason`` (Anthropic),
    ``candidates[0].finish_reason`` (Google), and ``finish_reason`` (LiteLLM).
    """
    try:
        # OpenAI / LiteLLM choices list
        choices = safe_getattr(response, "choices")
        if isinstance(choices, (list, tuple)) and choices:
            reason = safe_getattr(choices[0], "finish_reason")
            if reason is not None:
                return str(reason)

        # Anthropic
        reason = safe_getattr(response, "stop_reason")
        if reason is not None:
            return str(reason)

        # Google Gemini
        candidates = safe_getattr(response, "candidates")
        if isinstance(candidates, (list, tuple)) and candidates:
            reason = safe_getattr(candidates[0], "finish_reason")
            if reason is not None:
                return str(reason)

        # Generic fallback
        reason = safe_getattr(response, "finish_reason")
        if reason is not None:
            return str(reason)

    except Exception as exc:
        logger.debug("extract_finish_reason failed: %s", exc)

    return None


def extract_response_text(response: Any) -> Optional[str]:
    """
    Extract the generated text content from a provider-agnostic response.

    Tries the most common locations:
    - ``choices[0].message.content`` (OpenAI)
    - ``content[0].text`` (Anthropic)
    - ``candidates[0].content.parts[0].text`` (Google)
    - ``text`` / ``content`` top-level attributes
    """
    try:
        # OpenAI / LiteLLM
        choices = safe_getattr(response, "choices")
        if isinstance(choices, (list, tuple)) and choices:
            text = safe_getattr(choices[0], "message", "content")
            if text is not None:
                return str(text)

        # Anthropic
        content = safe_getattr(response, "content")
        if isinstance(content, (list, tuple)) and content:
            text = safe_getattr(content[0], "text")
            if text is not None:
                return str(text)
        if isinstance(content, str):
            return content

        # Google Gemini
        candidates = safe_getattr(response, "candidates")
        if isinstance(candidates, (list, tuple)) and candidates:
            parts = safe_getattr(candidates[0], "content", "parts")
            if isinstance(parts, (list, tuple)) and parts:
                text = safe_getattr(parts[0], "text")
                if text is not None:
                    return str(text)

        # Generic
        for attr in ("text", "output", "result"):
            val = safe_getattr(response, attr)
            if val and isinstance(val, str):
                return val

    except Exception as exc:
        logger.debug("extract_response_text failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# System-prompt extraction from a messages list
# ---------------------------------------------------------------------------


def extract_system_prompt(messages: Any) -> Optional[str]:
    """
    Find and return the system-role message content from a messages list.

    Handles both dict-style messages (``{"role": "system", "content": "…"}``)
    and object-style messages (``msg.role == "system"``).  When multiple
    system messages are present (rare but valid), they are joined with ``\\n``.

    Returns ``None`` when no system message is found or *messages* is empty.
    """
    if not messages:
        return None

    parts: list[str] = []
    try:
        iterable = messages if isinstance(messages, (list, tuple)) else [messages]
        for msg in iterable:
            role: Optional[str] = None
            content: Optional[str] = None

            if isinstance(msg, dict):
                role = msg.get("role")
                raw_content = msg.get("content")
            else:
                role = safe_getattr(msg, "role")
                raw_content = safe_getattr(msg, "content")

            if role and str(role).lower() == "system":
                if isinstance(raw_content, str):
                    content = raw_content
                elif isinstance(raw_content, (list, tuple)):
                    # Anthropic-style content blocks
                    texts = []
                    for block in raw_content:
                        t = (
                            block.get("text")
                            if isinstance(block, dict)
                            else safe_getattr(block, "text")
                        )
                        if t:
                            texts.append(str(t))
                    content = "\n".join(texts) if texts else None
                elif raw_content is not None:
                    content = str(raw_content)

                if content:
                    parts.append(content)

    except Exception as exc:
        logger.debug("extract_system_prompt failed: %s", exc)

    return "\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Tool-schema serialiser
# ---------------------------------------------------------------------------


def serialise_tools_list(tools: Any) -> list[dict[str, Any]]:
    """
    Normalise *tools* to a list of plain dicts (same shape as ``llm.tools`` JSON).

    Returns an empty list when *tools* is empty or serialisation fails.
    """
    if tools is None:
        return []
    try:
        tools_list = _resolve_tool_list(tools)
        if not tools_list:
            return []
        return [_serialise_single_tool(t) for t in tools_list]
    except Exception as exc:
        logger.debug("serialise_tools_list failed: %s", exc)
        return []


def serialize_tool_schema(tools: Any) -> Optional[str]:
    """
    Serialise any CrewAI / LangChain tool representation to a JSON string.

    Handles:
    - Plain ``list`` of dicts (already OpenAI-formatted)
    - ``list`` of objects with ``name``, ``description``, ``args_schema`` attrs
      (LangChain ``BaseTool`` / CrewAI ``Tool``)
    - Objects with a ``.tools`` list attribute
    - Single tool objects

    Returns ``None`` when *tools* is empty, ``None``, or serialisation fails.
    """
    serialised = serialise_tools_list(tools)
    if not serialised:
        return None
    try:
        return json.dumps(serialised, default=str)
    except Exception as exc:
        logger.debug("serialize_tool_schema failed: %s", exc)
        return None


def _tool_entry_display_name(entry: Any) -> str:
    """Best-effort tool name for span list attributes (flat or OpenAI function dict)."""
    if not isinstance(entry, dict):
        return "unknown"
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        return name
    fn = entry.get("function")
    if isinstance(fn, dict):
        inner = fn.get("name")
        if isinstance(inner, str) and inner.strip():
            return inner
    return "unknown"


def _tool_entry_display_description(entry: Any) -> str:
    """Best-effort description; aligns with LangChain default when missing."""
    if not isinstance(entry, dict):
        return "No description"
    desc = entry.get("description")
    if isinstance(desc, str) and desc.strip():
        return desc
    fn = entry.get("function")
    if isinstance(fn, dict):
        inner = fn.get("description")
        if isinstance(inner, str) and inner.strip():
            return inner
    return "No description"


def merge_available_tools_attributes(
    attrs: dict[str, Any],
    tools: Any,
    prefix: str,
) -> None:
    """
    Populate ``{prefix}.available_tools.count|names|descriptions|schemas`` on *attrs*.

    *prefix* is typically ``\"agent\"``, ``\"llm\"``, or ``\"mcp\"`` so keys match
    other integrations (e.g. LangChain ``llm.available_tools.names``).
    """
    serialised = serialise_tools_list(tools)
    if not serialised:
        return
    try:
        attrs[f"{prefix}.available_tools.count"] = len(serialised)
        attrs[f"{prefix}.available_tools.names"] = [
            _tool_entry_display_name(e) for e in serialised
        ]
        attrs[f"{prefix}.available_tools.descriptions"] = [
            _tool_entry_display_description(e) for e in serialised
        ]
    except Exception as exc:
        logger.debug("merge_available_tools_attributes count/names failed: %s", exc)
        return
    try:
        attrs[f"{prefix}.available_tools.schemas"] = json.dumps(serialised, default=str)
    except Exception as exc:
        logger.debug("merge_available_tools_attributes schemas failed: %s", exc)


def _resolve_tool_list(tools: Any) -> list[Any]:
    """Normalise *tools* to a plain list, resolving container wrappers."""
    if isinstance(tools, (list, tuple)):
        return list(tools)
    # Object with a .tools attribute (e.g. ToolsSchema)
    inner = safe_getattr(tools, "tools")
    if isinstance(inner, (list, tuple)):
        return list(inner)
    # Single tool object → wrap
    if tools is not None:
        return [tools]
    return []


def _serialise_single_tool(tool: Any) -> dict[str, Any]:
    """Convert a single tool to a plain serialisable dict."""
    if isinstance(tool, dict):
        return {k: safe_serialize(v) for k, v in tool.items()}

    result: dict[str, Any] = {}

    # Name / description — always present on CrewAI / LangChain tools
    name = safe_getattr(tool, "name")
    if name:
        result["name"] = str(name)

    description = safe_getattr(tool, "description")
    if description:
        result["description"] = str(description)

    # args_schema (Pydantic model on LangChain / CrewAI tools)
    args_schema = safe_getattr(tool, "args_schema")
    if args_schema is not None:
        try:
            # Pydantic v2
            if hasattr(args_schema, "model_json_schema"):
                result["parameters"] = args_schema.model_json_schema()
            # Pydantic v1
            elif hasattr(args_schema, "schema"):
                result["parameters"] = args_schema.schema()
        except Exception:
            pass

    # OpenAI-style function wrapper (e.g. {"type": "function", "function": {...}})
    for attr in ("function", "spec"):
        fn_spec = safe_getattr(tool, attr)
        if fn_spec is not None:
            result[attr] = safe_serialize(fn_spec)

    # Fallback: capture remaining public attrs not already captured
    if not result:
        result = safe_serialize(tool)

    return result or {"repr": str(tool)}


# ---------------------------------------------------------------------------
# Duration helper
# ---------------------------------------------------------------------------


def duration_ms(start: float, end: Optional[float] = None) -> float:
    """
    Compute elapsed wall-clock time in milliseconds.

    Args:
        start: ``time.time()`` value at operation start.
        end:   ``time.time()`` value at operation end. Defaults to *now*.

    Returns:
        Non-negative duration in milliseconds (rounded to 3 decimal places).
    """
    if end is None:
        end = time.time()
    delta = max(0.0, end - start) * 1000.0
    return round(delta, 3)


def monotonic_now() -> float:
    """Return current monotonic clock value (seconds). Use for duration pairing."""
    return time.monotonic()


def duration_ms_monotonic(start: float, end: Optional[float] = None) -> float:
    """
    Compute elapsed time in milliseconds using the monotonic clock.

    Prefer this over :func:`duration_ms` when *start* was captured with
    ``time.monotonic()`` (immune to wall-clock adjustments).
    """
    if end is None:
        end = time.monotonic()
    delta = max(0.0, end - start) * 1000.0
    return round(delta, 3)


# ---------------------------------------------------------------------------
# Span helpers (shared across CrewAI handler mixins)
# ---------------------------------------------------------------------------


def resolve_agent_id(source: Any, event: Any) -> Optional[str]:
    """
    Return ``event.agent_id`` or ``source.id`` / ``source.agent_id``, else ``None``.

    Also checks ``event.delegating_agent_id`` (A2A) after ``event.agent_id``.

    Contract matches the duplicated module-level helpers on memory, knowledge,
    MCP, guardrail, tool, reasoning, LLM, and A2A handler modules (not the
    :func:`noveum_trace.integrations.crewai._handlers_agent._resolve_agent_id`
    variant, which falls back to ``id(source)``).
    """
    raw = (
        safe_getattr(event, "agent_id")
        or safe_getattr(event, "delegating_agent_id")
        or safe_getattr(source, "id")
        or safe_getattr(source, "agent_id")
    )
    return str(raw) if raw is not None else None


def set_span_attributes(span: Any, attrs: dict[str, Any]) -> None:
    """
    Write *attrs* onto *span*.

    Tries ``span.set_attributes`` first; on failure or absence, updates
    ``span.attributes`` when it supports ``.update`` (covers finished spans).
    """
    if not attrs or span is None:
        return
    try:
        if hasattr(span, "set_attributes"):
            span.set_attributes(attrs)
            return
    except Exception:
        pass
    try:
        attr_store = getattr(span, "attributes", None)
        if attr_store is not None and hasattr(attr_store, "update"):
            attr_store.update(attrs)
    except Exception as exc:
        logger.debug("set_span_attributes failed: %s", exc)


def finish_span_common(
    span: Any,
    *,
    start_t: Any,
    status: str,
    status_attr: str,
    duration_attr: str,
    error: Any,
    extra_attrs: Optional[dict[str, Any]] = None,
    log_label: str = "finish_span_common",
) -> dict[str, Any]:
    """
    Build terminal attributes, write them, optionally mark span ERROR, then finish.

    Returns the merged attribute dict for callers that need to log derived keys.
    """
    attrs: dict[str, Any] = {status_attr: status}

    if start_t is not None:
        attrs[duration_attr] = duration_ms_monotonic(start_t)

    if error is not None:
        attrs[ATTR_ERROR_TYPE] = type(error).__name__
        attrs[ATTR_ERROR_MESSAGE] = str(error)
        tb = getattr(error, "__traceback__", None)
        if tb is not None:
            attrs[ATTR_ERROR_STACKTRACE] = "".join(traceback.format_tb(tb))

    if extra_attrs:
        attrs.update(extra_attrs)

    try:
        set_span_attributes(span, attrs)

        if status == ATTR_STATUS_ERROR and hasattr(span, "set_status"):
            try:
                from noveum_trace.core.span import SpanStatus

                span.set_status(SpanStatus.ERROR, str(error) if error else "")
            except Exception:
                pass

        if hasattr(span, "finish"):
            span.finish()
    except Exception:
        logger.debug(
            "%s span.finish error:\n%s",
            log_label,
            traceback.format_exc(),
        )

    return attrs


# ---------------------------------------------------------------------------
# calculate_llm_cost wrapper
# ---------------------------------------------------------------------------


def calculate_llm_cost(
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> dict[str, Any]:
    """
    Estimate LLM API cost by delegating to ``noveum_trace.utils.llm_utils.estimate_cost``.

    Args:
        model:         Model name string (e.g. ``"gpt-4o"``).
        input_tokens:  Number of prompt / input tokens (``None`` treated as 0).
        output_tokens: Number of completion / output tokens (``None`` treated as 0).

    Returns:
        Dict with keys ``input``, ``output``, ``total``, ``currency``.
        Returns an empty dict on any failure so callers can safely do
        ``cost.get("total", 0.0)``.
    """
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
        logger.debug("calculate_llm_cost failed for model=%s: %s", model, exc)
        return {}


# ---------------------------------------------------------------------------
# CrewAI event/object helpers
# ---------------------------------------------------------------------------


def extract_agent_info(agent: Any) -> dict[str, Any]:
    """
    Extract identifying metadata from a CrewAI ``Agent`` object or event payload.

    Probes the common attributes exposed by ``crewai.Agent``:
    ``role``, ``goal``, ``backstory``, ``id``, ``llm``, ``tools``,
    ``allow_delegation``, ``max_iter``.
    """
    info: dict[str, Any] = {}
    if agent is None:
        return info

    for attr in ("role", "goal", "backstory", "id"):
        val = safe_getattr(agent, attr)
        if val is not None:
            info[attr] = truncate_str(str(val), 512)

    # LLM identity
    llm = safe_getattr(agent, "llm")
    if llm is not None:
        model = safe_getattr(llm, "model_name") or safe_getattr(llm, "model")
        if model:
            info["llm_model"] = str(model)

    # Tool names
    tools = safe_getattr(agent, "tools")
    if tools:
        try:
            info["tool_names"] = [str(safe_getattr(t, "name") or t) for t in tools]
        except Exception:
            pass

    for attr in ("allow_delegation", "max_iter", "max_rpm"):
        val = safe_getattr(agent, attr)
        if val is not None:
            info[attr] = val

    return info


def extract_task_info(task: Any) -> dict[str, Any]:
    """
    Extract identifying metadata from a CrewAI ``Task`` object or event payload.

    Probes ``description``, ``expected_output``, ``id``, ``name``,
    ``agent`` (→ role only), ``context``, ``output_file``, ``human_input``.
    """
    info: dict[str, Any] = {}
    if task is None:
        return info

    for attr in ("id", "name"):
        val = safe_getattr(task, attr)
        if val is not None:
            info[attr] = str(val)

    for attr in ("description", "expected_output"):
        val = safe_getattr(task, attr)
        if val is not None:
            info[attr] = truncate_str(str(val), 1024)

    # Assigned agent role (avoid full agent serialisation)
    agent = safe_getattr(task, "agent")
    if agent is not None:
        role = safe_getattr(agent, "role")
        if role:
            info["agent_role"] = str(role)

    for attr in ("output_file", "human_input", "async_execution"):
        val = safe_getattr(task, attr)
        if val is not None:
            info[attr] = val

    return info


def extract_crew_info(crew: Any) -> dict[str, Any]:
    """
    Extract identifying metadata from a CrewAI ``Crew`` object.

    Captures ``name``, ``id``, agent roles list, task count, ``process``
    (sequential / hierarchical), and ``memory`` flag.
    """
    info: dict[str, Any] = {}
    if crew is None:
        return info

    for attr in ("name", "id"):
        val = safe_getattr(crew, attr)
        if val is not None:
            info[attr] = str(val)

    agents = safe_getattr(crew, "agents") or []
    if agents:
        try:
            info["agent_roles"] = [str(safe_getattr(a, "role") or a) for a in agents]
            info["agent_count"] = len(agents)
        except Exception:
            pass

    tasks = safe_getattr(crew, "tasks") or []
    if tasks:
        info["task_count"] = len(tasks)

    process = safe_getattr(crew, "process")
    if process is not None:
        # CrewAI Process enum has a .value attribute
        info["process"] = str(safe_getattr(process, "value") or process)

    for attr in ("memory", "verbose", "max_rpm"):
        val = safe_getattr(crew, attr)
        if val is not None:
            info[attr] = val

    return info


def extract_tool_result(result: Any) -> str:
    """
    Safely stringify a tool execution result for span attributes.

    Handles plain strings, dicts, Pydantic models, and arbitrary objects.
    Result is truncated to 4 096 characters to avoid bloating spans.
    """
    try:
        if result is None:
            return ""
        if isinstance(result, str):
            return truncate_str(result, 4096)
        return truncate_str(safe_json_dumps(result), 4096)
    except Exception as exc:
        logger.debug("extract_tool_result failed: %s", exc)
        return str(result)[:4096]


def extract_llm_model_from_agent(agent: Any) -> Optional[str]:
    """
    Best-effort extraction of the LLM model name from a CrewAI agent.

    CrewAI agents expose the backing LLM via ``.llm``; the LLM object may be
    a ``langchain_openai.ChatOpenAI``, ``litellm.LLM``, a plain string, or
    any provider-specific class.
    """
    try:
        llm = safe_getattr(agent, "llm")
        if llm is None:
            return None
        if isinstance(llm, str):
            return llm
        for attr in ("model_name", "model", "model_id"):
            val = safe_getattr(llm, attr)
            if val and isinstance(val, str):
                return val
    except Exception as exc:
        logger.debug("extract_llm_model_from_agent failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Message list helpers
# ---------------------------------------------------------------------------


def messages_to_json(messages: Any) -> Optional[str]:
    """
    Serialise a messages list (dicts or message objects) to a JSON string.

    Returns ``None`` when *messages* is empty or serialisation fails.
    """
    if not messages:
        return None
    try:
        serialised = safe_serialize(messages)
        return json.dumps(serialised, default=str)
    except Exception as exc:
        logger.debug("messages_to_json failed: %s", exc)
        return None


def count_messages_by_role(messages: Any) -> dict[str, int]:
    """
    Count messages grouped by role.

    Returns e.g. ``{"system": 1, "user": 3, "assistant": 2}``.
    """
    counts: dict[str, int] = {}
    if not messages:
        return counts
    try:
        iterable = messages if isinstance(messages, (list, tuple)) else [messages]
        for msg in iterable:
            role = (
                msg.get("role") if isinstance(msg, dict) else safe_getattr(msg, "role")
            )
            if role:
                key = str(role).lower()
                counts[key] = counts.get(key, 0) + 1
    except Exception as exc:
        logger.debug("count_messages_by_role failed: %s", exc)
    return counts
