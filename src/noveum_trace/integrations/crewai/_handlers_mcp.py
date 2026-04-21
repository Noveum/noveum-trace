"""
MCP (Model Context Protocol) event handler mixin for NoveumCrewAIListener.

CrewAI supports MCP to connect agents to external tool servers over
stdio, SSE, or custom transports.  MCP events cover the full lifecycle:
connection establishment, tool execution, and configuration errors.

Span hierarchy::

    crewai.agent
      crewai.mcp.connection    ← one per MCP server connection attempt
      crewai.mcp.tool          ← one per MCP tool execution

Events handled:

  Connection lifecycle:
  - ``on_mcp_connection_started``    → open ``crewai.mcp`` span;
                                        capture server_name, url, transport type
  - ``on_mcp_connection_completed``  → close as SUCCESS; write duration_ms,
                                        available tools list
  - ``on_mcp_connection_failed``     → close as ERROR; write error_type, message

  Tool execution:
  - ``on_mcp_tool_execution_started``   → open ``crewai.mcp`` span;
                                           capture server_name, tool_name, arguments,
                                           ``mcp.available_tools.*`` when tools are present
  - ``on_mcp_tool_execution_completed`` → close as SUCCESS; write result, duration_ms
  - ``on_mcp_tool_execution_failed``    → close as ERROR; write error_type, message

  Config errors (no span lifecycle — annotate nearest open span):
  - ``on_mcp_config_fetch_failed``   → annotate agent/crew span with config
                                        fetch failure details

State consumed / mutated (declared in _CrewAIObserverState):
    _lock, _is_shutdown,
    _agent_spans, _crew_spans, _mcp_spans
"""

from __future__ import annotations

import logging
import re
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_AGENT_ROLE,
    ATTR_ERROR_STACKTRACE,
    ATTR_MCP_DURATION_MS,
    ATTR_MCP_INPUT,
    ATTR_MCP_KEY,
    ATTR_MCP_OUTPUT,
    ATTR_MCP_SERVER,
    ATTR_MCP_STATUS,
    ATTR_MCP_TOOL_NAME,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    MAX_TEXT_LENGTH,
    MAX_TOOL_OUTPUT_LENGTH,
    SPAN_MCP_CONNECTION,
    SPAN_MCP_TOOL,
)
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase
from noveum_trace.integrations.crewai.crewai_utils import (
    finish_span_common,
    merge_available_tools_attributes,
    monotonic_now,
)
from noveum_trace.integrations.crewai.crewai_utils import (
    resolve_agent_id as _resolve_agent_id,
)
from noveum_trace.integrations.crewai.crewai_utils import (
    safe_getattr,
    safe_json_dumps,
    set_span_attributes,
    truncate_str,
)

logger = logging.getLogger(__name__)

# mcp.operation values
_OP_CONNECTION = "connection"
_OP_TOOL_CALL = "tool_call"


class _MCPHandlersMixin(_CrewAIObserverMixinBase):
    """
    Handler methods for CrewAI MCP (Model Context Protocol) events.

    All public methods match the ``BaseEventListener`` callback signature::

        def on_mcp_connection_started(self, source, event): ...

    ``source`` is the MCP adapter or Agent; ``event`` carries the per-operation
    payload.  Every method is fully exception-shielded.

    All handlers no-op when ``capture_mcp`` is ``False`` on the listener.
    """

    # =========================================================================
    # MCP Connection — started / completed / failed
    # =========================================================================

    def on_mcp_connection_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.mcp`` span for an MCP server connection attempt.

        Attributes set at span open
        ---------------------------
        - ``mcp.key``           — unique key for this connection instance
        - ``mcp.operation``     — ``"connection"``
        - ``mcp.server``        — server name / identifier
        - ``mcp.url``           — server URL or endpoint address
        - ``mcp.transport``     — transport type: ``"stdio"`` | ``"sse"`` | ``"http"``
        - ``mcp.config``        — JSON snapshot of MCP server config (truncated;
                                  sensitive-looking dict keys redacted)
        - ``agent.role``        — role of the agent initiating the connection
        """
        if not self._is_active() or not self.capture_mcp:
            return
        try:
            mcp_key = _resolve_mcp_key(event, source)
            agent_id = _resolve_agent_id(source, event)

            attrs: dict[str, Any] = {
                ATTR_MCP_KEY: mcp_key,
                "mcp.operation": _OP_CONNECTION,
            }
            _populate_connection_attrs(attrs, source, event)

            start_t = monotonic_now()
            parent_span = self._get_agent_or_crew_span(agent_id)

            span = self._create_child_span(
                SPAN_MCP_CONNECTION,
                parent_span=parent_span,
                attributes=attrs,
            )

            with self._lock:
                self._mcp_spans[mcp_key] = {
                    "span": span,
                    "start_t": start_t,
                }

            logger.debug(
                "MCP connection span opened: mcp_key=%s server=%s",
                mcp_key,
                attrs.get(ATTR_MCP_SERVER, "?"),
            )
        except Exception:
            logger.debug("on_mcp_connection_started error:\n%s", traceback.format_exc())

    def on_mcp_connection_completed(self, source: Any, event: Any) -> None:
        """
        Close the MCP connection span as OK.

        Attributes written
        ------------------
        - ``mcp.available_tools``  — JSON list of tool names exposed by the server
        - ``mcp.tool_count``       — number of tools available after connection
        - ``mcp.status``           — ``"ok"``
        - ``mcp.duration_ms``      — wall-clock duration of the connection handshake
        """
        if not self._is_active() or not self.capture_mcp:
            return
        try:
            mcp_key = _resolve_mcp_key(event, source)
            extra: dict[str, Any] = {}

            tools = safe_getattr(event, "tools") or safe_getattr(
                event, "available_tools"
            )
            if tools is not None:
                try:
                    tool_list = list(tools)
                    tool_names = [str(safe_getattr(t, "name") or t) for t in tool_list]
                    extra["mcp.available_tools"] = safe_json_dumps(tool_names)
                    extra["mcp.tool_count"] = len(tool_names)
                except Exception as exc:
                    logger.debug(
                        "on_mcp_connection_completed: failed to serialize tools for mcp_key=%s: %s",
                        mcp_key,
                        exc,
                    )

            self._finish_mcp_span(mcp_key, ATTR_STATUS_SUCCESS, None, extra)
        except Exception:
            logger.debug(
                "on_mcp_connection_completed error:\n%s", traceback.format_exc()
            )

    def on_mcp_connection_failed(self, source: Any, event: Any) -> None:
        """
        Close the MCP connection span as ERROR.

        Attributes written
        ------------------
        - ``error.type``       — exception class name
        - ``error.message``    — error message
        - ``error.stacktrace`` — formatted traceback when available
        - ``mcp.error_type``   — short error category (``"connection_refused"``,
                                  ``"timeout"``, ``"auth"``, etc.) when provided
        - ``mcp.status``       — ``"error"``
        - ``mcp.duration_ms``  — wall-clock duration
        """
        if not self._is_active() or not self.capture_mcp:
            return
        try:
            mcp_key = _resolve_mcp_key(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            extra: dict[str, Any] = {}
            error_type = safe_getattr(event, "error_type") or safe_getattr(
                event, "failure_reason"
            )
            if error_type:
                extra["mcp.error_type"] = str(error_type)
            self._finish_mcp_span(mcp_key, ATTR_STATUS_ERROR, error, extra)
        except Exception:
            logger.debug("on_mcp_connection_failed error:\n%s", traceback.format_exc())

    # =========================================================================
    # MCP Tool Execution — started / completed / failed
    # =========================================================================

    def on_mcp_tool_execution_started(self, source: Any, event: Any) -> None:
        """
        Open a ``crewai.mcp`` span for an MCP tool call.

        Each call to a tool hosted on an MCP server gets its own span,
        separate from the connection span.

        Attributes set at span open
        ---------------------------
        - ``mcp.key``           — unique key for this tool execution instance
        - ``mcp.operation``     — ``"tool_call"``
        - ``mcp.server``        — MCP server name hosting the tool
        - ``mcp.tool_name``     — name of the tool being invoked
        - ``mcp.input``         — JSON of arguments passed to the tool
        - ``agent.role``        — role of the invoking agent (correlation)
        - ``mcp.available_tools.*`` — count, names, descriptions, schemas when
          ``tools`` / ``available_tools`` are present on *event* or *source*
        """
        if not self._is_active() or not self.capture_mcp:
            return
        try:
            mcp_key = _resolve_mcp_key(event, source)
            agent_id = _resolve_agent_id(source, event)

            attrs: dict[str, Any] = {
                ATTR_MCP_KEY: mcp_key,
                "mcp.operation": _OP_TOOL_CALL,
            }
            _populate_tool_attrs(attrs, source, event)

            start_t = monotonic_now()
            parent_span = self._get_agent_or_crew_span(agent_id)

            span = self._create_child_span(
                SPAN_MCP_TOOL,
                parent_span=parent_span,
                attributes=attrs,
            )

            with self._lock:
                self._mcp_spans[mcp_key] = {
                    "span": span,
                    "start_t": start_t,
                }

            logger.debug(
                "MCP tool span opened: mcp_key=%s server=%s tool=%s",
                mcp_key,
                attrs.get(ATTR_MCP_SERVER, "?"),
                attrs.get(ATTR_MCP_TOOL_NAME, "?"),
            )
        except Exception:
            logger.debug(
                "on_mcp_tool_execution_started error:\n%s", traceback.format_exc()
            )

    def on_mcp_tool_execution_completed(self, source: Any, event: Any) -> None:
        """
        Close the MCP tool execution span as OK.

        Attributes written
        ------------------
        - ``mcp.output``       — tool result text / JSON (≤ MAX_TOOL_OUTPUT_LENGTH)
        - ``mcp.result_type``  — type name of the result object (for schema tracking)
        - ``mcp.status``       — ``"ok"``
        - ``mcp.duration_ms``  — wall-clock duration of the tool call
        """
        if not self._is_active() or not self.capture_mcp:
            return
        try:
            mcp_key = _resolve_mcp_key(event, source)
            extra: dict[str, Any] = {}

            result = (
                safe_getattr(event, "result")
                or safe_getattr(event, "output")
                or safe_getattr(event, "response")
            )
            if result is not None:
                result_str = (
                    result if isinstance(result, str) else safe_json_dumps(result)
                )
                extra[ATTR_MCP_OUTPUT] = truncate_str(
                    result_str, MAX_TOOL_OUTPUT_LENGTH
                )
                extra["mcp.result_type"] = type(result).__name__

            self._finish_mcp_span(mcp_key, ATTR_STATUS_SUCCESS, None, extra)
        except Exception:
            logger.debug(
                "on_mcp_tool_execution_completed error:\n%s", traceback.format_exc()
            )

    def on_mcp_tool_execution_failed(self, source: Any, event: Any) -> None:
        """
        Close the MCP tool execution span as ERROR.

        Attributes written
        ------------------
        - ``error.type``       — exception class name
        - ``error.message``    — error message
        - ``error.stacktrace`` — formatted traceback when available
        - ``mcp.error_type``   — short category: ``"tool_not_found"``,
                                  ``"schema_validation"``, ``"timeout"``, etc.
        - ``mcp.status``       — ``"error"``
        - ``mcp.duration_ms``  — wall-clock duration
        """
        if not self._is_active() or not self.capture_mcp:
            return
        try:
            mcp_key = _resolve_mcp_key(event, source)
            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            extra: dict[str, Any] = {}
            error_type = safe_getattr(event, "error_type") or safe_getattr(
                event, "failure_reason"
            )
            if error_type:
                extra["mcp.error_type"] = str(error_type)
            self._finish_mcp_span(mcp_key, ATTR_STATUS_ERROR, error, extra)
        except Exception:
            logger.debug(
                "on_mcp_tool_execution_failed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Config fetch failed (no span lifecycle — annotate nearest open span)
    # =========================================================================

    def on_mcp_config_fetch_failed(self, source: Any, event: Any) -> None:
        """
        Annotate the nearest open agent or crew span with an MCP config error.

        ``MCPConfigFetchFailed`` fires when CrewAI cannot load or resolve the
        MCP server configuration (bad URL, missing env var, unreachable registry).
        There is no connection or tool span to attach to — the error is written
        to the agent span (or crew span as fallback) instead.

        Attributes written
        ------------------
        - ``mcp.config_fetch_failed``         — ``True``
        - ``mcp.config_fetch_error``          — error message string
        - ``mcp.config_fetch_error.type``     — exception class name
        - ``mcp.config_fetch_error.server``   — server name that failed to configure
        - ``mcp.config_fetch_error.config``   — partial config snapshot (truncated)
        """
        if not self._is_active() or not self.capture_mcp:
            return
        try:
            agent_id = _resolve_agent_id(source, event)
            span = self._get_agent_or_crew_span(agent_id)
            if span is None:
                logger.debug(
                    "on_mcp_config_fetch_failed: no open span to annotate "
                    "(agent_id=%s) — error dropped",
                    agent_id,
                )
                return

            error = safe_getattr(event, "error") or safe_getattr(event, "exception")
            error_str = (
                str(error) if error else str(safe_getattr(event, "message") or "")
            )

            err_attrs: dict[str, Any] = {"mcp.config_fetch_failed": True}
            if error_str:
                err_attrs["mcp.config_fetch_error"] = truncate_str(error_str, 1024)
            if error is not None:
                err_attrs["mcp.config_fetch_error.type"] = type(error).__name__
                tb = getattr(error, "__traceback__", None)
                if tb:
                    err_attrs[ATTR_ERROR_STACKTRACE] = "".join(traceback.format_tb(tb))

            server = (
                safe_getattr(event, "server_name")
                or safe_getattr(event, "server")
                or safe_getattr(source, "server_name")
            )
            if server:
                err_attrs["mcp.config_fetch_error.server"] = str(server)

            config = safe_getattr(event, "config") or safe_getattr(source, "config")
            if config is not None:
                err_attrs["mcp.config_fetch_error.config"] = truncate_str(
                    safe_json_dumps(_redact_config(config)), 512
                )

            set_span_attributes(span, err_attrs)
            logger.debug(
                "mcp.config_fetch_failed annotated on span: agent_id=%s server=%s",
                agent_id,
                server,
            )
        except Exception:
            logger.debug(
                "on_mcp_config_fetch_failed error:\n%s", traceback.format_exc()
            )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_agent_or_crew_span(
        self, agent_id: Optional[str], crew_id: Optional[str] = None
    ) -> Optional[Any]:
        """Return the best available parent span: agent → specific crew."""
        with self._lock:
            if agent_id and agent_id in self._agent_spans:
                return self._agent_spans[agent_id]
            if crew_id and crew_id in self._crew_spans:
                entry = self._crew_spans[crew_id]
                return entry.get("span") if isinstance(entry, dict) else None
            logger.debug(
                "_get_agent_or_crew_span: no matching parent (agent_id=%s crew_id=%s)",
                agent_id,
                crew_id,
            )
        return None

    def _finish_mcp_span(
        self,
        mcp_key: str,
        status: str,
        error: Any,
        extra_attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write final attributes onto the MCP span and close it."""
        with self._lock:
            entry = self._mcp_spans.pop(mcp_key, None)

        if entry is None:
            logger.debug("_finish_mcp_span: no open entry for mcp_key=%s", mcp_key)
            return

        span = entry["span"]
        start_t = entry.get("start_t")

        finish_span_common(
            span,
            start_t=start_t,
            status=status,
            status_attr=ATTR_MCP_STATUS,
            duration_attr=ATTR_MCP_DURATION_MS,
            error=error,
            extra_attrs=extra_attrs,
            log_label="_finish_mcp_span",
        )

        logger.debug("MCP span closed: mcp_key=%s status=%s", mcp_key, status)


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================

# Substrings matched against **dict keys** (lowercased) before writing config JSON.
_SENSITIVE_CONFIG_KEY_MARKERS = (
    "api_key",
    "authorization",
    "key",
    "password",
    "secret",
    "token",
)

_HIGH_RISK_CONTAINER_KEYS = frozenset(
    {"args", "command", "headers", "env", "environment", "params", "options"}
)
_CLI_SECRET_FLAG_RE = re.compile(r"^--(token|key|secret|password)$", re.IGNORECASE)
_HEADER_SECRET_RE = re.compile(r"(authorization\s*:|bearer\s+)", re.IGNORECASE)
_INLINE_SECRET_RE = re.compile(r"(token|key|secret|password)\s*=", re.IGNORECASE)


def _is_credential_like_string(value: str) -> bool:
    low = value.casefold()
    if any(marker in low for marker in _SENSITIVE_CONFIG_KEY_MARKERS):
        return True
    if _HEADER_SECRET_RE.search(value):
        return True
    if _INLINE_SECRET_RE.search(value):
        return True
    return False


def _redact_sequence(seq: Any, *, scan_strings: bool = False) -> Any:
    out: list[Any] = []
    redact_next = False
    for item in seq:
        if isinstance(item, str):
            if redact_next or _is_credential_like_string(item):
                out.append("<redacted>")
            else:
                out.append(item)
            redact_next = _CLI_SECRET_FLAG_RE.match(item.strip()) is not None
            continue

        out.append(_redact_config(item, scan_strings=scan_strings))
        redact_next = False

    if isinstance(seq, tuple):
        return tuple(out)
    return out


def _redact_config(value: Any, *, scan_strings: bool = False) -> Any:
    """
    Recursively copy *value*, replacing dict entries whose keys look like
    credentials with ``\"<redacted>\"`` so MCP configs can be logged safely.
    """
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            key_low = key.casefold()
            if any(m in key_low for m in _SENSITIVE_CONFIG_KEY_MARKERS):
                redacted[key] = "<redacted>"
                continue
            child_scan = scan_strings or key_low in _HIGH_RISK_CONTAINER_KEYS
            redacted[key] = _redact_config(v, scan_strings=child_scan)
        return redacted
    if isinstance(value, list):
        return _redact_sequence(value, scan_strings=scan_strings)
    if isinstance(value, tuple):
        return _redact_sequence(value, scan_strings=scan_strings)
    if isinstance(value, str) and scan_strings and _is_credential_like_string(value):
        return "<redacted>"
    return value


def _resolve_mcp_key(event: Any, source: Any) -> str:
    """
    Return a stable string key for this MCP operation instance.

    ``started_event_id`` is checked before ``event.id`` so completion/failure
    events that reference the original start event reuse the same key as the
    matching ``on_mcp_*_started`` handler.
    """
    return str(
        safe_getattr(event, "mcp_key")
        or safe_getattr(event, "connection_id")
        or safe_getattr(event, "execution_id")
        or safe_getattr(event, "started_event_id")
        or safe_getattr(event, "id")
        or safe_getattr(event, "run_id")
        or id(event)
    )


def _populate_connection_attrs(attrs: dict[str, Any], source: Any, event: Any) -> None:
    """Write connection-specific attributes into *attrs* in-place."""
    server = (
        safe_getattr(event, "server_name")
        or safe_getattr(event, "server")
        or safe_getattr(source, "server_name")
        or safe_getattr(source, "name")
    )
    if server:
        attrs[ATTR_MCP_SERVER] = truncate_str(str(server), 256)

    url = (
        safe_getattr(event, "url")
        or safe_getattr(event, "endpoint")
        or safe_getattr(source, "url")
    )
    if url:
        attrs["mcp.url"] = truncate_str(str(url), 512)

    transport = (
        safe_getattr(event, "transport")
        or safe_getattr(source, "transport")
        or safe_getattr(event, "transport_type")
    )
    if transport:
        attrs["mcp.transport"] = str(transport).lower()

    # Compact config snapshot (sensitive dict keys redacted before JSON)
    config = safe_getattr(event, "config") or safe_getattr(source, "config")
    if config is not None:
        attrs["mcp.config"] = truncate_str(safe_json_dumps(_redact_config(config)), 512)

    agent_role = safe_getattr(event, "agent_role") or safe_getattr(source, "role")
    if agent_role:
        attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)


def _populate_tool_attrs(attrs: dict[str, Any], source: Any, event: Any) -> None:
    """Write tool-execution-specific attributes into *attrs* in-place."""
    server = (
        safe_getattr(event, "server_name")
        or safe_getattr(event, "server")
        or safe_getattr(source, "server_name")
    )
    if server:
        attrs[ATTR_MCP_SERVER] = truncate_str(str(server), 256)

    tool_name = (
        safe_getattr(event, "tool_name")
        or safe_getattr(event, "tool")
        or safe_getattr(event, "name")
    )
    if tool_name:
        attrs[ATTR_MCP_TOOL_NAME] = truncate_str(str(tool_name), 256)

    arguments = (
        safe_getattr(event, "arguments")
        or safe_getattr(event, "args")
        or safe_getattr(event, "input")
        or safe_getattr(event, "params")
    )
    if arguments is not None:
        raw = arguments if isinstance(arguments, str) else safe_json_dumps(arguments)
        attrs[ATTR_MCP_INPUT] = truncate_str(raw, MAX_TEXT_LENGTH)

    agent_role = safe_getattr(event, "agent_role") or safe_getattr(source, "role")
    if agent_role:
        attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)

    tools = (
        safe_getattr(event, "tools")
        or safe_getattr(event, "available_tools")
        or safe_getattr(source, "tools")
        or safe_getattr(source, "available_tools")
    )
    if tools is not None:
        merge_available_tools_attributes(attrs, tools, "mcp")
