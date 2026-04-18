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
                                           capture server_name, tool_name, arguments
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
import traceback
from typing import Any, Optional

from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_AGENT_ROLE,
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_STACKTRACE,
    ATTR_ERROR_TYPE,
    ATTR_MCP_DURATION_MS,
    ATTR_MCP_INPUT,
    ATTR_MCP_KEY,
    ATTR_MCP_OUTPUT,
    ATTR_MCP_SERVER,
    ATTR_MCP_STATUS,
    ATTR_MCP_TOOL_NAME,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    MAX_DESCRIPTION_LENGTH,
    MAX_TEXT_LENGTH,
    MAX_TOOL_OUTPUT_LENGTH,
    SPAN_MCP_CONNECTION,
    SPAN_MCP_TOOL,
)
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase
from noveum_trace.integrations.crewai.crewai_utils import (
    duration_ms_monotonic,
    monotonic_now,
    safe_getattr,
    safe_json_dumps,
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
        - ``mcp.config``        — JSON snapshot of MCP server config (truncated)
        - ``agent.role``        — role of the agent initiating the connection
        """
        if not self._is_active():
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
            logger.debug(
                "on_mcp_connection_started error:\n%s", traceback.format_exc()
            )

    def on_mcp_connection_completed(self, source: Any, event: Any) -> None:
        """
        Close the MCP connection span as SUCCESS.

        Attributes written
        ------------------
        - ``mcp.available_tools``  — JSON list of tool names exposed by the server
        - ``mcp.tool_count``       — number of tools available after connection
        - ``mcp.status``           — ``"success"``
        - ``mcp.duration_ms``      — wall-clock duration of the connection handshake
        """
        if not self._is_active():
            return
        try:
            mcp_key = _resolve_mcp_key(event, source)
            extra: dict[str, Any] = {}

            tools = (
                safe_getattr(event, "tools")
                or safe_getattr(event, "available_tools")
            )
            if tools is not None:
                try:
                    tool_list = list(tools)
                    tool_names = [
                        str(safe_getattr(t, "name") or t) for t in tool_list
                    ]
                    extra["mcp.available_tools"] = safe_json_dumps(tool_names)
                    extra["mcp.tool_count"] = len(tool_names)
                except Exception:
                    pass

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
        if not self._is_active():
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
                "on_mcp_connection_failed error:\n%s", traceback.format_exc()
            )

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
        """
        if not self._is_active():
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
        Close the MCP tool execution span as SUCCESS.

        Attributes written
        ------------------
        - ``mcp.output``       — tool result text / JSON (≤ MAX_TOOL_OUTPUT_LENGTH)
        - ``mcp.result_type``  — type name of the result object (for schema tracking)
        - ``mcp.status``       — ``"success"``
        - ``mcp.duration_ms``  — wall-clock duration of the tool call
        """
        if not self._is_active():
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
                    result
                    if isinstance(result, str)
                    else safe_json_dumps(result)
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
        if not self._is_active():
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
        if not self._is_active():
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
                str(error) if error else
                str(safe_getattr(event, "message") or "")
            )

            err_attrs: dict[str, Any] = {"mcp.config_fetch_failed": True}
            if error_str:
                err_attrs["mcp.config_fetch_error"] = truncate_str(
                    error_str, 1024
                )
            if error is not None:
                err_attrs["mcp.config_fetch_error.type"] = type(error).__name__
                tb = getattr(error, "__traceback__", None)
                if tb:
                    err_attrs[ATTR_ERROR_STACKTRACE] = "".join(
                        traceback.format_tb(tb)
                    )

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
                    safe_json_dumps(config), 512
                )

            _set_span_attributes(span, err_attrs)
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

    def _get_agent_or_crew_span(self, agent_id: Optional[str]) -> Any:
        """Return the best available parent span: agent → any crew → None."""
        with self._lock:
            if agent_id and agent_id in self._agent_spans:
                return self._agent_spans[agent_id]
            # Fall back to any open crew span
            for entry in self._crew_spans.values():
                return entry.get("span")
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
            logger.debug(
                "_finish_mcp_span: no open entry for mcp_key=%s", mcp_key
            )
            return

        span = entry["span"]
        start_t = entry.get("start_t")

        attrs: dict[str, Any] = {ATTR_MCP_STATUS: status}

        if start_t is not None:
            attrs[ATTR_MCP_DURATION_MS] = duration_ms_monotonic(start_t)

        if error is not None:
            attrs[ATTR_ERROR_TYPE] = type(error).__name__
            attrs[ATTR_ERROR_MESSAGE] = str(error)
            tb = getattr(error, "__traceback__", None)
            if tb is not None:
                attrs[ATTR_ERROR_STACKTRACE] = "".join(traceback.format_tb(tb))

        if extra_attrs:
            attrs.update(extra_attrs)

        try:
            if hasattr(span, "set_attributes"):
                span.set_attributes(attrs)
            elif hasattr(span, "attributes"):
                span.attributes.update(attrs)

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
                "_finish_mcp_span span.finish() error:\n%s",
                traceback.format_exc(),
            )

        logger.debug(
            "MCP span closed: mcp_key=%s status=%s", mcp_key, status
        )


# =============================================================================
# Module-level helpers (pure functions — no state access)
# =============================================================================


def _resolve_mcp_key(event: Any, source: Any) -> str:
    """Return a stable string key for this MCP operation instance."""
    return str(
        safe_getattr(event, "mcp_key")
        or safe_getattr(event, "connection_id")
        or safe_getattr(event, "execution_id")
        or safe_getattr(event, "id")
        or safe_getattr(event, "run_id")
        or id(event)
    )


def _resolve_agent_id(source: Any, event: Any) -> Optional[str]:
    """Return the agent_id for this MCP event, or ``None``."""
    raw = (
        safe_getattr(event, "agent_id")
        or safe_getattr(source, "id")
        or safe_getattr(source, "agent_id")
    )
    return str(raw) if raw is not None else None


def _populate_connection_attrs(
    attrs: dict[str, Any], source: Any, event: Any
) -> None:
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

    # Compact config snapshot (exclude secrets / tokens)
    config = safe_getattr(event, "config") or safe_getattr(source, "config")
    if config is not None:
        attrs["mcp.config"] = truncate_str(safe_json_dumps(config), 512)

    agent_role = (
        safe_getattr(event, "agent_role")
        or safe_getattr(source, "role")
    )
    if agent_role:
        attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)


def _populate_tool_attrs(
    attrs: dict[str, Any], source: Any, event: Any
) -> None:
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
        raw = (
            arguments
            if isinstance(arguments, str)
            else safe_json_dumps(arguments)
        )
        attrs[ATTR_MCP_INPUT] = truncate_str(raw, MAX_TEXT_LENGTH)

    agent_role = (
        safe_getattr(event, "agent_role")
        or safe_getattr(source, "role")
    )
    if agent_role:
        attrs[ATTR_AGENT_ROLE] = truncate_str(str(agent_role), 256)


def _set_span_attributes(span: Any, attrs: dict[str, Any]) -> None:
    """Write *attrs* to *span* via ``set_attributes`` or direct dict update."""
    if not attrs or span is None:
        return
    try:
        if hasattr(span, "set_attributes"):
            span.set_attributes(attrs)
        elif hasattr(span, "attributes"):
            span.attributes.update(attrs)
    except Exception as exc:
        logger.debug("_set_span_attributes failed: %s", exc)
