"""
Noveum trace processor for the OpenAI Agents SDK.

:class:`NoveumTraceProcessor` implements the ``agents.tracing.TracingProcessor``
interface. It maps OpenAI Agents traces onto Noveum traces and OpenAI Agents
spans onto Noveum spans, preserving parent-child ordering, and attaches
``gen_ai``-compatible ``llm.*`` / ``tool.*`` / ``agent.*`` attributes.

Register it with the Agents SDK::

    import noveum_trace
    from agents import add_trace_processor
    from noveum_trace.integrations.openai_agents import NoveumTraceProcessor

    noveum_trace.init(project="my-project", api_key="...")
    add_trace_processor(NoveumTraceProcessor())

or use the convenience factory :func:`setup_openai_agents_tracing`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from noveum_trace.integrations.openai_agents import constants as C
from noveum_trace.integrations.openai_agents.utils import (
    coerce_iso_datetime,
    derive_provider,
    estimate_cost_safe,
    extract_model_config,
    extract_usage_tokens,
    to_serialisable,
    truncate_text,
)

logger = logging.getLogger(__name__)

try:
    from agents.tracing import TracingProcessor

    OPENAI_AGENTS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra installed

    OPENAI_AGENTS_AVAILABLE = False

    class TracingProcessor:  # type: ignore[no-redef]
        """
        Fallback stub used when ``openai-agents`` is not installed.

        Mirrors the lifecycle hooks of ``agents.tracing.TracingProcessor`` so the
        module imports (and unit tests run) without the optional dependency.
        """

        def on_trace_start(self, trace: Any) -> None: ...

        def on_trace_end(self, trace: Any) -> None: ...

        def on_span_start(self, span: Any) -> None: ...

        def on_span_end(self, span: Any) -> None: ...

        def shutdown(self) -> None: ...

        def force_flush(self) -> None: ...


def _span_data_type(span_data: Any) -> str:
    """Return the ``SpanData.type`` string (``"generation"``, ``"agent"``, …)."""
    span_type = getattr(span_data, "type", None)
    return str(span_type) if span_type is not None else ""


def _set_attributes(span: Any, attributes: dict[str, Any]) -> None:
    """Write *attributes* onto *span*, tolerating already-finished spans."""
    if not attributes or span is None:
        return
    try:
        span.set_attributes(attributes)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("failed to set span attributes: %s", exc)


def _finish_span(span: Any, end_time: Any) -> None:
    """Finish *span* with *end_time*, never raising into the host application."""
    if span is None:
        return
    try:
        span.finish(end_time)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("failed to finish span: %s", exc)


class NoveumTraceProcessor(TracingProcessor):
    """
    Bridge OpenAI Agents SDK tracing into Noveum.

    Instances are safe to register once via ``agents.add_trace_processor``; the
    processor keys concurrent traces/spans by their OpenAI ids, so multiple agent
    runs (including async) map cleanly onto separate Noveum traces.

    Args:
        client: Explicit :class:`~noveum_trace.core.client.NoveumClient`. When
            omitted the processor uses the globally initialised client
            (``noveum_trace.init(...)``).
        capture_inputs: Capture raw tool/function/custom inputs. Off by default
            (privacy-safe) because inputs can contain sensitive payloads.
        capture_outputs: Capture raw tool/function/LLM outputs. Off by default.
        capture_llm_messages: Capture full LLM prompt/response message arrays.
            Off by default — this is the most sensitive payload.
        capture_tool_schemas: Capture structural metadata such as an agent's tool
            and handoff names. On by default (names/structure, not argument values).
        capture_trace_metadata: Copy the OpenAI trace ``metadata`` / ``group_id``
            onto the Noveum trace. On by default (user-supplied workflow metadata).
        capture_cost: Estimate and attach LLM cost from model + token counts.
        trace_name_prefix: Prefix used when the OpenAI trace has no workflow name.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        capture_inputs: bool = False,
        capture_outputs: bool = False,
        capture_llm_messages: bool = False,
        capture_tool_schemas: bool = True,
        capture_trace_metadata: bool = True,
        capture_cost: bool = True,
        trace_name_prefix: str = C.DEFAULT_TRACE_NAME_PREFIX,
    ) -> None:
        self._injected_client = client
        self.capture_inputs = capture_inputs
        self.capture_outputs = capture_outputs
        self.capture_llm_messages = capture_llm_messages
        self.capture_tool_schemas = capture_tool_schemas
        self.capture_trace_metadata = capture_trace_metadata
        self.capture_cost = capture_cost
        self._trace_name_prefix = trace_name_prefix
        self._lock = threading.RLock()
        self._traces: dict[str, Any] = {}
        self._spans: dict[str, Any] = {}
        self._is_shutdown = False

    # ------------------------------------------------------------------
    # Client resolution
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return the injected client, else the global client, else ``None``."""
        if self._is_shutdown:
            return None
        if self._injected_client is not None:
            return self._injected_client
        try:
            from noveum_trace import get_client, is_initialized

            if is_initialized():
                return get_client()
        except Exception:  # pragma: no cover - defensive
            return None
        return None

    # ------------------------------------------------------------------
    # TracingProcessor lifecycle hooks
    # ------------------------------------------------------------------

    def on_trace_start(self, trace: Any) -> None:
        """Open a Noveum trace mirroring the OpenAI Agents trace."""
        try:
            client = self._get_client()
            if client is None:
                return
            trace_id = getattr(trace, "trace_id", None)
            if trace_id is None:
                return

            workflow_name = getattr(trace, "name", None)
            trace_name = (
                str(workflow_name)
                if workflow_name
                else f"{self._trace_name_prefix}.workflow"
            )

            attributes: dict[str, Any] = {}
            if workflow_name:
                attributes[C.ATTR_WORKFLOW_NAME] = str(workflow_name)
            if self.capture_trace_metadata:
                group_id = getattr(trace, "group_id", None)
                if group_id is not None:
                    attributes[C.ATTR_GROUP_ID] = str(group_id)
                metadata = getattr(trace, "metadata", None)
                if metadata:
                    attributes[C.ATTR_TRACE_METADATA] = to_serialisable(metadata)

            noveum_trace_obj = client.start_trace(
                name=trace_name,
                attributes=attributes,
                set_as_current=False,
            )
            with self._lock:
                self._traces[str(trace_id)] = noveum_trace_obj
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("on_trace_start failed: %s", exc, exc_info=True)

    def on_trace_end(self, trace: Any) -> None:
        """Finish (and export) the Noveum trace for a completed OpenAI trace."""
        try:
            trace_id = getattr(trace, "trace_id", None)
            if trace_id is None:
                return
            with self._lock:
                noveum_trace_obj = self._traces.pop(str(trace_id), None)
                orphan_ids = [
                    span_id
                    for span_id, span in self._spans.items()
                    if getattr(span, "trace_id", None)
                    == getattr(noveum_trace_obj, "trace_id", object())
                ]
                orphans = [self._spans.pop(span_id) for span_id in orphan_ids]
            if noveum_trace_obj is None:
                return
            for orphan in orphans:
                _finish_span(orphan, None)

            client = self._get_client()
            if client is not None:
                client.finish_trace(noveum_trace_obj)
            else:
                try:
                    noveum_trace_obj.finish()
                except Exception:  # pragma: no cover - defensive
                    pass
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("on_trace_end failed: %s", exc, exc_info=True)

    def on_span_start(self, span: Any) -> None:
        """Open a Noveum span, linking it to its parent when one exists."""
        try:
            trace_id = getattr(span, "trace_id", None)
            span_id = getattr(span, "span_id", None)
            if trace_id is None or span_id is None:
                return
            with self._lock:
                noveum_trace_obj = self._traces.get(str(trace_id))
            if noveum_trace_obj is None:
                return

            parent_span_id: Optional[str] = None
            parent_id = getattr(span, "parent_id", None)
            if parent_id is not None:
                with self._lock:
                    parent_noveum_span = self._spans.get(str(parent_id))
                if parent_noveum_span is not None:
                    parent_span_id = getattr(parent_noveum_span, "span_id", None)

            span_type = _span_data_type(getattr(span, "span_data", None))
            span_name = C.SPAN_NAME_BY_TYPE.get(span_type, C.SPAN_DEFAULT)
            start_time = coerce_iso_datetime(getattr(span, "started_at", None))

            noveum_span = noveum_trace_obj.create_span(
                name=span_name,
                parent_span_id=parent_span_id,
                attributes={C.ATTR_SPAN_TYPE: span_type} if span_type else {},
                start_time=start_time,
            )
            with self._lock:
                self._spans[str(span_id)] = noveum_span
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("on_span_start failed: %s", exc, exc_info=True)

    def on_span_end(self, span: Any) -> None:
        """Populate attributes and finish the Noveum span for a completed span."""
        try:
            span_id = getattr(span, "span_id", None)
            if span_id is None:
                return
            with self._lock:
                noveum_span = self._spans.pop(str(span_id), None)
            if noveum_span is None:
                return

            span_data = getattr(span, "span_data", None)
            span_type = _span_data_type(span_data)
            attributes = self._build_span_attributes(span_type, span_data)

            error = getattr(span, "error", None)
            if error:
                self._apply_error(noveum_span, error, attributes)
            else:
                attributes[C.ATTR_STATUS] = C.STATUS_OK

            _set_attributes(noveum_span, attributes)
            end_time = coerce_iso_datetime(getattr(span, "ended_at", None))
            _finish_span(noveum_span, end_time)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("on_span_end failed: %s", exc, exc_info=True)

    def shutdown(self) -> None:
        """Flush pending traces on SDK shutdown; never raises into the host app."""
        try:
            self._force_flush_client()
        finally:
            self._is_shutdown = True

    def force_flush(self) -> None:
        """Flush the Noveum client so buffered traces are exported promptly."""
        self._force_flush_client()

    # ------------------------------------------------------------------
    # Attribute mapping
    # ------------------------------------------------------------------

    def _build_span_attributes(self, span_type: str, span_data: Any) -> dict[str, Any]:
        """Dispatch to the per-``SpanData`` mapper for *span_type*."""
        attributes: dict[str, Any] = {}
        if span_data is None:
            return attributes
        try:
            if span_type == "agent":
                self._map_agent(span_data, attributes)
            elif span_type == "function":
                self._map_function(span_data, attributes)
            elif span_type == "generation":
                self._map_generation(span_data, attributes)
            elif span_type == "response":
                self._map_response(span_data, attributes)
            elif span_type == "handoff":
                self._map_handoff(span_data, attributes)
            elif span_type == "guardrail":
                self._map_guardrail(span_data, attributes)
            elif span_type == "custom":
                self._map_custom(span_data, attributes)
            elif span_type == "mcp_tools":
                self._map_mcp_tools(span_data, attributes)
            elif span_type in ("task", "turn"):
                self._map_task_turn(span_data, attributes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("attribute mapping failed for %s: %s", span_type, exc)
        return attributes

    def _map_agent(self, span_data: Any, attributes: dict[str, Any]) -> None:
        name = getattr(span_data, "name", None)
        if name:
            attributes[C.ATTR_AGENT_NAME] = str(name)
        output_type = getattr(span_data, "output_type", None)
        if output_type:
            attributes[C.ATTR_AGENT_OUTPUT_TYPE] = str(output_type)
        if self.capture_tool_schemas:
            handoffs = getattr(span_data, "handoffs", None)
            if handoffs:
                attributes[C.ATTR_AGENT_HANDOFFS] = to_serialisable(handoffs)
            tools = getattr(span_data, "tools", None)
            if tools:
                attributes[C.ATTR_AGENT_TOOLS] = to_serialisable(tools)

    def _map_function(self, span_data: Any, attributes: dict[str, Any]) -> None:
        name = getattr(span_data, "name", None)
        if name:
            attributes[C.ATTR_TOOL_NAME] = str(name)
        if self.capture_inputs:
            tool_input = getattr(span_data, "input", None)
            if tool_input is not None:
                attributes[C.ATTR_TOOL_INPUT] = truncate_text(
                    tool_input, C.MAX_TEXT_LENGTH
                )
        if self.capture_outputs:
            tool_output = getattr(span_data, "output", None)
            if tool_output is not None:
                attributes[C.ATTR_TOOL_OUTPUT] = truncate_text(
                    tool_output, C.MAX_TEXT_LENGTH
                )

    def _map_generation(self, span_data: Any, attributes: dict[str, Any]) -> None:
        model = getattr(span_data, "model", None)
        self._apply_model(model, attributes)
        self._apply_model_config(getattr(span_data, "model_config", None), attributes)
        self._apply_usage(model, getattr(span_data, "usage", None), attributes)
        if self.capture_llm_messages:
            llm_input = getattr(span_data, "input", None)
            if llm_input is not None:
                attributes[C.ATTR_LLM_INPUT] = to_serialisable(llm_input)
            llm_output = getattr(span_data, "output", None)
            if llm_output is not None:
                attributes[C.ATTR_LLM_OUTPUT] = to_serialisable(llm_output)

    def _map_response(self, span_data: Any, attributes: dict[str, Any]) -> None:
        response = getattr(span_data, "response", None)
        model = getattr(response, "model", None)
        self._apply_model(model, attributes)
        usage = getattr(response, "usage", None) or getattr(span_data, "usage", None)
        self._apply_usage(model, usage, attributes)
        response_id = getattr(response, "id", None)
        if response_id:
            attributes[C.ATTR_LLM_REQUEST_ID] = str(response_id)
        if self.capture_inputs:
            response_input = getattr(span_data, "input", None)
            if response_input is not None:
                attributes[C.ATTR_LLM_INPUT] = to_serialisable(response_input)
        if self.capture_outputs and response is not None:
            output_text = getattr(response, "output_text", None)
            if output_text:
                attributes[C.ATTR_LLM_OUTPUT] = truncate_text(
                    output_text, C.MAX_TEXT_LENGTH
                )

    def _map_handoff(self, span_data: Any, attributes: dict[str, Any]) -> None:
        from_agent = getattr(span_data, "from_agent", None)
        if from_agent:
            attributes[C.ATTR_HANDOFF_FROM] = str(from_agent)
        to_agent = getattr(span_data, "to_agent", None)
        if to_agent:
            attributes[C.ATTR_HANDOFF_TO] = str(to_agent)

    def _map_guardrail(self, span_data: Any, attributes: dict[str, Any]) -> None:
        name = getattr(span_data, "name", None)
        if name:
            attributes[C.ATTR_GUARDRAIL_NAME] = str(name)
        triggered = getattr(span_data, "triggered", None)
        if triggered is not None:
            attributes[C.ATTR_GUARDRAIL_TRIGGERED] = bool(triggered)

    def _map_custom(self, span_data: Any, attributes: dict[str, Any]) -> None:
        name = getattr(span_data, "name", None)
        if name:
            attributes[C.ATTR_CUSTOM_NAME] = str(name)
        if self.capture_inputs:
            data = getattr(span_data, "data", None)
            if data:
                attributes[C.ATTR_CUSTOM_DATA] = to_serialisable(data)

    def _map_mcp_tools(self, span_data: Any, attributes: dict[str, Any]) -> None:
        server = getattr(span_data, "server", None)
        if server:
            attributes[C.ATTR_MCP_SERVER] = str(server)
        if self.capture_tool_schemas:
            result = getattr(span_data, "result", None)
            if result:
                attributes[C.ATTR_MCP_TOOLS] = to_serialisable(result)

    def _map_task_turn(self, span_data: Any, attributes: dict[str, Any]) -> None:
        self._apply_usage(None, getattr(span_data, "usage", None), attributes)
        turn = getattr(span_data, "turn", None)
        if turn is not None:
            attributes[C.ATTR_TURN_NUMBER] = turn
        agent_name = getattr(span_data, "agent_name", None)
        if agent_name:
            attributes[C.ATTR_TURN_AGENT_NAME] = str(agent_name)
        name = getattr(span_data, "name", None)
        if name:
            attributes[C.ATTR_TASK_NAME] = str(name)

    def _apply_model(self, model: Any, attributes: dict[str, Any]) -> None:
        if not model:
            return
        attributes[C.ATTR_LLM_MODEL] = str(model)
        provider = derive_provider(str(model))
        if provider:
            attributes[C.ATTR_LLM_PROVIDER] = provider

    def _apply_model_config(
        self, model_config: Any, attributes: dict[str, Any]
    ) -> None:
        config = extract_model_config(model_config)
        if "temperature" in config:
            attributes[C.ATTR_LLM_TEMPERATURE] = config["temperature"]
        if "top_p" in config:
            attributes[C.ATTR_LLM_TOP_P] = config["top_p"]
        if "max_tokens" in config:
            attributes[C.ATTR_LLM_MAX_TOKENS] = config["max_tokens"]

    def _apply_usage(self, model: Any, usage: Any, attributes: dict[str, Any]) -> None:
        tokens = extract_usage_tokens(usage)
        if tokens["input_tokens"] is not None:
            attributes[C.ATTR_LLM_INPUT_TOKENS] = tokens["input_tokens"]
        if tokens["output_tokens"] is not None:
            attributes[C.ATTR_LLM_OUTPUT_TOKENS] = tokens["output_tokens"]
        if tokens["total_tokens"] is not None:
            attributes[C.ATTR_LLM_TOTAL_TOKENS] = tokens["total_tokens"]
        if self.capture_cost and model:
            cost = estimate_cost_safe(
                str(model), tokens["input_tokens"], tokens["output_tokens"]
            )
            if cost.get("total"):
                attributes[C.ATTR_LLM_COST_INPUT] = cost.get("input")
                attributes[C.ATTR_LLM_COST_OUTPUT] = cost.get("output")
                attributes[C.ATTR_LLM_COST_TOTAL] = cost.get("total")
                attributes[C.ATTR_LLM_COST_CURRENCY] = cost.get("currency")

    def _apply_error(
        self, noveum_span: Any, error: Any, attributes: dict[str, Any]
    ) -> None:
        """Attach error attributes and mark the span ERROR (non-fatal)."""
        attributes[C.ATTR_STATUS] = C.STATUS_ERROR
        message: Any = None
        data: Any = None
        if isinstance(error, dict):
            message = error.get("message")
            data = error.get("data")
        else:
            message = getattr(error, "message", None) or str(error)
            data = getattr(error, "data", None)
        if message is not None:
            attributes[C.ATTR_ERROR_MESSAGE] = truncate_text(message, C.MAX_TEXT_LENGTH)
        if data:
            attributes[C.ATTR_ERROR_DATA] = to_serialisable(data)
        try:
            from noveum_trace.core.span import SpanStatus

            noveum_span.set_status(
                SpanStatus.ERROR, str(message) if message else "error"
            )
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------
    # Flush helper
    # ------------------------------------------------------------------

    def _force_flush_client(self) -> None:
        try:
            client = self._get_client()
            if client is not None:
                client.flush()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("flush failed: %s", exc)


def setup_openai_agents_tracing(
    *,
    replace_processors: bool = False,
    **kwargs: Any,
) -> NoveumTraceProcessor:
    """
    Create a :class:`NoveumTraceProcessor` and register it with the Agents SDK.

    Requires ``noveum_trace.init(...)`` to have been called (or an explicit
    ``client=`` kwarg). By default the processor is *added* alongside the SDK's
    own exporter; pass ``replace_processors=True`` to replace all processors
    (disabling OpenAI's default trace upload).

    Args:
        replace_processors: Use ``set_trace_processors([...])`` instead of
            ``add_trace_processor(...)``.
        **kwargs: Forwarded to :class:`NoveumTraceProcessor` (capture flags, etc.).

    Returns:
        The registered :class:`NoveumTraceProcessor`.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
        TypeError: If ``api_key`` / ``project`` are passed (configure those via
            ``noveum_trace.init(...)``).
    """
    if not OPENAI_AGENTS_AVAILABLE:
        raise ImportError(
            "openai-agents is not installed. Install with: "
            'pip install "noveum-trace[openai-agents]"'
        )
    for key in ("api_key", "project"):
        if key in kwargs:
            raise TypeError(
                f"setup_openai_agents_tracing() does not accept {key!r}. "
                "Use noveum_trace.init(api_key=..., project=...) to configure the "
                "global client; the processor always uses get_client()."
            )

    processor = NoveumTraceProcessor(**kwargs)

    from agents import add_trace_processor, set_trace_processors

    if replace_processors:
        set_trace_processors([processor])
    else:
        add_trace_processor(processor)
    return processor
