"""
Constants for the OpenAI Agents SDK integration.

Structure: span name constants, a ``SpanData.type`` → span-name map, span
attribute key constants (reusing the canonical ``llm.*`` / ``tool.*`` keys the
rest of the SDK understands), status values, and numeric limits.

Span hierarchy (mirrors the OpenAI Agents trace/span tree)::

    <workflow trace>                 ← one per ``agents`` Trace
      openai_agents.agent            ← one per Agent invocation
        openai_agents.generation     ← Chat Completions generation call
        openai_agents.response       ← Responses API call
        openai_agents.function       ← tool / function-tool call
        openai_agents.handoff        ← agent-to-agent handoff
        openai_agents.guardrail      ← input / output guardrail
        openai_agents.mcp_tools      ← MCP ``list_tools`` call
        openai_agents.custom         ← ``custom_span`` created by user code
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Span name constants
# ---------------------------------------------------------------------------

SPAN_AGENT = "openai_agents.agent"
SPAN_FUNCTION = "openai_agents.function"
SPAN_GENERATION = "openai_agents.generation"
SPAN_RESPONSE = "openai_agents.response"
SPAN_HANDOFF = "openai_agents.handoff"
SPAN_GUARDRAIL = "openai_agents.guardrail"
SPAN_CUSTOM = "openai_agents.custom"
SPAN_MCP_TOOLS = "openai_agents.mcp_tools"
SPAN_TRANSCRIPTION = "openai_agents.transcription"
SPAN_SPEECH = "openai_agents.speech"
SPAN_SPEECH_GROUP = "openai_agents.speech_group"
SPAN_TASK = "openai_agents.task"
SPAN_TURN = "openai_agents.turn"
SPAN_DEFAULT = "openai_agents.span"

# ``SpanData.type`` string → Noveum span name.
SPAN_NAME_BY_TYPE: dict[str, str] = {
    "agent": SPAN_AGENT,
    "function": SPAN_FUNCTION,
    "generation": SPAN_GENERATION,
    "response": SPAN_RESPONSE,
    "handoff": SPAN_HANDOFF,
    "guardrail": SPAN_GUARDRAIL,
    "custom": SPAN_CUSTOM,
    "mcp_tools": SPAN_MCP_TOOLS,
    "transcription": SPAN_TRANSCRIPTION,
    "speech": SPAN_SPEECH,
    "speech_group": SPAN_SPEECH_GROUP,
    "task": SPAN_TASK,
    "turn": SPAN_TURN,
}

# ---------------------------------------------------------------------------
# Trace / workflow attribute keys   (prefix: openai_agents.*)
# ---------------------------------------------------------------------------

ATTR_WORKFLOW_NAME = "openai_agents.workflow_name"
ATTR_GROUP_ID = "openai_agents.group_id"
ATTR_TRACE_METADATA = "openai_agents.metadata"
ATTR_SPAN_TYPE = "openai_agents.span_type"
ATTR_STATUS = "openai_agents.status"

# ---------------------------------------------------------------------------
# Agent attribute keys   (prefix: agent.*)
# ---------------------------------------------------------------------------

ATTR_AGENT_NAME = "agent.name"
ATTR_AGENT_HANDOFFS = "agent.handoffs"
ATTR_AGENT_TOOLS = "agent.tools"
ATTR_AGENT_OUTPUT_TYPE = "agent.output_type"

# ---------------------------------------------------------------------------
# Tool / function attribute keys   (prefix: tool.*)
# ---------------------------------------------------------------------------

ATTR_TOOL_NAME = "tool.name"
ATTR_TOOL_INPUT = "tool.input"
ATTR_TOOL_OUTPUT = "tool.output"

# ---------------------------------------------------------------------------
# LLM attribute keys   (prefix: llm.* — consumed by the gen_ai crosswalk)
# ---------------------------------------------------------------------------

ATTR_LLM_MODEL = "llm.model"
ATTR_LLM_PROVIDER = "llm.provider"
ATTR_LLM_INPUT = "llm.input"
ATTR_LLM_OUTPUT = "llm.output"
ATTR_LLM_INPUT_TOKENS = "llm.input_tokens"
ATTR_LLM_OUTPUT_TOKENS = "llm.output_tokens"
ATTR_LLM_TOTAL_TOKENS = "llm.total_tokens"
ATTR_LLM_TEMPERATURE = "llm.temperature"
ATTR_LLM_MAX_TOKENS = "llm.max_tokens"
ATTR_LLM_TOP_P = "llm.top_p"
ATTR_LLM_REQUEST_ID = "llm.request_id"
ATTR_LLM_COST_INPUT = "llm.cost.input"
ATTR_LLM_COST_OUTPUT = "llm.cost.output"
ATTR_LLM_COST_TOTAL = "llm.cost.total"
ATTR_LLM_COST_CURRENCY = "llm.cost.currency"

# ---------------------------------------------------------------------------
# Handoff attribute keys   (prefix: handoff.*)
# ---------------------------------------------------------------------------

ATTR_HANDOFF_FROM = "handoff.from_agent"
ATTR_HANDOFF_TO = "handoff.to_agent"

# ---------------------------------------------------------------------------
# Guardrail attribute keys   (prefix: guardrail.*)
# ---------------------------------------------------------------------------

ATTR_GUARDRAIL_NAME = "guardrail.name"
ATTR_GUARDRAIL_TRIGGERED = "guardrail.triggered"

# ---------------------------------------------------------------------------
# MCP attribute keys   (prefix: mcp.*)
# ---------------------------------------------------------------------------

ATTR_MCP_SERVER = "mcp.server_name"
ATTR_MCP_TOOLS = "mcp.tools"

# ---------------------------------------------------------------------------
# Custom-span attribute keys   (prefix: custom.*)
# ---------------------------------------------------------------------------

ATTR_CUSTOM_NAME = "custom.name"
ATTR_CUSTOM_DATA = "custom.data"

# ---------------------------------------------------------------------------
# Turn / task attribute keys
# ---------------------------------------------------------------------------

ATTR_TURN_NUMBER = "turn.number"
ATTR_TURN_AGENT_NAME = "turn.agent_name"
ATTR_TASK_NAME = "task.name"

# ---------------------------------------------------------------------------
# Error / status attribute keys
# ---------------------------------------------------------------------------

ATTR_ERROR_TYPE = "error.type"
ATTR_ERROR_MESSAGE = "error.message"
ATTR_ERROR_DATA = "error.data"

STATUS_OK = "ok"
STATUS_ERROR = "error"

# ---------------------------------------------------------------------------
# Limits / defaults
# ---------------------------------------------------------------------------

MAX_TEXT_LENGTH = 8_192
DEFAULT_TRACE_NAME_PREFIX = "openai_agents"

# Informational — the minimum ``openai-agents`` release these mappings were
# verified against (see ``pyproject.toml`` for the enforced pin).
MIN_OPENAI_AGENTS_VERSION = "0.19.2"
