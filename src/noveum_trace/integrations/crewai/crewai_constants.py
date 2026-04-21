"""
Constants for CrewAI integration.

Structure: span name constants, span attribute key
constants, LLM settings attribute map, and numeric limits/defaults.

Span hierarchy::

    crewai.crew                  ← root: one per Crew.kickoff() call
      crewai.task                ← one per Task assigned to the Crew
        crewai.agent             ← one per Agent step executing the Task
          crewai.llm             ← one per LLM call made by the Agent
          crewai.tool            ← one per Tool invocation made by the Agent
          crewai.memory.query    ← memory read / search operation
          crewai.memory.save     ← memory write / store operation
          crewai.memory.retrieval← bulk context pull for a task prompt
          crewai.a2a.delegation  ← agent-to-agent delegation (hierarchical process)
          crewai.a2a.conversation← multi-turn A2A conversation
          crewai.mcp.connection  ← MCP server connection attempt
          crewai.mcp.tool        ← MCP tool call within a connection
      crewai.flow                ← root: one per Flow execution
        crewai.flow.method       ← one per @start/@listen/@router decorated method
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Span name constants
# ---------------------------------------------------------------------------

SPAN_CREW = "crewai.crew"
SPAN_TASK = "crewai.task"
SPAN_AGENT = "crewai.agent"
SPAN_LLM = "crewai.llm"
SPAN_TOOL = "crewai.tool"
SPAN_FLOW = "crewai.flow"
SPAN_FLOW_METHOD = "crewai.flow.method"

# Memory spans — one per operation sub-type
SPAN_MEMORY_QUERY = "crewai.memory.query"
SPAN_MEMORY_SAVE = "crewai.memory.save"
SPAN_MEMORY_RETRIEVAL = "crewai.memory.retrieval"
SPAN_KNOWLEDGE = "crewai.knowledge"
# Legacy alias kept so that any code referencing SPAN_MEMORY_OP still compiles;
# handlers must use the sub-type constants above.
SPAN_MEMORY_OP = SPAN_MEMORY_QUERY

# A2A spans — one per interaction sub-type
SPAN_A2A_DELEGATION = "crewai.a2a.delegation"
SPAN_A2A_CONVERSATION = "crewai.a2a.conversation"

# MCP spans — one per operation sub-type
SPAN_MCP_CONNECTION = "crewai.mcp.connection"
SPAN_MCP_TOOL = "crewai.mcp.tool"

# ---------------------------------------------------------------------------
# Crew-level attribute keys   (prefix: crew.*)
# ---------------------------------------------------------------------------

ATTR_CREW_ID = "crew.id"
ATTR_CREW_NAME = "crew.name"
ATTR_CREW_PROCESS = "crew.process"  # "sequential" | "hierarchical"
ATTR_CREW_AGENT_COUNT = "crew.agent_count"
ATTR_CREW_TASK_COUNT = "crew.task_count"
ATTR_CREW_AGENT_ROLES = "crew.agent_roles"  # JSON list of role strings
ATTR_CREW_AVAILABLE_AGENTS = "crew.available_agents"  # JSON list of role/name/id labels
ATTR_CREW_AVAILABLE_AGENT_COUNT = "crew.available_agent_count"
ATTR_CREW_MEMORY = "crew.memory"  # bool
ATTR_CREW_VERBOSE = "crew.verbose"  # bool
ATTR_CREW_MAX_RPM = "crew.max_rpm"
ATTR_CREW_OUTPUT = "crew.output"  # final crew result (truncated)
ATTR_CREW_TOTAL_TOKENS = "crew.total_tokens"
ATTR_CREW_TOTAL_COST = "crew.total_cost"
ATTR_CREW_DURATION_MS = "crew.duration_ms"
ATTR_CREW_STATUS = "crew.status"  # "ok" | "error"

# ---------------------------------------------------------------------------
# Task-level attribute keys   (prefix: task.*)
# ---------------------------------------------------------------------------

ATTR_TASK_ID = "task.id"
ATTR_TASK_NAME = "task.name"
ATTR_TASK_DESCRIPTION = "task.description"
ATTR_TASK_EXPECTED_OUTPUT = "task.expected_output"
ATTR_TASK_AGENT_ROLE = "task.agent_role"
ATTR_TASK_OUTPUT = "task.output"  # final task result (truncated)
ATTR_TASK_OUTPUT_FILE = "task.output_file"
ATTR_TASK_HUMAN_INPUT = "task.human_input"  # bool
ATTR_TASK_ASYNC = "task.async_execution"  # bool
ATTR_TASK_DURATION_MS = "task.duration_ms"
ATTR_TASK_STATUS = "task.status"  # "ok" | "error"
ATTR_TASK_CONTEXT = (
    "task.context_tasks"  # JSON list of upstream task descriptions (RAG chain)
)

# ---------------------------------------------------------------------------
# Agent-level attribute keys  (prefix: agent.*)
# ---------------------------------------------------------------------------

ATTR_AGENT_ID = "agent.id"
ATTR_AGENT_ROLE = "agent.role"
ATTR_AGENT_GOAL = "agent.goal"
ATTR_AGENT_BACKSTORY = "agent.backstory"
ATTR_AGENT_LLM_MODEL = "agent.llm_model"
ATTR_AGENT_TOOL_NAMES = "agent.tool_names"  # JSON list
ATTR_AGENT_ALLOW_DELEGATION = "agent.allow_delegation"  # bool
ATTR_AGENT_MAX_ITER = "agent.max_iter"
ATTR_AGENT_MAX_RPM = "agent.max_rpm"
ATTR_AGENT_STEP = "agent.step"  # current reasoning iteration
ATTR_AGENT_DURATION_MS = "agent.duration_ms"
ATTR_AGENT_STATUS = "agent.status"  # "ok" | "error"

# ---------------------------------------------------------------------------
# LLM-call attribute keys     (prefix: llm.*)
# ---------------------------------------------------------------------------

ATTR_LLM_MODEL = "llm.model"
ATTR_LLM_PROVIDER = "llm.provider"
ATTR_LLM_INPUT_TOKENS = "llm.input_tokens"
ATTR_LLM_OUTPUT_TOKENS = "llm.output_tokens"
ATTR_LLM_TOTAL_TOKENS = "llm.total_tokens"
ATTR_LLM_COST_INPUT = "llm.cost.input"
ATTR_LLM_COST_OUTPUT = "llm.cost.output"
ATTR_LLM_COST_TOTAL = "llm.cost.total"
ATTR_LLM_COST_CURRENCY = "llm.cost.currency"
ATTR_LLM_FINISH_REASON = "llm.finish_reason"
ATTR_LLM_SYSTEM_PROMPT = "llm.system_prompt"
ATTR_LLM_INPUT_MESSAGES = "llm.input_messages"  # JSON
ATTR_LLM_OUTPUT_TEXT = "llm.response"  # spec key
ATTR_LLM_THINKING_TEXT = "llm.thinking"  # chain-of-thought / extended thinking
# Provider-specific token extras
ATTR_LLM_CACHE_READ_TOKENS = "llm.cache_read_tokens"
ATTR_LLM_CACHE_CREATION_TOKENS = "llm.cache_creation_tokens"
ATTR_LLM_REASONING_TOKENS = "llm.reasoning_tokens"
ATTR_LLM_TOOLS = "llm.tools"  # JSON tool schema
ATTR_LLM_TEMPERATURE = "llm.temperature"
ATTR_LLM_MAX_TOKENS = "llm.max_tokens"
ATTR_LLM_TOP_P = "llm.top_p"
ATTR_LLM_SEED = "llm.seed"
ATTR_LLM_DURATION_MS = "llm.duration_ms"
ATTR_LLM_STREAMING = "llm.streaming"  # bool
ATTR_LLM_CALL_ID = "llm.call_id"  # internal correlation id

# ---------------------------------------------------------------------------
# Tool-call attribute keys    (prefix: tool.*)
# ---------------------------------------------------------------------------

ATTR_TOOL_NAME = "tool.name"
ATTR_TOOL_DESCRIPTION = "tool.description"
ATTR_TOOL_INPUT = "tool.input"  # JSON / str
ATTR_TOOL_OUTPUT = "tool.output"  # JSON / str (truncated)
ATTR_TOOL_RUN_ID = "tool.run_id"
ATTR_TOOL_DURATION_MS = "tool.duration_ms"
ATTR_TOOL_STATUS = "tool.status"  # "ok" | "error"
ATTR_TOOL_ERROR = "tool.error"  # error message on failure

# ---------------------------------------------------------------------------
# Flow attribute keys         (prefix: flow.*)
# ---------------------------------------------------------------------------

ATTR_FLOW_ID = "flow.id"
ATTR_FLOW_NAME = "flow.name"
ATTR_FLOW_STATE = "flow.state"  # JSON snapshot of Flow state
ATTR_FLOW_STRUCTURE = (
    "flow.structure"  # JSON graph: nodes, edges, start_methods, router_methods
)
ATTR_FLOW_PLOT_EMITTED = (
    "flow.plot_emitted"  # set when FlowPlotEvent fires (e.g. flow.plot())
)
ATTR_FLOW_DURATION_MS = "flow.duration_ms"
ATTR_FLOW_STATUS = "flow.status"  # "ok" | "error"

# Flow method (@start / @listen decorated method)
ATTR_FLOW_METHOD_NAME = "flow.method.name"
ATTR_FLOW_METHOD_ID = "flow.method.id"
ATTR_FLOW_METHOD_TRIGGER = "flow.method.trigger"  # triggering event / method name
ATTR_FLOW_METHOD_DURATION_MS = "flow.method.duration_ms"
ATTR_FLOW_METHOD_STATUS = "flow.method.status"

# ---------------------------------------------------------------------------
# Memory-op attribute keys    (prefix: memory.*)
# ---------------------------------------------------------------------------

ATTR_MEMORY_OP_ID = "memory.op_id"
ATTR_MEMORY_TYPE = "memory.type"  # "short_term" | "long_term" | "entity"
ATTR_MEMORY_OPERATION = "memory.operation"  # "read" | "write" | "search" | "reset"
ATTR_MEMORY_QUERY = "memory.query"  # search query (truncated)
ATTR_MEMORY_RESULT_COUNT = "memory.result_count"
ATTR_MEMORY_DURATION_MS = "memory.duration_ms"
ATTR_MEMORY_STATUS = "memory.status"

# ---------------------------------------------------------------------------
# Agent-to-agent (A2A) attribute keys  (prefix: a2a.*)
# ---------------------------------------------------------------------------

ATTR_A2A_CONTEXT_ID = "a2a.context_id"
ATTR_A2A_DELEGATING_AGENT = "a2a.delegating_agent"  # role of the delegator
ATTR_A2A_RECEIVING_AGENT = "a2a.receiving_agent"  # role of the receiver
ATTR_A2A_RESULT = "a2a.result"  # delegated task result
ATTR_A2A_STATUS = "a2a.status"

# ---------------------------------------------------------------------------
# MCP attribute keys          (prefix: mcp.*)
# ---------------------------------------------------------------------------

ATTR_MCP_KEY = "mcp.key"
ATTR_MCP_SERVER = "mcp.server_name"  # server/transport identifier
ATTR_MCP_TOOL_NAME = "mcp.tool_name"
ATTR_MCP_INPUT = "mcp.arguments"  # JSON
ATTR_MCP_OUTPUT = "mcp.result"  # JSON (truncated)
ATTR_MCP_DURATION_MS = "mcp.duration_ms"
ATTR_MCP_STATUS = "mcp.status"

# ---------------------------------------------------------------------------
# Common / shared attribute keys
# ---------------------------------------------------------------------------

ATTR_ERROR_TYPE = "error.type"
ATTR_ERROR_MESSAGE = "error.message"
ATTR_ERROR_STACKTRACE = "error.stacktrace"

ATTR_STATUS_SUCCESS = "ok"  # spec value — used as status attribute value
ATTR_STATUS_ERROR = "error"

# ---------------------------------------------------------------------------
# LLM settings attribute map
# ---------------------------------------------------------------------------
# Maps LLM object attribute names (as exposed by CrewAI's LLM wrapper,
# LangChain ChatModels, or LiteLLM) to the canonical ``llm.*`` span
# attribute key.  Used by handler code to iterate and capture only the
# settings that are actually set (non-None).

LLM_SETTINGS_ATTRIBUTE_MAP: dict[str, str] = {
    # Model identity
    "model_name": ATTR_LLM_MODEL,  # LangChain ChatOpenAI et al.
    "model": ATTR_LLM_MODEL,  # LiteLLM / most providers
    # Sampling parameters
    "temperature": ATTR_LLM_TEMPERATURE,
    "max_tokens": ATTR_LLM_MAX_TOKENS,
    "max_completion_tokens": ATTR_LLM_MAX_TOKENS,  # OpenAI o-series
    "top_p": ATTR_LLM_TOP_P,
    "seed": ATTR_LLM_SEED,
}

# ---------------------------------------------------------------------------
# Text / string limits
# ---------------------------------------------------------------------------

MAX_TEXT_LENGTH = 8_192  # default truncation for text outputs in spans
MAX_DESCRIPTION_LENGTH = 1_024  # task description, goal, backstory, etc.
MAX_TOOL_OUTPUT_LENGTH = 4_096  # tool result before truncation
MAX_SYSTEM_PROMPT_LENGTH = 4_096  # system prompt stored in span attribute

# Max buffered sent/received message dicts per A2A conversation (``_a2a_stream_buffers``).
MAX_A2A_CONVERSATION_MESSAGES = 1_000

# ---------------------------------------------------------------------------
# Numeric defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "unknown"
DEFAULT_LLM_PROVIDER = "unknown"
