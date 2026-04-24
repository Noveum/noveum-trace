"""
Noveum CrewAI Integration

Provides tracing for CrewAI agents, tasks, tools, and LLM calls.

Installation
------------
>>> pip install "noveum-trace[crewai]"

Setup
-----
>>> from noveum_trace import init
>>> from noveum_trace.integrations.crewai import NoveumCrewAIListener, setup_crewai_tracing
>>>
>>> # Option 1: Using setup factory
>>> init(project="my-project", api_key="...")
>>> listener = setup_crewai_tracing()
>>> crew.callback_function = listener
>>> crew.kickoff()
>>>
>>> # Option 2: Manual instantiation
>>> from noveum_trace import get_client
>>> client = get_client()
>>> listener = NoveumCrewAIListener(client)
>>> crew.callback_function = listener
>>> crew.kickoff()

Configuration Options
---------------------
All options passed to NoveumCrewAIListener or setup_crewai_tracing():

  capture_inputs          — Capture input messages, tool args, task prompts
  capture_outputs         — Capture LLM responses, tool outputs, task results
  capture_llm_messages    — Capture full message history (system + RAG)
  capture_tool_schemas    — Capture tool definitions and available functions
  capture_agent_snapshot  — Capture agent goal/backstory at start
  capture_crew_snapshot   — Capture crew agents/tasks at kickoff
  capture_memory          — Capture memory operations (query/save)
  capture_a2a             — Capture agent-to-agent delegation
  capture_mcp             — Capture MCP server calls
  capture_flow            — Capture CrewAI Flow events
  capture_reasoning       — Capture reasoning steps
  capture_streaming       — Accumulate streaming chunks
  capture_thinking        — Capture extended thinking tokens
  trace_name_prefix       — Prefix for trace names (default: "crewai")
  verbose                 — Enable debug logging
"""

from noveum_trace.integrations.crewai.crewai_listener import (
    NoveumCrewAIListener,
    setup_crewai_tracing,
)

__all__ = [
    "NoveumCrewAIListener",
    "setup_crewai_tracing",
]
