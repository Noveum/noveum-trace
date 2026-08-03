"""
Noveum OpenAI Agents SDK Integration.

Provides tracing for the `OpenAI Agents SDK <https://openai.github.io/openai-agents-python/>`_
by implementing a ``TracingProcessor`` that mirrors OpenAI Agents traces/spans as
Noveum traces/spans.

Installation
------------
>>> pip install "noveum-trace[openai-agents]"

Setup
-----
>>> import noveum_trace
>>> from agents import add_trace_processor
>>> from noveum_trace.integrations.openai_agents import NoveumTraceProcessor
>>>
>>> noveum_trace.init(project="my-project", api_key="...")
>>> add_trace_processor(NoveumTraceProcessor())

or, equivalently, using the convenience factory::

>>> from noveum_trace.integrations.openai_agents import setup_openai_agents_tracing
>>> noveum_trace.init(project="my-project", api_key="...")
>>> setup_openai_agents_tracing()

Configuration Options
---------------------
All options passed to ``NoveumTraceProcessor`` or ``setup_openai_agents_tracing``:

  capture_inputs          — Capture raw tool / function / custom inputs (default: off)
  capture_outputs         — Capture raw tool / function / LLM outputs (default: off)
  capture_llm_messages    — Capture full LLM prompt/response messages (default: off)
  capture_tool_schemas    — Capture agent tool / handoff names (default: on)
  capture_trace_metadata  — Copy OpenAI trace metadata / group_id (default: on)
  capture_cost            — Estimate LLM cost from tokens (default: on)
  trace_name_prefix       — Prefix for unnamed workflows (default: "openai_agents")
"""

from noveum_trace.integrations.openai_agents.processor import (
    NoveumTraceProcessor,
    setup_openai_agents_tracing,
)

__all__ = [
    "NoveumTraceProcessor",
    "setup_openai_agents_tracing",
]
