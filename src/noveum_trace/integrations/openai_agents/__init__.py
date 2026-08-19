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

  capture_inputs          — Capture raw tool / function / custom inputs (default: on)
  capture_outputs         — Capture raw tool / function outputs (default: on)
  capture_llm_messages    — Capture full LLM prompt/response messages, system
                            prompts and tool calls, on generation and response
                            spans (default: on)
  capture_tool_schemas    — Capture agent tool / handoff names and the tool
                            schemas offered to the model (default: on)
  capture_trace_metadata  — Copy OpenAI trace metadata / group_id (default: on;
                            may contain sensitive data — see the integration guide)
  capture_cost            — Estimate LLM cost from tokens (default: on)
  trace_name_prefix       — Prefix for unnamed workflows (default: "openai_agents")

Everything is captured by default; set the flags to ``False`` to reduce what is
sent. Note that the Agents SDK separately decides whether to record prompt and
response payloads at all — with ``RunConfig(trace_include_sensitive_data=False)``
or ``OPENAI_AGENTS_DONT_LOG_MODEL_DATA`` set, those payloads never reach any
processor and no flag here can recover them.
"""

from noveum_trace.integrations.openai_agents.processor import (
    NoveumTraceProcessor,
    setup_openai_agents_tracing,
)

__all__ = [
    "NoveumTraceProcessor",
    "setup_openai_agents_tracing",
]
