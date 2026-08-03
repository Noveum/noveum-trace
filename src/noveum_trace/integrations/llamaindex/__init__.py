"""
Noveum LlamaIndex Integration.

Provides tracing for `LlamaIndex <https://docs.llamaindex.ai/>`_ query pipelines
(query engines, retrievers, synthesizers, LLM and embedding calls) by registering
span + event handlers on LlamaIndex's modern instrumentation dispatcher.

Installation
------------
>>> pip install "noveum-trace[llamaindex]"

Setup
-----
>>> import noveum_trace
>>> from noveum_trace.integrations.llamaindex import setup_llamaindex_tracing
>>>
>>> noveum_trace.init(project="my-project", api_key="...")
>>> setup_llamaindex_tracing()
>>>
>>> # ... build and query your index as usual ...
>>> noveum_trace.flush()

Configuration Options
---------------------
All options passed to ``setup_llamaindex_tracing``:

  capture_inputs        — Capture query / retrieval query text (default: off)
  capture_outputs       — Capture response text and retrieved node content
                          (default: off)
  capture_llm_messages  — Capture full LLM prompt/response messages (default: off)
  trace_name_prefix     — Prefix for unnamed operations (default: "llamaindex")
"""

from noveum_trace.integrations.llamaindex.handler import (
    NoveumLlamaIndexEventHandler,
    NoveumLlamaIndexSpan,
    NoveumLlamaIndexSpanHandler,
    setup_llamaindex_tracing,
)

__all__ = [
    "NoveumLlamaIndexEventHandler",
    "NoveumLlamaIndexSpan",
    "NoveumLlamaIndexSpanHandler",
    "setup_llamaindex_tracing",
]
