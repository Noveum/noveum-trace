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

  capture_inputs        — Capture query / retrieval / rerank query text, rerank
                          input nodes, agent tool arguments (default: on)
  capture_outputs       — Capture LLM response text, query response text,
                          retrieved node content, query source nodes and
                          reranked output nodes (default: on)
  capture_llm_messages  — Capture LLM prompt messages, system prompts and
                          available tool schemas (default: on). LLM responses
                          follow capture_outputs, not this flag.
  capture_cost          — Estimate LLM cost from model and token counts
                          (default: on)
  capture_embedding_chunks
                        — Capture the text being embedded (default: off)
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
