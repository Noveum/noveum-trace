"""
Example: trace a LlamaIndex query pipeline with Noveum.

Install the integration extra (plus the LlamaIndex readers/LLM you use)::

    pip install "noveum-trace[llamaindex]" llama-index

Provide credentials and run::

    export NOVEUM_API_KEY=...
    export OPENAI_API_KEY=...
    python docs/examples/llamaindex_integration_example.py

``setup_llamaindex_tracing`` registers span + event handlers on LlamaIndex's
instrumentation dispatcher, so every query engine, retriever, synthesizer, LLM,
and embedding call is mirrored as a Noveum trace/span — no other code changes.
"""

from __future__ import annotations

import os

from llama_index.core import Document, VectorStoreIndex

import noveum_trace
from noveum_trace.integrations.llamaindex import setup_llamaindex_tracing


def main() -> None:
    noveum_trace.init(
        project="llamaindex-example",
        api_key=os.environ.get("NOVEUM_API_KEY"),
    )

    # Everything is captured by default — query text, retrieved node content and
    # the synthesized answer. capture_embedding_chunks is the one opt-in: it adds
    # the text being embedded, which on a large corpus is a lot of data.
    setup_llamaindex_tracing()

    documents = [
        Document(text="Noveum provides AI tracing and observability for LLM apps."),
        Document(text="LlamaIndex builds retrieval-augmented generation pipelines."),
    ]

    try:
        index = VectorStoreIndex.from_documents(documents)
        response = index.as_query_engine().query("What does Noveum do?")
        print(response)
    finally:
        # Flush buffered traces and release SDK resources, even on failure.
        noveum_trace.flush()
        noveum_trace.shutdown()


if __name__ == "__main__":
    main()
