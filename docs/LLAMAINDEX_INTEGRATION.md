# LlamaIndex Integration Guide

Trace [LlamaIndex](https://docs.llamaindex.ai/) query pipelines with Noveum. The
integration registers **span** and **event** handlers on LlamaIndex's modern
instrumentation dispatcher (`llama_index.core.instrumentation`) and mirrors every
operation — query engines, retrievers, synthesizers, LLM calls, embedding calls,
rerankers — as a Noveum trace/span, preserving the parent-child hierarchy that
LlamaIndex threads through its `active_span_id` context variable.

## Prerequisites

- Python 3.9+
- A Noveum project + API key
- `llama-index-core >= 0.11`

## Installation

```bash
pip install "noveum-trace[llamaindex]"
```

## Quick start

### 1. Initialize the SDK

```python
import noveum_trace

noveum_trace.init(project="my-project", api_key="...")
```

### 2. Register the handlers

```python
from noveum_trace.integrations.llamaindex import setup_llamaindex_tracing

setup_llamaindex_tracing()
```

### 3. Build and query your index

```python
from llama_index.core import Document, VectorStoreIndex

index = VectorStoreIndex.from_documents([Document(text="...")])
response = index.as_query_engine().query("...")

noveum_trace.flush()  # export buffered traces before exit
```

`setup_llamaindex_tracing()` attaches to the **root** dispatcher, so all
LlamaIndex activity is traced. Each top-level operation (e.g. a query) opens a new
Noveum trace; nested operations become child spans under it.

## What gets traced

| LlamaIndex activity | Noveum span type | Key attributes |
| --- | --- | --- |
| Query engine (`.query`) | `query` | `query.text`\*, `query.response`\* |
| Retriever (`.retrieve`) | `retrieval` | `retrieval.node_count`, `retrieval.scores`, `retrieval.nodes`\* |
| Synthesizer | `synthesize` | — |
| LLM (`chat`/`complete`/`predict`) | `llm` | `llm.model`, `llm.provider`, `llm.input_tokens`, `llm.output_tokens`, `llm.total_tokens`, `llm.input`\*, `llm.output`\* |
| Embedding | `embedding` | `embedding.model`, `embedding.chunk_count`, `embedding.vector_count` |
| Reranker / node postprocessor | `rerank` | `rerank.model`, `rerank.top_n`, `rerank.input_node_count`, `rerank.output_node_count` |

\* Captured only when the corresponding capture flag is enabled.

Every span also carries `llamaindex.operation` (the `Class.method` name) and
`llamaindex.span_type`.

## Configuration options and privacy

All options are accepted by `setup_llamaindex_tracing(...)`:

| Option | Default | Captures |
| --- | --- | --- |
| `capture_inputs` | `False` | Query / retrieval query text |
| `capture_outputs` | `False` | Response text and retrieved node content |
| `capture_llm_messages` | `False` | Full LLM prompt/response message arrays |
| `client` | global | Explicit Noveum client instead of the initialized global |
| `trace_name_prefix` | `"llamaindex"` | Prefix used when an operation name is unavailable |
| `dispatcher` | root | Dispatcher to register on (defaults to `get_dispatcher()`) |

**Privacy-safe by default.** Raw payloads that may contain sensitive data — query
text, retrieved document content, and LLM message content — are **not** captured
unless you opt in. Structural metadata that is always captured: operation name,
span type, model name and provider, token usage, node counts and similarity
scores, embedding counts, and error type/message.

```python
# Capture full payloads (e.g. in a trusted dev environment)
setup_llamaindex_tracing(
    capture_inputs=True,
    capture_outputs=True,
    capture_llm_messages=True,
)
```

## Version compatibility

| Component | Supported |
| --- | --- |
| `llama-index-core` | `>= 0.11, < 1.0` |
| Python | `>= 3.9` |

The integration targets the stable `llama_index.core.instrumentation` dispatcher
(available since 0.10.x). Handlers read event/span fields defensively, so newer
event types are recorded generically rather than raising.

## Resilience

Failures inside the handlers (a mapping error, an unreachable Noveum backend) are
logged at debug level and never propagate. LlamaIndex additionally swallows
handler exceptions in its dispatcher, so tracing can never break a query.

## Troubleshooting

- **No traces appear:** confirm `noveum_trace.init(...)` ran before
  `setup_llamaindex_tracing()`, and call `noveum_trace.flush()` before exit.
- **Query text / node content missing:** these are opt-in; enable
  `capture_inputs` / `capture_outputs` / `capture_llm_messages`.
- **Token counts missing:** LlamaIndex exposes usage on the provider-native
  `response.raw`; some LLM integrations do not populate it.

## Example script

See [`docs/examples/llamaindex_integration_example.py`](examples/llamaindex_integration_example.py).

## Next steps

- [LlamaIndex docs](https://docs.llamaindex.ai/)
- [LlamaIndex instrumentation docs](https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation/)
