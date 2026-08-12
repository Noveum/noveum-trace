# LlamaIndex Integration Guide

Trace [LlamaIndex](https://docs.llamaindex.ai/) query pipelines with Noveum. The
integration registers **span** and **event** handlers on LlamaIndex's modern
instrumentation dispatcher (`llama_index.core.instrumentation`) and mirrors every
operation — query engines, retrievers, synthesizers, LLM calls, embedding calls,
rerankers — as a Noveum trace/span, preserving the parent-child hierarchy that
LlamaIndex threads through its `active_span_id` context variable.

## Prerequisites

- Python 3.10+ (required by current `llama-index-core`)
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

noveum_trace.flush()     # export buffered traces
noveum_trace.shutdown()  # release SDK resources at application termination
```

`setup_llamaindex_tracing()` attaches to the **root** dispatcher, so all
LlamaIndex activity is traced. Each top-level operation (e.g. a query) opens a new
Noveum trace; nested operations become child spans under it.

## What gets traced

| LlamaIndex activity | Noveum span type | Key attributes |
| --- | --- | --- |
| Query engine (`.query`) | `query` | `query.text`, `query.response`, `query.source_nodes` |
| Retriever (`.retrieve`) | `retrieval` | `retrieval.query`, `retrieval.top_k`, `retrieval.node_count`, `retrieval.scores`, `retrieval.nodes` |
| Synthesizer | `synthesize` | — (the work shows up on the nested `llm` span) |
| LLM (`chat`/`complete`/`predict`) | `llm` | `llm.model`, `llm.provider`, `llm.system_prompt`, `llm.input`, `llm.output`, `llm.available_tools`, `llm.available_tool_count`, `llm.tool_calls`, `llm.tool_call_count`, `llm.input_tokens`, `llm.output_tokens`, `llm.total_tokens`, `llm.cached_input_tokens`, `llm.reasoning_tokens`, `llm.cost.*` |
| Agent tool call | (on the enclosing span) | `tool.name`, `tool.description`, `tool.input` |
| Embedding | `embedding` | `embedding.model`, `embedding.chunk_count`, `embedding.vector_count`, `embedding.dimensions`, `embedding.chunks`† |
| Reranker / node postprocessor | `rerank` | `rerank.model`, `rerank.top_n`, `rerank.query`, `rerank.input_node_count`, `rerank.output_node_count`, `rerank.input_nodes`, `rerank.output_nodes`, `rerank.input_scores`, `rerank.output_scores` |

† `embedding.chunks` requires `capture_embedding_chunks=True` — see
[Embedding capture](#embedding-capture).

Every span also carries `llamaindex.operation` (the `Class.method` name) and
`llamaindex.span_type`.

### What a "node" contains

A LlamaIndex **node** is one chunk of a source document — the unit the index
stores, the retriever returns and the synthesizer reads. `retrieval.nodes`,
`query.source_nodes` and the two `rerank.*_nodes` attributes are lists of:

```json
{
  "id": "a8c7c8e0-…",
  "score": 0.83,
  "text": "Paris is the capital of France. …",
  "metadata": {"file_name": "france.md", "page_label": "2"}
}
```

`text` is the node's **full chunk content**, untruncated — the actual passage
the model was given, so a trace can be replayed or turned into an evaluation
example. `metadata` carries whatever the loader attached (source file, page,
custom tags), which is how a retrieved chunk is traced back to its document.
`score` is the retriever's own similarity/distance for that hit.

### Retrieval scores and top-k

`retrieval.scores` is the ordered list of top-k similarity scores, best match
first — the distances for the closest hits. `retrieval.top_k` is the configured
`similarity_top_k`, read off the retriever instance (the retrieval events
themselves do not carry it).

The **total number of vectors in the vector store** is not available: LlamaIndex's
instrumentation reports per-call activity, not index statistics, and no event or
retriever attribute exposes a collection count. Getting it would mean querying
the vector store directly, which the tracing layer deliberately does not do.
What is recorded per embedding call is `embedding.vector_count` (vectors produced
by that call) and `embedding.dimensions` (vector width).

### Reranking

Both sides of a rerank are captured, not just counts: `rerank.input_nodes` /
`rerank.input_scores` are what went in, `rerank.output_nodes` /
`rerank.output_scores` are the reranked result, so the reordering is directly
comparable.

For an **LLM-backed reranker** (`LLMRerank` and friends), the scoring prompt runs
through the dispatcher as a nested `llm` span. Its reasoning is captured there as
`llm.output` — the rerank events themselves carry only node lists, so the model's
justification lives on the child span rather than on the `rerank` span.

## Configuration options and privacy

All options are accepted by `setup_llamaindex_tracing(...)`:

| Option | Default | Captures |
| --- | --- | --- |
| `capture_inputs` | `True` | Query text, retrieval/rerank query text, rerank input nodes, agent tool arguments |
| `capture_outputs` | `True` | Response text, retrieved node content, query source nodes, reranked output nodes |
| `capture_llm_messages` | `True` | Full LLM prompt/response messages, system prompts, available tool schemas |
| `capture_cost` | `True` | Estimated LLM cost from model + token counts |
| `capture_embedding_chunks` | `False` | The text being embedded (see below) |
| `client` | global | Explicit Noveum client instead of the initialized global |
| `trace_name_prefix` | `"llamaindex"` | Prefix used when an operation name is unavailable |
| `dispatcher` | root | Dispatcher to register on (defaults to `get_dispatcher()`) |

**Everything is captured by default**, matching the other Noveum integrations —
a RAG trace without the query, the retrieved chunks and the synthesized answer
cannot be replayed, debugged or turned into an evaluation dataset. Set any flag
to `False` to reduce what is sent:

```python
# Reduce what leaves the process
setup_llamaindex_tracing(
    capture_inputs=False,
    capture_outputs=False,
    capture_llm_messages=False,
)
```

**Payloads are never truncated.** Prompts, retrieved node text and synthesized
answers routinely run past any fixed character budget, and a clipped node is
useless for evaluation, so they are recorded in full.

### Embedding capture

`capture_embedding_chunks` is the one flag that defaults to **off**, and it is
deliberately separate from `capture_inputs`.

Indexing a corpus emits one embedding call per batch, so capturing the chunk
*text* on embedding spans copies the entire source corpus into the trace store —
fine for a small index you want to build evaluation datasets from, expensive for
a large production one. Turn it on when you need the embedded text; leave it off
for bulk indexing.

Either way you still get `embedding.chunk_count`, `embedding.vector_count` and
`embedding.dimensions`. The **embedding vectors themselves are never attached** to
a span under any setting: a float array is not readable in a trace viewer and not
useful to an evaluation. Note that retrieved chunk text is a separate matter —
that arrives on `retrieval.nodes` under `capture_outputs`, and is bounded by
top-k rather than by corpus size.

## Version compatibility

| Component | Supported |
| --- | --- |
| `llama-index-core` | `>= 0.11, < 1.0` |
| Python | `>= 3.10` (required by current `llama-index-core`) |

The integration targets the stable `llama_index.core.instrumentation` dispatcher
(available since 0.10.x). Handlers read event/span fields defensively, so newer
event types are recorded generically rather than raising.

## Resilience

Failures inside the handlers (a mapping error, an unreachable Noveum backend) are
logged at debug level and never propagate. LlamaIndex additionally swallows
handler exceptions in its dispatcher, so tracing can never break a query.

## Troubleshooting

- **No traces appear:** confirm `noveum_trace.init(...)` ran before
  `setup_llamaindex_tracing()` (it now raises if the SDK is not initialized and
  no explicit `client=` is given). For short-lived processes, call
  `noveum_trace.flush()` then `noveum_trace.shutdown()` before exit — `flush()`
  sends buffered traces and `shutdown()` releases SDK resources.
- **Query text / node content missing:** these are on by default; check that
  `capture_inputs` / `capture_outputs` / `capture_llm_messages` were not disabled.
- **Token counts missing:** LlamaIndex exposes usage on the provider-native
  `response.raw`; some LLM integrations do not populate it.
- **Cost missing:** cost is derived from the model name on the start event plus
  the token counts on the end event — if either is absent (as with `MockLLM`, or
  a model the registry does not price), no cost attributes are written.
- **Embedded text missing:** `capture_embedding_chunks` is opt-in; see
  [Embedding capture](#embedding-capture).

## Example script

See [`docs/examples/llamaindex_integration_example.py`](examples/llamaindex_integration_example.py).

## Next steps

- [LlamaIndex docs](https://docs.llamaindex.ai/)
- [LlamaIndex instrumentation docs](https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation/)
