"""
Constants for the LlamaIndex integration.

Structure: span attribute key constants (reusing the canonical ``llm.*`` keys the
rest of the SDK understands), an operation → span-type classifier map, status
values, and numeric limits.

Span hierarchy mirrors the LlamaIndex instrumentation span tree, e.g.::

    RetrieverQueryEngine.query            ← root (one Noveum trace)
      VectorIndexRetriever.retrieve       ← retrieval
        OpenAIEmbedding.get_query_embedding  ← embedding
      CompactAndRefine.synthesize         ← synthesize
        OpenAI.chat                        ← llm
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Span-type classification
# ---------------------------------------------------------------------------

SPAN_TYPE_LLM = "llm"
SPAN_TYPE_EMBEDDING = "embedding"
SPAN_TYPE_RETRIEVAL = "retrieval"
SPAN_TYPE_QUERY = "query"
SPAN_TYPE_SYNTHESIZE = "synthesize"
SPAN_TYPE_RERANK = "rerank"
SPAN_TYPE_AGENT = "agent"
SPAN_TYPE_OTHER = "other"

# Substring (lower-cased) → span-type, checked in order against the operation name
# (e.g. ``"VectorIndexRetriever.retrieve"``) when no event has classified the span.
OPERATION_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("retriev", SPAN_TYPE_RETRIEVAL),
    ("rerank", SPAN_TYPE_RERANK),
    ("postprocess", SPAN_TYPE_RERANK),
    ("embed", SPAN_TYPE_EMBEDDING),
    ("synthe", SPAN_TYPE_SYNTHESIZE),
    ("get_response", SPAN_TYPE_SYNTHESIZE),
    ("chat", SPAN_TYPE_LLM),
    ("complete", SPAN_TYPE_LLM),
    ("predict", SPAN_TYPE_LLM),
    ("query", SPAN_TYPE_QUERY),
    ("agent", SPAN_TYPE_AGENT),
)

# ---------------------------------------------------------------------------
# Common / trace attribute keys   (prefix: llamaindex.*)
# ---------------------------------------------------------------------------

ATTR_OPERATION = "llamaindex.operation"
ATTR_SPAN_TYPE = "llamaindex.span_type"
ATTR_FRAMEWORK = "llamaindex.framework"
ATTR_STATUS = "llamaindex.status"

FRAMEWORK_NAME = "llama_index"

# ---------------------------------------------------------------------------
# LLM attribute keys   (prefix: llm.* — consumed by the gen_ai crosswalk)
# ---------------------------------------------------------------------------

ATTR_LLM_MODEL = "llm.model"
ATTR_LLM_PROVIDER = "llm.provider"
ATTR_LLM_INPUT = "llm.input"
ATTR_LLM_OUTPUT = "llm.output"
ATTR_LLM_SYSTEM_PROMPT = "llm.system_prompt"
ATTR_LLM_AVAILABLE_TOOLS = "llm.available_tools"
ATTR_LLM_AVAILABLE_TOOL_COUNT = "llm.available_tool_count"
ATTR_LLM_TOOL_CALLS = "llm.tool_calls"
ATTR_LLM_TOOL_CALL_COUNT = "llm.tool_call_count"
ATTR_LLM_INPUT_TOKENS = "llm.input_tokens"
ATTR_LLM_OUTPUT_TOKENS = "llm.output_tokens"
ATTR_LLM_TOTAL_TOKENS = "llm.total_tokens"
ATTR_LLM_CACHED_INPUT_TOKENS = "llm.cached_input_tokens"
ATTR_LLM_REASONING_TOKENS = "llm.reasoning_tokens"
ATTR_LLM_COST_INPUT = "llm.cost.input"
ATTR_LLM_COST_OUTPUT = "llm.cost.output"
ATTR_LLM_COST_TOTAL = "llm.cost.total"
ATTR_LLM_COST_CURRENCY = "llm.cost.currency"

# ---------------------------------------------------------------------------
# Tool attribute keys   (prefix: tool.*)
# ---------------------------------------------------------------------------

ATTR_TOOL_NAME = "tool.name"
ATTR_TOOL_DESCRIPTION = "tool.description"
ATTR_TOOL_INPUT = "tool.input"

# ---------------------------------------------------------------------------
# Embedding attribute keys   (prefix: embedding.*)
# ---------------------------------------------------------------------------

ATTR_EMBEDDING_MODEL = "embedding.model"
ATTR_EMBEDDING_CHUNK_COUNT = "embedding.chunk_count"
ATTR_EMBEDDING_VECTOR_COUNT = "embedding.vector_count"
ATTR_EMBEDDING_DIMENSIONS = "embedding.dimensions"
ATTR_EMBEDDING_CHUNKS = "embedding.chunks"

# ---------------------------------------------------------------------------
# Retrieval attribute keys   (prefix: retrieval.*)
# ---------------------------------------------------------------------------

ATTR_RETRIEVAL_QUERY = "retrieval.query"
ATTR_RETRIEVAL_NODE_COUNT = "retrieval.node_count"
ATTR_RETRIEVAL_SCORES = "retrieval.scores"
ATTR_RETRIEVAL_NODES = "retrieval.nodes"
ATTR_RETRIEVAL_TOP_K = "retrieval.top_k"

# ---------------------------------------------------------------------------
# Query attribute keys   (prefix: query.*)
# ---------------------------------------------------------------------------

ATTR_QUERY_TEXT = "query.text"
ATTR_QUERY_RESPONSE = "query.response"
ATTR_QUERY_SOURCE_NODES = "query.source_nodes"

# ---------------------------------------------------------------------------
# Rerank attribute keys   (prefix: rerank.*)
# ---------------------------------------------------------------------------

ATTR_RERANK_MODEL = "rerank.model"
ATTR_RERANK_TOP_N = "rerank.top_n"
ATTR_RERANK_QUERY = "rerank.query"
ATTR_RERANK_INPUT_NODE_COUNT = "rerank.input_node_count"
ATTR_RERANK_OUTPUT_NODE_COUNT = "rerank.output_node_count"
ATTR_RERANK_INPUT_NODES = "rerank.input_nodes"
ATTR_RERANK_OUTPUT_NODES = "rerank.output_nodes"
ATTR_RERANK_INPUT_SCORES = "rerank.input_scores"
ATTR_RERANK_OUTPUT_SCORES = "rerank.output_scores"

# ---------------------------------------------------------------------------
# Error / status attribute keys
# ---------------------------------------------------------------------------

ATTR_ERROR_TYPE = "error.type"
ATTR_ERROR_MESSAGE = "error.message"

STATUS_OK = "ok"
STATUS_ERROR = "error"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Payloads are recorded in full. Prompts, retrieved node text and synthesized
# answers routinely exceed any fixed character budget, and a clipped node is
# not usable for evaluation or replay, so this integration does not truncate.
DEFAULT_TRACE_NAME_PREFIX = "llamaindex"

# Informational — the minimum ``llama-index-core`` release the instrumentation
# API used here was verified against (see ``pyproject.toml`` for the pin).
MIN_LLAMA_INDEX_CORE_VERSION = "0.11.0"
