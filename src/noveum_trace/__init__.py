"""
Noveum Trace SDK

A high-performance, OpenTelemetry-compliant tracing SDK for LLM applications.
Provides automatic instrumentation, real-time evaluation, and dataset creation.

Quick Start:
    ```python
    import noveum_trace

    # Simple initialization
    noveum_trace.init()

    # Your LLM calls are now automatically traced!
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(...)
    ```

For more advanced usage, see the documentation and examples.
"""

__version__ = "0.1.2"
__author__ = "Noveum Team"
__email__ = "team@noveum.ai"

# Multi-agent support
from .agents import (
    Agent,
    AgentConfig,
    AgentContext,
    AgentRegistry,
    get_agent_registry,
    get_current_agent,
    set_current_agent,
)

# Backward compatibility aliases
# Simplified decorators (unified approach)
from .agents.decorators import llm_trace, observe, trace, update_current_span
from .core.context import TraceContext
from .core.span import Span

# Core components (for advanced users)
from .core.tracer import NoveumTracer, TracerConfig

# Simplified API (recommended for most users)
from .init import (
    NoveumTrace,
    configure,
    disable_auto_instrumentation,
    enable_auto_instrumentation,
    flush,
    get_tracer,
    init,
    setup,
    shutdown,
)
from .instrumentation import anthropic, openai

# Instrumentation
from .instrumentation.decorators import (
    trace_function,
    trace_llm_call,
    trace_streaming_llm_call,
)

# Sinks
from .sinks.base import BaseSink
from .sinks.console import ConsoleSink, ConsoleSinkConfig
from .sinks.elasticsearch import ElasticsearchConfig, ElasticsearchSink
from .sinks.file import FileSink, FileSinkConfig
from .sinks.noveum import NoveumConfig, NoveumSink

# Types
from .types import (
    AISystem,
    LLMRequest,
    LLMResponse,
    Message,
    OperationType,
    SpanData,
    SpanKind,
    SpanStatus,
    TokenUsage,
)

# Exceptions
from .utils.exceptions import (
    ConfigurationError,
    NetworkError,
    NoveumTracingError,
    ValidationError,
)

# Main exports (what users typically need)
__all__ = [
    "AISystem",
    # Sorted alphabetically
    "Agent",
    "AgentConfig",
    "AgentContext",
    "AgentRegistry",
    "BaseSink",
    "ConfigurationError",
    "ConsoleSink",
    "ConsoleSinkConfig",
    "ElasticsearchConfig",
    "ElasticsearchSink",
    "FileSink",
    "FileSinkConfig",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "NetworkError",
    "NoveumConfig",
    "NoveumSink",
    "NoveumTrace",
    "NoveumTracer",
    "NoveumTracingError",
    "OperationType",
    "Span",
    "SpanData",
    "SpanKind",
    "SpanStatus",
    "TokenUsage",
    "TraceContext",
    "TracerConfig",
    "ValidationError",
    "anthropic",
    "configure",
    "disable_auto_instrumentation",
    "enable_auto_instrumentation",
    "flush",
    "get_agent_registry",
    "get_current_agent",
    "get_tracer",
    "init",
    "llm_trace",
    "observe",
    "openai",
    "set_current_agent",
    "setup",
    "shutdown",
    "trace",
    "trace_function",
    "trace_llm_call",
    "trace_streaming_llm_call",
    "update_current_span",
]
