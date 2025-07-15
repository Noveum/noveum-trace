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

__version__ = "0.1.0"
__author__ = "Noveum Team"
__email__ = "team@noveum.ai"

# Simplified API (recommended for most users)
from .init import (
    init,
    configure,
    setup,
    enable_auto_instrumentation,
    disable_auto_instrumentation,
    get_tracer,
    shutdown,
    flush,
    NoveumTrace,
)

# Core components (for advanced users)
from .core.tracer import NoveumTracer, TracerConfig
from .core.span import Span
from .core.context import TraceContext

# Sinks
from .sinks.base import BaseSink
from .sinks.file import FileSink, FileSinkConfig
from .sinks.noveum import NoveumSink, NoveumConfig
from .sinks.console import ConsoleSink, ConsoleSinkConfig

# Types
from .types import (
    SpanData, LLMRequest, LLMResponse, Message, TokenUsage,
    OperationType, AISystem, SpanKind, SpanStatus
)

# Instrumentation
from .instrumentation.decorators import trace_function, trace_llm_call
from .instrumentation import openai, anthropic

# Exceptions
from .utils.exceptions import (
    NoveumTracingError, ConfigurationError, NetworkError, ValidationError
)

# Multi-agent support
from .agents import (
    AgentRegistry, get_agent_registry,
    Agent, AgentConfig,
    AgentContext, get_current_agent, set_current_agent
)

# Simplified decorators (unified approach)
from .agents.decorators import trace, update_current_span

# Backward compatibility aliases
from .agents.decorators import observe, llm_trace

# Main exports (what users typically need)
__all__ = [
    # Simplified API
    "init",
    "configure", 
    "setup",
    "enable_auto_instrumentation",
    "disable_auto_instrumentation", 
    "get_tracer",
    "shutdown",
    "flush",
    "NoveumTrace",
    
    # Core components
    "NoveumTracer",
    "TracerConfig",
    "Span",
    "TraceContext",
    
    # Sinks
    "BaseSink",
    "FileSink",
    "FileSinkConfig",
    "NoveumSink", 
    "NoveumConfig",
    "ConsoleSink",
    "ConsoleSinkConfig",
    
    # Types
    "SpanData",
    "LLMRequest",
    "LLMResponse", 
    "Message",
    "TokenUsage",
    "OperationType",
    "AISystem",
    "SpanKind",
    "SpanStatus",
    
    # Instrumentation
    "trace_function",
    "trace_llm_call",
    "openai",
    "anthropic",
    
    # Multi-agent support
    "AgentRegistry",
    "get_agent_registry",
    "Agent",
    "AgentConfig",
    "AgentContext",
    "get_current_agent",
    "set_current_agent",
    "trace",
    "observe",
    "llm_trace",
    "update_current_span",
    
    # Exceptions
    "NoveumTracingError",
    "ConfigurationError", 
    "NetworkError",
    "ValidationError",
]

