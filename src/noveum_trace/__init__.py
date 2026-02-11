"""
Noveum Trace SDK - Cloud-first, flexible tracing for LLM applications.

This package provides comprehensive observability for LLM applications and
multi-agent systems through multiple flexible tracing approaches.

Import Patterns:
    Recommended: Direct imports from package root

    >>> from noveum_trace import init, trace_context, trace, NoveumClient
    >>> from noveum_trace import trace_llm, trace_agent, trace_tool
    >>> from noveum_trace import trace_llm_call, trace_operation

    Alternative: Module-level imports

    >>> import noveum_trace
    >>> noveum_trace.init(project="my-app")
    >>> noveum_trace.trace_context(...)

    Submodule imports (when needed):

    >>> from noveum_trace.decorators import trace
    >>> from noveum_trace.integrations import NoveumTraceCallbackHandler
    >>> from noveum_trace.core.client import NoveumClient

    For complete import documentation, see README.md

Example:
    Basic usage with decorators:

    >>> import noveum_trace
    >>> noveum_trace.init(project="my-app")
    >>>
    >>> @noveum_trace.trace
    >>> def my_function(data: str) -> str:
    ...     return process_data(data)

    Or using direct imports:

    >>> from noveum_trace import init, trace
    >>> init(project="my-app")
    >>>
    >>> @trace
    >>> def my_function(data: str) -> str:
    ...     return process_data(data)

    LLM tracing with context managers:

    >>> def process_query(query: str) -> str:
    ...     # Pre-processing (not traced)
    ...     cleaned_query = clean_query(query)
    ...
    ...     # LLM call (traced)
    ...     with noveum_trace.trace_llm_call(model="gpt-4") as span:
    ...         response = openai.chat.completions.create(...)
    ...         span.set_attribute("llm.tokens", response.usage.total_tokens)
    ...
    ...     # Post-processing (not traced)
    ...     return format_response(response)

    Or using direct imports:

    >>> from noveum_trace import trace_llm_call
    >>> with trace_llm_call(model="gpt-4") as span:
    ...     response = openai.chat.completions.create(...)

"""

__version__ = "1.5.7"
__author__ = "Noveum Team"
__email__ = "engineering@noveum.ai"
__license__ = "Apache-2.0"

import threading
from typing import Any, Optional

# Agent imports
from noveum_trace.agents import (
    AgentGraph,
    AgentNode,
    AgentWorkflow,
    cleanup_agent,
    cleanup_agent_graph,
    cleanup_agent_workflow,
    cleanup_all_registries,
    cleanup_by_ttl,
    create_agent,
    create_agent_graph,
    create_agent_workflow,
    enforce_size_limits,
    get_agent,
    get_agent_graph,
    get_agent_workflow,
    get_registry_stats,
    temporary_agent_context,
)
from noveum_trace.agents import trace_agent_operation as trace_agent_op

# Context manager imports
from noveum_trace.context_managers import (
    create_child_span,
)
from noveum_trace.context_managers import trace_agent as trace_agent_operation
from noveum_trace.context_managers import (
    trace_batch_operation,
    trace_function_calls,
)
from noveum_trace.context_managers import trace_llm as trace_llm_call
from noveum_trace.context_managers import (
    trace_operation,
    trace_pipeline_stage,
)

# Core imports
from noveum_trace.core.client import NoveumClient
from noveum_trace.core.config import DEFAULT_ENDPOINT, configure, get_config
from noveum_trace.core.context import (
    ContextualTrace,
    get_current_span,
    get_current_trace,
    trace_context,
)
from noveum_trace.core.span import Span
from noveum_trace.core.trace import Trace

# Decorator imports
from noveum_trace.decorators.agent import trace_agent
from noveum_trace.decorators.base import trace
from noveum_trace.decorators.llm import trace_llm
from noveum_trace.decorators.retrieval import trace_retrieval
from noveum_trace.decorators.tool import trace_tool

# Streaming imports
from noveum_trace.streaming import (
    create_anthropic_streaming_callback,
    create_openai_streaming_callback,
    streaming_llm,
    trace_streaming,
)

# Thread imports
from noveum_trace.threads import (
    ThreadContext,
    create_thread,
    delete_thread,
    get_thread,
    list_threads,
    trace_thread_llm,
)

# Utility imports
from noveum_trace.utils.exceptions import (
    ConfigurationError,
    InitializationError,
    InstrumentationError,
    NoveumTraceError,
    TracingError,
    TransportError,
)

# Global client instance
_client: Optional[NoveumClient] = None
_client_lock = threading.Lock()


def init(
    project: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    environment: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Initialize the Noveum Trace SDK.

    This function sets up the global tracing client.

    Args:
        project: Project name for organizing traces
        api_key: Noveum API key (defaults to NOVEUM_API_KEY env var)
        endpoint: API endpoint (defaults to https://api.noveum.ai/api)
        environment: Environment name (dev, staging, prod)
        **kwargs: Additional configuration options including:
                 - transport_config: Transport layer configuration
                 - tracing_config: Tracing behavior configuration
                 - security_config: Security settings configuration
                 - integrations_config: Integration settings

    Example:
        >>> import noveum_trace
        >>> noveum_trace.init(
        ...     project="my-llm-app",
        ...     environment="production",
        ...     transport_config={
        ...         "batch_size": 10,
        ...         "batch_timeout": 1.0
        ...     }
        ... )
    """
    global _client

    with _client_lock:
        # Check if client is already initialized
        if _client is not None:
            return

        # Configure the SDK - handle endpoint explicitly to ensure defaults work
        config = {}
        if project is not None:
            config["project"] = project
        if api_key is not None:
            config["api_key"] = api_key
        # Always include endpoint in config to ensure consistent behavior
        # If None, use the default endpoint
        config["endpoint"] = endpoint if endpoint is not None else DEFAULT_ENDPOINT
        if environment is not None:
            config["environment"] = environment

        # Handle special config parameter mappings
        config_mappings = {
            "transport_config": "transport",
            "tracing_config": "tracing",
            "security_config": "security",
            "integrations_config": "integrations",
        }

        # Process kwargs and map config parameters to correct keys
        for key, value in kwargs.items():
            if key in config_mappings:
                # Map transport_config -> transport, etc.
                config[config_mappings[key]] = value
            else:
                # Pass through other kwargs as-is
                config[key] = value

        configure(config)

        # Initialize the client
        _client = NoveumClient()

        # Register with core module to avoid circular imports
        from noveum_trace.core import _register_client

        _register_client(_client)


def shutdown() -> None:
    """
    Shutdown the Noveum Trace SDK.

    This function flushes any pending traces and cleanly shuts down
    the tracing client.
    """
    global _client
    with _client_lock:
        if _client:
            _client.shutdown()
            _client = None

            # Unregister from core module
            from noveum_trace.core import _unregister_client

            _unregister_client()


def flush() -> None:
    """
    Flush any pending traces to the Noveum platform.

    This function blocks until all pending traces have been sent
    or the flush timeout is reached.
    """
    global _client
    with _client_lock:
        if _client:
            _client.flush()


def get_client() -> NoveumClient:
    """
    Get the global Noveum client instance.

    Returns:
        The global NoveumClient instance

    Raises:
        InitializationError: If the SDK has not been initialized
    """
    global _client
    with _client_lock:
        if _client is None:
            raise InitializationError(
                "Noveum Trace SDK not initialized. Call noveum_trace.init() first."
            )
        return _client


def is_initialized() -> bool:
    """
    Check if the Noveum Trace SDK has been initialized.

    Returns:
        True if initialized, False otherwise
    """
    global _client
    with _client_lock:
        return _client is not None


# Plugin system
def register_plugin(plugin: Any) -> None:
    """
    Register a custom plugin with the Noveum Trace SDK.

    Args:
        plugin: Plugin instance implementing the BasePlugin interface
    """
    # from noveum_trace.plugins import register_plugin as _register_plugin
    # _register_plugin(plugin)
    raise NotImplementedError("Plugin system not yet implemented")


def list_plugins() -> list[str]:
    """
    List all registered plugins.

    Returns:
        List of registered plugin names
    """
    # from noveum_trace.plugins import list_plugins as _list_plugins
    # return _list_plugins()
    raise NotImplementedError("Plugin system not yet implemented")


# Convenience functions for manual instrumentation
def start_trace(name: str, **kwargs: Any) -> "ContextualTrace":
    """
    Manually start a new trace.

    Args:
        name: Trace name
        **kwargs: Additional trace attributes

    Returns:
        New ContextualTrace instance that can be used as a context manager
    """
    client = get_client()
    return client.create_contextual_trace(name, **kwargs)


def start_span(name: str, **kwargs: Any) -> Span:
    """
    Manually start a new span.

    Args:
        name: Span name
        **kwargs: Additional span attributes

    Returns:
        New Span instance
    """
    client = get_client()
    return client.start_span(name, **kwargs)


# Integrations imports (conditional)
_integration_exports = []
try:
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    _integration_exports.append("NoveumTraceCallbackHandler")
except ImportError:
    # LangChain not installed
    pass

try:
    from noveum_trace.integrations.livekit import (
        LiveKitSTTWrapper,
        LiveKitTTSWrapper,
    )

    _integration_exports.extend(["LiveKitSTTWrapper", "LiveKitTTSWrapper"])
except ImportError:
    # LiveKit not installed
    pass

# Export public API
__all__ = [
    # Core functions
    "init",
    "shutdown",
    "flush",
    "configure",
    "get_config",
    "get_client",
    "is_initialized",
    # Decorators
    "trace",
    "trace_llm",
    "trace_agent",
    "trace_tool",
    "trace_retrieval",
    # Context managers
    "trace_context",
    "trace_llm_call",
    "trace_agent_operation",
    "trace_operation",
    "trace_batch_operation",
    "trace_pipeline_stage",
    "create_child_span",
    "trace_function_calls",
    # Streaming
    "trace_streaming",
    "streaming_llm",
    "create_openai_streaming_callback",
    "create_anthropic_streaming_callback",
    # Threads
    "create_thread",
    "get_thread",
    "delete_thread",
    "list_threads",
    "trace_thread_llm",
    "ThreadContext",
    # Agents
    "create_agent",
    "get_agent",
    "create_agent_graph",
    "get_agent_graph",
    "create_agent_workflow",
    "get_agent_workflow",
    "trace_agent_op",
    "AgentNode",
    "AgentGraph",
    "AgentWorkflow",
    # Agent cleanup
    "cleanup_agent",
    "cleanup_agent_graph",
    "cleanup_agent_workflow",
    "cleanup_all_registries",
    "cleanup_by_ttl",
    "enforce_size_limits",
    "get_registry_stats",
    "temporary_agent_context",
    # Context management
    "get_current_trace",
    "get_current_span",
    # Manual instrumentation
    "start_trace",
    "start_span",
    # Core classes
    "Trace",
    "Span",
    "NoveumClient",
    # Plugin system
    "register_plugin",
    "list_plugins",
    # Integrations (conditional)
    *_integration_exports,
    # Exceptions
    "NoveumTraceError",
    "ConfigurationError",
    "TransportError",
    "InstrumentationError",
    "TracingError",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
