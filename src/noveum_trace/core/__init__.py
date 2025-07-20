"""
Core module for Noveum Trace SDK.

This module contains the fundamental tracing primitives including
configuration, context management, spans, traces, and the main client.
"""

import threading
from typing import Optional

from noveum_trace.core.client import NoveumClient
from noveum_trace.core.config import Config, configure, get_config
from noveum_trace.core.context import (
    TraceContext,
    get_current_span,
    get_current_trace,
    trace_context,
)
from noveum_trace.core.span import Span, SpanStatus
from noveum_trace.core.trace import Trace

__all__ = [
    "NoveumClient",
    "Config",
    "configure",
    "get_config",
    "TraceContext",
    "trace_context",
    "get_current_trace",
    "get_current_span",
    "Span",
    "SpanStatus",
    "Trace",
    "get_global_client",
    "is_client_initialized",
]


# Global client registry to avoid circular imports
_global_client: Optional[NoveumClient] = None
_client_lock = threading.Lock()


def _register_client(client: NoveumClient) -> None:
    """Register the global client instance (internal use only)."""
    global _global_client
    with _client_lock:
        _global_client = client


def _unregister_client() -> None:
    """Unregister the global client instance (internal use only)."""
    global _global_client
    with _client_lock:
        _global_client = None


def get_global_client() -> Optional[NoveumClient]:
    """
    Get the global client instance without raising errors.

    Returns:
        The global client instance if initialized, None otherwise
    """
    global _global_client
    with _client_lock:
        return _global_client


def is_client_initialized() -> bool:
    """
    Check if the global client is initialized.

    Returns:
        True if client is initialized, False otherwise
    """
    global _global_client
    with _client_lock:
        return _global_client is not None
