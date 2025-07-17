"""
Core module for Noveum Trace SDK.

This module contains the fundamental tracing primitives including
configuration, context management, spans, traces, and the main client.
"""

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
]
