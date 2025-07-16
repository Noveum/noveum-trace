"""
Multi-agent support for Noveum Trace SDK.

This module provides comprehensive multi-agent tracing capabilities including:
- Agent registry and management
- Agent-aware context management
- Cross-agent correlation and relationships
- Hierarchical agent structures

The decorators are imported separately to maintain clean separation.
"""

from .agent import Agent, AgentConfig
from .context import (
    AgentContext,
    AsyncAgentContext,
    get_current_agent,
    set_current_agent,
)
from .decorators import llm_trace, observe, trace, update_current_span
from .registry import AgentRegistry, get_agent_registry

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentContext",
    "AgentRegistry",
    "AsyncAgentContext",
    "get_agent_registry",
    "get_current_agent",
    "llm_trace",
    "observe",
    "set_current_agent",
    "trace",
    "update_current_span",
]
