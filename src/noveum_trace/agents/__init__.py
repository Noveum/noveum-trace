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
from .registry import AgentRegistry, get_agent_registry
from .context import AgentContext, AsyncAgentContext, get_current_agent, set_current_agent
from .decorators import trace, observe, llm_trace, update_current_span

__all__ = [
    "Agent",
    "AgentConfig", 
    "AgentRegistry",
    "get_agent_registry",
    "AgentContext",
    "AsyncAgentContext",
    "get_current_agent",
    "set_current_agent",
    "trace",
    "observe", 
    "llm_trace",
    "update_current_span",
]