"""
LangChain integration for Noveum Trace SDK.

This package provides a callback handler that automatically traces LangChain
operations including LLM calls, chains, agents, tools, and retrieval operations.
"""

from noveum_trace.integrations.langchain.langchain import NoveumTraceCallbackHandler

__all__ = ["NoveumTraceCallbackHandler"]
