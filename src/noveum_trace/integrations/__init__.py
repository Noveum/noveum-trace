"""
Integration modules for third-party frameworks and libraries.

This package contains integration modules that provide seamless tracing
capabilities for popular frameworks and libraries used in LLM applications.
"""

__all__ = []

# Conditional imports based on available dependencies
try:
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    __all__.append("NoveumTraceCallbackHandler")
except ImportError:
    # LangChain not installed
    pass
