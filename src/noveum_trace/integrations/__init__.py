"""
Integration modules for third-party frameworks and libraries.

This package contains integration modules that provide seamless tracing
capabilities for popular frameworks and libraries used in LLM applications.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = []

# Conditional imports based on available dependencies
try:
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    __all__.append("NoveumTraceCallbackHandler")
except ImportError:
    # LangChain not installed
    pass

# LiveKit integration
try:
    from noveum_trace.integrations.livekit import (
        LiveKitSTTWrapper,
        LiveKitTTSWrapper,
    )

    __all__.extend(["LiveKitSTTWrapper", "LiveKitTTSWrapper"])
except ImportError as e:
    logger.error(
        "Failed to import LiveKit integration modules. "
        "LiveKit integration features will not be available.",
        exc_info=e,
    )

# LiveKit session tracing integration
try:
    from noveum_trace.integrations.livekit import setup_livekit_tracing

    __all__.append("setup_livekit_tracing")
except ImportError as e:
    logger.error(
        "Failed to import LiveKit session tracing integration. "
        "LiveKit session tracing features will not be available.",
        exc_info=e,
    )
try:
    from noveum_trace.integrations.livekit import extract_job_context

    __all__.append("extract_job_context")
except ImportError as e:
    logger.error(
        "Failed to import LiveKit job context extraction. "
        "LiveKit job context extraction features will not be available.",
        exc_info=e,
    )
