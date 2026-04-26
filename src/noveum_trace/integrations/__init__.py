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
except ImportError:
    # LiveKit not installed
    pass

# LiveKit session tracing integration
try:
    from noveum_trace.integrations.livekit import setup_livekit_tracing

    __all__.append("setup_livekit_tracing")
except ImportError:
    # LiveKit not installed
    pass

try:
    from noveum_trace.integrations.livekit import extract_job_context

    __all__.append("extract_job_context")
except ImportError:
    # LiveKit not installed
    pass

# Pipecat integration
try:
    from noveum_trace.integrations.pipecat import (
        NoveumTraceObserver,
        setup_pipecat_tracing,
    )

    __all__.extend(["NoveumTraceObserver", "setup_pipecat_tracing"])
except ImportError:
    pass

# CrewAI integration (requires Python 3.10+ and crewai>=0.177.0)
try:
    from noveum_trace.integrations.crewai import (
        NoveumCrewAIListener,
        setup_crewai_tracing,
    )

    __all__.extend(["NoveumCrewAIListener", "setup_crewai_tracing"])
except ImportError:
    pass
