"""
LiveKit integration for Noveum Trace SDK.

This package provides wrappers for LiveKit STT, TTS, and LLM providers,
as well as session tracing with automatic tool extraction and conversation history.
"""

from noveum_trace.integrations.livekit.livekit_llm import (
    LiveKitLLMWrapper,
    extract_available_tools,
    serialize_chat_history,
    serialize_function_calls,
    serialize_tools_for_attributes,
)
from noveum_trace.integrations.livekit.livekit_session import setup_livekit_tracing
from noveum_trace.integrations.livekit.livekit_stt import LiveKitSTTWrapper
from noveum_trace.integrations.livekit.livekit_tts import LiveKitTTSWrapper
from noveum_trace.integrations.livekit.livekit_utils import extract_job_context

__all__ = [
    # Wrappers
    "LiveKitLLMWrapper",
    "LiveKitSTTWrapper",
    "LiveKitTTSWrapper",
    # Session tracing
    "setup_livekit_tracing",
    # Utility functions
    "extract_job_context",
    "extract_available_tools",
    "serialize_chat_history",
    "serialize_function_calls",
    "serialize_tools_for_attributes",
]
