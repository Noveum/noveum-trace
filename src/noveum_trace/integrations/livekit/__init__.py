"""
LiveKit integration for Noveum Trace SDK.

This package provides wrappers for LiveKit STT, TTS providers,
as well as session tracing with automatic tool extraction, conversation history,
and LLM metrics capture.

For voice agents using AgentSession, use setup_livekit_tracing() which
automatically captures:
- LLM metrics (tokens, cost, model, TTFT, latency)
- Conversation history
- Function calls
- Available tools
- Full conversation audio

Example:
    >>> from noveum_trace.integrations.livekit import (
    ...     setup_livekit_tracing,
    ...     LiveKitSTTWrapper,
    ...     LiveKitTTSWrapper,
    ... )
    >>>
    >>> @server.rtc_session()
    >>> async def my_agent(ctx: JobContext):
    ...     traced_stt = LiveKitSTTWrapper(stt=base_stt, ...)
    ...     traced_tts = LiveKitTTSWrapper(tts=base_tts, ...)
    ...     llm = openai_plugin.LLM(model="gpt-4o-mini")  # No wrapper needed!
    ...
    ...     session = AgentSession(stt=traced_stt, llm=llm, tts=traced_tts, ...)
    ...     setup_livekit_tracing(session)  # Captures LLM metrics automatically
    ...     await session.start(agent, room=ctx.room, participant=participant)
"""

# Import session tracing (recommended for AgentSession)
from noveum_trace.integrations.livekit.livekit_session import setup_livekit_tracing

# Import STT/TTS wrappers
from noveum_trace.integrations.livekit.livekit_stt import LiveKitSTTWrapper
from noveum_trace.integrations.livekit.livekit_tts import LiveKitTTSWrapper

# Import utility functions
from noveum_trace.integrations.livekit.livekit_utils import (
    extract_available_tools,
    extract_job_context,
    serialize_chat_history,
    serialize_function_calls,
    serialize_tools_for_attributes,
)

__all__ = [
    # Session tracing (recommended)
    "setup_livekit_tracing",
    # STT/TTS Wrappers
    "LiveKitSTTWrapper",
    "LiveKitTTSWrapper",
    # Utility functions
    "extract_job_context",
    "extract_available_tools",
    "serialize_chat_history",
    "serialize_function_calls",
    "serialize_tools_for_attributes",
]
