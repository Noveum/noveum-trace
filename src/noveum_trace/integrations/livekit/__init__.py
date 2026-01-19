"""
LiveKit integration for Noveum Trace SDK.

This package provides automatic tracing for LiveKit AgentSession and RealtimeSession,
as well as optional wrappers for STT and TTS providers.

For voice agents using AgentSession, use setup_livekit_tracing() which automatically
captures (no manual configuration needed):
- LLM metrics (tokens, cost, model, TTFT, latency) from metrics_collected events
- Conversation history from LiveKit's built-in ChatContext
- Function calls and outputs from conversation_item_added and function_tools_executed events
- Available tools extracted from the agent
- Full conversation audio from LiveKit's RecorderIO (when record=True)
- All AgentSession and RealtimeSession events as spans

The recommended approach is to use setup_livekit_tracing() which hooks into the session's
event system. Optionally, you can also wrap STT/TTS providers for individual operation tracing.

Example (Recommended - Automatic Tracing):
    >>> from livekit.agents import AgentSession, Agent
    >>> from noveum_trace.integrations.livekit import setup_livekit_tracing
    >>>
    >>> session = AgentSession(stt=stt, tts=tts, llm=llm)
    >>> setup_livekit_tracing(session)  # Automatic tracing for all events
    >>>
    >>> agent = Agent(instructions="You are helpful.")
    >>> await session.start(agent, record=True)  # Enables audio recording

Example (Optional - Manual STT/TTS Wrapping):
    >>> from noveum_trace.integrations.livekit import (
    ...     setup_livekit_tracing,
    ...     LiveKitSTTWrapper,
    ...     LiveKitTTSWrapper,
    ...     extract_job_context,
    ... )
    >>>
    >>> @server.rtc_session()
    >>> async def my_agent(ctx: JobContext):
    ...     job_context = await extract_job_context(ctx)
    ...     traced_stt = LiveKitSTTWrapper(
    ...         stt=base_stt,
    ...         session_id="session_123",
    ...         job_context=job_context
    ...     )
    ...     traced_tts = LiveKitTTSWrapper(
    ...         tts=base_tts,
    ...         session_id="session_123",
    ...         job_context=job_context
    ...     )
    ...
    ...     session = AgentSession(stt=traced_stt, llm=llm, tts=traced_tts)
    ...     setup_livekit_tracing(session)  # Still recommended for LLM metrics
    ...     await session.start(agent, record=True)
"""

# Import session tracing (recommended for AgentSession)
from noveum_trace.integrations.livekit.livekit_session import setup_livekit_tracing

# Import STT/TTS wrappers
from noveum_trace.integrations.livekit.livekit_stt import LiveKitSTTWrapper
from noveum_trace.integrations.livekit.livekit_tts import LiveKitTTSWrapper

# Import utility functions
from noveum_trace.integrations.livekit.livekit_utils import extract_job_context

__all__ = [
    # Session tracing (recommended)
    "setup_livekit_tracing",
    # STT/TTS Wrappers
    "LiveKitSTTWrapper",
    "LiveKitTTSWrapper",
    # Utility functions
    "extract_job_context",
]
