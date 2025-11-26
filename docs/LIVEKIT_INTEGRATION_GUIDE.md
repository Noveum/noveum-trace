# LiveKit Integration Guide

A step-by-step guide to add automatic tracing to your LiveKit agents using noveum-trace.

The complete solution consists of two components working together:
- **Session Tracing**: Automatic tracing of all AgentSession and RealtimeSession events
- **STT/TTS Tracing**: Detailed tracing of speech-to-text and text-to-speech operations with audio capture

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [How It Works](#how-it-works)
5. [Session Tracing Component](#session-tracing-component)
6. [STT/TTS Tracing Component](#stttts-tracing-component)
7. [Configuration Options](#configuration-options)
8. [Verification](#verification)
9. [Common Use Cases](#common-use-cases)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- A **LiveKit agent** project (or a new one to create)
- **LiveKit API credentials** (API key, secret, server URL)
- **Noveum API key** (get one at [noveum.ai](https://noveum.ai))

---

## Installation

### Step 1: Install Required Packages

```bash
# Install noveum-trace
pip install noveum-trace

# Install LiveKit and agents framework
pip install livekit livekit-agents

# Install your preferred STT/TTS providers
# Examples:
pip install livekit-plugins-deepgram   # For Deepgram STT
pip install livekit-plugins-cartesia   # For Cartesia TTS
pip install livekit-plugins-openai     # For OpenAI Whisper/TTS
pip install livekit-plugins-elevenlabs # For ElevenLabs TTS
```

### Step 2: Set Environment Variables

Create a `.env` file in your project root:

```bash
# LiveKit credentials
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret

# Noveum credentials
NOVEUM_API_KEY=your-noveum-api-key

# Optional: Specify project and environment
NOVEUM_PROJECT=my-livekit-agent
NOVEUM_ENVIRONMENT=production
```

---

## Quick Start

Here's the complete solution for tracing your LiveKit agent:

```python
import noveum_trace
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, cartesia
from noveum_trace.integrations.livekit import setup_livekit_tracing
from noveum_trace.integrations.livekit import LiveKitSTTWrapper, LiveKitTTSWrapper
from noveum_trace.integrations.livekit.livekit_utils import extract_job_context

# Initialize noveum-trace (do this once at startup)
noveum_trace.init(project="my-livekit-agent")

async def entrypoint(ctx: JobContext):
    # Extract job context for span attributes
    job_context = extract_job_context(ctx)
    
    # Wrap STT provider for detailed audio tracking
    traced_stt = LiveKitSTTWrapper(
        stt=deepgram.STT(model="nova-2", language="en-US"),
        session_id=ctx.job.id,
        job_context=job_context
    )
    
    # Wrap TTS provider for detailed audio tracking
    traced_tts = LiveKitTTSWrapper(
        tts=cartesia.TTS(model="sonic-english"),
        session_id=ctx.job.id,
        job_context=job_context
    )
    
    # Create session with traced providers
    session = AgentSession(stt=traced_stt, tts=traced_tts)
    
    # Enable session tracing (creates trace automatically at session.start())
    setup_livekit_tracing(session)
    
    # Create agent
    agent = Agent(instructions="You are a helpful assistant.")
    
    # Connect and start
    await ctx.connect()
    await session.start(agent)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

This complete solution provides:
- **Session Events**: Automatic tracing of all AgentSession and RealtimeSession events
- **STT/TTS Operations**: Detailed tracking with audio file capture
- **Automatic Trace Creation**: Trace is created when `session.start()` is called
- **Complete Observability**: Full visibility into your agent's behavior

---

## How It Works

The complete LiveKit integration consists of two components that work together:

1. **Session Tracing Component**: Automatically creates a trace when your session starts and tracks all AgentSession and RealtimeSession events
2. **STT/TTS Tracing Component**: Wraps your STT and TTS providers to capture detailed audio operations with file storage

These components are designed to work together - session tracing creates the trace context, and STT/TTS wrappers create spans within that context. You should use both components for complete observability.

---

## Session Tracing Component

The session tracing component automatically creates a trace when your agent session starts and creates spans for every event that fires. This component provides high-level visibility into your agent's behavior and state changes.

### What Gets Traced

**AgentSession Events** (10 events):
- `user_state_changed` - User state transitions (speaking/listening/away)
- `agent_state_changed` - Agent state transitions (initializing/idle/listening/thinking/speaking)
- `user_input_transcribed` - User speech transcriptions
- `conversation_item_added` - Messages added to conversation
- `agent_false_interruption` - False interruption detections
- `function_tools_executed` - Function/tool executions
- `metrics_collected` - Performance metrics
- `speech_created` - Agent speech generation starts
- `error` - Errors from LLM, STT, TTS, or RealtimeModel
- `close` - Session closure

**RealtimeSession Events** (7 events, when using RealtimeModel):
- `input_speech_started` - Server-side VAD detects speech start
- `input_speech_stopped` - Server-side VAD detects speech end
- `input_audio_transcription_completed` - Input audio transcription
- `generation_created` - New generation created
- `session_reconnected` - Session reconnection
- `metrics_collected` - Realtime model metrics
- `error` - Realtime model errors

### Basic Usage

```python
from livekit.agents import Agent, AgentSession, JobContext
from noveum_trace.integrations.livekit import setup_livekit_tracing
import noveum_trace

# Initialize once
noveum_trace.init(project="my-agent")

async def entrypoint(ctx: JobContext):
    session = AgentSession()
    agent = Agent(instructions="You are helpful.")
    
    # One line to enable tracing
    setup_livekit_tracing(session)
    
    await ctx.connect()
    await session.start(agent)  # Trace created automatically
```

### With RealtimeModel

Works automatically with RealtimeModel - no additional setup needed:

```python
from livekit.plugins import openai

session = AgentSession(
    llm=openai.realtime.RealtimeModel(model="gpt-realtime", voice="alloy")
)

setup_livekit_tracing(session)  # Automatically detects RealtimeSession
await session.start(agent)
```

### Configuration

```python
# Custom trace name prefix
setup_livekit_tracing(
    session,
    trace_name_prefix="my_agent"  # Traces will be: my_agent.agent_session.{job_id}
)

# Disable tracing (useful for testing)
setup_livekit_tracing(session, enabled=False)
```

### Event Data Serialization

All event objects are automatically serialized and added as span attributes. For example, a `user_input_transcribed` event will have attributes like:
- `user_input_transcribed.transcript` - The transcribed text
- `user_input_transcribed.is_final` - Whether it's a final transcript
- `user_input_transcribed.speaker_id` - Speaker ID (if available)
- `user_input_transcribed.language` - Detected language
- `user_input_transcribed.created_at` - Timestamp

### Parent Resolution

Spans use **context-based parent resolution** - each span automatically becomes a child of the currently active span in the context. This creates a natural hierarchy:
- Trace (root)
  - `livekit.user_state_changed` (child of trace)
  - `livekit.user_input_transcribed` (child of trace)
  - `livekit.function_tools_executed` (child of trace)
    - `livekit.function_tools_executed` nested calls (child of previous)

### Working with STT/TTS Tracing Component

The session tracing component is designed to work with the STT/TTS tracing component:
- **Session tracing creates the trace automatically** - you don't need to manually call `start_trace()`
- STT/TTS spans will automatically be children of the session trace
- Both components work seamlessly together
- Always use both components together for complete observability

---

## STT/TTS Tracing Component

The STT/TTS tracing component wraps your speech-to-text and text-to-speech providers to automatically capture detailed audio operations, transcripts, and metadata. This component works within the trace context created by the session tracing component.

### Step 1: Import the Wrappers

Add these imports to your agent file:

```python
import noveum_trace
from noveum_trace.integrations.livekit import (
    LiveKitSTTWrapper,
    LiveKitTTSWrapper,
)
```

### Step 2: Initialize Noveum Trace (if not already done)

Add this **once** at the top level of your application (before creating agents):

```python
import os

# Initialize with environment variables
noveum_trace.init(
    project=os.getenv("NOVEUM_PROJECT", "livekit-agent"),
    api_key=os.getenv("NOVEUM_API_KEY"),
    environment=os.getenv("NOVEUM_ENVIRONMENT", "production")
)
```

Or with explicit values:

```python
noveum_trace.init(
    project="my-voice-agent",
    api_key="your-api-key",
    environment="production"
)
```

### Step 3: Note About Trace Creation

When using the complete solution (both session tracing and STT/TTS tracing), **you don't need to manually create a trace**. The session tracing component automatically creates the trace when `session.start()` is called. The STT/TTS wrappers will automatically create spans within that trace context.

If you're only using STT/TTS tracing without session tracing (not recommended), you would need to manually create a trace:

```python
async def entrypoint(ctx: JobContext):
    # Only needed if NOT using session tracing
    with noveum_trace.start_trace(f"session_{ctx.job.id}"):
        # Your agent code with STT/TTS wrappers
        ...
```

### Step 4: Wrap Your STT Provider

**Before (without tracing):**

```python
stt = deepgram.STT(
    model="nova-2",
    language="en-US",
)
```

**After (with tracing):**

```python
from noveum_trace.integrations.livekit import LiveKitSTTWrapper

# Create base STT
base_stt = deepgram.STT(
    model="nova-2",
    language="en-US",
)

# Wrap it
traced_stt = LiveKitSTTWrapper(
    stt=base_stt,
    session_id=ctx.job.id,  # Use job ID for organizing audio files
    job_context={
        "job_id": ctx.job.id,
        "room_name": ctx.room.name,
        "participant": ctx.participant.identity if ctx.participant else None,
    }
)

# Use traced_stt everywhere you would use stt
```

### Step 5: Wrap Your TTS Provider

**Before (without tracing):**

```python
tts = cartesia.TTS(
    model="sonic-english",
    voice="a0e99841-438c-4a64-b679-ae501e7d6091",
)
```

**After (with tracing):**

```python
from noveum_trace.integrations.livekit import LiveKitTTSWrapper

# Create base TTS
base_tts = cartesia.TTS(
    model="sonic-english",
    voice="a0e99841-438c-4a64-b679-ae501e7d6091",
)

# Wrap it
traced_tts = LiveKitTTSWrapper(
    tts=base_tts,
    session_id=ctx.job.id,
    job_context={
        "job_id": ctx.job.id,
        "room_name": ctx.room.name,
    }
)

# Use traced_tts everywhere you would use tts
```

### Step 6: Use in Your Agent

The wrappers are **drop-in replacements** for the original providers:

```python
from livekit.agents.voice import AgentSession

async def entrypoint(ctx: JobContext):
    with noveum_trace.start_trace(f"session_{ctx.job.id}"):
        # Wrap providers
        traced_stt = LiveKitSTTWrapper(...)
        traced_tts = LiveKitTTSWrapper(...)
        
        # Create agent session with traced providers
        session = AgentSession(
            stt=traced_stt,  # ✅ Use traced version
            tts=traced_tts,  # ✅ Use traced version
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
        )
        
        # Connect and start
        await ctx.connect()
        await session.start()
```

---

## Configuration Options

### Job Context

The `job_context` parameter allows you to attach metadata to every span. Include any relevant information:

```python
job_context = {
    # Required
    "job_id": ctx.job.id,
    
    # Recommended
    "room_name": ctx.room.name,
    "room_sid": ctx.room.sid,
    "participant_identity": ctx.participant.identity,
    
    # Optional - add whatever is useful
    "agent_version": "1.2.3",
    "deployment": "us-west-2",
    "customer_id": "cust_123",
    "conversation_type": "support",
}
```

These will appear in spans as `job.job_id`, `job.room_name`, etc.

### Custom Audio Directory

By default, audio files are saved to `audio_files/{session_id}/`. You can customize this:

```python
from pathlib import Path

traced_stt = LiveKitSTTWrapper(
    stt=base_stt,
    session_id=ctx.job.id,
    job_context=job_context,
    audio_base_dir=Path("/mnt/audio_storage")  # Custom directory
)
```

### Extract Job Context Automatically

Use the utility function to extract context from LiveKit's JobContext:

```python
from noveum_trace.integrations.livekit.livekit_utils import extract_job_context

job_context = extract_job_context(ctx)
# Automatically extracts: job_id, room_name, room_sid, agent_id, worker_id, etc.
```

---

## Verification

### 1. Check Audio Files

After running your agent, verify audio files are being saved:

```bash
ls -la audio_files/
# Should show directories named after session IDs

ls -la audio_files/{session_id}/
# Should show WAV files like:
# stt_0001_1732386400000.wav
# stt_0002_1732386402500.wav
# tts_0001_1732386401000.wav
```

### 2. Check Traces

Log into your Noveum dashboard to see traces:

1. Go to [noveum.ai](https://noveum.ai)
2. Navigate to your project
3. You should see traces named `livekit_session_{job_id}`
4. Each trace should contain STT and TTS spans

### 3. Verify in Code

Add logging to confirm tracing is active:

```python
import logging

logger = logging.getLogger(__name__)

async def entrypoint(ctx: JobContext):
    logger.info(f"Starting session {ctx.job.id}")
    
    with noveum_trace.start_trace(f"session_{ctx.job.id}") as trace:
        logger.info(f"Trace created: {trace.trace_id}")
        
        # ... rest of code
        
        logger.info(f"Session {ctx.job.id} complete. Spans: {len(trace.spans)}")
```

---

## Common Use Cases

### Use Case 1: Basic Voice Agent

```python
async def entrypoint(ctx: JobContext):
    with noveum_trace.start_trace(f"session_{ctx.job.id}"):
        traced_stt = LiveKitSTTWrapper(
            stt=deepgram.STT(model="nova-2"),
            session_id=ctx.job.id,
            job_context={"job_id": ctx.job.id}
        )
        
        traced_tts = LiveKitTTSWrapper(
            tts=openai.TTS(voice="alloy"),
            session_id=ctx.job.id,
            job_context={"job_id": ctx.job.id}
        )
        
        session = AgentSession(stt=traced_stt, tts=traced_tts)
        await ctx.connect()
        await session.start()
```

### Use Case 2: Multiple Providers (Fallback)

```python
from livekit.agents.stt import FallbackAdapter

async def entrypoint(ctx: JobContext):
    with noveum_trace.start_trace(f"session_{ctx.job.id}"):
        # Primary STT
        primary_stt = LiveKitSTTWrapper(
            stt=deepgram.STT(model="nova-2"),
            session_id=ctx.job.id,
            job_context={"provider": "primary"}
        )
        
        # Fallback STT
        fallback_stt = LiveKitSTTWrapper(
            stt=openai.STT(),
            session_id=ctx.job.id,
            job_context={"provider": "fallback"}
        )
        
        # Use fallback adapter
        stt_with_fallback = FallbackAdapter(
            stt=primary_stt,
            fallback_stt=fallback_stt
        )
        
        # Both providers will be traced!
        session = AgentSession(stt=stt_with_fallback, ...)
        await session.start()
```

### Use Case 3: Different Languages

```python
async def entrypoint(ctx: JobContext):
    # Detect user's language
    user_language = detect_language(ctx)
    
    with noveum_trace.start_trace(f"session_{ctx.job.id}"):
        # Configure STT for user's language
        traced_stt = LiveKitSTTWrapper(
            stt=deepgram.STT(
                model="nova-2",
                language=user_language
            ),
            session_id=ctx.job.id,
            job_context={
                "job_id": ctx.job.id,
                "language": user_language  # Track language in spans
            }
        )
        
        # ... rest of code
```

### Use Case 4: Custom Pre/Post Processing

```python
async def entrypoint(ctx: JobContext):
    with noveum_trace.start_trace(f"session_{ctx.job.id}"):
        traced_stt = LiveKitSTTWrapper(...)
        traced_tts = LiveKitTTSWrapper(...)
        
        # Wrappers don't interfere with your processing
        stream = traced_stt.stream()
        
        async for event in stream:
            if event.type == "final_transcript":
                # Your custom processing
                text = event.alternatives[0].text
                processed_text = custom_filter(text)
                
                # TTS uses processed text
                tts_stream = traced_tts.synthesize(processed_text)
                # ... play audio
```

---

## Troubleshooting

### Issue: Audio Files Not Created

**Problem**: No audio files appear in `audio_files/` directory.

**Solutions**:

1. **Check permissions**:
   ```bash
   # Ensure write permissions
   mkdir -p audio_files
   chmod 755 audio_files
   ```

2. **Check disk space**:
   ```bash
   df -h
   # Ensure sufficient space
   ```

3. **Verify wrapper is used**:
   ```python
   # Add logging
   logger.info(f"Using STT: {type(traced_stt).__name__}")
   # Should print: "Using STT: LiveKitSTTWrapper"
   ```

### Issue: No Spans in Noveum

**Problem**: Traces appear empty (no spans).

**Solutions**:

1. **Verify trace is active**:
   ```python
   from noveum_trace import get_current_trace
   
   trace = get_current_trace()
   logger.info(f"Active trace: {trace.trace_id if trace else 'None'}")
   ```

2. **Check API key**:
   ```python
   import os
   logger.info(f"API key set: {'Yes' if os.getenv('NOVEUM_API_KEY') else 'No'}")
   ```

3. **Ensure trace context**:
   ```python
   # WRONG - trace not active
   trace = noveum_trace.start_trace(...)
   await session.start()
   
   # CORRECT - trace is active during session
   with noveum_trace.start_trace(...):
       await session.start()
   ```

### Issue: Import Error

**Problem**: `ImportError: cannot import name 'LiveKitSTTWrapper'`

**Solutions**:

1. **Check noveum-trace version**:
   ```bash
   pip show noveum-trace
   # Should be >= 1.0.1
   ```

2. **Reinstall if needed**:
   ```bash
   pip install --upgrade noveum-trace
   ```

3. **Verify LiveKit is installed**:
   ```bash
   pip install livekit livekit-agents
   ```

### Issue: Span Not Created

**Problem**: STT/TTS works but no spans are created.

**Solutions**:

1. **Check if trace exists**:
   ```python
   from noveum_trace import get_current_trace
   
   async def entrypoint(ctx: JobContext):
       trace = get_current_trace()
       if not trace:
           logger.warning("No active trace!")
       
       # Always use context manager
       with noveum_trace.start_trace(...):
           # Create wrappers inside the context
           traced_stt = LiveKitSTTWrapper(...)
   ```

2. **Check for exceptions**:
   - The wrapper silently catches exceptions to not disrupt your agent
   - Check logs for any errors

### Issue: High Disk Usage

**Problem**: Audio files taking up too much space.

**Solutions**:

1. **Implement cleanup**:
   ```python
   import os
   import time
   from pathlib import Path
   
   def cleanup_old_audio(max_age_hours=24):
       """Delete audio files older than max_age_hours"""
       audio_dir = Path("audio_files")
       cutoff = time.time() - (max_age_hours * 3600)
       
       for session_dir in audio_dir.iterdir():
           if session_dir.is_dir():
               for audio_file in session_dir.glob("*.wav"):
                   if audio_file.stat().st_mtime < cutoff:
                       audio_file.unlink()
   
   # Run periodically
   cleanup_old_audio(max_age_hours=24)
   ```

2. **Use external storage** (future enhancement):
   - Currently audio is saved locally
   - Consider syncing to S3/cloud storage separately

### Issue: Session Tracing Not Working

**Problem**: No trace or spans created when using `setup_livekit_tracing()`.

**Solutions**:

1. **Verify setup is called before start**:
   ```python
   # CORRECT
   session = AgentSession()
   setup_livekit_tracing(session)  # Before start
   await session.start(agent)
   
   # WRONG
   await session.start(agent)
   setup_livekit_tracing(session)  # Too late!
   ```

2. **Check if tracing is enabled**:
   ```python
   manager = setup_livekit_tracing(session, enabled=True)
   # Verify manager.enabled is True
   ```

3. **Check for import errors**:
   ```python
   from noveum_trace.integrations.livekit import setup_livekit_tracing
   # Should not raise ImportError
   ```

4. **Verify noveum-trace is initialized**:
   ```python
   import noveum_trace
   from noveum_trace import is_initialized
   
   noveum_trace.init(project="my-agent", api_key="...")
   assert is_initialized(), "Noveum trace not initialized"
   ```

### Issue: RealtimeSession Events Not Traced

**Problem**: Using RealtimeModel but RealtimeSession events aren't appearing.

**Solutions**:

1. **Verify RealtimeModel is used**:
   ```python
   from livekit.plugins import openai
   
   session = AgentSession(
       llm=openai.realtime.RealtimeModel(...)  # Must be RealtimeModel
   )
   ```

2. **Check if RealtimeSession is detected**:
   - The integration automatically detects RealtimeSession after `session.start()`
   - There's a small delay (0.1s) for session initialization
   - Events should appear after the first agent state change

3. **Verify event handlers are registered**:
   ```python
   # Check logs for: "RealtimeSession handlers registered"
   ```

### Issue: Performance Impact

**Problem**: Agent feels slower with tracing.

**Solutions**:

1. **Audio saving is asynchronous** - shouldn't impact performance
2. **Check disk I/O**:
   ```bash
   iostat -x 1
   # Monitor disk utilization
   ```

3. **Use faster storage** if needed (SSD recommended)

4. **Disable audio saving temporarily**:
   ```python
   # Patch save functions for testing
   from unittest.mock import patch
   
   with patch("noveum_trace.integrations.livekit.save_audio_frames"):
       traced_stt = LiveKitSTTWrapper(...)
       # Audio won't be saved, but spans still created
   ```

---

## Complete Examples

### Example 1: Complete Solution (Recommended)

Here's a complete, production-ready example using both components together:

```python
"""
LiveKit Voice Agent with Complete Tracing
"""

import logging
import os

import noveum_trace
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, openai
from noveum_trace.integrations.livekit import (
    LiveKitSTTWrapper,
    LiveKitTTSWrapper,
)
from noveum_trace.integrations.livekit import setup_livekit_tracing
from noveum_trace.integrations.livekit.livekit_utils import extract_job_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize noveum-trace
noveum_trace.init(
    project=os.getenv("NOVEUM_PROJECT", "my-voice-agent"),
    api_key=os.getenv("NOVEUM_API_KEY"),
    environment=os.getenv("NOVEUM_ENVIRONMENT", "production"),
)

async def entrypoint(ctx: JobContext):
    logger.info(f"Starting session {ctx.job.id} in room {ctx.room.name}")
    
    # Extract job context
    job_context = extract_job_context(ctx)
    
    # Wrap STT provider
    traced_stt = LiveKitSTTWrapper(
        stt=deepgram.STT(model="nova-2", language="en-US"),
        session_id=ctx.job.id,
        job_context=job_context,
    )
    
    # Wrap TTS provider
    traced_tts = LiveKitTTSWrapper(
        tts=openai.TTS(voice="alloy"),
        session_id=ctx.job.id,
        job_context=job_context,
    )
    
    # Create agent session
    session = AgentSession(stt=traced_stt, tts=traced_tts)
    
    # Setup session tracing (creates trace automatically at start)
    setup_livekit_tracing(session)
    
    # Create agent
    agent = Agent(instructions="You are a helpful customer support agent.")
    
    # Connect and start
    await ctx.connect()
    
    # Start the session
    # - Session tracing creates trace automatically
    # - STT/TTS wrappers create spans for each operation
    await session.start(agent)
    
    logger.info(f"Session {ctx.job.id} complete")
    logger.info(f"Audio files saved to: audio_files/{ctx.job.id}/")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

---

## Next Steps

- **View your traces**: Log into [noveum.ai](https://noveum.ai) to see your agent's traces
- **Analyze audio**: Find saved audio files in `audio_files/{session_id}/`
- **Read the API docs**: See [`LIVEKIT_INTEGRATION.md`](LIVEKIT_INTEGRATION.md) for detailed API reference
- **Check examples**: See [`livekit_integration_example.py`](examples/livekit_integration_example.py) for more examples

---

## Support

- **Documentation**: See [`LIVEKIT_INTEGRATION.md`](LIVEKIT_INTEGRATION.md)
- **Issues**: Report issues on GitHub
- **Email**: [support@noveum.ai](mailto:support@noveum.ai)

---

## Summary Checklist

### Complete Solution Setup:
- [ ] Install `noveum-trace` and LiveKit packages
- [ ] Set environment variables (API keys)
- [ ] Initialize `noveum_trace.init()` at startup
- [ ] Import `setup_livekit_tracing` from `noveum_trace.integrations.livekit`
- [ ] Import wrapper classes from `noveum_trace.integrations.livekit`
- [ ] Wrap STT provider with `LiveKitSTTWrapper`
- [ ] Wrap TTS provider with `LiveKitTTSWrapper`
- [ ] Create session with wrapped providers
- [ ] Call `setup_livekit_tracing(session)` before `session.start()`
- [ ] Test and verify traces in Noveum dashboard
- [ ] Verify audio files are saved to `audio_files/{session_id}/`
- [ ] Deploy and monitor in production

Happy tracing! 🎉

## Related Documentation

- **Events Reference**: See [`LIVEKIT_EVENTS_REFERENCE.md`](LIVEKIT_EVENTS_REFERENCE.md) for complete list of trackable events
- **API Reference**: See [`LIVEKIT_INTEGRATION.md`](LIVEKIT_INTEGRATION.md) for detailed API documentation
- **Examples**: See [`livekit_integration_example.py`](examples/livekit_integration_example.py) for more examples

