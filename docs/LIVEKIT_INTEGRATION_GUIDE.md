# LiveKit Integration Guide

A step-by-step guide to add automatic tracing to your LiveKit agents using noveum-trace.

The complete solution consists of two components working together:
- **Session Tracing**: Automatic tracing of all AgentSession and RealtimeSession events
- **STT/TTS Tracing**: Detailed tracing of speech-to-text and text-to-speech operations with audio capture
- **Full Conversation Audio**: Automatic upload of the complete conversation recording

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [How It Works](#how-it-works)
5. [Session Tracing Component](#session-tracing-component)
6. [STT/TTS Tracing Component](#stttts-tracing-component)
7. [Full Conversation Audio Recording](#full-conversation-audio-recording)
8. [Configuration Options](#configuration-options)
9. [Verification](#verification)
10. [Common Use Cases](#common-use-cases)
11. [Troubleshooting](#troubleshooting)
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
#Specify project and environment
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
noveum_trace.init(
    project="my-livekit-agent",
    api_key="your-noveum-api-key",
    environment="production"
)

async def entrypoint(ctx: JobContext):
    # Extract job context for span attributes
    job_context = await extract_job_context(ctx)
    
    # Wrap STT provider for detailed audio tracking (per-utterance)
    traced_stt = LiveKitSTTWrapper(
        stt=deepgram.STT(model="nova-2", language="en-US"),
        session_id=ctx.job.id,
        job_context=job_context
    )
    
    # Wrap TTS provider for detailed audio tracking (per-utterance)
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
    
    # Connect and start with recording enabled for full conversation audio
    await ctx.connect()
    await session.start(agent, record=True)  # record=True enables full audio capture

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

This complete solution provides:
- **Session Events**: Automatic tracing of all AgentSession and RealtimeSession events
- **STT/TTS Operations**: Detailed tracking with per-utterance audio capture
- **Full Conversation Audio**: Complete stereo recording uploaded at session end (user=left, agent=right)
- **LLM Metrics**: Token usage, cost, and latency captured from LiveKit events
- **Chat History**: Full conversation history in JSON format on generation spans
- **Available Tools**: Function/tool definitions captured in traces
- **Automatic Trace Creation**: Trace is created when `session.start()` is called

---

## How It Works

The complete LiveKit integration consists of three components that work together:

1. **Session Tracing Component**: Automatically creates a trace when your session starts and tracks all AgentSession and RealtimeSession events
2. **STT/TTS Tracing Component**: Wraps your STT and TTS providers to capture detailed per-utterance audio operations
3. **Full Conversation Audio**: LiveKit's RecorderIO captures the complete conversation as a stereo OGG file, which is automatically uploaded at session close

These components are designed to work together - session tracing creates the trace context, STT/TTS wrappers create spans for individual operations, and the full conversation audio provides a complete recording for review.

---


### What Gets Captured

| Component | Span Name | Data Captured |
|-----------|-----------|---------------|
| STT Wrapper | `stt.stream` | Per-utterance audio (WAV), transcript, confidence, language |
| TTS Wrapper | `tts.stream` | Per-utterance audio (WAV), input text, voice settings |
| Session Events | `livekit.*` | All AgentSession and RealtimeSession events |
| LLM Generation | `livekit.realtime.generation_created` | Chat history, available tools, function calls |
| LLM Metrics | (merged into spans) | Token usage, cost, latency, model info |
| Full Audio | `livekit.full_conversation` | Complete conversation (OGG stereo), uploaded at session end |

### Span Attributes

**STT Stream Spans** (`stt.stream`):
- `stt.audio_uuid`: UUID for audio retrieval
- `stt.transcript`: Transcribed text
- `stt.confidence`: Transcription confidence (0-1)
- `stt.language`: Detected language
- `stt.provider`: STT provider (e.g., "Deepgram")
- `stt.model`: Model used (e.g., "nova-2")

**TTS Stream Spans** (`tts.stream`):
- `tts.audio_uuid`: UUID for audio retrieval
- `tts.input_text`: Text that was synthesized
- `tts.provider`: TTS provider (e.g., "Cartesia")
- `tts.model`: Model used

**Generation Spans** (`livekit.realtime.generation_created`):
- `llm.conversation.history`: Full chat history as JSON
- `llm.conversation.message_count`: Number of messages
- `llm.available_tools.count`: Number of available tools
- `llm.available_tools.names`: List of tool names
- `llm.available_tools.schemas`: Full tool schemas as JSON
- `llm.function_calls`: Function calls made (when merged)
- `llm.function_outputs`: Function outputs (when merged)

**Full Conversation Audio** (`livekit.full_conversation`):
- `stt.audio_uuid`: UUID for audio retrieval
- `stt.audio_format`: "ogg"
- `stt.audio_channels`: "stereo"
- `stt.audio_channel_left`: "user"
- `stt.audio_channel_right`: "agent"

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
noveum_trace.init(
    project="my-agent",
    api_key="your-noveum-api-key",
    environment="production"
)

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

## Full Conversation Audio Recording

The integration automatically uploads the complete conversation audio when the session ends. This requires enabling LiveKit's RecorderIO.

### How to Enable Recording

#### Option 1: In Code (Recommended)

```python
# Enable recording when starting the session
await session.start(agent, record=True)
```

#### Option 2: CLI Flag (Console Mode)

When running in console mode, add the `--record` flag:

```bash
python your_agent.py --test console --record
```

### How It Works

1. **LiveKit's RecorderIO** captures audio during the session
2. Audio is saved as a **stereo OGG/Opus file**:
   - Left channel: User audio (microphone)
   - Right channel: Agent audio (TTS output)
3. **At session close**, the SDK:
   - Creates an `livekit.full_conversation` span
   - Uploads the OGG file to Noveum

### Recording Storage

- **Console mode**: `console-recordings/session-{timestamp}/audio.ogg`
- **Production mode**: Managed by LiveKit Cloud

### Span Format

The full conversation creates a span like:

```json
{
  "name": "livekit.full_conversation",
  "attributes": {
    "stt.audio_uuid": "e37942f0-77b6-4380-a652-defd33e60b7e",
    "stt.audio_format": "ogg",
    "stt.audio_channels": "stereo",
    "stt.audio_channel_left": "user",
    "stt.audio_channel_right": "agent",
    "stt.audio_source": "livekit_recorder_io"
  }
}
```

### When Recording is NOT Available

If `record=True` is not set, the SDK will log:
```log
_upload_full_conversation_audio: No recording available. Ensure session.start(record=True) was called.
```

Per-utterance STT/TTS audio will still be captured via the wrappers.

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

The `job_context` parameter allows you to attach metadata to every span. Use `extract_job_context()` to get the standard fields, then add any additional information:

#### Extract Job Context Automatically

Use the utility function to extract context from LiveKit's JobContext:

```python
from noveum_trace.integrations.livekit.livekit_utils import extract_job_context

job_context = await extract_job_context(ctx)
# Automatically extracts: job_id, room_name, room_sid, agent_id, worker_id, etc.
```

#### Adding Custom Fields

```python
from noveum_trace.integrations.livekit.livekit_utils import extract_job_context

# Get standard fields automatically
job_context = await extract_job_context(ctx)

# Add your custom metadata
job_context.update({
    "agent_version": "1.2.3",
    "deployment": "us-west-2",
    "customer_id": "cust_123",
    "conversation_type": "support",
})
```

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
   
   noveum_trace.init(
       project="my-agent",
       api_key="...",
       environment="production"
   )
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

### Issue: Full Conversation Audio Not Uploaded

**Problem**: No `livekit.full_conversation` span appears in traces.

**Solutions**:

1. **Enable recording in session.start()**:
   ```python
   # CORRECT - recording enabled
   await session.start(agent, record=True)
   
   # WRONG - recording not enabled
   await session.start(agent)
   ```

2. **For console mode, add --record flag**:
   ```bash
   # CORRECT
   python your_agent.py --test console --record
   
   # WRONG - no recording
   python your_agent.py --test console
   ```

3. **Check for the recording message in logs**:
   ```
   Recording   Session recording will be saved to console-recordings/session-{timestamp}
   ```

4. **Verify the session closes properly**:
   - Use a `complete_order` or similar tool to close the session gracefully
   - Avoid Ctrl+C which may interrupt the upload

5. **Check if RecorderIO is available**:
   ```python
   # The SDK checks session._recorder_io for the recording path
   # If None, recording wasn't enabled
   ```

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
    job_context = await extract_job_context(ctx)
    
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

## Related Documentation

- **Events Reference**: See [`LIVEKIT_EVENTS_REFERENCE.md`](LIVEKIT_EVENTS_REFERENCE.md) for complete list of trackable events
- **API Reference**: See [`LIVEKIT_INTEGRATION.md`](LIVEKIT_INTEGRATION.md) for detailed API documentation
- **Examples**: See [`livekit_integration_example.py`](examples/livekit_integration_example.py) for more examples

