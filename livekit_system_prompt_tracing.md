# LiveKit System Prompt Tracing - Implementation Plan

This document provides a step-by-step guide for implementing system prompt tracing in LiveKit AgentSession integration.

## Overview

The system prompt is captured at two locations:
1. **Trace level**: Extracted from `agent.instructions` when the trace is created
2. **speech_created event**: Extracted via background task waiting for `agent_activity` to become available

## Implementation Steps

### Step 1: Trace Level System Prompt

**File**: `src/noveum_trace/integrations/livekit/livekit_session.py`

**Location**: Inside `_LiveKitTracingManager._wrap_start_method()` method, within the `wrapped_start()` async function

**When**: During trace creation when `session.start(agent)` is called

**Implementation**:

Find the section where trace attributes are being built (around where `attributes` dictionary is created). Add the following code:

```python
# Extract agent instructions (system prompt) if available
if hasattr(agent, "instructions") and agent.instructions:
    attributes["llm.system_prompt"] = agent.instructions
elif hasattr(agent, "_instructions") and agent._instructions:
    attributes["llm.system_prompt"] = agent._instructions
```

**Placement**: This should be added BEFORE the trace is created with `client.start_trace()`. Typically right after adding agent label and before adding job context.

**Exact Location**: Look for code like:
```python
# Add agent label if available
if hasattr(agent, "label"):
    attributes["livekit.agent.label"] = agent.label

# <-- ADD SYSTEM PROMPT EXTRACTION HERE -->

# Add job context if available
try:
    from livekit.agents import get_job_context
    ...
```

**Result**: System prompt will be stored in trace attributes as `llm.system_prompt` and will appear at the trace level in exported traces.

**Verification**: Check that the trace JSON includes `"llm.system_prompt"` in the `attributes` section at the trace level.

---

### Step 2: Create Helper Function for Span Updates

**File**: `src/noveum_trace/integrations/livekit/livekit_session.py`

**Location**: Add as a new async function, typically near other helper functions like `_update_speech_span_with_chat_items()`

**Implementation**:

Create the following async function:

```python
async def _update_span_with_system_prompt(
    span: Any,
    manager: Any,
    max_wait_seconds: float = 5.0,
    check_interval: float = 0.1,
) -> None:
    """
    Wait for agent_activity to become available, then update span with system prompt.

    Args:
        span: Span to update
        manager: _LiveKitTracingManager instance
        max_wait_seconds: Maximum time to wait for agent_activity
        check_interval: Interval between checks
    """
    try:
        start_time = asyncio.get_event_loop().time()

        # Wait for agent_activity to become available
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_wait_seconds:
                return

            # Check if agent_activity is available (try both property and attribute)
            agent_activity = None
            
            # Try property first
            if hasattr(manager.session, "agent_activity"):
                try:
                    agent_activity = manager.session.agent_activity
                except Exception:
                    pass
            
            # Fallback to checking _activity attribute directly
            if not agent_activity and hasattr(manager.session, "_activity"):
                try:
                    agent_activity = manager.session._activity
                except Exception:
                    pass

            if agent_activity:
                if hasattr(agent_activity, "_agent") and agent_activity._agent:
                    agent = agent_activity._agent

                    # Extract system prompt from agent.instructions (most reliable source)
                    system_prompt = None
                    if hasattr(agent, "instructions") and agent.instructions:
                        system_prompt = agent.instructions
                    elif hasattr(agent, "_instructions") and agent._instructions:
                        system_prompt = agent._instructions

                    if system_prompt:
                        # Directly modify span.attributes (bypassing set_attribute since span may be finished)
                        # This follows the same pattern as _update_speech_span_with_chat_items
                        span.attributes["llm.system_prompt"] = system_prompt
                        return

            # Wait before next check
            await asyncio.sleep(check_interval)

    except Exception:
        pass
```

**Function Location**: Add this function as a module-level function, typically placed right after `_update_speech_span_with_chat_items()` function (around line 326 in the reference implementation).

**Key Points**:
- Uses async/await to wait for `agent_activity` to become available
- Checks both `session.agent_activity` (property) and `session._activity` (attribute) as fallback
- Extracts from `agent.instructions` or `agent._instructions` (same source used for LLM calls)
- Directly modifies `span.attributes` because the span may already be finished (similar to `_update_speech_span_with_chat_items`)
- Polls every 0.1 seconds for up to 5 seconds maximum

---

### Step 3: Add System Prompt Extraction for speech_created Events

**File**: `src/noveum_trace/integrations/livekit/livekit_session.py`

**Location**: Inside `_create_event_span()` function, AFTER the span is finished with `client.finish_span(span)`

**Implementation**:

Find the section where spans are finished (after `client.finish_span(span)`). Add the following code:

```python
# For speech_created events, start background task to update span with system prompt
# (waiting for agent_activity to become available)
if manager and event_type == "speech_created":
    # Check if we already have system prompt in attributes
    if "llm.system_prompt" not in attributes:
        # Start background task to wait for agent_activity and update span
        asyncio.create_task(
            _update_span_with_system_prompt(span, manager)
        )
```

**Placement**: This should be added AFTER `client.finish_span(span)` but BEFORE `return span`.

**Exact Location**: Look for code like:
```python
# Finish span immediately (events are instantaneous)
client.finish_span(span)

# <-- ADD SYSTEM PROMPT EXTRACTION HERE -->

return span
```

**Why after finish_span**: The span is finished immediately for event spans, but we can still update `span.attributes` directly (similar to how `_update_speech_span_with_chat_items` works).

**Result**: System prompt will be stored in the `speech_created` span attributes as `llm.system_prompt` after `agent_activity` becomes available.

**Verification**: Check that `speech_created` spans in the exported trace JSON include `"llm.system_prompt"` in their `attributes` section.

---

## Important Notes

1. **Why wait for agent_activity**: The `agent_activity` may not be immediately available when `speech_created` events fire. The background task polls until it becomes available (up to 5 seconds).

2. **Direct attribute modification**: We use `span.attributes["llm.system_prompt"] = system_prompt` instead of `span.set_attribute()` because:
   - The span may already be finished when the async task runs
   - `set_attribute()` checks `if self._finished: return self` and does nothing on finished spans
   - This pattern matches `_update_speech_span_with_chat_items` which also updates finished spans

3. **Source priority**: Always extract from `agent.instructions` first (most reliable), then fallback to `agent._instructions`. This is the same source used when making LLM calls to ChatOpenAI.

4. **Import requirement**: Ensure `asyncio` is imported at the top of the file (should already be present).

---

## Testing Checklist

- [ ] Trace level: Verify `llm.system_prompt` appears in trace attributes
- [ ] speech_created span: Verify `llm.system_prompt` appears in `speech_created` span attributes
- [ ] System prompt content: Verify the extracted prompt matches `agent.instructions`
- [ ] Timing: Verify system prompt appears even if `agent_activity` isn't immediately available

---

## Code Structure Reference

### File Structure
```
src/noveum_trace/integrations/livekit/livekit_session.py
├── _update_speech_span_with_chat_items()  # ~line 282
├── _update_span_with_system_prompt()      # <-- ADD HERE (~line 328)
├── _create_event_span()                   # ~line 458
│   └── [Step 3 code goes here]            # ~line 757
└── class _LiveKitTracingManager
    └── _wrap_start_method()
        └── wrapped_start()                # ~line 741
            └── [Step 1 code goes here]    # ~line 876
```

### Similar Pattern Reference

See `_update_speech_span_with_chat_items()` function (around line 282) for a similar pattern:
- Waiting for something to become available (`speech_handle.wait_for_playout()`)
- Updating finished span attributes directly (`span.attributes.update()`)
- Handling exceptions gracefully

This function follows the exact same pattern but waits for `agent_activity` instead of `speech_handle.wait_for_playout()`.

## Common Pitfalls to Avoid

1. **Don't use `span.set_attribute()`**: It won't work on finished spans. Always use direct attribute modification: `span.attributes["llm.system_prompt"] = system_prompt`

2. **Don't forget the check**: Always check `if "llm.system_prompt" not in attributes:` before starting the background task to avoid duplicate extraction

3. **Don't extract before span is finished**: The background task must be started AFTER `client.finish_span(span)` because event spans are finished immediately

4. **Handle both property and attribute**: Always check both `session.agent_activity` (property) and `session._activity` (attribute) because the implementation may vary
