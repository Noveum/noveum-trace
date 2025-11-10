# Call Site Tracking for LangChain Integration

## Overview

This document outlines the plan to track exact call site information (function name, file name, line number) for Tool/LLM calls in the LangChain integration. This will help with debugging, performance analysis, and understanding usage patterns.

## Current Architecture

### How Tracing Works Now

1. **Callback Handler**: `NoveumTraceCallbackHandler` receives events from LangChain
   - `on_llm_start()` - Called when LLM invocation begins
   - `on_tool_start()` - Called when tool execution begins
   - `on_chain_start()` - Called when chain execution begins
   - `on_agent_start()` - Called when agent execution begins
   - `on_retriever_start()` - Called when retriever execution begins

2. **Event Flow**: 
   ```
   User Code → LangChain → Callback Handler → Span Creation
   ```

3. **Current Challenge**: 
   - When callback handler methods execute, we're inside LangChain's execution context
   - We need to walk up the call stack to find the user's code that initiated the call

## Solution Design

### Approach: Stack Frame Inspection

Use Python's `inspect` module to walk the call stack and identify the first frame that belongs to user code (not LangChain or internal libraries).

### Implementation Plan

#### Step 1: Create Utility Function for Call Site Extraction

**Location**: `src/noveum_trace/integrations/langchain_utils.py`

**Function**: `extract_call_site_info()`

**Purpose**: Walk up the call stack to find the first user code frame

**Key Features**:
- Skip frames from LangChain libraries (`langchain`, `langchain_core`, `langchain_openai`, etc.)
- Skip frames from Noveum Trace code (`noveum_trace`)
- Skip frames from Python standard library
- Extract: filename, line number, function name, module name, code context

**Implementation Details**:

```python
import inspect
from typing import Optional, Any

def extract_call_site_info(skip_frames: int = 0) -> Optional[dict[str, Any]]:
    """
    Extract call site information from the call stack.
    
    Walks up the call stack to find the first frame that's NOT in:
    - LangChain code (langchain, langchain_core, langchain_openai, etc.)
    - Noveum Trace code (noveum_trace)
    - Python standard library
    
    Args:
        skip_frames: Number of frames to skip from the top (default: 0)
        
    Returns:
        Dict with call site info: {
            "call_site.file": str,
            "call_site.line": int,
            "call_site.function": str,
            "call_site.module": str,
            "call_site.code_context": Optional[str]  # The actual line of code
        }
        Returns None if no user code frame found
    """
    try:
        stack = inspect.stack()
        
        # Skip the current frame (this function) and any requested frames
        start_idx = 1 + skip_frames
        
        # Patterns to skip (internal libraries)
        skip_patterns = [
            'langchain',
            'noveum_trace',
            'site-packages',
            '<frozen',
            'importlib',
        ]
        
        for frame_info in stack[start_idx:]:
            frame = frame_info.frame
            filename = frame_info.filename
            
            # Skip if this frame is in a library we want to ignore
            if any(pattern in filename for pattern in skip_patterns):
                continue
            
            # Skip if this is a builtin or internal Python code
            if filename.startswith('<') or 'site-packages' in filename:
                continue
            
            # Found user code! Extract information
            try:
                code_context = None
                if frame_info.code_context and len(frame_info.code_context) > 0:
                    code_context = frame_info.code_context[0].strip()
                
                return {
                    "call_site.file": filename,
                    "call_site.line": frame_info.lineno,
                    "call_site.function": frame_info.function,
                    "call_site.module": frame_info.frame.f_globals.get('__name__', 'unknown'),
                    "call_site.code_context": code_context,
                }
            except Exception:
                # If extraction fails, continue to next frame
                continue
        
        # No user code frame found
        return None
        
    except Exception:
        # If stack inspection fails, return None (fail gracefully)
        return None
```

#### Step 2: Integrate into Callback Handler Methods

**Location**: `src/noveum_trace/integrations/langchain.py`

**Methods to Modify**:

1. **`on_llm_start()`** - Add call site info to LLM spans
2. **`on_tool_start()`** - Add call site info to tool spans
3. **`on_chain_start()`** - Optionally add for chains
4. **`on_agent_start()`** - Optionally add for agents
5. **`on_retriever_start()`** - Optionally add for retrievers

**Example Integration in `on_llm_start()`**:

```python
def on_llm_start(
    self,
    serialized: dict[str, Any],
    prompts: list[str],
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """Handle LLM start event."""
    if not self._ensure_client():
        return

    operation_name = get_operation_name("llm_start", serialized)
    
    try:
        # Extract Noveum-specific metadata
        noveum_config = extract_noveum_metadata(metadata)
        custom_name = noveum_config.get("name")
        parent_name = noveum_config.get("parent_name")

        # Use custom name if provided, otherwise use operation name
        span_name = custom_name if custom_name else operation_name

        # Resolve parent span ID based on mode
        parent_span_id = self._resolve_parent_span_id(parent_run_id, parent_name)

        # Get or create trace context
        trace, should_manage = self._get_or_create_trace_context(
            span_name, run_id, parent_run_id
        )

        # Extract call site information
        from noveum_trace.integrations.langchain_utils import extract_call_site_info
        call_site_info = extract_call_site_info(skip_frames=1)  # Skip this frame

        # Extract the actual model name and provider
        # ... existing model extraction code ...

        span_attributes: dict[str, Any] = {
            "langchain.run_id": str(run_id),
            "llm.model": extracted_model_name,
            "llm.provider": extracted_provider,
            "llm.operation": "completion",
            "llm.input.prompts": prompts[:5] if len(prompts) > 5 else prompts,
            "llm.input.prompt_count": len(prompts),
            **attribute_kwargs,
        }

        # Add call site information if available
        if call_site_info:
            span_attributes.update(call_site_info)

        # Create span (either in new trace or existing trace)
        span = self._client.start_span(
            name=span_name,
            parent_span_id=parent_span_id,
            attributes=span_attributes,
        )

        # ... rest of existing code ...
```

**Example Integration in `on_tool_start()`**:

```python
def on_tool_start(
    self,
    serialized: dict[str, Any],
    input_str: str,
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    inputs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """Handle tool start event."""
    if not self._ensure_client():
        return

    get_operation_name("tool_start", serialized)

    try:
        # Extract Noveum-specific metadata
        noveum_config = extract_noveum_metadata(metadata)
        custom_name = noveum_config.get("name")
        parent_name = noveum_config.get("parent_name")

        tool_name = serialized.get("name", "unknown") if serialized else "unknown"
        func_name = extract_tool_function_name(serialized)
        span_name = custom_name if custom_name else f"tool:{tool_name}:{func_name}"

        # Resolve parent span ID based on mode
        parent_span_id = self._resolve_parent_span_id(parent_run_id, parent_name)

        # Get or create trace context
        trace, should_manage = self._get_or_create_trace_context(
            span_name, run_id, parent_run_id
        )

        # Extract call site information
        from noveum_trace.integrations.langchain_utils import extract_call_site_info
        call_site_info = extract_call_site_info(skip_frames=1)

        # Prepare input attributes
        input_attrs = {
            "tool.input.input_str": input_str,
            # ... existing input handling ...
        }

        span = self._client.start_span(
            name=span_name,
            parent_span_id=parent_span_id,
            attributes={
                "langchain.run_id": str(run_id),
                "tool.name": tool_name,
                "tool.operation": func_name,
                **input_attrs,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["tags", "metadata", "inputs"]
                    and isinstance(v, (str, int, float, bool))
                },
                # Add call site information if available
                **(call_site_info or {}),
            },
        )

        # ... rest of existing code ...
```

#### Step 3: Handle Edge Cases

1. **Performance Considerations**:
   - Stack inspection has overhead - consider limiting depth
   - Cache results if same call site is hit multiple times (optional)
   - Make it optional via configuration flag

2. **Thread Safety**:
   - `inspect.stack()` is thread-safe
   - No additional synchronization needed

3. **Failure Handling**:
   - If extraction fails, continue without call site info
   - Log warnings only in debug mode
   - Never break existing functionality

4. **Frame Filtering**:
   - Skip frames from LangChain libraries
   - Skip frames from Noveum Trace code
   - Skip frames from Python standard library
   - Skip frames from site-packages

5. **Special Cases**:
   - Decorators: May need to skip additional frames
   - Async code: Stack may look different
   - Nested calls: Should find the outermost user call

#### Step 4: Optional Enhancements

1. **Configurable Depth**:
   - Allow limiting how far up the stack to search
   - Default: search all frames (with pattern filtering)

2. **Metadata Override**:
   - Allow users to pass call site info via metadata
   - Useful for testing or when stack inspection fails

3. **Code Context**:
   - Include the actual line of code (if available)
   - Helps with debugging

4. **Relative Paths**:
   - Optionally store relative file paths instead of absolute
   - Useful for portability

5. **Performance Mode**:
   - Add configuration flag to disable call site tracking
   - Useful for high-performance scenarios

## Example Usage

### Before Implementation

```python
# File: my_app.py, Line: 47
def my_function():
    llm = ChatOpenAI(callbacks=[callback_handler])
    response = llm.invoke("Hello")  # No call site tracking
```

**Span Attributes** (current):
```python
{
    "langchain.run_id": "...",
    "llm.model": "gpt-3.5-turbo",
    "llm.provider": "openai",
    "llm.operation": "completion",
    "llm.input.prompts": ["Hello"],
    "llm.input.prompt_count": 1,
}
```

### After Implementation

```python
# File: my_app.py, Line: 47
def my_function():
    llm = ChatOpenAI(callbacks=[callback_handler])
    response = llm.invoke("Hello")  # This line will be tracked
```

**Span Attributes** (with call site tracking):
```python
{
    "langchain.run_id": "...",
    "llm.model": "gpt-3.5-turbo",
    "llm.provider": "openai",
    "llm.operation": "completion",
    "llm.input.prompts": ["Hello"],
    "llm.input.prompt_count": 1,
    "call_site.file": "/path/to/my_app.py",
    "call_site.line": 47,
    "call_site.function": "my_function",
    "call_site.module": "my_app",
    "call_site.code_context": "response = llm.invoke(\"Hello\")"
}
```

## Benefits

1. **Debugging**: 
   - Quickly identify where LLM/tool calls originate
   - Trace issues back to source code
   - Understand call patterns

2. **Performance Analysis**:
   - Identify hotspots in code
   - Find frequently called locations
   - Optimize based on call site data

3. **Usage Patterns**:
   - Understand how integration is used
   - Track which functions make most calls
   - Analyze call distribution

4. **Error Tracking**:
   - Correlate errors with call sites
   - Identify problematic code locations
   - Improve error messages

## Considerations

### Performance Impact

- **Stack Inspection Overhead**: Walking the stack has a small performance cost
- **Mitigation**: 
  - Only inspect when needed (not in hot paths)
  - Consider making it optional via config
  - Limit stack depth if needed

### Accuracy

- **Complex Scenarios**: 
  - Decorators may add extra frames
  - Async code may have different stack structure
  - Nested calls may require careful frame selection

- **Mitigation**:
  - Test with various scenarios
  - Provide fallback mechanisms
  - Allow manual override via metadata

### Privacy/Security

- **File Paths**: May contain sensitive information
- **Mitigation**:
  - Consider sanitizing paths
  - Option to use relative paths
  - Allow disabling in sensitive environments

## Testing Strategy

### Unit Tests

1. **Test `extract_call_site_info()`**:
   - Test with various stack configurations
   - Test with LangChain frames in stack
   - Test with Noveum Trace frames in stack
   - Test with no user code frames
   - Test with nested calls

2. **Test Frame Filtering**:
   - Verify LangChain frames are skipped
   - Verify Noveum Trace frames are skipped
   - Verify standard library frames are skipped
   - Verify user code frames are captured

### Integration Tests

1. **Test LLM Call Site Tracking**:
   - Verify call site info appears in LLM spans
   - Test with different LLM invocation patterns
   - Test with chains containing LLMs

2. **Test Tool Call Site Tracking**:
   - Verify call site info appears in tool spans
   - Test with different tool invocation patterns
   - Test with agents using tools

3. **Test Edge Cases**:
   - Decorated functions
   - Async code
   - Nested calls
   - Error scenarios

### Example Test Cases

```python
def test_extract_call_site_info_with_user_code():
    """Test that user code frame is correctly identified."""
    def user_function():
        return extract_call_site_info()
    
    result = user_function()
    assert result is not None
    assert "call_site.file" in result
    assert "call_site.line" in result
    assert "call_site.function" in result
    assert result["call_site.function"] == "user_function"

def test_extract_call_site_info_skips_langchain():
    """Test that LangChain frames are skipped."""
    # Simulate LangChain call stack
    # Verify user code frame is found, not LangChain frame

def test_on_llm_start_includes_call_site():
    """Test that on_llm_start includes call site info."""
    handler = NoveumTraceCallbackHandler()
    # ... setup ...
    handler.on_llm_start(...)
    # Verify span has call_site attributes

def test_on_tool_start_includes_call_site():
    """Test that on_tool_start includes call site info."""
    handler = NoveumTraceCallbackHandler()
    # ... setup ...
    handler.on_tool_start(...)
    # Verify span has call_site attributes
```

## Implementation Checklist

- [ ] Create `extract_call_site_info()` function in `langchain_utils.py`
- [ ] Add unit tests for `extract_call_site_info()`
- [ ] Integrate into `on_llm_start()` method
- [ ] Integrate into `on_tool_start()` method
- [ ] Integrate into `on_chain_start()` method (optional)
- [ ] Integrate into `on_agent_start()` method (optional)
- [ ] Integrate into `on_retriever_start()` method (optional)
- [ ] Add integration tests for LLM call site tracking
- [ ] Add integration tests for tool call site tracking
- [ ] Test with decorators
- [ ] Test with async code
- [ ] Test with nested calls
- [ ] Test error scenarios
- [ ] Update documentation
- [ ] Add example usage in examples directory

## Future Enhancements

1. **Call Site Aggregation**: Track statistics per call site
2. **Performance Metrics**: Track timing per call site
3. **Error Correlation**: Correlate errors with call sites
4. **Visualization**: Show call sites in dashboard
5. **Filtering**: Allow filtering traces by call site

## Notes

- This feature should be **non-breaking** - existing functionality must continue to work
- Call site extraction should **fail gracefully** - if it fails, tracing should continue
- Performance impact should be **minimal** - consider making it optional if needed
- The feature should work with **all LangChain integration patterns** (LLMs, tools, chains, agents, retrievers)

