# Noveum Trace + LangChain Integration Specification

## Overview
This document outlines the implementation of LangChain callback handlers for Noveum Trace SDK, enabling automatic tracing of LangChain workflows without code modifications.

## Architecture

### Integration Approach
- **Callback Handler**: Implements LangChain's `BaseCallbackHandler` interface
- **Trace Granularity**: Individual traces for top-level operations, spans for nested operations
- **Transport**: Uses existing Noveum Trace transport configuration and batching

### Trace Hierarchy
```
Standalone Operations:
├── Trace: llm.{model_name}
├── Trace: chain.{chain_name}  
├── Trace: agent.{agent_name}
└── Trace: retrieval.{retriever_name}

Nested Operations:
Trace: chain.{parent_chain_name}
├── Span: llm.{model_name}
├── Span: tool.{tool_name}
├── Span: retrieval.{retriever_name}
└── Span: chain.{nested_chain_name}
```

## Implementation Details

### 1. Core Callback Handler

**File**: `src/noveum_trace/integrations/langchain.py`

```python
class NoveumTraceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for Noveum Trace integration."""
    
    def __init__(self):
        super().__init__()
        self._trace_stack = []  # Active traces
        self._span_stack = []   # Active spans
        self._current_trace = None  # Current trace context
    
    def _should_create_trace(self, event_type: str, serialized: dict) -> bool:
        """Determine if event should create new trace or just span."""
        if event_type in ['chain_start', 'agent_start']:
            return True  # Always create trace for chains/agents
        
        if event_type in ['llm_start', 'retriever_start']:
            return len(self._trace_stack) == 0  # Only if not nested
        
        return False
    
    def _get_operation_name(self, event_type: str, serialized: dict) -> str:
        """Generate standardized operation names."""
        name = serialized.get('name', 'unknown')
        
        if event_type == 'llm_start':
            return f"llm.{name}"
        elif event_type == 'chain_start':
            return f"chain.{name}"
        elif event_type == 'agent_start':
            return f"agent.{name}"
        elif event_type == 'retriever_start':
            return f"retrieval.{name}"
        elif event_type == 'tool_start':
            return f"tool.{name}"
        
        return f"{event_type}.{name}"
```

### 2. Event Handlers

#### LLM Events
```python
def on_llm_start(self, serialized: dict, prompts: list, run_id: UUID, **kwargs):
    """Handle LLM start event."""
    client = get_client()
    operation_name = self._get_operation_name('llm_start', serialized)
    
    if self._should_create_trace('llm_start', serialized):
        # Standalone LLM call - create new trace
        self._current_trace = client.start_trace(operation_name)
        self._trace_stack.append(self._current_trace)
    
    # Create span (either in new trace or existing trace)
    span = client.start_span(
        name=operation_name,
        attributes={
            'langchain.run_id': str(run_id),
            'llm.model': serialized.get('name', 'unknown'),
            'llm.provider': serialized.get('id', ['unknown'])[-1],
            'llm.prompts': prompts,
            **kwargs
        }
    )
    self._span_stack.append(span)

def on_llm_end(self, response: LLMResult, run_id: UUID, **kwargs):
    """Handle LLM end event."""
    if self._span_stack:
        span = self._span_stack.pop()
        
        # Add response data
        span.set_attributes({
            'llm.response': str(response.generations),
            'llm.usage': response.llm_output.get('token_usage', {}) if response.llm_output else {},
            'llm.finish_reason': response.llm_output.get('finish_reason') if response.llm_output else None
        })
        
        span.set_status(SpanStatus.OK)
        client = get_client()
        client.finish_span(span)
        
        # Finish trace if this was a standalone LLM call
        if self._current_trace and len(self._span_stack) == 0:
            client.finish_trace(self._current_trace)
            self._trace_stack.pop()
            self._current_trace = None

def on_llm_error(self, error: Exception, run_id: UUID, **kwargs):
    """Handle LLM error event."""
    if self._span_stack:
        span = self._span_stack.pop()
        span.record_exception(error)
        span.set_status(SpanStatus.ERROR, str(error))
        client = get_client()
        client.finish_span(span)
        
        # Finish trace if this was a standalone LLM call
        if self._current_trace and len(self._span_stack) == 0:
            client.finish_trace(self._current_trace)
            self._trace_stack.pop()
            self._current_trace = None
```

#### Chain Events
```python
def on_chain_start(self, serialized: dict, inputs: dict, run_id: UUID, **kwargs):
    """Handle chain start event."""
    client = get_client()
    operation_name = self._get_operation_name('chain_start', serialized)
    
    if self._should_create_trace('chain_start', serialized):
        # Create new trace for chain
        self._current_trace = client.start_trace(operation_name)
        self._trace_stack.append(self._current_trace)
    
    # Create span for chain
    span = client.start_span(
        name=operation_name,
        attributes={
            'langchain.run_id': str(run_id),
            'chain.name': serialized.get('name', 'unknown'),
            'chain.inputs': inputs,
            **kwargs
        }
    )
    self._span_stack.append(span)

def on_chain_end(self, outputs: dict, run_id: UUID, **kwargs):
    """Handle chain end event."""
    if self._span_stack:
        span = self._span_stack.pop()
        
        span.set_attributes({
            'chain.outputs': outputs
        })
        
        span.set_status(SpanStatus.OK)
        client = get_client()
        client.finish_span(span)
        
        # Finish trace if this was the top-level chain
        if self._current_trace and len(self._span_stack) == 0:
            client.finish_trace(self._current_trace)
            self._trace_stack.pop()
            self._current_trace = None

def on_chain_error(self, error: Exception, run_id: UUID, **kwargs):
    """Handle chain error event."""
    if self._span_stack:
        span = self._span_stack.pop()
        span.record_exception(error)
        span.set_status(SpanStatus.ERROR, str(error))
        client = get_client()
        client.finish_span(span)
        
        # Finish trace if this was the top-level chain
        if self._current_trace and len(self._span_stack) == 0:
            client.finish_trace(self._current_trace)
            self._trace_stack.pop()
            self._current_trace = None
```

#### Tool Events
```python
def on_tool_start(self, serialized: dict, input_str: str, run_id: UUID, **kwargs):
    """Handle tool start event."""
    client = get_client()
    operation_name = self._get_operation_name('tool_start', serialized)
    
    # Tools always create spans (never standalone traces)
    span = client.start_span(
        name=operation_name,
        attributes={
            'langchain.run_id': str(run_id),
            'tool.name': serialized.get('name', 'unknown'),
            'tool.input': input_str,
            **kwargs
        }
    )
    self._span_stack.append(span)

def on_tool_end(self, output: str, run_id: UUID, **kwargs):
    """Handle tool end event."""
    if self._span_stack:
        span = self._span_stack.pop()
        
        span.set_attributes({
            'tool.output': output
        })
        
        span.set_status(SpanStatus.OK)
        client = get_client()
        client.finish_span(span)

def on_tool_error(self, error: Exception, run_id: UUID, **kwargs):
    """Handle tool error event."""
    if self._span_stack:
        span = self._span_stack.pop()
        span.record_exception(error)
        span.set_status(SpanStatus.ERROR, str(error))
        client = get_client()
        client.finish_span(span)
```

#### Agent Events
```python
def on_agent_action(self, action: AgentAction, run_id: UUID, **kwargs):
    """Handle agent action event."""
    if self._span_stack:
        span = self._span_stack[-1]  # Add to current span
        span.add_event('agent_action', {
            'action.tool': action.tool,
            'action.tool_input': action.tool_input,
            'action.log': action.log
        })

def on_agent_finish(self, finish: AgentFinish, run_id: UUID, **kwargs):
    """Handle agent finish event."""
    if self._span_stack:
        span = self._span_stack[-1]  # Add to current span
        span.add_event('agent_finish', {
            'finish.return_values': finish.return_values,
            'finish.log': finish.log
        })
```

#### Retrieval Events
```python
def on_retriever_start(self, serialized: dict, query: str, run_id: UUID, **kwargs):
    """Handle retriever start event."""
    client = get_client()
    operation_name = self._get_operation_name('retriever_start', serialized)
    
    if self._should_create_trace('retriever_start', serialized):
        # Standalone retrieval - create new trace
        self._current_trace = client.start_trace(operation_name)
        self._trace_stack.append(self._current_trace)
    
    # Create span
    span = client.start_span(
        name=operation_name,
        attributes={
            'langchain.run_id': str(run_id),
            'retrieval.query': query,
            'retrieval.source': serialized.get('name', 'unknown'),
            **kwargs
        }
    )
    self._span_stack.append(span)

def on_retriever_end(self, documents: list, run_id: UUID, **kwargs):
    """Handle retriever end event."""
    if self._span_stack:
        span = self._span_stack.pop()
        
        span.set_attributes({
            'retrieval.documents_count': len(documents),
            'retrieval.documents': [doc.page_content[:200] + "..." for doc in documents]
        })
        
        span.set_status(SpanStatus.OK)
        client = get_client()
        client.finish_span(span)
        
        # Finish trace if this was a standalone retrieval
        if self._current_trace and len(self._span_stack) == 0:
            client.finish_trace(self._current_trace)
            self._trace_stack.pop()
            self._current_trace = None

def on_retriever_error(self, error: Exception, run_id: UUID, **kwargs):
    """Handle retriever error event."""
    if self._span_stack:
        span = self._span_stack.pop()
        span.record_exception(error)
        span.set_status(SpanStatus.ERROR, str(error))
        client = get_client()
        client.finish_span(span)
        
        # Finish trace if this was a standalone retrieval
        if self._current_trace and len(self._span_stack) == 0:
            client.finish_trace(self._current_trace)
            self._trace_stack.pop()
            self._current_trace = None
```

### 3. Integration Points

#### Main Package Integration
**File**: `src/noveum_trace/__init__.py`
```python
# Add to exports
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

__all__ = [
    # ... existing exports
    "NoveumTraceCallbackHandler",
]
```

#### Usage Examples
**File**: `docs/examples/langchain_callback_example.py`
```python
import noveum_trace
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Initialize Noveum Trace
noveum_trace.init(api_key="your-key", project="langchain-app")

# Create callback handler
callback_handler = NoveumTraceCallbackHandler()

# Use with LangChain
llm = OpenAI(callbacks=[callback_handler])
chain = LLMChain(llm=llm, prompt=prompt)

# Execute
result = chain.run("Hello world")
```

### 4. Error Handling Strategy

- **Error Spans**: Create spans for all error events
- **Error Propagation**: Mark parent spans as failed when child operations fail
- **Stack Cleanup**: Maintain stack integrity during error conditions
- **Trace Completion**: Ensure traces are properly finished even on errors

### 5. Transport Integration

- **Use Existing Transport**: Leverage current Noveum Trace transport configuration
- **Batching**: Utilize existing batching mechanisms
- **Configuration**: Inherit transport settings from main Noveum Trace config

### 6. File Structure

```
src/noveum_trace/
├── integrations/
│   ├── __init__.py
│   └── langchain.py          # Main callback handler
├── core/                     # Existing core
├── decorators/               # Existing decorators
├── context_managers.py       # Existing context managers
└── __init__.py              # Updated exports
```

### 7. Dependencies

- **LangChain**: Add as optional dependency
- **Version**: Target LangChain >= 0.1.0
- **Installation**: `pip install noveum-trace[langchain]`

## Implementation Phases

### Phase 1: Core Handler
- Implement basic callback handler
- Add LLM, Chain, Tool event support
- Basic error handling

### Phase 2: Advanced Features
- Add Agent and Retrieval support
- Improve error handling and stack management
- Add comprehensive testing

### Phase 3: Integration
- Package integration
- Documentation
- Examples and tutorials

## Success Criteria

- [ ] All LangChain events properly traced
- [ ] Correct trace/span hierarchy maintained
- [ ] Error handling robust and complete
- [ ] Integration seamless with existing Noveum Trace features
- [ ] Performance impact minimal
- [ ] Documentation comprehensive

---

This specification provides a complete roadmap for implementing LangChain integration with Noveum Trace while maintaining the existing architecture and design principles.
