# Tracing Setup Analysis

Based on the trace data provided, here's how your customer has likely implemented tracing:

## Key Findings

### 1. **SDK Version Usage**
The traces show multiple SDK versions being used:
- **Version 1.0.0** (most traces) - 8 traces
- **Version 0.4.0** - 3 traces  
- **Version 0.3.11** - 3 traces

This suggests the customer has been upgrading the SDK over time, or different parts of their codebase are using different versions.

### 2. **SDK Field Presence**
**All traces in the provided file have the `sdk` field.** The SDK field is added by the `HttpTransport._format_trace_for_export()` method (lines 442-446 in `src/noveum_trace/transport/http_transport.py`).

If you're seeing traces without the SDK field, it could mean:
- They're using `trace.to_dict()` directly without going through the transport layer
- They have a custom export method
- They're using an older version of the SDK that didn't add the SDK field
- The traces were manually created/modified

### 3. **Implementation Pattern**

Based on the trace structure, the customer is using:

#### **Auto-Trace Pattern (Decorator-based with Custom Wrapper)**
All trace names follow the pattern `auto_trace_{function_name}`, which indicates they're using **decorators** (not context managers). Here's the evidence:

**Evidence FOR Decorators:**
1. **Span names match function names exactly**: `generate_report_content`, `fetch_data`, `summarize_data` - these are the actual function names, which decorators automatically capture
2. **Auto-trace creation**: Both decorators (line 89 in `decorators/base.py`) and context managers (line 50 in `context_managers.py`) create `auto_trace_{name}` patterns
3. **Consistent attribute pattern**: All 41 spans have the same attribute structure, suggesting automated capture

**Evidence AGAINST Context Managers:**
1. **No manual attribute setting needed**: Context managers require manually setting `input.args`, `input.kwargs`, etc. in every function
2. **Function name capture**: The span names are the exact function names, which decorators automatically extract via `func.__name__`

**However**, the attributes show a **custom transformation**:
- Decorators would set: `function.name`, `function.module`, `function.args.{param_name}`, `function.result`
- But traces show: `input.args`, `input.kwargs`, `span.type`, `output.result`

This means they have a **custom decorator wrapper** that:
1. Uses the noveum decorator internally
2. Transforms/removes the default decorator attributes
3. Adds custom `input.args`, `input.kwargs`, `span.type`, `output.result` format

#### **Span Attributes Pattern**
The spans show a consistent pattern:
- `input.args` - Function positional arguments (serialized as string)
- `input.kwargs` - Function keyword arguments (serialized as string)  
- `span.type` - Type of span: "operation", "llm", etc.
- `output.result` - Function return value (serialized)

**Note:** The current SDK codebase uses different attribute naming:
- `function.args.{param_name}` (in base decorator)
- `tool.input.{param_name}` (in tool decorator)
- `agent.input.{param_name}` (in agent decorator)

But the traces show `input.args` and `input.kwargs`. This suggests:
1. The customer might be using a custom wrapper/decorator
2. They might be using a different version of the SDK
3. There's post-processing that transforms the attributes

### 4. **Initialization**

Based on the trace attributes, they initialized the SDK like this:

```python
import noveum_trace

noveum_trace.init(
    project="wealthink-research-automation-v0",
    environment="testing",  # or "testing_1" in some traces
    api_key="<their-api-key>"
)
```

### 5. **Function Tracing (Decorator Pattern with Custom Wrapper)**

The customer is likely using **decorators with a custom wrapper** that transforms the attributes. Here's how:

```python
from noveum_trace.decorators.base import trace
from functools import wraps
import inspect

def custom_trace(span_type="operation"):
    """
    Custom decorator wrapper that:
    1. Uses noveum's @trace decorator
    2. Transforms attributes to custom format
    3. Removes default decorator attributes
    """
    def decorator(func):
        # Use noveum's trace decorator
        @trace(capture_args=True, capture_result=True)
        @wraps(func)
        def wrapper(*args, **kwargs):
            from noveum_trace.core.context import get_current_span
            
            span = get_current_span()
            
            if span:
                # Remove default decorator attributes
                # (or they're transformed during export)
                
                # Set custom format attributes
                span.set_attribute("input.args", str(args))
                span.set_attribute("input.kwargs", str(kwargs))
                span.set_attribute("span.type", span_type)
            
            # Execute function (decorator handles result capture)
            result = func(*args, **kwargs)
            
            if span:
                # Transform result attribute
                span.set_attribute("output.result", str(result))
                span.set_attribute("status", "success")
            
            return result
        
        return wrapper
    return decorator

# Usage:
@custom_trace(span_type="operation")
def generate_report_content(research_obj):
    # ... function implementation
    return result

@custom_trace(span_type="llm")
def summarize_data(module_obj):
    # ... LLM call implementation
    return result

@custom_trace(span_type="operation")
def fetch_data(module_obj):
    # ... data fetching implementation
    return result
```

**OR** they might be post-processing the attributes after the decorator runs:

```python
def transform_span_attributes(span):
    """Transform decorator attributes to custom format."""
    # Get all attributes
    attrs = span.attributes.copy()
    
    # Collect function args
    args_dict = {k: v for k, v in attrs.items() if k.startswith("function.args.")}
    if args_dict:
        # Reconstruct args tuple (simplified)
        args_str = str(tuple(args_dict.values()))
        span.set_attribute("input.args", args_str)
        
        # Remove original attributes
        for key in list(attrs.keys()):
            if key.startswith("function.args.") or key.startswith("function."):
                span.attributes.pop(key, None)
    
    # Transform result
    if "function.result" in attrs:
        span.set_attribute("output.result", attrs["function.result"])
        span.attributes.pop("function.result", None)
    
    # Set span type
    span.set_attribute("span.type", attrs.get("function.type", "operation"))
```

### 6. **Custom Decorator Wrapper (Confirmed)**

The evidence strongly suggests they have a **custom decorator wrapper** that:
1. Uses noveum's `@trace` decorator internally
2. Transforms the default decorator attributes (`function.*`) to custom format (`input.*`, `output.*`)
3. Adds `span.type` attribute
4. Handles status setting

**Key Evidence:**
- **0 decorator-specific attributes found** in traces (no `function.name`, `function.module`, `function.args.{param}`)
- **41 spans** have consistent `input.args`, `input.kwargs`, `span.type`, `output.result` pattern
- **Span names match function names exactly** - decorators automatically capture `func.__name__`

**Most Likely Implementation:**

```python
# common/tracing/noveum.py
from noveum_trace.decorators.base import trace
from noveum_trace.core.context import get_current_span
from functools import wraps
import inspect

def traced(span_type="operation"):
    """
    Custom decorator that wraps noveum's @trace and transforms attributes.
    """
    def decorator(func):
        # Use noveum's trace decorator with custom handling
        @trace(capture_args=True, capture_result=True)
        @wraps(func)
        def wrapper(*args, **kwargs):
            span = get_current_span()
            
            if span:
                # Transform attributes after decorator sets them
                attrs = span.attributes.copy()
                
                # Collect function.args.* into input.args
                func_args = {}
                for key in list(attrs.keys()):
                    if key.startswith("function.args."):
                        param_name = key.replace("function.args.", "")
                        func_args[param_name] = attrs[key]
                        span.attributes.pop(key)
                
                # Set custom format
                span.set_attribute("input.args", str(args))
                span.set_attribute("input.kwargs", str(kwargs))
                span.set_attribute("span.type", span_type)
                
                # Remove other function.* attributes
                for key in list(attrs.keys()):
                    if key.startswith("function.") and key != "function.result":
                        span.attributes.pop(key, None)
            
            # Execute (decorator handles result)
            result = func(*args, **kwargs)
            
            if span:
                # Transform function.result to output.result
                if "function.result" in span.attributes:
                    result_value = span.attributes.pop("function.result")
                    span.set_attribute("output.result", str(result_value))
                else:
                    span.set_attribute("output.result", str(result))
                
                span.set_attribute("status", "success")
            
            return result
        
        return wrapper
    return decorator

# Usage in their code:
@traced(span_type="operation")
def generate_report_content(research_obj):
    # ... implementation
    return result

@traced(span_type="llm")
def summarize_data(module_obj):
    # ... LLM implementation
    return result
```

### 7. **Error Handling**

The traces show error capture is working:
- Error traces have `status: "error"`
- Error details in `error.type`, `error.message`, `error.stacktrace`
- Exception events are recorded

### 8. **Project Structure**

Based on the stack traces in error spans, their code structure is:
```
/home/jay/work/wealthink-research-automation/
├── common/
│   └── tracing/
│       ├── noveum.py
│       └── langfuse.py
└── research_src/
    └── module/
        └── main.py
```

They have a custom tracing module at `common/tracing/noveum.py` which likely contains their wrapper functions.

## Recommendations

1. **Check for custom decorator wrapper**: Look for files like `common/tracing/noveum.py` that likely contains a custom decorator wrapper that transforms noveum's default attributes
2. **Version consistency**: They're using multiple SDK versions (1.0.0, 0.4.0, 0.3.11) - recommend standardizing on one
3. **SDK field**: If some traces are missing the SDK field, they might be using `trace.to_dict()` directly - recommend always using the transport layer for export
4. **Attribute transformation**: Their custom wrapper transforms decorator attributes - verify this transformation is intentional and not causing data loss
5. **Function metadata**: The wrapper removes `function.name`, `function.module`, `function.qualname` - consider if this metadata should be preserved

