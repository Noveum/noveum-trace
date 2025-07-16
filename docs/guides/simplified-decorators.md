# Simplified Decorators Guide

The Noveum Trace SDK uses a **unified decorator approach** that simplifies tracing while maintaining full functionality. Instead of multiple decorators like competitors, we provide **one `@trace` decorator** with parameters for specialization.

## 🎯 Design Philosophy

After analyzing competitor SDKs (DeepEval, Phoenix, Braintrust), we found they all provide multiple decorators (`@observe`, `@traced`, `@trace`) that essentially do the same thing - wrap functions in spans and capture input/output. The only differences are:

1. **Naming conventions**
2. **Backend destinations**
3. **Default attributes**

Our approach: **One decorator to rule them all** with parameter-based specialization.

## 🚀 Basic Usage

### Simple Function Tracing

```python
from noveum_trace import trace

@trace
def process_data(data):
    return f"Processed: {data}"

@trace(name="custom-operation")
def custom_function():
    return "result"
```

### LLM-Specific Tracing

```python
@trace(type="llm", model="gpt-4", operation="chat")
def chat_completion(messages):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response

@trace(type="llm", model="claude-3", operation="completion")
def text_completion(prompt):
    response = anthropic.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": prompt}]
    )
    return response
```

### Component/Agent Tracing

```python
@trace(type="component", agent="data-processor")
def agent_task(task):
    return f"Agent completed: {task}"

@trace(type="component", component="retrieval-system")
def search_documents(query):
    return search_results
```

### Tool Call Tracing

```python
@trace(type="tool", tool_name="web_search")
def search_web(query):
    return search_results

@trace(type="tool", tool_name="calculator")
def calculate(expression):
    return result
```

## 🔄 Backward Compatibility

For users familiar with competitor patterns, we provide aliases:

```python
from noveum_trace import observe, llm_trace

# DeepEval-style (alias for @trace)
@observe
def component_function():
    return "result"

# Custom LLM tracing (alias for @trace with LLM params)
@llm_trace(model="gpt-4")
def llm_function():
    return "response"
```

## ⚙️ Decorator Parameters

### Core Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | `str` | Custom span name | `@trace(name="data-processing")` |
| `type` | `str` | Operation type | `@trace(type="llm")` |
| `tags` | `List[str]` | Custom tags | `@trace(tags=["critical", "user-facing"])` |
| `metadata` | `Dict` | Custom metadata | `@trace(metadata={"version": "1.0"})` |

### LLM-Specific Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `model` | `str` | Model name | `@trace(model="gpt-4")` |
| `operation` | `str` | LLM operation | `@trace(operation="chat")` |
| `provider` | `str` | AI provider | `@trace(provider="openai")` |

### Agent-Specific Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `agent` | `str` | Agent name | `@trace(agent="coordinator")` |
| `component` | `str` | Component name | `@trace(component="retrieval")` |
| `capability` | `str` | Agent capability | `@trace(capability="reasoning")` |

### Tool-Specific Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `tool_name` | `str` | Tool/function name | `@trace(tool_name="web_search")` |
| `tool_type` | `str` | Tool category | `@trace(tool_type="external_api")` |

## 🔧 Advanced Usage

### Dynamic Span Updates

```python
from noveum_trace import trace, update_current_span

@trace
def long_running_task():
    update_current_span(
        metadata={"step": "initialization"},
        progress=10
    )

    # Do initialization
    initialize()

    update_current_span(
        metadata={"step": "processing"},
        progress=50,
        attributes={"items_processed": 100}
    )

    # Process data
    process_data()

    update_current_span(
        metadata={"step": "completion"},
        progress=100,
        status="success"
    )

    return "completed"
```

### Conditional Tracing

```python
@trace(enabled=lambda: os.getenv("ENABLE_TRACING") == "true")
def conditional_function():
    return "result"

@trace(sample_rate=0.1)  # Trace 10% of calls
def high_volume_function():
    return "result"
```

### Error Handling

```python
@trace
def risky_function():
    try:
        # Risky operation
        result = perform_operation()
        return result
    except Exception as e:
        # Exception is automatically recorded in span
        raise
```

## 🏆 Comparison with Competitors

### DeepEval Pattern

```python
# DeepEval approach
from deepeval.tracing import observe

@observe
def component_function():
    return "result"
```

```python
# Noveum equivalent (cleaner)
from noveum_trace import trace

@trace(type="component")
def component_function():
    return "result"
```

### Phoenix Pattern

```python
# Phoenix approach
from phoenix.trace import trace as phoenix_trace

@phoenix_trace
def traced_function():
    return "result"
```

```python
# Noveum equivalent (same simplicity)
from noveum_trace import trace

@trace
def traced_function():
    return "result"
```

### Braintrust Pattern

```python
# Braintrust approach
from braintrust import traced

@traced
def traced_function():
    return "result"
```

```python
# Noveum equivalent (same simplicity + more features)
from noveum_trace import trace

@trace
def traced_function():
    return "result"
```

## 🎨 Best Practices

### 1. Use Descriptive Names

```python
# Good
@trace(name="user-authentication")
def authenticate_user(credentials):
    return auth_result

# Better
@trace(name="authenticate-user", type="security", tags=["auth", "critical"])
def authenticate_user(credentials):
    return auth_result
```

### 2. Leverage Type-Specific Parameters

```python
# Good
@trace
def call_llm():
    return response

# Better
@trace(type="llm", model="gpt-4", operation="chat")
def call_llm():
    return response
```

### 3. Use Metadata for Context

```python
@trace(
    type="component",
    agent="data-processor",
    metadata={
        "version": "2.1.0",
        "environment": "production",
        "feature_flags": ["new_algorithm"]
    }
)
def process_data(data):
    return processed_data
```

### 4. Combine with Agent Context

```python
from noveum_trace import Agent, AgentContext, trace

agent = Agent(AgentConfig(name="coordinator", agent_type="orchestrator"))

with AgentContext(agent):
    @trace(type="planning")
    def create_plan():
        return plan

    @trace(type="execution")
    def execute_plan(plan):
        return results
```

## 🔍 Under the Hood

The `@trace` decorator:

1. **Creates a span** with the function name (or custom name)
2. **Captures input arguments** (if enabled)
3. **Records execution time** automatically
4. **Captures return value** (if enabled)
5. **Handles exceptions** and marks span as error
6. **Adds type-specific attributes** based on parameters
7. **Integrates with agent context** if available
8. **Exports to configured sinks** (file, console, Noveum.ai, etc.)

## 🚀 Migration Guide

### From DeepEval

```python
# Before (DeepEval)
from deepeval.tracing import observe

@observe
def my_function():
    return "result"
```

```python
# After (Noveum)
from noveum_trace import trace  # or import observe for compatibility

@trace  # or @observe
def my_function():
    return "result"
```

### From Phoenix

```python
# Before (Phoenix)
from phoenix.trace import trace as phoenix_trace

@phoenix_trace
def my_function():
    return "result"
```

```python
# After (Noveum)
from noveum_trace import trace

@trace
def my_function():
    return "result"
```

### From Braintrust

```python
# Before (Braintrust)
from braintrust import traced

@traced
def my_function():
    return "result"
```

```python
# After (Noveum)
from noveum_trace import trace

@trace
def my_function():
    return "result"
```

## 🎯 Why This Approach Wins

1. **Simplicity**: One decorator instead of many
2. **Flexibility**: Parameters for specialization
3. **Consistency**: Same interface for all operation types
4. **Extensibility**: Easy to add new parameters
5. **Backward Compatibility**: Aliases for familiar patterns
6. **Better DX**: Less cognitive load, more productivity

The unified decorator approach reduces complexity while maintaining full functionality - exactly what developers need for production AI applications.
