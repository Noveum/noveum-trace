# Quick Start Guide

Get up and running with Noveum Trace SDK in minutes.

## 1. Simple Initialization

The easiest way to start tracing:

```python
import noveum_trace

# Initialize with default settings
noveum_trace.init(service_name="my-app")

# Your application code here
with noveum_trace.get_tracer().start_span("my-operation") as span:
    span.set_attribute("operation.type", "data_processing")
    # Do some work
    result = process_data()
    span.set_attribute("operation.result_count", len(result))

# Shutdown when done
noveum_trace.shutdown()
```

## 2. File-Based Logging

Save traces to local files:

```python
import noveum_trace

# Initialize with file logging
noveum_trace.init(
    service_name="my-app",
    log_directory="./traces"
)

# Your traces will be saved to ./traces/
```

## 3. Auto-Instrumentation

Automatically trace LLM calls:

```python
import noveum_trace
from noveum_trace.instrumentation import openai, anthropic

# Initialize tracing
noveum_trace.init(service_name="llm-app", log_directory="./traces")

# Enable auto-instrumentation
openai.instrument_openai()
anthropic.instrument_anthropic()

# Now all LLM calls are automatically traced!
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 4. Manual Tracing

For custom operations:

```python
import noveum_trace

noveum_trace.init(service_name="custom-app")
tracer = noveum_trace.get_tracer()

# Create nested spans
with tracer.start_span("main_operation") as main_span:
    main_span.set_attribute("user.id", "user123")
    
    # Nested operation
    with tracer.start_span("sub_operation") as sub_span:
        sub_span.set_attribute("operation.type", "llm_call")
        sub_span.set_attribute("gen_ai.system", "openai")
        sub_span.set_attribute("gen_ai.request.model", "gpt-4")
        
        # Add events
        sub_span.add_event("gen_ai.content.prompt", {
            "gen_ai.prompt": "What is AI?"
        })
        
        # Simulate processing
        import time
        time.sleep(0.1)
        
        sub_span.add_event("gen_ai.content.completion", {
            "gen_ai.completion": "AI is artificial intelligence..."
        })
        
        # Set usage metrics
        sub_span.set_attribute("gen_ai.usage.input_tokens", 10)
        sub_span.set_attribute("gen_ai.usage.output_tokens", 25)
```

## 5. Using Decorators

Simplify tracing with decorators:

```python
import noveum_trace
from noveum_trace.instrumentation import trace_function

noveum_trace.init(service_name="decorator-app")

@trace_function(name="data_processor")
def process_data(data):
    # This function is automatically traced
    return [item.upper() for item in data]

@trace_function(name="llm_call", operation_type="llm")
def call_llm(prompt):
    # This function is traced as an LLM operation
    return f"Response to: {prompt}"

# Use the functions normally
result = process_data(["hello", "world"])
response = call_llm("What is Python?")
```

## 6. Configuration Options

Customize tracing behavior:

```python
import noveum_trace

noveum_trace.init(
    service_name="configured-app",
    environment="production",
    log_directory="./production_traces",
    capture_content=True,          # Capture LLM inputs/outputs
    batch_size=50,                 # Batch size for performance
    batch_timeout_ms=500,          # Batch timeout
    enable_console_output=False,   # Disable console logging
    project_id="my-project",       # For Noveum.ai integration
    api_key="your-api-key"         # For Noveum.ai integration
)
```

## 7. Multiple Sinks

Send traces to multiple destinations:

```python
import noveum_trace
from noveum_trace.sinks import FileSink, ConsoleSink

# Create custom sinks
file_sink = FileSink({"directory": "./traces", "name": "file-sink"})
console_sink = ConsoleSink({"name": "console-sink"})

# Initialize with multiple sinks
noveum_trace.init(
    service_name="multi-sink-app",
    sinks=[file_sink, console_sink]
)
```

## 8. Error Handling

Trace errors and exceptions:

```python
import noveum_trace

noveum_trace.init(service_name="error-app")
tracer = noveum_trace.get_tracer()

with tracer.start_span("risky_operation") as span:
    try:
        # Some operation that might fail
        result = 1 / 0
    except Exception as e:
        # Record the exception
        span.record_exception(e)
        span.set_status("error", str(e))
        # Handle the error
        print(f"Error occurred: {e}")
```

## Next Steps

- Explore [Configuration Guide](../guides/configuration.md)
- Learn about [LLM Tracing](../guides/llm-tracing.md)
- Check out [Examples](../examples/)
- Read [API Reference](../api-reference/)

