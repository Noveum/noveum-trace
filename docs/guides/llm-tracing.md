# LLM Tracing Guide

This guide covers how to trace Large Language Model (LLM) operations using the Noveum Trace SDK.

## Overview

The Noveum Trace SDK provides comprehensive tracing for LLM operations, following OpenTelemetry semantic conventions for generative AI. This ensures compatibility with industry standards and observability tools.

## Auto-Instrumentation

### OpenAI Integration

Automatically trace OpenAI API calls:

```python
import noveum_trace
from noveum_trace.instrumentation import openai

# Initialize tracing
noveum_trace.init(service_name="openai-app", log_directory="./traces")

# Enable auto-instrumentation
openai.instrument_openai()

# Now all OpenAI calls are automatically traced
import openai
client = openai.OpenAI(api_key="your-api-key")

# This call will be automatically traced
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,
    max_tokens=150
)
```

### Anthropic Integration

Automatically trace Anthropic API calls:

```python
import noveum_trace
from noveum_trace.instrumentation import anthropic

# Initialize tracing
noveum_trace.init(service_name="anthropic-app", log_directory="./traces")

# Enable auto-instrumentation
anthropic.instrument_anthropic()

# Now all Anthropic calls are automatically traced
import anthropic
client = anthropic.Anthropic(api_key="your-api-key")

# This call will be automatically traced
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=150,
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)
```

## Manual LLM Tracing

For custom LLM providers or more control:

```python
import noveum_trace
import time

noveum_trace.init(service_name="custom-llm-app")
tracer = noveum_trace.get_tracer()

def call_custom_llm(prompt, model="custom-model"):
    with tracer.start_span("llm_call") as span:
        # Set LLM-specific attributes
        span.set_attribute("gen_ai.system", "custom")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", "chat")
        
        # Add input event
        span.add_event("gen_ai.content.prompt", {
            "gen_ai.prompt": prompt
        })
        
        # Simulate LLM call
        start_time = time.time()
        response = f"Response to: {prompt}"  # Your LLM call here
        end_time = time.time()
        
        # Add output event
        span.add_event("gen_ai.content.completion", {
            "gen_ai.completion": response
        })
        
        # Set usage metrics
        span.set_attribute("gen_ai.usage.input_tokens", len(prompt) // 4)
        span.set_attribute("gen_ai.usage.output_tokens", len(response) // 4)
        span.set_attribute("llm.latency_ms", (end_time - start_time) * 1000)
        
        return response

# Use the function
result = call_custom_llm("What is artificial intelligence?")
```

## Streaming LLM Calls

Trace streaming responses:

```python
import noveum_trace

noveum_trace.init(service_name="streaming-app")
tracer = noveum_trace.get_tracer()

def stream_llm_call(prompt):
    with tracer.start_span("streaming_llm_call") as span:
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("llm.streaming", True)
        
        # Add prompt event
        span.add_event("gen_ai.content.prompt", {
            "gen_ai.prompt": prompt
        })
        
        # Simulate streaming chunks
        chunks = ["Hello", " there!", " How", " can", " I", " help?"]
        full_response = ""
        
        for i, chunk in enumerate(chunks):
            full_response += chunk
            
            # Add chunk event
            span.add_event("gen_ai.content.chunk", {
                "gen_ai.completion.chunk": chunk,
                "chunk.index": i,
                "chunk.timestamp": time.time()
            })
            
            time.sleep(0.1)  # Simulate streaming delay
        
        # Add final completion event
        span.add_event("gen_ai.content.completion", {
            "gen_ai.completion": full_response
        })
        
        # Set final metrics
        span.set_attribute("gen_ai.usage.input_tokens", len(prompt) // 4)
        span.set_attribute("gen_ai.usage.output_tokens", len(full_response) // 4)
        span.set_attribute("llm.chunks_count", len(chunks))
        
        return full_response

# Use streaming function
response = stream_llm_call("Tell me a short story")
```

## LLM Decorators

Use decorators for simpler LLM tracing:

```python
import noveum_trace
from noveum_trace.instrumentation import trace_llm_call

noveum_trace.init(service_name="decorator-app")

@trace_llm_call(
    system="openai",
    model="gpt-3.5-turbo",
    operation="chat"
)
def ask_question(question):
    # Your LLM call implementation
    return f"Answer to: {question}"

@trace_llm_call(
    system="anthropic",
    model="claude-3-haiku",
    operation="completion"
)
def generate_text(prompt):
    # Your LLM call implementation
    return f"Generated text for: {prompt}"

# Use decorated functions
answer = ask_question("What is Python?")
text = generate_text("Write a poem about AI")
```

## Semantic Conventions

The SDK follows OpenTelemetry semantic conventions for generative AI:

### Required Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `gen_ai.system` | LLM system name | `"openai"`, `"anthropic"` |
| `gen_ai.request.model` | Model name | `"gpt-3.5-turbo"`, `"claude-3-haiku"` |
| `gen_ai.operation.name` | Operation type | `"chat"`, `"completion"` |

### Usage Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `gen_ai.usage.input_tokens` | Input token count | `150` |
| `gen_ai.usage.output_tokens` | Output token count | `75` |
| `gen_ai.usage.total_tokens` | Total token count | `225` |

### Request Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `gen_ai.request.temperature` | Temperature setting | `0.7` |
| `gen_ai.request.max_tokens` | Max tokens limit | `1000` |
| `gen_ai.request.top_p` | Top-p setting | `0.9` |

### Custom Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `llm.latency_ms` | Response latency | `1250.5` |
| `llm.streaming` | Streaming mode | `true` |
| `llm.chunks_count` | Number of chunks | `15` |

## Events

### Input Events

```python
span.add_event("gen_ai.content.prompt", {
    "gen_ai.prompt": "What is machine learning?"
})
```

### Output Events

```python
span.add_event("gen_ai.content.completion", {
    "gen_ai.completion": "Machine learning is a subset of AI..."
})
```

### Streaming Events

```python
span.add_event("gen_ai.content.chunk", {
    "gen_ai.completion.chunk": "Hello",
    "chunk.index": 0,
    "chunk.timestamp": time.time()
})
```

## Error Handling

Trace LLM errors and exceptions:

```python
import noveum_trace

noveum_trace.init(service_name="error-handling-app")
tracer = noveum_trace.get_tracer()

def safe_llm_call(prompt):
    with tracer.start_span("llm_call") as span:
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
        
        try:
            # Your LLM call here
            response = call_llm_api(prompt)
            
            span.add_event("gen_ai.content.completion", {
                "gen_ai.completion": response
            })
            
            return response
            
        except Exception as e:
            # Record the exception
            span.record_exception(e)
            span.set_status("error", str(e))
            
            # Add error-specific attributes
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            
            # Re-raise or handle as needed
            raise
```

## Multi-Turn Conversations

Trace conversational flows:

```python
import noveum_trace

noveum_trace.init(service_name="conversation-app")
tracer = noveum_trace.get_tracer()

class ConversationTracer:
    def __init__(self, conversation_id):
        self.conversation_id = conversation_id
        self.turn_count = 0
    
    def trace_turn(self, user_message, assistant_response):
        self.turn_count += 1
        
        with tracer.start_span("conversation_turn") as span:
            span.set_attribute("conversation.id", self.conversation_id)
            span.set_attribute("conversation.turn", self.turn_count)
            
            # Trace the LLM call
            with tracer.start_span("llm_call") as llm_span:
                llm_span.set_attribute("gen_ai.system", "openai")
                llm_span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
                
                # Add conversation context
                llm_span.add_event("gen_ai.content.prompt", {
                    "gen_ai.prompt": user_message
                })
                
                llm_span.add_event("gen_ai.content.completion", {
                    "gen_ai.completion": assistant_response
                })
                
                # Set turn-specific attributes
                llm_span.set_attribute("conversation.turn", self.turn_count)
                llm_span.set_attribute("gen_ai.usage.input_tokens", len(user_message) // 4)
                llm_span.set_attribute("gen_ai.usage.output_tokens", len(assistant_response) // 4)

# Usage
conversation = ConversationTracer("conv_123")
conversation.trace_turn("Hello!", "Hi there! How can I help you?")
conversation.trace_turn("What's the weather?", "I don't have access to weather data.")
```

## Batch Processing

Trace batch LLM operations:

```python
import noveum_trace

noveum_trace.init(service_name="batch-app")
tracer = noveum_trace.get_tracer()

def process_batch(prompts):
    with tracer.start_span("batch_llm_processing") as batch_span:
        batch_span.set_attribute("batch.size", len(prompts))
        batch_span.set_attribute("batch.id", "batch_123")
        
        results = []
        total_tokens = 0
        
        for i, prompt in enumerate(prompts):
            with tracer.start_span(f"batch_item_{i}") as item_span:
                item_span.set_attribute("batch.item_index", i)
                item_span.set_attribute("gen_ai.system", "openai")
                item_span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
                
                # Process individual item
                response = f"Response to: {prompt}"
                results.append(response)
                
                # Track tokens
                input_tokens = len(prompt) // 4
                output_tokens = len(response) // 4
                total_tokens += input_tokens + output_tokens
                
                item_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                item_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        
        # Set batch summary
        batch_span.set_attribute("batch.total_tokens", total_tokens)
        batch_span.set_attribute("batch.avg_tokens_per_item", total_tokens / len(prompts))
        
        return results

# Process a batch
prompts = ["What is AI?", "Explain ML", "Define NLP"]
responses = process_batch(prompts)
```

## Performance Monitoring

Monitor LLM performance:

```python
import noveum_trace
import time

noveum_trace.init(service_name="performance-app")
tracer = noveum_trace.get_tracer()

def monitored_llm_call(prompt):
    with tracer.start_span("monitored_llm_call") as span:
        # Performance tracking
        start_time = time.time()
        
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
        
        # Add performance markers
        span.add_event("llm.request_start", {"timestamp": start_time})
        
        # Simulate LLM call with timing
        time.sleep(0.5)  # Simulate network latency
        first_token_time = time.time()
        
        span.add_event("llm.first_token", {
            "timestamp": first_token_time,
            "ttft_ms": (first_token_time - start_time) * 1000
        })
        
        # Simulate completion
        time.sleep(0.3)
        end_time = time.time()
        
        # Set performance metrics
        total_latency = (end_time - start_time) * 1000
        ttft = (first_token_time - start_time) * 1000
        
        span.set_attribute("llm.latency_ms", total_latency)
        span.set_attribute("llm.time_to_first_token_ms", ttft)
        span.set_attribute("llm.tokens_per_second", 50 / (total_latency / 1000))
        
        span.add_event("llm.request_complete", {
            "timestamp": end_time,
            "total_latency_ms": total_latency
        })
        
        return "LLM response"

# Monitor performance
response = monitored_llm_call("Generate a summary")
```

## Best Practices

1. **Use auto-instrumentation when possible** for consistency
2. **Include relevant context** in span attributes
3. **Capture token usage** for cost tracking
4. **Monitor performance metrics** like latency and TTFT
5. **Handle errors gracefully** with proper exception recording
6. **Use meaningful span names** for better observability
7. **Consider privacy** when capturing content
8. **Batch operations** for better performance
9. **Use structured events** for better analysis
10. **Monitor SDK overhead** in production

