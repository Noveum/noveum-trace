# Flexible Tracing Approaches - Beyond Decorators

## 🎯 Problem Statement

While decorators are elegant for new code, they have limitations in real-world scenarios:

1. **Existing Code**: Can't easily wrap existing LLM calls without refactoring
2. **Third-Party Libraries**: Can't modify library code to add decorators
3. **Mixed Logic**: Functions with both LLM calls and other logic need granular tracing
4. **Dynamic Calls**: Runtime-determined LLM calls are hard to decorate
5. **Legacy Integration**: Existing codebases need minimal-change integration

## 🚀 Solution: Multiple Tracing Approaches

The Noveum Trace SDK supports multiple tracing patterns to handle all real-world scenarios:

### 1. Context Managers (Inline Tracing)
### 2. Automatic Instrumentation (Monkey Patching)
### 3. Manual Span Creation
### 4. Proxy Objects
### 5. Auto-Discovery and Patching

---

## 1. 🔄 Context Managers - Inline Tracing

### Basic Context Manager Usage

```python
import noveum_trace
from openai import OpenAI

client = OpenAI()

# Instead of decorating the whole function, trace specific operations
def process_user_query(user_input: str):
    # Some preprocessing logic
    cleaned_input = user_input.strip().lower()

    # Trace just the LLM call
    with noveum_trace.trace_llm(model="gpt-4", provider="openai") as span:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": cleaned_input}]
        )

        # Automatically capture LLM attributes
        span.set_attributes({
            "llm.input_tokens": response.usage.prompt_tokens,
            "llm.output_tokens": response.usage.completion_tokens,
            "llm.total_tokens": response.usage.total_tokens,
            "llm.response": response.choices[0].message.content
        })

    # More processing logic
    final_response = post_process_response(response.choices[0].message.content)
    return final_response
```

### Advanced Context Manager with Multiple Operations

```python
def complex_ai_workflow(user_query: str):
    results = {}

    # Step 1: Query enhancement
    with noveum_trace.trace_llm(model="gpt-3.5-turbo", operation="query_enhancement") as span:
        enhanced_query = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Enhance this query: {user_query}"}]
        )
        results["enhanced_query"] = enhanced_query.choices[0].message.content
        span.set_attribute("llm.enhanced_query", results["enhanced_query"])

    # Step 2: Information retrieval (non-LLM operation)
    with noveum_trace.trace_operation("vector_search") as span:
        search_results = vector_db.search(results["enhanced_query"])
        span.set_attributes({
            "search.query": results["enhanced_query"],
            "search.results_count": len(search_results),
            "search.top_score": search_results[0].score if search_results else 0
        })

    # Step 3: Response generation with context
    with noveum_trace.trace_llm(model="gpt-4", operation="response_generation") as span:
        context = "\n".join([r.content for r in search_results[:3]])

        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Context: {context}"},
                {"role": "user", "content": results["enhanced_query"]}
            ]
        )

        span.set_attributes({
            "llm.context_length": len(context),
            "llm.context_sources": len(search_results[:3]),
            "llm.final_response": final_response.choices[0].message.content
        })

    return final_response.choices[0].message.content
```

---

## 2. 🔧 Automatic Instrumentation (Monkey Patching)

### Auto-Instrument Popular Libraries

```python
# noveum_trace/integrations/auto_instrument.py

import noveum_trace

# Automatically instrument OpenAI
noveum_trace.auto_instrument("openai")

# Now ALL OpenAI calls are automatically traced
from openai import OpenAI
client = OpenAI()

# This call is automatically traced without any decorators or context managers
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Also works with other libraries
noveum_trace.auto_instrument("anthropic")
noveum_trace.auto_instrument("langchain")
noveum_trace.auto_instrument("llamaindex")
```

### Selective Auto-Instrumentation

```python
# Only instrument specific methods
noveum_trace.auto_instrument("openai", methods=["chat.completions.create", "embeddings.create"])

# Instrument with custom configuration
noveum_trace.auto_instrument("openai", config={
    "capture_inputs": True,
    "capture_outputs": True,
    "max_input_length": 1000,
    "redact_pii": True
})
```

### Implementation of Auto-Instrumentation

```python
# noveum_trace/integrations/openai_auto.py

import functools
from typing import Any, Dict
import openai
from noveum_trace import get_client, trace_llm

class OpenAIInstrumentation:
    """Automatic instrumentation for OpenAI library."""

    def __init__(self):
        self.original_methods = {}
        self.instrumented = False

    def instrument(self, config: Dict[str, Any] = None):
        """Instrument OpenAI library methods."""
        if self.instrumented:
            return

        config = config or {}

        # Instrument chat completions
        self._instrument_chat_completions(config)

        # Instrument embeddings
        self._instrument_embeddings(config)

        # Instrument image generation
        self._instrument_image_generation(config)

        self.instrumented = True

    def _instrument_chat_completions(self, config: Dict[str, Any]):
        """Instrument chat completions method."""
        original_create = openai.chat.completions.create
        self.original_methods['chat.completions.create'] = original_create

        @functools.wraps(original_create)
        def traced_create(*args, **kwargs):
            # Extract model from kwargs
            model = kwargs.get('model', 'unknown')

            with trace_llm(model=model, provider="openai") as span:
                try:
                    # Capture inputs if configured
                    if config.get('capture_inputs', True):
                        messages = kwargs.get('messages', [])
                        if messages:
                            span.set_attribute("llm.messages", str(messages)[:config.get('max_input_length', 2000)])

                    # Make the actual API call
                    response = original_create(*args, **kwargs)

                    # Capture outputs if configured
                    if config.get('capture_outputs', True):
                        if hasattr(response, 'usage'):
                            span.set_attributes({
                                "llm.input_tokens": response.usage.prompt_tokens,
                                "llm.output_tokens": response.usage.completion_tokens,
                                "llm.total_tokens": response.usage.total_tokens,
                            })

                        if hasattr(response, 'choices') and response.choices:
                            content = response.choices[0].message.content
                            span.set_attribute("llm.response", content[:config.get('max_output_length', 2000)])

                    span.set_status("ok")
                    return response

                except Exception as e:
                    span.record_exception(e)
                    span.set_status("error", str(e))
                    raise

        # Replace the original method
        openai.chat.completions.create = traced_create

    def uninstrument(self):
        """Remove instrumentation and restore original methods."""
        if not self.instrumented:
            return

        for method_path, original_method in self.original_methods.items():
            if method_path == 'chat.completions.create':
                openai.chat.completions.create = original_method

        self.original_methods.clear()
        self.instrumented = False

# Global instrumentation instance
_openai_instrumentation = OpenAIInstrumentation()

def auto_instrument_openai(config: Dict[str, Any] = None):
    """Auto-instrument OpenAI library."""
    _openai_instrumentation.instrument(config)

def uninstrument_openai():
    """Remove OpenAI instrumentation."""
    _openai_instrumentation.uninstrument()
```

---

## 3. 🎯 Manual Span Creation

### Direct Span Management

```python
import noveum_trace

def existing_function_with_llm_calls():
    # Your existing code...
    user_input = get_user_input()

    # Manually create and manage spans
    client = noveum_trace.get_client()

    # Create span for LLM call
    llm_span = client.start_span(
        name="gpt4_query_processing",
        attributes={
            "llm.model": "gpt-4",
            "llm.provider": "openai",
            "llm.operation": "chat_completion"
        }
    )

    try:
        # Your existing LLM call
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_input}]
        )

        # Add response attributes
        llm_span.set_attributes({
            "llm.input_tokens": response.usage.prompt_tokens,
            "llm.output_tokens": response.usage.completion_tokens,
            "llm.response": response.choices[0].message.content
        })

        llm_span.set_status("ok")

    except Exception as e:
        llm_span.record_exception(e)
        llm_span.set_status("error", str(e))
        raise
    finally:
        client.finish_span(llm_span)

    # Continue with your existing code...
    return process_response(response)
```

### Batch Span Creation for Multiple Operations

```python
def process_multiple_queries(queries: List[str]):
    client = noveum_trace.get_client()

    # Create parent span for the batch operation
    batch_span = client.start_span(
        name="batch_query_processing",
        attributes={
            "batch.size": len(queries),
            "batch.operation": "llm_processing"
        }
    )

    results = []

    try:
        for i, query in enumerate(queries):
            # Create child span for each query
            query_span = client.start_span(
                name=f"query_processing_{i}",
                parent_span_id=batch_span.span_id,
                attributes={
                    "query.index": i,
                    "query.text": query[:100]  # Truncated for storage
                }
            )

            try:
                # Process individual query
                result = process_single_query(query)
                results.append(result)

                query_span.set_attributes({
                    "query.success": True,
                    "query.result_length": len(str(result))
                })
                query_span.set_status("ok")

            except Exception as e:
                query_span.record_exception(e)
                query_span.set_status("error", str(e))
                results.append(None)
            finally:
                client.finish_span(query_span)

        # Set batch results
        batch_span.set_attributes({
            "batch.successful": sum(1 for r in results if r is not None),
            "batch.failed": sum(1 for r in results if r is None)
        })
        batch_span.set_status("ok")

    except Exception as e:
        batch_span.record_exception(e)
        batch_span.set_status("error", str(e))
        raise
    finally:
        client.finish_span(batch_span)

    return results
```

---

## 4. 🎭 Proxy Objects

### LLM Client Proxy

```python
# noveum_trace/proxies/llm_proxy.py

class TracedOpenAIClient:
    """Proxy for OpenAI client that automatically traces all calls."""

    def __init__(self, original_client, trace_config=None):
        self._original_client = original_client
        self._trace_config = trace_config or {}

        # Create traced versions of nested objects
        self.chat = TracedChatCompletions(original_client.chat, trace_config)
        self.embeddings = TracedEmbeddings(original_client.embeddings, trace_config)
        self.images = TracedImages(original_client.images, trace_config)

    def __getattr__(self, name):
        # Delegate other attributes to original client
        return getattr(self._original_client, name)

class TracedChatCompletions:
    """Traced version of chat completions."""

    def __init__(self, original_chat, trace_config):
        self._original_chat = original_chat
        self._trace_config = trace_config
        self.completions = TracedCompletions(original_chat.completions, trace_config)

class TracedCompletions:
    """Traced version of completions."""

    def __init__(self, original_completions, trace_config):
        self._original_completions = original_completions
        self._trace_config = trace_config

    def create(self, **kwargs):
        """Traced version of create method."""
        model = kwargs.get('model', 'unknown')

        with noveum_trace.trace_llm(model=model, provider="openai") as span:
            # Capture input attributes
            if self._trace_config.get('capture_inputs', True):
                messages = kwargs.get('messages', [])
                span.set_attribute("llm.messages", str(messages)[:2000])

            # Make the actual call
            response = self._original_completions.create(**kwargs)

            # Capture output attributes
            if self._trace_config.get('capture_outputs', True):
                if hasattr(response, 'usage'):
                    span.set_attributes({
                        "llm.input_tokens": response.usage.prompt_tokens,
                        "llm.output_tokens": response.usage.completion_tokens,
                        "llm.total_tokens": response.usage.total_tokens,
                    })

            return response

# Usage
from openai import OpenAI

# Create traced client
original_client = OpenAI()
traced_client = TracedOpenAIClient(original_client)

# Now all calls through traced_client are automatically traced
response = traced_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Agent Proxy for Existing Agent Instances

```python
class TracedAgentProxy:
    """Proxy for existing agent instances to add tracing."""

    def __init__(self, agent, agent_type="unknown"):
        self._agent = agent
        self._agent_type = agent_type

    def __getattr__(self, name):
        attr = getattr(self._agent, name)

        # If it's a callable method, wrap it with tracing
        if callable(attr):
            return self._wrap_method(attr, name)
        return attr

    def _wrap_method(self, method, method_name):
        @functools.wraps(method)
        def traced_method(*args, **kwargs):
            with noveum_trace.trace_agent(
                agent_type=self._agent_type,
                operation=method_name
            ) as span:
                try:
                    result = method(*args, **kwargs)
                    span.set_attributes({
                        "agent.method": method_name,
                        "agent.args_count": len(args),
                        "agent.kwargs_count": len(kwargs)
                    })
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status("error", str(e))
                    raise

        return traced_method

# Usage with existing agent
existing_agent = SomeAgentClass()
traced_agent = TracedAgentProxy(existing_agent, agent_type="custom_agent")

# All method calls are now traced
result = traced_agent.process_query("What is AI?")
```

---

## 5. 🔍 Auto-Discovery and Patching

### Automatic Library Detection

```python
# noveum_trace/auto_discovery.py

class AutoDiscovery:
    """Automatically discover and instrument AI libraries."""

    SUPPORTED_LIBRARIES = {
        'openai': {
            'methods': ['chat.completions.create', 'embeddings.create'],
            'instrumentation': 'noveum_trace.integrations.openai_auto'
        },
        'anthropic': {
            'methods': ['messages.create'],
            'instrumentation': 'noveum_trace.integrations.anthropic_auto'
        },
        'langchain': {
            'methods': ['LLMChain.run', 'ChatOpenAI.__call__'],
            'instrumentation': 'noveum_trace.integrations.langchain_auto'
        }
    }

    def discover_and_instrument(self):
        """Discover installed libraries and instrument them."""
        instrumented = []

        for lib_name, config in self.SUPPORTED_LIBRARIES.items():
            if self._is_library_installed(lib_name):
                try:
                    self._instrument_library(lib_name, config)
                    instrumented.append(lib_name)
                except Exception as e:
                    print(f"Failed to instrument {lib_name}: {e}")

        return instrumented

    def _is_library_installed(self, lib_name: str) -> bool:
        """Check if a library is installed."""
        try:
            __import__(lib_name)
            return True
        except ImportError:
            return False

    def _instrument_library(self, lib_name: str, config: dict):
        """Instrument a specific library."""
        instrumentation_module = config['instrumentation']
        # Dynamic import and instrumentation
        # Implementation details...

# Usage
auto_discovery = AutoDiscovery()
instrumented_libs = auto_discovery.discover_and_instrument()
print(f"Auto-instrumented: {instrumented_libs}")
```

---

## 6. 🔧 Enhanced SDK API

### New SDK Methods for Flexible Tracing

```python
# Enhanced noveum_trace/__init__.py

# Context managers
from .context_managers import trace_llm, trace_agent, trace_operation

# Auto-instrumentation
from .auto_instrument import auto_instrument, uninstrument

# Proxies
from .proxies import TracedOpenAIClient, TracedAgentProxy

# Manual tracing
from .manual import start_span, finish_span, get_current_span

# Convenience functions
def trace_existing_function(func, span_name=None, attributes=None):
    """Trace an existing function without modifying it."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with trace_operation(span_name or func.__name__) as span:
            if attributes:
                span.set_attributes(attributes)
            return func(*args, **kwargs)
    return wrapper

def trace_llm_call(llm_func, model=None, provider=None):
    """Trace a specific LLM function call."""
    @functools.wraps(llm_func)
    def wrapper(*args, **kwargs):
        with trace_llm(model=model, provider=provider) as span:
            result = llm_func(*args, **kwargs)
            # Auto-extract common LLM attributes
            if hasattr(result, 'usage'):
                span.set_attributes({
                    "llm.input_tokens": getattr(result.usage, 'prompt_tokens', 0),
                    "llm.output_tokens": getattr(result.usage, 'completion_tokens', 0),
                })
            return result
    return wrapper

# Global auto-instrumentation
def enable_auto_tracing(libraries=None, config=None):
    """Enable automatic tracing for specified libraries."""
    libraries = libraries or ['openai', 'anthropic', 'langchain']
    config = config or {}

    for lib in libraries:
        try:
            auto_instrument(lib, config.get(lib, {}))
        except Exception as e:
            print(f"Failed to auto-instrument {lib}: {e}")
```

---

## 7. 📚 Real-World Usage Examples

### Example 1: Existing Codebase Integration

```python
# Existing code that you can't easily refactor
import openai
from some_library import ExistingAgent

# Minimal change approach - just add auto-instrumentation
import noveum_trace
noveum_trace.init(api_key="your-key", project="existing-app")
noveum_trace.enable_auto_tracing(['openai'])

# Your existing code works unchanged, but now it's traced!
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Existing agent also gets traced with proxy
agent = ExistingAgent()
traced_agent = noveum_trace.TracedAgentProxy(agent, "existing_agent")
result = traced_agent.process("some input")
```

### Example 2: Mixed Logic Function

```python
def complex_business_logic(user_request: str):
    # Business logic that can't be easily decorated

    # Step 1: Validate input (no tracing needed)
    if not user_request.strip():
        return "Invalid input"

    # Step 2: Trace just the LLM enhancement
    with noveum_trace.trace_llm(model="gpt-3.5-turbo", operation="input_enhancement"):
        enhanced_request = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Enhance: {user_request}"}]
        ).choices[0].message.content

    # Step 3: Database lookup (trace as operation)
    with noveum_trace.trace_operation("database_lookup") as span:
        db_results = database.search(enhanced_request)
        span.set_attribute("db.results_count", len(db_results))

    # Step 4: Business logic processing (no tracing)
    processed_results = apply_business_rules(db_results)

    # Step 5: Final LLM call for response generation
    with noveum_trace.trace_llm(model="gpt-4", operation="response_generation"):
        final_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Generate response based on data"},
                {"role": "user", "content": str(processed_results)}
            ]
        ).choices[0].message.content

    return final_response
```

### Example 3: Third-Party Library Integration

```python
# Working with LangChain without modifying LangChain code
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Auto-instrument LangChain
noveum_trace.auto_instrument("langchain", config={
    "capture_inputs": True,
    "capture_outputs": True,
    "trace_chains": True
})

# Your existing LangChain code now gets traced automatically
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["product"],
    template="Write a marketing copy for {product}"
)
chain = LLMChain(llm=llm, prompt=prompt)

# This call is automatically traced
result = chain.run("wireless headphones")
```

---

## 8. 🎯 Best Practices and Recommendations

### When to Use Each Approach

| Scenario | Recommended Approach | Reason |
|----------|---------------------|---------|
| New code development | Decorators | Clean, declarative, easy to read |
| Existing codebase | Auto-instrumentation | Minimal code changes |
| Mixed logic functions | Context managers | Granular control over what's traced |
| Third-party libraries | Auto-instrumentation + Proxies | No source code modification needed |
| Complex workflows | Manual span creation | Full control over span hierarchy |
| Legacy systems | Proxy objects | Wrapper pattern, no refactoring |

### Performance Considerations

```python
# Configure tracing for production
noveum_trace.init(
    api_key="your-key",
    project="production-app",
    config={
        "sampling_rate": 0.1,  # Sample 10% of traces
        "auto_instrumentation": {
            "capture_inputs": False,  # Reduce overhead
            "capture_outputs": True,
            "max_input_length": 500
        },
        "batch_size": 100,  # Larger batches for efficiency
        "async_export": True  # Non-blocking export
    }
)
```

### Error Handling

```python
# Robust error handling for instrumentation
try:
    noveum_trace.enable_auto_tracing(['openai', 'anthropic'])
except Exception as e:
    # Graceful degradation - app continues without tracing
    print(f"Tracing initialization failed: {e}")
    # Optionally set up fallback logging
```

---

## 9. 🚀 Implementation Priority

### Phase 1: Core Flexibility (Immediate)
- [ ] Context managers for inline tracing
- [ ] Manual span creation API
- [ ] Basic auto-instrumentation for OpenAI

### Phase 2: Advanced Instrumentation (Next Sprint)
- [ ] Proxy objects for existing instances
- [ ] Auto-discovery of installed libraries
- [ ] LangChain and Anthropic auto-instrumentation

### Phase 3: Production Features (Future)
- [ ] Advanced sampling strategies
- [ ] Performance optimization
- [ ] Comprehensive library support

This flexible approach ensures that Noveum Trace SDK can be adopted in any codebase, regardless of architecture or existing patterns, while maintaining the simplicity that makes decorators attractive for new development.
