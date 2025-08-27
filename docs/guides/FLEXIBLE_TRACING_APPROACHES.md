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
### 2. Manual Span Creation

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

## 2. 🎯 Manual Span Creation

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

## 3. 📚 Real-World Usage Examples

### Example 1: Mixed Logic Function

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

## 4. 🎯 Best Practices and Recommendations

### When to Use Each Approach

| Scenario | Recommended Approach | Reason |
|----------|---------------------|---------|
| New code development | Decorators | Clean, declarative, easy to read |
| Existing codebase | Context managers | Minimal code changes, granular control |
| Mixed logic functions | Context managers | Granular control over what's traced |
| Third-party libraries | Context managers | No source code modification needed |
| Complex workflows | Manual span creation | Full control over span hierarchy |
| Legacy systems | Context managers | Wrapper pattern, no refactoring |

### Performance Considerations

```python
# Configure tracing for production
noveum_trace.init(
    api_key="your-key",
    project="production-app",
    config={
        "sampling_rate": 0.1,  # Sample 10% of traces
        "batch_size": 100,  # Larger batches for efficiency
        "async_export": True  # Non-blocking export
    }
)
```

---

## 5. 🚀 Implementation Priority

### Phase 1: Core Flexibility (Immediate)
- [ ] Context managers for inline tracing
- [ ] Manual span creation API

### Phase 2: Production Features (Future)
- [ ] Advanced sampling strategies
- [ ] Performance optimization
- [ ] Comprehensive library support

This flexible approach ensures that Noveum Trace SDK can be adopted in any codebase, regardless of architecture or existing patterns, while maintaining the simplicity that makes decorators attractive for new development.
