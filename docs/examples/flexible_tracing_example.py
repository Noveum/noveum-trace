"""
Flexible Tracing Approaches for Noveum Trace SDK

This example demonstrates various ways to trace LLM calls and agents
without requiring decorators on every function.
"""

import json
import os
import time
from typing import Any

# Load environment variables (install python-dotenv if needed)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "python-dotenv not installed. Environment variables will be read from system only."
    )
    pass

# Import OpenAI for examples
from openai import OpenAI

# Initialize Noveum Trace SDK
import noveum_trace

# Import the new flexible tracing approaches
from noveum_trace.context_managers import trace_llm, trace_operation

# Validate required environment variables
noveum_api_key = os.getenv("NOVEUM_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not noveum_api_key:
    raise ValueError(
        "NOVEUM_API_KEY environment variable is required. "
        "Please set it before running this example."
    )
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is required. "
        "Please set it before running this example."
    )

noveum_trace.init(
    api_key=noveum_api_key,
    project="flexible-tracing-demo",
    environment="development",
)

client = OpenAI(api_key=openai_api_key)

# =============================================================================
# APPROACH 1: CONTEXT MANAGERS
# =============================================================================


def process_user_query_with_context_manager(user_query: str) -> dict[str, Any]:
    """
    Process a user query using context managers for granular tracing.

    This approach allows you to trace specific parts of a function
    without decorating the entire function.
    """
    print(f"Processing query: {user_query}")
    results = {}

    # Step 1: Preprocess the query (not traced)
    cleaned_query = user_query.strip().lower()

    # Step 2: Enhance the query with LLM (traced)
    with trace_llm(model="gpt-3.5-turbo", operation="query_enhancement") as span:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query enhancement assistant.",
                    },
                    {
                        "role": "user",
                        "content": f"Enhance this search query: {cleaned_query}",
                    },
                ],
            )

            enhanced_query = response.choices[0].message.content
            results["enhanced_query"] = enhanced_query

            # Add attributes to the span
            span.set_attributes(
                {
                    "llm.input_tokens": response.usage.prompt_tokens,
                    "llm.output_tokens": response.usage.completion_tokens,
                    "llm.total_tokens": response.usage.total_tokens,
                    "llm.enhanced_query": enhanced_query,
                }
            )
        except Exception as e:
            # The context manager will automatically record the exception
            print(f"Error enhancing query: {e}")
            results["enhanced_query"] = cleaned_query

    # Step 3: Simulate database lookup (traced as operation)
    with trace_operation("database_lookup") as span:
        # Simulate database query
        time.sleep(0.5)
        search_results = [
            {"id": 1, "title": "Result 1", "relevance": 0.95},
            {"id": 2, "title": "Result 2", "relevance": 0.85},
            {"id": 3, "title": "Result 3", "relevance": 0.75},
        ]
        results["search_results"] = search_results

        # Add attributes to the span
        span.set_attributes(
            {
                "db.query": results["enhanced_query"],
                "db.results_count": len(search_results),
                "db.top_relevance": (
                    search_results[0]["relevance"] if search_results else 0
                ),
            }
        )

    # Step 4: Generate response with LLM (traced)
    with trace_llm(model="gpt-4", operation="response_generation") as span:
        try:
            context = json.dumps(search_results[:2])

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"Use this context to answer: {context}",
                    },
                    {"role": "user", "content": cleaned_query},
                ],
            )

            final_response = response.choices[0].message.content
            results["final_response"] = final_response

            # Add attributes to the span
            span.set_attributes(
                {
                    "llm.input_tokens": response.usage.prompt_tokens,
                    "llm.output_tokens": response.usage.completion_tokens,
                    "llm.total_tokens": response.usage.total_tokens,
                    "llm.context_length": len(context),
                    "llm.response_length": len(final_response),
                }
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            results["final_response"] = "Sorry, I couldn't generate a response."

    # Step 5: Post-processing (not traced)
    results["processed_at"] = time.time()

    return results


# =============================================================================
# APPROACH 2: MANUAL SPAN CREATION
# =============================================================================


def legacy_function_with_manual_spans(query: str):
    """
    Demonstrate manual span creation for legacy code.

    This approach allows adding tracing to existing code with minimal changes.
    """
    print("\nDemonstrating manual span creation...")

    # Get the client for manual span management
    client = noveum_trace.get_client()

    # Create a trace if none exists
    trace = None
    if not noveum_trace.core.context.get_current_trace():
        trace = client.start_trace("manual_trace")

    # Create span for the operation
    span = client.start_span(
        name="legacy_function",
        attributes={
            "function.name": "legacy_function_with_manual_spans",
            "function.query": query,
        },
    )

    try:
        # Simulate some work
        time.sleep(1)

        # Create a child span for a sub-operation
        child_span = client.start_span(
            name="legacy_sub_operation",
            parent_span_id=span.span_id,
            attributes={"sub_operation.type": "processing"},
        )

        try:
            # Simulate sub-operation
            time.sleep(0.5)
            result = f"Processed: {query.upper()}"

            # Add result attributes
            child_span.set_attributes(
                {
                    "sub_operation.result_length": len(result),
                    "sub_operation.success": True,
                }
            )

            child_span.set_status("ok")

        except Exception as e:
            child_span.record_exception(e)
            child_span.set_status("error", str(e))
            raise
        finally:
            # Always finish the child span
            client.finish_span(child_span)

        # Add result to parent span
        span.set_attributes(
            {"function.result": result, "function.duration_ms": 1500})

        span.set_status("ok")
        print(f"Manual span result: {result}")

        return result

    except Exception as e:
        span.record_exception(e)
        span.set_status("error", str(e))
        raise
    finally:
        # Always finish the span
        client.finish_span(span)

        # Finish the trace if we created one
        if trace:
            client.finish_trace(trace)


# =============================================================================
# APPROACH 3: MIXED APPROACH FOR COMPLEX WORKFLOWS
# =============================================================================


def complex_workflow_mixed_approach(user_input: str):
    """
    Demonstrate a mixed approach for complex workflows.

    This combines multiple tracing approaches for flexibility.
    """
    print("\nDemonstrating mixed approach for complex workflows...")

    # Start a trace for the entire workflow
    with noveum_trace.trace_operation("complex_workflow") as workflow_span:
        workflow_span.set_attributes(
            {"workflow.input": user_input, "workflow.start_time": time.time()}
        )

        results = {}

        # Step 1: Use context manager for custom operation
        with trace_operation("custom_processing") as process_span:
            # Simulate processing
            time.sleep(0.5)
            processed_input = f"Processed: {user_input}"

            process_span.set_attributes(
                {
                    "process.input_length": len(user_input),
                    "process.output_length": len(processed_input),
                }
            )

            results["processed_input"] = processed_input

        # Step 2: Use manual span for legacy code
        client = noveum_trace.get_client()
        legacy_span = client.start_span(
            name="legacy_operation",
            parent_span_id=workflow_span.span_id,
            attributes={"legacy.operation": "data_transformation"},
        )

        try:
            # Simulate legacy operation
            time.sleep(0.7)
            transformed_data = processed_input.replace(" ", "_")

            legacy_span.set_attributes(
                {"legacy.result": transformed_data, "legacy.success": True}
            )

            results["transformed_data"] = transformed_data
            legacy_span.set_status("ok")

        except Exception as e:
            legacy_span.record_exception(e)
            legacy_span.set_status("error", str(e))
            raise
        finally:
            client.finish_span(legacy_span)

        # Step 3: Use context manager for final LLM call
        with trace_llm(model="gpt-3.5-turbo", operation="final_processing") as llm_span:
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Summarize this data: {results}"}
                ],
            )

            final_result = response.choices[0].message.content
            results["final_result"] = final_result

            llm_span.set_attributes(
                {
                    "llm.input_tokens": response.usage.prompt_tokens,
                    "llm.output_tokens": response.usage.completion_tokens,
                    "llm.total_tokens": response.usage.total_tokens,
                }
            )

        # Update workflow span with final results
        workflow_span.set_attributes(
            {
                "workflow.end_time": time.time(),
                "workflow.steps_completed": 4,
                "workflow.success": True,
            }
        )

        print(f"Complex workflow result: {final_result[:50]}...")
        return results


# =============================================================================
# MAIN EXAMPLE RUNNER
# =============================================================================


def run_flexible_tracing_examples():
    """Run all flexible tracing examples."""
    print("🎯 Demonstrating Flexible Tracing Approaches with Noveum Trace SDK\n")

    # Example 1: Context Managers
    print("\n📌 EXAMPLE 1: CONTEXT MANAGERS")
    result1 = process_user_query_with_context_manager(
        "What is the capital of France?")
    print(f"Result: {result1['final_response'][:50]}...\n")

    # Example 2: Manual Span Creation
    print("\n📌 EXAMPLE 2: MANUAL SPAN CREATION")
    legacy_function_with_manual_spans("legacy system query")

    # Example 3: Mixed Approach
    print("\n📌 EXAMPLE 3: MIXED APPROACH")
    complex_workflow_mixed_approach("Demonstrate flexible tracing approaches")

    print("\n✅ All flexible tracing examples completed successfully!")
    print("\n📊 Trace data has been sent to Noveum API for analysis.")


if __name__ == "__main__":
    run_flexible_tracing_examples()
