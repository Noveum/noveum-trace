"""
Comprehensive Example: Noveum Trace SDK

This example demonstrates all key features of the Noveum Trace SDK:
- Simple initialization with project ID
- Unified @trace decorator with different parameters
- Multi-agent workflows
- LLM tracing
- Tool call tracing
- Dynamic span updates
- Error handling
"""

import random
import time

import noveum_trace
from noveum_trace import Agent, AgentConfig, AgentContext, trace, update_current_span


def main():
    """Main example function demonstrating all SDK features."""

    print("🚀 Noveum Trace SDK - Comprehensive Example")
    print("=" * 50)

    # 1. Initialize the SDK with project configuration
    print("\n1. Initializing SDK...")
    tracer = noveum_trace.init(
        project_id="comprehensive-example",
        project_name="Comprehensive AI Application",
        org_id="org-demo",
        user_id="user-demo",
        session_id="session-demo",
        environment="development",
        file_logging=True,
        log_directory="./example_traces",
        auto_instrument=True,
        capture_content=True,
        custom_headers={"X-Example-Version": "1.0.0"},
    )
    print(f"✅ SDK initialized for project: {tracer.config.project_id}")

    # 2. Basic function tracing
    print("\n2. Basic Function Tracing...")

    @trace
    def simple_function(data):
        """Simple function with basic tracing."""
        time.sleep(0.1)  # Simulate work
        return f"Processed: {data}"

    @trace(name="custom-data-processing")
    def custom_named_function(data):
        """Function with custom span name."""
        time.sleep(0.05)
        return f"Custom processed: {data}"

    result1 = simple_function("user input")
    result2 = custom_named_function("more data")
    print(f"✅ Basic tracing: {result1}, {result2}")

    # 3. LLM Tracing (simulated)
    print("\n3. LLM Tracing...")

    @trace(type="llm", model="gpt-4", operation="chat")
    def mock_openai_call(messages):
        """Simulated OpenAI API call."""
        time.sleep(0.2)  # Simulate API latency

        # Update span with LLM-specific info
        update_current_span(
            attributes={
                "llm.request.messages": len(messages),
                "llm.request.temperature": 0.7,
                "llm.usage.input_tokens": 100,
                "llm.usage.output_tokens": 50,
                "llm.latency_ms": 200,
            }
        )

        return {
            "choices": [{"message": {"content": "AI response to your query"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    @trace(type="llm", model="claude-3", operation="completion")
    def mock_anthropic_call(prompt):
        """Simulated Anthropic API call."""
        time.sleep(0.15)

        update_current_span(
            attributes={
                "llm.request.prompt_length": len(prompt),
                "llm.usage.input_tokens": 80,
                "llm.usage.output_tokens": 40,
                "llm.latency_ms": 150,
            }
        )

        return {"content": "Claude's response to your prompt"}

    mock_openai_call([{"role": "user", "content": "Hello!"}])
    mock_anthropic_call("Explain quantum computing")
    print("✅ LLM tracing: Got responses from both models")

    # 4. Multi-Agent Workflow
    print("\n4. Multi-Agent Workflow...")

    # Define agents
    coordinator = Agent(
        AgentConfig(
            name="task-coordinator",
            agent_type="orchestrator",
            id="coord-001",
            capabilities={"planning", "coordination"},
            tags={"critical", "orchestrator"},
        )
    )

    data_worker = Agent(
        AgentConfig(
            name="data-processor",
            agent_type="worker",
            id="worker-001",
            capabilities={"data_processing", "analysis"},
            tags={"worker", "data"},
        )
    )

    llm_worker = Agent(
        AgentConfig(
            name="llm-specialist",
            agent_type="worker",
            id="worker-002",
            capabilities={"llm_calls", "text_generation"},
            tags={"worker", "llm"},
        )
    )

    # Coordinator plans the task
    with AgentContext(coordinator):

        @trace(type="planning", agent="task-coordinator")
        def plan_complex_task(user_request):
            """Coordinator plans how to handle the user request."""
            update_current_span(metadata={"step": "analyzing_request"}, progress=25)

            # Analyze request
            time.sleep(0.1)

            update_current_span(metadata={"step": "creating_plan"}, progress=75)

            # Create execution plan
            plan = {
                "steps": [
                    {
                        "agent": "data-processor",
                        "task": "process_data",
                        "data": user_request,
                    },
                    {
                        "agent": "llm-specialist",
                        "task": "generate_response",
                        "context": "processed_data",
                    },
                ],
                "estimated_duration": "2s",
            }

            update_current_span(
                metadata={"step": "plan_complete"},
                progress=100,
                attributes={"plan_steps": len(plan["steps"])},
            )

            return plan

        task_plan = plan_complex_task("Analyze sales data and generate insights")
        print(f"✅ Task planned with {len(task_plan['steps'])} steps")

    # Data worker processes data
    with AgentContext(data_worker):

        @trace(type="data_processing", agent="data-processor")
        def process_sales_data(raw_data):
            """Data worker processes the sales data."""
            update_current_span(
                metadata={"step": "data_validation", "input_size": len(str(raw_data))},
                progress=20,
            )

            # Validate data
            time.sleep(0.05)

            update_current_span(metadata={"step": "data_cleaning"}, progress=50)

            # Clean data
            time.sleep(0.1)

            update_current_span(metadata={"step": "data_analysis"}, progress=80)

            # Analyze data
            time.sleep(0.08)
            processed_data = {
                "total_sales": 150000,
                "growth_rate": 12.5,
                "top_products": ["Product A", "Product B"],
                "insights": ["Sales increased 12.5%", "Product A is trending"],
            }

            update_current_span(
                metadata={"step": "analysis_complete"},
                progress=100,
                attributes={
                    "records_processed": 1000,
                    "insights_generated": len(processed_data["insights"]),
                },
            )

            return processed_data

        processed_data = process_sales_data("raw sales data")
        print(f"✅ Data processed: {processed_data['total_sales']} total sales")

    # LLM worker generates response
    with AgentContext(llm_worker):

        @trace(type="text_generation", agent="llm-specialist")
        def generate_insights_report(data):
            """LLM worker generates a human-readable report."""
            update_current_span(metadata={"step": "prompt_preparation"}, progress=25)

            # Prepare prompt
            prompt = f"Generate insights report for: {data}"
            time.sleep(0.05)

            update_current_span(
                metadata={"step": "llm_call"},
                progress=75,
                attributes={"prompt_length": len(prompt)},
            )

            # Simulate LLM call
            time.sleep(0.2)
            report = f"""
            Sales Insights Report:
            - Total Sales: ${data['total_sales']:,}
            - Growth Rate: {data['growth_rate']}%
            - Key Insights: {', '.join(data['insights'])}
            - Top Products: {', '.join(data['top_products'])}
            """

            update_current_span(
                metadata={"step": "report_generated"},
                progress=100,
                attributes={
                    "report_length": len(report),
                    "insights_included": len(data["insights"]),
                },
            )

            return report.strip()

        final_report = generate_insights_report(processed_data)
        print(f"✅ Report generated: {len(final_report)} characters")

    # 5. Tool Call Tracing
    print("\n5. Tool Call Tracing...")

    @trace(type="tool", name="web_search")
    def search_web(query):
        """Simulated web search tool."""
        time.sleep(0.1)

        update_current_span(
            attributes={
                "tool.query": query,
                "tool.results_count": 10,
                "tool.latency_ms": 100,
            }
        )

        return f"Search results for: {query}"

    @trace(type="tool", name="calculator")
    def calculate(expression):
        """Calculator tool."""
        try:
            result = eval(expression)  # Note: Don't use eval in production!

            update_current_span(
                attributes={
                    "tool.expression": expression,
                    "tool.result": result,
                    "tool.success": True,
                }
            )

            return result
        except Exception as e:
            update_current_span(
                attributes={
                    "tool.expression": expression,
                    "tool.error": str(e),
                    "tool.success": False,
                }
            )
            raise

    @trace(type="tool", name="data_validator")
    def validate_data(data):
        """Data validation tool."""
        time.sleep(0.05)

        is_valid = len(data) > 0 and isinstance(data, (str, dict, list))

        update_current_span(
            attributes={
                "tool.data_type": type(data).__name__,
                "tool.data_size": len(str(data)),
                "tool.is_valid": is_valid,
            }
        )

        return {"valid": is_valid, "data_type": type(data).__name__}

    search_web("AI trends 2024")
    calc_result = calculate("(100 + 50) * 1.2")
    validate_data({"test": "data"})

    print(f"✅ Tool calls completed: search, calculation ({calc_result}), validation")

    # 6. Error Handling
    print("\n6. Error Handling...")

    @trace(type="error_demo")
    def function_with_error():
        """Function that demonstrates error handling."""
        update_current_span(metadata={"step": "starting_risky_operation"})

        # Simulate some work before error
        time.sleep(0.05)

        # Simulate random error
        if random.random() < 0.7:  # 70% chance of error
            raise ValueError("Simulated error for demonstration")

        return "Success!"

    try:
        error_result = function_with_error()
        print(f"✅ Error demo: {error_result}")
    except ValueError as e:
        print(f"✅ Error demo: Caught and traced error - {e}")

    # 7. Complex Workflow with Nested Spans
    print("\n7. Complex Nested Workflow...")

    @trace(name="complex-ai-workflow")
    def complex_ai_workflow(user_query):
        """Complex workflow demonstrating nested spans."""

        update_current_span(
            metadata={"workflow_stage": "initialization"},
            attributes={"user_query": user_query},
        )

        # Step 1: Validate input
        @trace(name="input-validation", type="validation")
        def validate_input(query):
            time.sleep(0.02)
            return len(query) > 0

        if not validate_input(user_query):
            raise ValueError("Invalid input")

        # Step 2: Process with multiple agents
        results = []

        with AgentContext(data_worker):

            @trace(name="data-enrichment", type="data_processing")
            def enrich_query(query):
                time.sleep(0.1)
                return f"enriched_{query}"

            enriched_query = enrich_query(user_query)
            results.append(enriched_query)

        with AgentContext(llm_worker):

            @trace(name="llm-processing", type="llm")
            def process_with_llm(query):
                time.sleep(0.15)
                return f"llm_processed_{query}"

            llm_result = process_with_llm(enriched_query)
            results.append(llm_result)

        # Step 3: Final processing
        @trace(name="result-compilation", type="compilation")
        def compile_results(results_list):
            time.sleep(0.05)
            return {"final_result": " -> ".join(results_list)}

        final_result = compile_results(results)

        update_current_span(
            metadata={"workflow_stage": "completed"},
            attributes={
                "steps_completed": 3,
                "agents_used": 2,
                "final_result_length": len(str(final_result)),
            },
        )

        return final_result

    complex_result = complex_ai_workflow("Analyze market trends")
    print(f"✅ Complex workflow completed: {len(str(complex_result))} chars")

    # 8. Performance Summary
    print("\n8. Performance Summary...")

    @trace(name="performance-summary")
    def generate_performance_summary():
        """Generate a summary of the example performance."""

        # Simulate gathering metrics
        time.sleep(0.1)

        summary = {
            "total_operations": 15,
            "agents_used": 3,
            "llm_calls": 2,
            "tool_calls": 3,
            "errors_handled": 1,
            "success_rate": "93%",
        }

        update_current_span(attributes=summary)

        return summary

    perf_summary = generate_performance_summary()
    print(f"✅ Performance summary: {perf_summary['total_operations']} operations")

    # 9. Cleanup
    print("\n9. Cleanup...")

    # Flush any pending traces
    noveum_trace.flush(timeout_ms=2000)

    # Shutdown the SDK
    noveum_trace.shutdown()

    print("✅ SDK shutdown complete")

    print("\n" + "=" * 50)
    print("🎉 Comprehensive example completed successfully!")
    print("\nKey features demonstrated:")
    print("  ✓ Simple initialization with project ID")
    print("  ✓ Unified @trace decorator with parameters")
    print("  ✓ Multi-agent workflows with context")
    print("  ✓ LLM tracing with metrics")
    print("  ✓ Tool call tracing")
    print("  ✓ Dynamic span updates")
    print("  ✓ Error handling and recovery")
    print("  ✓ Complex nested workflows")
    print("  ✓ Performance monitoring")
    print("\nCheck the './example_traces' directory for generated trace files!")


if __name__ == "__main__":
    main()
