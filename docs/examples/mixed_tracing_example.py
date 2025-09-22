"""
Mixed Tracing Example for Noveum Trace SDK.

This example demonstrates how to mix Noveum Trace SDK context managers
(agent_operation, agent_task) with LangChain operations to show that
tracing works seamlessly across different tracing approaches.

The example shows:
1. Using agent_operation context manager
2. Creating a span and adding attributes
3. Calling LangChain LLM with callback handler
4. Using agent_task context manager
5. All operations are properly nested and traced

Prerequisites:
    pip install noveum-trace[langchain]
    pip install langchain langchain-openai langchain-community

Environment Variables:
    NOVEUM_API_KEY: Your Noveum API key
    OPENAI_API_KEY: Your OpenAI API key (for LLM examples)
"""

import os
import time

from dotenv import load_dotenv

import noveum_trace
from noveum_trace import NoveumTraceCallbackHandler
from noveum_trace.context_managers import trace_agent, trace_operation

load_dotenv()


def mixed_tracing_example():
    """Example: Mixed tracing with SDK context managers and LangChain."""
    print("=== Mixed Tracing Example ===")
    print("Demonstrating agent_operation -> LangChain -> agent_task flow")
    print()

    try:
        from langchain_openai import ChatOpenAI

        # Initialize Noveum Trace with batch size 1 for immediate sending
        noveum_trace.init(
            endpoint="https://noveum.free.beeceptor.com",
            transport_config={"batch_size": 1, "batch_timeout": 5.0},
        )

        # Create LangChain callback handler
        callback_handler = NoveumTraceCallbackHandler()

        # Create LangChain LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.7, callbacks=[callback_handler]
        )

        print("1. Starting agent_operation context...")

        # Use agent_operation context manager
        with trace_agent(
            agent_type="research_agent",
            operation="knowledge_retrieval",
            capabilities=["web_search", "analysis", "synthesis"],
            attributes={
                "agent.session_id": "session_123",
                "agent.priority": "high",
                "agent.domain": "research",
            },
        ) as agent_span:

            print("   ✓ Agent operation span created")
            print("   ✓ Agent type: research_agent")
            print("   ✓ Operation: knowledge_retrieval")

            # Add some custom attributes to the agent span
            agent_span.set_attributes(
                {
                    "agent.start_time": time.time(),
                    "agent.request_id": "req_456",
                    "agent.user_id": "user_789",
                }
            )

            print("   ✓ Added custom attributes to agent span")

            # Create a custom span within the agent operation
            print("\n2. Creating custom span within agent operation...")

            with trace_operation(
                "data_preparation",
                attributes={
                    "prep.step": "query_formulation",
                    "prep.timestamp": time.time(),
                    "prep.source": "user_input",
                },
            ) as prep_span:

                print("   ✓ Custom span created for data preparation")

                # Simulate some data preparation work
                query = "What are the latest developments in artificial intelligence?"
                prep_span.set_attributes(
                    {
                        "prep.query_length": len(query),
                        "prep.query_type": "research_question",
                    }
                )

                print(f"   ✓ Prepared query: {query[:50]}...")

                # Now call LangChain LLM - this will create its own spans via callback
                print("\n3. Calling LangChain LLM with callback handler...")

                try:
                    response = llm.invoke(query)
                    print(
                        f"   ✓ LangChain response received: {response.content[:100]}..."
                    )

                    # Add response info to our span
                    prep_span.set_attributes(
                        {
                            "prep.response_length": len(response.content),
                            "prep.response_success": True,
                        }
                    )

                except Exception as e:
                    print(f"   ✗ LangChain call failed: {e}")
                    prep_span.set_attributes(
                        {"prep.response_success": False, "prep.error": str(e)}
                    )
                    raise

            print("\n4. Starting agent_task context...")

            # Use agent_task context manager (nested within agent_operation)
            with trace_agent(
                agent_type="analysis_agent",
                operation="content_analysis",
                capabilities=["text_analysis", "summarization"],
                attributes={
                    "task.id": "task_789",
                    "task.priority": "medium",
                    "task.dependencies": ["knowledge_retrieval"],
                },
            ) as task_span:

                print("   ✓ Agent task span created")
                print("   ✓ Task type: content_analysis")

                # Add task-specific attributes
                task_span.set_attributes(
                    {
                        "task.start_time": time.time(),
                        "task.parent_operation": "knowledge_retrieval",
                        "task.expected_duration": 30.0,
                    }
                )

                # Simulate some analysis work
                print("   ✓ Performing content analysis...")
                time.sleep(0.5)  # Simulate processing time

                # Add analysis results
                task_span.set_attributes(
                    {
                        "analysis.completed": True,
                        "analysis.duration": 0.5,
                        "analysis.complexity": "medium",
                    }
                )

                print("   ✓ Analysis completed")

                # Another LangChain call within the task
                print("\n5. Making another LangChain call within agent_task...")

                try:
                    follow_up_query = (
                        "Can you summarize the key points from your previous response?"
                    )
                    follow_up_response = llm.invoke(follow_up_query)
                    print(
                        f"   ✓ Follow-up response: {follow_up_response.content[:100]}..."
                    )

                    task_span.set_attributes(
                        {
                            "task.follow_up_success": True,
                            "task.final_response_length": len(
                                follow_up_response.content
                            ),
                        }
                    )

                except Exception as e:
                    print(f"   ✗ Follow-up call failed: {e}")
                    task_span.set_attributes(
                        {"task.follow_up_success": False, "task.error": str(e)}
                    )

            # Update the main agent span with final results
            agent_span.set_attributes(
                {
                    "agent.end_time": time.time(),
                    "agent.status": "completed",
                    "agent.tasks_completed": 1,
                }
            )

            print("\n6. Agent operation completed successfully!")
            print("   ✓ All spans properly nested and traced")
            print("   ✓ LangChain operations integrated seamlessly")
            print("   ✓ Mixed tracing approach working correctly")

    except ImportError as e:
        print(f"Skipping example - required packages not installed: {e}")
    except Exception as e:
        print(f"Error in mixed tracing example: {e}")
        import traceback

        traceback.print_exc()


def advanced_mixed_tracing_example():
    """Advanced example with more complex nesting and error handling."""
    print("\n" + "=" * 60)
    print("=== Advanced Mixed Tracing Example ===")
    print("Demonstrating complex nesting with error handling")
    print()

    try:
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

        # Create callback handler
        callback_handler = NoveumTraceCallbackHandler()

        # Create LangChain components
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.3, callbacks=[callback_handler]
        )

        prompt = PromptTemplate(
            input_variables=["topic", "context"],
            template="Based on the context: {context}\n\nProvide insights about: {topic}",
        )

        chain = LLMChain(llm=llm, prompt=prompt, callbacks=[callback_handler])

        print("1. Starting complex agent workflow...")

        # Main agent operation
        with trace_agent(
            agent_type="workflow_agent",
            operation="complex_analysis",
            capabilities=["multi_step_analysis", "error_handling", "chaining"],
            attributes={
                "workflow.id": "wf_001",
                "workflow.version": "1.0",
                "workflow.complexity": "high",
            },
        ) as _:

            print("   ✓ Main workflow agent started")

            # Step 1: Data collection
            with trace_operation(
                "data_collection", attributes={"step": 1, "phase": "input"}
            ) as data_span:

                print("   ✓ Data collection phase")
                context_data = "Recent AI developments include GPT-4, multimodal models, and improved reasoning capabilities."
                topic = "AI safety and ethics"

                data_span.set_attributes(
                    {
                        "data.context_length": len(context_data),
                        "data.topic": topic,
                        "data.collection_method": "manual",
                    }
                )

            # Step 2: LangChain chain execution
            with trace_operation(
                "langchain_analysis", attributes={"step": 2, "phase": "processing"}
            ) as analysis_span:

                print("   ✓ LangChain chain execution")

                try:
                    result = chain.run(topic=topic, context=context_data)
                    print(f"   ✓ Chain result: {result[:100]}...")

                    analysis_span.set_attributes(
                        {
                            "analysis.success": True,
                            "analysis.result_length": len(result),
                            "analysis.chain_type": "LLMChain",
                        }
                    )

                except Exception as e:
                    print(f"   ✗ Chain execution failed: {e}")
                    analysis_span.set_attributes(
                        {"analysis.success": False, "analysis.error": str(e)}
                    )
                    raise

            # Step 3: Nested agent task for validation
            with trace_agent(
                agent_type="validation_agent",
                operation="result_validation",
                capabilities=["validation", "quality_check"],
                attributes={
                    "validation.step": 3,
                    "validation.phase": "quality_assurance",
                },
            ) as _:

                print("   ✓ Validation agent started")

                # Another LangChain call for validation
                with trace_operation(
                    "validation_check",
                    attributes={"validation.type": "quality_assessment"},
                ) as validation_span:

                    try:
                        validation_query = f"Rate the quality of this analysis on a scale of 1-10: {result[:200]}"
                        validation_response = llm.invoke(validation_query)
                        print(
                            f"   ✓ Validation response: {validation_response.content}"
                        )

                        validation_span.set_attributes(
                            {
                                "validation.success": True,
                                "validation.response": validation_response.content,
                            }
                        )

                    except Exception as e:
                        print(f"   ✗ Validation failed: {e}")
                        validation_span.set_attributes(
                            {"validation.success": False, "validation.error": str(e)}
                        )

            # Final step: Results compilation
            with trace_operation(
                "results_compilation", attributes={"step": 4, "phase": "output"}
            ) as results_span:

                print("   ✓ Compiling final results")
                results_span.set_attributes(
                    {"results.status": "completed", "results.workflow_success": True}
                )

            print("\n   ✓ Complex workflow completed successfully!")
            print("   ✓ All nested spans properly created and linked")
            print("   ✓ Mixed tracing approach handled complex scenarios")

    except ImportError as e:
        print(f"Skipping advanced example - required packages not installed: {e}")
    except Exception as e:
        print(f"Error in advanced mixed tracing example: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all mixed tracing examples."""
    print("Noveum Trace - Mixed Tracing Examples")
    print("=" * 50)
    print("This example demonstrates mixing Noveum Trace SDK context managers")
    print("with LangChain operations to show seamless integration.")
    print()

    # Check if API keys are set
    if not os.getenv("NOVEUM_API_KEY"):
        print("Warning: NOVEUM_API_KEY not set. Using mock mode.")
        print()

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may fail.")
        print()

    # Run examples
    mixed_tracing_example()
    advanced_mixed_tracing_example()

    print("\n" + "=" * 60)
    print("=== Examples Complete ===")
    print("Check your Noveum dashboard to see the traced operations!")
    print("You should see:")
    print("- Nested spans from agent_operation -> agent_task")
    print("- LangChain operations integrated within the spans")
    print("- Proper parent-child relationships in the trace")
    print("- Mixed tracing approaches working together seamlessly")

    # Flush any pending traces
    noveum_trace.flush()


if __name__ == "__main__":
    main()
