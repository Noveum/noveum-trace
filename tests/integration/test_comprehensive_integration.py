#!/usr/bin/env python3
"""
Comprehensive integration test for all Noveum Trace features.
Tests multiple providers, frameworks, and advanced features.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables
from dotenv import load_dotenv

import noveum_trace
from noveum_trace.agents.context import AgentContext

load_dotenv()


def test_multi_provider_scenario():
    """Test a complex scenario using multiple providers."""
    print("🌐 Testing Multi-Provider Scenario...")

    # Initialize tracer
    noveum_trace.init(
        project_id="comprehensive_test", file_logging=True, log_directory="test_traces"
    )

    try:
        import anthropic
        import openai

        # Create agents for different providers
        openai_agent_config = noveum_trace.AgentConfig(
            name="OpenAI_Agent",
            description="Agent using OpenAI GPT models",
            version="1.0.0",
            capabilities={"text_generation", "function_calling"},
            metadata={"provider": "openai", "model": "gpt-3.5-turbo"},
        )

        anthropic_agent_config = noveum_trace.AgentConfig(
            name="Anthropic_Agent",
            description="Agent using Anthropic Claude models",
            version="1.0.0",
            capabilities={"text_generation", "analysis"},
            metadata={"provider": "anthropic", "model": "claude-3-haiku"},
        )

        registry = noveum_trace.get_agent_registry()
        openai_agent = registry.register_agent(openai_agent_config)
        anthropic_agent = registry.register_agent(anthropic_agent_config)

        # OpenAI client
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Anthropic client
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Multi-step workflow
        @noveum_trace.trace(name="multi_provider_workflow")
        def multi_provider_workflow(user_query: str) -> Dict[str, Any]:
            results = {}

            # Step 1: Use OpenAI for initial processing
            with AgentContext(openai_agent):

                @noveum_trace.trace(name="openai_processing")
                def openai_process(query: str) -> str:
                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that provides structured analysis.",
                            },
                            {
                                "role": "user",
                                "content": f"Analyze this query and provide key insights: {query}",
                            },
                        ],
                        max_tokens=150,
                    )
                    return response.choices[0].message.content

                results["openai_analysis"] = openai_process(user_query)

            # Step 2: Use Anthropic for follow-up
            with AgentContext(anthropic_agent):

                @noveum_trace.trace(name="anthropic_processing")
                def anthropic_process(analysis: str) -> str:
                    response = anthropic_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=150,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Based on this analysis: {analysis}\n\nProvide a concise summary and recommendation.",
                            }
                        ],
                    )
                    return response.content[0].text

                results["anthropic_summary"] = anthropic_process(
                    results["openai_analysis"]
                )

            return results

        # Execute the workflow
        user_query = "What are the best practices for implementing AI in healthcare?"
        results = multi_provider_workflow(user_query)

        print(f"✅ OpenAI Analysis: {results['openai_analysis'][:100]}...")
        print(f"✅ Anthropic Summary: {results['anthropic_summary'][:100]}...")

        return True

    except Exception as e:
        print(f"❌ Multi-provider scenario test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_concurrent_tracing():
    """Test concurrent tracing with multiple threads."""
    print("\n🔄 Testing Concurrent Tracing...")

    # Initialize tracer
    noveum_trace.init(
        project_id="concurrent_test", file_logging=True, log_directory="test_traces"
    )

    try:
        import openai

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        def worker_function(worker_id: int) -> str:
            """Worker function that makes LLM calls."""

            @noveum_trace.trace(name=f"worker_{worker_id}")
            def process_with_llm(prompt: str) -> str:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                )
                return response.choices[0].message.content

            return process_with_llm(f"Tell me a fact about the number {worker_id}")

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker_function, i) for i in range(1, 4)]
            results = [future.result() for future in futures]

        print(f"✅ Concurrent results: {len(results)} workers completed")
        for i, result in enumerate(results, 1):
            print(f"  Worker {i}: {result[:50]}...")

        return True

    except Exception as e:
        print(f"❌ Concurrent tracing test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_error_handling():
    """Test error handling and recovery."""
    print("\n⚠️ Testing Error Handling...")

    # Initialize tracer
    noveum_trace.init(
        project_id="error_test", file_logging=True, log_directory="test_traces"
    )

    try:
        import openai

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        @noveum_trace.trace(name="error_prone_function")
        def error_prone_function(should_fail: bool) -> str:
            if should_fail:
                raise ValueError("Intentional test error")

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=20,
            )
            return response.choices[0].message.content

        # Test successful case
        success_result = error_prone_function(False)
        print(f"✅ Success case: {success_result}")

        # Test error case
        try:
            error_prone_function(True)
        except ValueError as e:
            print(f"✅ Error case handled: {e}")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_performance_monitoring():
    """Test performance monitoring and metrics."""
    print("\n📊 Testing Performance Monitoring...")

    # Initialize tracer
    noveum_trace.init(
        project_id="performance_test", file_logging=True, log_directory="test_traces"
    )

    try:
        import openai

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        @noveum_trace.trace(name="performance_test_function")
        def timed_function(delay: float) -> str:
            time.sleep(delay)  # Simulate processing time

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Count to 3"}],
                max_tokens=20,
            )
            return response.choices[0].message.content

        # Test with different delays
        delays = [0.1, 0.2, 0.3]
        for delay in delays:
            result = timed_function(delay)
            print(f"✅ Function with {delay}s delay: {result}")

        return True

    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_nested_trace_contexts():
    """Test deeply nested trace contexts."""
    print("\n🪆 Testing Nested Trace Contexts...")

    # Initialize tracer
    noveum_trace.init(
        project_id="nested_test", file_logging=True, log_directory="test_traces"
    )

    try:
        import openai

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        @noveum_trace.trace(name="level_1")
        def level_1_function(data: str) -> str:

            @noveum_trace.trace(name="level_2")
            def level_2_function(processed_data: str) -> str:

                @noveum_trace.trace(name="level_3_llm_call")
                def level_3_llm_call(final_data: str) -> str:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "user",
                                "content": f"Process this data: {final_data}",
                            }
                        ],
                        max_tokens=30,
                    )
                    return response.choices[0].message.content

                return level_3_llm_call(f"Level 2 processed: {processed_data}")

            return level_2_function(f"Level 1 processed: {data}")

        result = level_1_function("Initial data")
        print(f"✅ Nested trace result: {result}")

        return True

    except Exception as e:
        print(f"❌ Nested trace contexts test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_langchain_with_agents():
    """Test LangChain integration with Noveum agents."""
    print("\n🦜🤖 Testing LangChain with Noveum Agents...")

    # Initialize tracer
    noveum_trace.init(
        project_id="langchain_agent_test",
        file_logging=True,
        log_directory="test_traces",
    )

    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        # Create agent
        agent_config = noveum_trace.AgentConfig(
            name="LangChain_Agent",
            description="Agent using LangChain with OpenAI",
            version="1.0.0",
            capabilities={"langchain", "openai"},
            metadata={"framework": "langchain", "provider": "openai"},
        )

        registry = noveum_trace.get_agent_registry()
        agent = registry.register_agent(agent_config)

        with AgentContext(agent):

            @noveum_trace.trace(name="langchain_processing")
            def process_with_langchain(query: str) -> str:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                )

                response = llm.invoke([HumanMessage(content=query)])
                return response.content

            result = process_with_langchain(
                "What is the significance of quantum computing?"
            )
            print(f"✅ LangChain + Agent result: {result[:100]}...")

        return True

    except ImportError as e:
        print(f"⚠️ LangChain not available: {e}")
        return False
    except Exception as e:
        print(f"❌ LangChain with agents test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()


def test_all_sink_types():
    """Test different sink types."""
    print("\n🗂️ Testing Different Sink Types...")

    try:
        from noveum_trace.sinks.console import ConsoleSink, ConsoleSinkConfig
        from noveum_trace.sinks.file import FileSink, FileSinkConfig

        # Test file sink
        file_config = FileSinkConfig(
            name="test-file-sink",
            directory="test_traces",
            file_format="jsonl",
            max_file_size_mb=10,
        )

        file_sink = FileSink(file_config)
        print(f"✅ File sink created: {file_sink}")

        # Test console sink
        console_config = ConsoleSinkConfig(name="test-console-sink", pretty_print=True)

        console_sink = ConsoleSink(console_config)
        print(f"✅ Console sink created: {console_sink}")

        return True

    except Exception as e:
        print(f"❌ Sink types test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Comprehensive Integration Tests...")
    print("=" * 60)

    results = []
    results.append(test_multi_provider_scenario())
    results.append(test_concurrent_tracing())
    results.append(test_error_handling())
    results.append(test_performance_monitoring())
    results.append(test_nested_trace_contexts())
    results.append(test_langchain_with_agents())
    results.append(test_all_sink_types())

    print("\n" + "=" * 60)
    print(f"📊 Final Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("🎉 All comprehensive tests passed!")
        print(
            "✅ Noveum Trace is working correctly across all providers and frameworks!"
        )
    else:
        print("❌ Some comprehensive tests failed")
        failed_tests = len(results) - sum(results)
        print(f"❌ {failed_tests} test(s) failed")
        sys.exit(1)
