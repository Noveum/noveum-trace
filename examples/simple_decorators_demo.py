"""
Simple Decorators Demo

This example demonstrates the simplified decorator patterns in the Noveum Trace SDK,
showing how they compare to competitor SDKs like DeepEval, Phoenix, and Braintrust.

The demo includes:
- @trace decorator for simple function tracing
- @observe decorator for component-level observability
- @llm_trace decorator for LLM operation tracing
- update_current_span for dynamic span updates
"""

import asyncio
import time
from typing import Any, Dict, List

# Import Noveum Trace SDK with simplified decorators
import noveum_trace
from noveum_trace import (
    AgentConfig,
    AgentContext,
    get_agent_registry,
    get_current_agent,
    llm_trace,
    observe,
    trace,
    update_current_span,
)
from noveum_trace.sinks.console import ConsoleSink, ConsoleSinkConfig
from noveum_trace.types import CustomHeaders


# Mock LLM response for demonstration
class MockLLMResponse:
    def __init__(self, content: str, model: str = "gpt-4"):
        self.choices = [MockChoice(content)]
        self.usage = MockUsage()
        self.model = model
        self.id = f"mock-{int(time.time())}"


class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)
        self.finish_reason = "stop"


class MockMessage:
    def __init__(self, content: str):
        self.content = content


class MockUsage:
    def __init__(self):
        self.prompt_tokens = 25
        self.completion_tokens = 75
        self.total_tokens = 100


# ============================================================================
# 1. Basic @trace Decorator Examples
# ============================================================================


@trace
def simple_function(data: str) -> str:
    """Simple function with basic tracing."""
    # Function is automatically traced with minimal overhead
    time.sleep(0.01)  # Simulate work
    return f"Processed: {data}"


@trace(name="custom_operation", capture_args=True, capture_result=True)
def advanced_function(input_data: Dict, config: Dict) -> Dict:
    """Function with advanced tracing configuration."""
    # Custom span name and argument/result capture
    time.sleep(0.02)  # Simulate work

    result = {
        "status": "completed",
        "input_size": len(str(input_data)),
        "config_applied": config.get("mode", "default"),
    }

    return result


@trace(kind=noveum_trace.SpanKind.CLIENT)
async def async_api_call(endpoint: str) -> Dict:
    """Async function with CLIENT span kind."""
    # Simulate API call
    await asyncio.sleep(0.05)

    return {"endpoint": endpoint, "status_code": 200, "response_time_ms": 50}


# ============================================================================
# 2. @observe Decorator Examples (DeepEval-inspired)
# ============================================================================


@observe(metrics=["accuracy", "latency"])
def data_processor(data: List[str]) -> List[str]:
    """Component with observability metrics."""
    # Process data with metrics collection
    start_time = time.time()

    processed_data = []
    for item in data:
        # Simulate processing
        time.sleep(0.001)
        processed_item = item.upper()
        processed_data.append(processed_item)

    end_time = time.time()

    # Update span with component-specific information
    update_current_span(
        input=data,
        output=processed_data,
        metadata={
            "items_processed": len(data),
            "processing_time_ms": (end_time - start_time) * 1000,
            "accuracy_score": 0.95,  # Mock accuracy
        },
    )

    return processed_data


@observe(
    name="llm_component",
    metrics=["response_quality", "token_efficiency"],
    capture_input=True,
    capture_output=True,
)
def llm_component(prompt: str, model: str = "gpt-4") -> str:
    """LLM component with detailed observability."""
    # Simulate LLM call
    time.sleep(0.1)

    response = f"AI Response to: {prompt}"

    # Update span with LLM-specific metrics
    update_current_span(
        metadata={
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "temperature": 0.7,
            "quality_score": 0.88,  # Mock quality score
        }
    )

    return response


@observe(name="validation_component")
async def async_validator(data: Dict) -> bool:
    """Async component with validation logic."""
    # Simulate async validation
    await asyncio.sleep(0.02)

    is_valid = all(key in data for key in ["id", "name", "type"])

    update_current_span(
        input=data,
        output=is_valid,
        metadata={
            "validation_rules": ["id_required", "name_required", "type_required"],
            "validation_result": "passed" if is_valid else "failed",
            "data_keys": list(data.keys()),
        },
    )

    return is_valid


# ============================================================================
# 3. @llm_trace Decorator Examples
# ============================================================================


@llm_trace(model="gpt-4", operation="chat", ai_system="openai")
def chat_completion(messages: List[Dict]) -> MockLLMResponse:
    """Chat completion with automatic LLM tracing."""
    # Simulate OpenAI chat completion
    time.sleep(0.08)

    last_message = messages[-1]["content"] if messages else ""
    response_content = (
        f"AI: I understand you said '{last_message}'. Here's my response."
    )

    return MockLLMResponse(response_content)


@llm_trace(model="text-embedding-ada-002", operation="embedding")
async def create_embedding(text: str) -> List[float]:
    """Embedding creation with LLM tracing."""
    # Simulate embedding API call
    await asyncio.sleep(0.1)
    print(f"Creating embedding for: {text[:50]}...")

    # Return simulated embedding vector
    return [0.1, 0.2, 0.3] * 128  # 384-dimensional vector


@llm_trace(
    name="custom_llm_operation",
    model="claude-3",
    operation="completion",
    ai_system="anthropic",
)
def custom_llm_call(prompt: str, max_tokens: int = 100) -> MockLLMResponse:
    """Custom LLM operation with full configuration."""
    # Simulate Anthropic API call
    time.sleep(0.06)

    response_content = (
        f"Claude: {prompt[:50]}... [Generated response with {max_tokens} max tokens]"
    )

    return MockLLMResponse(response_content, model="claude-3")


# ============================================================================
# 4. Complex Workflow Examples
# ============================================================================


@trace(name="document_processing_workflow")
def process_document(document: Dict) -> Dict:
    """Complex workflow combining multiple decorated functions."""
    update_current_span(
        input=document,
        metadata={
            "workflow_type": "document_processing",
            "document_type": document.get("type", "unknown"),
        },
    )

    # Step 1: Validate document
    is_valid = validate_document(document)
    if not is_valid:
        raise ValueError("Invalid document")

    # Step 2: Extract text
    text_content = extract_text(document)

    # Step 3: Process with LLM
    summary = summarize_text(text_content)

    # Step 4: Generate final result
    result = {
        "document_id": document.get("id"),
        "validation_status": "passed",
        "text_length": len(text_content),
        "summary": summary,
        "processed_at": time.time(),
    }

    update_current_span(
        output=result, metadata={"processing_completed": True, "steps_executed": 4}
    )

    return result


@observe(name="document_validator")
def validate_document(document: Dict) -> bool:
    """Document validation component."""
    required_fields = ["id", "type", "content"]
    is_valid = all(field in document for field in required_fields)

    update_current_span(
        input=document,
        output=is_valid,
        metadata={
            "required_fields": required_fields,
            "present_fields": list(document.keys()),
            "validation_result": "passed" if is_valid else "failed",
        },
    )

    return is_valid


@trace(name="text_extraction")
def extract_text(document: Dict) -> str:
    """Text extraction from document."""
    # Simulate text extraction
    time.sleep(0.01)

    content = document.get("content", "")
    extracted_text = (
        f"Extracted text from {document.get('type', 'unknown')} document: {content}"
    )

    update_current_span(
        metadata={
            "extraction_method": "mock_extractor",
            "original_length": len(content),
            "extracted_length": len(extracted_text),
        }
    )

    return extracted_text


@llm_trace(model="gpt-4", operation="chat")
def summarize_text(text: str) -> str:
    """Text summarization using LLM."""
    # Simulate LLM summarization
    time.sleep(0.05)

    # Mock summary
    summary = f"Summary: {text[:100]}..." if len(text) > 100 else f"Summary: {text}"

    return summary


# ============================================================================
# 5. Agent-Aware Workflow
# ============================================================================


@trace(name="multi_agent_workflow")
async def run_multi_agent_workflow(task: str) -> Dict:
    """Workflow that uses multiple agents."""
    update_current_span(input=task, metadata={"workflow_type": "multi_agent"})

    results = {}

    # Get agent registry
    registry = get_agent_registry()

    # Create agents if they don't exist
    if "processor" not in registry:
        processor_config = AgentConfig(
            name="processor",
            agent_type="data_processor",
            capabilities={"data_processing"},
        )
        registry.register_agent(processor_config)

    if "analyzer" not in registry:
        analyzer_config = AgentConfig(
            name="analyzer", agent_type="data_analyzer", capabilities={"data_analysis"}
        )
        registry.register_agent(analyzer_config)

    # Process with first agent
    processor_agent = registry.get_agent("processor")
    with AgentContext(processor_agent):
        processed_data = await process_with_agent(task, "processing")
        results["processed"] = processed_data

    # Analyze with second agent
    analyzer_agent = registry.get_agent("analyzer")
    with AgentContext(analyzer_agent):
        analysis_result = await process_with_agent(processed_data, "analysis")
        results["analyzed"] = analysis_result

    update_current_span(
        output=results,
        metadata={"agents_used": ["processor", "analyzer"], "workflow_completed": True},
    )

    return results


@observe(name="agent_processing")
async def process_with_agent(data: Any, operation_type: str) -> str:
    """Process data with current agent context."""
    current_agent = get_current_agent()

    update_current_span(
        input=data,
        metadata={
            "operation_type": operation_type,
            "agent_name": current_agent.name if current_agent else "unknown",
            "agent_type": current_agent.agent_type if current_agent else "unknown",
        },
    )

    # Simulate processing
    await asyncio.sleep(0.02)

    result = f"{operation_type.title()} result for: {str(data)[:50]}"

    update_current_span(output=result, metadata={"processing_completed": True})

    return result


# ============================================================================
# Main Demo Function
# ============================================================================


async def main():
    """Main function demonstrating all decorator patterns."""

    # Initialize Noveum Trace SDK
    noveum_trace.init(
        project_name="decorators-demo",
        custom_headers=CustomHeaders(
            project_id="demo-project", org_id="noveum-examples"
        ),
        sinks=[
            ConsoleSink(ConsoleSinkConfig(format_json=True, include_timestamp=True))
        ],
    )

    print("🎯 Noveum Trace SDK - Simplified Decorators Demo")
    print("=" * 60)

    # 1. Basic @trace decorator examples
    print("\n1️⃣ Basic @trace Decorator Examples")
    print("-" * 40)

    result1 = simple_function("test data")
    print(f"Simple function result: {result1}")

    result2 = advanced_function(
        {"key": "value", "data": [1, 2, 3]}, {"mode": "advanced", "debug": True}
    )
    print(f"Advanced function result: {result2}")

    result3 = await async_api_call("/api/users")
    print(f"Async API call result: {result3}")

    # 2. @observe decorator examples
    print("\n2️⃣ @observe Decorator Examples (DeepEval-style)")
    print("-" * 50)

    data_list = ["hello", "world", "noveum", "trace"]
    processed_data = data_processor(data_list)
    print(f"Data processor result: {processed_data}")

    llm_result = llm_component("What is machine learning?")
    print(f"LLM component result: {llm_result}")

    validation_data = {"id": "123", "name": "Test", "type": "document"}
    is_valid = await async_validator(validation_data)
    print(f"Async validator result: {is_valid}")

    # 3. @llm_trace decorator examples
    print("\n3️⃣ @llm_trace Decorator Examples")
    print("-" * 40)

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    chat_result = chat_completion(messages)
    print(f"Chat completion: {chat_result.choices[0].message.content}")

    embedding_result = await create_embedding("Sample text for embedding")
    print(f"Embedding dimensions: {len(embedding_result)}")

    custom_result = custom_llm_call("Explain quantum computing", max_tokens=150)
    print(f"Custom LLM call: {custom_result.choices[0].message.content}")

    # 4. Complex workflow example
    print("\n4️⃣ Complex Workflow Example")
    print("-" * 35)

    document = {
        "id": "doc-123",
        "type": "article",
        "content": "This is a sample document content that needs to be processed and summarized.",
    }

    try:
        workflow_result = process_document(document)
        print(f"Document processing completed: {workflow_result['document_id']}")
        print(f"Summary: {workflow_result['summary']}")
    except Exception as e:
        print(f"Workflow error: {e}")

    # 5. Multi-agent workflow
    print("\n5️⃣ Multi-Agent Workflow Example")
    print("-" * 40)

    multi_agent_result = await run_multi_agent_workflow(
        "Analyze customer feedback data"
    )
    print(f"Multi-agent workflow result: {multi_agent_result}")

    # Display final statistics
    print("\n📊 Final Statistics")
    print("-" * 25)

    registry = get_agent_registry()
    if len(registry) > 0:
        stats = registry.get_registry_stats()
        print(f"Total agents: {stats['total_agents']}")
        print(f"Total traces: {stats['total_traces']}")
        print(f"Active traces: {stats['active_traces']}")

    # Shutdown
    print("\n🔄 Shutting down...")
    noveum_trace.shutdown()

    print("✅ Demo completed successfully!")
    print("\n💡 Key Takeaways:")
    print("   • @trace: Simple function tracing with minimal configuration")
    print("   • @observe: Component-level observability with metrics")
    print("   • @llm_trace: Specialized LLM operation tracing")
    print("   • update_current_span: Dynamic span updates during execution")
    print("   • Agent-aware: Automatic agent context resolution")


if __name__ == "__main__":
    asyncio.run(main())
