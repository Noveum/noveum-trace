"""
Complete example demonstrating the Noveum Trace SDK.

This example shows:
- Setting up multiple sinks (Elasticsearch and Noveum.ai)
- Manual span creation and management
- Decorator-based automatic instrumentation
- LLM operation tracing with streaming support
- Error handling and exception recording
- Context propagation and span relationships
- Performance monitoring and metrics
"""

import asyncio
import time
import random
from typing import List, Dict, Any, AsyncGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the Noveum Trace SDK
from noveum_trace import (
    # Core components
    NoveumTracer, TracerConfig,
    # Sinks
    ElasticsearchSink, ElasticsearchConfig,
    NoveumSink, NoveumConfig,
    # Decorators
    trace_function, trace_llm_call, trace_streaming_llm_call,
    # Types
    SpanKind, OperationType, AISystem,
    LLMRequest, LLMResponse, TokenUsage, ModelParameters, Message
)


def setup_comprehensive_tracing():
    """Set up comprehensive tracing with multiple sinks."""
    
    # Configure Elasticsearch sink
    es_config = ElasticsearchConfig(
        name="elasticsearch-sink",
        hosts=["localhost:9200"],
        index_prefix="noveum-demo-traces",
        username="elastic",  # Optional
        password="password",  # Optional
        create_index_template=True,
        bulk_chunk_size=100,
        use_ssl=False,  # Set to True for production
        verify_certs=True,
    )
    es_sink = ElasticsearchSink(es_config)
    
    # Configure Noveum.ai sink
    noveum_config = NoveumConfig(
        name="noveum-sink",
        api_key="nv_demo_key_12345",  # Replace with real API key
        project_id="demo-project",
        enable_real_time_evaluation=True,
        enable_dataset_creation=True,
        evaluation_models=["gpt-4", "claude-3"],
        evaluation_metrics=["relevance", "accuracy", "safety", "coherence"],
        anonymize_pii=True,
        content_filtering=True,
    )
    noveum_sink = NoveumSink(noveum_config)
    
    # Configure tracer with comprehensive settings
    tracer_config = TracerConfig(
        service_name="noveum-demo-app",
        service_version="1.0.0",
        environment="development",
        
        # Sampling configuration
        sampling_rate=1.0,  # 100% sampling for demo
        max_spans_per_trace=1000,
        
        # Performance configuration
        max_queue_size=5000,
        batch_size=50,
        batch_timeout_ms=3000,
        max_export_timeout_ms=30000,
        
        # Resource configuration
        max_attribute_length=2048,
        max_event_count_per_span=64,
        max_link_count_per_span=32,
        
        # Feature flags
        enable_auto_instrumentation=True,
        enable_context_propagation=True,
        capture_llm_content=True,  # Enable for demo purposes
        
        # Sinks
        sinks=[es_sink, noveum_sink],
    )
    
    return NoveumTracer(tracer_config)


# Example 1: User Authentication Service
@trace_function(
    name="authenticate_user",
    kind=SpanKind.INTERNAL,
    capture_args=True,
    capture_result=True
)
def authenticate_user(username: str, password: str, ip_address: str) -> Dict[str, Any]:
    """Authenticate a user with comprehensive tracing."""
    logger.info(f"Authenticating user: {username} from {ip_address}")
    
    # Simulate authentication delay
    time.sleep(random.uniform(0.05, 0.15))
    
    # Simulate authentication logic
    if username == "admin" and password == "secret123":
        return {
            "user_id": "user_12345",
            "username": username,
            "role": "administrator",
            "authenticated": True,
            "session_token": f"token_{random.randint(100000, 999999)}",
            "permissions": ["read", "write", "admin"],
            "last_login": time.time(),
        }
    elif username == "user" and password == "password":
        return {
            "user_id": "user_67890",
            "username": username,
            "role": "user",
            "authenticated": True,
            "session_token": f"token_{random.randint(100000, 999999)}",
            "permissions": ["read"],
            "last_login": time.time(),
        }
    else:
        return {
            "authenticated": False,
            "error": "Invalid credentials",
            "error_code": "AUTH_FAILED",
            "retry_after": 5,
        }


# Example 2: LLM Chat Completion Service
@trace_llm_call(
    operation_type=OperationType.CHAT,
    ai_system=AISystem.OPENAI,
    model="gpt-4",
    capture_content=True
)
def chat_completion(
    messages: List[Dict[str, str]], 
    temperature: float = 0.7,
    max_tokens: int = 150,
    model: str = "gpt-4"
) -> Dict[str, Any]:
    """Simulate OpenAI chat completion with comprehensive tracing."""
    logger.info(f"Processing chat completion with {len(messages)} messages")
    
    # Simulate API call latency with realistic variation
    base_latency = 0.5
    token_latency = max_tokens * 0.002  # 2ms per token
    network_jitter = random.uniform(-0.1, 0.2)
    total_latency = base_latency + token_latency + network_jitter
    
    time.sleep(total_latency)
    
    # Simulate different response types based on input
    last_message = messages[-1]["content"].lower() if messages else ""
    
    if "error" in last_message or "fail" in last_message:
        # Simulate API error
        raise Exception("Simulated OpenAI API error: Rate limit exceeded")
    
    # Generate realistic response
    if "code" in last_message or "python" in last_message:
        response_content = """Here's a Python example:

```python
def hello_world():
    print("Hello, World!")
    return "success"

hello_world()
```

This function demonstrates basic Python syntax and function definition."""
    elif "explain" in last_message:
        response_content = "I'd be happy to explain that concept. Let me break it down into key components and provide a clear, step-by-step explanation with relevant examples."
    else:
        response_content = "Thank you for your question. I understand what you're asking and I'll provide a helpful and accurate response based on the information you've provided."
    
    # Calculate realistic token usage
    input_tokens = sum(len(msg["content"].split()) for msg in messages) * 1.3  # Rough tokenization
    output_tokens = len(response_content.split()) * 1.3
    total_tokens = input_tokens + output_tokens
    
    return {
        "id": f"chatcmpl-{random.randint(100000, 999999)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": int(input_tokens),
            "completion_tokens": int(output_tokens),
            "total_tokens": int(total_tokens)
        }
    }


# Example 3: Streaming LLM Completion
@trace_streaming_llm_call(
    operation_type=OperationType.COMPLETION,
    ai_system=AISystem.OPENAI,
    model="gpt-4"
)
def streaming_completion(prompt: str, model: str = "gpt-4") -> AsyncGenerator[Dict[str, Any], None]:
    """Simulate streaming completion with realistic timing."""
    logger.info(f"Starting streaming completion for prompt: {prompt[:50]}...")
    
    # Generate response in chunks
    full_response = "The answer to your question involves several important considerations. First, we need to understand the fundamental principles at play. These principles guide how we approach the problem and determine the most effective solution. Let me walk you through each step of the process in detail."
    
    words = full_response.split()
    chunk_size = random.randint(1, 3)  # Variable chunk sizes
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_content = " " + " ".join(chunk_words) if i > 0 else " ".join(chunk_words)
        
        # Simulate realistic streaming delays
        if i == 0:
            time.sleep(0.2)  # Initial delay (TTFB)
        else:
            time.sleep(random.uniform(0.05, 0.15))  # Inter-token delay
        
        yield {
            "id": f"chatcmpl-stream-{random.randint(100000, 999999)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk_content},
                "finish_reason": None if i + chunk_size < len(words) else "stop"
            }]
        }


# Example 4: Complex Business Logic with Nested Spans
@trace_function(
    name="process_user_request",
    kind=SpanKind.SERVER,
    capture_args=True
)
def process_user_request(
    user_id: str, 
    request_type: str, 
    payload: Dict[str, Any],
    session_token: str
) -> Dict[str, Any]:
    """Process a complex user request with multiple nested operations."""
    logger.info(f"Processing {request_type} request for user {user_id}")
    
    # Step 1: Validate session
    session_valid = validate_session(session_token, user_id)
    if not session_valid:
        return {"error": "Invalid session", "status": 401}
    
    # Step 2: Rate limiting check
    if not check_rate_limit(user_id):
        return {"error": "Rate limit exceeded", "status": 429, "retry_after": 60}
    
    # Step 3: Process based on request type
    try:
        if request_type == "chat":
            result = handle_chat_request(payload, user_id)
        elif request_type == "analysis":
            result = handle_analysis_request(payload, user_id)
        elif request_type == "streaming":
            result = handle_streaming_request(payload, user_id)
        else:
            return {"error": "Unknown request type", "status": 400}
        
        # Step 4: Log successful request
        log_request_success(user_id, request_type, result)
        
        return {
            "status": 200,
            "result": result,
            "request_id": f"req_{random.randint(100000, 999999)}",
            "processing_time_ms": random.randint(100, 500),
            "timestamp": time.time(),
        }
        
    except Exception as e:
        # Step 5: Handle and log errors
        log_request_error(user_id, request_type, str(e))
        return {
            "error": "Internal server error",
            "status": 500,
            "error_id": f"err_{random.randint(100000, 999999)}",
            "timestamp": time.time(),
        }


@trace_function(name="validate_session")
def validate_session(session_token: str, user_id: str) -> bool:
    """Validate user session token."""
    time.sleep(0.02)  # Simulate database lookup
    
    # Simple validation logic
    return session_token.startswith("token_") and len(session_token) > 10


@trace_function(name="check_rate_limit")
def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limits."""
    time.sleep(0.01)  # Simulate cache lookup
    
    # Simulate rate limiting (allow most requests)
    return random.random() > 0.05  # 5% chance of rate limiting


@trace_function(name="handle_chat_request")
def handle_chat_request(payload: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Handle chat request with LLM integration."""
    messages = payload.get("messages", [])
    temperature = payload.get("temperature", 0.7)
    model = payload.get("model", "gpt-4")
    
    # Add system message for context
    system_message = {
        "role": "system",
        "content": f"You are a helpful assistant for user {user_id}. Be concise and helpful."
    }
    full_messages = [system_message] + messages
    
    # Call LLM
    llm_response = chat_completion(full_messages, temperature=temperature, model=model)
    
    return {
        "type": "chat_response",
        "response": llm_response,
        "model_used": model,
        "message_count": len(messages),
    }


@trace_function(name="handle_analysis_request")
def handle_analysis_request(payload: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Handle data analysis request."""
    data = payload.get("data", [])
    analysis_type = payload.get("analysis_type", "basic")
    
    # Simulate analysis processing
    time.sleep(random.uniform(0.1, 0.3))
    
    # Generate analysis results
    analysis_result = {
        "analysis_type": analysis_type,
        "data_points": len(data),
        "summary_stats": {
            "count": len(data),
            "avg_length": sum(len(str(item)) for item in data) / len(data) if data else 0,
            "unique_items": len(set(str(item) for item in data)),
        },
        "insights": [
            "Data distribution appears normal",
            "No significant outliers detected",
            "Quality score: 85/100"
        ],
        "recommendations": [
            "Consider additional data validation",
            "Monitor for seasonal patterns",
            "Implement automated quality checks"
        ]
    }
    
    return {
        "type": "analysis_response",
        "analysis": analysis_result,
        "processing_time_ms": random.randint(100, 300),
    }


@trace_function(name="handle_streaming_request")
def handle_streaming_request(payload: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Handle streaming request."""
    prompt = payload.get("prompt", "")
    model = payload.get("model", "gpt-4")
    
    # Collect streaming response
    full_response = ""
    chunk_count = 0
    
    for chunk in streaming_completion(prompt, model=model):
        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if content:
            full_response += content
            chunk_count += 1
    
    return {
        "type": "streaming_response",
        "full_response": full_response,
        "chunk_count": chunk_count,
        "model_used": model,
    }


@trace_function(name="log_request_success")
def log_request_success(user_id: str, request_type: str, result: Dict[str, Any]) -> None:
    """Log successful request for analytics."""
    time.sleep(0.005)  # Simulate logging delay
    logger.info(f"Request success: {user_id} - {request_type}")


@trace_function(name="log_request_error")
def log_request_error(user_id: str, request_type: str, error: str) -> None:
    """Log request error for monitoring."""
    time.sleep(0.005)  # Simulate logging delay
    logger.error(f"Request error: {user_id} - {request_type} - {error}")


# Example 5: Error Handling and Recovery
@trace_function(name="resilient_operation", capture_exceptions=True)
def resilient_operation(operation_type: str, retry_count: int = 3) -> Dict[str, Any]:
    """Demonstrate error handling and retry logic with tracing."""
    
    for attempt in range(retry_count):
        try:
            # Simulate operation that might fail
            if random.random() < 0.3:  # 30% chance of failure
                raise Exception(f"Simulated {operation_type} failure on attempt {attempt + 1}")
            
            # Simulate successful operation
            time.sleep(random.uniform(0.1, 0.2))
            
            return {
                "operation": operation_type,
                "status": "success",
                "attempt": attempt + 1,
                "result": f"Operation {operation_type} completed successfully"
            }
            
        except Exception as e:
            if attempt == retry_count - 1:  # Last attempt
                raise  # Re-raise the exception
            else:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff


# Example 6: Async Operations
@trace_function(name="async_data_processing")
async def async_data_processing(data_batch: List[str]) -> Dict[str, Any]:
    """Demonstrate async operation tracing."""
    logger.info(f"Processing {len(data_batch)} items asynchronously")
    
    # Simulate async processing
    await asyncio.sleep(0.2)
    
    # Process each item
    results = []
    for item in data_batch:
        # Simulate item processing
        await asyncio.sleep(0.01)
        processed_item = {
            "original": item,
            "processed": item.upper(),
            "length": len(item),
            "timestamp": time.time()
        }
        results.append(processed_item)
    
    return {
        "processed_count": len(results),
        "results": results,
        "total_length": sum(len(item) for item in data_batch),
        "processing_time_ms": 200 + len(data_batch) * 10,
    }


async def run_async_examples(tracer):
    """Run async examples with proper tracing."""
    logger.info("Running async examples...")
    
    # Async data processing
    test_data = ["item1", "item2", "item3", "longer_item_name", "test"]
    result = await async_data_processing(test_data)
    logger.info(f"Async processing completed: {result['processed_count']} items")


def run_comprehensive_demo():
    """Run a comprehensive demonstration of the SDK."""
    logger.info("Starting Noveum Trace SDK Comprehensive Demo")
    logger.info("=" * 60)
    
    # Set up tracing
    tracer = setup_comprehensive_tracing()
    
    try:
        # Example 1: User Authentication Flow
        logger.info("\n1. Testing User Authentication Flow")
        auth_results = []
        
        # Test successful authentication
        auth_result = authenticate_user("admin", "secret123", "192.168.1.100")
        auth_results.append(auth_result)
        logger.info(f"Admin auth: {auth_result['authenticated']}")
        
        # Test user authentication
        auth_result = authenticate_user("user", "password", "192.168.1.101")
        auth_results.append(auth_result)
        logger.info(f"User auth: {auth_result['authenticated']}")
        
        # Test failed authentication
        auth_result = authenticate_user("hacker", "wrongpass", "192.168.1.102")
        auth_results.append(auth_result)
        logger.info(f"Failed auth: {auth_result['authenticated']}")
        
        # Example 2: LLM Chat Completions
        logger.info("\n2. Testing LLM Chat Completions")
        
        chat_messages = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "user", "content": "Explain Python functions with code examples"},
            {"role": "user", "content": "How does error handling work in programming?"}
        ]
        
        for i, message in enumerate(chat_messages):
            try:
                response = chat_completion([message], temperature=0.7)
                logger.info(f"Chat {i+1}: {response['usage']['total_tokens']} tokens")
            except Exception as e:
                logger.error(f"Chat {i+1} failed: {e}")
        
        # Example 3: Complex Business Logic
        logger.info("\n3. Testing Complex Business Logic")
        
        # Successful chat request
        chat_payload = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "temperature": 0.8,
            "model": "gpt-4"
        }
        
        result = process_user_request(
            user_id="user_12345",
            request_type="chat",
            payload=chat_payload,
            session_token="token_123456"
        )
        logger.info(f"Chat request result: {result['status']}")
        
        # Analysis request
        analysis_payload = {
            "data": ["sample1", "sample2", "sample3", "sample4"],
            "analysis_type": "comprehensive"
        }
        
        result = process_user_request(
            user_id="user_12345",
            request_type="analysis",
            payload=analysis_payload,
            session_token="token_123456"
        )
        logger.info(f"Analysis request result: {result['status']}")
        
        # Streaming request
        streaming_payload = {
            "prompt": "Explain the benefits of distributed tracing",
            "model": "gpt-4"
        }
        
        result = process_user_request(
            user_id="user_12345",
            request_type="streaming",
            payload=streaming_payload,
            session_token="token_123456"
        )
        logger.info(f"Streaming request result: {result['status']}")
        
        # Example 4: Error Handling and Resilience
        logger.info("\n4. Testing Error Handling and Resilience")
        
        operations = ["data_sync", "cache_refresh", "model_update"]
        for operation in operations:
            try:
                result = resilient_operation(operation, retry_count=3)
                logger.info(f"{operation}: {result['status']} (attempt {result['attempt']})")
            except Exception as e:
                logger.error(f"{operation}: Failed after all retries - {e}")
        
        # Example 5: Async Operations
        logger.info("\n5. Testing Async Operations")
        asyncio.run(run_async_examples(tracer))
        
        # Example 6: Sink Health Monitoring
        logger.info("\n6. Checking Sink Health")
        for sink in tracer.config.sinks:
            health_status = sink.health_check()
            metrics = sink.metrics
            logger.info(f"Sink '{sink.name}': {'Healthy' if health_status else 'Unhealthy'}")
            logger.info(f"  - Success rate: {metrics.success_rate:.1f}%")
            logger.info(f"  - Spans sent: {metrics.spans_sent}")
            logger.info(f"  - Average latency: {metrics.average_latency_ms:.2f}ms")
        
        logger.info("\n" + "=" * 60)
        logger.info("Demo completed successfully!")
        logger.info("Check your Elasticsearch and Noveum.ai dashboards for trace data.")
        
        # Flush any remaining spans
        logger.info("\nFlushing remaining spans...")
        flush_success = tracer.flush(timeout_ms=10000)
        logger.info(f"Flush {'successful' if flush_success else 'timed out'}")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
    
    finally:
        # Always shutdown the tracer
        logger.info("Shutting down tracer...")
        shutdown_success = tracer.shutdown(timeout_ms=15000)
        logger.info(f"Shutdown {'successful' if shutdown_success else 'timed out'}")


if __name__ == "__main__":
    run_comprehensive_demo()

