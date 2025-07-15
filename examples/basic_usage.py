"""
Basic usage example for the Noveum Trace SDK.
"""

import time
from noveum_trace import (
    NoveumTracer, TracerConfig,
    OperationType, AISystem
)
from noveum_trace.core.tracer import set_current_tracer
from noveum_trace.instrumentation.decorators import trace_function
from noveum_trace.sinks.base import BaseSink, SinkConfig


class ConsoleSink(BaseSink):
    """Simple console sink for demonstration."""
    
    def __init__(self):
        config = SinkConfig(name="console-sink")
        super().__init__(config)
    
    def _initialize(self) -> None:
        print("Console sink initialized")
    
    def _send_batch(self, spans) -> None:
        print(f"Console sink received {len(spans)} spans:")
        for span in spans:
            print(f"  - {span.name} ({span.span_id})")
    
    def _health_check(self) -> bool:
        return True


def main():
    """Run basic usage example."""
    print("Noveum Trace SDK - Basic Usage Example")
    print("=" * 50)
    
    # Create console sink
    console_sink = ConsoleSink()
    
    # Configure tracer
    config = TracerConfig(
        service_name="basic-example",
        service_version="1.0.0",
        environment="development",
        sinks=[console_sink],
        batch_size=1,  # Process immediately for demo
        batch_timeout_ms=100,
    )
    
    # Create tracer
    tracer = NoveumTracer(config)
    
    # Set as global tracer for decorators
    set_current_tracer(tracer)
    
    try:
        # Example 1: Manual span creation
        print("\n1. Manual span creation:")
        with tracer.start_span("user_request") as span:
            span.set_attribute("user.id", "user123")
            span.set_attribute("request.type", "api_call")
            
            # Simulate some work
            time.sleep(0.1)
            
            span.add_event("processing_started", {"step": "validation"})
            
            # Nested span
            with tracer.start_span("database_query") as db_span:
                db_span.set_attribute("db.operation", "SELECT")
                db_span.set_attribute("db.table", "users")
                time.sleep(0.05)
            
            span.add_event("processing_completed", {"result": "success"})
        
        # Example 2: Using decorators
        print("\n2. Using decorators:")
        
        @trace_function(name="process_data", capture_args=True)
        def process_data(data_type, count):
            time.sleep(0.05)
            return f"Processed {count} items of type {data_type}"
        
        def mock_llm_call(prompt):
            # Simulate LLM call
            time.sleep(0.1)
            return {
                "choices": [{"message": {"content": f"Response to: {prompt}"}}],
                "usage": {"total_tokens": 50}
            }
        
        # Call decorated functions
        result = process_data("documents", 10)
        print(f"Process result: {result}")
        
        llm_result = mock_llm_call("Hello, how are you?")
        print(f"LLM result: {llm_result['choices'][0]['message']['content']}")
        
        # Example 3: Error handling
        print("\n3. Error handling:")
        
        with tracer.start_span("error_example") as span:
            try:
                # Simulate an error
                raise ValueError("This is a test error")
            except ValueError as e:
                span.record_exception(e)
                print(f"Recorded exception: {e}")
        
        # Give time for background processing
        print("\n4. Waiting for span processing...")
        time.sleep(0.5)
        
        # Flush remaining spans
        print("\n5. Flushing spans...")
        tracer.flush(timeout_ms=2000)
        
        print("\nExample completed successfully!")
        
    finally:
        # Always shutdown the tracer
        print("\nShutting down tracer...")
        tracer.shutdown(timeout_ms=5000)


if __name__ == "__main__":
    main()

