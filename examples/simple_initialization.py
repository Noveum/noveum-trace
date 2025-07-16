"""
Simple initialization example demonstrating the easy-to-use API.

This example shows how simple it is to get started with Noveum Trace SDK,
matching the simplicity of competitors like DeepEval, Phoenix, and Braintrust.
"""

import os

import noveum_trace


def main():
    """Demonstrate simple initialization patterns."""
    print("Noveum Trace SDK - Simple Initialization Examples")
    print("=" * 55)

    # Example 1: Simplest possible initialization
    print("\n1. Simplest initialization (file logging only):")
    print("   noveum_trace.init()")

    tracer = noveum_trace.init()
    print(f"   ✅ Initialized with service: {tracer.config.service_name}")

    # Example 2: With service name
    print("\n2. With custom service name:")
    print("   noveum_trace.init(service_name='my-llm-app')")

    noveum_trace.shutdown()  # Clean up previous tracer
    tracer = noveum_trace.init(service_name="my-llm-app")
    print(f"   ✅ Initialized with service: {tracer.config.service_name}")

    # Example 3: Production configuration
    print("\n3. Production configuration:")
    print("   noveum_trace.init(")
    print("       api_key='your-api-key',")
    print("       project_id='your-project',")
    print("       environment='production',")
    print("       capture_content=False")
    print("   )")

    noveum_trace.shutdown()
    tracer = noveum_trace.init(
        # api_key="your-api-key",  # Would enable Noveum.ai sink
        # project_id="your-project",
        environment="production",
        capture_content=False,
        service_name="production-llm-service",
    )
    print(f"   ✅ Initialized for {tracer.config.environment} environment")

    # Example 4: Context manager pattern
    print("\n4. Context manager pattern:")
    print("   with noveum_trace.NoveumTrace(service_name='temp-service'):")
    print("       # Your code here")
    print("       pass")

    noveum_trace.shutdown()
    with noveum_trace.NoveumTrace(service_name="temp-service") as tracer:
        print(
            f"   ✅ Context manager active with service: {tracer.config.service_name}"
        )
    print("   ✅ Automatic cleanup completed")

    # Example 5: Environment variable configuration
    print("\n5. Environment variable configuration:")
    print("   export NOVEUM_API_KEY='your-api-key'")
    print("   export NOVEUM_PROJECT_ID='your-project'")
    print("   noveum_trace.init()  # Automatically uses env vars")

    # Set example env vars
    os.environ["NOVEUM_API_KEY"] = "demo-api-key"
    os.environ["NOVEUM_PROJECT_ID"] = "demo-project"

    tracer = noveum_trace.init(service_name="env-configured-service")
    print("   ✅ Configured from environment variables")

    # Clean up
    del os.environ["NOVEUM_API_KEY"]
    del os.environ["NOVEUM_PROJECT_ID"]

    # Example 6: Manual LLM call tracing (without auto-instrumentation)
    print("\n6. Manual tracing example:")

    @noveum_trace.trace_function(name="my_llm_function")
    def call_llm():
        """Example function that would call an LLM."""
        import time

        time.sleep(0.1)  # Simulate LLM call
        return "Hello from LLM!"

    result = call_llm()
    print(f"   ✅ Manual trace completed: {result}")

    # Flush and show stats
    print("\n7. Flushing traces...")
    noveum_trace.flush()

    # Get current tracer and show info
    current_tracer = noveum_trace.get_tracer()
    if current_tracer:
        print(f"   Service: {current_tracer.config.service_name}")
        print(f"   Environment: {current_tracer.config.environment}")
        print(f"   Sinks: {len(current_tracer.config.sinks)}")

    # Final cleanup
    print("\n8. Shutting down...")
    noveum_trace.shutdown()
    print("   ✅ Shutdown complete")

    print("\n🎉 All examples completed successfully!")
    print("\nNext steps:")
    print("- Check the ./traces directory for generated trace files")
    print("- Try the auto-instrumentation example with real LLM calls")
    print("- Explore the advanced configuration options")


if __name__ == "__main__":
    main()
