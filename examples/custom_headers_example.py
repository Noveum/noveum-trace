"""
Example demonstrating custom headers and project-based configuration.

This example shows how to use the updated Noveum Trace SDK with:
- Project-based configuration (instead of service-based)
- Custom headers for projectId, orgId, and other metadata
- File-based logging with custom headers included
"""

import json
import time

import noveum_trace


def main():
    print("🚀 Testing Noveum Trace SDK with Custom Headers and Project Configuration")

    # Initialize with project-based configuration and custom headers
    tracer = noveum_trace.init(
        project_name="my-ai-chatbot",
        project_id="proj_12345",
        org_id="org_67890",
        user_id="user_abc123",
        session_id="session_xyz789",
        environment="development",
        file_logging=True,
        log_directory="./project_traces",
        custom_headers={
            "x-deployment-id": "deploy_v1.2.3",
            "x-region": "us-west-2",
            "x-team": "ai-platform",
        },
    )

    print(f"✅ Tracer initialized for project: {tracer.config.project_name}")
    print("📁 Logs will be saved to: ./project_traces")

    # Test basic tracing with project context
    with tracer.start_span("chatbot_conversation") as conversation_span:
        conversation_span.set_attribute("conversation.type", "customer_support")
        conversation_span.set_attribute("conversation.language", "en")

        # Simulate user message processing
        with tracer.start_span("process_user_message") as message_span:
            message_span.set_attribute(
                "message.content", "Hello, I need help with my account"
            )
            message_span.set_attribute("message.intent", "account_support")
            time.sleep(0.1)  # Simulate processing time

        # Simulate LLM response generation
        with tracer.start_span("generate_llm_response") as llm_span:
            llm_span.set_attribute("llm.model", "gpt-4")
            llm_span.set_attribute("llm.provider", "openai")
            llm_span.set_attribute("llm.tokens.input", 25)
            llm_span.set_attribute("llm.tokens.output", 150)
            time.sleep(0.2)  # Simulate LLM call

            llm_span.set_attribute(
                "response.content",
                "I'd be happy to help you with your account. Can you please provide your account ID?",
            )

    # Test manual span creation with custom attributes
    with tracer.start_span("analytics_tracking") as analytics_span:
        analytics_span.set_attribute("analytics.event", "conversation_completed")
        analytics_span.set_attribute("analytics.satisfaction_score", 4.5)
        analytics_span.set_attribute("analytics.resolution_time_ms", 300)

    print("📊 Generated sample traces with project and custom header context")

    # Flush to ensure all traces are written
    tracer.flush()

    print("✅ All traces flushed to file sink")

    # Check the generated trace file
    import glob
    import os

    trace_files = glob.glob("./project_traces/*.jsonl")
    if trace_files:
        latest_file = max(trace_files, key=os.path.getctime)
        print(f"📄 Latest trace file: {latest_file}")

        # Read and display a sample trace
        with open(latest_file) as f:
            lines = f.readlines()
            if lines:
                sample_trace = json.loads(lines[0])
                print("🔍 Sample trace data:")
                print(
                    f"  - Project: {sample_trace.get('attributes', {}).get('noveum.project.name', 'N/A')}"
                )
                print(
                    f"  - Org ID: {sample_trace.get('attributes', {}).get('noveum.org.id', 'N/A')}"
                )
                print(
                    f"  - Project ID: {sample_trace.get('attributes', {}).get('noveum.project.id', 'N/A')}"
                )
                print(
                    f"  - User ID: {sample_trace.get('attributes', {}).get('noveum.user.id', 'N/A')}"
                )
                print(
                    f"  - Session ID: {sample_trace.get('attributes', {}).get('noveum.session.id', 'N/A')}"
                )
                print(
                    f"  - Custom Headers: {sample_trace.get('attributes', {}).get('x-deployment-id', 'N/A')}"
                )

    # Shutdown tracer
    noveum_trace.shutdown()
    print("🔄 Tracer shutdown complete")


if __name__ == "__main__":
    main()
