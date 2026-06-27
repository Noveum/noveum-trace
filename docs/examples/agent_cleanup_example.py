#!/usr/bin/env python3
"""
Agent Registry Cleanup Example for Noveum Trace SDK.

⚠️  NOTE: This example uses agent APIs that are currently not actively maintained.
For production use, please use context managers (trace_agent_operation) or
LangChain/LangGraph integrations instead.

This example demonstrates how to use the registry cleanup mechanisms
to prevent memory leaks in long-running applications.
"""

import os
import time

import noveum_trace
from noveum_trace import (
    cleanup_agent,
    cleanup_agent_graph,
    cleanup_agent_workflow,
    cleanup_all_registries,
    cleanup_by_ttl,
    create_agent,
    create_agent_graph,
    create_agent_workflow,
    enforce_size_limits,
    get_registry_stats,
    temporary_agent_context,
)

# Validate required environment variables
api_key = os.getenv("NOVEUM_API_KEY")
if not api_key:
    raise ValueError(
        "NOVEUM_API_KEY environment variable is required. "
        "Please set it before running this example."
    )


def main():
    """Main example function."""

    # Initialize the SDK
    noveum_trace.init(
        api_key=api_key,
        project="agent-cleanup-demo",
        environment="development",
    )

    print("🧹 Agent Registry Cleanup Demo")
    print("=" * 50)

    # Example 1: Basic registry usage and cleanup
    print("\n1. Creating agents and checking registry stats...")

    # Create some agents
    for i in range(5):
        create_agent(f"agent_{i}", agent_type="worker")

    # Create some graphs and workflows
    for i in range(3):
        create_agent_graph(f"graph_{i}", name=f"Graph {i}")
        create_agent_workflow(f"workflow_{i}", name=f"Workflow {i}")

    # Check registry stats
    stats = get_registry_stats()
    print(f"Registry stats: {stats}")

    # Example 2: Individual cleanup
    print("\n2. Individual cleanup...")

    removed = cleanup_agent("agent_0")
    print(f"Removed agent_0: {removed}")

    removed = cleanup_agent_graph("graph_0")
    print(f"Removed graph_0: {removed}")

    removed = cleanup_agent_workflow("workflow_0")
    print(f"Removed workflow_0: {removed}")

    stats = get_registry_stats()
    print(f"Registry stats after cleanup: {stats}")

    # Example 3: TTL-based cleanup
    print("\n3. TTL-based cleanup...")

    # Wait a moment, then clean up entries older than 1 second
    time.sleep(1.1)
    cleaned = cleanup_by_ttl(ttl_seconds=1.0)
    print(f"TTL cleanup results: {cleaned}")

    stats = get_registry_stats()
    print(f"Registry stats after TTL cleanup: {stats}")

    # Example 4: Size limit enforcement
    print("\n4. Size limit enforcement...")

    # Create many agents to test size limits
    for i in range(15):
        create_agent(f"bulk_agent_{i}", agent_type="bulk")

    print("Created 15 new agents")

    # Manually enforce size limits
    evicted = enforce_size_limits()
    print(f"Size enforcement results: {evicted}")

    stats = get_registry_stats()
    print(f"Registry stats after size enforcement: {stats}")

    # Example 5: Temporary context manager
    print("\n5. Temporary context manager...")

    stats_before = get_registry_stats()
    print(f"Stats before context: {stats_before}")

    with temporary_agent_context():
        # Create temporary agents
        _temp_agent = create_agent("temp_agent", agent_type="temporary")
        _temp_graph = create_agent_graph("temp_graph", name="Temporary Graph")
        _temp_workflow = create_agent_workflow(
            "temp_workflow", name="Temporary Workflow"
        )

        print("Created temporary agents, graphs, and workflows")

        stats_during = get_registry_stats()
        print(f"Stats during context: {stats_during}")

    # After exiting context, temporary items should be cleaned up
    stats_after = get_registry_stats()
    print(f"Stats after context (should be same as before): {stats_after}")

    # Example 6: Complete cleanup
    print("\n6. Complete cleanup...")

    cleared = cleanup_all_registries()
    print(f"Cleared all registries: {cleared}")

    final_stats = get_registry_stats()
    print(f"Final registry stats: {final_stats}")

    # Example 7: Configuration via environment variables
    print("\n7. Registry limits configuration...")
    print("Current limits:")
    print(f"  MAX_AGENTS: {os.getenv('NOVEUM_MAX_AGENTS', '1000')}")
    print(f"  MAX_AGENT_GRAPHS: {os.getenv('NOVEUM_MAX_AGENT_GRAPHS', '100')}")
    print(f"  MAX_AGENT_WORKFLOWS: {os.getenv('NOVEUM_MAX_AGENT_WORKFLOWS', '100')}")

    print(
        """
    💡 Tip: You can configure registry limits via environment variables:
    - NOVEUM_MAX_AGENTS=500
    - NOVEUM_MAX_AGENT_GRAPHS=50
    - NOVEUM_MAX_AGENT_WORKFLOWS=50
    """
    )

    # Flush any pending traces
    noveum_trace.flush()
    print("\n✅ All traces sent and registries cleaned!")


def demonstrate_production_usage():
    """Demonstrate how to use cleanup in production applications."""

    print("\n🏭 Production Usage Patterns")
    print("=" * 50)

    print(
        """
    1. **Periodic TTL Cleanup** (recommended for most applications):
       ```python
       import threading
       import time

       def periodic_cleanup():
           while True:
               time.sleep(300)  # Every 5 minutes
               cleanup_by_ttl(ttl_seconds=3600)  # 1 hour TTL

       cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
       cleanup_thread.start()
       ```

    2. **Size-based Cleanup** (for memory-constrained environments):
       ```python
       # Set strict limits via environment variables
       os.environ['NOVEUM_MAX_AGENTS'] = '100'
       os.environ['NOVEUM_MAX_AGENT_GRAPHS'] = '10'

       # Size limits are automatically enforced on creation
       ```

    3. **Context-based Cleanup** (for temporary operations):
       ```python
       with temporary_agent_context():
           # Create agents for a specific operation
           agent = create_agent("task_agent")
           # Automatically cleaned up when exiting context
       ```

    4. **Manual Cleanup** (for specific use cases):
       ```python
       # Clean up specific agents when done
       cleanup_agent("finished_agent_id")

       # Or clean up all registries during shutdown
       cleanup_all_registries()
       ```

    5. **Health Monitoring**:
       ```python
       stats = get_registry_stats()
       if stats['agents']['utilization'] > 80:
           # Take action: cleanup, alerting, etc.
           cleanup_by_ttl(ttl_seconds=1800)  # More aggressive cleanup
       ```
    """
    )


if __name__ == "__main__":
    try:
        main()
        demonstrate_production_usage()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Always clean up in finally block
        cleanup_all_registries()
        print("🧹 Final cleanup completed")
