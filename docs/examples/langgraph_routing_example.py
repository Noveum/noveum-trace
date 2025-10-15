"""
LangGraph Routing Decision Tracking Example

This example demonstrates how to use the new routing decision tracking
feature in Noveum Trace to capture routing decisions as separate spans.
"""

import os
from typing import Literal, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

# Import Noveum Trace
from noveum_trace.integrations import NoveumTraceCallbackHandler

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# STATE DEFINITION
# =============================================================================

class CounterState(TypedDict):
    """Simple counter state for demonstration."""
    count: int
    max_count: int
    status: str

# =============================================================================
# ROUTING FUNCTION WITH CALLBACK TRACKING
# =============================================================================

def route_counter(state: CounterState, config: RunnableConfig) -> Literal["increment", "finish"]:
    """
    Basic routing function that decides whether to continue or finish.
    
    This function:
    1. Receives the current state
    2. Receives the config (including callbacks)
    3. Makes a routing decision
    4. Emits a custom event with routing information
    5. Returns the next node name
    """
    
    # Extract data from state
    current_count = state["count"]
    max_count = state["max_count"]
    
    # Make routing decision
    if current_count >= max_count:
        decision = "finish"
        reason = f"Count {current_count} >= max {max_count}"
        confidence = 1.0
    else:
        decision = "increment"
        reason = f"Count {current_count} < max {max_count}"
        confidence = 0.8
    
    # Calculate alternatives
    alternatives = ["increment", "finish"]
    alternatives.remove(decision)
    
    # Emit routing event (if callbacks available)
    if config:
        try:
            # Get callback manager from config
            callbacks = config.get("callbacks")
            if callbacks and hasattr(callbacks, 'on_custom_event'):
                # callbacks is directly a CallbackManager, not a list
                callback_manager = callbacks
                
                # Emit custom event with routing information
                callback_manager.on_custom_event(
                    "langgraph.routing_decision",
                    {
                        # Routing information
                        "source_node": "check",           # Where routing happens
                        "target_node": decision,          # Where we're going
                        "decision": decision,             # The decision value
                        
                        # Context
                        "reason": reason,                 # Why this decision
                        "confidence": confidence,         # Confidence score
                        
                        # State snapshot
                        "state_snapshot": {
                            "current_count": current_count,
                            "max_count": max_count,
                            "status": state.get("status", "unknown"),
                        },
                        
                        # Metadata
                        "routing_type": "counter_based",
                        "alternatives": alternatives,
                        "custom_field": "example_value",  # Custom field example
                    }
                )
        except Exception as e:
            # Gracefully handle callback errors (don't break routing)
            print(f"Warning: Could not emit routing event: {e}")
    
    # Return routing decision
    return decision

# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def check_node(state: CounterState) -> CounterState:
    """Check current count and status."""
    print(f"✓ Check: count={state['count']}, max={state['max_count']}")
    state["status"] = "checking"
    return state

def increment_node(state: CounterState) -> CounterState:
    """Increment the counter."""
    state["count"] += 1
    state["status"] = f"incremented to {state['count']}"
    print(f"✓ Increment: count={state['count']}")
    return state

def finish_node(state: CounterState) -> CounterState:
    """Finish execution."""
    state["status"] = "finished"
    print(f"✓ Finish: final count={state['count']}")
    return state

# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_counter_graph() -> StateGraph:
    """
    Create a simple counter graph with routing.
    
    Graph structure:
        START → check → [routing decision]
                           ├─→ increment → check (loop)
                           └─→ finish → END
    """
    # Create graph
    workflow = StateGraph(CounterState)
    
    # Add nodes
    workflow.add_node("check", check_node)
    workflow.add_node("increment", increment_node)
    workflow.add_node("finish", finish_node)
    
    # Set entry point
    workflow.set_entry_point("check")
    
    # Add conditional edge with routing function
    workflow.add_conditional_edges(
        "check",
        route_counter,  # ← This is our routing function with callback tracking
        {
            "increment": "increment",
            "finish": "finish"
        }
    )
    
    # Add edges
    workflow.add_edge("increment", "check")  # Loop back
    workflow.add_edge("finish", END)
    
    return workflow

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def run_counter_example():
    """Run the counter example with Noveum Trace."""
    
    # Setup Noveum Trace (optional - for demo purposes)
    try:
        import noveum_trace
        noveum_trace.init(
            project=os.getenv("NOVEUM_PROJECT", "routing-example"),
            api_key=os.getenv("NOVEUM_API_KEY"),
            environment=os.getenv("NOVEUM_ENVIRONMENT", "demo"),
        )
        print("✅ Noveum Trace initialized successfully")
    except Exception as e:
        print(f"Note: Noveum Trace not configured: {e}")
    
    # Create callback handler
    handler = NoveumTraceCallbackHandler()
    
    # Create graph
    graph = create_counter_graph()
    app = graph.compile()
    
    print("=" * 80)
    print("🤖 COUNTER GRAPH WITH ROUTING TRACKING")
    print("=" * 80)
    
    # Run with callbacks
    result = app.invoke(
        {"count": 0, "max_count": 3, "status": "initialized"},
        config={"callbacks": [handler]}
    )
    
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print(f"  Count: {result['count']}")
    print(f"  Status: {result['status']}")
    print("=" * 80)
    
    return result

# Run the example
if __name__ == "__main__":
    run_counter_example()