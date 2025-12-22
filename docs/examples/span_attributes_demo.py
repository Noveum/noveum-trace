"""
Standalone demo showing two ways to add attributes to a span inside a context
manager: using helper methods and direct dict mutation.
"""

import os

from dotenv import load_dotenv

import noveum_trace
from noveum_trace import trace_operation


def run_demo() -> None:
    load_dotenv()

    # Initialize tracing (falls back to mock mode if no API key)
    noveum_trace.init(transport_config={"batch_size": 1, "batch_timeout": 5.0})

    # Create a span via the context manager
    with trace_operation("attribute_dual_methods") as span:
        # Method 1: helper APIs (preferred)
        span.set_attribute("demo.method", "set_attribute")
        span.set_attributes({"demo.batch.a": 1, "demo.batch.b": {"nested": True}})

        # Method 2: direct dict mutation on a real span
        span.attributes["demo.raw"] = ["x", "y", "z"]

        # Show what we set
        print("Span attributes now:", span.attributes)


if __name__ == "__main__":
    run_demo()



