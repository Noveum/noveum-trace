"""
Example: trace an OpenAI Agents SDK run with Noveum.

Install the integration extra::

    pip install "noveum-trace[openai-agents]"

Provide credentials and run::

    export NOVEUM_API_KEY=...
    export OPENAI_API_KEY=...
    python docs/examples/openai_agents_integration_example.py

The :class:`NoveumTraceProcessor` mirrors every OpenAI Agents trace/span
(agent runs, tool calls, LLM generations, handoffs, guardrails) as a Noveum
trace/span. It is registered alongside the SDK's default processor, so OpenAI's
own trace upload keeps working; pass ``replace_processors=True`` to
``setup_openai_agents_tracing`` if you want Noveum to be the only exporter.
"""

from __future__ import annotations

import asyncio
import os

from agents import Agent, Runner, add_trace_processor, function_tool

import noveum_trace
from noveum_trace.integrations.openai_agents import NoveumTraceProcessor


@function_tool
def get_weather(city: str) -> str:
    """Return a canned weather report for the given city."""
    return f"The weather in {city} is sunny."


async def main() -> None:
    noveum_trace.init(
        project="openai-agents-example",
        api_key=os.environ.get("NOVEUM_API_KEY"),
    )

    # Capture inputs/outputs is opt-in (privacy-safe defaults). Enable it here so
    # the example traces show the tool arguments and results.
    add_trace_processor(NoveumTraceProcessor(capture_inputs=True, capture_outputs=True))

    agent = Agent(
        name="Weather assistant",
        instructions="You are a helpful assistant. Use the tools when needed.",
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What is the weather in San Francisco?")
    print(result.final_output)

    # Flush buffered traces before the process exits.
    noveum_trace.flush()


if __name__ == "__main__":
    asyncio.run(main())
