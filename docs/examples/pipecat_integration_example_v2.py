#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Pipecat drive-thru order bot — Noveum Trace SDK v2 integration.

Demonstrates the new two-call NoveumPipecatTracer API.  Compare with
pipecat_integration_example.py (legacy, seven-step manual wiring) to see
what changed.

Noveum-specific additions vs. a plain Pipecat bot (6 lines total):

    import noveum_trace                                                 # +1
    from noveum_trace.integrations.pipecat import NoveumPipecatTracer  # +1

    noveum_trace.init(api_key=..., project=...)                        # +1
    tracer = NoveumPipecatTracer(record_audio=True, ...)               # +1
    pipeline = tracer.observe_pipeline(pipeline)                        # +1
    task = await tracer.register_task_handlers(task, transport=...)    # +1

Everything else — transport, pipeline    , PipelineTask — is stock Pipecat.
No manual AudioBufferProcessor needed: observe_pipeline auto-inserts one.

Features enabled via NoveumPipecatTracer flags (all opt-out unless noted):
  - observe_pipeline auto-inserts AudioBufferProcessor and optionally registers
    an OTEL SpanProcessor for custom spans
  - register_task_handlers taps transport for pre-filter raw audio
  - PipelineParams.enable_metrics / enable_usage_metrics auto-patched True
  - ErrorFrame / FatalErrorFrame → span errors + trace events
  - SystemLogFrame (opt-in, capture_system_logs=True) → span events
  - LLMUsageMetricsFrame (newer Pipecat standalone frame) → tokens + cost
  - transport type, room URL, runner idle timeout stamped on root trace;
    pass runner_args= to register_task_handlers to enrich further

Plain OTEL spans emitted anywhere in customer code (see add_item_to_order and
confirm_order below) are automatically captured by NoveumCustomSpanProcessor
and nested under the active pipecat.turn — no noveum_trace import required in
the business-logic code.  Requires: pip install 'noveum-trace[pipecat-otel]'
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

import noveum_trace
from noveum_trace.integrations.pipecat import NoveumPipecatTracer

try:
    from opentelemetry import trace as otel_trace

    _tracer = otel_trace.get_tracer(__name__)
    OTEL_AVAILABLE = True
except ImportError:
    otel_trace = None  # type: ignore[assignment]
    _tracer = None
    OTEL_AVAILABLE = False

load_dotenv(override=True)


# Simulated order storage
current_order = {"items": [], "total": 0.0}

# Menu pricing
MENU = {
    "burger": 5.99,
    "cheeseburger": 6.99,
    "double_burger": 8.99,
    "fries": 2.99,
    "large_fries": 3.99,
    "small_drink": 1.99,
    "medium_drink": 2.49,
    "large_drink": 2.99,
    "milkshake": 4.99,
    "chicken_nuggets": 4.99,
    "salad": 5.99,
}


async def add_item_to_order(params: FunctionCallParams):
    """Add an item to the current order."""
    item = params.arguments.get("item", "").lower().replace(" ", "_")
    quantity = params.arguments.get("quantity", 1)

    if item not in MENU:
        await params.result_callback(
            {
                "success": False,
                "message": f"Sorry, we don't have {item} on our menu.",
            }
        )
        return

    price = MENU[item]

    # Plain OTEL span — captured automatically by NoveumCustomSpanProcessor
    # and nested under the active pipecat.turn (requires capture_custom_spans=True).
    # No noveum_trace import needed here; the processor intercepts it transparently.
    if _tracer is not None:
        with _tracer.start_as_current_span("menu.price_lookup") as otel_span:
            otel_span.set_attribute("menu.item", item)
            otel_span.set_attribute("menu.price", price)
            otel_span.set_attribute("menu.quantity", quantity)

    current_order["items"].append({"name": item, "quantity": quantity, "price": price})
    current_order["total"] += price * quantity

    logger.info(
        f"Added {quantity}x {item} to order. New total: ${current_order['total']:.2f}"
    )

    await params.result_callback(
        {
            "success": True,
            "item": item,
            "quantity": quantity,
            "price": price,
            "current_total": current_order["total"],
        }
    )


async def remove_item_from_order(params: FunctionCallParams):
    """Remove an item from the current order."""
    item = params.arguments.get("item", "").lower().replace(" ", "_")

    for order_item in current_order["items"]:
        if order_item["name"] == item:
            current_order["items"].remove(order_item)
            current_order["total"] -= order_item["price"] * order_item["quantity"]
            logger.info(
                f"Removed {item} from order. New total: ${current_order['total']:.2f}"
            )

            await params.result_callback(
                {
                    "success": True,
                    "item": item,
                    "current_total": current_order["total"],
                }
            )
            return

    await params.result_callback(
        {
            "success": False,
            "message": f"{item} is not in your order.",
        }
    )


async def view_current_order(params: FunctionCallParams):
    """View the current order."""
    if not current_order["items"]:
        await params.result_callback(
            {
                "success": True,
                "items": [],
                "total": 0.0,
                "message": "Your order is empty.",
            }
        )
        return

    items_summary = []
    for item in current_order["items"]:
        items_summary.append(
            {
                "name": item["name"].replace("_", " "),
                "quantity": item["quantity"],
                "price": item["price"],
            }
        )

    logger.info(f"Current order: {items_summary}, Total: ${current_order['total']:.2f}")

    await params.result_callback(
        {
            "success": True,
            "items": items_summary,
            "total": current_order["total"],
        }
    )


async def confirm_order(params: FunctionCallParams):
    """Confirm and finalize the order."""
    if not current_order["items"]:
        await params.result_callback(
            {
                "success": False,
                "message": "Cannot confirm an empty order.",
            }
        )
        return

    logger.info(f"Order confirmed! Total: ${current_order['total']:.2f}")
    logger.info(f"Items: {current_order['items']}")

    order_summary = {
        "success": True,
        "order_number": "12345",
        "items": current_order["items"],
        "total": current_order["total"],
        "message": f"Order confirmed! Your total is ${current_order['total']:.2f}. Please pull forward to the next window.",
    }

    # Plain OTEL span — no Noveum import needed; captured automatically and
    # nested under the active pipecat.turn when capture_custom_spans=True.
    if _tracer is not None:
        with _tracer.start_as_current_span("order.finalize") as otel_span:
            current_order["items"] = []
            current_order["total"] = 0.0
            otel_span.set_attribute("order.item_count", len(order_summary["items"]))
            otel_span.set_attribute("order.number", order_summary["order_number"])
            otel_span.set_attribute("order.total_usd", order_summary["total"])
    else:
        current_order["items"] = []
        current_order["total"] = 0.0

    await params.result_callback(order_summary)


# Stock transport params — no Noveum wrappers needed.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=RNNoiseFilter(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=RNNoiseFilter(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=RNNoiseFilter(),
    ),
}


async def run_bot(transport, runner_args: RunnerArguments, tracer: NoveumPipecatTracer):
    logger.info("Starting drive-thru order bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash",
        system_instruction="""You are a friendly drive-thru order taker at a fast food restaurant.

Your menu includes:
- Burgers: burger ($5.99), cheeseburger ($6.99), double burger ($8.99)
- Sides: fries ($2.99), large fries ($3.99), chicken nuggets ($4.99), salad ($5.99)
- Drinks: small drink ($1.99), medium drink ($2.49), large drink ($2.99), milkshake ($4.99)

Your job:
1. Greet the customer warmly
2. Take their order using the add_item_to_order function
3. Confirm items and make suggestions
4. Use view_current_order to review their order
5. When they're done, use confirm_order to finalize
6. If they want to cancel or end the call early, use end_call

Be conversational, friendly, and efficient. Keep responses concise since this is voice-based.
If they ask for something not on the menu, politely let them know and suggest alternatives.""",
    )

    llm.register_function("add_item_to_order", add_item_to_order)
    llm.register_function("remove_item_from_order", remove_item_from_order)
    llm.register_function("view_current_order", view_current_order)
    llm.register_function("confirm_order", confirm_order)

    add_item_schema = FunctionSchema(
        name="add_item_to_order",
        description="Add an item to the customer's order. Use this when the customer requests an item.",
        properties={
            "item": {
                "type": "string",
                "description": "The menu item name (e.g., 'burger', 'fries', 'medium_drink')",
            },
            "quantity": {
                "type": "integer",
                "description": "The quantity of the item (default 1)",
                "default": 1,
            },
        },
        required=["item"],
    )

    remove_item_schema = FunctionSchema(
        name="remove_item_from_order",
        description="Remove an item from the customer's order.",
        properties={
            "item": {
                "type": "string",
                "description": "The menu item name to remove",
            },
        },
        required=["item"],
    )

    view_order_schema = FunctionSchema(
        name="view_current_order",
        description="View the current order with all items and total price.",
        properties={},
        required=[],
    )

    confirm_order_schema = FunctionSchema(
        name="confirm_order",
        description="Confirm and finalize the customer's order. Use this when the customer is done ordering.",
        properties={},
        required=[],
    )

    end_call_schema = FunctionSchema(
        name="end_call",
        description=(
            "End the drive-thru call after a brief goodbye. Use when the customer wants to cancel or hang up."
        ),
        properties={},
        required=[],
    )

    tools = ToolsSchema(
        standard_tools=[
            add_item_schema,
            remove_item_schema,
            view_order_schema,
            confirm_order_schema,
            end_call_schema,
        ]
    )

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech-to-Text
            user_aggregator,  # User responses
            llm,  # LLM with function calling
            tts,  # Text-to-Speech
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
            # AudioBufferProcessor is auto-inserted at the tail by
            # tracer.observe_pipeline() — no manual setup needed.
        ]
    )

    # --- Noveum Trace wiring (2 calls) ---
    # observe_pipeline auto-appends AudioBufferProcessor when absent.
    pipeline = tracer.observe_pipeline(pipeline)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    task = await tracer.register_task_handlers(
        task,
        transport=transport,  # raw audio tap + transport type + room_url
        runner_args=runner_args,  # room URL, idle timeout → root trace attributes
    )
    # --- end Noveum wiring ---

    async def end_call(params: FunctionCallParams):
        """End the call: fixed goodbye via TTS, then graceful task shutdown."""
        logger.info("Customer ended the call")
        await params.result_callback(
            {"success": True, "message": "Call ended."},
            properties=FunctionCallResultProperties(run_llm=False),
        )
        await task.queue_frames(
            [TTSSpeakFrame("Thank you for visiting! Have a great day!")]
        )
        await task.stop_when_done()

    llm.register_function("end_call", end_call)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Customer connected to drive-thru")
        context.add_message(
            {
                "role": "user",
                "content": "Greet the customer and ask for their order.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Customer disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    noveum_trace.init(
        api_key=os.getenv("NOVEUM_API_KEY"),
        project=os.getenv("NOVEUM_PROJECT", "pipecat-drive-thru"),
    )

    tracer = NoveumPipecatTracer(
        record_audio=True,
        record_raw_input_audio=True,  # taps transport.input().push_audio_frame
        capture_custom_spans=True,  # plain-OTEL spans auto-nested under active turn
        auto_enable_metrics=True,  # ensures MetricsFrame is always emitted
        capture_errors=True,  # ErrorFrame/FatalErrorFrame → span errors
        capture_system_logs=False,  # SystemLogFrame opt-in (high volume)
        capture_session_metadata=True,  # room URL + transport type on root trace
    )

    # Stock transport — no Noveum*Transport class-swap needed.
    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args, tracer)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
