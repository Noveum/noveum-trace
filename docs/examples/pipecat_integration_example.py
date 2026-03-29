#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
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
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

import noveum_trace
from noveum_trace.integrations.pipecat import NoveumTraceObserver

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

    # Find and remove the item
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

    # Clear the order after confirmation
    current_order["items"] = []
    current_order["total"] = 0.0

    await params.result_callback(order_summary)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting drive-thru order bot")

    noveum_trace.init(
        api_key=os.getenv("NOVEUM_API_KEY"),
        project=os.getenv("NOVEUM_PROJECT", "pipecat-drive-thru"),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
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
        ),
    )

    # Register function handlers
    llm.register_function("add_item_to_order", add_item_to_order)
    llm.register_function("remove_item_from_order", remove_item_from_order)
    llm.register_function("view_current_order", view_current_order)
    llm.register_function("confirm_order", confirm_order)

    # Define function schemas
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

    # Records the full stereo conversation (user=left, bot=right) for Noveum Trace.
    # NoveumTraceObserver auto-detects this in attach_to_task() and uploads a
    # pipecat.full_conversation span at the end of the session.
    audio_buffer = AudioBufferProcessor(num_channels=2)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech-to-Text
            user_aggregator,  # User responses
            llm,  # LLM with function calling
            tts,  # Text-to-Speech
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
            audio_buffer,  # Full-conversation stereo recording
        ]
    )

    trace_obs = NoveumTraceObserver(record_audio=True)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[trace_obs],
    )

    trace_obs.attach_to_task(task)

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
        # Kick off the conversation with a greeting
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
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
