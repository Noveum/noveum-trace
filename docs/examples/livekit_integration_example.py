"""
Drive-Thru Agent Example with LiveKit STT/TTS Integration

This example demonstrates a realistic drive-thru ordering agent that:
- Takes customer orders via voice
- Uses LLM for natural conversation
- Integrates with LiveKit for STT/TTS
- Automatically traces all operations with noveum-trace

Features (via setup_livekit_tracing):
- Chat history tracked in generation spans
- Function calls captured in session events (same span as LLM call)
- Available tools extracted from agent
- Full conversation audio uploaded at session end (stereo OGG)
- Per-utterance STT/TTS audio captured via wrappers
- LLM metrics (tokens, cost, latency) captured from LiveKit events

Two modes:
- Default: Runs with dummy inputs/outputs (text simulation)
- --test: Actually runs with LiveKit voice agent

Recording (for full conversation audio):
- Recording is automatically enabled via setup_livekit_tracing (record=True by default)
- Audio is saved as stereo OGG (left=user, right=agent) and uploaded to Noveum
- Local files are automatically cleaned up after upload (cleanup_audio_files=True by default)

Prerequisites:
    - livekit
    - livekit-agents
    - livekit-plugins-deepgram (or any other STT plugin)
    - livekit-plugins-cartesia (or any other TTS plugin)
    - openai (for LLM, required)

Install:
    pip install livekit livekit-agents livekit-plugins-deepgram livekit-plugins-cartesia noveum-trace openai python-dotenv

Environment Variables:
    Create a .env file in the project root with:
    - OPENAI_API_KEY (required for LLM)
    - DEEPGRAM_API_KEY (required for STT in --test mode)
    - CARTESIA_API_KEY (required for TTS in --test mode)
    - LIVEKIT_URL (required for --test mode)
    - LIVEKIT_API_KEY (required for --test mode)
    - LIVEKIT_API_SECRET (required for --test mode)
    - NOVEUM_API_KEY (optional, for sending traces)
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Optional

from livekit.agents import (
    Agent,
    AgentServer,
    JobContext,
    RunContext,
    ToolError,
    cli,
    function_tool,
)
from livekit.agents.voice import AgentSession
from livekit.plugins import cartesia, deepgram
from livekit.plugins import openai as openai_plugin
from openai import OpenAI
from pydantic import Field

import noveum_trace
from noveum_trace.integrations.livekit import (
    LiveKitSTTWrapper,
    LiveKitTTSWrapper,
    setup_livekit_tracing,
)
from noveum_trace.integrations.livekit.livekit_utils import extract_job_context

# Disable LiveKit's built-in OpenTelemetry to prevent telemetry errors
os.environ.setdefault("OTEL_SDK_DISABLED", "true")


# Load environment variables from .env file in project root
try:
    from dotenv import load_dotenv

    # Get project root (go up from docs/examples/ to project root)
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Also try loading from current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass


# Import LiveKit wrappers

# =============================================================================
# MENU & ORDER MANAGEMENT
# =============================================================================

MENU = {
    "burger": {"price": 5.99, "description": "Classic burger"},
    "fries": {"price": 2.99, "description": "French fries"},
    "drink": {"price": 1.99, "description": "Soft drink"},
    "coke": {"price": 1.99, "description": "Coca-Cola"},
}


class Order:
    """Simple order tracking."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.status = "collecting"

    def add_item(self, item: str, quantity: int = 1) -> bool:
        """Add item to order."""
        if item.lower() in MENU:
            for _ in range(quantity):
                self.items.append(
                    {"name": item.lower(), "price": MENU[item.lower()]["price"]}
                )
            return True
        return False

    def get_total(self) -> float:
        """Calculate total price."""
        return sum(item["price"] for item in self.items)

    def get_summary(self) -> str:
        """Get order summary."""
        if not self.items:
            return "No items in order"
        item_counts: dict[str, int] = {}
        for item in self.items:
            name = item["name"]
            item_counts[name] = item_counts.get(name, 0) + 1

        summary_parts = []
        for name, count in item_counts.items():
            summary_parts.append(f"{count}x {name}")
        return ", ".join(summary_parts)


@dataclass
class Userdata:
    """User data for the agent session."""

    order: Order
    # Reference to session for closing
    session: Optional["AgentSession"] = None


# =============================================================================
# LLM CONVERSATION HANDLER
# =============================================================================


@noveum_trace.trace_llm(provider="openai", metadata={"model": "gpt-4o-mini"})
def call_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call LLM for conversation (traced automatically).

    Requires OpenAI API key to be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it with: export OPENAI_API_KEY=your-key"
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


def create_system_prompt(order: Order) -> str:
    """Create system prompt for the drive-thru agent."""
    # Filter out "coke" from menu display since it's the same as "drink"
    menu_display = {k: v for k, v in MENU.items() if k != "coke"}
    menu_text = "\n".join(
        [f"- {name}: ${info['price']:.2f}" for name, info in menu_display.items()]
    )
    menu_text += "\n- coke: $1.99 (same as drink)"

    return f"""You are a friendly drive-thru order taker. Your job is to:
1. Greet customers warmly
2. Take their order from this menu:
{menu_text}
3. Confirm items as they order
4. Ask if they want anything else
5. Tell them the total when they're done

Note: "coke" refers to a drink. Sizes (small, medium, large) don't affect price.

Current order: {order.get_summary()}
Be concise and friendly. Keep responses short (1-2 sentences)."""


# =============================================================================
# DRIVE-THRU AGENT WITH TOOLS
# =============================================================================


class DriveThruAgent(Agent):
    """Drive-thru ordering agent with tools."""

    def __init__(self, *, userdata: Userdata) -> None:
        menu_text = "\n".join(
            [f"- {name}: ${info['price']:.2f}" for name, info in MENU.items()]
        )

        instructions = f"""You are a friendly drive-thru order taker. Your job is to:
1. Greet customers warmly
2. Take their order from this menu:
{menu_text}
3. Use the add_item_to_order tool to add items when the customer orders them
4. Confirm items as they order
5. Ask if they want anything else
6. When customer says "bye", "done", "that's all", "thank you bye" or indicates they're finished,
   use the complete_order tool to finalize and end the conversation

Note: "coke" refers to a drink. Sizes (small, medium, large) don't affect price.

Be concise and friendly. Keep responses short (1-2 sentences)."""

        super().__init__(
            instructions=instructions,
            tools=[self.build_add_item_tool(), self.build_complete_order_tool()],
        )

    def build_add_item_tool(self) -> Callable[..., Any]:
        """Build the add_item_to_order tool."""
        available_items = list(MENU.keys())

        @function_tool
        async def add_item_to_order(
            ctx: RunContext[Userdata],
            item: Annotated[
                str,
                Field(
                    description="The item to add to the order.",
                    json_schema_extra={"enum": available_items},
                ),
            ],
            quantity: Annotated[
                int,
                Field(
                    description="Quantity of the item (default: 1)",
                    ge=1,
                ),
            ] = 1,
        ) -> str:
            """
            Call this when the user orders an item. Use this tool to add items to their order.

            Examples:
            - User says "I want a burger" -> call with item="burger", quantity=1
            - User says "I want a coke" -> call with item="drink", quantity=1 (coke is a drink)
            - User says "Two burgers" -> call with item="burger", quantity=2
            """
            # Handle "coke" as "drink"
            if item.lower() == "coke":
                item = "drink"

            if item.lower() not in MENU:
                raise ToolError(
                    f"error: {item} is not on the menu. Available items: {', '.join(available_items)}"
                )

            success = ctx.userdata.order.add_item(item.lower(), quantity)
            if not success:
                raise ToolError(f"error: failed to add {item} to order")

            total = ctx.userdata.order.get_total()
            summary = ctx.userdata.order.get_summary()
            return f"Added {quantity}x {item} to your order. Current order: {summary}. Total: ${total:.2f}"

        return add_item_to_order

    def build_complete_order_tool(self) -> Callable[..., Any]:
        """Build the complete_order tool that ends the session."""

        @function_tool
        async def complete_order(
            ctx: RunContext[Userdata],
        ) -> str:
            """
            Call this when the customer is done ordering and wants to end the conversation.
            Use when customer says: "bye", "done", "that's all", "thank you", "goodbye", etc.
            This will finalize the order and end the call.
            """
            order = ctx.userdata.order
            total = order.get_total()
            summary = order.get_summary()

            # Mark order as complete
            order.status = "completed"

            # Schedule session close after a short delay (to allow goodbye message)
            async def close_session() -> None:
                await asyncio.sleep(3)  # Wait for TTS to finish goodbye
                if ctx.userdata.session:
                    print("📞 Ending call - order complete")
                    await ctx.userdata.session.aclose()

            asyncio.create_task(close_session())

            if order.items:
                return f"Order complete! {summary}. Total: ${total:.2f}. Thank you, have a great day!"
            else:
                return "No items ordered. Thank you for visiting, have a great day!"

        return complete_order


# =============================================================================
# TEXT SIMULATION AGENT (for default mode)
# =============================================================================


class DriveThruAgentText:
    """Drive-thru ordering agent for text simulation."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.order = Order()
        self.conversation_history: list[dict[str, str]] = []

    def process_customer_message(self, text: str) -> str:
        """
        Process customer message and return agent response.

        This is traced automatically via the LLM call.
        """
        # Add to conversation history
        self.conversation_history.append({"role": "customer", "content": text})

        # Try to extract order items from text
        text_lower = text.lower()
        # Handle "coke" as "drink"
        if "coke" in text_lower and "drink" not in text_lower:
            self.order.add_item("drink", 1)
        else:
            for item in MENU.keys():
                if item in text_lower:
                    # Extract quantity if mentioned
                    quantity = 1
                    words = text_lower.split()
                    for i, word in enumerate(words):
                        if (
                            word.isdigit()
                            and i + 1 < len(words)
                            and words[i + 1] == item
                        ):
                            quantity = int(word)
                            break

                    self.order.add_item(item, quantity)

        # Get LLM response
        system_prompt = create_system_prompt(self.order)
        user_prompt = text

        # If order has items, include total in system prompt
        if self.order.items:
            total = self.order.get_total()
            system_prompt += f"\n\nCurrent order total: ${total:.2f}"

        response = call_llm(user_prompt, system_prompt)

        # Replace placeholder with actual total
        if "${:.2f}" in response:
            response = response.replace("${:.2f}", f"${self.order.get_total():.2f}")

        # Add to conversation history
        self.conversation_history.append({"role": "agent", "content": response})

        return response

    def get_order_summary(self) -> dict[str, Any]:
        """Get final order summary."""
        return {
            "items": self.order.items,
            "total": self.order.get_total(),
            "summary": self.order.get_summary(),
        }


# =============================================================================
# LIVEKIT AGENT SERVER
# =============================================================================

# Initialize noveum-trace once at module level
if not noveum_trace.is_initialized():
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "drive-thru-agent"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment=os.getenv("NOVEUM_ENVIRONMENT", "production"),
    )

server = AgentServer()


@server.rtc_session()
async def drive_thru_agent(ctx: JobContext) -> None:
    """
    LiveKit agent entrypoint for voice interaction.

    This sets up STT/TTS with tracing and handles the conversation.

    All tracing is automatic:
    - LiveKitSTTWrapper: Creates stt.stream spans with per-utterance audio + transcript
    - LiveKitTTSWrapper: Creates tts.stream spans with per-utterance audio + input text
    - setup_livekit_tracing: Handles session events including:
        - LLM generation events with chat history and available tools
        - Function calls merged into generation spans (not separate spans)
        - Full conversation audio upload at session end (stereo OGG)
        - LLM metrics (tokens, cost, latency) from LiveKit events
    """
    job_context = await extract_job_context(ctx)

    session_id = ctx.job.id

    # Create STT provider and wrap with tracing
    # Per-utterance audio is uploaded by the wrapper
    base_stt = deepgram.STT(model="nova-2", language="en-US")
    traced_stt = LiveKitSTTWrapper(
        stt=base_stt, session_id=session_id, job_context=job_context
    )

    # Create TTS provider and wrap with tracing
    # Per-utterance audio is uploaded by the wrapper
    base_tts = cartesia.TTS(
        model="sonic-english",
        voice="a0e99841-438c-4a64-b679-ae501e7d6091",  # Friendly voice
    )
    traced_tts = LiveKitTTSWrapper(
        tts=base_tts, session_id=session_id, job_context=job_context
    )

    # Create userdata with order
    order = Order()
    userdata = Userdata(order=order)

    # Create LLM (setup_livekit_tracing handles LLM tracing via session events)
    # It captures: available tools, chat history, function calls, LLM metrics
    llm = openai_plugin.LLM(model="gpt-4o-mini", temperature=0.7)

    # Create agent session with wrapped STT/TTS providers
    session = AgentSession[Userdata](
        userdata=userdata,
        stt=traced_stt,
        llm=llm,
        tts=traced_tts,
    )

    # Store session reference in userdata for the complete_order tool
    userdata.session = session

    # Setup automatic tracing for the agent session
    # This creates the trace, handles session events, and uploads full conversation audio
    # LiveKit's RecorderIO handles this as a stereo OGG file (left=user, right=agent)
    setup_livekit_tracing(session)

    print(f"🍔 Drive-thru agent connected to room: {ctx.room.name}")
    print(f"📝 Session ID: {session_id}")

    # Start session with the agent (which has tools)
    # The recording is automatically uploaded to Noveum at session close
    await session.start(agent=DriveThruAgent(userdata=userdata), room=ctx.room)

    # Note: In console mode, the session continues running after start() returns.
    # Use 'bye' or 'done' to end gracefully (triggers complete_order tool)
    print("🎤 Agent is listening... (say 'bye' or 'done' to end)")


# =============================================================================
# DEFAULT MODE: TEXT SIMULATION
# =============================================================================


async def run_text_simulation() -> None:
    """
    Run drive-thru agent in text mode (default).

    Simulates a conversation with dummy inputs/outputs.
    """
    print("=" * 60)
    print("🍔 DRIVE-THRU AGENT - TEXT SIMULATION MODE")
    print("=" * 60)
    print()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required.")
        print("   Set it with: export OPENAI_API_KEY=your-key")
        sys.exit(1)

    # Create trace
    with noveum_trace.start_trace("drive_thru_simulation") as trace:
        trace.set_attributes({"mode": "text_simulation"})

        # Create agent for text simulation
        agent = DriveThruAgentText(session_id="sim-session-001")

        # Simulate conversation - fixed flow
        customer_messages = [
            "hi",
            "I want a coke",
            "I want a small coke, that's all, please add it",
            "thank you",
        ]

        print("Starting conversation...\n")

        for i, message in enumerate(customer_messages, 1):
            print(f"[Turn {i}]")
            print(f"👤 Customer: {message}")

            # Process message (this creates LLM spans)
            response = agent.process_customer_message(message)
            print(f"🤖 Agent: {response}")
            print()

        # Show final order
        order_summary = agent.get_order_summary()
        print("-" * 60)
        print("📋 FINAL ORDER")
        print("-" * 60)
        print(f"Items: {order_summary['summary']}")
        print(f"Total: ${order_summary['total']:.2f}")
        print()

        print("✅ Simulation complete!")
        print("📊 Check Noveum dashboard for traces:")
        print("   - LLM spans for each conversation turn")
        print("   - Order processing spans")
        print()
        print(
            "💡 To run with actual voice agent, use: python livekit_integration_example.py --test console"
        )


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main() -> None:
    """Main entry point."""
    # Print configuration info regardless of mode
    print("🍔 Drive-Thru Agent Example - Two modes available:")
    print("   Default: Text simulation with dummy inputs/outputs")
    print("   --test:  Actual LiveKit voice agent (requires LiveKit setup)")
    print()

    # Check for --test flag before argparse (to avoid LiveKit CLI intercepting it)
    test_mode = "--test" in sys.argv

    if test_mode:
        # Remove --test from sys.argv so LiveKit CLI doesn't see it
        sys.argv = [arg for arg in sys.argv if arg != "--test"]

        # Check for required API keys
        missing_keys = []

        if not os.getenv("OPENAI_API_KEY"):
            missing_keys.append("OPENAI_API_KEY")

        if not os.getenv("DEEPGRAM_API_KEY"):
            missing_keys.append("DEEPGRAM_API_KEY")

        if not os.getenv("CARTESIA_API_KEY"):
            missing_keys.append("CARTESIA_API_KEY")

        if missing_keys:
            print("❌ Error: Missing required API keys:")
            for key in missing_keys:
                print(f"   - {key}")
            print("\n   Set them with:")
            for key in missing_keys:
                print(f"   export {key}=your-key")
            sys.exit(1)

        print("🍔 Starting LiveKit drive-thru agent...")
        print("   Make sure you have set:")
        print("   - OPENAI_API_KEY (required)")
        print("   - DEEPGRAM_API_KEY (required)")
        print("   - CARTESIA_API_KEY (required)")
        print("   - LIVEKIT_URL")
        print("   - LIVEKIT_API_KEY")
        print("   - LIVEKIT_API_SECRET")
        print("   - NOVEUM_API_KEY (optional)")
        print()

        # Use livekit-agents CLI with AgentServer
        cli.run_app(server)
    else:
        # Run text simulation
        asyncio.run(run_text_simulation())


if __name__ == "__main__":
    main()
