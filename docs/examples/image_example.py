"""
Image Caption Generation Examples with Noveum Trace Integration

This example demonstrates:
1. A LangGraph agent that generates captions for images
2. A LangChain chain that generates captions for images
3. Integration with Noveum Trace for both examples
4. Using vision models (Claude with vision) for image understanding

Prerequisites:
    pip install noveum-trace[langchain]
    pip install langchain langchain-anthropic langgraph pillow python-dotenv

Environment Variables (loaded from .env file):
    Create a .env file in the project root with:
        NOVEUM_API_KEY=your_noveum_api_key
        ANTHROPIC_API_KEY=your_anthropic_api_key
        NOVEUM_PROJECT=image-captioning  # optional, defaults to "image-captioning"
        NOVEUM_ENVIRONMENT=dev           # optional, defaults to "dev"
"""

import base64
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from PIL import Image

import noveum_trace
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

load_dotenv()


def setup_noveum_trace():
    """Initialize Noveum Trace with API keys from .env file."""
    api_key = os.getenv("NOVEUM_API_KEY")
    if not api_key:
        print("⚠️  Warning: NOVEUM_API_KEY not found in environment variables.")
        print("   Please create a .env file with: NOVEUM_API_KEY=your_key")

    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "image-captioning"),
        api_key=api_key,
        environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
    )


def load_image_as_base64(image_path: str) -> str:
    """
    Load an image file and convert it to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded image string with data URI prefix
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        img = Image.open(image_path)
        format_map = {"JPEG": "jpeg", "PNG": "png", "GIF": "gif", "WEBP": "webp"}
        img_format = format_map.get(img.format, "jpeg")
        return f"data:image/{img_format};base64,{base64_image}"


def example_chain_caption_generator(image_path: str) -> str:
    """
    Example 1: LangChain Chain for image caption generation.

    This creates a chain that:
    1. Analyzes the image using a vision model
    2. Generates a descriptive caption

    Args:
        image_path: Path to the image file

    Returns:
        Generated caption string
    """
    print("=" * 80)
    print("EXAMPLE 1: LangChain Chain - Image Caption Generator")
    print("=" * 80)

    setup_noveum_trace()
    callback_handler = NoveumTraceCallbackHandler()
    base64_image = load_image_as_base64(image_path)
    print(f"✓ Loaded image from: {image_path}")

    vision_llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.7,
        callbacks=[callback_handler],
    )

    print("\n📸 Analyzing image with vision model...")
    vision_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": "Analyze this image and generate a concise, descriptive caption (one sentence). Focus on the main subject, setting, and key visual elements.",
                    },
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                ],
            )
        ]
    )

    caption_chain = vision_prompt | vision_llm
    caption_response = caption_chain.invoke({"image_url": base64_image})
    final_caption = caption_response.content
    print(f"✓ Caption generated: {final_caption}")

    print("\n" + "=" * 80)
    print("✅ Chain execution completed")
    print("=" * 80)

    return final_caption


class CaptionState(TypedDict):
    """State for the caption generation agent."""

    image_path: str
    image_base64: str
    final_caption: str


def generate_caption_node(state: CaptionState) -> CaptionState:
    """Node that generates a caption from the image using a vision model."""
    print("\n📸 Generating caption from image...")

    # No need to pass callbacks here - LangGraph propagates them from config
    vision_llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.7,
    )

    messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this image and generate a concise, descriptive caption (one sentence). Focus on the main subject, setting, and key visual elements.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": state["image_base64"]},
                },
            ]
        )
    ]

    response = vision_llm.invoke(messages)
    caption = response.content.strip()
    print(f"✓ Caption generated: {caption}")

    # Only return the fields we modified (not the entire state with the image)
    return {"final_caption": caption}


def create_caption_agent() -> StateGraph:
    """Create the LangGraph agent for caption generation."""
    workflow = StateGraph(CaptionState)

    workflow.add_node("generate_caption", generate_caption_node)
    workflow.set_entry_point("generate_caption")
    workflow.add_edge("generate_caption", END)

    return workflow


def example_agent_caption_generator(image_path: str) -> str:
    """
    Example 2: LangGraph Agent for image caption generation.

    This creates an agent that generates a caption from an image using a vision model.

    Args:
        image_path: Path to the image file

    Returns:
        Generated caption string
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: LangGraph Agent - Image Caption Generator")
    print("=" * 80)

    setup_noveum_trace()
    base64_image = load_image_as_base64(image_path)
    print(f"✓ Loaded image from: {image_path}")

    workflow = create_caption_agent()
    app = workflow.compile()
    print("✅ Caption agent compiled")

    initial_state: CaptionState = {
        "image_path": image_path,
        "image_base64": base64_image,
        "final_caption": "",
    }

    print("\n" + "=" * 80)
    print("🚀 STARTING AGENT EXECUTION")
    print("=" * 80)

    callback_handler = NoveumTraceCallbackHandler(use_langchain_assigned_parent=True)
    config = {
        "callbacks": [callback_handler],
        "metadata": {
            "agent_type": "caption_generator",
            "image_path": image_path,
        },
        "tags": ["caption_agent", "langgraph", "noveum_trace", "vision"],
        "recursion_limit": 50,
    }

    try:
        final_state = app.invoke(initial_state, config=config)

        print("\n" + "=" * 80)
        print("✅ AGENT EXECUTION COMPLETED")
        print("=" * 80)

        print("\n📊 EXECUTION SUMMARY:")
        print(
            f"   • Final caption length: {len(final_state['final_caption'])} characters"
        )

        print("\n📝 FINAL CAPTION:")
        print("─" * 80)
        print(final_state["final_caption"])
        print("─" * 80)

        return final_state["final_caption"]

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


def main():
    """Run both examples."""
    noveum_key = os.getenv("NOVEUM_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not noveum_key:
        print("❌ Error: NOVEUM_API_KEY not found in environment variables.")
        print("   Please create a .env file in the project root with:")
        print("   NOVEUM_API_KEY=your_noveum_api_key")
        return

    if not anthropic_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment variables.")
        print("   Please create a .env file in the project root with:")
        print("   ANTHROPIC_API_KEY=your_anthropic_api_key")
        return

    # Hardcoded path to the sample image in docs/examples/sample_image/
    sample_image_path = Path(__file__).parent / "sample_image" / "sample_img.jpeg"

    if not sample_image_path.exists():
        print(f"❌ Error: Image not found at {sample_image_path}")
        print(
            "Please ensure sample_img.jpeg exists in docs/examples/sample_image/ directory."
        )
        return

    print("\n" + "=" * 80)
    print("🖼️  IMAGE CAPTION GENERATION EXAMPLES")
    print("=" * 80)
    print(f"Using image: {sample_image_path}")
    print("✓ API keys loaded from .env file")
    print("=" * 80)

    try:
        chain_caption = example_chain_caption_generator(str(sample_image_path))
        print(f"\n📋 Chain Result: {chain_caption}")
    except Exception as e:
        print(f"\n❌ Chain example failed: {e}")

    try:
        agent_caption = example_agent_caption_generator(str(sample_image_path))
        print(f"\n📋 Agent Result: {agent_caption}")
    except Exception as e:
        print(f"\n❌ Agent example failed: {e}")

    print("\n" + "=" * 80)
    print("🎉 ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nCheck your Noveum Trace dashboard to see:")
    print("  • Chain-level spans for the LangChain example")
    print("  • Graph-level spans for the LangGraph agent")
    print("  • Vision model calls (Claude with vision)")
    print("  • Node-level spans in the agent workflow")

    noveum_trace.flush()


if __name__ == "__main__":
    main()
