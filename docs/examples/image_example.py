"""
Image Caption Generation Examples with Noveum Trace Integration

This example demonstrates:
1. A LangGraph agent that generates captions for images
2. A LangChain chain that generates captions for images
3. Integration with Noveum Trace for both examples
4. Using vision models (GPT-4 Vision) for image understanding

Prerequisites:
    pip install noveum-trace[langchain]
    pip install langchain langchain-openai langgraph pillow python-dotenv

Environment Variables (loaded from .env file):
    Create a .env file in the project root with:
        NOVEUM_API_KEY=your_noveum_api_key
        OPENAI_API_KEY=your_openai_api_key
        NOVEUM_PROJECT=image-captioning  # optional, defaults to "image-captioning"
        NOVEUM_ENVIRONMENT=dev           # optional, defaults to "dev"
"""

import os
import base64
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
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

    This creates a sequential chain that:
    1. Analyzes the image using a vision model
    2. Generates an initial caption
    3. Refines the caption to be more engaging

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

    vision_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        callbacks=[callback_handler],
    )

    text_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        callbacks=[callback_handler],
    )

    print("\n📸 Step 1: Analyzing image with vision model...")
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

    initial_caption_chain = vision_prompt | vision_llm
    initial_caption = initial_caption_chain.invoke({"image_url": base64_image})
    initial_caption_text = initial_caption.content
    print(f"✓ Initial caption: {initial_caption_text}")

    print("\n✨ Step 2: Refining caption...")
    refinement_prompt = ChatPromptTemplate.from_template(
        """Here is an initial image caption: {caption}

Enhance this caption to be more vivid, engaging, and descriptive while keeping it concise (one sentence). 
Make it suitable for social media or a photography portfolio."""
    )

    refinement_chain = refinement_prompt | text_llm

    refined_caption = refinement_chain.invoke({"caption": initial_caption_text})
    final_caption = refined_caption.content
    print(f"✓ Refined caption: {final_caption}")

    print("\n" + "=" * 80)
    print("✅ Chain execution completed")
    print("=" * 80)

    return final_caption


class CaptionState(TypedDict):
    """State for the caption generation agent."""

    image_path: str
    image_base64: str
    messages: Annotated[list, add_messages]
    current_caption: str
    iteration_count: int
    final_caption: str


def analyze_image_node(state: CaptionState) -> CaptionState:
    """Node that analyzes the image using a vision model."""
    print("\n🔍 ANALYZE IMAGE NODE")

    callback_handler = NoveumTraceCallbackHandler()
    vision_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        callbacks=[callback_handler],
    )

    messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this image in detail. Describe the main subject, setting, colors, mood, and any notable features. Be comprehensive but concise.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": state["image_base64"]},
                },
            ]
        )
    ]

    response = vision_llm.invoke(messages)
    analysis = response.content
    print(f"✓ Image analysis completed ({len(analysis)} characters)")

    state["messages"].append(HumanMessage(content="Analyze this image and generate a caption."))
    state["messages"].append(AIMessage(content=f"Image Analysis:\n{analysis}"))
    return state


def generate_caption_node(state: CaptionState) -> CaptionState:
    """Node that generates an initial caption based on the analysis."""
    print("\n✍️  GENERATE CAPTION NODE")

    callback_handler = NoveumTraceCallbackHandler()
    text_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        callbacks=[callback_handler],
    )

    analysis = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and "Image Analysis:" in msg.content:
            analysis = msg.content.replace("Image Analysis:\n", "")
            break

    prompt = f"""Based on this image analysis, generate a creative and engaging caption:

{analysis}

Create a caption that:
- Is one sentence long
- Captures the essence and mood of the image
- Is suitable for social media or photography portfolio
- Is vivid and descriptive

Caption:"""

    response = text_llm.invoke([HumanMessage(content=prompt)])
    caption = response.content.strip()
    print(f"✓ Generated caption: {caption}")

    state["current_caption"] = caption
    state["iteration_count"] += 1
    state["messages"].append(AIMessage(content=f"Generated Caption: {caption}"))
    return state


def refine_caption_node(state: CaptionState) -> CaptionState:
    """Node that refines the caption to make it more engaging."""
    print("\n✨ REFINE CAPTION NODE")

    callback_handler = NoveumTraceCallbackHandler()
    text_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.8,
        callbacks=[callback_handler],
    )

    prompt = f"""Here is a caption for an image: {state['current_caption']}

Refine this caption to be more vivid, engaging, and polished while keeping it concise (one sentence). 
Make it stand out and capture attention. Return only the refined caption, nothing else."""

    response = text_llm.invoke([HumanMessage(content=prompt)])
    refined_caption = response.content.strip()
    print(f"✓ Refined caption: {refined_caption}")

    state["current_caption"] = refined_caption
    state["iteration_count"] += 1
    state["messages"].append(AIMessage(content=f"Refined Caption: {refined_caption}"))
    return state


def should_refine(state: CaptionState) -> Literal["refine", "finalize"]:
    """Decision node: determine if caption needs refinement or is ready."""
    if state["iteration_count"] < 2:
        print("\n🔄 Caption needs refinement")
        return "refine"
    else:
        print("\n✅ Caption is ready")
        return "finalize"


def finalize_caption_node(state: CaptionState) -> CaptionState:
    """
    Final node that sets the final caption.
    """
    print("\n🎯 FINALIZE CAPTION NODE")

    state["final_caption"] = state["current_caption"]
    state["messages"].append(AIMessage(content=f"Final Caption: {state['final_caption']}"))

    print(f"✓ Final caption set: {state['final_caption']}")

    return state


def create_caption_agent() -> StateGraph:
    """Create the LangGraph agent for caption generation."""
    workflow = StateGraph(CaptionState)

    workflow.add_node("analyze_image", analyze_image_node)
    workflow.add_node("generate_caption", generate_caption_node)
    workflow.add_node("refine_caption", refine_caption_node)
    workflow.add_node("finalize", finalize_caption_node)

    workflow.set_entry_point("analyze_image")
    workflow.add_edge("analyze_image", "generate_caption")

    workflow.add_conditional_edges(
        "generate_caption",
        should_refine,
        {
            "refine": "refine_caption",
            "finalize": "finalize",
        },
    )

    workflow.add_conditional_edges(
        "refine_caption",
        should_refine,
        {
            "refine": "refine_caption",
            "finalize": "finalize",
        },
    )

    workflow.add_edge("finalize", END)
    return workflow


def example_agent_caption_generator(image_path: str) -> str:
    """
    Example 2: LangGraph Agent for image caption generation.

    This creates an agent that:
    1. Analyzes the image using a vision model
    2. Generates an initial caption
    3. Refines the caption iteratively
    4. Finalizes the best caption

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
        "messages": [],
        "current_caption": "",
        "iteration_count": 0,
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
        print(f"   • Iterations performed: {final_state['iteration_count']}")
        print(f"   • Final caption length: {len(final_state['final_caption'])} characters")

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
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not noveum_key:
        print("❌ Error: NOVEUM_API_KEY not found in environment variables.")
        print("   Please create a .env file in the project root with:")
        print("   NOVEUM_API_KEY=your_noveum_api_key")
        return
    
    if not openai_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("   Please create a .env file in the project root with:")
        print("   OPENAI_API_KEY=your_openai_api_key")
        return
    
    project_root = Path(__file__).parent.parent.parent
    sample_image_path = project_root / "sample_img.jpeg"

    if not sample_image_path.exists():
        print(f"❌ Error: Image not found at {sample_image_path}")
        print("Please ensure sample_img.jpeg exists in the project root.")
        return

    print("\n" + "=" * 80)
    print("🖼️  IMAGE CAPTION GENERATION EXAMPLES")
    print("=" * 80)
    print(f"Using image: {sample_image_path}")
    print(f"✓ API keys loaded from .env file")
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
    print("  • Vision model calls (GPT-4 Vision)")
    print("  • Text model calls for refinement")
    print("  • Node-level spans in the agent workflow")

    noveum_trace.flush()


if __name__ == "__main__":
    main()
