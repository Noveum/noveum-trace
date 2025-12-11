"""
Multimodal Examples for Noveum Trace SDK

This file demonstrates how to use the extended Noveum Trace SDK
for tracing multimodal AI applications including images, audio,
video, and multimodal LLM operations.
"""

import os
import time
from typing import Any, cast

# Load environment variables (install python-dotenv if needed)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "python-dotenv not installed. Environment variables will be read from system only."
    )
    pass

import noveum_trace

# Import context managers for tracing
from noveum_trace import trace_llm_call, trace_operation, trace_agent_operation


# Note: This example uses context managers for tracing instead of decorators.
# All tracing is done using context managers as recommended in the README.


# Initialize SDK with multimodal configuration
noveum_trace.init(
    api_key=os.getenv("NOVEUM_API_KEY"),
    project="example-project-testing-multimodal-examples",
    environment="dev-aman",
)

# =============================================================================
# IMAGE GENERATION & PROCESSING EXAMPLES
# =============================================================================


def generate_product_image(
    description: str, style: str = "photorealistic"
) -> dict[str, Any]:
    """Generate product images using DALL-E 3."""
    with trace_operation("image_generation") as span:
        span.set_attributes(
            {
                "model": "dall-e-3",
                "provider": "openai",
                "style": style,
                "description": description,
            }
        )

        print(f"Generating image: {description} in {style} style")

        # Simulate OpenAI DALL-E API call
        time.sleep(2)  # Simulate API latency

        result = {
            "image_url": f"https://example.com/generated_image_{hash(description)}.png",
            "revised_prompt": f"Professional {description} in {style} style, high quality, detailed",
            "format": "PNG",
            "size": "1024x1024",
            "generation_time_ms": 2000,
            "cost": 0.04,
        }

        span.set_attribute("result.image_url", result["image_url"])
        span.set_attribute("result.cost", result["cost"])

        return result


def detect_objects_in_image(image_path: str) -> dict[str, Any]:
    """Detect objects in an image using YOLO."""
    with trace_operation("object_detection") as span:
        span.set_attributes(
            {
                "model": "yolo-v8",
                "provider": "ultralytics",
                "image_path": image_path,
            }
        )

        print(f"Detecting objects in: {image_path}")

        # Simulate object detection
        time.sleep(1)

        result = {
            "objects": [
                {"class": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]},
                {"class": "car", "confidence": 0.87, "bbox": [300, 150, 500, 250]},
                {"class": "dog", "confidence": 0.92, "bbox": [50, 200, 150, 280]},
            ],
            "processing_time_ms": 1000,
            "model_version": "yolo-v8n",
            "image_size": "1920x1080",
        }

        objects = cast(list[dict[str, Any]], result["objects"])
        span.set_attribute("result.objects_count", len(objects))

        return result


# =============================================================================
# AUDIO & VOICE PROCESSING EXAMPLES
# =============================================================================


def transcribe_customer_call(audio_file: str) -> dict[str, Any]:
    """Transcribe customer service call using Whisper."""
    with trace_operation("speech_to_text") as span:
        span.set_attributes(
            {
                "model": "whisper-large-v3",
                "provider": "openai",
                "language": "auto-detect",
                "audio_file": audio_file,
            }
        )

        print(f"Transcribing audio: {audio_file}")

        # Simulate Whisper API call
        time.sleep(3)

        result = {
            "text": "Hello, I'm calling about my recent order. I haven't received it yet and it's been over a week.",
            "language": "en",
            "confidence": 0.94,
            "duration_ms": 8500,
            "processing_time_ms": 3000,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello, I'm calling about my recent order.",
                },
                {
                    "start": 2.5,
                    "end": 8.5,
                    "text": "I haven't received it yet and it's been over a week.",
                },
            ],
        }

        span.set_attribute("result.confidence", result["confidence"])
        span.set_attribute("result.language", result["language"])

        return result


def generate_customer_response(
    text: str, emotion: str = "professional"
) -> dict[str, Any]:
    """Generate spoken response for customer service."""
    with trace_operation("text_to_speech") as span:
        span.set_attributes(
            {
                "model": "eleven-labs-v2",
                "provider": "eleven-labs",
                "voice_id": "premium_voice_001",
                "emotion": emotion,
            }
        )

        print(f"Generating speech: {text[:50]}...")

        # Simulate TTS API call
        time.sleep(2)

        result = {
            "audio_url": f"https://example.com/generated_audio_{hash(text)}.wav",
            "duration_ms": len(text) * 50,  # Rough estimate
            "voice_id": "premium_voice_001",
            "emotion": emotion,
            "format": "WAV",
            "sample_rate": 44100,
            "generation_time_ms": 2000,
            "cost": 0.02,
        }

        span.set_attribute("result.audio_url", result["audio_url"])
        span.set_attribute("result.cost", result["cost"])

        return result


def handle_customer_service_call(audio_input: str) -> dict[str, Any]:
    """Handle complete customer service voice interaction."""
    with trace_agent_operation(
        agent_type="customer_service", operation="voice_interaction"
    ) as span:
        span.set_attributes(
            {
                "capabilities": [
                    "speech_recognition",
                    "nlp",
                    "speech_synthesis",
                    "sentiment_analysis",
                ],
                "language": "en-US",
            }
        )

        print("Processing customer service call...")

        # Step 1: Transcribe customer input
        transcription = transcribe_customer_call(audio_input)

        # Step 2: Analyze sentiment and intent
        sentiment_analysis = analyze_customer_sentiment(transcription["text"])

        # Step 3: Generate appropriate response
        response_text = generate_service_response(
            transcription["text"], sentiment_analysis["sentiment"]
        )

        # Step 4: Convert response to speech
        audio_response = generate_customer_response(
            response_text,
            emotion=(
                "empathetic"
                if sentiment_analysis["sentiment"] == "negative"
                else "professional"
            ),
        )

        result = {
            "transcription": transcription,
            "sentiment_analysis": sentiment_analysis,
            "response_text": response_text,
            "audio_response": audio_response,
            "call_id": f"call_{int(time.time())}",
            "total_processing_time_ms": 7000,
            "resolution_status": "resolved",
        }

        span.set_attribute("result.resolution_status", result["resolution_status"])

        return result


# =============================================================================
# VIDEO PROCESSING EXAMPLES
# =============================================================================


def create_product_demo_video(script: str, style: str = "commercial") -> dict[str, Any]:
    """Generate product demonstration video."""
    with trace_operation("video_generation") as span:
        span.set_attributes(
            {
                "model": "runway-gen2",
                "provider": "runway",
                "duration_seconds": 10,
                "style": style,
            }
        )

        print(f"Creating video: {script[:50]}...")

        # Simulate video generation
        time.sleep(15)  # Video generation takes longer

        result = {
            "video_url": f"https://example.com/generated_video_{hash(script)}.mp4",
            "duration_ms": 10000,
            "resolution": "1920x1080",
            "fps": 30,
            "format": "MP4",
            "file_size_bytes": 25000000,  # ~25MB
            "generation_time_ms": 15000,
            "cost": 0.50,
            "style": style,
        }

        span.set_attribute("result.video_url", result["video_url"])
        span.set_attribute("result.cost", result["cost"])

        return result


def moderate_video_content(video_path: str) -> dict[str, Any]:
    """Analyze video content for moderation."""
    with trace_operation("video_analysis") as span:
        span.set_attributes(
            {
                "analysis_type": "content_moderation",
                "model": "video-analyzer-v2",
                "provider": "custom",
                "video_path": video_path,
            }
        )

        print(f"Analyzing video content: {video_path}")

        # Simulate video analysis
        time.sleep(5)

        result = {
            "moderation_result": "approved",
            "confidence": 0.96,
            "flags": [],
            "content_categories": ["product_demo", "commercial"],
            "inappropriate_content": False,
            "analysis_time_ms": 5000,
            "frames_analyzed": 300,
            "audio_analysis": {
                "inappropriate_language": False,
                "volume_levels": "normal",
            },
        }

        span.set_attribute("result.moderation_result", result["moderation_result"])
        span.set_attribute("result.confidence", result["confidence"])

        return result


# =============================================================================
# MULTIMODAL LLM EXAMPLES
# =============================================================================


def analyze_product_image(image_path: str, analysis_request: str) -> dict[str, Any]:
    """Analyze product image using GPT-4 Vision."""
    with trace_llm_call(model="gpt-4-vision", provider="openai") as span:
        span.set_attributes(
            {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
                "image_path": image_path,
                "analysis_request": analysis_request,
            }
        )

        print(f"Analyzing image with GPT-4 Vision: {analysis_request}")

        # Simulate multimodal LLM call
        time.sleep(3)

        result = {
            "analysis": "This image shows a modern wireless headphone in matte black finish. The product appears to be professionally photographed with good lighting. Key features visible include: over-ear design, adjustable headband, and premium build quality. The image would be suitable for e-commerce use.",
            "confidence": 0.92,
            "detected_objects": ["headphones", "product"],
            "image_quality_score": 8.5,
            "marketing_suitability": "high",
            "processing_time_ms": 3000,
            "tokens_used": 150,
            "cost": 0.008,
        }

        span.set_attribute("llm.tokens_used", result["tokens_used"])
        span.set_attribute("llm.cost", result["cost"])

        return result


def create_social_media_post(
    image_path: str, audio_description: str, target_audience: str
) -> dict[str, Any]:
    """Create social media post from image and audio description."""
    with trace_llm_call(model="claude-3-vision", provider="anthropic") as span:
        span.set_attributes(
            {
                "input_modalities": ["text", "image", "audio"],
                "output_modalities": ["text"],
                "image_path": image_path,
                "target_audience": target_audience,
            }
        )

        print(f"Creating social media post for {target_audience}")

        # Simulate multimodal processing
        time.sleep(4)

        result = {
            "post_text": "🎧 Experience premium sound quality with our latest wireless headphones! Perfect for music lovers who demand the best. #AudioTech #WirelessFreedom #PremiumSound",
            "hashtags": [
                "#AudioTech",
                "#WirelessFreedom",
                "#PremiumSound",
                "#MusicLovers",
            ],
            "engagement_prediction": 8.2,
            "target_audience_match": 0.94,
            "tone": "enthusiastic",
            "processing_time_ms": 4000,
            "total_tokens": 200,
            "cost": 0.012,
        }

        span.set_attribute("llm.tokens_used", result["total_tokens"])
        span.set_attribute("llm.cost", result["cost"])

        return result


# =============================================================================
# COMPLEX MULTIMODAL WORKFLOWS
# =============================================================================


def create_marketing_campaign(
    product_name: str, target_audience: str, campaign_type: str
) -> dict[str, Any]:
    """Create complete marketing campaign with multimodal content."""
    with trace_agent_operation(
        agent_type="content_creator", operation="marketing_campaign_creation"
    ) as span:
        span.set_attributes(
            {
                "capabilities": [
                    "image_analysis",
                    "text_generation",
                    "video_creation",
                    "social_media",
                ],
                "product_name": product_name,
                "target_audience": target_audience,
                "campaign_type": campaign_type,
            }
        )

        print(f"Creating {campaign_type} campaign for {product_name}")

        campaign_results: dict[str, Any] = {}

        # Step 1: Generate product images
        product_images = []
        for style in ["lifestyle", "product_shot", "action"]:
            image = generate_product_image(
                f"{product_name} for {target_audience}", style
            )
            product_images.append(image)

        campaign_results["product_images"] = product_images

        # Step 2: Analyze best image for video creation
        best_image = product_images[0]  # Simplified selection
        image_analysis = analyze_product_image(
            best_image["image_url"],
            f"Analyze this {product_name} image for video marketing potential",
        )

        campaign_results["image_analysis"] = image_analysis

        # Step 3: Create promotional video
        video_script = f"Introducing the revolutionary {product_name} - designed specifically for {target_audience}"
        promo_video = create_product_demo_video(video_script, "commercial")

        campaign_results["promo_video"] = promo_video

        # Step 4: Generate social media content
        social_post = create_social_media_post(
            best_image["image_url"],
            f"Audio description of {product_name}",
            target_audience,
        )

        campaign_results["social_media"] = social_post

        # Step 5: Content moderation
        moderation_result = moderate_video_content(promo_video["video_url"])

        campaign_results["moderation"] = moderation_result

        result = {
            "campaign_id": f"campaign_{int(time.time())}",
            "product_name": product_name,
            "target_audience": target_audience,
            "campaign_type": campaign_type,
            "content_created": campaign_results,
            "total_cost": sum([img["cost"] for img in product_images])
            + promo_video["cost"]
            + social_post["cost"],
            "estimated_reach": 50000,
            "completion_status": "ready_for_review",
        }

        span.set_attribute("result.campaign_id", result["campaign_id"])
        span.set_attribute("result.total_cost", result["total_cost"])

        return result


# =============================================================================
# VOICE-FIRST AI ASSISTANT EXAMPLE
# =============================================================================


def voice_assistant_interaction(
    audio_input: str, context_images: list[str] = None
) -> dict[str, Any]:
    """Handle complex voice assistant interaction with visual context."""
    with trace_agent_operation(
        agent_type="voice_assistant", operation="multimodal_interaction"
    ) as span:
        span.set_attributes(
            {
                "capabilities": [
                    "speech_recognition",
                    "multimodal_understanding",
                    "task_execution",
                    "speech_synthesis",
                ],
            }
        )

        print("Processing voice assistant interaction...")

        # Step 1: Transcribe voice input
        transcription = transcribe_customer_call(audio_input)
        user_request = transcription["text"]

        # Step 2: Analyze any provided images for context
        visual_context = []
        if context_images:
            for image_path in context_images:
                analysis = analyze_product_image(
                    image_path,
                    f"Analyze this image in context of user request: {user_request}",
                )
                visual_context.append(analysis)

        # Step 3: Process request with multimodal understanding
        response_text = process_multimodal_request(user_request, visual_context)

        # Step 4: Generate spoken response
        audio_response = generate_customer_response(response_text, "helpful")

        result = {
            "interaction_id": f"voice_interaction_{int(time.time())}",
            "user_request": user_request,
            "visual_context": visual_context,
            "response_text": response_text,
            "audio_response": audio_response,
            "interaction_type": "multimodal_voice",
            "success": True,
            "total_processing_time_ms": 8000,
        }

        span.set_attribute("result.success", result["success"])

        return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def analyze_customer_sentiment(text: str) -> dict[str, Any]:
    """Analyze customer sentiment from text."""
    with trace_llm_call(model="gpt-4", provider="openai") as span:
        span.set_attribute("llm.operation", "sentiment_analysis")

        print(f"Analyzing sentiment: {text[:50]}...")
        time.sleep(1)

        result = {
            "sentiment": (
                "negative" if "haven't received" in text.lower() else "neutral"
            ),
            "confidence": 0.89,
            "emotions": ["frustration", "concern"],
            "urgency": "medium",
        }

        span.set_attribute("result.sentiment", result["sentiment"])
        span.set_attribute("result.confidence", result["confidence"])

        return result


def generate_service_response(customer_text: str, sentiment: str) -> str:
    """Generate appropriate customer service response."""
    with trace_llm_call(model="gpt-4", provider="openai") as span:
        span.set_attributes(
            {
                "llm.operation": "response_generation",
                "sentiment": sentiment,
            }
        )

        print(f"Generating response for {sentiment} sentiment...")
        time.sleep(1)

        if sentiment == "negative":
            response = "I sincerely apologize for the delay with your order. Let me look into this immediately and provide you with a tracking update and expedited shipping at no extra cost."
        else:
            response = "Thank you for contacting us. I'd be happy to help you with your inquiry. Let me check on that for you right away."

        span.set_attribute("result.response_length", len(response))

        return response


def process_multimodal_request(text_request: str, visual_context: list[dict[str, Any]]) -> str:
    """Process multimodal request combining text and visual information."""
    with trace_llm_call(model="gpt-4", provider="openai") as span:
        span.set_attributes(
            {
                "llm.operation": "multimodal_processing",
                "has_visual_context": bool(visual_context),
                "visual_context_count": len(visual_context) if visual_context else 0,
            }
        )

        print("Processing multimodal request...")
        time.sleep(2)

        context_summary = "Based on the images provided, " if visual_context else ""
        response = f"{context_summary}I can help you with that request. Here's what I found based on your question: '{text_request}'"

        span.set_attribute("result.response_length", len(response))

        return response


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


def run_multimodal_examples() -> None:
    """Run all multimodal examples to demonstrate capabilities."""
    print("🎯 Running Multimodal Noveum Trace SDK Examples\n")

    # Example 1: Image Generation Workflow
    print("📸 Example 1: Product Image Generation")
    image_result = generate_product_image(
        "wireless headphones", "commercial photography"
    )
    print(f"Generated image: {image_result['image_url']}\n")

    # Example 2: Voice Bot Interaction
    print("🎤 Example 2: Customer Service Voice Bot")
    voice_result = handle_customer_service_call("customer_complaint.wav")
    print(f"Call resolved: {voice_result['resolution_status']}\n")

    # Example 3: Video Content Creation
    print("🎬 Example 3: Product Demo Video")
    video_result = create_product_demo_video(
        "Showcase the amazing features of our new headphones"
    )
    print(f"Video created: {video_result['video_url']}\n")

    # Example 4: Multimodal Analysis
    print("🔍 Example 4: Multimodal Product Analysis")
    analysis_result = analyze_product_image(
        "headphones.jpg", "Analyze this product for marketing potential"
    )
    print(f"Marketing suitability: {analysis_result['marketing_suitability']}\n")

    # Example 5: Complete Marketing Campaign
    print("🚀 Example 5: Complete Marketing Campaign")
    campaign_result = create_marketing_campaign(
        "UltraSound Pro Headphones", "young professionals", "product_launch"
    )
    print(f"Campaign created: {campaign_result['campaign_id']}")
    print(f"Total cost: ${campaign_result['total_cost']:.2f}")
    print(f"Estimated reach: {campaign_result['estimated_reach']:,} people\n")

    # Example 6: Voice Assistant with Visual Context
    print("🤖 Example 6: Multimodal Voice Assistant")
    assistant_result = voice_assistant_interaction(
        "user_question.wav", context_images=["product1.jpg", "product2.jpg"]
    )
    print(f"Interaction completed: {assistant_result['interaction_id']}\n")

    print("✅ All multimodal examples completed successfully!")
    print(
        "\n📊 Trace data has been sent to Noveum API for analysis and dataset creation."
    )


if __name__ == "__main__":
    run_multimodal_examples()
