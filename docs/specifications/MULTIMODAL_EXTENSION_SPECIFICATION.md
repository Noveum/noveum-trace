# Noveum Trace SDK - Multimodal Extension Specification

## 🎯 Executive Summary

This document outlines a comprehensive strategy for extending the Noveum Trace SDK to support multimodal AI applications including images, voice bots, videos, and other generative AI models. The extension maintains backward compatibility while adding powerful new capabilities for tracing complex multimodal workflows.

## 📋 Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Multimodal Extension Strategy](#multimodal-extension-strategy)
3. [New Decorators & APIs](#new-decorators--apis)
4. [Data Schema Extensions](#data-schema-extensions)
5. [Binary Data Handling](#binary-data-handling)
6. [Integration Examples](#integration-examples)
7. [Performance Considerations](#performance-considerations)
8. [Implementation Roadmap](#implementation-roadmap)

## 🏗️ Current Architecture Analysis

### Existing Decorator Structure
```
noveum_trace/decorators/
├── base.py          # Core decorator functionality
├── llm.py           # LLM-specific tracing
├── agent.py         # Agent workflow tracing
├── tool.py          # Tool usage tracing
└── retrieval.py     # Retrieval operation tracing
```

### Extension Points Identified
1. **Span Attributes**: Already flexible with `Dict[str, Any]`
2. **Decorator Pattern**: Easily extensible for new modalities
3. **Event System**: Can capture multimodal events
4. **Transport Layer**: Can handle different data types

## 🚀 Multimodal Extension Strategy

### 1. New Modality Support

#### 1.1 Image Generation & Processing
```python
@trace_image_generation(
    model="dall-e-3",
    capture_inputs=True,
    capture_outputs=True,
    store_images=True
)
def generate_image(prompt: str, style: str = "photorealistic"):
    # Image generation logic
    return generated_image

@trace_image_processing(
    operation="object_detection",
    model="yolo-v8"
)
def detect_objects(image_path: str):
    # Object detection logic
    return detected_objects
```

#### 1.2 Voice & Audio Processing
```python
@trace_speech_to_text(
    model="whisper-large-v3",
    language="auto-detect"
)
def transcribe_audio(audio_file: str):
    # Speech-to-text logic
    return transcription

@trace_text_to_speech(
    model="eleven-labs",
    voice_id="premium_voice"
)
def synthesize_speech(text: str, voice_settings: dict):
    # Text-to-speech logic
    return audio_file

@trace_voice_bot(
    bot_type="conversational",
    capabilities=["speech_recognition", "nlp", "speech_synthesis"]
)
def handle_voice_interaction(audio_input: bytes):
    # Voice bot processing
    return audio_response
```

#### 1.3 Video Generation & Analysis
```python
@trace_video_generation(
    model="runway-gen2",
    duration_seconds=10
)
def generate_video(prompt: str, style_reference: str):
    # Video generation logic
    return video_file

@trace_video_analysis(
    analysis_type="scene_detection",
    model="video-analyzer-v1"
)
def analyze_video_content(video_path: str):
    # Video analysis logic
    return analysis_results
```

#### 1.4 Multimodal AI Models
```python
@trace_multimodal_llm(
    model="gpt-4-vision",
    modalities=["text", "image"]
)
def analyze_image_with_text(image: bytes, question: str):
    # Multimodal analysis
    return analysis_result

@trace_multimodal_generation(
    model="dall-e-3-with-gpt4",
    input_modalities=["text"],
    output_modalities=["image", "text"]
)
def generate_image_with_description(prompt: str):
    # Generate image and description
    return {"image": image_data, "description": description}
```

### 2. New Decorator Implementations

#### 2.1 Image Generation Decorator
```python
# noveum_trace/decorators/image.py

from typing import Optional, Dict, Any, Union, List
import functools
import base64
import hashlib
from PIL import Image
import io

def trace_image_generation(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    store_images: bool = False,
    image_storage_strategy: str = "hash_reference",  # "hash_reference", "base64", "url"
    max_image_size_mb: float = 10.0,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Decorator for tracing image generation operations.

    Args:
        model: Image generation model name
        provider: Model provider (openai, stability, midjourney, etc.)
        capture_inputs: Whether to capture input prompts and parameters
        capture_outputs: Whether to capture output metadata
        store_images: Whether to store actual image data
        image_storage_strategy: How to handle image storage
        max_image_size_mb: Maximum image size to store
        metadata: Additional metadata
        tags: Tags for categorization
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from noveum_trace import get_client, is_initialized
            from noveum_trace.core.context import get_current_trace

            if not is_initialized():
                return func(*args, **kwargs)

            client = get_client()
            trace = get_current_trace()

            # Auto-create trace if none exists
            auto_trace = None
            if trace is None:
                auto_trace = client.start_trace(f"image_generation_{func.__name__}")
                trace = auto_trace

            # Create span for image generation
            span = client.start_span(
                name=f"image_generation.{func.__name__}",
                attributes={
                    "image.operation": "generation",
                    "image.model": model,
                    "image.provider": provider,
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                }
            )

            # Add metadata and tags
            if metadata:
                span.set_attributes(metadata)
            if tags:
                for key, value in tags.items():
                    span.add_tag(key, value)

            try:
                # Capture inputs
                if capture_inputs:
                    input_data = _extract_image_generation_inputs(args, kwargs)
                    span.set_attributes({
                        "image.prompt": input_data.get("prompt"),
                        "image.style": input_data.get("style"),
                        "image.size": input_data.get("size"),
                        "image.quality": input_data.get("quality"),
                        "image.n_images": input_data.get("n_images", 1),
                    })

                # Execute function
                result = func(*args, **kwargs)

                # Capture outputs
                if capture_outputs:
                    output_data = _extract_image_generation_outputs(result)
                    span.set_attributes({
                        "image.output_format": output_data.get("format"),
                        "image.output_size": output_data.get("size"),
                        "image.output_dimensions": output_data.get("dimensions"),
                        "image.file_size_bytes": output_data.get("file_size"),
                    })

                    # Store image data if requested
                    if store_images and output_data.get("image_data"):
                        image_reference = _store_image_data(
                            output_data["image_data"],
                            image_storage_strategy,
                            max_image_size_mb
                        )
                        span.set_attribute("image.stored_reference", image_reference)

                span.set_status("ok")
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status("error", str(e))
                raise
            finally:
                client.finish_span(span)
                if auto_trace:
                    client.finish_trace(auto_trace)

        return wrapper

    # Handle both @trace_image_generation and @trace_image_generation() usage
    if callable(model):
        func = model
        model = None
        return decorator(func)

    return decorator

def _extract_image_generation_inputs(args, kwargs) -> Dict[str, Any]:
    """Extract relevant inputs from function arguments."""
    inputs = {}

    # Common parameter extraction
    if args:
        inputs["prompt"] = str(args[0]) if args[0] else None

    # Extract from kwargs
    for key in ["prompt", "style", "size", "quality", "n", "n_images"]:
        if key in kwargs:
            inputs[key] = kwargs[key]

    return inputs

def _extract_image_generation_outputs(result) -> Dict[str, Any]:
    """Extract relevant outputs from function result."""
    outputs = {}

    if isinstance(result, dict):
        # Handle dictionary results
        if "image" in result:
            outputs["image_data"] = result["image"]
        if "url" in result:
            outputs["image_url"] = result["url"]
        if "format" in result:
            outputs["format"] = result["format"]
    elif hasattr(result, 'save'):
        # Handle PIL Image objects
        outputs["image_data"] = result
        outputs["format"] = getattr(result, 'format', 'unknown')
        outputs["dimensions"] = f"{result.width}x{result.height}"
    elif isinstance(result, (str, bytes)):
        # Handle raw image data or file paths
        outputs["image_data"] = result

    return outputs

def _store_image_data(image_data, strategy: str, max_size_mb: float) -> str:
    """Store image data according to the specified strategy."""

    if strategy == "hash_reference":
        # Store hash reference only
        if isinstance(image_data, bytes):
            image_hash = hashlib.sha256(image_data).hexdigest()
        else:
            # Convert PIL Image to bytes for hashing
            img_bytes = io.BytesIO()
            image_data.save(img_bytes, format='PNG')
            image_hash = hashlib.sha256(img_bytes.getvalue()).hexdigest()

        return f"hash:{image_hash}"

    elif strategy == "base64":
        # Store as base64 (size limited)
        if hasattr(image_data, 'save'):
            # PIL Image
            img_bytes = io.BytesIO()
            image_data.save(img_bytes, format='PNG')
            img_data = img_bytes.getvalue()
        else:
            img_data = image_data

        # Check size limit
        size_mb = len(img_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            return f"size_exceeded:{size_mb:.2f}MB"

        return f"base64:{base64.b64encode(img_data).decode('utf-8')}"

    elif strategy == "url":
        # Store as URL reference (if available)
        if isinstance(image_data, str) and image_data.startswith(('http://', 'https://')):
            return f"url:{image_data}"
        else:
            return "url:not_available"

    return "unknown_strategy"
```

#### 2.2 Voice Bot Decorator
```python
# noveum_trace/decorators/voice.py

def trace_voice_bot(
    bot_type: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    language: Optional[str] = None,
    capture_audio: bool = False,
    audio_storage_strategy: str = "hash_reference",
    metadata: Optional[Dict[str, Any]] = None,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Decorator for tracing voice bot interactions.

    Args:
        bot_type: Type of voice bot (conversational, command, etc.)
        capabilities: List of bot capabilities
        language: Primary language for the interaction
        capture_audio: Whether to capture audio data
        audio_storage_strategy: How to handle audio storage
        metadata: Additional metadata
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from noveum_trace import get_client, is_initialized
            from noveum_trace.core.context import get_current_trace

            if not is_initialized():
                return func(*args, **kwargs)

            client = get_client()
            trace = get_current_trace()

            # Auto-create trace if none exists
            auto_trace = None
            if trace is None:
                auto_trace = client.start_trace(f"voice_bot_{func.__name__}")
                trace = auto_trace

            # Create span for voice interaction
            span = client.start_span(
                name=f"voice_bot.{func.__name__}",
                attributes={
                    "voice.bot_type": bot_type,
                    "voice.capabilities": capabilities,
                    "voice.language": language,
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                }
            )

            try:
                # Add interaction start event
                span.add_event("voice_interaction_start", {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input_type": "audio"
                })

                # Execute function
                result = func(*args, **kwargs)

                # Capture interaction results
                if isinstance(result, dict):
                    span.set_attributes({
                        "voice.response_type": result.get("type", "unknown"),
                        "voice.confidence_score": result.get("confidence"),
                        "voice.intent_detected": result.get("intent"),
                        "voice.response_duration_ms": result.get("duration_ms"),
                    })

                # Add interaction end event
                span.add_event("voice_interaction_end", {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "output_type": "audio"
                })

                span.set_status("ok")
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status("error", str(e))
                raise
            finally:
                client.finish_span(span)
                if auto_trace:
                    client.finish_trace(auto_trace)

        return wrapper

    return decorator
```

#### 2.3 Multimodal LLM Decorator
```python
# noveum_trace/decorators/multimodal.py

def trace_multimodal_llm(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    input_modalities: Optional[List[str]] = None,
    output_modalities: Optional[List[str]] = None,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    store_media: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Decorator for tracing multimodal LLM operations.

    Args:
        model: Multimodal model name (gpt-4-vision, claude-3-vision, etc.)
        provider: Model provider
        input_modalities: List of input modalities (text, image, audio, video)
        output_modalities: List of output modalities
        capture_inputs: Whether to capture input data
        capture_outputs: Whether to capture output data
        store_media: Whether to store media data
        metadata: Additional metadata
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from noveum_trace import get_client, is_initialized
            from noveum_trace.core.context import get_current_trace

            if not is_initialized():
                return func(*args, **kwargs)

            client = get_client()
            trace = get_current_trace()

            # Auto-create trace if none exists
            auto_trace = None
            if trace is None:
                auto_trace = client.start_trace(f"multimodal_llm_{func.__name__}")
                trace = auto_trace

            # Create span for multimodal operation
            span = client.start_span(
                name=f"multimodal_llm.{func.__name__}",
                attributes={
                    "llm.model": model,
                    "llm.provider": provider,
                    "llm.type": "multimodal",
                    "llm.input_modalities": input_modalities,
                    "llm.output_modalities": output_modalities,
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                }
            )

            try:
                # Capture multimodal inputs
                if capture_inputs:
                    input_data = _extract_multimodal_inputs(args, kwargs)
                    span.set_attributes({
                        "llm.text_input": input_data.get("text"),
                        "llm.image_count": input_data.get("image_count", 0),
                        "llm.audio_duration_ms": input_data.get("audio_duration_ms"),
                        "llm.video_duration_ms": input_data.get("video_duration_ms"),
                        "llm.total_input_size_bytes": input_data.get("total_size_bytes"),
                    })

                # Execute function
                result = func(*args, **kwargs)

                # Capture multimodal outputs
                if capture_outputs:
                    output_data = _extract_multimodal_outputs(result)
                    span.set_attributes({
                        "llm.text_output": output_data.get("text"),
                        "llm.output_modalities_detected": output_data.get("modalities"),
                        "llm.confidence_scores": output_data.get("confidence_scores"),
                        "llm.total_output_size_bytes": output_data.get("total_size_bytes"),
                    })

                span.set_status("ok")
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status("error", str(e))
                raise
            finally:
                client.finish_span(span)
                if auto_trace:
                    client.finish_trace(auto_trace)

        return wrapper

    return decorator

def _extract_multimodal_inputs(args, kwargs) -> Dict[str, Any]:
    """Extract multimodal input information."""
    inputs = {}
    total_size = 0

    # Analyze arguments for different modalities
    for arg in args:
        if isinstance(arg, str):
            inputs["text"] = arg[:500]  # Truncate for storage
            total_size += len(arg.encode('utf-8'))
        elif isinstance(arg, bytes):
            # Could be image, audio, or video data
            total_size += len(arg)
        elif hasattr(arg, 'read'):
            # File-like object
            try:
                size = arg.seek(0, 2)  # Seek to end to get size
                arg.seek(0)  # Reset position
                total_size += size
            except:
                pass

    # Check kwargs for specific modality indicators
    if 'images' in kwargs:
        inputs["image_count"] = len(kwargs['images']) if isinstance(kwargs['images'], list) else 1

    inputs["total_size_bytes"] = total_size
    return inputs

def _extract_multimodal_outputs(result) -> Dict[str, Any]:
    """Extract multimodal output information."""
    outputs = {}

    if isinstance(result, dict):
        outputs["text"] = result.get("text", "")[:500]  # Truncate
        outputs["modalities"] = list(result.keys())

        # Calculate total output size
        total_size = 0
        for key, value in result.items():
            if isinstance(value, (str, bytes)):
                total_size += len(value) if isinstance(value, bytes) else len(value.encode('utf-8'))
        outputs["total_size_bytes"] = total_size

    return outputs
```

### 3. Enhanced Span Attributes for Multimodal Data

#### 3.1 Image Attributes
```python
# Standard image attributes
IMAGE_ATTRIBUTES = {
    "image.operation": str,           # generation, processing, analysis
    "image.model": str,               # dall-e-3, stable-diffusion, etc.
    "image.provider": str,            # openai, stability, midjourney
    "image.prompt": str,              # Generation prompt
    "image.style": str,               # Style parameters
    "image.size": str,                # Image dimensions
    "image.format": str,              # PNG, JPEG, WebP
    "image.quality": str,             # Quality settings
    "image.file_size_bytes": int,     # File size
    "image.generation_time_ms": float, # Generation time
    "image.cost": float,              # Generation cost
    "image.stored_reference": str,    # Storage reference
}
```

#### 3.2 Audio Attributes
```python
# Standard audio attributes
AUDIO_ATTRIBUTES = {
    "audio.operation": str,           # transcription, synthesis, analysis
    "audio.model": str,               # whisper, eleven-labs, etc.
    "audio.provider": str,            # openai, eleven-labs, google
    "audio.language": str,            # Language code
    "audio.duration_ms": float,       # Audio duration
    "audio.sample_rate": int,         # Sample rate
    "audio.channels": int,            # Number of channels
    "audio.format": str,              # MP3, WAV, FLAC
    "audio.file_size_bytes": int,     # File size
    "audio.transcription": str,       # Transcribed text
    "audio.confidence_score": float,  # Transcription confidence
    "audio.voice_id": str,            # Voice identifier for TTS
    "audio.cost": float,              # Processing cost
}
```

#### 3.3 Video Attributes
```python
# Standard video attributes
VIDEO_ATTRIBUTES = {
    "video.operation": str,           # generation, analysis, processing
    "video.model": str,               # runway, pika, etc.
    "video.provider": str,            # runway, pika, openai
    "video.prompt": str,              # Generation prompt
    "video.duration_ms": float,       # Video duration
    "video.fps": int,                 # Frames per second
    "video.resolution": str,          # Video resolution
    "video.format": str,              # MP4, AVI, MOV
    "video.file_size_bytes": int,     # File size
    "video.frame_count": int,         # Total frames
    "video.generation_time_ms": float, # Generation time
    "video.cost": float,              # Generation cost
    "video.analysis_results": dict,   # Analysis results
}
```

### 4. Binary Data Handling Strategy

#### 4.1 Storage Strategies
```python
class MediaStorageStrategy(Enum):
    """Strategies for handling binary media data."""
    HASH_REFERENCE = "hash_reference"    # Store hash only
    BASE64_INLINE = "base64_inline"      # Store as base64 (size limited)
    URL_REFERENCE = "url_reference"      # Store URL reference
    EXTERNAL_STORAGE = "external_storage" # Upload to external storage
    METADATA_ONLY = "metadata_only"      # Store metadata only
```

#### 4.2 Media Storage Manager
```python
# noveum_trace/utils/media_storage.py

class MediaStorageManager:
    """Manages storage of binary media data in traces."""

    def __init__(self, config: MediaStorageConfig):
        self.config = config
        self.storage_backends = {
            "s3": S3StorageBackend(config.s3_config),
            "gcs": GCSStorageBackend(config.gcs_config),
            "azure": AzureStorageBackend(config.azure_config),
        }

    def store_media(
        self,
        media_data: Union[bytes, str],
        media_type: str,
        strategy: MediaStorageStrategy
    ) -> str:
        """Store media data according to strategy."""

        if strategy == MediaStorageStrategy.HASH_REFERENCE:
            return self._create_hash_reference(media_data)

        elif strategy == MediaStorageStrategy.BASE64_INLINE:
            return self._store_as_base64(media_data, media_type)

        elif strategy == MediaStorageStrategy.URL_REFERENCE:
            return self._extract_url_reference(media_data)

        elif strategy == MediaStorageStrategy.EXTERNAL_STORAGE:
            return self._upload_to_external_storage(media_data, media_type)

        elif strategy == MediaStorageStrategy.METADATA_ONLY:
            return self._extract_metadata_only(media_data, media_type)

        return "unknown_strategy"

    def _create_hash_reference(self, media_data: Union[bytes, str]) -> str:
        """Create a hash reference for the media."""
        if isinstance(media_data, str):
            media_data = media_data.encode('utf-8')

        media_hash = hashlib.sha256(media_data).hexdigest()
        return f"hash:{media_hash}"

    def _store_as_base64(self, media_data: Union[bytes, str], media_type: str) -> str:
        """Store media as base64 with size limits."""
        if isinstance(media_data, str):
            media_data = media_data.encode('utf-8')

        size_mb = len(media_data) / (1024 * 1024)
        if size_mb > self.config.max_inline_size_mb:
            return f"size_exceeded:{size_mb:.2f}MB"

        b64_data = base64.b64encode(media_data).decode('utf-8')
        return f"base64:{media_type}:{b64_data}"

    def _upload_to_external_storage(self, media_data: bytes, media_type: str) -> str:
        """Upload to configured external storage."""
        backend = self.storage_backends.get(self.config.default_backend)
        if not backend:
            return "no_backend_configured"

        try:
            url = backend.upload(media_data, media_type)
            return f"external:{self.config.default_backend}:{url}"
        except Exception as e:
            return f"upload_failed:{str(e)}"

@dataclass
class MediaStorageConfig:
    """Configuration for media storage."""
    default_strategy: MediaStorageStrategy = MediaStorageStrategy.HASH_REFERENCE
    max_inline_size_mb: float = 10.0
    default_backend: str = "s3"
    s3_config: Optional[Dict] = None
    gcs_config: Optional[Dict] = None
    azure_config: Optional[Dict] = None
```

### 5. Integration Examples

#### 5.1 Complete Image Generation Workflow
```python
import noveum_trace
from noveum_trace import trace_image_generation, trace_llm, trace

# Initialize SDK
noveum_trace.init(
    api_key="your-api-key",
    project="multimodal-app",
    config={
        "media_storage": {
            "default_strategy": "hash_reference",
            "max_inline_size_mb": 5.0
        }
    }
)

@trace_image_generation(
    model="dall-e-3",
    provider="openai",
    store_images=True,
    image_storage_strategy="base64"
)
def generate_product_image(description: str, style: str = "photorealistic"):
    """Generate product images using DALL-E 3."""
    import openai

    response = openai.images.generate(
        model="dall-e-3",
        prompt=f"{description} in {style} style",
        size="1024x1024",
        quality="standard",
        n=1
    )

    return {
        "image_url": response.data[0].url,
        "revised_prompt": response.data[0].revised_prompt,
        "format": "PNG",
        "size": "1024x1024"
    }

@trace_llm(model="gpt-4", provider="openai")
def enhance_image_prompt(basic_prompt: str):
    """Enhance image generation prompt using GPT-4."""
    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert at creating detailed image generation prompts."},
            {"role": "user", "content": f"Enhance this image prompt for better results: {basic_prompt}"}
        ]
    )

    return response.choices[0].message.content

@trace(name="complete_image_generation_workflow")
def create_marketing_image(product_name: str, target_audience: str):
    """Complete workflow for creating marketing images."""

    # Step 1: Create enhanced prompt
    basic_prompt = f"Professional product photo of {product_name} for {target_audience}"
    enhanced_prompt = enhance_image_prompt(basic_prompt)

    # Step 2: Generate image
    image_result = generate_product_image(enhanced_prompt, "commercial photography")

    # Step 3: Return complete result
    return {
        "original_prompt": basic_prompt,
        "enhanced_prompt": enhanced_prompt,
        "image_url": image_result["image_url"],
        "revised_prompt": image_result["revised_prompt"]
    }

# Usage
result = create_marketing_image("wireless headphones", "young professionals")
```

#### 5.2 Voice Bot Conversation Flow
```python
from noveum_trace import trace_voice_bot, trace_speech_to_text, trace_text_to_speech

@trace_speech_to_text(
    model="whisper-large-v3",
    language="auto-detect"
)
def transcribe_user_input(audio_file: str):
    """Transcribe user voice input."""
    import whisper

    model = whisper.load_model("large-v3")
    result = model.transcribe(audio_file)

    return {
        "text": result["text"],
        "language": result["language"],
        "confidence": result.get("confidence", 0.0)
    }

@trace_llm(model="gpt-4", provider="openai")
def process_user_intent(transcribed_text: str):
    """Process user intent from transcribed text."""
    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant. Respond naturally and concisely."},
            {"role": "user", "content": transcribed_text}
        ]
    )

    return response.choices[0].message.content

@trace_text_to_speech(
    model="eleven-labs",
    voice_id="premium_voice"
)
def synthesize_response(response_text: str):
    """Convert text response to speech."""
    # ElevenLabs TTS implementation
    audio_data = eleven_labs_tts(response_text, voice_id="premium_voice")

    return {
        "audio_data": audio_data,
        "duration_ms": len(audio_data) / 16,  # Assuming 16kHz
        "format": "WAV"
    }

@trace_voice_bot(
    bot_type="conversational",
    capabilities=["speech_recognition", "nlp", "speech_synthesis"],
    language="en-US"
)
def handle_voice_conversation(audio_input_file: str):
    """Complete voice conversation handling."""

    # Step 1: Transcribe audio input
    transcription = transcribe_user_input(audio_input_file)

    # Step 2: Process intent and generate response
    response_text = process_user_intent(transcription["text"])

    # Step 3: Convert response to speech
    audio_response = synthesize_response(response_text)

    return {
        "transcription": transcription,
        "response_text": response_text,
        "audio_response": audio_response,
        "conversation_id": "conv_123",
        "duration_ms": audio_response["duration_ms"]
    }

# Usage
conversation_result = handle_voice_conversation("user_input.wav")
```

#### 5.3 Multimodal AI Analysis
```python
from noveum_trace import trace_multimodal_llm

@trace_multimodal_llm(
    model="gpt-4-vision",
    provider="openai",
    input_modalities=["text", "image"],
    output_modalities=["text"],
    store_media=True
)
def analyze_image_with_context(image_path: str, question: str, context: str = ""):
    """Analyze image with textual context using GPT-4 Vision."""
    import openai
    import base64

    # Encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Context: {context}\n\nQuestion: {question}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=500
    )

    return {
        "analysis": response.choices[0].message.content,
        "usage": response.usage.dict(),
        "model": "gpt-4-vision-preview"
    }

@trace(name="multimodal_content_analysis")
def analyze_social_media_post(image_path: str, caption: str):
    """Analyze social media post for brand safety and engagement potential."""

    # Analyze image content
    content_analysis = analyze_image_with_context(
        image_path,
        "Describe what you see in this image and identify any potential brand safety concerns.",
        f"This is a social media post with caption: '{caption}'"
    )

    # Analyze engagement potential
    engagement_analysis = analyze_image_with_context(
        image_path,
        "Rate the engagement potential of this image on a scale of 1-10 and explain why.",
        f"Social media context with caption: '{caption}'"
    )

    return {
        "content_analysis": content_analysis["analysis"],
        "engagement_analysis": engagement_analysis["analysis"],
        "total_tokens": content_analysis["usage"]["total_tokens"] + engagement_analysis["usage"]["total_tokens"]
    }

# Usage
analysis = analyze_social_media_post("post_image.jpg", "Check out our new product! #innovation")
```

### 6. Performance Considerations

#### 6.1 Binary Data Optimization
```python
# Configuration for handling large media files
MEDIA_PERFORMANCE_CONFIG = {
    "max_inline_size_mb": 5.0,           # Max size for inline storage
    "compression_enabled": True,          # Enable compression
    "compression_quality": 0.8,           # Compression quality
    "async_upload": True,                 # Async external uploads
    "batch_media_uploads": True,          # Batch multiple uploads
    "cdn_acceleration": True,             # Use CDN for media delivery
}
```

#### 6.2 Memory Management
```python
class MediaMemoryManager:
    """Manages memory usage for media processing."""

    def __init__(self, max_memory_mb: float = 500.0):
        self.max_memory_mb = max_memory_mb
        self.current_usage_mb = 0.0
        self.media_cache = {}

    def can_process_media(self, estimated_size_mb: float) -> bool:
        """Check if media can be processed within memory limits."""
        return (self.current_usage_mb + estimated_size_mb) <= self.max_memory_mb

    def process_with_memory_limit(self, media_data: bytes, processor_func):
        """Process media with memory management."""
        estimated_size = len(media_data) / (1024 * 1024)

        if not self.can_process_media(estimated_size):
            # Use streaming processing or external processing
            return self._stream_process(media_data, processor_func)

        return processor_func(media_data)
```

### 7. Backend API Extensions

#### 7.1 Enhanced Elasticsearch Schema
```json
{
  "mappings": {
    "properties": {
      "spans": {
        "type": "nested",
        "properties": {
          "attributes": {
            "properties": {
              "image.operation": {"type": "keyword"},
              "image.model": {"type": "keyword"},
              "image.provider": {"type": "keyword"},
              "image.size": {"type": "keyword"},
              "image.format": {"type": "keyword"},
              "image.file_size_bytes": {"type": "long"},
              "image.generation_time_ms": {"type": "float"},
              "image.cost": {"type": "float"},
              "image.stored_reference": {"type": "keyword"},

              "audio.operation": {"type": "keyword"},
              "audio.model": {"type": "keyword"},
              "audio.language": {"type": "keyword"},
              "audio.duration_ms": {"type": "float"},
              "audio.confidence_score": {"type": "float"},
              "audio.cost": {"type": "float"},

              "video.operation": {"type": "keyword"},
              "video.model": {"type": "keyword"},
              "video.duration_ms": {"type": "float"},
              "video.resolution": {"type": "keyword"},
              "video.fps": {"type": "integer"},
              "video.cost": {"type": "float"},

              "multimodal.input_modalities": {"type": "keyword"},
              "multimodal.output_modalities": {"type": "keyword"},
              "multimodal.total_cost": {"type": "float"}
            }
          }
        }
      }
    }
  }
}
```

#### 7.2 New API Endpoints
```python
# Additional endpoints for multimodal data

@app.get("/v1/analytics/multimodal-summary")
async def get_multimodal_analytics(
    project: str,
    modalities: List[str] = Query(None),
    time_range: str = "24h"
):
    """Get analytics summary for multimodal operations."""
    return {
        "summary": {
            "total_operations": 1250,
            "by_modality": {
                "image": 650,
                "audio": 400,
                "video": 150,
                "multimodal": 50
            },
            "total_cost": 45.67,
            "avg_processing_time_ms": 2340.5
        },
        "cost_breakdown": {
            "image_generation": 25.30,
            "audio_processing": 12.45,
            "video_generation": 7.92
        }
    }

@app.get("/v1/media/{media_reference}")
async def get_media_data(media_reference: str):
    """Retrieve stored media data by reference."""
    # Implementation for retrieving stored media
    pass

@app.post("/v1/datasets/multimodal")
async def create_multimodal_dataset(request: MultimodalDatasetRequest):
    """Create dataset specifically for multimodal evaluation."""
    return {
        "dataset_id": "dataset_multimodal_123",
        "modalities_included": ["text", "image", "audio"],
        "sample_count": 500,
        "estimated_completion": "2025-07-16T21:00:00Z"
    }
```

## 🗺️ Implementation Roadmap

### Phase 1: Core Multimodal Infrastructure (Weeks 1-4)
- [ ] **Week 1-2**: Implement media storage manager and strategies
- [ ] **Week 3-4**: Create base multimodal decorator framework

### Phase 2: Image & Audio Support (Weeks 5-8)
- [ ] **Week 5-6**: Implement image generation and processing decorators
- [ ] **Week 7-8**: Implement audio/voice processing decorators

### Phase 3: Video & Advanced Multimodal (Weeks 9-12)
- [ ] **Week 9-10**: Implement video processing decorators
- [ ] **Week 11-12**: Implement multimodal LLM decorators

### Phase 4: Integration & Optimization (Weeks 13-16)
- [ ] **Week 13-14**: Backend API extensions for multimodal data
- [ ] **Week 15-16**: Performance optimization and testing

### Phase 5: Advanced Features (Weeks 17-20)
- [ ] **Week 17-18**: Advanced analytics and visualization
- [ ] **Week 19-20**: Production deployment and monitoring

## 🎯 Success Metrics

### Technical Metrics
- **Media Processing Latency**: <200ms overhead for media handling
- **Storage Efficiency**: >80% compression for large media files
- **Memory Usage**: <500MB for typical multimodal workloads

### Business Metrics
- **Multimodal Coverage**: Support for 95% of common multimodal AI use cases
- **Developer Adoption**: 50% of users utilizing multimodal features within 6 months
- **Cost Efficiency**: <$0.01 per multimodal operation traced

## 🎉 Conclusion

The multimodal extension of the Noveum Trace SDK provides comprehensive support for modern AI applications that work with images, audio, video, and multiple modalities simultaneously. This extension maintains the SDK's core principles of simplicity and performance while adding powerful new capabilities for tracing complex multimodal workflows.

The implementation strategy ensures backward compatibility while providing a clear path for organizations to adopt multimodal tracing as their AI applications evolve beyond text-only interactions.

---

**Document Version**: 1.0
**Last Updated**: 2025-07-16
**Implementation Target**: Q4 2025
