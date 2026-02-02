"""
Utility functions for parsing and structuring LangChain message objects.

This module provides functions to convert LangChain messages into structured
dictionaries instead of stringifying them, significantly reducing trace size
and improving readability.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def message_to_dict(msg: Any) -> dict[str, Any]:
    """
    Convert LangChain message to dict using model_dump() or dict().

    Tries Pydantic v2 (model_dump) first, then falls back to v1 (dict),
    and finally to manual extraction if neither works.

    Args:
        msg: LangChain message object

    Returns:
        Dictionary representation of the message
    """
    try:
        # Try Pydantic v2 first
        if hasattr(msg, "model_dump") and callable(msg.model_dump):
            return msg.model_dump()
        # Fallback to Pydantic v1
        elif hasattr(msg, "dict") and callable(msg.dict):
            return msg.dict()
        # Last resort: manual extraction
        else:
            result = {
                "type": type(msg).__name__,
                "content": str(msg.content) if hasattr(msg, "content") else None,
            }
            # Add common optional fields
            if hasattr(msg, "id"):
                result["id"] = msg.id
            if hasattr(msg, "name"):
                result["name"] = msg.name
            return result
    except Exception as e:
        logger.debug(f"Error converting message to dict: {e}")
        return {
            "type": type(msg).__name__,
            "content": (
                str(msg.content)
                if hasattr(msg, "content")
                else "Error extracting content"
            ),
            "error": str(e),
        }


def is_langchain_message(obj: Any) -> bool:
    """
    Check if object is a LangChain message.

    All LangChain messages inherit from BaseMessage, so we check if
    'BaseMessage' is in the object's class hierarchy (MRO).

    Args:
        obj: Object to check

    Returns:
        True if object is a LangChain message, False otherwise
    """
    try:
        # Check if BaseMessage is in the Method Resolution Order (MRO)
        # This works for all message types: HumanMessage, AIMessage, etc.
        mro = type(obj).__mro__
        return any(
            cls.__name__ == "BaseMessage" and cls.__module__.startswith("langchain")
            for cls in mro
        )
    except Exception:
        return False


def parse_messages_list(messages: list[Any]) -> dict[str, list[Any]]:
    """
    Parse a list of LangChain messages into structured data.

    Separates messages into three categories:
    1. Regular messages (human, ai, system, chat, remove)
    2. Tool calls (extracted from AIMessage.tool_calls)
    3. Tool results (tool and function messages)

    Args:
        messages: List of LangChain message objects

    Returns:
        Dict with keys: 'messages', 'tool_calls', 'tool_results'
    """
    result: dict[str, list[Any]] = {
        "messages": [],
        "tool_calls": [],
        "tool_results": [],
    }

    for msg in messages:
        try:
            # Get message type
            msg_type = msg.type if hasattr(msg, "type") else type(msg).__name__.lower()

            # Handle regular messages (human, system, chat, remove)
            if msg_type in ["human", "system", "chat", "remove"]:
                result["messages"].append(message_to_dict(msg))

            # Handle AI messages (extract tool calls separately)
            elif msg_type == "ai":
                # Extract tool calls first
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            # Handle both dict and object tool calls
                            if isinstance(tc, dict):
                                result["tool_calls"].append(
                                    {
                                        "name": tc.get("name"),
                                        "args": tc.get("args"),
                                        "id": tc.get("id"),
                                    }
                                )
                            else:
                                result["tool_calls"].append(
                                    {
                                        "name": getattr(tc, "name", None),
                                        "args": getattr(tc, "args", None),
                                        "id": getattr(tc, "id", None),
                                    }
                                )
                        except Exception as e:
                            logger.debug(f"Error extracting tool call: {e}")

                # Add AI message without tool_calls to avoid duplication
                msg_dict = message_to_dict(msg)
                # Clear tool_calls and invalid_tool_calls to reduce size
                msg_dict["tool_calls"] = []
                if "invalid_tool_calls" in msg_dict:
                    msg_dict["invalid_tool_calls"] = []
                result["messages"].append(msg_dict)

            # Handle tool results
            elif msg_type in ["tool", "function"]:
                result["tool_results"].append(message_to_dict(msg))

            # Unknown message type - add to messages as-is
            else:
                logger.debug(f"Unknown message type: {msg_type}")
                result["messages"].append(message_to_dict(msg))

        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            # Add error placeholder to avoid losing the message entirely
            result["messages"].append(
                {
                    "type": "error",
                    "content": f"Error parsing message: {str(e)}",
                    "original_type": type(msg).__name__,
                }
            )

    return result


def process_chain_inputs_outputs(data: dict[str, Any]) -> dict[str, Any]:
    """
    Process chain inputs/outputs dict, detecting and parsing messages.

    For each key-value pair:
    - If value is a list of LangChain messages, parse into structured format
    - Otherwise, stringify the value (existing behavior)

    Args:
        data: Input or output dict from chain

    Returns:
        Dict of attributes ready for span, with keys like:
        - 'messages.messages': list of message dicts
        - 'messages.tool_calls': list of tool call dicts
        - 'messages.tool_results': list of tool result dicts
        - 'iteration': stringified value
        - etc.
    """
    attributes: dict[str, Any] = {}

    for key, value in data.items():
        try:
            # Check if value is a list of messages
            if isinstance(value, list) and value and is_langchain_message(value[0]):
                # Parse messages into structured format
                parsed = parse_messages_list(value)
                attributes[f"{key}.messages"] = parsed["messages"]
                attributes[f"{key}.tool_calls"] = parsed["tool_calls"]
                attributes[f"{key}.tool_results"] = parsed["tool_results"]
            else:
                # Keep other keys as stringified (existing behavior)
                attributes[key] = str(value)
        except Exception as e:
            logger.error(f"Error processing key '{key}': {e}")
            # Fallback to stringify if parsing fails
            attributes[key] = str(value)

    return attributes


def extract_images_from_messages(
    message_dicts: list[list[dict[str, Any]]],
) -> list[str]:
    """
    Extract image URLs and data URIs from message dictionaries.

    Looks for images in message content with type "image_url" that contain
    either HTTP/HTTPS URLs or data URIs (base64-encoded images).

    Args:
        message_dicts: List of message batches as dictionaries (already converted)

    Returns:
        List of image URLs/data URIs found in message content
    """
    images = []

    try:
        for batch in message_dicts:
            for msg in batch:
                # Get content from dict
                content = msg.get("content") if isinstance(msg, dict) else None

                if content is None:
                    continue

                if isinstance(content, list):
                    # Multimodal content (text + images)
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url = item.get("image_url")
                            if isinstance(image_url, dict):
                                url = image_url.get("url")
                            elif isinstance(image_url, str):
                                url = image_url
                            else:
                                url = None

                            if url and url not in images:
                                images.append(url)

    except Exception as e:
        logger.debug(f"Error extracting images from messages: {e}")

    return images


def download_image_from_url(url: str) -> Optional[tuple[bytes, str]]:
    """
    Download an image from a URL and return the image bytes and format.

    Args:
        url: HTTP/HTTPS URL of the image

    Returns:
        Tuple of (image_bytes, image_format) if successful, None otherwise
    """
    try:
        from io import BytesIO

        import requests
        from PIL import Image

        # Download image with timeout
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()

        # Read image data
        image_bytes = response.content

        # Detect image format using PIL
        try:
            img = Image.open(BytesIO(image_bytes))
            format_map = {"JPEG": "jpeg", "PNG": "png", "GIF": "gif", "WEBP": "webp"}
            # img.format can be None, so handle that case
            img_format = format_map.get(img.format or "JPEG", "jpeg")
        except Exception:
            # If PIL can't detect, validate Content-Type header
            content_type = response.headers.get("content-type", "").lower()

            # Validate that content-type starts with "image/"
            if not content_type or not content_type.startswith("image/"):
                raise ValueError(
                    f"Invalid or missing image content-type. "
                    f"Expected 'image/*', got: {content_type or 'None'}"
                ) from None

            # Extract image subtype and map to format
            if "png" in content_type:
                img_format = "png"
            elif "jpeg" in content_type or "jpg" in content_type:
                img_format = "jpeg"
            elif "gif" in content_type:
                img_format = "gif"
            elif "webp" in content_type:
                img_format = "webp"
            else:
                # If subtype is not recognized but starts with "image/", extract it
                # content_type might be like "image/svg+xml" or other formats
                subtype = content_type.split("/")[1].split(";")[0].strip()
                raise ValueError(
                    f"Unsupported image format: {subtype}. "
                    f"Supported formats: png, jpeg, gif, webp"
                ) from None

        return (image_bytes, img_format)

    except Exception as e:
        logger.warning(f"Failed to download image from URL {url}: {e}")
        return None


def extract_and_process_images(
    message_dicts: list[list[dict[str, Any]]],
    trace_id: str,
    span_id: str,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """
    Extract base64 images, upload them, and replace with UUID references.

    This function processes message dictionaries to:
    1. Find base64-encoded images (data:image/...)
    2. Upload each image to the platform
    3. Replace the base64 data with image:// UUID references
    4. Return the list of image UUIDs and modified messages

    Args:
        message_dicts: List of message batches as dictionaries (already converted)
        trace_id: Trace ID to associate images with
        span_id: Span ID to associate images with

    Returns:
        Tuple of (image_uuids, modified_message_dicts)
    """
    from noveum_trace import get_client
    from noveum_trace.utils.image_utils import (
        format_image_reference,
        generate_image_uuid,
        is_base64_image,
        parse_base64_image,
    )

    image_uuids = []
    modified_message_dicts = []

    try:
        # Get client for uploading images
        client = get_client()
        if not client:
            logger.debug("No client available, skipping image upload")
            return ([], message_dicts)

        # Deep copy message_dicts to avoid modifying the original
        import copy

        modified_message_dicts = copy.deepcopy(message_dicts)

        # Process each batch
        for batch in modified_message_dicts:
            for msg in batch:
                # Get content from dict
                content = msg.get("content") if isinstance(msg, dict) else None

                if content is None:
                    continue

                if isinstance(content, list):
                    # Multimodal content (text + images)
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url_data = item.get("image_url")

                            # Extract URL from image_url field
                            if isinstance(image_url_data, dict):
                                url = image_url_data.get("url")
                            elif isinstance(image_url_data, str):
                                url = image_url_data
                            else:
                                continue

                            if not url:
                                continue

                            # Process base64 images
                            if is_base64_image(url):
                                try:
                                    # Parse base64 image
                                    parsed = parse_base64_image(url)
                                    image_data = parsed["image_data"]
                                    image_format = parsed["format"]

                                    # Generate UUID for this image
                                    image_uuid = generate_image_uuid()

                                    # Prepare metadata
                                    metadata = {"format": image_format}

                                    # Upload image
                                    client.export_image(
                                        image_data=image_data,
                                        trace_id=trace_id,
                                        span_id=span_id,
                                        image_uuid=image_uuid,
                                        metadata=metadata,
                                    )

                                    # Add to list of UUIDs
                                    image_uuids.append(image_uuid)

                                    # Replace base64 with UUID reference
                                    image_reference = format_image_reference(image_uuid)
                                    if isinstance(image_url_data, dict):
                                        image_url_data["url"] = image_reference
                                    elif isinstance(image_url_data, str):
                                        item["image_url"] = image_reference

                                    logger.debug(
                                        f"Uploaded base64 image {image_uuid} and replaced in message"
                                    )

                                except Exception as e:
                                    logger.warning(
                                        f"Failed to process base64 image: {e}",
                                        exc_info=True,
                                    )
                                    # Keep original base64 if upload fails

                            # Process HTTP/HTTPS URLs
                            elif url.startswith(("http://", "https://")):
                                try:
                                    # Download image from URL
                                    download_result = download_image_from_url(url)
                                    if download_result is None:
                                        logger.debug(
                                            f"Skipping image URL (download failed): {url}"
                                        )
                                        continue

                                    image_data, image_format = download_result

                                    # Generate UUID for this image
                                    image_uuid = generate_image_uuid()

                                    # Prepare metadata (include original URL)
                                    metadata = {
                                        "format": image_format,
                                        "original_url": url,
                                    }

                                    # Upload image
                                    client.export_image(
                                        image_data=image_data,
                                        trace_id=trace_id,
                                        span_id=span_id,
                                        image_uuid=image_uuid,
                                        metadata=metadata,
                                    )

                                    # Add to list of UUIDs
                                    image_uuids.append(image_uuid)

                                    # Replace URL with UUID reference
                                    image_reference = format_image_reference(image_uuid)
                                    if isinstance(image_url_data, dict):
                                        image_url_data["url"] = image_reference
                                    elif isinstance(image_url_data, str):
                                        item["image_url"] = image_reference

                                    logger.debug(
                                        f"Downloaded and uploaded image {image_uuid} from URL and replaced in message"
                                    )

                                except Exception as e:
                                    logger.warning(
                                        f"Failed to process image URL {url}: {e}",
                                        exc_info=True,
                                    )
                                    # Keep original URL if upload fails

    except Exception as e:
        logger.error(f"Error processing images from messages: {e}", exc_info=True)
        # Return original messages if processing fails
        return ([], message_dicts)

    return (image_uuids, modified_message_dicts)


def process_images_from_prompts(
    prompts: list[str],
    trace_id: str,
    span_id: str,
) -> tuple[list[str], list[str]]:
    """
    Process images embedded in JSON prompts.

    This handles the edge case where prompts are JSON strings containing
    image_url fields (common in on_llm_start). It will:
    1. Parse JSON strings to extract message dicts
    2. Process images (download, upload, replace with UUIDs)
    3. Convert back to JSON strings

    Args:
        prompts: List of prompt strings (may contain JSON with images)
        trace_id: Trace ID to associate images with
        span_id: Span ID to associate images with

    Returns:
        Tuple of (image_uuids, processed_prompts)
    """
    image_uuids: list[str] = []
    processed_prompts: list[str] = prompts  # Default to original prompts

    if not prompts:
        return ([], prompts)

    # Check if any prompts contain embedded JSON with images
    has_embedded_images = any(
        isinstance(p, str) and '"type": "image_url"' in p for p in prompts
    )

    if not has_embedded_images:
        return ([], prompts)

    try:
        import json as json_module

        # Parse prompts to extract message dicts
        message_dicts_list = []
        for prompt in prompts:
            if isinstance(prompt, str) and '"type": "image_url"' in prompt:
                try:
                    if prompt.strip().startswith("[") or prompt.strip().startswith("{"):
                        parsed = json_module.loads(prompt)
                        if isinstance(parsed, dict):
                            parsed = [parsed]
                        if isinstance(parsed, list):
                            # Wrap in batch format
                            message_dicts_list.append(parsed)
                except Exception:
                    pass

        # Process images if we found any message dicts
        if not message_dicts_list:
            return ([], prompts)

        combined_message_dicts = message_dicts_list
        image_uuids, processed_message_dicts = extract_and_process_images(
            combined_message_dicts, trace_id, span_id
        )

        # Convert processed message dicts back to JSON strings
        processed_prompts = []
        for prompt in prompts:
            if isinstance(prompt, str) and '"type": "image_url"' in prompt:
                # This prompt had images, use processed version
                try:
                    if prompt.strip().startswith("[") or prompt.strip().startswith("{"):
                        # Find matching processed message dict and convert back
                        if processed_message_dicts:
                            processed_prompts.append(
                                json_module.dumps(processed_message_dicts.pop(0))
                            )
                        else:
                            processed_prompts.append(prompt)
                    else:
                        processed_prompts.append(prompt)
                except Exception:
                    processed_prompts.append(prompt)
            else:
                processed_prompts.append(prompt)

    except Exception as e:
        logger.warning(f"Error processing images from prompts: {e}", exc_info=True)
        return ([], prompts)

    return (image_uuids, processed_prompts)
