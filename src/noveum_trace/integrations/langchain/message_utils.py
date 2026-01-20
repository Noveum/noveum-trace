"""
Utility functions for parsing and structuring LangChain message objects.

This module provides functions to convert LangChain messages into structured
dictionaries instead of stringifying them, significantly reducing trace size
and improving readability.
"""

import logging
from typing import Any

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
