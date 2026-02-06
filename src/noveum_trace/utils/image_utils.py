"""
Utility functions for image processing and handling.

This module provides functions for parsing base64-encoded images,
extracting image metadata, and generating image UUIDs.
"""

import base64
import re
import uuid
from typing import Any, Optional

from noveum_trace.utils.logging import get_sdk_logger

logger = get_sdk_logger("utils.image_utils")


def is_base64_image(url: str) -> bool:
    """
    Check if URL is a base64-encoded image data URI.

    Args:
        url: URL string to check

    Returns:
        True if URL is a base64 image data URI, False otherwise
    """
    return isinstance(url, str) and url.startswith("data:image/")


def parse_base64_image(data_uri: str) -> dict[str, Any]:
    """
    Parse a base64 data URI and extract image data and format.

    Args:
        data_uri: Data URI string (e.g., 'data:image/png;base64,ABC123...')

    Returns:
        Dict with 'image_data' (bytes), 'format' (str), 'mime_type' (str)

    Raises:
        ValueError: If data_uri is not a valid base64 image data URI
    """
    if not is_base64_image(data_uri):
        raise ValueError(f"Invalid base64 image data URI: {data_uri[:50]}...")

    try:
        # Parse format: data:image/{format};base64,{base64_data}
        # Match pattern: data:image/{format};base64,{data}
        pattern = r"^data:image/([^;]+);base64,(.+)$"
        match = re.match(pattern, data_uri)

        if not match:
            raise ValueError(
                "Invalid data URI format. Expected 'data:image/{format};base64,{data}'"
            )

        image_format = match.group(1)
        base64_data = match.group(2)

        # Decode base64 to bytes
        try:
            image_data = base64.b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 data: {e}") from e

        # Construct mime type
        mime_type = f"image/{image_format}"

        return {
            "image_data": image_data,
            "format": image_format,
            "mime_type": mime_type,
        }

    except Exception as e:
        logger.error(f"Error parsing base64 image: {e}")
        raise


def generate_image_uuid() -> str:
    """
    Generate UUID for image upload.

    Returns:
        UUID string for image identification
    """
    return str(uuid.uuid4())


def format_image_reference(image_uuid: str) -> str:
    """
    Format image UUID as a reference string.

    Args:
        image_uuid: UUID of the uploaded image

    Returns:
        Formatted reference string (e.g., 'image://uuid')
    """
    return f"image://{image_uuid}"


def extract_uuid_from_reference(reference: str) -> Optional[str]:
    """
    Extract UUID from an image reference string.

    Args:
        reference: Image reference string (e.g., 'image://uuid')

    Returns:
        UUID string if valid reference, None otherwise
    """
    if not isinstance(reference, str) or not reference.startswith("image://"):
        return None

    return reference[len("image://") :]
