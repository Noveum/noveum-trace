"""Tests for image utility functions."""

import base64

import pytest

from noveum_trace.utils.image_utils import (
    extract_uuid_from_reference,
    format_image_reference,
    generate_image_uuid,
    is_base64_image,
    parse_base64_image,
)


class TestIsBase64Image:
    """Tests for is_base64_image function."""

    def test_valid_base64_image(self):
        """Test that valid base64 image URIs are identified."""
        assert is_base64_image("data:image/png;base64,ABC123")
        assert is_base64_image("data:image/jpeg;base64,XYZ789")
        assert is_base64_image("data:image/gif;base64,...")

    def test_http_url_not_base64(self):
        """Test that HTTP URLs are not identified as base64 images."""
        assert not is_base64_image("https://example.com/image.png")
        assert not is_base64_image("http://example.com/image.jpg")

    def test_non_image_data_uri(self):
        """Test that non-image data URIs are not identified as base64 images."""
        assert not is_base64_image("data:text/plain;base64,SGVsbG8=")

    def test_invalid_types(self):
        """Test that non-string inputs are handled correctly."""
        assert not is_base64_image(None)
        assert not is_base64_image(123)
        assert not is_base64_image(b"bytes")


class TestParseBase64Image:
    """Tests for parse_base64_image function."""

    def test_parse_valid_png(self):
        """Test parsing a valid PNG base64 image."""
        # Create a simple base64 encoded image (1x1 red pixel PNG)
        image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{base64_data}"

        result = parse_base64_image(data_uri)

        assert result["format"] == "png"
        assert result["mime_type"] == "image/png"
        assert result["image_data"] == image_bytes

    def test_parse_valid_jpeg(self):
        """Test parsing a valid JPEG base64 image."""
        image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{base64_data}"

        result = parse_base64_image(data_uri)

        assert result["format"] == "jpeg"
        assert result["mime_type"] == "image/jpeg"
        assert result["image_data"] == image_bytes

    def test_parse_invalid_data_uri(self):
        """Test that invalid data URIs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid base64 image data URI"):
            parse_base64_image("https://example.com/image.png")

    def test_parse_invalid_format(self):
        """Test that invalid data URI format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid data URI format"):
            parse_base64_image("data:image/png,NOTBASE64")

    def test_parse_invalid_base64(self):
        """Test that invalid base64 data raises ValueError."""
        with pytest.raises(ValueError, match="Failed to decode base64 data"):
            parse_base64_image("data:image/png;base64,!!INVALID!!")


class TestGenerateImageUuid:
    """Tests for generate_image_uuid function."""

    def test_generates_valid_uuid(self):
        """Test that a valid UUID is generated."""
        uuid = generate_image_uuid()
        assert isinstance(uuid, str)
        assert len(uuid) == 36  # Standard UUID format
        assert uuid.count("-") == 4  # UUID has 4 hyphens

    def test_generates_unique_uuids(self):
        """Test that unique UUIDs are generated."""
        uuid1 = generate_image_uuid()
        uuid2 = generate_image_uuid()
        assert uuid1 != uuid2


class TestFormatImageReference:
    """Tests for format_image_reference function."""

    def test_format_reference(self):
        """Test formatting an image UUID as a reference."""
        uuid = "abc-123-def-456"
        reference = format_image_reference(uuid)
        assert reference == "image://abc-123-def-456"

    def test_format_with_real_uuid(self):
        """Test formatting with a real UUID."""
        uuid = generate_image_uuid()
        reference = format_image_reference(uuid)
        assert reference.startswith("image://")
        assert reference[8:] == uuid


class TestExtractUuidFromReference:
    """Tests for extract_uuid_from_reference function."""

    def test_extract_valid_reference(self):
        """Test extracting UUID from a valid reference."""
        uuid = "abc-123-def-456"
        reference = "image://abc-123-def-456"
        extracted = extract_uuid_from_reference(reference)
        assert extracted == uuid

    def test_extract_with_real_uuid(self):
        """Test extracting a real UUID from a reference."""
        uuid = generate_image_uuid()
        reference = format_image_reference(uuid)
        extracted = extract_uuid_from_reference(reference)
        assert extracted == uuid

    def test_extract_invalid_reference(self):
        """Test that invalid references return None."""
        assert extract_uuid_from_reference("https://example.com/image.png") is None
        assert extract_uuid_from_reference("data:image/png;base64,ABC") is None

    def test_extract_invalid_types(self):
        """Test that non-string inputs return None."""
        assert extract_uuid_from_reference(None) is None
        assert extract_uuid_from_reference(123) is None
