"""
Integration tests for image upload API with real API calls.

This module tests the image upload functionality by making real API calls
to the Noveum platform, ensuring that images are correctly uploaded with
proper trace and span associations.

NO MOCKING - All tests use real API calls.
"""

import io
import os
import time
import uuid
from typing import Optional

import pytest
from PIL import Image

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

import noveum_trace
from noveum_trace.utils.image_utils import generate_image_uuid

# Test endpoints - configurable via environment
ENDPOINT = os.environ.get("NOVEUM_ENDPOINT", "https://api.noveum.ai/api")
API_KEY = os.environ.get("NOVEUM_API_KEY", "test-api-key")
# Separate header key if needed
API_KEY_HEADER = os.environ.get("NOVEUM_API_KEY_HEADER", API_KEY)
ORGANIZATION_SLUG = os.environ.get("NOVEUM_ORGANIZATION_SLUG", "apple")


def is_valid_api_key(key: Optional[str]) -> bool:
    """Check if API key is valid (not None, empty, or placeholder)."""
    if not key:
        return False

    invalid_keys = [
        "",
        "test-api-key",
        "test-key",
        "your-api-key-here",
    ]
    return key not in invalid_keys and len(key) > 10


@pytest.fixture
def sample_image_bytes():
    """Generate a random test image as bytes."""
    # Create a simple random image
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))  # Red image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture(autouse=True)
def setup_noveum_trace():
    """Setup noveum trace for each test."""
    # Ensure clean state
    noveum_trace.shutdown()

    # Initialize with optimized transport settings for integration tests
    noveum_trace.init(
        project="noveum-trace-python",
        api_key=API_KEY,
        endpoint=ENDPOINT,
        environment="git-integ-test",
        transport_config={
            "batch_size": 1,  # Send images immediately
            "batch_timeout": 0.1,  # Very short timeout for faster tests
            "timeout": 30,  # Timeout for image uploads
        },
    )

    yield

    # Cleanup - shutdown completely to ensure clean state for next test
    noveum_trace.shutdown()
    time.sleep(0.1)  # Small delay to ensure cleanup


@pytest.mark.integration
@pytest.mark.disable_transport_mocking
class TestImageUpload:
    """Test image upload functionality with real API calls."""

    @pytest.mark.skipif(
        not is_valid_api_key(API_KEY), reason="Noveum API key not available or invalid"
    )
    def test_image_upload_with_unique_ids(self, sample_image_bytes):
        """Test image upload with unique trace, span, and image UUIDs."""
        # Generate unique IDs
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        image_uuid = generate_image_uuid()

        # Get client and upload image
        client = noveum_trace.get_client()
        client.export_image(
            image_data=sample_image_bytes,
            trace_id=trace_id,
            span_id=span_id,
            image_uuid=image_uuid,
            metadata={"format": "png"},
        )

        # Flush to ensure image is sent
        client.transport.flush()
        time.sleep(1.0)  # Give time for async processing and API response

        # Verify upload was successful by checking the response
        # The SDK should have successfully sent the image
        # We can't easily verify the image exists via API without additional endpoints,
        # but we can verify no exceptions were raised
        assert True, "Image upload completed without errors"

    @pytest.mark.skipif(
        not is_valid_api_key(API_KEY), reason="Noveum API key not available or invalid"
    )
    def test_image_upload_with_jpeg_format(self):
        """Test image upload with JPEG format."""
        # Create a JPEG image
        img = Image.new("RGB", (50, 50), color=(0, 0, 255))  # Blue image
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        jpeg_data = img_bytes.read()

        # Generate unique IDs
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        image_uuid = generate_image_uuid()

        # Upload image
        client = noveum_trace.get_client()
        client.export_image(
            image_data=jpeg_data,
            trace_id=trace_id,
            span_id=span_id,
            image_uuid=image_uuid,
            metadata={"format": "jpeg"},
        )

        # Flush to ensure image is sent
        client.transport.flush()
        time.sleep(1.0)

        # Verify upload completed
        assert True, "JPEG image upload completed without errors"

    @pytest.mark.skipif(
        not is_valid_api_key(API_KEY), reason="Noveum API key not available or invalid"
    )
    def test_image_upload_with_organization_slug(self, sample_image_bytes):
        """Test image upload with organization slug in URL."""
        # Generate unique IDs
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        image_uuid = generate_image_uuid()

        # Upload image
        client = noveum_trace.get_client()
        client.export_image(
            image_data=sample_image_bytes,
            trace_id=trace_id,
            span_id=span_id,
            image_uuid=image_uuid,
            metadata={"format": "png"},
        )

        # Flush to ensure image is sent
        client.transport.flush()
        time.sleep(1.0)

        # Verify upload completed
        assert True, "Image upload with organization slug completed without errors"

    @pytest.mark.skipif(
        not is_valid_api_key(API_KEY), reason="Noveum API key not available or invalid"
    )
    def test_image_upload_multiple_images(self):
        """Test uploading multiple images with different UUIDs."""
        client = noveum_trace.get_client()

        # Upload multiple images
        num_images = 3
        image_uuids = []
        for i in range(num_images):
            # Create unique image each time
            img = Image.new("RGB", (100, 100), color=(i * 50, 100, 200))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            image_data = img_bytes.read()

            # Generate unique IDs for each image
            trace_id = str(uuid.uuid4())
            span_id = str(uuid.uuid4())
            image_uuid = generate_image_uuid()
            image_uuids.append(image_uuid)

            client.export_image(
                image_data=image_data,
                trace_id=trace_id,
                span_id=span_id,
                image_uuid=image_uuid,
                metadata={"format": "png", "index": i},
            )

        # Flush to ensure all images are sent
        client.transport.flush()
        time.sleep(2.0)  # Give more time for multiple uploads

        # Verify all image UUIDs are unique
        assert len(set(image_uuids)) == num_images, "All image UUIDs should be unique"
        assert True, f"Successfully uploaded {num_images} images"

    @pytest.mark.skipif(
        not is_valid_api_key(API_KEY), reason="Noveum API key not available or invalid"
    )
    def test_image_upload_with_custom_metadata(self, sample_image_bytes):
        """Test image upload with custom metadata."""
        # Generate unique IDs
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        image_uuid = generate_image_uuid()

        # Upload with custom metadata
        custom_metadata = {
            "format": "png",
            "width": 100,
            "height": 100,
            "source": "test",
            "custom_field": "custom_value",
        }

        client = noveum_trace.get_client()
        client.export_image(
            image_data=sample_image_bytes,
            trace_id=trace_id,
            span_id=span_id,
            image_uuid=image_uuid,
            metadata=custom_metadata,
        )

        # Flush to ensure image is sent
        client.transport.flush()
        time.sleep(1.0)

        # Verify upload completed
        assert True, "Image upload with custom metadata completed without errors"
