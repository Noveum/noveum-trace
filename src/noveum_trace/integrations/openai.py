"""
OpenAI integration for Noveum Trace SDK.

This module provides automatic instrumentation for OpenAI API calls.
"""

import logging
from typing import Any

from noveum_trace.decorators.llm import trace_llm
from noveum_trace.utils.exceptions import TracingError

logger = logging.getLogger(__name__)

_original_create = None
_patched = False


def patch_openai() -> None:
    """
    Patch OpenAI client to automatically trace API calls.

    This function monkey-patches the OpenAI client to automatically
    add tracing to all completion calls.

    Raises:
        TracingError: If patching fails
    """
    global _patched, _original_create

    if _patched:
        logger.warning("OpenAI is already patched")
        return

    try:
        import openai
    except ImportError as e:
        raise TracingError(
            "OpenAI package not found. Install with: pip install openai"
        ) from e

    try:
        # Patch the chat completions create method
        if hasattr(openai, "OpenAI"):
            # OpenAI v1.x
            _patch_openai_v1(openai)
        else:
            # OpenAI v0.x (legacy)
            _patch_openai_legacy(openai)

        _patched = True
        logger.info("OpenAI integration patched successfully")

    except Exception as e:
        raise TracingError(f"Failed to patch OpenAI: {e}") from e


def unpatch_openai() -> None:
    """
    Remove OpenAI patching and restore original functionality.
    """
    global _patched, _original_create

    if not _patched:
        return

    try:
        import openai

        # Restore original methods
        if _original_create and hasattr(openai, "OpenAI"):
            openai.OpenAI.chat.completions.create = _original_create

        _patched = False
        _original_create = None
        logger.info("OpenAI integration unpatched")

    except Exception as e:
        logger.error(f"Failed to unpatch OpenAI: {e}")


def _patch_openai_v1(openai_module: Any) -> None:
    """Patch OpenAI v1.x client."""
    global _original_create

    # Store original method
    _original_create = openai_module.OpenAI.chat.completions.create

    # Create traced wrapper
    @trace_llm(
        capture_prompts=True,
        capture_completions=True,
        capture_tokens=True,
        estimate_costs=True,
    )
    def traced_create(self: Any, **kwargs: Any) -> Any:
        """Traced version of OpenAI chat completions create."""
        return _original_create(self, **kwargs)

    # Apply patch
    openai_module.OpenAI.chat.completions.create = traced_create


def _patch_openai_legacy(openai_module: Any) -> None:
    """Patch OpenAI v0.x (legacy) client."""
    # Implementation for legacy OpenAI client
    # This would patch the older API format
    pass


def is_patched() -> bool:
    """
    Check if OpenAI integration is currently patched.

    Returns:
        True if patched, False otherwise
    """
    return _patched


def get_integration_info() -> dict[str, Any]:
    """
    Get information about the OpenAI integration.

    Returns:
        Dictionary with integration information
    """
    try:
        import openai

        openai_version = getattr(openai, "__version__", "unknown")
    except ImportError:
        openai_version = None

    return {
        "name": "openai",
        "patched": _patched,
        "openai_version": openai_version,
        "supported": openai_version is not None,
    }
