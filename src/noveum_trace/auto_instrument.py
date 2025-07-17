"""
Auto-instrumentation module for Noveum Trace SDK.

This module provides automatic instrumentation capabilities for popular
AI and ML libraries, allowing tracing without code modifications.
"""

import functools
import importlib
import warnings
from typing import Any, Optional

from noveum_trace.context_managers import trace_llm
from noveum_trace.core.span import SpanStatus
from noveum_trace.utils.exceptions import NoveumTraceError


class InstrumentationRegistry:
    """Registry for managing instrumentation of different libraries."""

    def __init__(self) -> None:
        self.instrumented_libraries: dict[str, Any] = {}
        self.original_methods: dict[str, dict[str, Any]] = {}

    def register_instrumentation(
        self, library_name: str, instrumentation_class: type[Any]
    ) -> None:
        """Register an instrumentation class for a library."""
        self.instrumented_libraries[library_name] = instrumentation_class

    def is_instrumented(self, library_name: str) -> bool:
        """Check if a library is currently instrumented."""
        return library_name in self.original_methods

    def instrument(
        self, library_name: str, config: Optional[dict[str, Any]] = None
    ) -> None:
        """Instrument a library."""
        if self.is_instrumented(library_name):
            warnings.warn(f"{library_name} is already instrumented", stacklevel=2)
            return

        if library_name not in self.instrumented_libraries:
            raise NoveumTraceError(f"No instrumentation available for {library_name}")

        instrumentation_class = self.instrumented_libraries[library_name]
        instrumentation = instrumentation_class(config or {})

        try:
            original_methods = instrumentation.instrument()
            self.original_methods[library_name] = {
                "instrumentation": instrumentation,
                "original_methods": original_methods,
            }
        except Exception as e:
            raise NoveumTraceError(f"Failed to instrument {library_name}: {e}") from e

    def uninstrument(self, library_name: str) -> None:
        """Remove instrumentation from a library."""
        if not self.is_instrumented(library_name):
            warnings.warn(f"{library_name} is not instrumented", stacklevel=2)
            return

        instrumentation_data = self.original_methods[library_name]
        instrumentation = instrumentation_data["instrumentation"]

        try:
            instrumentation.uninstrument()
            del self.original_methods[library_name]
        except Exception as e:
            raise NoveumTraceError(f"Failed to uninstrument {library_name}: {e}") from e

    def uninstrument_all(self) -> None:
        """Remove instrumentation from all libraries."""
        libraries = list(self.original_methods.keys())
        for library_name in libraries:
            try:
                self.uninstrument(library_name)
            except Exception as e:
                warnings.warn(
                    f"Failed to uninstrument {library_name}: {e}", stacklevel=2
                )


# Global registry instance
_instrumentation_registry = InstrumentationRegistry()


class BaseInstrumentation:
    """Base class for library instrumentation."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.original_methods: dict[str, Any] = {}

    def instrument(self) -> dict[str, Any]:
        """Instrument the library. Returns original methods for restoration."""
        raise NotImplementedError("This method must be implemented by subclasses")

    def uninstrument(self) -> None:
        """Remove instrumentation and restore original methods."""
        raise NotImplementedError("This method must be implemented by subclasses")

    def _is_library_available(self, library_name: str) -> bool:
        """Check if a library is available for instrumentation."""
        try:
            importlib.import_module(library_name)
            return True
        except ImportError:
            return False


class OpenAIInstrumentation(BaseInstrumentation):
    """Instrumentation for OpenAI library."""

    def instrument(self) -> dict[str, Any]:
        """Instrument OpenAI library methods."""
        if not self._is_library_available("openai"):
            raise NoveumTraceError("OpenAI library not found")

        import openai

        # Instrument chat completions
        self._instrument_chat_completions(openai)

        # Instrument embeddings
        self._instrument_embeddings(openai)

        # Instrument image generation
        self._instrument_images(openai)

        return self.original_methods

    def _instrument_chat_completions(self, openai_module: Any) -> None:
        """Instrument chat completions."""
        try:
            original_create = openai_module.chat.completions.create
            self.original_methods["chat.completions.create"] = original_create

            @functools.wraps(original_create)
            def traced_create(*args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "unknown")

                with trace_llm(
                    model=model, provider="openai", operation="chat_completion"
                ) as span:
                    try:
                        # Capture inputs if configured
                        if self.config.get("capture_inputs", True):
                            messages = kwargs.get("messages", [])
                            if messages:
                                # Truncate messages for storage
                                max_length = self.config.get("max_input_length", 2000)
                                messages_str = str(messages)[:max_length]
                                span.set_attributes(
                                    {
                                        "llm.messages": messages_str,
                                        "llm.message_count": len(messages),
                                    }
                                )

                        # Make the actual API call
                        response = original_create(*args, **kwargs)

                        # Capture outputs if configured
                        if self.config.get("capture_outputs", True):
                            if hasattr(response, "usage") and response.usage:
                                span.set_attributes(
                                    {
                                        "llm.input_tokens": response.usage.prompt_tokens,
                                        "llm.output_tokens": response.usage.completion_tokens,
                                        "llm.total_tokens": response.usage.total_tokens,
                                    }
                                )

                            if hasattr(response, "choices") and response.choices:
                                content = response.choices[0].message.content
                                max_output_length = self.config.get(
                                    "max_output_length", 2000
                                )
                                span.set_attribute(
                                    "llm.response", content[:max_output_length]
                                )

                        # Calculate cost if configured
                        if self.config.get("calculate_cost", False):
                            if hasattr(response, "usage") and response.usage:
                                cost = self._calculate_openai_cost(
                                    model, response.usage
                                )
                                span.set_attribute("llm.cost", cost)

                        span.set_status(SpanStatus.OK)
                        return response

                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(SpanStatus.ERROR, str(e))
                        raise

            # Replace the original method
            openai_module.chat.completions.create = traced_create

        except AttributeError as e:
            warnings.warn(f"Could not instrument chat completions: {e}", stacklevel=2)

    def _instrument_embeddings(self, openai_module: Any) -> None:
        """Instrument embeddings."""
        try:
            original_create = openai_module.embeddings.create
            self.original_methods["embeddings.create"] = original_create

            @functools.wraps(original_create)
            def traced_embeddings_create(*args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "unknown")

                with trace_llm(
                    model=model, provider="openai", operation="embeddings"
                ) as span:
                    try:
                        # Capture input information
                        if self.config.get("capture_inputs", True):
                            input_data = kwargs.get("input", [])
                            if isinstance(input_data, list):
                                span.set_attributes(
                                    {
                                        "llm.input_count": len(input_data),
                                        "llm.input_type": "list",
                                    }
                                )
                            else:
                                span.set_attributes(
                                    {
                                        "llm.input_count": 1,
                                        "llm.input_type": "string",
                                        "llm.input_length": len(str(input_data)),
                                    }
                                )

                        # Make the actual API call
                        response = original_create(*args, **kwargs)

                        # Capture output information
                        if self.config.get("capture_outputs", True):
                            if hasattr(response, "usage") and response.usage:
                                span.set_attribute(
                                    "llm.total_tokens", response.usage.total_tokens
                                )

                            if hasattr(response, "data") and response.data:
                                span.set_attributes(
                                    {
                                        "llm.embeddings_count": len(response.data),
                                        "llm.embedding_dimensions": (
                                            len(response.data[0].embedding)
                                            if response.data
                                            else 0
                                        ),
                                    }
                                )

                        span.set_status(SpanStatus.OK)
                        return response

                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(SpanStatus.ERROR, str(e))
                        raise

            openai_module.embeddings.create = traced_embeddings_create

        except AttributeError as e:
            warnings.warn(f"Could not instrument embeddings: {e}", stacklevel=2)

    def _instrument_images(self, openai_module: Any) -> None:
        """Instrument image generation."""
        try:
            original_generate = openai_module.images.generate
            self.original_methods["images.generate"] = original_generate

            @functools.wraps(original_generate)
            def traced_images_generate(*args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "dall-e-2")

                with trace_llm(
                    model=model, provider="openai", operation="image_generation"
                ) as span:
                    try:
                        # Capture input information
                        if self.config.get("capture_inputs", True):
                            prompt = kwargs.get("prompt", "")
                            span.set_attributes(
                                {
                                    "image.prompt": prompt[:500],  # Truncate prompt
                                    "image.size": kwargs.get("size", "unknown"),
                                    "image.quality": kwargs.get("quality", "standard"),
                                    "image.n": kwargs.get("n", 1),
                                }
                            )

                        # Make the actual API call
                        response = original_generate(*args, **kwargs)

                        # Capture output information
                        if self.config.get("capture_outputs", True):
                            if hasattr(response, "data") and response.data:
                                span.set_attributes(
                                    {
                                        "image.generated_count": len(response.data),
                                        "image.urls": [
                                            img.url
                                            for img in response.data
                                            if hasattr(img, "url")
                                        ],
                                    }
                                )

                        span.set_status(SpanStatus.OK)
                        return response

                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(SpanStatus.ERROR, str(e))
                        raise

            openai_module.images.generate = traced_images_generate

        except AttributeError as e:
            warnings.warn(f"Could not instrument image generation: {e}", stacklevel=2)

    def _calculate_openai_cost(self, model: str, usage: Any) -> float:
        """Calculate approximate cost for OpenAI API calls."""
        # Simplified cost calculation - would need real pricing data
        cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
        }

        base_cost = cost_per_1k_tokens.get(model, 0.002)
        if hasattr(usage, "total_tokens"):
            return (usage.total_tokens / 1000) * base_cost
        return 0.0

    def uninstrument(self) -> None:
        """Remove OpenAI instrumentation."""
        import openai

        # Restore original methods
        for method_path, original_method in self.original_methods.items():
            if method_path == "chat.completions.create":
                openai.chat.completions.create = original_method
            elif method_path == "embeddings.create":
                openai.embeddings.create = original_method
            elif method_path == "images.generate":
                openai.images.generate = original_method


class AnthropicInstrumentation(BaseInstrumentation):
    """Instrumentation for Anthropic library."""

    def instrument(self) -> dict[str, Any]:
        """Instrument Anthropic library methods."""
        if not self._is_library_available("anthropic"):
            raise NoveumTraceError("Anthropic library not found")

        import anthropic

        # Instrument messages
        self._instrument_messages(anthropic)

        return self.original_methods

    def _instrument_messages(self, anthropic_module: Any) -> None:
        """Instrument Anthropic messages."""
        try:
            # This would need to be adapted based on actual Anthropic API structure
            original_create = anthropic_module.messages.create
            self.original_methods["messages.create"] = original_create

            @functools.wraps(original_create)
            def traced_messages_create(*args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "unknown")

                with trace_llm(
                    model=model, provider="anthropic", operation="messages"
                ) as span:
                    try:
                        # Capture inputs
                        if self.config.get("capture_inputs", True):
                            messages = kwargs.get("messages", [])
                            span.set_attributes(
                                {
                                    "llm.messages": str(messages)[:2000],
                                    "llm.message_count": len(messages),
                                }
                            )

                        # Make the actual API call
                        response = original_create(*args, **kwargs)

                        # Capture outputs
                        if self.config.get("capture_outputs", True):
                            # Adapt based on actual Anthropic response structure
                            if hasattr(response, "content"):
                                span.set_attribute(
                                    "llm.response", str(response.content)[:2000]
                                )

                        span.set_status(SpanStatus.OK)
                        return response

                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(SpanStatus.ERROR, str(e))
                        raise

            anthropic_module.messages.create = traced_messages_create

        except AttributeError as e:
            warnings.warn(f"Could not instrument Anthropic messages: {e}", stacklevel=2)

    def uninstrument(self) -> None:
        """Remove Anthropic instrumentation."""
        import anthropic

        for method_path, original_method in self.original_methods.items():
            if method_path == "messages.create":
                if hasattr(anthropic, "messages"):
                    anthropic.messages.create = original_method


class LangChainInstrumentation(BaseInstrumentation):
    """Instrumentation for LangChain library."""

    def instrument(self) -> dict[str, Any]:
        """Instrument LangChain library methods."""
        if not self._is_library_available("langchain"):
            raise NoveumTraceError("LangChain library not found")

        # This would require more complex instrumentation
        # as LangChain has many different components
        warnings.warn(
            "LangChain instrumentation is not fully implemented yet", stacklevel=2
        )

        return self.original_methods

    def uninstrument(self) -> None:
        """Remove LangChain instrumentation."""
        pass


# Register available instrumentations
_instrumentation_registry.register_instrumentation("openai", OpenAIInstrumentation)
_instrumentation_registry.register_instrumentation(
    "anthropic", AnthropicInstrumentation
)
_instrumentation_registry.register_instrumentation(
    "langchain", LangChainInstrumentation
)


# Public API functions


def auto_instrument(library: str, config: Optional[dict[str, Any]] = None) -> bool:
    """
    Automatically instrument a library for tracing.

    Args:
        library: Name of the library to instrument ('openai', 'anthropic', etc.)
        config: Configuration options for instrumentation

    Returns:
        True if instrumentation was successful, False otherwise

    Example:
        # Basic instrumentation
        auto_instrument('openai')

        # With configuration
        auto_instrument('openai', {
            'capture_inputs': True,
            'capture_outputs': True,
            'max_input_length': 1000,
            'calculate_cost': True
        })
    """
    try:
        _instrumentation_registry.instrument(library, config)
        return True
    except Exception as e:
        warnings.warn(f"Failed to instrument {library}: {e}", stacklevel=2)
        return False


def uninstrument(library: str) -> bool:
    """
    Remove instrumentation from a library.

    Args:
        library: Name of the library to uninstrument

    Returns:
        True if uninstrumentation was successful, False otherwise
    """
    try:
        _instrumentation_registry.uninstrument(library)
        return True
    except Exception as e:
        warnings.warn(f"Failed to uninstrument {library}: {e}", stacklevel=2)
        return False


def uninstrument_all() -> None:
    """Remove instrumentation from all libraries."""
    _instrumentation_registry.uninstrument_all()


def is_instrumented(library: str) -> bool:
    """
    Check if a library is currently instrumented.

    Args:
        library: Name of the library to check

    Returns:
        True if the library is instrumented, False otherwise
    """
    return _instrumentation_registry.is_instrumented(library)


def get_instrumented_libraries() -> list[str]:
    """
    Get list of currently instrumented libraries.

    Returns:
        List of instrumented library names
    """
    return list(_instrumentation_registry.original_methods.keys())


def enable_auto_tracing(
    libraries: Optional[list[str]] = None,
    config: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, bool]:
    """
    Enable automatic tracing for multiple libraries.

    Args:
        libraries: List of libraries to instrument. If None, instruments all available.
        config: Per-library configuration options

    Returns:
        Dictionary mapping library names to instrumentation success status

    Example:
        # Instrument all available libraries
        enable_auto_tracing()

        # Instrument specific libraries
        enable_auto_tracing(['openai', 'anthropic'])

        # With per-library configuration
        enable_auto_tracing(
            ['openai', 'anthropic'],
            {
                'openai': {'capture_inputs': True, 'calculate_cost': True},
                'anthropic': {'capture_inputs': False}
            }
        )
    """
    if libraries is None:
        libraries = ["openai", "anthropic", "langchain"]

    config = config or {}
    results = {}

    for library in libraries:
        library_config = config.get(library, {})
        results[library] = auto_instrument(library, library_config)

    return results


def get_available_instrumentations() -> list[str]:
    """
    Get list of available instrumentations.

    Returns:
        List of library names that can be instrumented
    """
    return list(_instrumentation_registry.instrumented_libraries.keys())


# Utility functions for configuration


def get_default_config(library: str) -> dict[str, Any]:
    """
    Get default configuration for a library.

    Args:
        library: Library name

    Returns:
        Default configuration dictionary
    """
    defaults = {
        "openai": {
            "capture_inputs": True,
            "capture_outputs": True,
            "max_input_length": 2000,
            "max_output_length": 2000,
            "calculate_cost": False,
        },
        "anthropic": {
            "capture_inputs": True,
            "capture_outputs": True,
            "max_input_length": 2000,
            "max_output_length": 2000,
        },
        "langchain": {
            "capture_inputs": True,
            "capture_outputs": True,
            "trace_chains": True,
        },
    }

    return defaults.get(library, {})  # type: ignore


def create_production_config() -> dict[str, dict[str, Any]]:
    """
    Create production-optimized configuration for all libraries.

    Returns:
        Production configuration dictionary
    """
    return {
        "openai": {
            "capture_inputs": False,  # Reduce overhead
            "capture_outputs": True,
            "max_output_length": 500,
            "calculate_cost": True,
        },
        "anthropic": {
            "capture_inputs": False,
            "capture_outputs": True,
            "max_output_length": 500,
        },
        "langchain": {
            "capture_inputs": False,
            "capture_outputs": True,
            "trace_chains": False,  # Reduce noise
        },
    }
