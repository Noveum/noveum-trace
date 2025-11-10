"""
LLM utility functions for Noveum Trace SDK.

This module provides utility functions for working with LLM providers,
token counting, cost estimation, and metadata extraction.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional, Union

from noveum_trace.utils import tokenizers


@dataclass
class ModelInfo:
    """Information about an LLM model."""

    provider: str
    name: str
    context_window: int
    max_output_tokens: int
    input_cost_per_1m: float  # Cost per 1M input tokens in USD
    output_cost_per_1m: float  # Cost per 1M output tokens in USD
    supports_vision: bool = False
    supports_audio: bool = False
    supports_function_calling: bool = False
    training_cutoff: Optional[str] = None


# Comprehensive model definitions with pricing as of October 29, 2025
# Note: Pricing information is current as of October 29, 2025. Please verify latest pricing from official provider websites.
MODEL_REGISTRY: dict[str, ModelInfo] = {
    # OpenAI GPT-4.1 Family
    "gpt-4.1": ModelInfo(
        provider="openai",
        name="gpt-4.1",
        context_window=1047576,
        max_output_tokens=32768,
        input_cost_per_1m=3.00,
        output_cost_per_1m=12.00,
        supports_function_calling=True,
        training_cutoff="Jun 2024",
    ),
    "gpt-4.1-mini": ModelInfo(
        provider="openai",
        name="gpt-4.1-mini",
        context_window=1047576,
        max_output_tokens=32768,
        input_cost_per_1m=0.80,
        output_cost_per_1m=3.20,
        supports_function_calling=True,
        training_cutoff="Jun 2024",
    ),
    "gpt-4.1-nano": ModelInfo(
        provider="openai",
        name="gpt-4.1-nano",
        context_window=1047576,
        max_output_tokens=32768,
        input_cost_per_1m=0.20,
        output_cost_per_1m=0.80,
        supports_function_calling=True,
        training_cutoff="Jun 2024",
    ),
    # OpenAI Reasoning Models
    "o1": ModelInfo(
        provider="openai",
        name="o1",
        context_window=200000,
        max_output_tokens=100000,
        input_cost_per_1m=15.00,
        output_cost_per_1m=60.00,
        supports_function_calling=False,
        training_cutoff="Oct 2023",
    ),
    "o1-mini": ModelInfo(
        provider="openai",
        name="o1-mini",
        context_window=128000,
        max_output_tokens=65536,
        input_cost_per_1m=1.10,
        output_cost_per_1m=4.40,
        supports_function_calling=False,
        training_cutoff="Oct 2023",
    ),
    "o1-pro": ModelInfo(
        provider="openai",
        name="o1-pro",
        context_window=200000,
        max_output_tokens=100000,
        input_cost_per_1m=150.00,
        output_cost_per_1m=600.00,
        supports_function_calling=False,
        training_cutoff="Oct 2023",
    ),
    # OpenAI GPT-4 Family
    "gpt-4o": ModelInfo(
        provider="openai",
        name="gpt-4o",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1m=5.00,
        output_cost_per_1m=20.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Oct 2023",
    ),
    "gpt-4o-2024-11-20": ModelInfo(
        provider="openai",
        name="gpt-4o-2024-11-20",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Oct 2023",
    ),
    "gpt-4o-2024-08-06": ModelInfo(
        provider="openai",
        name="gpt-4o-2024-08-06",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Oct 2023",
    ),
    "gpt-4o-mini": ModelInfo(
        provider="openai",
        name="gpt-4o-mini",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1m=0.60,
        output_cost_per_1m=2.40,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Oct 2023",
    ),
    "gpt-4-turbo": ModelInfo(
        provider="openai",
        name="gpt-4-turbo",
        context_window=128000,
        max_output_tokens=4096,
        input_cost_per_1m=10.00,
        output_cost_per_1m=30.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Dec 2023",
    ),
    "gpt-4": ModelInfo(
        provider="openai",
        name="gpt-4",
        context_window=8192,
        max_output_tokens=8192,
        input_cost_per_1m=30.00,
        output_cost_per_1m=60.00,
        supports_function_calling=True,
        training_cutoff="Sep 2021",
    ),
    # OpenAI GPT-5 Family
    "gpt-5": ModelInfo(
        provider="openai",
        name="gpt-5",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="2025",
    ),
    "gpt-5-mini": ModelInfo(
        provider="openai",
        name="gpt-5-mini",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1m=0.25,
        output_cost_per_1m=2.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="2025",
    ),
    "gpt-5-nano": ModelInfo(
        provider="openai",
        name="gpt-5-nano",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.40,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="2025",
    ),
    "gpt-5-pro": ModelInfo(
        provider="openai",
        name="gpt-5-pro",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1m=15.00,
        output_cost_per_1m=120.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="2025",
    ),
    # OpenAI GPT-3.5 Family
    "gpt-3.5-turbo": ModelInfo(
        provider="openai",
        name="gpt-3.5-turbo",
        context_window=16385,
        max_output_tokens=4096,
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.50,
        supports_function_calling=True,
        training_cutoff="Sep 2021",
    ),
    "gpt-3.5-turbo-0125": ModelInfo(
        provider="openai",
        name="gpt-3.5-turbo-0125",
        context_window=16385,
        max_output_tokens=4096,
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.50,
        supports_function_calling=True,
        training_cutoff="Sep 2021",
    ),
    # Google Gemini Family
    "gemini-2.5-flash": ModelInfo(
        provider="google",
        name="gemini-2.5-flash",
        context_window=1000000,
        max_output_tokens=8192,
        input_cost_per_1m=0.30,
        output_cost_per_1m=2.50,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    "gemini-2.5-pro": ModelInfo(
        provider="google",
        name="gemini-2.5-pro",
        context_window=2000000,
        max_output_tokens=8192,
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    "gemini-2.0-flash": ModelInfo(
        provider="google",
        name="gemini-2.0-flash",
        context_window=1000000,
        max_output_tokens=8192,
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    # Gemini 1.5 pricing published per 1K characters; converted to per 1M tokens assuming 4 characters per token.
    "gemini-1.5-pro": ModelInfo(
        provider="google",
        name="gemini-1.5-pro",
        context_window=2000000,
        max_output_tokens=8192,
        input_cost_per_1m=1.25,
        output_cost_per_1m=5.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    "gemini-1.5-flash": ModelInfo(
        provider="google",
        name="gemini-1.5-flash",
        context_window=1000000,
        max_output_tokens=8192,
        input_cost_per_1m=0.075,
        output_cost_per_1m=0.30,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    # Anthropic Claude Family
    "claude-3.7-sonnet": ModelInfo(
        provider="anthropic",
        name="claude-3.7-sonnet",
        context_window=200000,
        max_output_tokens=128000,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    "claude-3.5-sonnet": ModelInfo(
        provider="anthropic",
        name="claude-3.5-sonnet",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    "claude-3.5-haiku": ModelInfo(
        provider="anthropic",
        name="claude-3.5-haiku",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    "claude-3-opus": ModelInfo(
        provider="anthropic",
        name="claude-3-opus",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Aug 2023",
    ),
    # Anthropic Claude 4 Family
    "claude-opus-4": ModelInfo(
        provider="anthropic",
        name="claude-opus-4",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="2025",
    ),
    "claude-sonnet-4": ModelInfo(
        provider="anthropic",
        name="claude-sonnet-4",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="2025",
    ),
    # Meta Llama Family
    "llama-3.3-70b": ModelInfo(
        provider="meta",
        name="llama-3.3-70b",
        context_window=128000,
        max_output_tokens=2048,
        input_cost_per_1m=0.23,
        output_cost_per_1m=0.40,
        supports_function_calling=True,
        training_cutoff="Dec 2024",
    ),
    "llama-3.1-405b": ModelInfo(
        provider="meta",
        name="llama-3.1-405b",
        context_window=128000,
        max_output_tokens=2048,
        input_cost_per_1m=1.79,
        output_cost_per_1m=1.79,
        supports_function_calling=True,
        training_cutoff="Jul 2024",
    ),
    "llama-3.1-70b": ModelInfo(
        provider="meta",
        name="llama-3.1-70b",
        context_window=128000,
        max_output_tokens=2048,
        input_cost_per_1m=0.23,
        output_cost_per_1m=0.40,
        supports_function_calling=True,
        training_cutoff="Jul 2024",
    ),
    # DeepSeek Models
    "deepseek-v3": ModelInfo(
        provider="deepseek",
        name="deepseek-v3",
        context_window=128000,
        max_output_tokens=8192,
        input_cost_per_1m=0.14,
        output_cost_per_1m=0.28,
        supports_function_calling=True,
        training_cutoff="2024",
    ),
    "deepseek-r1": ModelInfo(
        provider="deepseek",
        name="deepseek-r1",
        context_window=128000,
        max_output_tokens=8192,
        input_cost_per_1m=0.55,
        output_cost_per_1m=2.19,
        supports_function_calling=True,
        training_cutoff="2024",
    ),
    # Mistral Models
    "mistral-large-2": ModelInfo(
        provider="mistral",
        name="mistral-large-2",
        context_window=128000,
        max_output_tokens=4096,
        input_cost_per_1m=2.00,
        output_cost_per_1m=6.00,
        supports_function_calling=True,
        training_cutoff="2024",
    ),
    "mistral-small-2409": ModelInfo(
        provider="mistral",
        name="mistral-small-2409",
        context_window=128000,
        max_output_tokens=4096,
        input_cost_per_1m=0.20,
        output_cost_per_1m=0.60,
        supports_function_calling=True,
        training_cutoff="2024",
    ),
    "mixtral-8x7b": ModelInfo(
        provider="mistral",
        name="mixtral-8x7b",
        context_window=32000,
        max_output_tokens=4096,
        input_cost_per_1m=0.50,
        output_cost_per_1m=0.50,
        supports_function_calling=True,
        training_cutoff="2024",
    ),
    # Cohere Models
    "command-r-plus": ModelInfo(
        provider="cohere",
        name="command-r-plus",
        context_window=128000,
        max_output_tokens=4096,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_function_calling=True,
        training_cutoff="2024",
    ),
    "command-r": ModelInfo(
        provider="cohere",
        name="command-r",
        context_window=128000,
        max_output_tokens=4096,
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.50,
        supports_function_calling=True,
        training_cutoff="2024",
    ),
}

# Model name aliases and variations
MODEL_ALIASES = {
    # OpenAI aliases
    "gpt-4-turbo-preview": "gpt-4-turbo",
    "gpt-4-1106-preview": "gpt-4-turbo",
    "gpt-4-0125-preview": "gpt-4-turbo",
    "gpt-4-0613": "gpt-4",
    "gpt-4-0314": "gpt-4",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo",
    # OpenAI GPT-5 aliases
    "gpt5": "gpt-5",
    "gpt5-mini": "gpt-5-mini",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-pro": "gpt-5-pro",
    # Anthropic aliases
    "claude-3-opus-20240229": "claude-3-opus",
    "claude-3-sonnet-20240229": "claude-3.5-sonnet",
    "claude-3-haiku-20240307": "claude-3.5-haiku",
    "claude-3.5-sonnet-20241022": "claude-3.5-sonnet",
    "claude-3.5-haiku-20241022": "claude-3.5-haiku",
    # Anthropic Claude 4 aliases
    "claude-4-opus": "claude-opus-4",
    "claude-4-sonnet": "claude-sonnet-4",
    # Google aliases
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-1.5-flash",
    "text-embedding-004": "gemini-text-embedding",
    # Common shortened names
    "gpt4": "gpt-4",
    "gpt4o": "gpt-4o",
    "gpt35": "gpt-3.5-turbo",
    "claude": "claude-3.5-sonnet",
    "gemini": "gemini-1.5-pro",
}


def detect_llm_provider(arguments: dict[str, Any]) -> Optional[str]:
    """
    Detect LLM provider from function arguments.

    Args:
        arguments: Function arguments

    Returns:
        Detected provider name or None
    """
    # Check for model parameter
    model = arguments.get("model", "")
    if isinstance(model, str) and model:
        model_info = get_model_info(model)
        if model_info:
            return model_info.provider

    # Check for provider-specific parameters
    if any(key in arguments for key in ["messages", "max_tokens", "temperature"]):
        # Common OpenAI parameters
        if "messages" in arguments and isinstance(arguments["messages"], list):
            return "openai"

    # Check for anthropic-specific patterns
    if any(key in arguments for key in ["max_tokens_to_sample", "stop_sequences"]):
        return "anthropic"

    # Check string representations for provider hints
    args_str = str(arguments).lower()
    if "anthropic" in args_str:
        return "anthropic"
    elif "openai" in args_str:
        return "openai"
    elif "google" in args_str or "gemini" in args_str:
        return "google"
    elif "claude" in args_str:
        return "anthropic"

    return None


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """
    Get model information by name.

    Args:
        model_name: Model name (with support for aliases)

    Returns:
        ModelInfo object or None if not found
    """
    if not isinstance(model_name, str):
        return None

    # Normalize model name
    normalized_name = normalize_model_name(model_name)

    # Check aliases first
    if normalized_name in MODEL_ALIASES:
        normalized_name = MODEL_ALIASES[normalized_name]

    # Return model info
    return MODEL_REGISTRY.get(normalized_name)


def normalize_model_name(model: str) -> str:
    """
    Normalize model name for consistent tracking.

    Args:
        model: Raw model name

    Returns:
        Normalized model name
    """
    if not isinstance(model, str):
        return str(model)

    # Remove version suffixes and normalize
    model = model.lower().strip()

    # Remove common prefixes/suffixes
    prefixes_to_remove = ["openai/", "anthropic/", "google/", "meta/", "microsoft/"]
    for prefix in prefixes_to_remove:
        if model.startswith(prefix):
            model = model[len(prefix) :]

    # Handle date-based versions
    model = re.sub(r"-\d{8}$", "", model)  # Remove YYYYMMDD
    model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)  # Remove YYYY-MM-DD

    # Handle version numbers at the end
    model = re.sub(r"-v\d+(\.\d+)*$", "", model)  # Remove version numbers

    return model


def estimate_token_count(
    text: Union[str, list[dict[str, Any]], Any],
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> int:
    """Estimate token count for text or messages.

    Args:
        text: Text string, list of messages, or other content
        model: Optional model name to leverage provider-specific tokenizers
        provider: Optional provider override (e.g., ``"openai"``)

    Returns:
        Estimated token count
    """

    if isinstance(text, list):
        total_tokens = 0
        for message in text:
            if isinstance(message, dict):
                if "content" in message:
                    total_tokens += estimate_token_count(
                        message["content"], model=model, provider=provider
                    )
                total_tokens += 10  # Overhead for message structure
            else:
                total_tokens += estimate_token_count(
                    message, model=model, provider=provider
                )
        return max(1, total_tokens)

    if isinstance(text, dict):
        total_tokens = 0
        for key, value in text.items():
            total_tokens += estimate_token_count(key, model=model, provider=provider)
            total_tokens += estimate_token_count(value, model=model, provider=provider)
        return max(1, total_tokens)

    if not isinstance(text, str):
        text = str(text)

    try:
        provider_tokens = tokenizers.count_tokens(text, model=model, provider=provider)
    except Exception:
        provider_tokens = None

    if provider_tokens is not None:
        return provider_tokens

    if not text:
        return 1

    words = len(text.split())
    chars = len(text)

    if text.strip().startswith("{") and text.strip().endswith("}"):
        return max(1, int(chars / 3))
    if text.count("\n") > 5:
        return max(1, int(chars / 3.5))
    if len([w for w in text.split() if len(w) > 10]) > words * 0.3:
        return max(1, int(chars / 3))
    return max(1, int(chars / 4))


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    """
    Estimate API cost based on model and token usage.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Dictionary with cost information
    """
    model_info = get_model_info(model)

    if not model_info:
        # Fallback pricing for unknown models
        input_cost_per_1m = 1.0
        output_cost_per_1m = 2.0
        provider = "unknown"
    else:
        input_cost_per_1m = model_info.input_cost_per_1m
        output_cost_per_1m = model_info.output_cost_per_1m
        provider = model_info.provider

    # Calculate costs (prices are per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(total_cost, 8),
        "currency": "USD",
        "model": model,
        "provider": provider,
        "input_cost_per_1m": input_cost_per_1m,
        "output_cost_per_1m": output_cost_per_1m,
    }


def extract_llm_metadata(response: Any) -> dict[str, Any]:
    """
    Extract metadata from LLM response objects.

    Args:
        response: LLM response object

    Returns:
        Dictionary of extracted metadata
    """
    metadata: dict[str, Any] = {}

    def _get_value(obj: Any, name: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def _update_usage(usage_obj: Any) -> None:
        prompt = _get_value(usage_obj, "prompt_tokens")
        if prompt is not None:
            metadata.setdefault("llm.usage.prompt_tokens", prompt)
            metadata.setdefault("llm.usage.input_tokens", prompt)

        completion = _get_value(usage_obj, "completion_tokens")
        if completion is not None:
            metadata.setdefault("llm.usage.completion_tokens", completion)
            metadata.setdefault("llm.usage.output_tokens", completion)

        total = _get_value(usage_obj, "total_tokens")
        if total is not None:
            metadata.setdefault("llm.usage.total_tokens", total)

        input_tokens = _get_value(usage_obj, "input_tokens")
        if input_tokens is not None:
            metadata.setdefault("llm.usage.input_tokens", input_tokens)

        output_tokens = _get_value(usage_obj, "output_tokens")
        if output_tokens is not None:
            metadata.setdefault("llm.usage.output_tokens", output_tokens)

    def _update_usage_metadata(usage_obj: Any) -> None:
        prompt = _get_value(usage_obj, "prompt_token_count")
        if prompt is not None:
            metadata.setdefault("llm.usage.prompt_tokens", prompt)
            metadata.setdefault("llm.usage.input_tokens", prompt)

        completion = _get_value(usage_obj, "candidates_token_count")
        if completion is not None:
            metadata.setdefault("llm.usage.completion_tokens", completion)
            metadata.setdefault("llm.usage.output_tokens", completion)

        total = _get_value(usage_obj, "total_token_count")
        if total is not None:
            metadata.setdefault("llm.usage.total_tokens", total)

    usage = _get_value(response, "usage")
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is not None:
        _update_usage(usage)

    usage_metadata = _get_value(response, "usage_metadata")
    if usage_metadata is None and isinstance(response, dict):
        usage_metadata = response.get("usage_metadata")
    if usage_metadata is not None:
        _update_usage_metadata(usage_metadata)

    meta = _get_value(response, "meta")
    if meta is not None:
        tokens_info = _get_value(meta, "tokens")
        if tokens_info is not None:
            input_tokens = _get_value(tokens_info, "input_tokens")
            if input_tokens is not None:
                metadata.setdefault("llm.usage.input_tokens", input_tokens)
            output_tokens = _get_value(tokens_info, "output_tokens")
            if output_tokens is not None:
                metadata.setdefault("llm.usage.output_tokens", output_tokens)
            prompt_tokens = _get_value(tokens_info, "prompt_tokens")
            if prompt_tokens is not None:
                metadata.setdefault("llm.usage.prompt_tokens", prompt_tokens)

    # Extract model information
    model_name = _get_value(response, "model")
    if model_name is not None:
        metadata["llm.model"] = model_name
        model_info = get_model_info(model_name)
        if model_info:
            metadata.setdefault("llm.provider", model_info.provider)
            metadata.setdefault("llm.context_window", model_info.context_window)
            metadata.setdefault("llm.max_output_tokens", model_info.max_output_tokens)

    # Extract system fingerprint for OpenAI
    system_fingerprint = _get_value(response, "system_fingerprint")
    if system_fingerprint is not None:
        metadata["llm.system_fingerprint"] = system_fingerprint

    # Extract finish reason
    choices = _get_value(response, "choices")
    if choices:
        choice = choices[0]
        finish_reason = _get_value(choice, "finish_reason")
        if finish_reason is not None:
            metadata["llm.finish_reason"] = finish_reason

    created = _get_value(response, "created")
    if created is not None:
        metadata["llm.created"] = created

    if "llm.usage.total_tokens" not in metadata:
        prompt_tokens = metadata.get("llm.usage.input_tokens") or metadata.get(
            "llm.usage.prompt_tokens"
        )
        completion_tokens = metadata.get("llm.usage.output_tokens") or metadata.get(
            "llm.usage.completion_tokens"
        )
        if prompt_tokens is not None and completion_tokens is not None:
            metadata["llm.usage.total_tokens"] = prompt_tokens + completion_tokens

    return metadata


def validate_model_compatibility(model: str, messages: list[Any]) -> dict[str, Any]:
    """
    Validate model compatibility and provide suggestions.

    Args:
        model: Model name
        messages: List of messages

    Returns:
        Dictionary with validation results and suggestions
    """
    model_info = get_model_info(model)

    result: dict[str, Any] = {
        "valid": True,
        "warnings": [],
        "suggestions": [],
        "model_info": model_info,
    }

    if not model_info:
        result["valid"] = False
        result["warnings"].append(f"Model '{model}' not found in registry")

        # Suggest similar models
        normalized = normalize_model_name(model)
        suggestions: list[str] = []
        for known_model in MODEL_REGISTRY.keys():
            if normalized in known_model or known_model in normalized:
                suggestions.append(known_model)

        if suggestions:
            result["suggestions"] = suggestions[:3]  # Top 3 suggestions
        else:
            # Fallback suggestions based on common patterns
            if "gpt" in normalized:
                result["suggestions"] = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
            elif "claude" in normalized:
                result["suggestions"] = ["claude-3.5-sonnet", "claude-3.5-haiku"]
            elif "gemini" in normalized:
                result["suggestions"] = ["gemini-1.5-pro", "gemini-1.5-flash"]

    if model_info:
        # Estimate token usage
        estimated_tokens = estimate_token_count(
            messages, model=model, provider=model_info.provider
        )

        if estimated_tokens > model_info.context_window * 0.9:
            result["warnings"].append(
                f"Input may exceed context window ({estimated_tokens} tokens estimated, "
                f"limit is {model_info.context_window})"
            )

        # Check for vision content without vision support
        if not model_info.supports_vision:
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    if isinstance(content, list):
                        for item in content:
                            if (
                                isinstance(item, dict)
                                and item.get("type") == "image_url"
                            ):
                                result["warnings"].append(
                                    f"Model '{model}' does not support vision/image inputs"
                                )
                                break

    return result


def get_supported_models(provider: Optional[str] = None) -> list[str]:
    """
    Get list of supported models, optionally filtered by provider.

    Args:
        provider: Optional provider filter

    Returns:
        List of model names
    """
    if provider:
        return [
            name for name, info in MODEL_REGISTRY.items() if info.provider == provider
        ]
    return list(MODEL_REGISTRY.keys())


def extract_prompt_template_variables(prompt: str) -> list[str]:
    """
    Extract template variables from a prompt string.

    Args:
        prompt: Prompt string with template variables

    Returns:
        List of variable names found in the prompt
    """
    variables = set()

    # Find variables in {variable} format
    variables.update(re.findall(r"\{([^}]+)\}", prompt))

    # Find variables in {{variable}} format (Jinja2 style)
    variables.update(re.findall(r"\{\{([^}]+)\}\}", prompt))

    # Find variables in ${variable} format
    variables.update(re.findall(r"\$\{([^}]+)\}", prompt))

    return [var.strip() for var in variables]


def sanitize_llm_content(content: str) -> str:
    """
    Sanitize LLM content for safe logging.

    Args:
        content: Content to sanitize

    Returns:
        Sanitized content
    """
    if not isinstance(content, str):
        content = str(content)

    # Remove or mask sensitive patterns
    patterns = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
        (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]"),
        (r"\b[A-Z0-9]{20,}\b", "[TOKEN]"),  # API keys, tokens
        (r"sk-[a-zA-Z0-9]{32,}", "[API_KEY]"),  # OpenAI API keys
        (r"xoxb-[0-9]{12}-[0-9]{12}-[a-zA-Z0-9]{24}", "[SLACK_TOKEN]"),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content
