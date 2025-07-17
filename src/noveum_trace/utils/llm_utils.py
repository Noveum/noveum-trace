"""
LLM utility functions for Noveum Trace SDK.

This module provides utility functions for working with LLM providers,
token counting, cost estimation, and metadata extraction.
"""

import re
from typing import Any, Optional, Union


def detect_llm_provider(arguments: dict[str, Any]) -> Optional[str]:
    """
    Detect LLM provider from function arguments.

    Args:
        arguments: Function arguments

    Returns:
        Detected provider name or None
    """
    # Check for OpenAI patterns
    if any(key in arguments for key in ["model", "messages", "prompt"]):
        model = arguments.get("model", "")
        if isinstance(model, str):
            if model.startswith(
                ("gpt-", "text-", "davinci", "curie", "babbage", "ada")
            ):
                return "openai"
            elif model.startswith("claude"):
                return "anthropic"
            elif "gemini" in model.lower():
                return "google"
            elif "llama" in model.lower():
                return "meta"

    # Check for provider-specific parameters
    if "anthropic" in str(arguments).lower():
        return "anthropic"
    elif "openai" in str(arguments).lower():
        return "openai"

    return None


def estimate_token_count(text: Union[str, list[dict[str, Any]]]) -> int:
    """
    Estimate token count for text or messages.

    This is a rough estimation based on common tokenization patterns.
    For production use, consider using tiktoken or similar libraries.

    Args:
        text: Text string or list of messages

    Returns:
        Estimated token count
    """
    if isinstance(text, list):
        # Handle messages format
        total_tokens = 0
        for message in text:
            if isinstance(message, dict) and "content" in message:
                total_tokens += estimate_token_count(message["content"])
            else:
                total_tokens += estimate_token_count(str(message))
        return total_tokens

    if not isinstance(text, str):
        text = str(text)

    # Simple estimation: ~4 characters per token on average
    # This is a rough approximation and varies by model and language
    return max(1, len(text) // 4)


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
    # Pricing data (as of 2024 - should be updated regularly)
    pricing = {
        # OpenAI GPT-4 models
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        # OpenAI GPT-3.5 models
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        # Anthropic Claude models
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        # Default fallback
        "default": {"input": 0.001, "output": 0.002},
    }

    # Get pricing for model (fallback to default)
    model_pricing = pricing.get(model, pricing["default"])

    # Calculate costs (prices are per 1K tokens)
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "currency": "USD",
        "model": model,
    }


def extract_llm_metadata(response: Any) -> dict[str, Any]:
    """
    Extract metadata from LLM response objects.

    Args:
        response: LLM response object

    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}

    # Handle OpenAI response format
    if hasattr(response, "usage"):
        usage = response.usage
        metadata.update(
            {
                "llm.usage.prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "llm.usage.completion_tokens": getattr(usage, "completion_tokens", 0),
                "llm.usage.total_tokens": getattr(usage, "total_tokens", 0),
            }
        )

    if hasattr(response, "model"):
        metadata["llm.model"] = response.model

    # Handle Anthropic response format
    if hasattr(response, "usage"):
        usage = response.usage
        metadata.update(
            {
                "llm.usage.input_tokens": getattr(usage, "input_tokens", 0),
                "llm.usage.output_tokens": getattr(usage, "output_tokens", 0),
            }
        )

    # Extract finish reason
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "finish_reason"):
            metadata["llm.finish_reason"] = choice.finish_reason

    return metadata


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

    # Common normalizations
    normalizations = {
        r"gpt-4-\d+k": "gpt-4",
        r"gpt-3\.5-turbo-\d+k": "gpt-3.5-turbo",
        r"claude-3-opus-\d+": "claude-3-opus",
        r"claude-3-sonnet-\d+": "claude-3-sonnet",
        r"claude-3-haiku-\d+": "claude-3-haiku",
    }

    for pattern, replacement in normalizations.items():
        if re.match(pattern, model):
            return replacement

    return model


def extract_prompt_template_variables(prompt: str) -> list[str]:
    """
    Extract template variables from a prompt string.

    Args:
        prompt: Prompt string with template variables

    Returns:
        List of variable names found in the prompt
    """
    # Find variables in {variable} format
    variables = re.findall(r"\{([^}]+)\}", prompt)

    # Find variables in {{variable}} format (Jinja2 style)
    jinja_variables = re.findall(r"\{\{([^}]+)\}\}", prompt)

    # Combine and deduplicate
    all_variables = list(set(variables + jinja_variables))

    return [var.strip() for var in all_variables]


def sanitize_llm_content(content: str, max_length: int = 1000) -> str:
    """
    Sanitize LLM content for safe logging.

    Args:
        content: Content to sanitize
        max_length: Maximum length of content

    Returns:
        Sanitized content
    """
    if not isinstance(content, str):
        content = str(content)

    # Truncate if too long
    if len(content) > max_length:
        content = content[: max_length - 3] + "..."

    # Remove or mask sensitive patterns
    # Email addresses
    content = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", content
    )

    # Phone numbers (simple pattern)
    content = re.sub(r"\b\d{3}-\d{3}-\d{4}\b", "[PHONE]", content)

    # Credit card numbers (simple pattern)
    content = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]", content)

    return content
