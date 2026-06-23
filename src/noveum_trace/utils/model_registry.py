"""
Model registry for Noveum Trace SDK.

Contains ModelInfo dataclass, MODEL_REGISTRY, and MODEL_ALIASES.
Pricing is per 1M tokens in USD. Verify latest pricing from provider docs.
Last updated: June 2026.
"""

from dataclasses import dataclass
from typing import Optional


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


# ---------------------------------------------------------------------------
# MODEL_REGISTRY
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, ModelInfo] = {
    # -----------------------------------------------------------------------
    # OpenAI GPT-4.1 Family
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # OpenAI Reasoning Models
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # OpenAI GPT-4 Family
    # -----------------------------------------------------------------------
    "gpt-4o": ModelInfo(
        provider="openai",
        name="gpt-4o",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
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
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
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
    # -----------------------------------------------------------------------
    # OpenAI GPT-5 Family
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # OpenAI GPT-3.5 Family
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # Google Gemini Family
    # -----------------------------------------------------------------------
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
    # Gemini 1.5 pricing published per 1K characters; converted to per 1M tokens
    # assuming 4 characters per token.
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
    # -----------------------------------------------------------------------
    # Anthropic Claude — Legacy (Claude 2 / Instant)
    # -----------------------------------------------------------------------
    "claude-instant-1.2": ModelInfo(
        provider="anthropic",
        name="claude-instant-1.2",
        context_window=100000,
        max_output_tokens=4096,
        input_cost_per_1m=0.80,
        output_cost_per_1m=2.40,
        supports_function_calling=False,
        training_cutoff="Early 2023",
    ),
    "claude-2.0": ModelInfo(
        provider="anthropic",
        name="claude-2.0",
        context_window=100000,
        max_output_tokens=4096,
        input_cost_per_1m=8.00,
        output_cost_per_1m=24.00,
        supports_function_calling=False,
        training_cutoff="Early 2023",
    ),
    "claude-2.1": ModelInfo(
        provider="anthropic",
        name="claude-2.1",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=8.00,
        output_cost_per_1m=24.00,
        supports_function_calling=False,
        training_cutoff="Early 2023",
    ),
    # -----------------------------------------------------------------------
    # Anthropic Claude 3 Family
    # -----------------------------------------------------------------------
    "claude-3-haiku": ModelInfo(
        provider="anthropic",
        name="claude-3-haiku",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=0.25,
        output_cost_per_1m=1.25,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Aug 2023",
    ),
    "claude-3-haiku-20240307": ModelInfo(
        provider="anthropic",
        name="claude-3-haiku-20240307",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=0.25,
        output_cost_per_1m=1.25,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Aug 2023",
    ),
    "claude-3-sonnet": ModelInfo(
        provider="anthropic",
        name="claude-3-sonnet",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Aug 2023",
    ),
    "claude-3-sonnet-20240229": ModelInfo(
        provider="anthropic",
        name="claude-3-sonnet-20240229",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Aug 2023",
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
    "claude-3-opus-20240229": ModelInfo(
        provider="anthropic",
        name="claude-3-opus-20240229",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Aug 2023",
    ),
    # -----------------------------------------------------------------------
    # Anthropic Claude 3.5 Family
    # -----------------------------------------------------------------------
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
    "claude-3.5-haiku-20241022": ModelInfo(
        provider="anthropic",
        name="claude-3.5-haiku-20241022",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
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
    "claude-3.5-sonnet-20240620": ModelInfo(
        provider="anthropic",
        name="claude-3.5-sonnet-20240620",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    "claude-3.5-sonnet-20241022": ModelInfo(
        provider="anthropic",
        name="claude-3.5-sonnet-20241022",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    # -----------------------------------------------------------------------
    # Anthropic Claude 3.7 Family
    # -----------------------------------------------------------------------
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
    "claude-3.7-sonnet-20250219": ModelInfo(
        provider="anthropic",
        name="claude-3.7-sonnet-20250219",
        context_window=200000,
        max_output_tokens=128000,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="Apr 2024",
    ),
    # -----------------------------------------------------------------------
    # Anthropic Claude 4 Family
    # -----------------------------------------------------------------------
    "claude-haiku-4": ModelInfo(
        provider="anthropic",
        name="claude-haiku-4",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
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
    # Versioned Claude 4 IDs (date/patch suffixes)
    "claude-haiku-4-5": ModelInfo(
        provider="anthropic",
        name="claude-haiku-4-5",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1m=1.00,
        output_cost_per_1m=5.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="2025",
    ),
    "claude-sonnet-4-6": ModelInfo(
        provider="anthropic",
        name="claude-sonnet-4-6",
        context_window=200000,
        max_output_tokens=64000,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="2025",
    ),
    "claude-opus-4-8": ModelInfo(
        provider="anthropic",
        name="claude-opus-4-8",
        context_window=200000,
        max_output_tokens=32768,
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        supports_vision=True,
        supports_function_calling=True,
        training_cutoff="2025",
    ),
    # -----------------------------------------------------------------------
    # Meta Llama Family
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # DeepSeek Models
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # Mistral Models
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # Cohere Models
    # -----------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# MODEL_ALIASES — maps variant names → canonical registry keys
# ---------------------------------------------------------------------------
MODEL_ALIASES: dict[str, str] = {
    # OpenAI
    "gpt-4-turbo-preview": "gpt-4-turbo",
    "gpt-4-1106-preview": "gpt-4-turbo",
    "gpt-4-0125-preview": "gpt-4-turbo",
    "gpt-4-0613": "gpt-4",
    "gpt-4-0314": "gpt-4",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo",
    # OpenAI GPT-5 shorthand
    "gpt5": "gpt-5",
    "gpt5-mini": "gpt-5-mini",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-pro": "gpt-5-pro",
    # Common shorthand
    "gpt4": "gpt-4",
    "gpt4o": "gpt-4o",
    "gpt35": "gpt-3.5-turbo",
    # Anthropic — legacy versioned IDs already in registry as canonical entries;
    # these aliases cover alternate spellings / pre-normalisation names.
    "claude-3-opus-20240229": "claude-3-opus-20240229",  # identity (normaliser strips date)
    "claude-3-sonnet-20240229": "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307": "claude-3-haiku-20240307",
    "claude-3.5-sonnet-20241022": "claude-3.5-sonnet-20241022",
    "claude-3.5-haiku-20241022": "claude-3.5-haiku-20241022",
    # Alternate Claude 4 spellings
    "claude-4-opus": "claude-opus-4",
    "claude-4-sonnet": "claude-sonnet-4",
    "claude-4-haiku": "claude-haiku-4",
    # Shorthand
    "claude": "claude-3.5-sonnet",
    # Google
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-1.5-flash",
    "gemini": "gemini-1.5-pro",
    "text-embedding-004": "gemini-text-embedding",
}
