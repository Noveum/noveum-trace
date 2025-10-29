"""Provider-specific token counting utilities.

This module centralizes integrations with provider tokenizers while ensuring
graceful degradation when optional dependencies are unavailable. Each
tokenizer returns ``None`` when a provider-specific count cannot be produced,
allowing callers to fall back to heuristic estimates.

All helpers accept loose ``Any`` input to align with the diverse payloads used
by SDK integrations (strings, dicts, chat messages, etc.).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Optional

try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None

try:  # pragma: no cover - optional dependency
    from anthropic import Tokenizer as AnthropicTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AnthropicTokenizer = None


def _normalize_content(content: Any) -> str:
    """Serialize arbitrary prompt structures into a single string."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, bytes):
        try:
            return content.decode("utf-8")
        except Exception:
            return content.decode("utf-8", errors="ignore")

    if isinstance(content, list):
        return "\n".join(_normalize_content(item) for item in content)

    if isinstance(content, dict):
        parts: list[str] = []
        for key, value in content.items():
            parts.append(f"{key}:{_normalize_content(value)}")
        return "\n".join(parts)

    return str(content)


def _infer_provider(model: Optional[str]) -> Optional[str]:
    if not model:
        return None

    name = model.lower()
    if name.startswith("gpt") or "openai" in name:
        return "openai"
    if "claude" in name or "anthropic" in name:
        return "anthropic"
    if "gemini" in name or "google" in name:
        return "google"
    if "cohere" in name or "command" in name:
        return "cohere"
    if "mistral" in name or "mixtral" in name:
        return "mistral"
    if "llama" in name or "meta" in name:
        return "meta"
    if "deepseek" in name:
        return "deepseek"
    return None


@lru_cache(maxsize=32)
def _get_tiktoken_encoding(model: Optional[str]):  # pragma: no cover - cache
    if tiktoken is None:
        return None

    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            pass

    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_anthropic_tokenizer():  # pragma: no cover - optional dependency
    if AnthropicTokenizer is None:
        return None
    try:
        return AnthropicTokenizer()
    except Exception:
        return None


def _count_openai_tokens(text: str, model: Optional[str]) -> Optional[int]:
    encoding = _get_tiktoken_encoding(model)
    if encoding is None:
        return None

    try:
        return max(1, len(encoding.encode(text)))
    except Exception:
        return None


def _count_anthropic_tokens(text: str, model: Optional[str]) -> Optional[int]:
    tokenizer = _get_anthropic_tokenizer()
    if tokenizer is not None:
        try:
            return max(1, int(tokenizer.count_tokens(text)))  # type: ignore[arg-type]
        except Exception:
            pass

    # Fall back to the OpenAI tokenizer which closely approximates Claude
    # tokenization for English text.
    return _count_openai_tokens(text, model or "claude-3.5-sonnet")


def _count_gemini_tokens(text: str, model: Optional[str]) -> Optional[int]:
    """Approximate Gemini token counts based on documented pricing."""

    if not text:
        return 1

    name = (model or "").lower()

    if "1.5" in name:
        # Gemini 1.5 pricing is published per 1K characters, so approximate
        # tokens using four characters per token.
        divisor = 4
    elif "flash" in name:
        # Gemini 2.x Flash models operate on compact tokens.
        divisor = 3.5
    else:
        divisor = 4

    return max(1, math.ceil(len(text) / divisor))


def count_tokens(
    content: Any,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[int]:
    """Return provider-specific token counts when possible."""

    provider_name = (provider or _infer_provider(model) or "").lower()
    text = _normalize_content(content)

    if not text:
        return 1

    if provider_name == "openai":
        return _count_openai_tokens(text, model)

    if provider_name == "anthropic":
        return _count_anthropic_tokens(text, model)

    if provider_name in {"google", "gemini"}:
        return _count_gemini_tokens(text, model)

    # Cohere, Mistral, DeepSeek, Meta Llama currently fall back to heuristics.
    return None


__all__ = ["count_tokens"]

