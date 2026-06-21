from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import httpx

from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.types import BlockResponseMode, ParsedRequest, ParsedResponse


class ProviderAdapter(ABC):
    """Translate httpx Request/Response ↔ ParsedRequest/ParsedResponse.

    Also responsible for building the synthetic block response so each provider
    gets an error body that its SDK can parse cleanly.
    """

    provider_name: ClassVar[str]
    host_patterns: ClassVar[list[str]]  # e.g. ["api.openai.com", "api.groq.com"]
    body_signature_keys: ClassVar[
        list[str]
    ]  # fallback: keys that identify this provider's request body
    # Request headers unique to this provider. When non-empty, the body-shape
    # fallback requires at least one to be present. This prevents a provider
    # whose body signature is a *subset* of another's (OpenAI's {model, messages}
    # ⊂ Anthropic's {model, messages, max_tokens}) from being misclassified on a
    # custom base_url. Empty = no header requirement.
    header_signatures: ClassVar[list[str]] = []

    @abstractmethod
    def parse_request(self, req: httpx.Request) -> ParsedRequest: ...

    @abstractmethod
    def parse_response(
        self, req: httpx.Request, resp: httpx.Response
    ) -> ParsedResponse: ...

    @abstractmethod
    def synthetic_block_response(
        self,
        req: httpx.Request,
        decision: PolicyDecision,
        mode: BlockResponseMode = BlockResponseMode.provider_error,
    ) -> httpx.Response: ...


class AdapterRegistry:
    """Resolves the right ProviderAdapter for an httpx.Request.

    Resolution order:
    1. host_patterns match against request URL host
    2. body_signature_keys fallback for custom base_url (e.g. Azure OpenAI),
       gated by header_signatures so subset body signatures don't collide
    """

    def __init__(self, adapters: Optional[list[ProviderAdapter]] = None) -> None:
        self._adapters: list[ProviderAdapter] = list(adapters or [])

    def register(self, adapter: ProviderAdapter) -> None:
        self._adapters.append(adapter)

    def for_request(self, req: httpx.Request) -> Optional[ProviderAdapter]:
        host = req.url.host
        for adapter in self._adapters:
            if any(pattern in host for pattern in adapter.host_patterns):
                return adapter

        # Body-shape fallback for custom base_url — most specific signature wins.
        try:
            body = json.loads(req.content)
        except Exception:
            return None

        candidates = sorted(
            self._adapters,
            key=lambda a: len(a.body_signature_keys),
            reverse=True,
        )
        for adapter in candidates:
            # An adapter that declares unique headers must see one of them before
            # its body signature is considered — otherwise an OpenAI body carrying
            # max_tokens would match Anthropic's superset signature first.
            if adapter.header_signatures and not any(
                h in req.headers for h in adapter.header_signatures
            ):
                continue
            if all(k in body for k in adapter.body_signature_keys):
                return adapter

        return None


# Module-level default registry populated by the two concrete adapters below.
# Transport code imports this singleton; tests can replace it.
_default_registry: Optional[AdapterRegistry] = None


def default_registry() -> AdapterRegistry:
    global _default_registry
    if _default_registry is None:
        from noveum_trace.guard.transport.adapters.anthropic_adapter import (
            AnthropicAdapter,
        )
        from noveum_trace.guard.transport.adapters.openai_adapter import OpenAIAdapter

        _default_registry = AdapterRegistry([OpenAIAdapter(), AnthropicAdapter()])
    return _default_registry
