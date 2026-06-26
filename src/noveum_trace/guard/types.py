from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from noveum_trace.guard.api_client import GuardAPIClient


class EnforcementMode(str, Enum):
    strict = "strict"  # pre-call waits on backend; post reconciles
    non_strict = "non_strict"  # pre decides locally; post reports


class Action(str, Enum):
    block = "block"
    allow = "allow"


class Phase(str, Enum):
    pre = "pre"
    post = "post"


class BlockResponseMode(str, Enum):
    provider_error = "provider_error"  # HTTP 403 → SDK raises, no retry
    synthetic_success = "synthetic_success"  # HTTP 200 with canned body


@dataclass
class ParsedRequest:
    provider: str  # "openai" | "anthropic"
    model: str
    messages: list[Any]
    stream: bool
    max_tokens: Optional[int]
    estimated_input_tokens: int
    raw_body: bytes  # original serialized body for rewrite


@dataclass
class ParsedResponse:
    model: str
    text: Optional[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class PolicyContext:
    project_id: str
    organization_id: Optional[str]
    environment: str
    trace_id: Optional[str]
    span_id: Optional[str]
    call_id: str  # UUID minted per call in the transport; flows pre → post → release


@dataclass
class PolicyDeps:
    api: GuardAPIClient  # the shared stub seam
