from noveum_trace.guard._state import attach_policy, detach_policy, refresh
from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.api_client_http import HttpGuardAPIClient
from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.exceptions import GuardBackendUnavailable, NoveumGuardBlocked
from noveum_trace.guard.integrations.bedrock import instrument_bedrock
from noveum_trace.guard.integrations.crewai import NoveumCrewAIInterceptor
from noveum_trace.guard.policies import AbstractPolicy, CostCapPolicy
from noveum_trace.guard.poller import PolicyPoller
from noveum_trace.guard.transport import (
    NoveumAsyncTransport,
    NoveumTransport,
    async_http_client,
    http_client,
)
from noveum_trace.guard.transport.adapters.base import default_registry
from noveum_trace.guard.types import (
    Action,
    BlockResponseMode,
    EnforcementMode,
    ParsedRequest,
    ParsedResponse,
    Phase,
    PolicyContext,
    PolicyDeps,
)


def supported_providers() -> list[str]:
    """Provider names NovaGuard can enforce policies on, across both mechanisms.

    Most entries (OpenAI + OpenAI-compatible hosts, Anthropic) come from the
    default ProviderAdapter registry: enforcement requires one explicit
    ``http_client()``/``async_http_client()`` call PER client instance you
    construct (``openai.OpenAI(http_client=noveum_trace.guard.http_client())``).

    "bedrock" is covered differently: AWS Bedrock (boto3/botocore) never
    produces an httpx.Request, so it can't go through that registry — see the
    module docstring in ``guard.transport.adapters.base`` for why. Instead,
    ``instrument_bedrock()`` is called ONCE, process-wide, typically right
    after ``noveum_trace.init(...)`` — it patches botocore itself, so it
    covers every bedrock-runtime client the app builds afterward (or already
    built), from any boto3.Session(), not just one client instance. Both
    mechanisms are equally one line and equally explicit opt-in; they differ
    in scope — per-client-instance (OpenAI/Anthropic) vs. process-wide
    (Bedrock) — not in whether wiring is required at all.

    Google Vertex AI / the native Gemini SDK are still not covered by either
    mechanism.
    """
    return sorted({*default_registry().provider_names(), "bedrock"})


__all__ = [
    # Core
    "GuardAPIClient",
    "HttpGuardAPIClient",
    "PolicyDecision",
    "PolicyEngine",
    "NoveumGuardBlocked",
    "GuardBackendUnavailable",
    "PolicyPoller",
    # Policies
    "AbstractPolicy",
    "CostCapPolicy",
    # Transport
    "NoveumTransport",
    "NoveumAsyncTransport",
    "http_client",
    "async_http_client",
    # Integrations
    "NoveumCrewAIInterceptor",
    "instrument_bedrock",
    # Types / enums
    "Action",
    "BlockResponseMode",
    "EnforcementMode",
    "ParsedRequest",
    "ParsedResponse",
    "Phase",
    "PolicyContext",
    "PolicyDeps",
    # Runtime policy management
    "attach_policy",
    "detach_policy",
    "refresh",
    # Coverage introspection
    "supported_providers",
]
