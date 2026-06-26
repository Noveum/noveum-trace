"""Generic NovaGuard example — framework-agnostic.

This example demonstrates the two recommended integration patterns for NovaGuard
when you are not using a specific framework like CrewAI:

Pattern A — Transport-level (recommended):
    Wrap the httpx client so every LLM call passes through Guard automatically.
    Works with openai, anthropic, and any httpx-based client.

Pattern B — Interceptor-level (manual):
    Call before_llm_call / after_llm_call around your LLM invocations manually.
    Use this when you cannot inject a custom httpx transport.

Both patterns use the same backend-polled policy configuration — you do NOT need
to pass policies explicitly when guard_enabled=True; the backend poller fetches
and upserts them automatically.
"""

from __future__ import annotations

import uuid

import noveum_trace
from noveum_trace.guard import (
    CostCapPolicy,
    EnforcementMode,
    GuardAPIClient,
    NoveumGuardBlocked,
    PolicyEngine,
    PolicyPoller,
    async_http_client,
    http_client,
)
from noveum_trace.guard.types import PolicyContext

# ---------------------------------------------------------------------------
# Pattern A: Transport-level guard (zero-code enforcement after init)
# ---------------------------------------------------------------------------


# 1. Initialize the SDK with guard enabled.
#    The backend poller will automatically fetch your configured policies
#    from the Noveum dashboard and keep them in sync without any local config.
noveum_trace.init(
    project="my-llm-app",
    api_key="your-noveum-api-key",  # or set NOVEUM_API_KEY env var
    guard_enabled=True,
    environment="production",
)

# 2. Create a guarded httpx client and pass it to your LLM provider SDK.
#    Every call is automatically checked against all active policies.
guarded_client = http_client()

# Example with OpenAI:
try:
    import openai

    openai_client = openai.OpenAI(
        api_key="your-openai-api-key",
        http_client=guarded_client,  # <-- Guard is wired in here
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Explain quantum entanglement briefly."}],
        max_tokens=200,
    )
    print("OpenAI response:", response.choices[0].message.content)

except Exception as e:
    # NoveumGuardBlocked is raised (as an HTTP 403) when a policy blocks the call.
    print(f"Call blocked or failed: {e}")


# Example with Anthropic (async variant):
async def anthropic_example() -> None:
    try:
        import anthropic

        async with async_http_client() as guarded_async_client:
            async with anthropic.AsyncAnthropic(
                api_key="your-anthropic-api-key",
                http_client=guarded_async_client,
            ) as anthropic_client:
                message = await anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=200,
                    messages=[
                        {"role": "user", "content": "What is the capital of France?"}
                    ],
                )
                print("Anthropic response:", message.content[0].text)

    except Exception as e:
        print(f"Call blocked or failed: {e}")


# ---------------------------------------------------------------------------
# Pattern B: Manual interceptor (no transport injection needed)
# ---------------------------------------------------------------------------


def manual_guard_example() -> None:
    """Show how to manually enforce Guard around any LLM call."""

    # Build the engine.  When guard_enabled=True in init() you can skip this —
    # just import from noveum_trace.guard._state instead.
    api = GuardAPIClient(api_key="your-noveum-api-key")
    engine = PolicyEngine(api_client=api)

    # Policies can be added locally (backwards-compat) OR fetched from the backend.
    # With guard_enabled=True and no explicit policies, the poller handles this.
    cap = CostCapPolicy(
        max_usd=5.00,
        mode=EnforcementMode.strict,
        project_id="my-llm-app",
    )
    engine.attach(cap)

    ctx = PolicyContext(
        project_id="my-llm-app",
        organization_id=None,
        environment="production",
        trace_id=None,
        span_id=None,
        call_id=str(uuid.uuid4()),
    )

    # Start the poller so policies stay in sync with the backend.
    poller = PolicyPoller(engine, project_id="my-llm-app")
    poller.start()

    # NoveumCrewAIInterceptor implements the pre/post call contract and works with
    # any framework — the name reflects its origin, not a CrewAI requirement.
    from noveum_trace.guard.integrations.crewai import NoveumCrewAIInterceptor

    interceptor = NoveumCrewAIInterceptor(engine, ctx)

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100,
        "provider": "openai",
    }

    try:
        call_id, ran = interceptor.before_llm_call(payload)

        # --- your actual LLM call goes here ---
        response = {
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 12, "completion_tokens": 30},
            "choices": [{"message": {"content": "Hello! How can I help?"}}],
        }
        # --- end LLM call ---

        interceptor.after_llm_call(call_id, payload, response, ran)
        print("Response:", response["choices"][0]["message"]["content"])

    except NoveumGuardBlocked as e:
        print(f"Call blocked by policy '{e.policy_name}': {e.reason}")
    finally:
        poller.stop()


if __name__ == "__main__":
    manual_guard_example()
    noveum_trace.shutdown()
