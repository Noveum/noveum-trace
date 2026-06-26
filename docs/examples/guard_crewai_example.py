"""NovaGuard + CrewAI integration example.

This example shows how to enforce NovaGuard cost-cap (and any other active)
policies inside a CrewAI multi-agent workflow.

There are two recommended approaches:

Approach 1 — Transport-level (strongly preferred):
    Pass a guarded httpx client to ``crewai.LLM(client_params=...)`` so every
    LLM call is intercepted transparently with zero agent-code changes.

Approach 2 — Interceptor-level (fallback):
    Use ``NoveumCrewAIInterceptor`` to manually gate each LLM call.  Use this
    only if you cannot inject a custom transport (e.g. when using a CrewAI
    version that doesn't support ``client_params``).

Backend-polled policies (recommended):
    With ``guard_enabled=True`` and no ``policies=`` argument, the poller
    automatically fetches and syncs policies from the Noveum dashboard.
    This is the single-source-of-truth path recommended by the team.

Local policies (backwards-compat / dev/test):
    You can still pass ``policies=[CostCapPolicy(...)]`` to ``init()`` for
    local testing.  A ``DeprecationWarning`` is emitted when this is combined
    with ``guard_enabled=True``.
"""

from __future__ import annotations

import noveum_trace
from noveum_trace.guard import (
    CostCapPolicy,
    EnforcementMode,
    NoveumCrewAIInterceptor,
    NoveumGuardBlocked,
    http_client,
)

# get_context / get_engine are not in the public __all__; import directly from
# _state for now. Prefer the zero-arg http_client() approach (Approach 1) which
# resolves engine/context automatically without touching internal modules.
from noveum_trace.guard._state import get_context, get_engine

# ---------------------------------------------------------------------------
# 1. Initialize the SDK
# ---------------------------------------------------------------------------

# Recommended: backend-polled policies only — no local policies= needed.
noveum_trace.init(
    project="crewai-demo",
    api_key="your-noveum-api-key",  # or NOVEUM_API_KEY env var
    guard_enabled=True,
    environment="production",
)

# Dev/test override: specify policies locally.
# NOTE: emits DeprecationWarning when combined with guard_enabled=True.
# noveum_trace.init(
#     project="crewai-demo",
#     api_key="your-noveum-api-key",
#     guard_enabled=True,
#     policies=[CostCapPolicy(max_usd=10.0, mode=EnforcementMode.strict)],
# )


# ---------------------------------------------------------------------------
# Approach 1: Transport-level (recommended)
# ---------------------------------------------------------------------------


def run_crew_transport_approach() -> None:
    """Wire the guarded client into CrewAI's LLM so enforcement is transparent."""
    try:
        from crewai import LLM, Agent, Crew, Task
    except ImportError:
        print("crewai not installed — skipping transport approach demo.")
        return

    # http_client() returns an httpx.Client; use it as a context manager so the
    # underlying connection pool is closed after the crew run completes.
    with http_client() as guarded_client:
        llm = LLM(
            model="gpt-4o-mini",
            api_key="your-openai-api-key",
            # Pass the guarded httpx client so Guard intercepts every provider call.
            client_params={"http_client": guarded_client},
        )

        researcher = Agent(
            role="Research Analyst",
            goal="Find key facts about quantum computing",
            backstory="Expert researcher with deep knowledge of quantum physics.",
            llm=llm,
            verbose=True,
        )

        research_task = Task(
            description="Summarize the top 3 applications of quantum computing in 100 words.",
            expected_output="A concise 100-word summary of quantum computing applications.",
            agent=researcher,
        )

        crew = Crew(agents=[researcher], tasks=[research_task], verbose=True)

        try:
            result = crew.kickoff()
            print("Crew result:", result)
        except Exception as e:
            # Guard blocks surface as HTTP 403 errors from the provider SDK.
            print(f"Crew blocked by Guard: {e}")


# ---------------------------------------------------------------------------
# Approach 2: Interceptor-level (manual fallback)
# ---------------------------------------------------------------------------


def run_crew_interceptor_approach() -> None:
    """Manually gate LLM calls via NoveumCrewAIInterceptor.

    Use this when you cannot inject a custom httpx transport into CrewAI.
    """
    # Retrieve the engine and context that init() already bootstrapped.
    engine = get_engine()
    ctx = get_context()
    if engine is None or ctx is None:
        raise RuntimeError("Guard not initialized — call noveum_trace.init() first.")

    interceptor = NoveumCrewAIInterceptor(engine, ctx)

    # Simulate the payload your CrewAI LLM wrapper would produce.
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful research assistant."},
            {
                "role": "user",
                "content": "What are the top 3 uses of quantum computing?",
            },
        ],
        "max_tokens": 200,
        "provider": "openai",
    }

    try:
        # Gate the call — raises NoveumGuardBlocked if any policy blocks.
        call_id, ran = interceptor.before_llm_call(payload)

        # ---- your actual LLM call goes here --------------------------------
        # e.g. response = your_llm_wrapper.call(payload)
        # For this demo we use a mock response:
        response = {
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 45, "completion_tokens": 120},
            "choices": [
                {
                    "message": {
                        "content": (
                            "1. Cryptography & security\n"
                            "2. Drug discovery & molecular simulation\n"
                            "3. Optimization problems in logistics"
                        )
                    }
                }
            ],
        }
        # --------------------------------------------------------------------

        # Reconcile spend after the call completes.
        interceptor.after_llm_call(call_id, payload, response, ran)
        print("LLM response:", response["choices"][0]["message"]["content"])

    except NoveumGuardBlocked as e:
        # Handle the block gracefully — do NOT retry automatically.
        print(f"[GuardBlocked] Policy '{e.policy_name}' blocked the call: {e.reason}")
        # Optionally surface a user-friendly error to the CrewAI task output.
        raise


# ---------------------------------------------------------------------------
# Approach 3: Attaching a policy at runtime (after init)
# ---------------------------------------------------------------------------


def attach_policy_at_runtime() -> None:
    """Attach a new policy after init() without restarting the SDK.

    Useful when you want to tighten limits mid-run (e.g. after a budget alert).
    The poller will force_refresh() immediately so the new policy takes effect
    on the very next call.
    """
    from noveum_trace.guard import attach_policy

    new_cap = CostCapPolicy(
        max_usd=2.00,  # tighter cap
        mode=EnforcementMode.non_strict,
        project_id="crewai-demo",
    )
    attach_policy(new_cap)
    print("Attached tighter cost cap — all subsequent calls are now limited to $2.00")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Approach 1: Transport-level (requires crewai installed) ===")
    run_crew_transport_approach()

    print("\n=== Approach 2: Interceptor-level (manual) ===")
    run_crew_interceptor_approach()

    print("\n=== Approach 3: Runtime policy attachment ===")
    attach_policy_at_runtime()

    noveum_trace.shutdown()
