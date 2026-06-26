"""Real-call ("real time") integration tests for NovaGuard.

Unlike the other files under tests/integration/guard/, these exercise the full
Guard stack against the *live* provider API — no mock inner transport. They
prove the transport, cost reconciliation, and atomic spend reserve behave under
real network latency and real usage numbers, not just deterministic mocks.

Every test skips unless a valid ``ANTHROPIC_API_KEY`` is present, so the suite is a
no-op until keys are configured. Run explicitly with:

    pytest tests/integration/guard/test_real_calls.py -m integration -v

Covered plan steps:
  Step 1 — sync call through noveum_trace.guard.http_client()
  Step 6 — async call through noveum_trace.guard.async_http_client()
           (exercises NoveumAsyncTransport + response.aread())
  Step 7 — concurrency under a real cap: 20 threads, tight max_usd, real calls.
           Expect a mix of allowed/blocked outcomes and current_spend <= cap.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import pytest

# Load .env so locally-exported provider keys are picked up automatically.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv optional
    pass

import noveum_trace
from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.policies.cost_cap import CostCapPolicy
from noveum_trace.guard.transport.adapters.openai_adapter import OpenAIAdapter
from noveum_trace.guard.types import EnforcementMode, PolicyContext

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:  # provider SDK optional
    anthropic = None  # type: ignore[assignment]
    ANTHROPIC_AVAILABLE = False

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = os.environ.get("NOVEUM_GUARD_TEST_MODEL", "claude-haiku-4-5-20251001")


def _is_valid_key(key: Optional[str]) -> bool:
    invalid = {"", "your-anthropic-api-key-here", "test-key", "sk-test", "sk-fake"}
    return bool(key) and key not in invalid and len(key) > 10


def _should_test_anthropic() -> bool:
    return ANTHROPIC_AVAILABLE and _is_valid_key(ANTHROPIC_API_KEY)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _should_test_anthropic(),
        reason="ANTHROPIC_API_KEY not set/valid or anthropic not installed",
    ),
]


# ---------------------------------------------------------------------------
# Results Storage
# ---------------------------------------------------------------------------


class ResultsCollector:
    """Collects test results with detailed metrics and writes them to JSON."""

    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: list[dict] = []
        self.session_id = datetime.now().isoformat()

    def record(
        self,
        test_name: str,
        status: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        spend_usd: float = 0.0,
        reserved_usd: Optional[float] = None,
        actual_usd: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        reserve_ms: Optional[float] = None,
        provider_call_ms: Optional[float] = None,
        reconcile_ms: Optional[float] = None,
        total_ms: Optional[float] = None,
        block_reason: Optional[str] = None,
        current_spend: Optional[float] = None,
        cap_usd: Optional[float] = None,
        inflight_count: int = 0,
        peak_inflight: Optional[int] = None,
        threads: Optional[int] = None,
        admitted: Optional[int] = None,
        blocked: Optional[int] = None,
        details: Optional[dict] = None,
    ) -> None:
        """Record a test result with comprehensive metrics."""
        # Build metrics object
        metrics = {}

        # 1. Provider/model info
        if provider:
            metrics["provider"] = provider
        if model:
            metrics["model"] = model

        # 2. Cost tracking
        metrics["spend_usd"] = spend_usd
        if reserved_usd is not None:
            metrics["reserved_usd"] = reserved_usd
            if actual_usd is not None:
                metrics["actual_usd"] = actual_usd
                metrics["released_usd"] = max(0, reserved_usd - actual_usd)

        # 3. Token usage
        if input_tokens is not None or output_tokens is not None:
            tokens = {}
            if input_tokens is not None:
                tokens["input_tokens"] = input_tokens
            if output_tokens is not None:
                tokens["output_tokens"] = output_tokens
            if input_tokens is not None and output_tokens is not None:
                tokens["total_tokens"] = input_tokens + output_tokens
            metrics["tokens"] = tokens

        # 4. Timing metrics
        if any(t is not None for t in [reserve_ms, provider_call_ms, reconcile_ms]):
            timing = {}
            if reserve_ms is not None:
                timing["reserve_ms"] = round(reserve_ms, 2)
            if provider_call_ms is not None:
                timing["provider_call_ms"] = round(provider_call_ms, 2)
            if reconcile_ms is not None:
                timing["reconcile_ms"] = round(reconcile_ms, 2)
            if total_ms is not None:
                timing["total_ms"] = round(total_ms, 2)
            metrics["timing"] = timing

        # 5. Block details
        if block_reason:
            metrics["block"] = {
                "reason": block_reason,
            }
            if current_spend is not None:
                metrics["block"]["current_spend"] = current_spend
            if cap_usd is not None:
                metrics["block"]["cap_usd"] = cap_usd

        # 6. Concurrency details
        if threads is not None:
            metrics["concurrency"] = {
                "threads": threads,
            }
            if admitted is not None:
                metrics["concurrency"]["admitted"] = admitted
            if blocked is not None:
                metrics["concurrency"]["blocked"] = blocked
            if peak_inflight is not None:
                metrics["concurrency"]["peak_inflight"] = peak_inflight

        # Add inflight count
        metrics["inflight_count"] = inflight_count

        # Build result record
        result = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "status": status,
            "metrics": metrics,
        }

        # Add extra details if provided
        if details:
            result["details"] = details

        self.results.append(result)

    def write(self) -> None:
        """Write all results to a JSON file."""
        output_file = (
            self.output_dir / f"results_{self.session_id.replace(':', '-')}.json"
        )
        data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r["status"] == "passed"),
            "failed": sum(1 for r in self.results if r["status"] == "failed"),
            "results": self.results,
        }
        output_file.write_text(json.dumps(data, indent=2))
        print(f"\nTest results saved to: {output_file}")


_results_collector = ResultsCollector()


@pytest.fixture(autouse=True, scope="module")
def _auto_write_results():
    """Auto-write results after all tests in this module complete."""
    yield
    _results_collector.write()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(project_id: str) -> PolicyContext:
    return PolicyContext(
        project_id=project_id,
        organization_id=None,
        environment="real-test",
        trace_id=None,
        span_id=None,
        call_id=str(uuid.uuid4()),
    )


def _build_stack(
    project_id: str, max_usd: float
) -> tuple[PolicyEngine, PolicyContext, GuardAPIClient]:
    """A real-call Guard stack: fresh in-memory api + a single CostCapPolicy."""
    api = GuardAPIClient()
    engine = PolicyEngine(api_client=api)
    engine.attach(
        CostCapPolicy(
            max_usd=max_usd, mode=EnforcementMode.strict, project_id=project_id
        )
    )
    return engine, _ctx(project_id), api


def _messages() -> list[dict]:
    return [{"role": "user", "content": "Reply with one word: done."}]


def _estimated_reserved_usd(max_tokens: int) -> float:
    """Worst-case reservation for one call — the same value the policy reserves.

    Used to size a cap that guarantees a partial-admit mix under concurrency.
    """
    payload = json.dumps(
        {"model": MODEL, "messages": _messages(), "max_tokens": max_tokens}
    ).encode()
    req = httpx.Request(
        "POST", "https://api.openai.com/v1/chat/completions", content=payload
    )
    parsed = OpenAIAdapter().parse_request(req)
    probe = CostCapPolicy(max_usd=1.0, project_id="probe")
    return probe._estimate_reserved_usd(parsed)


# ---------------------------------------------------------------------------
# Step 1 — sync call through the Guard transport
# ---------------------------------------------------------------------------


class TestStep1Sync:
    def test_real_call_allowed_and_spend_reconciled(self):
        project_id = "guard-real-sync"
        # Generous cap so the call is admitted.
        engine, ctx, api = _build_stack(project_id, max_usd=1.0)

        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )

        start_total = time.time()
        resp = client.messages.create(model=MODEL, messages=_messages(), max_tokens=16)
        provider_call_ms = (time.time() - start_total) * 1000

        assert resp.content[0].text  # real response came back
        spend = api.current_spend(project_id)
        # Spend is positive (a real call happened) and reconciled below the cap.
        assert spend > 0.0
        assert spend <= 1.0
        # After reconcile no reservation should remain inflight.
        assert api.inflight_count() == 0

        reserved = _estimated_reserved_usd(16)
        _results_collector.record(
            "test_real_call_allowed_and_spend_reconciled",
            "passed",
            provider=resp.model.split("-")[0] if resp.model else "anthropic",
            model=resp.model or MODEL,
            spend_usd=spend,
            reserved_usd=reserved,
            actual_usd=spend,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            provider_call_ms=provider_call_ms,
            inflight_count=api.inflight_count(),
            details={"response": resp.content[0].text[:50]},
        )

    def test_real_call_blocked_when_cap_exhausted(self):
        project_id = "guard-real-sync-blocked"
        cap = 1.0
        engine, ctx, api = _build_stack(project_id, max_usd=cap)
        # Pre-exhaust the cap so the pre() reserve is rejected before any network I/O.
        api._spend[project_id] = 1.0

        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )

        with pytest.raises(anthropic.PermissionDeniedError):
            client.messages.create(model=MODEL, messages=_messages(), max_tokens=16)

        # A pre-block reserves nothing, so spend is unchanged.
        assert api.current_spend(project_id) == pytest.approx(cap)

        reserved = _estimated_reserved_usd(16)
        _results_collector.record(
            "test_real_call_blocked_when_cap_exhausted",
            "passed",
            provider="anthropic",
            model=MODEL,
            spend_usd=api.current_spend(project_id),
            block_reason="cap_exhausted",
            current_spend=api.current_spend(project_id),
            cap_usd=cap,
            reserved_usd=reserved,
            details={"guard_action": "blocked_at_pre"},
        )


# ---------------------------------------------------------------------------
# Step 6 — async call (NoveumAsyncTransport + aread())
# ---------------------------------------------------------------------------


class TestStep6Async:
    async def test_real_async_call_allowed_and_spend_reconciled(self):
        project_id = "guard-real-async"
        engine, ctx, api = _build_stack(project_id, max_usd=1.0)

        client = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.async_http_client(engine, ctx),
        )
        try:
            start_total = time.time()
            resp = await client.messages.create(
                model=MODEL, messages=_messages(), max_tokens=16
            )
            provider_call_ms = (time.time() - start_total) * 1000
        finally:
            await client.close()

        assert resp.content[0].text
        spend = api.current_spend(project_id)
        assert spend > 0.0
        assert spend <= 1.0
        assert api.inflight_count() == 0

        reserved = _estimated_reserved_usd(16)
        _results_collector.record(
            "test_real_async_call_allowed_and_spend_reconciled",
            "passed",
            provider=resp.model.split("-")[0] if resp.model else "anthropic",
            model=resp.model or MODEL,
            spend_usd=spend,
            reserved_usd=reserved,
            actual_usd=spend,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            provider_call_ms=provider_call_ms,
            inflight_count=api.inflight_count(),
            details={"response": resp.content[0].text[:50]},
        )

    async def test_real_async_call_blocked_when_cap_exhausted(self):
        project_id = "guard-real-async-blocked"
        cap = 1.0
        engine, ctx, api = _build_stack(project_id, max_usd=cap)
        api._spend[project_id] = 1.0

        client = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.async_http_client(engine, ctx),
        )
        try:
            with pytest.raises(anthropic.PermissionDeniedError):
                await client.messages.create(
                    model=MODEL, messages=_messages(), max_tokens=16
                )
        finally:
            await client.close()

        assert api.current_spend(project_id) == pytest.approx(cap)

        reserved = _estimated_reserved_usd(16)
        _results_collector.record(
            "test_real_async_call_blocked_when_cap_exhausted",
            "passed",
            provider="anthropic",
            model=MODEL,
            spend_usd=api.current_spend(project_id),
            block_reason="cap_exhausted",
            current_spend=api.current_spend(project_id),
            cap_usd=cap,
            reserved_usd=reserved,
            details={"guard_action": "blocked_at_pre"},
        )


# ---------------------------------------------------------------------------
# Step 7 — concurrency under a real cap
# ---------------------------------------------------------------------------


class TestStep7ConcurrencyRealCap:
    def test_twenty_threads_real_calls_respect_cap(self):
        project_id = "guard-real-concurrency"
        n_threads = 20
        max_tokens = 16
        target_admit = 8  # aim for a mix: ~8 admitted, ~12 blocked

        # Size the cap off the actual worst-case reservation so the mix is
        # deterministic regardless of the model's price.
        reserved = _estimated_reserved_usd(max_tokens)
        cap = reserved * (target_admit + 0.5)

        engine, ctx, api = _build_stack(project_id, max_usd=cap)

        outcomes: list[int] = []
        errors: list[Exception] = []
        lock = threading.Lock()
        peak_inflight = [0]  # track max inflight during execution

        def _call() -> None:
            # Each thread gets its own client; they share the engine/api (and cap).
            client = anthropic.Anthropic(
                api_key=ANTHROPIC_API_KEY,
                http_client=noveum_trace.guard.http_client(engine, ctx),
            )
            try:
                client.messages.create(
                    model=MODEL, messages=_messages(), max_tokens=max_tokens
                )
                with lock:
                    outcomes.append(200)
                    # Track peak inflight
                    peak_inflight[0] = max(peak_inflight[0], api.inflight_count())
            except anthropic.PermissionDeniedError:
                with lock:
                    outcomes.append(403)
                    peak_inflight[0] = max(peak_inflight[0], api.inflight_count())
            except Exception as exc:  # surface unexpected failures
                with lock:
                    errors.append(exc)
            finally:
                client.close()

        start_concurrent = time.time()
        threads = [threading.Thread(target=_call) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_concurrent_ms = (time.time() - start_concurrent) * 1000

        assert errors == [], f"unexpected errors: {errors}"
        assert len(outcomes) == n_threads
        # Only allow/block — nothing leaked through as another status.
        assert set(outcomes) <= {200, 403}
        # The atomic reserve must produce a real mix under live latency.
        assert 200 in outcomes, "expected some calls to be admitted"
        assert 403 in outcomes, "expected the cap to block some calls"
        # Cap is never exceeded, even with concurrent reserves against real calls.
        assert api.current_spend(project_id) <= cap + 1e-9
        # All reservations reconciled — nothing stuck inflight.
        assert api.inflight_count() == 0

        admitted = sum(1 for o in outcomes if o == 200)
        blocked = sum(1 for o in outcomes if o == 403)
        _results_collector.record(
            "test_twenty_threads_real_calls_respect_cap",
            "passed",
            provider="anthropic",
            model=MODEL,
            spend_usd=api.current_spend(project_id),
            threads=n_threads,
            admitted=admitted,
            blocked=blocked,
            cap_usd=cap,
            peak_inflight=peak_inflight[0],
            provider_call_ms=total_concurrent_ms,
            inflight_count=api.inflight_count(),
            details={
                "concurrency_test": True,
            },
        )


# ---------------------------------------------------------------------------
# Streaming — Guard passes through, no cost reconciliation
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_streaming_request_passes_through(self):
        """Streaming requests are reconciled when the stream is fully consumed.

        Guard wraps the response stream in _SyncStreamReconciler, which buffers
        every SSE chunk. When the stream is exhausted, it parses Anthropic's
        message_start / message_delta events to extract actual token usage and
        calls post_call() — so inflight_count returns to 0 and spend reflects
        real tokens, not the worst-case reservation.
        """
        project_id = "guard-streaming"
        engine, ctx, api = _build_stack(project_id, max_usd=1.0)

        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.http_client(engine, ctx),
        )

        start_total = time.time()
        with client.messages.stream(
            model=MODEL, messages=_messages(), max_tokens=16
        ) as stream:
            full_text = ""
            for text in stream.text_stream:
                full_text += text
            # get_final_message() is available after the text stream is exhausted
            final_msg = stream.get_final_message()
            sdk_input = final_msg.usage.input_tokens
            sdk_output = final_msg.usage.output_tokens
        provider_call_ms = (time.time() - start_total) * 1000

        assert full_text  # real response came back
        spend = api.current_spend(project_id)
        assert spend > 0.0
        assert spend <= 1.0
        # After stream is fully consumed the reconciler must have fired
        assert api.inflight_count() == 0

        reserved = _estimated_reserved_usd(16)
        _results_collector.record(
            "test_streaming_request_passes_through",
            "passed",
            provider="anthropic",
            model=MODEL,
            spend_usd=spend,
            reserved_usd=reserved,
            actual_usd=spend,
            input_tokens=sdk_input,
            output_tokens=sdk_output,
            provider_call_ms=provider_call_ms,
            inflight_count=api.inflight_count(),
            details={
                "response": full_text[:50],
                "reconciliation": "sse_parse",
            },
        )

    async def test_async_streaming_request_passes_through(self):
        """Async streaming also bypasses cost reconciliation."""
        project_id = "guard-async-streaming"
        engine, ctx, api = _build_stack(project_id, max_usd=1.0)

        client = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=noveum_trace.guard.async_http_client(engine, ctx),
        )

        try:
            start_total = time.time()
            full_text = ""
            async with client.messages.stream(
                model=MODEL, messages=_messages(), max_tokens=16
            ) as stream:
                async for text in stream.text_stream:
                    full_text += text
                # get_final_message() is available after the text stream is exhausted
                final_msg = await stream.get_final_message()
                sdk_input = final_msg.usage.input_tokens
                sdk_output = final_msg.usage.output_tokens
            provider_call_ms = (time.time() - start_total) * 1000

            assert full_text  # real response came back
            spend = api.current_spend(project_id)
            assert spend > 0.0
            assert spend <= 1.0
            # After stream is fully consumed the reconciler must have fired
            assert api.inflight_count() == 0

            reserved = _estimated_reserved_usd(16)
            _results_collector.record(
                "test_async_streaming_request_passes_through",
                "passed",
                provider="anthropic",
                model=MODEL,
                spend_usd=spend,
                reserved_usd=reserved,
                actual_usd=spend,
                input_tokens=sdk_input,
                output_tokens=sdk_output,
                provider_call_ms=provider_call_ms,
                inflight_count=api.inflight_count(),
                details={
                    "response": full_text[:50],
                    "reconciliation": "sse_parse",
                },
            )
        finally:
            await client.close()
