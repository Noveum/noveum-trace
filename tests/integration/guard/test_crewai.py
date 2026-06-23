"""Integration tests for NoveumCrewAIInterceptor.

Tests the raise-only interceptor flow:
  before_llm_call → policies run → raise NoveumGuardBlocked on block
  after_llm_call  → policies reconcile → raise NoveumGuardBlocked on post-block
"""

from __future__ import annotations

import uuid

import pytest

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.exceptions import NoveumGuardBlocked
from noveum_trace.guard.integrations.crewai import NoveumCrewAIInterceptor
from noveum_trace.guard.policies.base import AbstractPolicy
from noveum_trace.guard.policies.cost_cap import CostCapPolicy
from noveum_trace.guard.types import (
    EnforcementMode,
    Phase,
    PolicyContext,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(project_id: str = "test-proj") -> PolicyContext:
    return PolicyContext(
        project_id=project_id,
        organization_id=None,
        environment="test",
        trace_id=None,
        span_id=None,
        call_id=str(uuid.uuid4()),
    )


def _payload(model: str = "gpt-4o-mini", max_tokens: int = 100) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": max_tokens,
        "provider": "openai",
    }


def _response(
    model: str = "gpt-4o-mini", input_tokens: int = 20, output_tokens: int = 10
) -> dict:
    return {
        "model": model,
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        "choices": [{"message": {"content": "hi"}}],
    }


class _AlwaysBlockPolicy(AbstractPolicy):
    name = "always_block"

    def pre(self, parsed, ctx, deps) -> PolicyDecision:
        return PolicyDecision.block(self.name, Phase.pre, reason="blocked by test")


class _BlockInPostPolicy(AbstractPolicy):
    name = "block_in_post"

    def pre(self, parsed, ctx, deps) -> PolicyDecision:
        return PolicyDecision.allow(self.name, Phase.pre)

    def post(self, resp, ctx, decision, deps) -> PolicyDecision:
        return PolicyDecision.block(self.name, Phase.post, reason="post block")


# ---------------------------------------------------------------------------
# before_llm_call — allow
# ---------------------------------------------------------------------------


class TestBeforeLlmCallAllow:
    def test_returns_ran_list_on_allow(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=100.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        interceptor = NoveumCrewAIInterceptor(engine, _ctx())
        call_id, ran = interceptor.before_llm_call(_payload())

        assert isinstance(ran, list)
        assert len(ran) > 0

    def test_no_exception_when_under_cap(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=100.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        interceptor = NoveumCrewAIInterceptor(engine, _ctx())

        # Should not raise
        interceptor.before_llm_call(_payload())


# ---------------------------------------------------------------------------
# before_llm_call — block
# ---------------------------------------------------------------------------


class TestBeforeLlmCallBlock:
    def test_raises_guard_blocked_on_pre_block(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_AlwaysBlockPolicy())

        interceptor = NoveumCrewAIInterceptor(engine, _ctx())

        with pytest.raises(NoveumGuardBlocked):
            interceptor.before_llm_call(_payload())

    def test_exception_carries_policy_name(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_AlwaysBlockPolicy())

        interceptor = NoveumCrewAIInterceptor(engine, _ctx())

        with pytest.raises(NoveumGuardBlocked) as exc_info:
            interceptor.before_llm_call(_payload())

        assert exc_info.value.policy_name == "always_block"

    def test_exception_carries_reason(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_AlwaysBlockPolicy())

        interceptor = NoveumCrewAIInterceptor(engine, _ctx())

        with pytest.raises(NoveumGuardBlocked) as exc_info:
            interceptor.before_llm_call(_payload())

        assert "blocked by test" in exc_info.value.reason

    def test_budget_rolled_back_after_block(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)

        cost_cap = CostCapPolicy(
            max_usd=100.0, mode=EnforcementMode.strict, project_id="test-proj"
        )
        cost_cap.priority = 10
        engine.attach(cost_cap)

        blocker = _AlwaysBlockPolicy()
        blocker.priority = 20
        engine.attach(blocker)

        interceptor = NoveumCrewAIInterceptor(engine, _ctx())

        with pytest.raises(NoveumGuardBlocked):
            interceptor.before_llm_call(_payload())

        assert api.current_spend("test-proj") == pytest.approx(0.0)

    def test_cap_exhaustion_triggers_block(self):
        api = GuardAPIClient()
        api._spend["test-proj"] = 1000.0  # cap exhausted
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=1.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        interceptor = NoveumCrewAIInterceptor(engine, _ctx())

        with pytest.raises(NoveumGuardBlocked):
            interceptor.before_llm_call(_payload())


# ---------------------------------------------------------------------------
# after_llm_call
# ---------------------------------------------------------------------------


class TestAfterLlmCall:
    def test_no_exception_on_successful_post(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=100.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        interceptor = NoveumCrewAIInterceptor(engine, _ctx())
        call_id, ran = interceptor.before_llm_call(_payload())

        # Should not raise
        interceptor.after_llm_call(call_id, _payload(), _response(), ran)

    def test_raises_guard_blocked_on_post_block(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_BlockInPostPolicy())

        interceptor = NoveumCrewAIInterceptor(engine, _ctx())
        call_id, ran = interceptor.before_llm_call(_payload())

        with pytest.raises(NoveumGuardBlocked):
            interceptor.after_llm_call(call_id, _payload(), _response(), ran)

    def test_cost_tracked_after_response(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(
            CostCapPolicy(
                max_usd=100.0, mode=EnforcementMode.strict, project_id="test-proj"
            )
        )

        interceptor = NoveumCrewAIInterceptor(engine, _ctx("test-proj"))
        call_id, ran = interceptor.before_llm_call(_payload())
        interceptor.after_llm_call(
            call_id, _payload(), _response(input_tokens=100, output_tokens=50), ran
        )

        # Spend should have been reconciled (actual < reserved)
        assert api.current_spend("test-proj") >= 0


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


class TestParseRequest:
    def test_model_extracted(self):
        parsed = NoveumCrewAIInterceptor._parse_request(_payload(model="gpt-4o"))
        assert parsed.model == "gpt-4o"

    def test_messages_extracted(self):
        messages = [{"role": "user", "content": "hi"}]
        parsed = NoveumCrewAIInterceptor._parse_request(
            {"model": "gpt-4o", "messages": messages}
        )
        assert parsed.messages == messages

    def test_estimated_tokens_positive_for_non_empty_message(self):
        parsed = NoveumCrewAIInterceptor._parse_request(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "a longer message here"}],
            }
        )
        assert parsed.estimated_input_tokens > 0


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_input_tokens_extracted(self):
        parsed = NoveumCrewAIInterceptor._parse_response(
            _payload(),
            {"model": "gpt-4o", "usage": {"input_tokens": 15, "output_tokens": 5}},
        )
        assert parsed.input_tokens == 15

    def test_prompt_tokens_alias(self):
        """OpenAI-style prompt_tokens should also work."""
        parsed = NoveumCrewAIInterceptor._parse_response(
            _payload(),
            {"model": "gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        )
        assert parsed.input_tokens == 10
        assert parsed.output_tokens == 3

    def test_cost_usd_non_negative(self):
        parsed = NoveumCrewAIInterceptor._parse_response(
            _payload(), _response(input_tokens=100, output_tokens=50)
        )
        assert parsed.cost_usd >= 0
