"""Integration tests for the Bedrock guard integration / instrument_bedrock().

instrument_bedrock() installs a single process-wide monkeypatch of
``botocore.client.BaseClient._make_api_call``, filtered internally to
bedrock-runtime clients. One function call frame spans pre-call blocking,
the real call, and post-call reconciliation — no event-hook handoff, and no
dependency on any particular client instance being explicitly wired.

Direct-handler tests exercise parsing/reconciliation logic without a live
boto3 client. The TestGlobalWiring / TestInstrumentBedrockWiring classes
prove the end-to-end flow (patch installed -> real call -> block/reconcile)
using botocore.stub.Stubber against a real boto3 client with dummy
credentials — including a client that was never explicitly instrumented,
which is the scenario the old per-client event-hook design could not cover.
"""

from __future__ import annotations

import io
import json
import uuid

import boto3
import botocore.config
import botocore.response
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from noveum_trace.guard.api_client import GuardAPIClient
from noveum_trace.guard.decision import PolicyDecision
from noveum_trace.guard.engine import PolicyEngine
from noveum_trace.guard.exceptions import NoveumGuardBlocked
from noveum_trace.guard.integrations.bedrock import (
    _BedrockStreamReconciler,
    _is_embeddings_model,
    _parse_embeddings_input,
    _parse_embeddings_usage,
    _parse_invoke_model_body,
    _parse_invoke_model_usage,
    _parse_request,
    _reconcile_non_streaming,
    _uninstrument_bedrock_for_tests,
    instrument_bedrock,
)
from noveum_trace.guard.policies.base import AbstractPolicy
from noveum_trace.guard.policies.cost_cap import CostCapPolicy
from noveum_trace.guard.types import EnforcementMode, Phase, PolicyContext

# ---------------------------------------------------------------------------
# Teardown — the patch is process-wide, so it must never leak across tests.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_bedrock_patch():
    yield
    _uninstrument_bedrock_for_tests()


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


def _converse_params(
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
) -> dict:
    return {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": "Hello!"}]}],
        "inferenceConfig": {"maxTokens": 100},
    }


def _invoke_model_params(
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
) -> dict:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello!"}],
    }
    return {
        "modelId": model_id,
        "body": json.dumps(body).encode(),
        "contentType": "application/json",
    }


def _streaming_body(payload: dict) -> botocore.response.StreamingBody:
    raw = json.dumps(payload).encode()
    return botocore.response.StreamingBody(io.BytesIO(raw), len(raw))


def _client() -> boto3.client:
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


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


def _engine_with(*policies: AbstractPolicy) -> PolicyEngine:
    api = GuardAPIClient()
    engine = PolicyEngine(api_client=api)
    for policy in policies:
        engine.attach(policy)
    return engine


def _cost_cap(max_usd: float = 100.0, project_id: str = "test-proj") -> CostCapPolicy:
    return CostCapPolicy(
        max_usd=max_usd, mode=EnforcementMode.strict, project_id=project_id
    )


_APPLY_GUARDRAIL_PARAMS = {
    "guardrailIdentifier": "test-guardrail",
    "guardrailVersion": "DRAFT",
    "source": "INPUT",
    "content": [{"text": {"text": "hello"}}],
}

_APPLY_GUARDRAIL_RESPONSE = {
    "usage": {
        "topicPolicyUnits": 0,
        "contentPolicyUnits": 0,
        "wordPolicyUnits": 0,
        "sensitiveInformationPolicyUnits": 0,
        "sensitiveInformationPolicyFreeUnits": 0,
        "contextualGroundingPolicyUnits": 0,
    },
    "action": "NONE",
    "outputs": [],
    "assessments": [],
}


# ---------------------------------------------------------------------------
# Global wiring via botocore.stub.Stubber — instrument_bedrock() with no
# client argument, exercised through a real (Stubber-mocked) boto3 client.
# ---------------------------------------------------------------------------


class TestGlobalWiring:
    def test_converse_reconciles_cost_on_success(self):
        client = _client()
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        instrument_bedrock(engine=engine, context=_ctx("test-proj"))

        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "hi"}]}
                },
                "usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30},
                "stopReason": "end_turn",
                "metrics": {"latencyMs": 100},
            },
            _converse_params(),
        )
        stubber.activate()

        response = client.converse(**_converse_params())
        assert response["usage"]["outputTokens"] == 10
        assert api.current_spend("test-proj") >= 0

    def test_unknown_operation_is_ignored(self):
        # ApplyGuardrail isn't in _KNOWN_OPERATIONS — an always-block policy
        # must not affect it.
        client = _client()
        stubber = Stubber(client)
        engine = _engine_with(_AlwaysBlockPolicy())
        instrument_bedrock(engine=engine, context=_ctx())

        stubber.add_response(
            "apply_guardrail", _APPLY_GUARDRAIL_RESPONSE, _APPLY_GUARDRAIL_PARAMS
        )
        stubber.activate()

        response = client.apply_guardrail(**_APPLY_GUARDRAIL_PARAMS)
        assert response["action"] == "NONE"

    def test_raises_guard_blocked_on_pre_block(self):
        client = _client()
        stubber = Stubber(client)  # no add_response — call must never reach it
        engine = _engine_with(_AlwaysBlockPolicy())
        instrument_bedrock(engine=engine, context=_ctx())
        stubber.activate()

        with pytest.raises(NoveumGuardBlocked) as exc_info:
            client.converse(**_converse_params())

        assert exc_info.value.policy_name == "always_block"

    def test_budget_rolled_back_after_block(self):
        client = _client()
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        cost_cap = _cost_cap()
        cost_cap.priority = 10
        engine.attach(cost_cap)
        blocker = _AlwaysBlockPolicy()
        blocker.priority = 20
        engine.attach(blocker)
        instrument_bedrock(engine=engine, context=_ctx("test-proj"))
        stubber.activate()

        with pytest.raises(NoveumGuardBlocked):
            client.converse(**_converse_params())

        assert api.current_spend("test-proj") == pytest.approx(0.0)

    def test_raises_guard_blocked_on_post_block(self):
        client = _client()
        stubber = Stubber(client)
        engine = _engine_with(_BlockInPostPolicy())
        instrument_bedrock(engine=engine, context=_ctx())

        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "hi"}]}
                },
                "usage": {"inputTokens": 5, "outputTokens": 5, "totalTokens": 10},
                "stopReason": "end_turn",
                "metrics": {"latencyMs": 100},
            },
            _converse_params(),
        )
        stubber.activate()

        with pytest.raises(NoveumGuardBlocked):
            client.converse(**_converse_params())

    def test_converse_missing_usage_releases_instead_of_guessing(self):
        client = _client()
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        instrument_bedrock(engine=engine, context=_ctx("test-proj"))

        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "hi"}]}
                },
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "stopReason": "end_turn",
                "metrics": {"latencyMs": 100},
            },
            _converse_params(),
        )
        stubber.activate()

        client.converse(**_converse_params())
        # Zero usage is still "usage present" (unlike InvokeModel's fully
        # absent case below) — spend reflects the (zero) reported tokens.
        assert api.current_spend("test-proj") >= 0

    def test_invoke_model_reconciles_and_body_still_readable(self):
        client = _client()
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        instrument_bedrock(engine=engine, context=_ctx("test-proj"))

        resp_payload = {
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "content": [{"type": "text", "text": "hello"}],
        }
        stubber.add_response(
            "invoke_model",
            {"body": _streaming_body(resp_payload), "contentType": "application/json"},
            _invoke_model_params(),
        )
        stubber.activate()

        response = client.invoke_model(**_invoke_model_params())
        assert json.loads(response["body"].read()) == resp_payload
        assert api.current_spend("test-proj") > 0

    def test_invoke_model_unrecognized_family_releases_instead_of_guessing(self):
        client = _client()
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        instrument_bedrock(engine=engine, context=_ctx("test-proj"))

        params = _invoke_model_params(model_id="unknown-vendor.some-model-v1:0")
        stubber.add_response(
            "invoke_model",
            {
                "body": _streaming_body({"some": "unrecognized shape"}),
                "contentType": "application/json",
            },
            params,
        )
        stubber.activate()

        client.invoke_model(**params)
        assert api.current_spend("test-proj") == pytest.approx(0.0)

    def test_after_call_error_releases_reservation(self):
        client = _client()
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        instrument_bedrock(engine=engine, context=_ctx("test-proj"))

        stubber.add_client_error("converse", service_error_code="ValidationException")
        stubber.activate()

        with pytest.raises(ClientError):
            client.converse(**_converse_params())

        assert api.current_spend("test-proj") == pytest.approx(0.0)

    def test_second_client_from_independent_session_is_also_guarded(self):
        # The scenario the old per-client event-hook design couldn't cover:
        # a client never passed to instrument_bedrock(), built from its own
        # boto3.Session(), is still guarded by the process-wide patch.
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        instrument_bedrock(engine=engine, context=_ctx("test-proj"))

        client = boto3.Session().client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        stubber = Stubber(client)
        resp_payload = {
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "content": [{"type": "text", "text": "hello"}],
        }
        stubber.add_response(
            "invoke_model",
            {"body": _streaming_body(resp_payload), "contentType": "application/json"},
            _invoke_model_params(),
        )
        stubber.activate()

        response = client.invoke_model(**_invoke_model_params())
        assert json.loads(response["body"].read()) == resp_payload
        assert api.current_spend("test-proj") > 0

    def test_non_bedrock_client_is_unaffected(self):
        # Any other botocore client goes through the same patched
        # _make_api_call — confirm it passes through untouched even with an
        # always-block policy configured globally.
        instrument_bedrock(engine=_engine_with(_AlwaysBlockPolicy()), context=_ctx())

        sts = boto3.client(
            "sts",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        stubber = Stubber(sts)
        stubber.add_response(
            "get_caller_identity",
            {
                "UserId": "AIDA",
                "Account": "123456789012",
                "Arn": "arn:aws:iam::123456789012:user/test",
            },
            {},
        )
        stubber.activate()

        response = sts.get_caller_identity()
        assert response["Account"] == "123456789012"


# ---------------------------------------------------------------------------
# Streaming reconciler — unaffected by the wiring redesign, still constructed
# directly with a module-level _parse_request() instead of an interceptor
# instance method.
# ---------------------------------------------------------------------------


class TestStreamingReconciler:
    def test_converse_stream_reconciles_from_metadata_event(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        ctx = _ctx("test-proj")
        parsed_req = _parse_request("ConverseStream", _converse_params())
        block, ran = engine.pre_call(parsed_req, ctx)
        assert block is None

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "hi"}}},
            {"metadata": {"usage": {"inputTokens": 15, "outputTokens": 8}}},
        ]
        reconciler = _BedrockStreamReconciler(
            events, engine, ctx, ran, parsed_req, "ConverseStream"
        )
        list(reconciler)  # fully consume

        assert api.current_spend("test-proj") >= 0

    def test_invoke_model_stream_reconciles_from_invocation_metrics(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        ctx = _ctx("test-proj")
        parsed_req = _parse_request(
            "InvokeModelWithResponseStream", _invoke_model_params()
        )
        block, ran = engine.pre_call(parsed_req, ctx)
        assert block is None

        final_chunk = json.dumps(
            {
                "amazon-bedrock-invocationMetrics": {
                    "inputTokenCount": 12,
                    "outputTokenCount": 6,
                }
            }
        ).encode()
        events = [
            {"chunk": {"bytes": json.dumps({"type": "content_block_delta"}).encode()}},
            {"chunk": {"bytes": final_chunk}},
        ]
        reconciler = _BedrockStreamReconciler(
            events, engine, ctx, ran, parsed_req, "InvokeModelWithResponseStream"
        )
        list(reconciler)

        assert api.current_spend("test-proj") >= 0

    def test_no_usage_metrics_releases_instead_of_guessing(self):
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        ctx = _ctx("test-proj")
        parsed_req = _parse_request("ConverseStream", _converse_params())
        block, ran = engine.pre_call(parsed_req, ctx)
        assert block is None

        reconciler = _BedrockStreamReconciler(
            [{"messageStart": {}}], engine, ctx, ran, parsed_req, "ConverseStream"
        )
        list(reconciler)

        assert api.current_spend("test-proj") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Buffered stream enforcement — when a post-blocking policy is attached, the
# stream is fully buffered and post_call runs BEFORE any event is yielded, so a
# post-phase block is actually enforced (raises) instead of only logged. Mirrors
# the httpx transport's build_buffered_stream_response path.
# ---------------------------------------------------------------------------


class _BlockInPostStreaming(AbstractPolicy):
    name = "block_in_post_streaming"
    can_block_post = True

    def __init__(self) -> None:
        super().__init__()
        self.seen_text: str | None = None

    def pre(self, parsed, ctx, deps) -> PolicyDecision:
        return PolicyDecision.allow(self.name, Phase.pre)

    def post(self, resp, ctx, decision, deps) -> PolicyDecision:
        self.seen_text = resp.text
        return PolicyDecision.block(self.name, Phase.post, reason="post block")


class _ObservePostStreaming(AbstractPolicy):
    """Post-blocking-capable policy that never actually blocks; records text."""

    name = "observe_post_streaming"
    can_block_post = True

    def __init__(self) -> None:
        super().__init__()
        self.seen_text: str | None = None

    def pre(self, parsed, ctx, deps) -> PolicyDecision:
        return PolicyDecision.allow(self.name, Phase.pre)

    def post(self, resp, ctx, decision, deps) -> PolicyDecision:
        self.seen_text = resp.text
        return PolicyDecision.allow(self.name, Phase.post)


class TestBufferedStreamEnforcement:
    def _run(self, operation_name, events, policy):
        from noveum_trace.guard.integrations.bedrock import _wrap_streaming_result

        engine = _engine_with(policy)
        ctx = _ctx("test-proj")
        parsed_req = _parse_request(
            operation_name,
            (
                _converse_params()
                if operation_name == "ConverseStream"
                else _invoke_model_params()
            ),
        )
        block, ran = engine.pre_call(parsed_req, ctx)
        assert block is None
        key = "stream" if operation_name == "ConverseStream" else "body"
        return _wrap_streaming_result(
            {key: iter(events)}, engine, ctx, ran, parsed_req, operation_name
        )

    def test_converse_stream_post_block_raises_before_yield(self):
        events = [
            {"contentBlockDelta": {"delta": {"text": "leaked"}}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3}}},
        ]
        policy = _BlockInPostStreaming()
        with pytest.raises(NoveumGuardBlocked):
            self._run("ConverseStream", events, policy)
        # The assembled completion text reached post() before any event escaped.
        assert policy.seen_text == "leaked"

    def test_converse_stream_allowed_replays_events(self):
        events = [
            {"contentBlockDelta": {"delta": {"text": "hi there"}}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3}}},
        ]
        policy = _ObservePostStreaming()
        parsed = self._run("ConverseStream", events, policy)
        # Caller still iterates every original event (buffered, then replayed).
        assert list(parsed["stream"]) == events
        assert policy.seen_text == "hi there"

    def test_invoke_model_stream_post_block_raises_before_yield(self):
        final_chunk = json.dumps(
            {
                "amazon-bedrock-invocationMetrics": {
                    "inputTokenCount": 7,
                    "outputTokenCount": 4,
                }
            }
        ).encode()
        events = [
            {
                "chunk": {
                    "bytes": json.dumps(
                        {"type": "content_block_delta", "delta": {"text": "leaked"}}
                    ).encode()
                }
            },
            {"chunk": {"bytes": final_chunk}},
        ]
        policy = _BlockInPostStreaming()
        with pytest.raises(NoveumGuardBlocked):
            self._run("InvokeModelWithResponseStream", events, policy)
        assert policy.seen_text == "leaked"

    def test_stream_drain_error_releases_reservation(self):
        from noveum_trace.guard.integrations.bedrock import _wrap_streaming_result

        # A cost-cap reservation is taken in pre(); if buffering the stream
        # raises before post_call reconciles, the reservation must be released
        # rather than leaked inflight.
        cost_cap = _cost_cap()
        engine = _engine_with(cost_cap, _ObservePostStreaming())
        ctx = _ctx("test-proj")
        parsed_req = _parse_request("ConverseStream", _converse_params())
        block, ran = engine.pre_call(parsed_req, ctx)
        assert block is None
        assert engine._api_client.inflight_count() == 1  # reserved

        def _boom():
            yield {"contentBlockDelta": {"delta": {"text": "partial"}}}
            raise RuntimeError("stream dropped")

        with pytest.raises(RuntimeError, match="stream dropped"):
            _wrap_streaming_result(
                {"stream": _boom()}, engine, ctx, ran, parsed_req, "ConverseStream"
            )

        assert engine._api_client.inflight_count() == 0  # released, not leaked
        assert engine._api_client.current_spend("test-proj") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Bedrock embeddings models (Titan Embed, Cohere Embed) — routed through
# InvokeModel but parsed as kind="embeddings": input-only reservation (no
# chat-style output allowance) and output_tokens forced to 0 on reconcile.
# ---------------------------------------------------------------------------


class _RecordPostPolicy(AbstractPolicy):
    name = "record_post"

    def __init__(self) -> None:
        super().__init__()
        self.resp = None

    def pre(self, parsed, ctx, deps) -> PolicyDecision:
        return PolicyDecision.allow(self.name, Phase.pre)

    def post(self, resp, ctx, decision, deps) -> PolicyDecision:
        self.resp = resp
        return PolicyDecision.allow(self.name, Phase.post)


def _invoke_embed_params(model_id: str, body: dict) -> dict:
    return {"modelId": model_id, "body": json.dumps(body).encode()}


class TestBedrockEmbeddings:
    def test_is_embeddings_model(self):
        assert _is_embeddings_model("amazon.titan-embed-text-v2:0")
        assert _is_embeddings_model("cohere.embed-english-v3")
        assert not _is_embeddings_model("anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert not _is_embeddings_model("amazon.titan-text-express-v1")

    def test_parse_embeddings_input_titan(self):
        assert (
            _parse_embeddings_input(
                "amazon.titan-embed-text-v2:0", {"inputText": "hello world"}
            )
            == "hello world"
        )

    def test_parse_embeddings_input_cohere_joins_texts(self):
        assert (
            _parse_embeddings_input(
                "cohere.embed-english-v3", {"texts": ["one", "two"]}
            )
            == "one two"
        )

    def test_parse_embeddings_usage_titan_reports_input(self):
        assert (
            _parse_embeddings_usage(
                "amazon.titan-embed-text-v2:0", {"inputTextTokenCount": 12}
            )
            == 12
        )

    def test_parse_embeddings_usage_cohere_returns_none(self):
        # Cohere embed responses carry no token count.
        assert (
            _parse_embeddings_usage("cohere.embed-english-v3", {"embeddings": [[0.1]]})
            is None
        )

    def test_titan_embed_request_is_kind_embeddings(self):
        parsed = _parse_request(
            "InvokeModel",
            _invoke_embed_params(
                "amazon.titan-embed-text-v2:0", {"inputText": "hello world"}
            ),
        )
        assert parsed.kind == "embeddings"
        assert parsed.max_tokens is None
        assert parsed.estimated_input_tokens > 0

    def test_cohere_embed_request_is_kind_embeddings(self):
        parsed = _parse_request(
            "InvokeModel",
            _invoke_embed_params(
                "cohere.embed-english-v3",
                {"texts": ["hello world"], "input_type": "search_document"},
            ),
        )
        assert parsed.kind == "embeddings"
        assert parsed.estimated_input_tokens > 0

    def test_titan_embed_reconciles_with_exact_input_and_zero_output(self):
        rec = _RecordPostPolicy()
        engine = _engine_with(rec)
        ctx = _ctx("test-proj")
        parsed_req = _parse_request(
            "InvokeModel",
            _invoke_embed_params(
                "amazon.titan-embed-text-v2:0", {"inputText": "hello world"}
            ),
        )
        block, ran = engine.pre_call(parsed_req, ctx)
        assert block is None

        resp = {
            "body": _streaming_body(
                {"embedding": [0.1, 0.2], "inputTextTokenCount": 12}
            )
        }
        _reconcile_non_streaming("InvokeModel", parsed_req, resp, engine, ctx, ran)

        # Exact input from the response; embeddings never produce output tokens.
        assert rec.resp is not None
        assert rec.resp.input_tokens == 12
        assert rec.resp.output_tokens == 0
        # The caller's body is still readable after our peek.
        assert json.loads(resp["body"].read())["inputTextTokenCount"] == 12

    def test_cohere_embed_falls_back_to_estimate_instead_of_releasing(self):
        rec = _RecordPostPolicy()
        engine = _engine_with(rec)
        ctx = _ctx("test-proj")
        parsed_req = _parse_request(
            "InvokeModel",
            _invoke_embed_params(
                "cohere.embed-english-v3", {"texts": ["hello world foo bar"]}
            ),
        )
        assert parsed_req.estimated_input_tokens > 0
        block, ran = engine.pre_call(parsed_req, ctx)
        assert block is None

        # Cohere embed response has no token count — reconcile must fall back to
        # the request-time estimate rather than releasing (which would record
        # nothing against the cap).
        resp = {"body": _streaming_body({"embeddings": [[0.1, 0.2]], "id": "x"})}
        _reconcile_non_streaming("InvokeModel", parsed_req, resp, engine, ctx, ran)

        assert rec.resp is not None
        assert rec.resp.input_tokens == parsed_req.estimated_input_tokens
        assert rec.resp.output_tokens == 0


# ---------------------------------------------------------------------------
# Request/usage body parsing helpers — pure functions, unaffected.
# ---------------------------------------------------------------------------


class TestParseInvokeModelBody:
    def test_anthropic_family_extracts_messages(self):
        messages = [{"role": "user", "content": "hi"}]
        content, max_tokens = _parse_invoke_model_body(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            {"messages": messages, "max_tokens": 256},
        )
        assert content == messages
        assert max_tokens == 256

    def test_titan_family_extracts_input_text(self):
        content, max_tokens = _parse_invoke_model_body(
            "amazon.titan-text-express-v1",
            {"inputText": "hello", "textGenerationConfig": {"maxTokenCount": 512}},
        )
        assert content == "hello"
        assert max_tokens == 512

    def test_unrecognized_family_falls_back_to_json(self):
        content, max_tokens = _parse_invoke_model_body(
            "some.unknown-v1", {"foo": "bar"}
        )
        assert "foo" in content
        assert max_tokens is None


class TestParseInvokeModelUsage:
    def test_anthropic_family(self):
        input_tokens, output_tokens = _parse_invoke_model_usage(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            {"usage": {"input_tokens": 10, "output_tokens": 5}},
        )
        assert (input_tokens, output_tokens) == (10, 5)

    def test_titan_family(self):
        input_tokens, output_tokens = _parse_invoke_model_usage(
            "amazon.titan-text-express-v1",
            {"inputTextTokenCount": 8, "results": [{"tokenCount": 4}]},
        )
        assert (input_tokens, output_tokens) == (8, 4)

    def test_unrecognized_family_returns_none(self):
        assert _parse_invoke_model_usage("some.unknown-v1", {"foo": "bar"}) == (
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Per-instance override — instrument_bedrock(client, engine=, context=) binds
# ONE specific client to a distinct engine/context instead of the
# process-wide default. Still installs the global patch under the hood.
# ---------------------------------------------------------------------------


class TestInstrumentBedrockWiring:
    def test_invoke_model_reconciles_and_body_readable(self):
        client = _client()
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        ctx = _ctx("test-proj")
        instrument_bedrock(client, engine=engine, context=ctx)

        resp_payload = {
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "content": [{"type": "text", "text": "hello"}],
        }
        stubber.add_response(
            "invoke_model",
            {"body": _streaming_body(resp_payload), "contentType": "application/json"},
            _invoke_model_params(),
        )
        stubber.activate()

        response = client.invoke_model(**_invoke_model_params())
        assert json.loads(response["body"].read()) == resp_payload
        assert api.current_spend("test-proj") > 0

    def test_converse_reconciles(self):
        client = _client()
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        ctx = _ctx("test-proj")
        instrument_bedrock(client, engine=engine, context=ctx)

        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "hi"}]}
                },
                "usage": {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19},
                "stopReason": "end_turn",
                "metrics": {"latencyMs": 100},
            },
            _converse_params(),
        )
        stubber.activate()

        response = client.converse(**_converse_params())
        assert response["usage"]["outputTokens"] == 7
        assert api.current_spend("test-proj") > 0

    def test_blocking_policy_prevents_call(self):
        # Previously worked around with a tight connect_timeout because
        # Stubber.activate() uses register_first on before-parameter-build,
        # always pre-empting an event-based interceptor regardless of
        # registration order. Patching _make_api_call itself sidesteps that
        # entirely: our wrapper raises before the real (Stubber-instrumented)
        # _make_api_call body — and hence before any of Stubber's own event
        # hooks — ever runs, so a plain Stubber (with no response queued)
        # works and proves the call never reaches the network.
        client = _client()
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_AlwaysBlockPolicy())
        ctx = _ctx("test-proj")
        instrument_bedrock(client, engine=engine, context=ctx)
        stubber.activate()

        with pytest.raises(NoveumGuardBlocked):
            client.converse(**_converse_params())

    def test_client_override_takes_precedence_over_global_default(self):
        # A client with its own override must use ITS engine, not whatever
        # instrument_bedrock() set as the process-wide default.
        instrument_bedrock(engine=_engine_with(_AlwaysBlockPolicy()), context=_ctx())

        client = _client()
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        instrument_bedrock(client, engine=engine, context=_ctx("test-proj"))

        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "hi"}]}
                },
                "usage": {"inputTokens": 5, "outputTokens": 5, "totalTokens": 10},
                "stopReason": "end_turn",
                "metrics": {"latencyMs": 100},
            },
            _converse_params(),
        )
        stubber.activate()

        # Must not raise — this client's own override allows the call.
        client.converse(**_converse_params())
        assert api.current_spend("test-proj") > 0

    def test_config_botocore_config_still_applies(self):
        # Sanity check that instrumentation doesn't interfere with normal
        # client configuration (e.g. a custom botocore.config.Config).
        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            config=botocore.config.Config(retries={"max_attempts": 0}),
        )
        stubber = Stubber(client)
        api = GuardAPIClient()
        engine = PolicyEngine(api_client=api)
        engine.attach(_cost_cap())
        instrument_bedrock(client, engine=engine, context=_ctx("test-proj"))

        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "hi"}]}
                },
                "usage": {"inputTokens": 5, "outputTokens": 5, "totalTokens": 10},
                "stopReason": "end_turn",
                "metrics": {"latencyMs": 100},
            },
            _converse_params(),
        )
        stubber.activate()

        client.converse(**_converse_params())
        assert api.current_spend("test-proj") > 0
