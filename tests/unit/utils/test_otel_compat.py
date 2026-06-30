"""Unit tests for OpenTelemetry compatibility helpers."""

import uuid

from noveum_trace.utils import otel_compat


class TestIdConversion:
    def test_trace_id_strips_dashes_and_lowercases(self):
        u = uuid.uuid4()
        result = otel_compat.to_otel_trace_id(str(u))
        assert result == u.hex
        assert len(result) == 32
        assert "-" not in result

    def test_trace_id_is_lossless_for_uuid(self):
        u = "9803895D-E802-4B92-AB83-BEE8F3440EBB"
        assert otel_compat.to_otel_trace_id(u) == "9803895de8024b92ab83bee8f3440ebb"

    def test_span_id_truncates_to_16_hex(self):
        result = otel_compat.to_otel_span_id(str(uuid.uuid4()))
        assert len(result) == 16
        assert "-" not in result

    def test_span_id_is_deterministic(self):
        sid = str(uuid.uuid4())
        assert otel_compat.to_otel_span_id(sid) == otel_compat.to_otel_span_id(sid)

    def test_none_ids_return_none(self):
        assert otel_compat.to_otel_trace_id(None) is None
        assert otel_compat.to_otel_span_id(None) is None
        assert otel_compat.to_otel_trace_id("") is None


class TestGenAiCrosswalk:
    def test_basic_mapping(self):
        attrs = {
            "llm.model": "claude-sonnet-4",
            "llm.provider": "anthropic",
            "llm.operation": "chat",
            "llm.input_tokens": 10,
            "llm.output_tokens": 5,
            "llm.temperature": 0.7,
        }
        result = otel_compat.derive_gen_ai_attributes(attrs)
        assert result["gen_ai.request.model"] == "claude-sonnet-4"
        assert result["gen_ai.provider.name"] == "anthropic"
        assert result["gen_ai.operation.name"] == "chat"
        assert result["gen_ai.usage.input_tokens"] == 10
        assert result["gen_ai.usage.output_tokens"] == 5
        assert result["gen_ai.request.temperature"] == 0.7

    def test_does_not_mutate_input(self):
        attrs = {"llm.model": "m"}
        otel_compat.derive_gen_ai_attributes(attrs)
        assert "gen_ai.request.model" not in attrs

    def test_finish_reason_wrapped_in_array(self):
        result = otel_compat.derive_gen_ai_attributes({"llm.finish_reason": "stop"})
        assert result["gen_ai.response.finish_reasons"] == ["stop"]

    def test_finish_reason_list_passthrough(self):
        result = otel_compat.derive_gen_ai_attributes(
            {"llm.finish_reason": ["stop", "length"]}
        )
        assert result["gen_ai.response.finish_reasons"] == ["stop", "length"]

    def test_token_fallback_keys(self):
        result = otel_compat.derive_gen_ai_attributes({"llm.prompt_tokens": 42})
        assert result["gen_ai.usage.input_tokens"] == 42

    def test_skips_none_empty_dict_and_mock_placeholders(self):
        attrs = {
            "llm.model": None,
            "llm.top_p": {},
            "llm.finish_reason": "<Mock object>",
        }
        result = otel_compat.derive_gen_ai_attributes(attrs)
        assert result == {}

    def test_does_not_overwrite_existing_gen_ai_key(self):
        attrs = {"llm.model": "legacy", "gen_ai.request.model": "explicit"}
        result = otel_compat.derive_gen_ai_attributes(attrs)
        assert "gen_ai.request.model" not in result


class TestSpanKind:
    def test_llm_name_is_client(self):
        assert otel_compat.infer_span_kind("llm.chat", {}) == otel_compat.KIND_CLIENT

    def test_http_name_is_client(self):
        assert (
            otel_compat.infer_span_kind("http.request", {}) == otel_compat.KIND_CLIENT
        )

    def test_gen_ai_attr_is_client(self):
        assert (
            otel_compat.infer_span_kind("something", {"gen_ai.request.model": "m"})
            == otel_compat.KIND_CLIENT
        )

    def test_default_is_internal(self):
        assert (
            otel_compat.infer_span_kind("agent.step", {}) == otel_compat.KIND_INTERNAL
        )


class TestStatus:
    def test_ok(self):
        assert otel_compat.to_otel_status("ok", None) == {"code": "OK"}

    def test_error_includes_message(self):
        assert otel_compat.to_otel_status("error", "boom") == {
            "code": "ERROR",
            "message": "boom",
        }

    def test_timeout_and_cancelled_map_to_error(self):
        assert otel_compat.to_otel_status("timeout", None)["code"] == "ERROR"
        assert otel_compat.to_otel_status("cancelled", None)["code"] == "ERROR"

    def test_unset_default(self):
        assert otel_compat.to_otel_status(None, None) == {"code": "UNSET"}

    def test_ok_drops_message(self):
        assert "message" not in otel_compat.to_otel_status("ok", "ignored")


class TestSpanName:
    def test_operation_and_model(self):
        attrs = {"llm.model": "claude-sonnet-4", "llm.operation": "chat"}
        assert otel_compat.otel_span_name("llm.chat", attrs) == "chat claude-sonnet-4"

    def test_derives_operation_from_name(self):
        attrs = {"llm.model": "gpt-4"}
        assert otel_compat.otel_span_name("llm.completion", attrs) == "completion gpt-4"

    def test_falls_back_to_name_without_model(self):
        assert otel_compat.otel_span_name("agent.step", {}) == "agent.step"


class TestResource:
    def test_full_resource(self):
        resource = otel_compat.build_resource(
            service_name="proj",
            sdk_version="1.5.17",
            environment="production",
            service_version="v2",
        )
        assert resource["service.name"] == "proj"
        assert resource["telemetry.sdk.name"] == otel_compat.DEFAULT_SCOPE_NAME
        assert resource["telemetry.sdk.version"] == "1.5.17"
        assert resource["telemetry.sdk.language"] == "python"
        assert resource["deployment.environment"] == "production"
        assert resource["service.version"] == "v2"

    def test_omits_absent_optional_fields(self):
        resource = otel_compat.build_resource(
            service_name=None, sdk_version="1.0", environment=None
        )
        assert "service.name" not in resource
        assert "deployment.environment" not in resource
        assert "service.version" not in resource
