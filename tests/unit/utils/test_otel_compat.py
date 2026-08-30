"""Unit tests for OpenTelemetry compatibility helpers."""

import uuid

from noveum_trace.utils import otel_compat


class TestIdConversion:
    """Tests for W3C trace and span ID conversion functions."""

    def test_trace_id_strips_dashes_and_lowercases(self) -> None:
        """Verify that UUID trace IDs are converted to lowercase 32-hex format."""
        u = uuid.uuid4()
        result = otel_compat.to_otel_trace_id(str(u))
        assert result == u.hex
        assert len(result) == 32
        assert "-" not in result

    def test_trace_id_is_lossless_for_uuid(self) -> None:
        """Verify that UUID trace ID conversion is deterministic and lossless."""
        u = "9803895D-E802-4B92-AB83-BEE8F3440EBB"
        assert otel_compat.to_otel_trace_id(u) == "9803895de8024b92ab83bee8f3440ebb"

    def test_span_id_truncates_to_16_hex(self) -> None:
        """Verify that span IDs are truncated to 16-hex characters."""
        result = otel_compat.to_otel_span_id(str(uuid.uuid4()))
        assert len(result) == 16
        assert "-" not in result

    def test_span_id_is_deterministic(self) -> None:
        """Verify that the same span ID produces identical 16-hex output."""
        sid = str(uuid.uuid4())
        assert otel_compat.to_otel_span_id(sid) == otel_compat.to_otel_span_id(sid)

    def test_none_ids_return_none(self) -> None:
        """Verify that None or empty ID strings return None."""
        assert otel_compat.to_otel_trace_id(None) is None
        assert otel_compat.to_otel_span_id(None) is None
        assert otel_compat.to_otel_trace_id("") is None


class TestGenAiCrosswalk:
    """Tests for OpenTelemetry GenAI semantic convention attribute derivation."""

    def test_basic_mapping(self) -> None:
        """Verify basic crosswalk of LLM attributes to GenAI conventions."""
        attrs = {
            "llm.model": "claude-sonnet-4",
            "llm.provider": "anthropic",
            "llm.operation": "chat",
            "llm.input_tokens": 10,
            "llm.output_tokens": 5,
            "llm.temperature": 0.7,
        }
        result = otel_compat.derive_gen_ai_attributes(attrs)
        assert result["gen_ai.system"] == "anthropic"
        assert result["gen_ai.request.model"] == "claude-sonnet-4"
        assert result["gen_ai.provider.name"] == "anthropic"
        assert result["gen_ai.operation.name"] == "chat"
        assert result["gen_ai.usage.input_tokens"] == 10
        assert result["gen_ai.usage.output_tokens"] == 5
        assert result["gen_ai.request.temperature"] == 0.7

    def test_does_not_mutate_input(self) -> None:
        """Verify that derive_gen_ai_attributes does not mutate the input dictionary."""
        attrs = {"llm.model": "m"}
        otel_compat.derive_gen_ai_attributes(attrs)
        assert "gen_ai.request.model" not in attrs

    def test_system_and_penalties_mapping(self) -> None:
        """Verify mapping of gen_ai.system, presence_penalty, and frequency_penalty."""
        attrs = {
            "llm.provider": "openai",
            "llm.presence_penalty": 0.5,
            "llm.frequency_penalty": 0.2,
        }
        result = otel_compat.derive_gen_ai_attributes(attrs)
        assert result["gen_ai.system"] == "openai"
        assert result["gen_ai.request.presence_penalty"] == 0.5
        assert result["gen_ai.request.frequency_penalty"] == 0.2

    def test_system_google_normalizes_to_gemini(self) -> None:
        """Verify that provider 'google' is normalized to 'gemini' for gen_ai.system."""
        result = otel_compat.derive_gen_ai_attributes({"llm.provider": "google"})
        assert result["gen_ai.system"] == "gemini"
        assert result["gen_ai.provider.name"] == "google"

    def test_penalties_input_prefixed_fallback(self) -> None:
        """Verify fallback to llm.input.* keys for penalties."""
        attrs = {
            "llm.input.presence_penalty": 0.3,
            "llm.input.frequency_penalty": 0.1,
        }
        result = otel_compat.derive_gen_ai_attributes(attrs)
        assert result["gen_ai.request.presence_penalty"] == 0.3
        assert result["gen_ai.request.frequency_penalty"] == 0.1

    def test_stop_sequences_wrapped_in_array(self) -> None:
        """Verify that a scalar stop sequence string is normalized into an array."""
        result = otel_compat.derive_gen_ai_attributes({"llm.stop": "STOP"})
        assert result["gen_ai.request.stop_sequences"] == ["STOP"]

    def test_stop_sequences_list_passthrough(self) -> None:
        """Verify that a list of stop sequences passes through unchanged."""
        result = otel_compat.derive_gen_ai_attributes(
            {"llm.stop_sequences": ["\n\n", "END"]}
        )
        assert result["gen_ai.request.stop_sequences"] == ["\n\n", "END"]

    def test_stop_sequences_input_prefixed(self) -> None:
        """Verify fallback to llm.input.stop for stop sequences."""
        result = otel_compat.derive_gen_ai_attributes({"llm.input.stop": ["END"]})
        assert result["gen_ai.request.stop_sequences"] == ["END"]

    def test_finish_reason_wrapped_in_array(self) -> None:
        """Verify that a scalar finish_reason is wrapped in a list."""
        result = otel_compat.derive_gen_ai_attributes({"llm.finish_reason": "stop"})
        assert result["gen_ai.response.finish_reasons"] == ["stop"]

    def test_finish_reason_list_passthrough(self) -> None:
        """Verify that a list of finish reasons passes through unchanged."""
        result = otel_compat.derive_gen_ai_attributes(
            {"llm.finish_reason": ["stop", "length"]}
        )
        assert result["gen_ai.response.finish_reasons"] == ["stop", "length"]

    def test_token_fallback_keys(self) -> None:
        """Verify that fallback token count keys are properly mapped."""
        result = otel_compat.derive_gen_ai_attributes({"llm.prompt_tokens": 42})
        assert result["gen_ai.usage.input_tokens"] == 42

    def test_skips_none_empty_dict_and_mock_placeholders(self) -> None:
        """Verify that None, empty dicts, and placeholder mock strings are skipped."""
        attrs = {
            "llm.model": None,
            "llm.top_p": {},
            "llm.finish_reason": "<Mock object>",
            "llm.stop": None,
        }
        result = otel_compat.derive_gen_ai_attributes(attrs)
        assert result == {}

    def test_does_not_overwrite_existing_gen_ai_key(self) -> None:
        """Verify that explicitly provided gen_ai.* keys are preserved."""
        attrs = {
            "llm.model": "legacy",
            "gen_ai.request.model": "explicit",
            "llm.provider": "legacy_provider",
            "gen_ai.system": "explicit_system",
            "llm.stop": "legacy_stop",
            "gen_ai.request.stop_sequences": ["explicit_stop"],
        }
        result = otel_compat.derive_gen_ai_attributes(attrs)
        assert "gen_ai.request.model" not in result
        assert "gen_ai.system" not in result
        assert "gen_ai.request.stop_sequences" not in result


class TestSpanKind:
    """Tests for OpenTelemetry span kind inference."""

    def test_llm_name_is_client(self) -> None:
        """Verify that llm.* span names infer KIND_CLIENT."""
        assert otel_compat.infer_span_kind("llm.chat", {}) == otel_compat.KIND_CLIENT

    def test_http_name_is_client(self) -> None:
        """Verify that http.* span names infer KIND_CLIENT."""
        assert (
            otel_compat.infer_span_kind("http.request", {}) == otel_compat.KIND_CLIENT
        )

    def test_gen_ai_attr_is_client(self) -> None:
        """Verify that spans with gen_ai.* attributes infer KIND_CLIENT."""
        assert (
            otel_compat.infer_span_kind("something", {"gen_ai.request.model": "m"})
            == otel_compat.KIND_CLIENT
        )

    def test_default_is_internal(self) -> None:
        """Verify that other span types default to KIND_INTERNAL."""
        assert (
            otel_compat.infer_span_kind("agent.step", {}) == otel_compat.KIND_INTERNAL
        )


class TestStatus:
    """Tests for OpenTelemetry status code and message mapping."""

    def test_ok(self) -> None:
        """Verify that ok status maps to OK code without message."""
        assert otel_compat.to_otel_status("ok", None) == {"code": "OK"}

    def test_error_includes_message(self) -> None:
        """Verify that error status includes error message."""
        assert otel_compat.to_otel_status("error", "boom") == {
            "code": "ERROR",
            "message": "boom",
        }

    def test_timeout_and_cancelled_map_to_error(self) -> None:
        """Verify that timeout and cancelled statuses map to ERROR."""
        assert otel_compat.to_otel_status("timeout", None)["code"] == "ERROR"
        assert otel_compat.to_otel_status("cancelled", None)["code"] == "ERROR"

    def test_unset_default(self) -> None:
        """Verify that missing status defaults to UNSET."""
        assert otel_compat.to_otel_status(None, None) == {"code": "UNSET"}

    def test_ok_drops_message(self) -> None:
        """Verify that ok status ignores status message per OTel specs."""
        assert "message" not in otel_compat.to_otel_status("ok", "ignored")


class TestSpanName:
    """Tests for OpenTelemetry span name formatting."""

    def test_operation_and_model(self) -> None:
        """Verify span name formatting when operation and model are known."""
        attrs = {"llm.model": "claude-sonnet-4", "llm.operation": "chat"}
        assert otel_compat.otel_span_name("llm.chat", attrs) == "chat claude-sonnet-4"

    def test_derives_operation_from_name(self) -> None:
        """Verify derivation of operation name from span name prefix."""
        attrs = {"llm.model": "gpt-4"}
        assert otel_compat.otel_span_name("llm.completion", attrs) == "completion gpt-4"

    def test_falls_back_to_name_without_model(self) -> None:
        """Verify fallback to original span name when model is absent."""
        assert otel_compat.otel_span_name("agent.step", {}) == "agent.step"


class TestResource:
    """Tests for OpenTelemetry resource attribute building."""

    def test_full_resource(self) -> None:
        """Verify building a full resource dictionary with all metadata."""
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

    def test_omits_absent_optional_fields(self) -> None:
        """Verify that absent optional resource fields are omitted."""
        resource = otel_compat.build_resource(
            service_name=None, sdk_version="1.0", environment=None
        )
        assert "service.name" not in resource
        assert "deployment.environment" not in resource
        assert "service.version" not in resource
