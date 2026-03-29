"""Unit tests for Pipecat integration utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_extract_metrics_data_llm_usage() -> None:
    """LLMUsageMetricsData uses nested value (LLMTokenUsage) from pipecat.metrics.metrics."""
    pytest.importorskip("pipecat.metrics.metrics")
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData

    from noveum_trace.integrations.pipecat.pipecat_utils import extract_metrics_data

    usage = LLMTokenUsage(
        prompt_tokens=5,
        completion_tokens=7,
        total_tokens=12,
    )
    item = LLMUsageMetricsData(processor="test-llm", model="gpt-4o-mini", value=usage)
    frame = MetricsFrame(data=[item])

    out = extract_metrics_data(frame)
    assert out["prompt_tokens"] == 5
    assert out["completion_tokens"] == 7
    assert out["total_tokens"] == 12
    assert out["llm_model"] == "gpt-4o-mini"


def test_extract_metrics_data_ttfb() -> None:
    pytest.importorskip("pipecat.metrics.metrics")
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import TTFBMetricsData

    from noveum_trace.integrations.pipecat.pipecat_utils import extract_metrics_data

    item = TTFBMetricsData(processor="LLM", model=None, value=0.25)
    frame = MetricsFrame(data=[item])
    out = extract_metrics_data(frame)
    assert out["ttfb_seconds"] == pytest.approx(0.25)
    assert "LLM" in out["ttfb_processor"]


def test_upload_audio_frames_uses_passed_client() -> None:
    pytest.importorskip("pipecat.frames.frames")
    from pipecat.frames.frames import AudioRawFrame

    from noveum_trace.integrations.pipecat.pipecat_utils import upload_audio_frames

    client = MagicMock()
    pcm = b"\x00\x00" * 160  # 160 samples mono 16-bit
    frames = [AudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)]

    ok = upload_audio_frames(
        frames,
        audio_uuid="uuid-test",
        audio_type="stt",
        trace_id="t1",
        span_id="s1",
        client=client,
    )
    assert ok is True
    client.export_audio.assert_called_once()
    call_kw = client.export_audio.call_args[1]
    assert call_kw["trace_id"] == "t1"
    assert call_kw["span_id"] == "s1"
    assert call_kw["audio_uuid"] == "uuid-test"
    assert len(call_kw["audio_data"]) > 0


def test_merge_llm_pending_stash() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import merge_llm_pending_stash

    existing: dict = {"messages": "[1]"}
    merge_llm_pending_stash(existing, {"tools": "[]"})
    assert existing["messages"] == "[1]"
    assert existing["tools"] == "[]"
    merge_llm_pending_stash(existing, {"messages": "[2]"})
    assert existing["messages"] == "[2]"


def test_merge_appended_messages_json() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import (
        merge_appended_messages_json,
    )

    assert (
        merge_appended_messages_json(None, [{"role": "user"}]) == '[{"role": "user"}]'
    )
    out = merge_appended_messages_json('[{"role":"system"}]', [{"role": "user"}])
    assert '"system"' in out and '"user"' in out


def test_serialize_tools_field_list() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import serialize_tools_field

    schema = [{"type": "function", "function": {"name": "x"}}]
    s = serialize_tools_field(schema)
    assert s is not None and "function" in s


def test_serialize_tool_choice_field() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import (
        serialize_tool_choice_field,
    )

    assert '"auto"' in serialize_tool_choice_field("auto")  # type: ignore[arg-type]


def test_extract_stt_confidence() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_stt_confidence

    assert extract_stt_confidence(None) is None
    assert (
        extract_stt_confidence({"channel": {"alternatives": [{"confidence": 0.95}]}})
        == 0.95
    )


def test_llm_token_usage_separate_cache_fields() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import _llm_token_usage_to_dict

    class _U:
        prompt_tokens = 1
        completion_tokens = 2
        total_tokens = 3
        cache_read_input_tokens = 100
        cache_creation_input_tokens = 200

    out = _llm_token_usage_to_dict(_U())
    assert out["cache_read_tokens"] == 100
    assert out["cache_creation_tokens"] == 200


def test_extract_service_settings_empty() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_service_settings

    assert extract_service_settings(object()) == {}


def test_extract_service_settings_full() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_service_settings

    class _Lang:
        value = "en-US"

    class _Settings:
        model = "gpt-4o"
        voice = "alloy"
        language = _Lang()
        system_instruction = "Be helpful"
        temperature = 0.7
        max_tokens = 100
        top_p = 0.9
        seed = 42

    proc = MagicMock()
    proc._settings = _Settings()
    out = extract_service_settings(proc)
    assert out["model"] == "gpt-4o"
    assert out["voice"] == "alloy"
    assert out["language"] == "en-US"
    assert out["system_instruction"] == "Be helpful"
    assert out["temperature"] == 0.7
    assert out["max_tokens"] == 100
    assert out["top_p"] == 0.9
    assert out["seed"] == 42


def test_extract_service_settings_system_prompt_fallback() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_service_settings

    class _Settings:
        system_prompt = "From prompt"

    proc = MagicMock()
    proc._settings = _Settings()
    assert extract_service_settings(proc)["system_instruction"] == "From prompt"


def test_extract_llm_context_data_none() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_llm_context_data

    assert extract_llm_context_data(None) == {}


def test_extract_llm_context_data_messages() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_llm_context_data

    ctx = MagicMock()
    ctx.get_messages = MagicMock(return_value=[{"role": "user", "content": "hi"}])
    ctx.tools = None
    out = extract_llm_context_data(ctx)
    assert "messages" in out
    assert "user" in out["messages"]


def test_extract_llm_context_data_tools_list() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_llm_context_data

    ctx = MagicMock()
    ctx.get_messages = MagicMock(return_value=[])
    ctx.tools = [{"type": "function", "function": {"name": "x"}}]
    out = extract_llm_context_data(ctx)
    assert "tools" in out


def test_merge_llm_pending_stash_empty_updates() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import merge_llm_pending_stash

    existing: dict = {"a": "1"}
    merge_llm_pending_stash(existing, {})
    merge_llm_pending_stash(existing, {"b": ""})
    assert existing == {"a": "1"}


def test_json_dumps_messages() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import json_dumps_messages

    assert json_dumps_messages([]) is None
    assert json_dumps_messages(None) is None
    s = json_dumps_messages([{"role": "user"}])
    assert s is not None and "user" in s


def test_merge_appended_messages_json_parse_error_fallback() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import (
        merge_appended_messages_json,
    )

    out = merge_appended_messages_json("not-json", [{"role": "user"}])
    assert out is not None and "user" in out


def test_resolve_tools_to_list_and_coerce() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import (
        _coerce_function_schemas,
        _resolve_tools_to_list,
    )

    class _Schema:
        def to_default_dict(self) -> dict:
            return {"name": "fn"}

    assert _resolve_tools_to_list([{"a": 1}]) == [{"a": 1}]
    assert _coerce_function_schemas([_Schema()]) == [{"name": "fn"}]

    class _ToolsSchema:
        standard_tools = [_Schema()]

    assert _resolve_tools_to_list(_ToolsSchema()) == [{"name": "fn"}]

    class _Legacy:
        tools = [{"x": 1}]

    assert _resolve_tools_to_list(_Legacy()) == [{"x": 1}]


def test_serialize_tools_field_sentinels() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import serialize_tools_field

    assert serialize_tools_field(None) is None
    assert serialize_tools_field(False) is None

    class _NG:
        def __repr__(self) -> str:
            return "<NOT_GIVEN>"

    assert serialize_tools_field(_NG()) is None


def test_serialize_tool_choice_field_fallback_str() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import (
        serialize_tool_choice_field,
    )

    class _Bad:
        def __repr__(self) -> str:
            return "bad"

    with patch(
        "noveum_trace.integrations.pipecat.pipecat_utils.json.dumps",
        side_effect=TypeError("x"),
    ):
        assert serialize_tool_choice_field(_Bad()) == "bad"


def test_truncate_for_trace_attr() -> None:
    from noveum_trace.integrations.pipecat.pipecat_constants import (
        MAX_TEXT_BUFFER_LENGTH,
    )
    from noveum_trace.integrations.pipecat.pipecat_utils import truncate_for_trace_attr

    assert truncate_for_trace_attr("short") == "short"
    long_text = "a" * (MAX_TEXT_BUFFER_LENGTH + 10)
    out = truncate_for_trace_attr(long_text)
    assert out.endswith("...")
    assert len(out) == MAX_TEXT_BUFFER_LENGTH


def test_llm_token_usage_model_dump() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import _llm_token_usage_to_dict

    u = MagicMock()
    u.model_dump = MagicMock(
        return_value={
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "reasoning_tokens": 5,
        }
    )
    out = _llm_token_usage_to_dict(u)
    assert out["total_tokens"] == 3
    assert out["reasoning_tokens"] == 5


def test_llm_token_usage_computes_total() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import _llm_token_usage_to_dict

    class _U:
        prompt_tokens = 10
        completion_tokens = 20
        total_tokens = None

    out = _llm_token_usage_to_dict(_U())
    assert out["total_tokens"] == 30


def test_extract_metrics_data_processing_and_tts() -> None:
    pytest.importorskip("pipecat.metrics.metrics")
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import ProcessingMetricsData, TTSUsageMetricsData

    from noveum_trace.integrations.pipecat.pipecat_utils import extract_metrics_data

    frame = MetricsFrame(
        data=[
            ProcessingMetricsData(processor="p", value=0.5),
            TTSUsageMetricsData(processor="tts", value=120),
        ]
    )
    out = extract_metrics_data(frame)
    assert out["processing_seconds"] == pytest.approx(0.5)
    assert out["tts_characters"] == 120


def test_extract_metrics_data_text_aggregation_if_available() -> None:
    pytest.importorskip("pipecat.metrics.metrics")
    try:
        from pipecat.metrics.metrics import TextAggregationMetricsData
    except ImportError:
        pytest.skip("TextAggregationMetricsData not available")

    from pipecat.frames.frames import MetricsFrame

    from noveum_trace.integrations.pipecat.pipecat_utils import extract_metrics_data

    frame = MetricsFrame(data=[TextAggregationMetricsData(processor="agg", value=0.33)])
    out = extract_metrics_data(frame)
    assert out["text_aggregation_seconds"] == pytest.approx(0.33)


def test_extract_metrics_data_smart_turn_if_available() -> None:
    pytest.importorskip("pipecat.metrics.metrics")
    try:
        from pipecat.metrics.metrics import SmartTurnMetricsData
    except ImportError:
        pytest.skip("SmartTurnMetricsData not available")

    from pipecat.frames.frames import MetricsFrame

    from noveum_trace.integrations.pipecat.pipecat_utils import extract_metrics_data

    frame = MetricsFrame(
        data=[
            SmartTurnMetricsData(
                processor="st",
                is_complete=True,
                probability=0.9,
                e2e_processing_time_ms=10.0,
                inference_time_ms=2.0,
                server_total_time_ms=3.0,
            )
        ]
    )
    out = extract_metrics_data(frame)
    assert out["turn_eou_is_complete"] is True
    assert out["turn_eou_confidence"] == pytest.approx(0.9)
    assert "turn_eou_inference_ms" in out


def test_extract_stt_confidence_object_paths() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_stt_confidence

    r = MagicMock()
    r.confidence = 0.88
    assert extract_stt_confidence(r) == pytest.approx(0.88)

    class _Alt:
        confidence = 0.77

    class _Ch:
        alternatives = [_Alt()]

    class _R2:
        channel = _Ch()

    assert extract_stt_confidence(_R2()) == pytest.approx(0.77)


def test_extract_frame_text() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_frame_text

    f = MagicMock()
    f.text = "hello"
    assert extract_frame_text(f) == "hello"
    assert extract_frame_text(object()) is None


def test_extract_function_call_data() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import (
        extract_function_call_data,
    )

    class _Frame:
        function_name = "get_weather"
        tool_call_id = "call_1"
        arguments = {"city": "NYC"}
        result = "sunny"
        run_llm = True

    out = extract_function_call_data(_Frame())
    assert out["function_name"] == "get_weather"
    assert out["tool_call_id"] == "call_1"
    assert "city" in out["arguments"]
    assert out["run_llm"] is True


def test_serialize_processor_info() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import serialize_processor_info

    p = MagicMock()
    p.name = "stt-1"
    p._settings = None
    info = serialize_processor_info(p)
    assert info["name"] == "stt-1"
    assert info["class"] == "MagicMock"


def test_calculate_llm_cost() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import calculate_llm_cost

    out = calculate_llm_cost("gpt-4o-mini", 100, 50)
    assert isinstance(out, dict)
    assert "total" in out or out == {}


def test_frames_to_wav_and_duration() -> None:
    pytest.importorskip("pipecat.frames.frames")
    from pipecat.frames.frames import AudioRawFrame

    from noveum_trace.integrations.pipecat.pipecat_utils import (
        _frames_to_wav_bytes,
        calculate_audio_duration_ms,
    )

    # 320 stereo samples (2 bytes each) = 640 bytes mono 16-bit → 320 samples at 16 kHz → 20 ms
    pcm = b"\x00\x01" * 320
    frames = [AudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)]
    wav = _frames_to_wav_bytes(frames)
    assert wav.startswith(b"RIFF")
    assert calculate_audio_duration_ms(frames) == pytest.approx(20.0, rel=0.01)
    assert calculate_audio_duration_ms([]) == 0.0


def test_upload_audio_frames_empty_and_no_client() -> None:
    pytest.importorskip("pipecat.frames.frames")
    from pipecat.frames.frames import AudioRawFrame

    from noveum_trace.integrations.pipecat.pipecat_utils import upload_audio_frames

    assert upload_audio_frames([], "u", "stt", "t", "s") is False

    # Empty PCM still yields a non-empty WAV header; upload proceeds.
    frames = [AudioRawFrame(audio=b"", sample_rate=16000, num_channels=1)]
    client = MagicMock()
    assert upload_audio_frames(frames, "u", "stt", "t", "s", client=client) is True
    client.export_audio.assert_called_once()

    pcm = b"\x00\x00" * 10
    frames2 = [AudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)]
    with patch_get_client_none():
        assert upload_audio_frames(frames2, "u", "stt", "t", "s") is False


def patch_get_client_none():
    return patch("noveum_trace.get_client", return_value=None)
