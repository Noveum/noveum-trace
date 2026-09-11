"""Unit tests for Pipecat integration utilities."""

from __future__ import annotations

import types
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


def test_normalize_metrics_data_preserves_every_item_and_identity() -> None:
    pytest.importorskip("pipecat.metrics.metrics")
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData

    from noveum_trace.integrations.pipecat.pipecat_utils import normalize_metrics_data

    frame = MetricsFrame(
        data=[
            TTFBMetricsData(processor="stt-a", model="nova", value=0.1),
            TTFBMetricsData(processor="llm-b", model="gpt", value=0.2),
            ProcessingMetricsData(processor="tts-c", model="sonic", value=0.3),
        ]
    )
    records = normalize_metrics_data(frame)

    assert len(records) == 3
    assert [record.family for record in records] == ["ttfb", "ttfb", "processing"]
    assert [record.value for record in records] == pytest.approx([0.1, 0.2, 0.3])
    assert [record.unit for record in records] == ["seconds", "seconds", "seconds"]
    assert [record.processor for record in records] == ["stt-a", "llm-b", "tts-c"]
    assert [record.model for record in records] == ["nova", "gpt", "sonic"]


def test_normalize_metrics_data_preserves_zero_values() -> None:
    pytest.importorskip("pipecat.metrics.metrics")
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData

    from noveum_trace.integrations.pipecat.pipecat_utils import (
        extract_metrics_data,
        normalize_metrics_data,
    )

    frame = MetricsFrame(
        data=[
            TTFBMetricsData(processor="llm", value=0.0),
            ProcessingMetricsData(processor="llm", value=0.0),
        ]
    )
    assert [record.value for record in normalize_metrics_data(frame)] == [0.0, 0.0]
    assert extract_metrics_data(frame)["ttfb_seconds"] == 0.0
    assert extract_metrics_data(frame)["processing_seconds"] == 0.0


def test_normalize_metrics_data_keeps_structured_usage_per_item() -> None:
    pytest.importorskip("pipecat.metrics.metrics")
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData

    from noveum_trace.integrations.pipecat.pipecat_utils import normalize_metrics_data

    frame = MetricsFrame(
        data=[
            LLMUsageMetricsData(
                processor="llm-a",
                model="model-a",
                value=LLMTokenUsage(
                    prompt_tokens=1, completion_tokens=2, total_tokens=3
                ),
            ),
            LLMUsageMetricsData(
                processor="llm-b",
                model="model-b",
                value=LLMTokenUsage(
                    prompt_tokens=4, completion_tokens=5, total_tokens=9
                ),
            ),
        ]
    )
    records = normalize_metrics_data(frame)
    assert [record.value for record in records] == [
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
    ]
    assert [record.processor for record in records] == ["llm-a", "llm-b"]


def test_normalize_metrics_data_preserves_unknown_items() -> None:
    pytest.importorskip("pipecat.metrics.metrics")
    from noveum_trace.integrations.pipecat.pipecat_utils import normalize_metrics_data

    class _FutureMetricsData:
        processor = "future-processor"
        model = "future-model"
        value = 0

    frame = MagicMock()
    frame.data = [_FutureMetricsData()]
    [record] = normalize_metrics_data(frame)
    assert record.native_class == "_FutureMetricsData"
    assert record.family == "unknown"
    assert record.value == 0
    assert record.processor == "future-processor"
    assert record.model == "future-model"


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


def test_thinking_settings_provider_default_disabled_flash_25() -> None:
    """D3: gemini-2.5-flash with no app ThinkingConfig → Pipecat disables thinking
    (thinking_budget=0). We mirror that so reasoning_tokens=0 is interpretable."""
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_service_settings

    class _Settings:
        model = "gemini-2.5-flash"
        thinking = None

    proc = MagicMock()
    proc._settings = _Settings()
    out = extract_service_settings(proc)
    assert out["thinking_budget"] == 0
    assert out["thinking_enabled"] is False
    assert out["thinking_config_source"] == "provider_default"


def test_thinking_settings_provider_default_minimal_flash_3() -> None:
    """D3: gemini-3*flash with no app config → Pipecat default thinking_level=minimal."""
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_service_settings

    class _Settings:
        model = "gemini-3-flash"
        thinking = None

    proc = MagicMock()
    proc._settings = _Settings()
    out = extract_service_settings(proc)
    assert out["thinking_level"] == "minimal"
    assert out["thinking_enabled"] is True
    assert out["thinking_config_source"] == "provider_default"


def test_thinking_settings_app_configured() -> None:
    """D3: an explicit app ThinkingConfig is captured verbatim, source='app'."""
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_service_settings

    class _Thinking:
        thinking_budget = 2048
        thinking_level = None
        include_thoughts = True

    class _Settings:
        model = "gemini-2.5-pro"
        thinking = _Thinking()

    proc = MagicMock()
    proc._settings = _Settings()
    out = extract_service_settings(proc)
    assert out["thinking_budget"] == 2048
    assert out["include_thoughts"] is True
    assert out["thinking_enabled"] is True
    assert out["thinking_config_source"] == "app"


def test_thinking_settings_absent_for_non_google() -> None:
    """D3: services without a ``thinking`` attribute emit no thinking keys."""
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_service_settings

    class _Settings:
        model = "gpt-4o"

    proc = MagicMock()
    proc._settings = _Settings()
    out = extract_service_settings(proc)
    assert not any(k.startswith("thinking") for k in out)


def test_extract_stt_result_data_words_and_request_id() -> None:
    # §4: per-word timing/diarization array + request id from a Deepgram-shaped result.
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_stt_result_data

    word = types.SimpleNamespace(
        word="hi", start=0.0, end=0.2, confidence=0.9, punctuated_word="Hi", speaker=0
    )
    result = types.SimpleNamespace(
        channel=types.SimpleNamespace(
            alternatives=[types.SimpleNamespace(words=[word])]
        ),
        metadata=types.SimpleNamespace(request_id="req-1"),
    )
    out = extract_stt_result_data(result)
    assert out["words"] == [
        {
            "word": "hi",
            "start": 0.0,
            "end": 0.2,
            "confidence": 0.9,
            "punctuated_word": "Hi",
            "speaker": 0,
        }
    ]
    assert out["request_id"] == "req-1"


def test_extract_stt_result_data_dict_shaped_and_truncation() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_stt_result_data

    words = [{"word": f"w{i}", "start": float(i)} for i in range(5)]
    result = {"channel": {"alternatives": [{"words": words}]}, "metadata": {}}
    out = extract_stt_result_data(result, max_words=3)
    assert len(out["words"]) == 3
    assert out["words_truncated"] is True
    assert "request_id" not in out


def test_extract_stt_result_data_empty() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import extract_stt_result_data

    assert extract_stt_result_data(None) == {}
    assert extract_stt_result_data(object()) == {}


def test_derive_provider_from_registry_and_class() -> None:
    from noveum_trace.integrations.pipecat.pipecat_utils import derive_provider

    class GoogleLLMService:
        pass

    class DeepgramSTTService:
        pass

    class ElevenLabsTTSService:
        pass

    # Model registry is authoritative when a known model is given.
    assert derive_provider(GoogleLLMService(), "gemini-2.5-flash") == "google"
    # Class-name fallback (role + Service suffix stripped, lowercased).
    assert derive_provider(DeepgramSTTService()) == "deepgram"
    assert derive_provider(ElevenLabsTTSService()) == "elevenlabs"
    assert derive_provider(GoogleLLMService()) == "google"
    assert derive_provider(None) is None


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
            "input_audio_tokens": 6,
            "output_audio_tokens": 7,
            "cache_read_input_audio_tokens": 8,
        }
    )
    out = _llm_token_usage_to_dict(u)
    assert out["total_tokens"] == 3
    assert out["reasoning_tokens"] == 5
    assert out["input_audio_tokens"] == 6
    assert out["output_audio_tokens"] == 7
    assert out["cache_read_input_audio_tokens"] == 8


def test_normalize_optional_ttfa_and_stt_usage(monkeypatch) -> None:
    pytest.importorskip("pipecat.metrics.metrics")
    from pipecat.metrics import metrics

    from noveum_trace.integrations.pipecat.pipecat_utils import (
        normalize_metrics_data,
    )

    class _TTFA:
        processor = "voice"
        model = "model"
        ttfa = 0.4
        ttfb = 0.25
        leading_silence = 0.15

    class _STTUsage:
        processor = "speech"
        model = None
        value = types.SimpleNamespace(audio_seconds=1.5)

    monkeypatch.setattr(metrics, "TTFAMetricsData", _TTFA, raising=False)
    monkeypatch.setattr(metrics, "STTUsageMetricsData", _STTUsage, raising=False)
    records = normalize_metrics_data(types.SimpleNamespace(data=[_TTFA(), _STTUsage()]))

    assert records[0].family == "ttfa"
    assert records[0].value == {
        "ttfa": 0.4,
        "ttfb": 0.25,
        "leading_silence": 0.15,
    }
    assert records[1].family == "stt_usage"
    assert records[1].value == {"audio_seconds": 1.5}


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
