"""
Value-asserting regression tests for the pipecat Utils / constants / error-capture
subsystem (PIPECAT_TEST_PLAN.md §J, UC-1..11).

Covers:
  - ``pipecat_utils.calculate_llm_cost``       — estimate_cost key remap (UC-1)
  - ``pipecat_utils.extract_service_settings`` — NOT_GIVEN / falsy drop, precedence,
                                                 language enum/plain (UC-2, UC-3)
  - ``pipecat_utils.extract_metrics_data``     — plain TurnMetricsData routing (UC-4)
  - tools serialization through real ToolsSchema/FunctionSchema (UC-5, UC-11)
  - ``pipecat_utils.upload_audio_frames``      — export metadata + WAV header (UC-6)
  - ``_error_capture._ErrorCaptureMixin``      — error status/event propagation,
                                                 trace-event drop, gating + level filter
                                                 (UC-7, UC-8, UC-9)
  - ``pipecat_constants.MAX_STT_AUDIO_FRAMES`` — value + bounded-append consumption (UC-10)

These drive real ``Trace``/``Span`` objects and real pipecat payloads (skipping
cleanly when an optional dependency is absent) and assert names, values, status
strings and event payloads — never "a mock was called".
"""

from __future__ import annotations

import json
import types
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pipecat.frames.frames")

from noveum_trace.core.trace import Trace  # noqa: E402
from noveum_trace.integrations.pipecat import pipecat_constants as C  # noqa: E402
from noveum_trace.integrations.pipecat import pipecat_utils as U  # noqa: E402
from noveum_trace.integrations.pipecat.pipecat_observer import (  # noqa: E402
    NoveumTraceObserver,
)

_EC = "noveum_trace.utils.llm_utils.estimate_cost"


# --------------------------------------------------------------------------- #
# UC-1 — calculate_llm_cost remaps estimate_cost keys                          #
# --------------------------------------------------------------------------- #
def test_uc1_calculate_llm_cost_remaps_estimate_cost_keys() -> None:
    # Guards: the *_cost -> input/output/total key remap (was a tautology test).
    with patch(
        _EC,
        return_value={
            "input_cost": 0.01,
            "output_cost": 0.02,
            "total_cost": 0.03,
            "currency": "USD",
        },
    ) as mock_estimate:
        out = U.calculate_llm_cost("gpt-4o", 100, 50)

    assert out == {"input": 0.01, "output": 0.02, "total": 0.03, "currency": "USD"}
    # input_tokens / output_tokens forwarded as keyword args, in that order.
    mock_estimate.assert_called_once_with("gpt-4o", input_tokens=100, output_tokens=50)


def test_uc1_calculate_llm_cost_returns_empty_dict_on_error() -> None:
    # Guards: estimate_cost raising -> {} (caller treats {} as "no cost"), never bubbles.
    with patch(_EC, side_effect=RuntimeError("boom")):
        assert U.calculate_llm_cost("gpt-4o", 1, 1) == {}


# --------------------------------------------------------------------------- #
# UC-2 — extract_service_settings drops NOT_GIVEN sentinels + falsy fields     #
# --------------------------------------------------------------------------- #
def test_uc2_extract_service_settings_drops_not_given_and_falsy() -> None:
    # Guards: NOT_GIVEN sentinel + empty-string fields leaking into spans as values.
    openai = pytest.importorskip("openai")

    raw = types.SimpleNamespace(
        model="gpt-4o",
        voice="",  # falsy -> dropped
        temperature=0.5,
        top_p=openai.NOT_GIVEN,  # sentinel -> dropped
    )
    out = U.extract_service_settings(types.SimpleNamespace(_settings=raw))

    assert out["model"] == "gpt-4o"
    assert out["temperature"] == 0.5
    assert "top_p" not in out
    assert "voice" not in out


# --------------------------------------------------------------------------- #
# UC-3 — system_instruction precedence + language enum/plain                   #
# --------------------------------------------------------------------------- #
def test_uc3_system_instruction_first_wins() -> None:
    # Guards: system_instruction wins over system_prompt (first-wins loop order).
    raw = types.SimpleNamespace(
        model="m",
        system_instruction="be brief",
        system_prompt="other",
    )
    out = U.extract_service_settings(types.SimpleNamespace(_settings=raw))
    assert out["system_instruction"] == "be brief"


def test_uc3_language_enum_uses_value() -> None:
    # Guards: language objects with .value serialise to the enum value, not repr.
    class _Lang:
        value = "en-US"

    out = U.extract_service_settings(
        types.SimpleNamespace(_settings=types.SimpleNamespace(language=_Lang()))
    )
    assert out["language"] == "en-US"


def test_uc3_language_plain_string_passthrough() -> None:
    # Guards: a plain-string language (no .value) is passed through unchanged.
    out = U.extract_service_settings(
        types.SimpleNamespace(_settings=types.SimpleNamespace(language="fr"))
    )
    assert out["language"] == "fr"


# --------------------------------------------------------------------------- #
# UC-4 — extract_metrics_data routes plain TurnMetricsData                     #
# --------------------------------------------------------------------------- #
def test_uc4_extract_metrics_data_plain_turn_metrics() -> None:
    # Guards: plain-Turn branch keys + SmartTurn-vs-Turn elif order (no SmartTurn keys).
    pytest.importorskip("pipecat.metrics.metrics")
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import TurnMetricsData

    tmd = TurnMetricsData(
        processor="TurnTracker",
        is_complete=True,
        probability=0.8,
        e2e_processing_time_ms=12.0,
    )
    out = U.extract_metrics_data(MetricsFrame(data=[tmd]))

    assert out == {
        "turn_eou_is_complete": True,
        "turn_eou_confidence": 0.8,
        "turn_eou_processing_time_ms": 12.0,
    }
    # SmartTurn-only keys must NOT appear for a plain TurnMetricsData.
    assert "turn_eou_inference_ms" not in out
    assert "turn_eou_server_total_ms" not in out


# --------------------------------------------------------------------------- #
# UC-5 — tools serialization through real ToolsSchema / FunctionSchema         #
# --------------------------------------------------------------------------- #
def test_uc5_serialize_tools_field_real_tools_schema() -> None:
    # Guards: the real .standard_tools -> to_default_dict() serialization pipeline.
    pytest.importorskip("pipecat.adapters.schemas.tools_schema")
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    fs = FunctionSchema(
        name="get_weather",
        description="Get weather",
        properties={"city": {"type": "string"}},
        required=["city"],
    )
    ts = ToolsSchema(standard_tools=[fs])

    parsed = json.loads(U.serialize_tools_field(ts))
    assert parsed == [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


def test_uc5_extract_llm_context_data_messages_and_real_tools() -> None:
    # Guards: context messages + real ToolsSchema both serialise to expected JSON.
    pytest.importorskip("pipecat.adapters.schemas.tools_schema")
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    fs = FunctionSchema(
        name="get_weather",
        description="Get weather",
        properties={"city": {"type": "string"}},
        required=["city"],
    )
    ctx = types.SimpleNamespace(
        get_messages=lambda: [{"role": "user", "content": "hi"}],
        tools=ToolsSchema(standard_tools=[fs]),
    )
    out = U.extract_llm_context_data(ctx)

    assert json.loads(out["messages"]) == [{"role": "user", "content": "hi"}]
    assert json.loads(out["tools"]) == [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


# --------------------------------------------------------------------------- #
# UC-11 — _resolve_tools_to_list returns None (not []) for empty containers    #
# --------------------------------------------------------------------------- #
def test_uc11_resolve_tools_to_list_empty_returns_none() -> None:
    # Guards: empty tools containers serialising to '[]' and polluting llm.tools.
    pytest.importorskip("pipecat.adapters.schemas.tools_schema")
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    assert U._resolve_tools_to_list([]) is None
    assert U._resolve_tools_to_list(ToolsSchema(standard_tools=[])) is None
    # Legacy / other container exposing an empty .tools list.
    assert U._resolve_tools_to_list(types.SimpleNamespace(tools=[])) is None


# --------------------------------------------------------------------------- #
# UC-6 — upload_audio_frames metadata + WAV header values                     #
# --------------------------------------------------------------------------- #
def test_uc6_upload_audio_frames_metadata_and_wav_header() -> None:
    # Guards: export metadata contract (duration_ms/format/type) + WAV framing + verbatim ids.
    from pipecat.frames.frames import AudioRawFrame

    # 160 samples mono 16 kHz -> 160 / 16000 * 1000 = 10.0 ms.
    frame = AudioRawFrame(audio=b"\x00\x01" * 160, sample_rate=16000, num_channels=1)
    client = MagicMock()

    ok = U.upload_audio_frames(
        [frame], "uuid-1", "stt", "trace-1", "span-1", client=client
    )
    assert ok is True

    kwargs = client.export_audio.call_args.kwargs
    assert kwargs["metadata"] == {
        "duration_ms": 10.0,
        "format": "wav",
        "type": "stt",
    }
    assert kwargs["audio_data"].startswith(b"RIFF")
    assert kwargs["trace_id"] == "trace-1"
    assert kwargs["span_id"] == "span-1"
    assert kwargs["audio_uuid"] == "uuid-1"


# --------------------------------------------------------------------------- #
# Error-capture helpers                                                        #
# --------------------------------------------------------------------------- #
def _error_observer() -> tuple[NoveumTraceObserver, Trace, object, object, object]:
    """Observer with capture_errors=True wired to a real Trace + turn/llm/tts spans."""
    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    llm = trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)
    tts = trace.create_span(name="pipecat.tts", parent_span_id=turn.span_id)

    obs = NoveumTraceObserver(capture_errors=True)
    obs._trace = trace
    obs._current_turn_span = turn
    obs._active_llm_span = llm
    obs._active_tts_span = tts
    return obs, trace, turn, llm, tts


def _error_frame(message: str = "boom"):
    from pipecat.frames.frames import ErrorFrame

    return ErrorFrame(error=message)


# --------------------------------------------------------------------------- #
# UC-7 — _handle_error sets status on active spans + one turn event            #
# --------------------------------------------------------------------------- #
async def test_uc7_handle_error_marks_spans_and_emits_turn_event() -> None:
    # Guards: error status propagation to active llm/tts/turn spans + the single turn event.
    obs, trace, turn, llm, tts = _error_observer()

    await obs._handle_error(types.SimpleNamespace(frame=_error_frame("boom")))

    for span in (llm, tts, turn):
        assert span.attributes["pipecat_span_status"] == "error"
        assert span.attributes["pipecat_span_status_message"] == "boom"

    error_events = [e for e in turn.events if e.name == "pipecat.error"]
    assert len(error_events) == 1
    evt = error_events[0]
    assert evt.attributes["error.message"] == "boom"
    assert evt.attributes["error.type"] == "ErrorFrame"


# --------------------------------------------------------------------------- #
# UC-8 — _handle_error annotates trace attribute (green) but drops the event   #
# --------------------------------------------------------------------------- #
async def test_uc8_handle_error_sets_trace_attributes() -> None:
    # Guards (green): the trace-level status attributes ARE written so the dashboard
    # surfaces the error even when no child span is open.
    obs, trace, turn, llm, tts = _error_observer()

    await obs._handle_error(types.SimpleNamespace(frame=_error_frame("boom")))

    assert trace.attributes["pipecat_span_status"] == "error"
    assert trace.attributes["pipecat_span_status_message"] == "boom"


async def test_uc8_handle_error_records_trace_level_event() -> None:
    # Guards: the trace-level pipecat.error event now persists (Trace.events).
    # Previously dropped silently because Trace had no .events (the append
    # AttributeError'd into a bare except); a regression that drops trace events
    # — or reverts Trace.events — makes this red. Was UC-8 xfail; the fix landed.
    obs, trace, turn, llm, tts = _error_observer()

    await obs._handle_error(types.SimpleNamespace(frame=_error_frame("boom")))

    err_events = [e for e in trace.events if e.name == "pipecat.error"]
    assert len(err_events) == 1
    assert err_events[0].attributes["error.message"] == "boom"
    # ...and it reaches the export payload, not just the in-memory object.
    serialized = trace.to_dict()["events"]
    assert any(e["name"] == "pipecat.error" for e in serialized)


# --------------------------------------------------------------------------- #
# UC-9 — gating + level filtering                                             #
# --------------------------------------------------------------------------- #
async def test_uc9_handle_error_noop_when_capture_disabled() -> None:
    # Guards: capture_errors=False opt-out gate — nothing on spans/turn/trace.
    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn")
    llm = trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)

    obs = NoveumTraceObserver(capture_errors=False)
    obs._trace = trace
    obs._current_turn_span = turn
    obs._active_llm_span = llm

    await obs._handle_error(types.SimpleNamespace(frame=_error_frame("boom")))

    assert "pipecat_span_status" not in llm.attributes
    assert "pipecat_span_status" not in turn.attributes
    assert "pipecat_span_status" not in trace.attributes
    assert not turn.events


async def test_uc9_system_log_records_only_warning_error_critical() -> None:
    # Guards: capture_system_logs opt-in gate + warning/error/critical-only level filter.
    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn")

    # Gate closed by default: nothing recorded.
    closed = NoveumTraceObserver(capture_system_logs=False)
    closed._trace = trace
    closed._current_turn_span = turn
    await closed._handle_system_log(
        types.SimpleNamespace(
            frame=types.SimpleNamespace(level="warning", message="careful")
        )
    )
    assert not turn.events

    # Gate open: info dropped, empty message dropped, warning recorded once.
    obs = NoveumTraceObserver(capture_system_logs=True)
    obs._trace = trace
    obs._current_turn_span = turn

    await obs._handle_system_log(
        types.SimpleNamespace(
            frame=types.SimpleNamespace(level="info", message="noise")
        )
    )
    await obs._handle_system_log(
        types.SimpleNamespace(frame=types.SimpleNamespace(level="warning", message=""))
    )
    await obs._handle_system_log(
        types.SimpleNamespace(
            frame=types.SimpleNamespace(level="warning", message="careful")
        )
    )

    log_events = [e for e in turn.events if e.name == "pipecat.system_log"]
    assert len(log_events) == 1
    assert log_events[0].attributes["log.level"] == "warning"
    assert log_events[0].attributes["log.message"] == "careful"


# --------------------------------------------------------------------------- #
# UC-10 — MAX_STT_AUDIO_FRAMES value + bounded-append consumption              #
# --------------------------------------------------------------------------- #
def test_uc10_max_stt_audio_frames_value_and_overflow_trim() -> None:
    # Guards: the STT audio memory cap value + that overflow drops oldest frames.
    assert C.MAX_STT_AUDIO_FRAMES == 3000

    obs = NoveumTraceObserver()
    buffer: list[int] = []
    for i in range(C.MAX_STT_AUDIO_FRAMES + 5):
        obs._bounded_append_stt_frame(buffer, i)

    # Capped at the constant; the 5 oldest (0..4) were trimmed, newest retained.
    assert len(buffer) == C.MAX_STT_AUDIO_FRAMES
    assert buffer[0] == 5
    assert buffer[-1] == C.MAX_STT_AUDIO_FRAMES + 4
