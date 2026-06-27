"""
Unit tests for the ``livekit_utils.py`` helper functions (Section A of
tests/integration/livekit/LIVEKIT_TEST_PLAN.md).

Before this file only ``upload_audio_frames`` was directly tested; ~18 helper
functions — including the public ``extract_job_context`` and the heavily-branchy
``create_span_attributes`` / ``serialize_*`` — were not. These are value-asserting
unit tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents.stt import SpeechData  # noqa: E402

from noveum_trace.integrations.livekit import livekit_utils as U  # noqa: E402


# --------------------------------------------------------------------------- #
# A1 — create_constants_metadata
# --------------------------------------------------------------------------- #
def test_create_constants_metadata_structure_and_values():
    md = U.create_constants_metadata()
    defaults = md["config"]["defaults"]
    # spot-check one from each group + a couple of exact values
    for key in (
        "STT_TRANSCRIPT_DEFAULT_VALUE",
        "TTS_INPUT_TEXT_DEFAULT_VALUE",
        "AUDIO_DURATION_MS_DEFAULT_VALUE",
        "SYSTEM_PROMPT_MAX_WAIT_SECONDS",
        "MAX_CONVERSATION_HISTORY",
    ):
        assert key in defaults
    assert defaults["STT_TRANSCRIPT_DEFAULT_VALUE"] == ""
    assert defaults["MAX_CONVERSATION_HISTORY"] == 1000
    assert defaults["SYSTEM_PROMPT_MAX_WAIT_SECONDS"] == 5.0


# --------------------------------------------------------------------------- #
# A2 — calculate_audio_duration_ms
# --------------------------------------------------------------------------- #
def test_calculate_audio_duration_ms():
    assert U.calculate_audio_duration_ms([]) == 0.0
    frames = [SimpleNamespace(duration=0.5), SimpleNamespace(duration=0.5)]
    assert U.calculate_audio_duration_ms(frames) == 1000.0


# --------------------------------------------------------------------------- #
# A3/A4 — save_audio_frames / save_audio_buffer
# --------------------------------------------------------------------------- #
def test_save_audio_frames_empty_writes_empty_file(tmp_path):
    out = tmp_path / "sub" / "empty.wav"
    U.save_audio_frames([], out)
    assert out.exists() and out.read_bytes() == b""


def test_save_audio_frames_writes_wav_bytes(tmp_path):
    out = tmp_path / "audio.wav"
    combined = Mock()
    combined.to_wav_bytes.return_value = b"WAVDATA"
    with patch.object(U, "rtc", create=True) as rtc:
        rtc.combine_audio_frames.return_value = combined
        U.save_audio_frames([Mock(), Mock()], out)
    assert out.read_bytes() == b"WAVDATA"


def test_save_audio_frames_noop_when_livekit_unavailable(tmp_path):
    out = tmp_path / "x.wav"
    with patch.object(U, "LIVEKIT_AVAILABLE", False):
        U.save_audio_frames([Mock()], out)
    assert not out.exists()


def test_save_audio_buffer_delegates_to_save_audio_frames(tmp_path):
    out = tmp_path / "b.wav"
    buffer = [Mock(), Mock()]
    with patch.object(U, "save_audio_frames") as save:
        U.save_audio_buffer(buffer, out)
    save.assert_called_once_with(buffer, out)


# --------------------------------------------------------------------------- #
# A5 — upload_audio_file
# --------------------------------------------------------------------------- #
def test_upload_audio_file_missing_file_returns_false(tmp_path):
    assert U.upload_audio_file(tmp_path / "nope.ogg", "u", "stt", "t", "s") is False


def test_upload_audio_file_empty_file_returns_false(tmp_path):
    p = tmp_path / "empty.ogg"
    p.write_bytes(b"")
    assert U.upload_audio_file(p, "u", "stt", "t", "s") is False


def test_upload_audio_file_no_client_returns_false(tmp_path):
    p = tmp_path / "a.ogg"
    p.write_bytes(b"data")
    with patch("noveum_trace.get_client", return_value=None):
        assert U.upload_audio_file(p, "u", "stt", "t", "s") is False


def test_upload_audio_file_success_exports_with_metadata(
    tmp_path, client_with_mocked_transport
):
    client = client_with_mocked_transport
    p = tmp_path / "rec.ogg"
    p.write_bytes(b"oggbytes")

    ok = U.upload_audio_file(
        p, "uuid-1", "conversation", "trace-1", "span-1", content_type="audio/ogg"
    )

    assert ok is True
    _, kwargs = client.transport.export_audio.call_args
    assert kwargs["trace_id"] == "trace-1"
    assert kwargs["span_id"] == "span-1"
    assert kwargs["audio_uuid"] == "uuid-1"
    meta = kwargs["metadata"]
    assert meta["format"] == "ogg"  # from .ogg suffix
    assert meta["type"] == "conversation"
    assert meta["content_type"] == "audio/ogg"
    assert meta["file_size_bytes"] == len(b"oggbytes")


# --------------------------------------------------------------------------- #
# A6 — get_conversation_history_from_session
# --------------------------------------------------------------------------- #
def test_history_no_attr_returns_empty():
    assert U.get_conversation_history_from_session(SimpleNamespace()) == {}


def test_history_none_returns_empty():
    assert U.get_conversation_history_from_session(SimpleNamespace(history=None)) == {}


def test_history_prefers_to_dict():
    session = SimpleNamespace(
        history=SimpleNamespace(to_dict=lambda: {"items": [1, 2]})
    )
    assert U.get_conversation_history_from_session(session) == {"items": [1, 2]}


def test_history_fallback_uses_model_dump():
    item = SimpleNamespace(model_dump=lambda **kw: {"role": "user"})
    session = SimpleNamespace(history=SimpleNamespace(items=[item]))
    assert U.get_conversation_history_from_session(session) == {
        "items": [{"role": "user"}]
    }


# --------------------------------------------------------------------------- #
# A7/A9 — recorder path resolution
# --------------------------------------------------------------------------- #
def test_get_recorder_audio_path_variants(tmp_path):
    assert U.get_recorder_audio_path(SimpleNamespace()) is None
    assert (
        U.get_recorder_audio_path(SimpleNamespace(_recorder_io=SimpleNamespace()))
        is None
    )
    # nonexistent path -> None
    s = SimpleNamespace(_recorder_io=SimpleNamespace(output_path="/no/such/x.ogg"))
    assert U.get_recorder_audio_path(s) is None
    # existing str path coerced to Path
    f = tmp_path / "rec.ogg"
    f.write_bytes(b"x")
    s2 = SimpleNamespace(_recorder_io=SimpleNamespace(output_path=str(f)))
    assert U.get_recorder_audio_path(s2) == f


def test_resolve_recorder_audio_path_falls_back_to_output_path(tmp_path):
    # exists -> returned
    f = tmp_path / "ok.ogg"
    f.write_bytes(b"x")
    s = SimpleNamespace(_recorder_io=SimpleNamespace(output_path=str(f)))
    assert U.resolve_recorder_audio_path(s) == f
    # not-yet-existing -> still returns the configured Path
    s2 = SimpleNamespace(_recorder_io=SimpleNamespace(output_path="/tmp/pending.ogg"))
    assert U.resolve_recorder_audio_path(s2) == Path("/tmp/pending.ogg")
    # nothing configured -> None
    assert U.resolve_recorder_audio_path(SimpleNamespace()) is None


# --------------------------------------------------------------------------- #
# A8 — finalize_recorder_io
# --------------------------------------------------------------------------- #
async def test_finalize_recorder_io_none_is_noop():
    await U.finalize_recorder_io(None)  # must not raise


async def test_finalize_recorder_io_skips_when_closed():
    rec = SimpleNamespace(closed=True, aclose=AsyncMock())
    await U.finalize_recorder_io(rec)
    rec.aclose.assert_not_called()


async def test_finalize_recorder_io_closes_when_open():
    rec = SimpleNamespace(closed=False, aclose=AsyncMock())
    await U.finalize_recorder_io(rec)
    rec.aclose.assert_awaited_once()


async def test_finalize_recorder_io_swallows_errors():
    rec = SimpleNamespace(closed=False, aclose=AsyncMock(side_effect=RuntimeError("x")))
    await U.finalize_recorder_io(rec)  # must not raise


# --------------------------------------------------------------------------- #
# A10 — wait_for_audio_path
# --------------------------------------------------------------------------- #
async def test_wait_for_audio_path_none_returns_false():
    assert await U.wait_for_audio_path(None) is False


async def test_wait_for_audio_path_present_returns_true(tmp_path):
    f = tmp_path / "a.ogg"
    f.write_bytes(b"data")
    assert await U.wait_for_audio_path(f) is True


async def test_wait_for_audio_path_absent_times_out(tmp_path):
    f = tmp_path / "never.ogg"
    assert (
        await U.wait_for_audio_path(f, max_wait_seconds=0.2, poll_interval=0.05)
        is False
    )


# --------------------------------------------------------------------------- #
# A11 — get_session_system_prompt
# --------------------------------------------------------------------------- #
def test_get_session_system_prompt_from_instructions():
    agent = SimpleNamespace(instructions="SP")
    session = SimpleNamespace(agent_activity=SimpleNamespace(_agent=agent))
    assert U.get_session_system_prompt(session) == "SP"


def test_get_session_system_prompt_fallback_private():
    agent = SimpleNamespace(_instructions="SP2")  # no public instructions
    session = SimpleNamespace(agent_activity=SimpleNamespace(_agent=agent))
    assert U.get_session_system_prompt(session) == "SP2"


def test_get_session_system_prompt_none_when_unavailable():
    assert U.get_session_system_prompt(SimpleNamespace()) is None


# --------------------------------------------------------------------------- #
# A12/A13 — audio dir + filename helpers
# --------------------------------------------------------------------------- #
def test_ensure_audio_directory(tmp_path):
    d = U.ensure_audio_directory("sess-1", base_dir=tmp_path)
    assert d == tmp_path / "sess-1"
    assert d.is_dir()


def test_generate_audio_filename():
    assert (
        U.generate_audio_filename("stt", 1, timestamp=1700000000000)
        == "stt_0001_1700000000000.wav"
    )
    assert re.fullmatch(r"tts_0042_\d+\.wav", U.generate_audio_filename("tts", 42))


# --------------------------------------------------------------------------- #
# A14/A15 — mock detection + safe stringification
# --------------------------------------------------------------------------- #
def test_is_mock_object():
    assert U._is_mock_object(Mock()) is True
    assert U._is_mock_object(AsyncMock()) is True
    assert U._is_mock_object("hello") is False
    assert U._is_mock_object(123) is False


async def test_safe_str_handles_none_mock_value_and_coroutine():
    assert await U._safe_str(None) == "unknown"
    assert await U._safe_str(None, default="N/A") == "N/A"
    assert await U._safe_str("value") == "value"
    assert await U._safe_str(Mock()) == "unknown"

    async def coro():
        return "awaited"

    assert await U._safe_str(coro()) == "awaited"


# --------------------------------------------------------------------------- #
# A16 — extract_job_context (public API)
# --------------------------------------------------------------------------- #
async def test_extract_job_context_full():
    ctx = SimpleNamespace(
        job=SimpleNamespace(id="job1", room=SimpleNamespace(sid="rs", name="rn")),
        room=SimpleNamespace(name="room1", sid="rsid"),
        agent=SimpleNamespace(id="ag1"),
        worker_id="w1",
        participant=SimpleNamespace(identity="user1", sid="psid"),
    )
    out = await U.extract_job_context(ctx)
    assert out == {
        "job_id": "job1",
        "job_room_sid": "rs",
        "job_room_name": "rn",
        "room_name": "room1",
        "room_sid": "rsid",
        "agent_id": "ag1",
        "worker_id": "w1",
        "participant_identity": "user1",
        "participant_sid": "psid",
    }


async def test_extract_job_context_filters_mocks_and_empty():
    assert await U.extract_job_context(SimpleNamespace()) == {}
    # a mock job is filtered out entirely
    assert await U.extract_job_context(SimpleNamespace(job=Mock())) == {}


# --------------------------------------------------------------------------- #
# A17 — create_span_attributes job-prefix normalization
# --------------------------------------------------------------------------- #
def test_create_span_attributes_normalizes_job_prefix():
    attrs = U.create_span_attributes(
        provider="deepgram",
        model="nova-2",
        operation_type="stt",
        audio_uuid="u",
        audio_duration_ms=100.0,
        job_context={"job.already": "a", "job_id": "b", "room_name": "c"},
        extra="x",
    )
    assert attrs["stt.provider"] == "deepgram"
    assert attrs["stt.model"] == "nova-2"
    assert attrs["stt.audio_uuid"] == "u"
    assert attrs["stt.audio_duration_ms"] == 100.0
    assert attrs["job.already"] == "a"  # dotted prefix kept verbatim
    assert attrs["job.id"] == "b"  # job_ -> job.
    assert attrs["job.room_name"] == "c"  # bare -> job. prefixed
    assert "metadata" in attrs
    assert attrs["extra"] == "x"


# --------------------------------------------------------------------------- #
# A18/A19 — tool extraction + serialization
# --------------------------------------------------------------------------- #
def test_extract_available_tools_from_tools_attr():
    tool = SimpleNamespace(
        name="get_weather", description="desc", args_schema={"type": "object"}
    )
    tools = U.extract_available_tools(SimpleNamespace(tools=[tool]))
    assert tools == [
        {
            "name": "get_weather",
            "description": "desc",
            "args_schema": {"type": "object"},
        }
    ]


def test_extract_available_tools_name_and_doc_fallbacks():
    def my_tool():
        """A docstring description."""

    tools = U.extract_available_tools(SimpleNamespace(_tools=[my_tool]))
    assert tools[0]["name"] == "my_tool"
    assert tools[0]["description"] == "A docstring description."


def test_extract_available_tools_empty():
    assert U.extract_available_tools(SimpleNamespace()) == []
    assert U.extract_available_tools(SimpleNamespace(tools=[])) == []


def test_serialize_tools_for_attributes():
    assert U.serialize_tools_for_attributes([]) == {}
    tools = [{"name": "a", "description": "da"}, {"name": "b", "description": "db"}]
    attrs = U.serialize_tools_for_attributes(tools)
    assert attrs["llm.available_tools.count"] == 2
    assert attrs["llm.available_tools.names"] == ["a", "b"]
    assert attrs["llm.available_tools.descriptions"] == ["da", "db"]
    assert json.loads(attrs["llm.available_tools.schemas"]) == tools


# --------------------------------------------------------------------------- #
# A20 — serialize_chat_history
# --------------------------------------------------------------------------- #
def test_serialize_chat_history_dict_messages():
    out = U.serialize_chat_history(
        [
            {"role": "user", "content": "hi", "name": "bob"},
            {"role": "assistant", "content": [{"text": "a"}, {"text": "b"}]},
        ]
    )
    assert out[0] == {"role": "user", "content": "hi", "name": "bob"}
    assert out[1] == {"role": "assistant", "content": "a\nb"}


def test_serialize_chat_history_object_messages_with_enum_role():
    msg = SimpleNamespace(role=SimpleNamespace(value="user"), text_content="hello")
    out = U.serialize_chat_history([msg])
    assert out == [{"role": "user", "content": "hello"}]


# --------------------------------------------------------------------------- #
# A21 — serialize_function_calls
# --------------------------------------------------------------------------- #
def test_serialize_function_calls_dict_and_object_args():
    out = U.serialize_function_calls(
        [
            {"name": "f", "arguments": '{"a":1}', "call_id": "c1"},
            SimpleNamespace(name="g", arguments={"a": 1}, call_id="c2"),
            SimpleNamespace(name="h", arguments='{"b":2}', call_id="c3"),
        ]
    )
    assert out[0] == {"name": "f", "arguments": '{"a":1}', "call_id": "c1"}
    # non-string arguments get JSON-encoded
    assert json.loads(out[1]["arguments"]) == {"a": 1}
    assert out[1]["call_id"] == "c2"
    # already-string arguments pass through verbatim
    assert out[2]["arguments"] == '{"b":2}'


# --------------------------------------------------------------------------- #
# A23 — _serialize_chat_items classification (untested elsewhere)
# --------------------------------------------------------------------------- #
def test_serialize_chat_items_empty():
    assert U._serialize_chat_items([]) == {}


def test_serialize_chat_items_classifies_messages_calls_outputs():
    items = [
        SimpleNamespace(type="message", role="user", content="hi", interrupted=False),
        SimpleNamespace(type="function_call", name="f", arguments='{"a":1}'),
        SimpleNamespace(
            type="function_call_output", name="f", output="ok", is_error=False
        ),
    ]
    out = U._serialize_chat_items(items)
    assert out["speech.chat_items.count"] == 3
    assert out["speech.messages"] == [
        {"role": "user", "content": "hi", "interrupted": False}
    ]
    assert out["speech.function_calls"] == [{"name": "f", "arguments": '{"a":1}'}]
    assert out["speech.function_outputs"] == [
        {"name": "f", "output": "ok", "is_error": False}
    ]


def test_serialize_chat_items_infers_type_and_joins_list_content():
    item = SimpleNamespace(role="assistant", content=[{"text": "a"}, {"text": "b"}])
    out = U._serialize_chat_items([item])
    assert out["speech.messages"][0]["content"] == "a\nb"


# --------------------------------------------------------------------------- #
# A22 — _serialize_event_data on a REAL LiveKit dataclass (catches drift)
# --------------------------------------------------------------------------- #
def test_serialize_event_data_real_speechdata():
    data = SpeechData(language="en", text="hi there", confidence=0.9)
    out = U._serialize_event_data(data, "stt")
    assert out["stt.text"] == "hi there"
    assert out["stt.language"] == "en"
    assert out["stt.confidence"] == 0.9
