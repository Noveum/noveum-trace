"""Unit tests for Pipecat integration constants."""

from __future__ import annotations

from noveum_trace.integrations.pipecat import pipecat_constants as c


def test_span_name_constants() -> None:
    assert c.SPAN_CONVERSATION == "pipecat.conversation"
    assert c.SPAN_TURN == "pipecat.turn"
    assert c.SPAN_STT == "pipecat.stt"
    assert c.SPAN_LLM == "pipecat.llm"
    assert c.SPAN_TTS == "pipecat.tts"


def test_buffer_and_turn_defaults() -> None:
    assert c.MAX_CONVERSATION_HISTORY == 50
    assert c.MAX_TEXT_BUFFER_LENGTH == 50_000
    assert c.MAX_FRAME_DEDUP_HISTORY == 20_000
    assert c.DEFAULT_TURN_END_TIMEOUT_SECS == 2.5


def test_audio_defaults() -> None:
    assert c.AUDIO_DURATION_MS_DEFAULT_VALUE == 0.0
    assert c.AUDIO_SAMPLE_RATE_DEFAULT == 16_000
    assert c.AUDIO_NUM_CHANNELS_DEFAULT == 1
    assert c.AUDIO_BYTES_PER_SAMPLE == 2


def test_attribute_defaults() -> None:
    assert c.STT_TEXT_DEFAULT_VALUE == ""
    assert c.LLM_OUTPUT_DEFAULT_VALUE == ""
    assert c.LLM_MODEL_DEFAULT_VALUE == "unknown"
    assert c.TTS_INPUT_TEXT_DEFAULT_VALUE == ""
