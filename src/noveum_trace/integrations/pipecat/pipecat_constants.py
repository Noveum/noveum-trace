"""
Constants for Pipecat integration.

Mirrors livekit_constants.py structure and naming conventions.
"""

# ---------------------------------------------------------------------------
# Span name constants
# ---------------------------------------------------------------------------
SPAN_CONVERSATION = "pipecat.conversation"
SPAN_TURN = "pipecat.turn"
SPAN_STT = "pipecat.stt"
SPAN_LLM = "pipecat.llm"
SPAN_TTS = "pipecat.tts"

# ---------------------------------------------------------------------------
# Buffer and history limits
# ---------------------------------------------------------------------------
MAX_CONVERSATION_HISTORY = 50
MAX_TEXT_BUFFER_LENGTH = 50_000
# Dedupe observer callbacks when the same frame id is pushed twice (Pipecat pattern).
# Sized to comfortably cover a busy turn: ~100 LLM text frames + ~500 TTS audio frames
# (20 ms each, 10 s utterance) + ~50 misc × ~3 pipeline-processor hops each ≈ 1 950.
# 4 000 gives comfortable headroom; keys are plain ints (~28 B each) so ~112 KB total.
MAX_FRAME_DEDUP_HISTORY = 20_000

# ---------------------------------------------------------------------------
# Turn management defaults
# ---------------------------------------------------------------------------
DEFAULT_TURN_END_TIMEOUT_SECS = 2.5

# ---------------------------------------------------------------------------
# Audio defaults (mirrors livekit_constants.py)
# ---------------------------------------------------------------------------
AUDIO_DURATION_MS_DEFAULT_VALUE = 0.0
AUDIO_SAMPLE_RATE_DEFAULT = 16_000
AUDIO_NUM_CHANNELS_DEFAULT = 1
AUDIO_BYTES_PER_SAMPLE = 2  # 16-bit PCM

# ---------------------------------------------------------------------------
# STT attribute defaults
# ---------------------------------------------------------------------------
STT_TEXT_DEFAULT_VALUE = ""
STT_LANGUAGE_DEFAULT_VALUE = None
STT_USER_ID_DEFAULT_VALUE = None

# ---------------------------------------------------------------------------
# LLM attribute defaults
# ---------------------------------------------------------------------------
LLM_OUTPUT_DEFAULT_VALUE = ""
LLM_MODEL_DEFAULT_VALUE = "unknown"

# ---------------------------------------------------------------------------
# TTS attribute defaults
# ---------------------------------------------------------------------------
TTS_INPUT_TEXT_DEFAULT_VALUE = ""
TTS_VOICE_DEFAULT_VALUE = None
TTS_MODEL_DEFAULT_VALUE = None
