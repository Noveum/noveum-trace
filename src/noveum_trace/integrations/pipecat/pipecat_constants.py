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
# Busy-turn order of magnitude: ~100 LLM text frames + ~500 TTS audio frames
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
