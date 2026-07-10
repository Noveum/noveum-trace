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
# Cap STT PCM frame lists (post-filter and raw pre-filter). ~3k frames ≈ 60s at
# 20 ms/frame for mono 16 kHz; drops oldest on overflow (long utterances / VAD pre-roll).
MAX_STT_AUDIO_FRAMES = 3_000

# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------
# Providers that report reasoning/thinking tokens *disjoint* from
# ``completion_tokens``, so billable output = completion_tokens + reasoning_tokens.
#
# Gemini (both the batch ``GoogleLLMService`` and ``GeminiLiveLLMService``) reports
# ``candidates_token_count`` / ``response_token_count`` exclusive of
# ``thoughts_token_count``.  OpenAI-compatible providers instead report
# ``reasoning_tokens`` as a breakdown *inside* ``completion_tokens``
# (``completion_tokens_details.reasoning_tokens``), so pricing
# ``completion + reasoning`` there would bill the reasoning tokens twice.
#
# Matched case-insensitively against the metrics processor name first, then the
# model name.
REASONING_DISJOINT_PROVIDERS = ("google", "gemini")

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
