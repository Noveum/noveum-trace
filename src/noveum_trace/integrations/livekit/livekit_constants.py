"""
Constants for LiveKit integration.
"""

# Default values for transcript and confidence
STT_TRANSCRIPT_DEFAULT_VALUE = ""
STT_CONFIDENCE_DEFAULT_VALUE = 0.0

# Default values for optional STT attributes
STT_LANGUAGE_DEFAULT_VALUE = None
STT_START_TIME_DEFAULT_VALUE = None
STT_END_TIME_DEFAULT_VALUE = None
STT_SPEAKER_ID_DEFAULT_VALUE = None
STT_IS_PRIMARY_SPEAKER_DEFAULT_VALUE = None

# Default values for TTS attributes
TTS_INPUT_TEXT_DEFAULT_VALUE = ""
TTS_SEGMENT_ID_DEFAULT_VALUE = None
TTS_REQUEST_ID_DEFAULT_VALUE = None
TTS_DELTA_TEXT_DEFAULT_VALUE = None
TTS_SAMPLE_RATE_DEFAULT_VALUE = None
TTS_NUM_CHANNELS_DEFAULT_VALUE = None

# Default values for audio duration
AUDIO_DURATION_MS_DEFAULT_VALUE = 0.0

# Timing constants for system prompt updates
SYSTEM_PROMPT_MAX_WAIT_SECONDS = 5.0
SYSTEM_PROMPT_CHECK_INTERVAL_SECONDS = 0.1

# Default values for LLM attributes
LLM_RESPONSE_DEFAULT_VALUE = ""
LLM_MODEL_DEFAULT_VALUE = "unknown"
LLM_PROVIDER_DEFAULT_VALUE = "unknown"
LLM_REQUEST_ID_DEFAULT_VALUE = None
LLM_TEMPERATURE_DEFAULT_VALUE = None
LLM_TOP_P_DEFAULT_VALUE = None
LLM_MAX_TOKENS_DEFAULT_VALUE = None

# Maximum buffer sizes to prevent memory growth and oversized attributes
# Note: Larger values allow more context but increase memory usage and attribute sizes
MAX_CONVERSATION_HISTORY = 1000  # Increased for larger conversation context
MAX_PENDING_FUNCTION_CALLS = 100
MAX_PENDING_FUNCTION_OUTPUTS = 100
