"""
Utility functions for Pipecat integration.

Mirrors livekit_utils.py structure: frame data extraction, audio conversion,
and upload helpers tailored to Pipecat's AudioRawFrame objects.
"""

from __future__ import annotations

import io
import json
import logging
import wave
from typing import Any, Optional

from noveum_trace.integrations.pipecat.pipecat_constants import (
    AUDIO_BYTES_PER_SAMPLE,
    AUDIO_DURATION_MS_DEFAULT_VALUE,
    AUDIO_NUM_CHANNELS_DEFAULT,
    AUDIO_SAMPLE_RATE_DEFAULT,
    MAX_TEXT_BUFFER_LENGTH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service settings extraction
# ---------------------------------------------------------------------------


def extract_service_settings(processor: Any) -> dict[str, Any]:
    """
    Safely read model/voice/language from a processor's _settings object.

    Supports LLMSettings, TTSSettings, and STTSettings from
    pipecat.services.settings (checked via hasattr, no hard import).
    """
    settings: dict[str, Any] = {}
    try:
        raw = getattr(processor, "_settings", None)
        if raw is None:
            return settings

        # Falsy guard is intentional for model/voice/language — an empty string is
        # not a useful value for these fields and should be treated as absent.
        if hasattr(raw, "model") and raw.model:
            settings["model"] = str(raw.model)

        if hasattr(raw, "voice") and raw.voice:
            settings["voice"] = str(raw.voice)

        if hasattr(raw, "language") and raw.language:
            lang = raw.language
            settings["language"] = str(lang.value if hasattr(lang, "value") else lang)

        # `is not None` (not falsy) is deliberate here: an empty-string system prompt
        # is a valid operator intent (explicitly clearing the default instruction) and
        # must be recorded, unlike an absent model name above.
        for attr in ("system_instruction", "system_prompt"):
            val = getattr(raw, attr, None)
            if val is not None:
                settings["system_instruction"] = str(val)
                break

        # LLM numeric settings — capture all that are present and non-None
        for attr in (
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "seed",
        ):
            val = getattr(raw, attr, None)
            if val is not None and val != {}:
                settings[attr] = val

    except Exception as e:
        logger.debug("Failed to extract service settings: %s", e)

    return settings


# ---------------------------------------------------------------------------
# LLM context data extraction
# ---------------------------------------------------------------------------


def extract_llm_context_data(context: Any) -> dict[str, Any]:
    """
    Safely serialise messages and tools from an LLMContext / OpenAILLMContext.

    Returns a dict with zero, one, or both of:
      ``messages`` — JSON string of the message list (``context.get_messages()``)
      ``tools``    — JSON string of the tool schema (``context.tools``), omitted
                     when tools is ``NOT_GIVEN``, ``None``, or an empty list.
    """
    result: dict[str, Any] = {}
    if context is None:
        return result

    # Messages
    try:
        get_msgs = getattr(context, "get_messages", None)
        messages = (
            get_msgs() if callable(get_msgs) else getattr(context, "messages", None)
        )
        if messages:
            result["messages"] = json.dumps(messages, default=str)
    except Exception as e:
        logger.debug("Failed to serialise LLM messages: %s", e)

    # Tools — guard against NOT_GIVEN sentinel and empty collections
    try:
        tools = getattr(context, "tools", None)
        # Pipecat uses openai's NOT_GIVEN sentinel; it is falsy-like but not None
        if tools is not None and tools is not False:
            # Check it is not the NOT_GIVEN sentinel (has no standard bool)
            tools_repr = repr(tools)
            if "NOT_GIVEN" not in tools_repr:
                tools_list = _resolve_tools_to_list(tools)
                if tools_list:
                    result["tools"] = json.dumps(tools_list, default=str)
    except Exception as e:
        logger.debug("Failed to serialise LLM tools: %s", e)

    return result


def merge_llm_pending_stash(
    existing: dict[str, Any], extracted: dict[str, Any]
) -> None:
    """Merge non-empty keys from ``extracted`` into ``existing`` (in place)."""
    for key, val in extracted.items():
        if val:
            existing[key] = val


def system_prompt_from_messages_json(json_str: str) -> str | None:
    """
    Return the ``system`` role content from a JSON-encoded messages list.

    Scans the list for the first message with ``role == "system"`` and returns
    its content as a string. Returns ``None`` when the input is absent, the
    JSON is invalid, or no system message is found.
    """
    if not json_str:
        return None
    try:
        for msg in json.loads(json_str):
            if not (isinstance(msg, dict) and msg.get("role") == "system"):
                continue
            content = msg.get("content") or ""
            if content:
                return content if isinstance(content, str) else json.dumps(content)
            break
    except Exception:
        pass
    return None


def json_dumps_messages(messages: Any) -> str | None:
    """Serialise a message list to JSON, or ``None`` if empty / invalid."""
    if not messages:
        return None
    try:
        return json.dumps(messages, default=str)
    except Exception as e:
        logger.debug("Failed to json.dumps messages: %s", e)
        return None


def merge_appended_messages_json(
    existing_json: str | None, new_messages: Any
) -> str | None:
    """
    Parse ``existing_json`` as a list (if present), extend with ``new_messages``,
    and re-serialise. Falls back to serialising only ``new_messages``.
    """
    if not new_messages:
        return existing_json
    try:
        if existing_json:
            prev = json.loads(existing_json)
            if isinstance(prev, list):
                return json.dumps(prev + list(new_messages), default=str)
    except Exception as e:
        logger.debug("merge_appended_messages_json parse failed: %s", e)
    return json_dumps_messages(list(new_messages))


def _resolve_tools_to_list(tools: Any) -> list[Any] | None:
    """
    Resolve any Pipecat tools container to a plain serialisable list.

    Handles:
    - Plain ``list`` — returned as-is (items may be raw dicts or ``FunctionSchema``).
    - ``ToolsSchema`` — reads ``.standard_tools`` and converts each ``FunctionSchema``
      via ``.to_default_dict()``.
    - Legacy objects with a ``.tools`` list attribute.

    ``FunctionSchema`` objects that don't appear in a ``ToolsSchema`` are also
    converted via ``.to_default_dict()`` when present.
    """
    if isinstance(tools, list):
        return _coerce_function_schemas(tools) or None

    # ToolsSchema (pipecat.adapters.schemas.tools_schema) stores its tools in
    # standard_tools, not in a .tools attribute.
    if hasattr(tools, "standard_tools"):
        raw = tools.standard_tools
        if isinstance(raw, list) and raw:
            return _coerce_function_schemas(raw)
        return None

    # Legacy / other containers
    inner = getattr(tools, "tools", None)
    if isinstance(inner, list) and inner:
        return _coerce_function_schemas(inner)

    return None


def _coerce_function_schemas(items: list[Any]) -> list[Any]:
    """Convert any ``FunctionSchema`` objects in *items* to plain dicts."""
    result = []
    for item in items:
        if hasattr(item, "to_default_dict"):
            result.append(item.to_default_dict())
        else:
            result.append(item)
    return result


def serialize_tools_field(tools: Any) -> str | None:
    """
    Serialise Pipecat ``LLMSetToolsFrame.tools`` (list, ``ToolsSchema``, or sentinel).
    """
    if tools is None or tools is False:
        return None
    tools_repr = repr(tools)
    if "NOT_GIVEN" in tools_repr:
        return None
    try:
        tools_list = _resolve_tools_to_list(tools)
        if tools_list:
            return json.dumps(tools_list, default=str)
    except Exception as e:
        logger.debug("Failed to serialize tools field: %s", e)
    return None


def serialize_tool_choice_field(tool_choice: Any) -> str | None:
    """Serialise ``LLMSetToolChoiceFrame.tool_choice`` for ``llm.tool_choice``."""
    if tool_choice is None:
        return None
    try:
        return json.dumps(tool_choice, default=str)
    except Exception as e:
        logger.debug("Failed to serialize tool_choice: %s", e)
        return str(tool_choice)


def truncate_for_trace_attr(text: str, max_len: int = MAX_TEXT_BUFFER_LENGTH) -> str:
    """Truncate long strings for span attributes."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------


def _llm_token_usage_to_dict(usage: Any) -> dict[str, Any]:
    """Flatten LLMTokenUsage (Pydantic) or similar to prompt/completion/total keys."""
    out: dict[str, Any] = {}
    if usage is None:
        return out
    try:
        if hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif hasattr(usage, "dict"):
            data = usage.dict()
        else:
            data = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "cache_read_input_tokens": getattr(
                    usage, "cache_read_input_tokens", None
                ),
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", None
                ),
                "reasoning_tokens": getattr(usage, "reasoning_tokens", None),
            }
        for key, out_key in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
            ("cache_read_input_tokens", "cache_read_tokens"),
            ("cache_creation_input_tokens", "cache_creation_tokens"),
            ("reasoning_tokens", "reasoning_tokens"),
        ):
            val = data.get(key)
            if val is not None:
                out[out_key] = int(val)
        if "total_tokens" not in out:
            p = out.get("prompt_tokens")
            c = out.get("completion_tokens")
            if p is not None and c is not None:
                out["total_tokens"] = p + c
    except Exception as e:
        logger.debug("Failed to flatten LLM token usage: %s", e)
    return out


def extract_metrics_data(frame: Any) -> dict[str, Any]:
    """
    Iterate frame.data, route by isinstance, and return a flat metrics dict.

    Handles: TTFBMetricsData, LLMUsageMetricsData, ProcessingMetricsData,
             TTSUsageMetricsData, TextAggregationMetricsData, TurnMetricsData,
             SmartTurnMetricsData.

    Metric payload types are defined in ``pipecat.metrics.metrics`` (not ``frames``).
    """
    result: dict[str, Any] = {}
    try:
        from pipecat.metrics.metrics import (
            LLMUsageMetricsData,
            ProcessingMetricsData,
            TTFBMetricsData,
            TTSUsageMetricsData,
        )

        TextAggregationMetricsData: type[Any] | None = None
        try:
            from pipecat.metrics.metrics import TextAggregationMetricsData as _TAG
        except ImportError:
            pass
        else:
            TextAggregationMetricsData = _TAG

        TurnMetricsData: type[Any] | None = None
        try:
            from pipecat.metrics.metrics import TurnMetricsData as _TMD
        except ImportError:
            pass
        else:
            TurnMetricsData = _TMD

        SmartTurnMetricsData: type[Any] | None = None
        try:
            from pipecat.metrics.metrics import SmartTurnMetricsData as _STMD
        except ImportError:
            pass
        else:
            SmartTurnMetricsData = _STMD

        for item in getattr(frame, "data", []):
            try:
                if isinstance(item, TTFBMetricsData):
                    val = getattr(item, "value", None) or getattr(item, "ttfb", None)
                    if val is not None:
                        result["ttfb_seconds"] = float(val)
                    processor = getattr(item, "processor", None)
                    if processor:
                        result["ttfb_processor"] = str(processor)

                elif isinstance(item, LLMUsageMetricsData):
                    usage = getattr(item, "value", None)
                    merged = _llm_token_usage_to_dict(usage)
                    result.update(merged)
                    model = getattr(item, "model", None)
                    if model:
                        result["llm_model"] = str(model)

                elif isinstance(item, ProcessingMetricsData):
                    val = getattr(item, "value", None)
                    if val is not None:
                        result["processing_seconds"] = float(val)

                elif isinstance(item, TTSUsageMetricsData):
                    val = getattr(item, "value", None)
                    if val is not None:
                        result["tts_characters"] = int(val)

                elif TextAggregationMetricsData and isinstance(
                    item, TextAggregationMetricsData
                ):
                    val = getattr(item, "value", None)
                    if val is not None:
                        result["text_aggregation_seconds"] = float(val)

                elif SmartTurnMetricsData and isinstance(item, SmartTurnMetricsData):
                    if getattr(item, "is_complete", None) is not None:
                        result["turn_eou_is_complete"] = bool(item.is_complete)
                    if getattr(item, "probability", None) is not None:
                        result["turn_eou_confidence"] = float(item.probability)
                    if getattr(item, "e2e_processing_time_ms", None) is not None:
                        result["turn_eou_processing_time_ms"] = float(
                            item.e2e_processing_time_ms
                        )
                    if getattr(item, "inference_time_ms", None) is not None:
                        result["turn_eou_inference_ms"] = float(item.inference_time_ms)
                    if getattr(item, "server_total_time_ms", None) is not None:
                        result["turn_eou_server_total_ms"] = float(
                            item.server_total_time_ms
                        )

                elif TurnMetricsData and isinstance(item, TurnMetricsData):
                    if getattr(item, "is_complete", None) is not None:
                        result["turn_eou_is_complete"] = bool(item.is_complete)
                    if getattr(item, "probability", None) is not None:
                        result["turn_eou_confidence"] = float(item.probability)
                    if getattr(item, "e2e_processing_time_ms", None) is not None:
                        result["turn_eou_processing_time_ms"] = float(
                            item.e2e_processing_time_ms
                        )

            except Exception as item_err:
                logger.debug(
                    "Failed to parse metrics item %s: %s", type(item).__name__, item_err
                )

    except ImportError:
        logger.debug(
            "Could not import Pipecat metrics data types from pipecat.metrics.metrics"
        )
    except Exception as e:
        logger.debug("Failed to extract metrics data: %s", e)

    return result


# ---------------------------------------------------------------------------
# STT provider result helpers
# ---------------------------------------------------------------------------


def extract_stt_confidence(result: Any) -> Optional[float]:
    """
    Best-effort utterance-level confidence from an STT provider ``result`` object.

    Deepgram exposes ``channel.alternatives[0].confidence``; other providers may
    expose ``confidence`` on the root or nested structures.
    """
    if result is None:
        return None
    try:
        direct = getattr(result, "confidence", None)
        if direct is not None:
            return float(direct)
    except (TypeError, ValueError):
        pass
    try:
        channel = getattr(result, "channel", None)
        if channel is not None:
            alts = getattr(channel, "alternatives", None)
            if alts and len(alts) > 0:
                c0 = getattr(alts[0], "confidence", None)
                if c0 is not None:
                    return float(c0)
    except (TypeError, ValueError, IndexError):
        pass
    try:
        if isinstance(result, dict):
            ch = result.get("channel")
            if isinstance(ch, dict):
                alts = ch.get("alternatives") or []
                if alts and isinstance(alts[0], dict):
                    c = alts[0].get("confidence")
                    if c is not None:
                        return float(c)
    except (TypeError, ValueError, IndexError):
        pass
    return None


# ---------------------------------------------------------------------------
# Frame text extraction
# ---------------------------------------------------------------------------


def extract_frame_text(frame: Any) -> Optional[str]:
    """Return text content from any text-carrying frame type."""
    try:
        text = getattr(frame, "text", None)
        if text is not None:
            return str(text)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Function call extraction
# ---------------------------------------------------------------------------


def extract_function_call_data(frame: Any) -> dict[str, Any]:
    """
    Extract function name, tool_call_id, arguments, result, run_llm from
    FunctionCallInProgressFrame or FunctionCallResultFrame.
    """
    data: dict[str, Any] = {}
    try:
        if hasattr(frame, "function_name"):
            data["function_name"] = str(frame.function_name)
        if hasattr(frame, "tool_call_id"):
            data["tool_call_id"] = str(frame.tool_call_id)
        if hasattr(frame, "arguments"):
            args = frame.arguments
            if isinstance(args, str):
                data["arguments"] = args
            elif args is not None:
                try:
                    data["arguments"] = json.dumps(args, default=str)
                except Exception:
                    data["arguments"] = str(args)
        if hasattr(frame, "result"):
            res = frame.result
            if isinstance(res, str):
                data["result"] = res
            elif res is not None:
                try:
                    data["result"] = json.dumps(res, default=str)
                except Exception:
                    data["result"] = str(res)
        if hasattr(frame, "run_llm"):
            data["run_llm"] = bool(frame.run_llm)
    except Exception as e:
        logger.debug("Failed to extract function call data: %s", e)

    return data


# ---------------------------------------------------------------------------
# Processor info
# ---------------------------------------------------------------------------


def serialize_processor_info(processor: Any) -> dict[str, Any]:
    """Return processor name, class, and settings probe."""
    info: dict[str, Any] = {}
    try:
        if hasattr(processor, "name"):
            info["name"] = str(processor.name)
        info["class"] = type(processor).__name__
        settings = extract_service_settings(processor)
        if settings:
            info["settings"] = settings
    except Exception as e:
        logger.debug("Failed to serialize processor info: %s", e)
    return info


# ---------------------------------------------------------------------------
# LLM cost
# ---------------------------------------------------------------------------


def calculate_llm_cost(
    model: str, input_tokens: int, output_tokens: int
) -> dict[str, Any]:
    """Delegate to noveum_trace.utils.llm_utils.estimate_cost()."""
    try:
        from noveum_trace.utils.llm_utils import estimate_cost

        cost_info = estimate_cost(
            model, input_tokens=input_tokens, output_tokens=output_tokens
        )
        return {
            "input": cost_info.get("input_cost", 0.0),
            "output": cost_info.get("output_cost", 0.0),
            "total": cost_info.get("total_cost", 0.0),
            "currency": cost_info.get("currency", "USD"),
        }
    except Exception as e:
        logger.debug("Failed to calculate LLM cost: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _frames_to_wav_bytes(frames: list[Any]) -> bytes:
    """
    Convert a list of Pipecat AudioRawFrame objects to WAV bytes.

    Each frame is expected to have:
      - .audio   bytes   raw PCM (16-bit LE)
      - .sample_rate  int
      - .num_channels int
    """
    if not frames:
        return b""

    first = frames[0]
    sample_rate: int = int(
        getattr(first, "sample_rate", AUDIO_SAMPLE_RATE_DEFAULT)
        or AUDIO_SAMPLE_RATE_DEFAULT
    )
    num_channels: int = int(
        getattr(first, "num_channels", AUDIO_NUM_CHANNELS_DEFAULT)
        or AUDIO_NUM_CHANNELS_DEFAULT
    )

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(AUDIO_BYTES_PER_SAMPLE)
        wf.setframerate(sample_rate)
        for frame in frames:
            audio = getattr(frame, "audio", b"") or b""
            if audio:
                wf.writeframes(audio)

    return buf.getvalue()


def calculate_audio_duration_ms(frames: list[Any]) -> float:
    """
    Sum PCM frame durations: num_samples / sample_rate * 1000.

    bytes_per_sample = 2 (16-bit), so num_samples = len(audio) / (2 * channels).
    """
    if not frames:
        return AUDIO_DURATION_MS_DEFAULT_VALUE

    total_ms = 0.0
    for frame in frames:
        audio = getattr(frame, "audio", None)
        if not audio:
            continue
        sample_rate = int(
            getattr(frame, "sample_rate", AUDIO_SAMPLE_RATE_DEFAULT)
            or AUDIO_SAMPLE_RATE_DEFAULT
        )
        num_channels = int(
            getattr(frame, "num_channels", AUDIO_NUM_CHANNELS_DEFAULT)
            or AUDIO_NUM_CHANNELS_DEFAULT
        )
        num_samples = len(audio) / (AUDIO_BYTES_PER_SAMPLE * num_channels)
        total_ms += (num_samples / sample_rate) * 1000.0

    return total_ms


def upload_audio_frames(
    frames: list[Any],
    audio_uuid: str,
    audio_type: str,
    trace_id: str,
    span_id: str,
    client: Any = None,
) -> bool:
    """
    Convert Pipecat AudioRawFrame list to WAV bytes and upload to Noveum.

    Same pattern as livekit_utils.upload_audio_frames() but works with
    Pipecat's AudioRawFrame objects (raw PCM) instead of LiveKit's rtc.AudioFrame.

    Args:
        client: Optional ``NoveumClient``. When omitted, uses ``get_client()``.

    Returns True if upload was queued successfully.
    """
    try:
        if not frames:
            logger.debug("No frames to upload")
            return False

        audio_bytes = _frames_to_wav_bytes(frames)
        if not audio_bytes:
            logger.debug("Empty WAV bytes, skipping upload")
            return False

        if client is None:
            from noveum_trace import get_client

            client = get_client()
        if not client:
            logger.info("No client available, skipping audio upload")
            return False

        duration_ms = calculate_audio_duration_ms(frames)
        metadata = {
            "duration_ms": duration_ms,
            "format": "wav",
            "type": audio_type,
        }

        client.export_audio(
            audio_data=audio_bytes,
            trace_id=trace_id,
            span_id=span_id,
            audio_uuid=audio_uuid,
            metadata=metadata,
        )
        logger.debug("Queued audio upload: %s", audio_uuid)
        return True

    except Exception as e:
        logger.warning("Failed to export audio %s: %s", audio_uuid, e, exc_info=True)
        return False
