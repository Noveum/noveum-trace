"""
Metrics frame handler mixin for NoveumTraceObserver.

Handles:
  - MetricsFrame — parse ``TTFBMetricsData``, ``LLMUsageMetricsData``,
    ``ProcessingMetricsData``, ``TTSUsageMetricsData``, ``TextAggregationMetricsData``,
    ``TurnMetricsData``, ``SmartTurnMetricsData``
    and scatter the values across the appropriate active span.
"""

from __future__ import annotations

import logging
from typing import Any

from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
from noveum_trace.integrations.pipecat.pipecat_utils import (
    calculate_llm_cost,
    extract_metrics_data,
)

logger = logging.getLogger(__name__)


class _MetricsHandlerMixin(_PipecatObserverMixinBase):
    """Handler method for ``MetricsFrame`` processing."""

    # State attributes declared in NoveumTraceObserver.__init__:
    #   _active_llm_span, _active_tts_span, _metrics_accumulator

    async def _handle_metrics(self, data: Any) -> None:
        """
        Parse ``MetricsFrame`` and distribute values to the appropriate span.

        Routing logic:

        - **TTFB** (``TTFBMetricsData``) — if the processor name contains ``"tts"``
          (and not ``"llm"``), written to ``tts.time_to_first_byte_ms`` on the LLM/TTS
          target span; otherwise written to ``llm.time_to_first_token_ms`` on the LLM
          target span (with TTS target as fallback).
        - **Processing time** (``ProcessingMetricsData``) — written to
          ``llm.processing_ms`` on the LLM target span.
        - **Token usage** (``LLMUsageMetricsData``) — written to
          ``llm.input_tokens``, ``llm.output_tokens``, ``llm.total_tokens`` and
          optional cache/reasoning token fields on the LLM target span. Also
          calculates per-span cost and increments conversation-level accumulators.
        - **TTS characters** (``TTSUsageMetricsData``) — written to
          ``tts.characters`` on the TTS target span.
        - **Text aggregation** (``TextAggregationMetricsData``) — written to
          ``tts.text_aggregation_ms`` on the TTS target span (time from first LLM
          token to first sentence sent to TTS).
        - **Turn / EOU** (``TurnMetricsData`` / ``SmartTurnMetricsData``) — written to
          the current ``pipecat.turn`` span as ``turn.eou_*`` attributes.

        Each "target span" is the live active span when present, falling back to
        ``_last_llm_span`` / ``_last_tts_span`` — the most-recently-closed span of
        that type — for the common case where Pipecat emits ``MetricsFrame`` after
        the span has already been finished.  Attributes
        are written directly to ``span.attributes`` (bypassing the ``_finished`` guard
        on ``set_attribute``) because spans are serialised at trace-export time, not
        at ``span.finish()`` time.
        """
        frame = data.frame
        metrics = extract_metrics_data(frame)
        if not metrics:
            return

        # Prefer the live (still-open) span; fall back to the most-recently-closed
        # span for MetricsFrame data that arrives after the span has finished.

        llm_target = self._active_llm_span or self._last_llm_span

        tts_target = self._active_tts_span or self._last_tts_span

        turn_target = self._current_turn_span

        # ------------------------------------------------------------------ #
        # TTFB → LLM span (llm.*) or TTS span (tts.*)                        #
        # ------------------------------------------------------------------ #
        if "ttfb_seconds" in metrics:
            ttfb_ms = metrics["ttfb_seconds"] * 1000
            ttfb_proc = metrics.get("ttfb_processor", "").lower()
            tts_only = tts_target and "tts" in ttfb_proc and "llm" not in ttfb_proc
            if tts_only:
                tts_target.attributes["tts.time_to_first_byte_ms"] = ttfb_ms
            elif llm_target:
                llm_target.attributes["llm.time_to_first_token_ms"] = ttfb_ms
            elif tts_target:  # fallback
                tts_target.attributes["tts.time_to_first_byte_ms"] = ttfb_ms

        # ------------------------------------------------------------------ #
        # Processing time                                                      #
        # ------------------------------------------------------------------ #
        if "processing_seconds" in metrics and llm_target:
            llm_target.attributes["llm.processing_ms"] = (
                metrics["processing_seconds"] * 1000
            )

        # ------------------------------------------------------------------ #
        # Token usage → LLM target span                                       #
        # ------------------------------------------------------------------ #
        if llm_target and any(
            k in metrics for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            prompt = metrics.get("prompt_tokens", 0) or 0
            completion = metrics.get("completion_tokens", 0) or 0
            total = metrics.get("total_tokens", prompt + completion)

            llm_target.attributes["llm.input_tokens"] = prompt
            llm_target.attributes["llm.output_tokens"] = completion
            llm_target.attributes["llm.total_tokens"] = total

            if "cache_read_tokens" in metrics:
                llm_target.attributes["llm.cache_read_tokens"] = metrics[
                    "cache_read_tokens"
                ]
            if "cache_creation_tokens" in metrics:
                llm_target.attributes["llm.cache_creation_tokens"] = metrics[
                    "cache_creation_tokens"
                ]
            if "reasoning_tokens" in metrics:
                llm_target.attributes["llm.reasoning_tokens"] = metrics[
                    "reasoning_tokens"
                ]

            model = metrics.get("llm_model") or llm_target.attributes.get(
                "llm.model", ""
            )
            if model:
                llm_target.attributes["llm.model"] = model
                cost = calculate_llm_cost(model, prompt, completion)
                if cost:
                    llm_target.attributes["llm.cost.input"] = cost["input"]
                    llm_target.attributes["llm.cost.output"] = cost["output"]
                    llm_target.attributes["llm.cost.total"] = cost["total"]
                    llm_target.attributes["llm.cost.currency"] = cost["currency"]
                    self._metrics_accumulator["total_cost"] = (
                        self._metrics_accumulator["total_cost"] + cost["total"]
                    )

            self._metrics_accumulator["total_input_tokens"] += prompt

            self._metrics_accumulator["total_output_tokens"] += completion

        # ------------------------------------------------------------------ #
        # TTS characters → TTS target span                                    #
        # ------------------------------------------------------------------ #
        if "tts_characters" in metrics and tts_target:
            tts_target.attributes["tts.characters"] = metrics["tts_characters"]

        # ------------------------------------------------------------------ #
        # Text aggregation → TTS target span                                  #
        # ------------------------------------------------------------------ #
        if "text_aggregation_seconds" in metrics and tts_target:
            tts_target.attributes["tts.text_aggregation_ms"] = (
                metrics["text_aggregation_seconds"] * 1000
            )

        # ------------------------------------------------------------------ #
        # Turn / EOU metrics → current turn span (or buffer if no turn)      #
        # ------------------------------------------------------------------ #
        eou_keys = (
            "turn_eou_is_complete",
            "turn_eou_confidence",
            "turn_eou_processing_time_ms",
            "turn_eou_inference_ms",
            "turn_eou_server_total_ms",
        )
        has_eou = any(k in metrics for k in eou_keys)
        if has_eou:
            if turn_target:
                # Live path: turn span exists, write directly
                if "turn_eou_is_complete" in metrics:
                    turn_target.attributes["turn.eou_is_complete"] = metrics[
                        "turn_eou_is_complete"
                    ]
                if "turn_eou_confidence" in metrics:
                    turn_target.attributes["turn.eou_confidence"] = metrics[
                        "turn_eou_confidence"
                    ]
                if "turn_eou_processing_time_ms" in metrics:
                    turn_target.attributes["turn.eou_processing_time_ms"] = metrics[
                        "turn_eou_processing_time_ms"
                    ]
                if "turn_eou_inference_ms" in metrics:
                    turn_target.attributes["turn.eou_inference_ms"] = metrics[
                        "turn_eou_inference_ms"
                    ]
                if "turn_eou_server_total_ms" in metrics:
                    turn_target.attributes["turn.eou_server_total_ms"] = metrics[
                        "turn_eou_server_total_ms"
                    ]
                # Clear any previously buffered EOU (now applied)

                self._pending_turn_eou_metrics.clear()
            else:
                # Buffer path: no turn span yet, stash for next turn
                for k in eou_keys:
                    if k in metrics:

                        self._pending_turn_eou_metrics[k] = metrics[k]
                logger.debug(
                    "Buffered EOU metrics (no active turn): %s",
                    [k for k in eou_keys if k in metrics],
                )
