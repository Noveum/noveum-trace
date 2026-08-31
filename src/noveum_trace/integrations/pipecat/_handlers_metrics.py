"""Typed, per-item Pipecat metrics routing for ``NoveumTraceObserver``."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from noveum_trace.integrations.pipecat._observer_state import _PipecatObserverMixinBase
from noveum_trace.integrations.pipecat._processor_registry import (
    PROCESSOR_ROLE_LLM,
    PROCESSOR_ROLE_STT,
    PROCESSOR_ROLE_TTS,
)
from noveum_trace.integrations.pipecat.pipecat_utils import (
    NormalizedMetricData,
    calculate_llm_cost,
    normalize_metrics_data,
)

_ROLE_SPECIFIC_FAMILIES = {
    "llm_usage": PROCESSOR_ROLE_LLM,
    "tts_usage": PROCESSOR_ROLE_TTS,
    "text_aggregation": PROCESSOR_ROLE_TTS,
}


class _MetricsHandlerMixin(_PipecatObserverMixinBase):
    """Route each native metric independently and never guess an owner."""

    async def _handle_metrics(self, data: Any) -> None:
        frame_id = getattr(getattr(data, "frame", None), "id", None)
        for item_index, metric in enumerate(normalize_metrics_data(data.frame)):
            observation_id = f"{frame_id}:{item_index}"
            if metric.family in ("turn", "smart_turn"):
                self._apply_turn_metric(metric)
                continue
            if metric.family == "unknown":
                self._emit_unattributed_metric_span(
                    data, metric, "zero_matches", [], []
                )
                continue

            required_role = _ROLE_SPECIFIC_FAMILIES.get(metric.family)
            processors = self._processor_registry.resolve_metric_processors(
                getattr(data, "source", None), metric.processor, required_role
            )
            if len(processors) != 1:
                reason = "multiple_matches" if len(processors) > 1 else "zero_matches"
                operation_candidates = [
                    target
                    for processor in processors
                    for target in self._metric_operation_targets(processor)
                    if required_role is None or target[0] == required_role
                ]
                self._emit_unattributed_metric_span(
                    data, metric, reason, processors, operation_candidates
                )
                continue

            processor = processors[0]
            targets = self._metric_operation_targets(processor)
            if required_role is not None:
                targets = [target for target in targets if target[0] == required_role]
            if len(targets) != 1:
                reason = "multiple_matches" if len(targets) > 1 else "zero_matches"
                self._emit_unattributed_metric_span(
                    data, metric, reason, processors, targets
                )
                continue

            target_role, target = targets[0]
            if self._is_pre_start_metric(data, target_role, target):
                self._emit_unattributed_metric_span(
                    data, metric, "zero_matches", processors, []
                )
                continue
            self._apply_metric(metric, target_role, target, observation_id)

    def _is_pre_start_metric(self, data: Any, role: str, operation: Any) -> bool:
        """Reject a queued metric created before the current invocation started."""
        metric_frame_id = getattr(getattr(data, "frame", None), "id", None)
        if role == PROCESSOR_ROLE_LLM:
            if (
                operation.phase != "active"
                or not operation.settled_predecessor_at_start
            ):
                return False
            start_frame_id = operation.start_frame_id
        elif role == PROCESSOR_ROLE_STT:
            start_frame_id = self._stt_start_frame_id
        elif role == PROCESSOR_ROLE_TTS:
            start_frame_id = self._tts_start_frame_id
        else:
            return False
        return (
            isinstance(metric_frame_id, int)
            and isinstance(start_frame_id, int)
            and metric_frame_id < start_frame_id
        )

    def _metric_operation_targets(self, processor: Any) -> list[tuple[str, Any]]:
        """Return active-or-metrics-pending operations for one exact processor."""
        targets: list[tuple[str, Any]] = []
        if processor.has_role(PROCESSOR_ROLE_LLM):
            operation = self._llm_operations.get_metrics_target(processor.processor)
            if operation is not None:
                targets.append((PROCESSOR_ROLE_LLM, operation))
        if processor.has_role(PROCESSOR_ROLE_STT):
            if self._active_stt_span is not None:
                if self._stt_metric_processor is None:
                    self._stt_metric_processor = processor.processor
                if self._stt_metric_processor is processor.processor:
                    targets.append((PROCESSOR_ROLE_STT, self._active_stt_span))
        if processor.has_role(PROCESSOR_ROLE_TTS):
            if (
                self._active_tts_span is not None
                and self._tts_source_processor is processor.processor
            ):
                targets.append((PROCESSOR_ROLE_TTS, self._active_tts_span))
        return targets

    def _apply_metric(
        self,
        metric: NormalizedMetricData,
        role: str,
        target: Any,
        observation_id: str,
    ) -> None:
        if role == PROCESSOR_ROLE_LLM:
            self._apply_llm_metric(metric, target, observation_id)
            return

        span = target
        fingerprint = self._span_metric_fingerprint(
            span,
            metric,
            observation_id if metric.family == "processing" else None,
        )
        span_id = str(getattr(span, "span_id", id(span)))
        seen = self._metric_fingerprints.setdefault(span_id, set())
        if fingerprint in seen:
            return
        seen.add(fingerprint)

        if metric.family == "ttfb" and metric.value is not None:
            value_ms = float(metric.value) * 1000
            span.attributes[f"{role}.ttfb_ms"] = value_ms
            if role == PROCESSOR_ROLE_TTS:
                span.attributes["tts.time_to_first_byte_ms"] = value_ms
        elif metric.family == "processing" and metric.value is not None:
            value_ms = float(metric.value) * 1000
            key = f"{role}.processing_observations_ms"
            values = list(span.attributes.get(key, []))
            values.append(value_ms)
            span.attributes[key] = values
            span.attributes.setdefault(f"{role}.processing_ms", value_ms)
        elif role == PROCESSOR_ROLE_TTS and metric.family == "tts_usage":
            if metric.value is not None:
                span.attributes["tts.characters"] = int(metric.value)
        elif role == PROCESSOR_ROLE_TTS and metric.family == "text_aggregation":
            if metric.value is not None:
                span.attributes["tts.text_aggregation_ms"] = float(metric.value) * 1000

    def _apply_llm_metric(
        self, metric: NormalizedMetricData, operation: Any, observation_id: str
    ) -> None:
        policy = "replace"
        if metric.family == "ttfb":
            policy = "first"
        elif metric.family == "processing":
            policy = "none"
        changed = self._llm_operations.record_metric(
            operation,
            family=metric.family,
            value=metric.value,
            unit=metric.unit,
            model=metric.model,
            channel="metrics_frame",
            native_type=metric.native_class,
            fingerprint=(
                f"{operation.operation_id}:processing:{observation_id}"
                if metric.family == "processing"
                else None
            ),
            canonical_policy=policy,
        )
        if not changed:
            return

        span = operation.span
        if metric.family == "ttfb" and metric.value is not None:
            value_ms = float(metric.value) * 1000
            span.attributes.setdefault("llm.ttfb_ms", value_ms)
            # Compatibility alias. Pipecat TTFB is not necessarily a text token.
            span.attributes.setdefault("llm.time_to_first_token_ms", value_ms)
        elif metric.family == "processing" and metric.value is not None:
            values = list(span.attributes.get("llm.processing_observations_ms", []))
            values.append(float(metric.value) * 1000)
            span.attributes["llm.processing_observations_ms"] = values
            span.attributes.setdefault("llm.processing_ms", values[0])
        elif metric.family == "llm_usage" and isinstance(metric.value, dict):
            self._write_llm_usage(span, operation, metric.value, metric.model)

    @staticmethod
    def _write_llm_usage(
        span: Any, operation: Any, usage: dict[str, Any], model: Optional[str]
    ) -> None:
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or 0)
        span.attributes["llm.input_tokens"] = prompt
        span.attributes["llm.output_tokens"] = completion
        span.attributes["llm.total_tokens"] = total
        for source_key, attr_key in (
            ("cache_read_tokens", "llm.cache_read_tokens"),
            ("cache_creation_tokens", "llm.cache_creation_tokens"),
            ("reasoning_tokens", "llm.reasoning_tokens"),
        ):
            if source_key in usage:
                span.attributes[attr_key] = usage[source_key]

        effective_model = model or operation.model or span.attributes.get("llm.model")
        if effective_model:
            operation.model = effective_model
            span.attributes["llm.model"] = effective_model
            cost = calculate_llm_cost(effective_model, prompt, completion)
            if cost:
                span.attributes["llm.cost.input"] = cost["input"]
                span.attributes["llm.cost.output"] = cost["output"]
                span.attributes["llm.cost.total"] = cost["total"]
                span.attributes["llm.cost.currency"] = cost["currency"]

    def _apply_turn_metric(self, metric: NormalizedMetricData) -> None:
        value = metric.value if isinstance(metric.value, dict) else {}
        mapped = {
            "turn_eou_is_complete": value.get("is_complete"),
            "turn_eou_confidence": value.get("probability"),
            "turn_eou_processing_time_ms": value.get("e2e_processing_time_ms"),
            "turn_eou_inference_ms": value.get("inference_time_ms"),
            "turn_eou_server_total_ms": value.get("server_total_time_ms"),
        }
        mapped = {key: val for key, val in mapped.items() if val is not None}
        if not mapped:
            return
        if self._current_turn_span is None:
            self._pending_turn_eou_metrics.update(mapped)
            return
        attr_names = {
            "turn_eou_is_complete": "turn.eou_is_complete",
            "turn_eou_confidence": "turn.eou_confidence",
            "turn_eou_processing_time_ms": "turn.eou_processing_time_ms",
            "turn_eou_inference_ms": "turn.eou_inference_ms",
            "turn_eou_server_total_ms": "turn.eou_server_total_ms",
        }
        for key, val in mapped.items():
            self._current_turn_span.attributes[attr_names[key]] = val
        self._pending_turn_eou_metrics.clear()

    def _emit_unattributed_metric_span(
        self,
        data: Any,
        metric: NormalizedMetricData,
        reason: str,
        processor_candidates: list[Any],
        operation_candidates: list[tuple[str, Any]],
    ) -> None:
        if self._trace is None:
            return
        source = getattr(data, "source", None)
        destination = getattr(data, "destination", None)
        frame = getattr(data, "frame", None)
        attrs: dict[str, Any] = {
            "metric.native_class": metric.native_class,
            "metric.family": metric.family,
            "metric.value": metric.value,
            "metric.unit": metric.unit,
            "metric.reported_processor": metric.processor,
            "metric.model": metric.model,
            "metric.attribution_result": reason,
            "metric.processor_candidate_count": len(processor_candidates),
            "metric.processor_candidates": [
                {"name": candidate.name, "roles": sorted(candidate.roles)}
                for candidate in processor_candidates[:10]
            ],
            "metric.operation_candidate_count": len(operation_candidates),
            # Compatibility alias now truthfully means operation candidates.
            "metric.candidate_count": len(operation_candidates),
            "metric.source_processor": self._processor_name(source),
            "metric.destination_processor": self._processor_name(destination),
            "metric.frame_id": getattr(frame, "id", None),
            "metric.direction": str(getattr(data, "direction", "")),
            "metric.producer_timestamp": getattr(data, "timestamp", None),
            "metric.rollup_eligible": False,
        }
        span = self._create_child_span(
            f"pipecat.metric.{metric.family}",
            parent_span=self._current_turn_span,
            attributes={
                key: value for key, value in attrs.items() if value is not None
            },
        )
        if span is not None:
            self._finish_managed_span(span)

    @staticmethod
    def _processor_name(processor: Any) -> Optional[str]:
        if processor is None:
            return None
        name = getattr(processor, "name", None)
        return str(name) if isinstance(name, str) and name else type(processor).__name__

    @staticmethod
    def _span_metric_fingerprint(
        span: Any,
        metric: NormalizedMetricData,
        observation_id: Optional[str] = None,
    ) -> str:
        payload = {
            "span_id": str(getattr(span, "span_id", id(span))),
            "family": metric.family,
            "value": metric.value,
            "unit": metric.unit,
            "model": metric.model,
            "native_class": metric.native_class,
            "observation_id": observation_id,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
