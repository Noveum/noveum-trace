"""Per-processor LLM invocation correlation for the Pipecat observer.

Finished spans are retained only as short-lived correlation records so late
metrics can still be attached without extending the operation's duration.
Processor identity is exact Python object identity; object addresses never
appear in public operation IDs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

_UNSET = object()
_PUBLIC_LABEL_RE = re.compile(r"[^a-z0-9]+")
_CANONICAL_POLICIES = {"first", "replace", "none"}


class LLMOperationAlreadyActiveError(RuntimeError):
    """Raised when one processor starts an indistinguishable overlapping call."""


class LLMOperationNotOwnedError(ValueError):
    """Raised when an operation belongs to another registry."""


@dataclass(frozen=True)
class LLMMetricObservation:
    """One normalized metric observation associated with an invocation."""

    family: str
    value: Any
    unit: Optional[str]
    model: Optional[str]
    channel: Optional[str]
    native_type: Optional[str]
    fingerprint: str
    observed_at: Optional[float]


@dataclass
class LLMOperationRecord:
    """Mutable state for exactly one LLM processor invocation."""

    operation_id: str
    processor_key: str
    invocation_sequence: int
    processor_name: str
    provider: Optional[str]
    model: Optional[str]
    span: Any = field(repr=False, compare=False)
    parent_span: Any = field(default=None, repr=False, compare=False)
    source_processor: Any = field(default=None, repr=False, compare=False)
    pending_input: Optional[dict[str, Any]] = None
    pending_input_was_set: bool = False
    output_chunks: list[str] = field(default_factory=list)
    thought_chunks: list[str] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    thought_signatures: list[str] = field(default_factory=list)
    markers: list[dict[str, Any]] = field(default_factory=list)
    requested_function_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    executed_function_calls: list[dict[str, Any]] = field(default_factory=list)
    function_call_results: list[dict[str, Any]] = field(default_factory=list)
    raw_metrics: list[LLMMetricObservation] = field(default_factory=list)
    canonical_metrics: dict[str, LLMMetricObservation] = field(default_factory=dict)
    metric_fingerprints: set[str] = field(default_factory=set)
    error: Optional[dict[str, Any]] = None
    terminal_status: Optional[str] = None
    response_started_at: Optional[float] = None
    start_frame_id: Optional[int] = None
    settled_predecessor_at_start: bool = False
    logical_end_at: Optional[float] = None
    output_complete: Optional[bool] = None
    termination_reason: Optional[str] = None
    phase: str = "active"

    @property
    def output_text(self) -> str:
        return "".join(self.output_chunks)

    def finish_open_thought(self, signature: str = "") -> None:
        if not self.thought_chunks:
            return
        self.thoughts.append("".join(self.thought_chunks))
        self.thought_signatures.append(signature)
        self.thought_chunks.clear()


@dataclass(frozen=True)
class LLMAccountingRecord:
    """Correlation-free snapshot used for idempotent conversation rollups."""

    operation_id: str
    processor_key: str
    invocation_sequence: int
    processor_name: str
    provider: Optional[str]
    model: Optional[str]
    terminal_status: Optional[str]
    output_complete: Optional[bool]
    termination_reason: Optional[str]
    response_started_at: Optional[float]
    logical_end_at: Optional[float]
    canonical_metrics: tuple[LLMMetricObservation, ...]

    def metric(self, family: str) -> Optional[LLMMetricObservation]:
        return next(
            (item for item in self.canonical_metrics if item.family == family), None
        )


@dataclass
class _ProcessorState:
    processor: Any = field(repr=False, compare=False)
    public_key: str
    ordinal: int
    display_name: str
    aliases: set[str] = field(default_factory=set)
    invocation_sequence: int = 0
    active: Optional[LLMOperationRecord] = None
    metrics_pending: Optional[LLMOperationRecord] = None
    pending_input: Any = field(default=_UNSET, repr=False)


class LLMOperationRegistry:
    """Own active and late-metric LLM state per exact processor instance."""

    def __init__(self) -> None:
        self._processors: dict[int, _ProcessorState] = {}
        self._aliases: dict[str, set[int]] = {}
        self._operations: dict[str, LLMOperationRecord] = {}
        self._ledger: list[LLMAccountingRecord] = []
        self._next_processor_ordinal = 0

    @property
    def ledger(self) -> tuple[LLMAccountingRecord, ...]:
        return tuple(copy.deepcopy(self._ledger))

    @property
    def active_operations(self) -> tuple[LLMOperationRecord, ...]:
        return tuple(
            state.active
            for state in self._ordered_processor_states()
            if state.active is not None
        )

    @property
    def metrics_pending_operations(self) -> tuple[LLMOperationRecord, ...]:
        return tuple(
            state.metrics_pending
            for state in self._ordered_processor_states()
            if state.metrics_pending is not None
        )

    def register_processor(
        self, processor: Any, processor_name: Optional[str] = None
    ) -> str:
        state = self._state(processor, processor_name=processor_name, create=True)
        assert state is not None
        if processor_name:
            self.register_alias(processor, processor_name)
        return state.public_key

    def register_alias(self, processor: Any, processor_name: str) -> None:
        state = self._state(processor, processor_name=processor_name, create=True)
        assert state is not None
        alias = str(processor_name)
        state.aliases.add(alias)
        self._aliases.setdefault(alias, set()).add(id(processor))

    def processor_keys_for_alias(self, processor_name: str) -> tuple[str, ...]:
        return tuple(
            state.public_key for state in self._states_for_alias(processor_name)
        )

    def operations_for_alias(
        self, processor_name: str
    ) -> tuple[LLMOperationRecord, ...]:
        operations = []
        for state in self._states_for_alias(processor_name):
            operation = state.active or state.metrics_pending
            if operation is not None:
                operations.append(operation)
        return tuple(operations)

    def set_pending_input(self, processor: Any, value: Mapping[str, Any]) -> None:
        state = self._state(processor, create=True)
        assert state is not None
        state.pending_input = copy.deepcopy(dict(value))

    def merge_pending_input(self, processor: Any, value: Mapping[str, Any]) -> None:
        state = self._state(processor, create=True)
        assert state is not None
        current = {} if state.pending_input is _UNSET else dict(state.pending_input)
        current.update(copy.deepcopy(dict(value)))
        state.pending_input = current

    def pending_input_for(self, processor: Any) -> Optional[dict[str, Any]]:
        state = self._state(processor, create=False)
        if state is None or state.pending_input is _UNSET:
            return None
        return copy.deepcopy(dict(state.pending_input))

    def start(
        self,
        processor: Any,
        *,
        span: Any,
        parent_span: Any = None,
        processor_name: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        started_at: Optional[float] = None,
        start_frame_id: Optional[int] = None,
    ) -> LLMOperationRecord:
        state = self._state(processor, processor_name=processor_name, create=True)
        assert state is not None
        if state.active is not None:
            raise LLMOperationAlreadyActiveError(
                f"processor {state.public_key} already has active operation "
                f"{state.active.operation_id}"
            )
        settled_predecessor = state.metrics_pending is not None
        if settled_predecessor:
            self._settle_state(state)

        state.invocation_sequence += 1
        display_name = processor_name or state.display_name
        if processor_name:
            self.register_alias(processor, processor_name)
        operation_id = (
            f"llm-{_public_label(display_name)}-{state.ordinal}-"
            f"{state.invocation_sequence}"
        )
        pending_was_set = state.pending_input is not _UNSET
        pending = copy.deepcopy(state.pending_input) if pending_was_set else None
        state.pending_input = _UNSET
        operation = LLMOperationRecord(
            operation_id=operation_id,
            processor_key=state.public_key,
            invocation_sequence=state.invocation_sequence,
            processor_name=display_name,
            provider=provider,
            model=model,
            span=span,
            parent_span=parent_span,
            source_processor=processor,
            pending_input=pending,
            pending_input_was_set=pending_was_set,
            response_started_at=time.monotonic() if started_at is None else started_at,
            start_frame_id=start_frame_id,
            settled_predecessor_at_start=settled_predecessor,
        )
        state.active = operation
        self._operations[operation.operation_id] = operation
        return operation

    def get_active(self, processor: Any) -> Optional[LLMOperationRecord]:
        state = self._state(processor, create=False)
        return state.active if state is not None else None

    def get_metrics_target(self, processor: Any) -> Optional[LLMOperationRecord]:
        state = self._state(processor, create=False)
        return (state.active or state.metrics_pending) if state is not None else None

    def get_by_operation_id(self, operation_id: str) -> Optional[LLMOperationRecord]:
        return self._operations.get(operation_id)

    def complete(
        self,
        processor: Any,
        *,
        logical_end_at: Optional[float] = None,
        output_complete: bool = True,
        termination_reason: str = "response_end",
        terminal_status: str = "ok",
    ) -> Optional[LLMOperationRecord]:
        state = self._state(processor, create=False)
        if state is None or state.active is None:
            return None
        if state.metrics_pending is not None:
            self._settle_state(state)
        operation = state.active
        state.active = None
        operation.finish_open_thought()
        operation.logical_end_at = (
            time.monotonic() if logical_end_at is None else logical_end_at
        )
        operation.output_complete = output_complete
        operation.termination_reason = termination_reason
        operation.terminal_status = terminal_status
        operation.phase = "metrics_pending"
        state.metrics_pending = operation
        return operation

    def record_metric(
        self,
        operation: LLMOperationRecord,
        *,
        family: str,
        value: Any,
        unit: Optional[str] = None,
        model: Optional[str] = None,
        channel: Optional[str] = None,
        native_type: Optional[str] = None,
        fingerprint: Optional[str] = None,
        observed_at: Optional[float] = None,
        canonical_policy: str = "replace",
    ) -> bool:
        self._assert_owned(operation)
        if canonical_policy not in _CANONICAL_POLICIES:
            raise ValueError("canonical_policy must be one of: first, replace, none")
        copied = copy.deepcopy(value)
        fingerprint = fingerprint or make_metric_fingerprint(
            operation.operation_id,
            family,
            copied,
            unit=unit,
            model=model,
            channel=channel,
            native_type=native_type,
        )
        if fingerprint in operation.metric_fingerprints:
            return False
        observation = LLMMetricObservation(
            family=family,
            value=copied,
            unit=unit,
            model=model,
            channel=channel,
            native_type=native_type,
            fingerprint=fingerprint,
            observed_at=observed_at,
        )
        operation.metric_fingerprints.add(fingerprint)
        operation.raw_metrics.append(observation)
        if canonical_policy == "replace":
            operation.canonical_metrics[family] = observation
        elif canonical_policy == "first":
            operation.canonical_metrics.setdefault(family, observation)
        return True

    def settle(self, processor: Any) -> Optional[LLMAccountingRecord]:
        state = self._state(processor, create=False)
        return self._settle_state(state) if state is not None else None

    def settle_all(self) -> tuple[LLMAccountingRecord, ...]:
        settled = []
        for state in self._ordered_processor_states():
            accounting = self._settle_state(state)
            if accounting is not None:
                settled.append(accounting)
        return tuple(settled)

    def clear(self) -> None:
        self._processors.clear()
        self._aliases.clear()
        self._operations.clear()
        self._ledger.clear()
        self._next_processor_ordinal = 0

    def sum_metric_fields(self, family: str, fields: Iterable[str]) -> dict[str, float]:
        names = tuple(fields)
        totals = dict.fromkeys(names, 0.0)
        for accounting in self._ledger:
            metric = accounting.metric(family)
            if metric is None or not isinstance(metric.value, Mapping):
                continue
            for name in names:
                value = metric.value.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[name] += float(value)
        return totals

    def _state(
        self,
        processor: Any,
        *,
        processor_name: Optional[str] = None,
        create: bool,
    ) -> Optional[_ProcessorState]:
        object_key = id(processor)
        state = self._processors.get(object_key)
        if state is not None:
            if state.processor is not processor:
                raise RuntimeError("processor object identity was unexpectedly reused")
            return state
        if not create:
            return None
        self._next_processor_ordinal += 1
        display_name = processor_name or _processor_display_name(processor)
        state = _ProcessorState(
            processor=processor,
            public_key=f"processor-{self._next_processor_ordinal}",
            ordinal=self._next_processor_ordinal,
            display_name=display_name,
        )
        self._processors[object_key] = state
        self.register_alias(processor, display_name)
        return state

    def _states_for_alias(self, processor_name: str) -> tuple[_ProcessorState, ...]:
        states = (
            self._processors[key]
            for key in self._aliases.get(str(processor_name), set())
        )
        return tuple(sorted(states, key=lambda item: item.ordinal))

    def _ordered_processor_states(self) -> tuple[_ProcessorState, ...]:
        return tuple(sorted(self._processors.values(), key=lambda item: item.ordinal))

    def _assert_owned(self, operation: LLMOperationRecord) -> None:
        if self._operations.get(operation.operation_id) is not operation:
            raise LLMOperationNotOwnedError(
                f"operation {operation.operation_id!r} is not owned by this registry"
            )

    def _settle_state(self, state: _ProcessorState) -> Optional[LLMAccountingRecord]:
        operation = state.metrics_pending
        if operation is None:
            return None
        accounting = LLMAccountingRecord(
            operation_id=operation.operation_id,
            processor_key=operation.processor_key,
            invocation_sequence=operation.invocation_sequence,
            processor_name=operation.processor_name,
            provider=operation.provider,
            model=operation.model,
            terminal_status=operation.terminal_status,
            output_complete=operation.output_complete,
            termination_reason=operation.termination_reason,
            response_started_at=operation.response_started_at,
            logical_end_at=operation.logical_end_at,
            canonical_metrics=tuple(
                copy.deepcopy(tuple(operation.canonical_metrics.values()))
            ),
        )
        operation.phase = "settled"
        state.metrics_pending = None
        self._operations.pop(operation.operation_id, None)
        self._ledger.append(accounting)
        return copy.deepcopy(accounting)


def make_metric_fingerprint(
    operation_id: str,
    family: str,
    value: Any,
    *,
    unit: Optional[str] = None,
    model: Optional[str] = None,
    channel: Optional[str] = None,
    native_type: Optional[str] = None,
) -> str:
    """Build a stable, address-free metric fingerprint."""
    encoded = json.dumps(
        {
            "operation_id": operation_id,
            "family": family,
            "value": _stable_metric_value(value),
            "unit": unit,
            "model": model,
            "channel": channel,
            "native_type": native_type,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _processor_display_name(processor: Any) -> str:
    for field_name in ("name", "_name"):
        value = getattr(processor, field_name, None)
        if isinstance(value, str) and value:
            return value
    return type(processor).__name__


def _public_label(value: str) -> str:
    label = _PUBLIC_LABEL_RE.sub("-", str(value).lower()).strip("-")
    return (label or "processor")[:48].rstrip("-")


def _stable_metric_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _stable_metric_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_metric_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_stable_metric_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    return {"unsupported_type": type(value).__qualname__}
