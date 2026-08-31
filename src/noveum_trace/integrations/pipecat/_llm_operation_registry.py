"""Per-processor LLM invocation correlation for the Pipecat observer.

The registry deliberately separates a finished span from its short-lived
correlation record.  An operation can therefore be logically complete while still
accepting late metrics without extending the span's duration.

Processor identity is exact Python object identity.  Object addresses are used only
as private dictionary keys and are never included in operation IDs or accounting
records.
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
    """Raised when an operation is passed to a registry that did not create it."""


@dataclass(frozen=True)
class LLMMetricObservation:
    """One normalized metric observation associated with an LLM invocation."""

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
    """Mutable state owned by exactly one LLM processor invocation."""

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
        """Return the response text accumulated for this operation."""

        return "".join(self.output_chunks)

    def finish_open_thought(self, signature: str = "") -> None:
        """Move an unfinished thought buffer into the completed thought lists."""

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
        """Return this operation's canonical metric for ``family``, if present."""

        return next(
            (metric for metric in self.canonical_metrics if metric.family == family),
            None,
        )


@dataclass
class _ProcessorState:
    """Private state retaining the real object so its identity cannot be reused."""

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
    """Own active and late-metric LLM operation state per processor instance."""

    def __init__(self) -> None:
        self._processors: dict[int, _ProcessorState] = {}
        self._aliases: dict[str, set[int]] = {}
        self._operations: dict[str, LLMOperationRecord] = {}
        self._ledger: list[LLMAccountingRecord] = []
        self._next_processor_ordinal = 0

    @property
    def ledger(self) -> tuple[LLMAccountingRecord, ...]:
        """Return accounting snapshots without exposing mutable registry storage."""

        return tuple(copy.deepcopy(self._ledger))

    @property
    def active_operations(self) -> tuple[LLMOperationRecord, ...]:
        """Return all active operations in processor registration order."""

        return tuple(
            state.active
            for state in self._ordered_processor_states()
            if state.active is not None
        )

    @property
    def metrics_pending_operations(self) -> tuple[LLMOperationRecord, ...]:
        """Return logically complete operations still eligible for late metrics."""

        return tuple(
            state.metrics_pending
            for state in self._ordered_processor_states()
            if state.metrics_pending is not None
        )

    def register_processor(
        self, processor: Any, processor_name: Optional[str] = None
    ) -> str:
        """Register ``processor`` and return its address-free session key."""

        state = self._state(processor, processor_name=processor_name, create=True)
        assert state is not None
        if processor_name:
            self.register_alias(processor, processor_name)
        return state.public_key

    def register_alias(self, processor: Any, processor_name: str) -> None:
        """Associate one exact Pipecat processor name with an object identity."""

        state = self._state(processor, processor_name=processor_name, create=True)
        assert state is not None
        alias = str(processor_name)
        state.aliases.add(alias)
        self._aliases.setdefault(alias, set()).add(id(processor))

    def processor_keys_for_alias(self, processor_name: str) -> tuple[str, ...]:
        """Return every registered processor key with this exact name."""

        states = self._states_for_alias(processor_name)
        return tuple(state.public_key for state in states)

    def operations_for_alias(
        self, processor_name: str
    ) -> tuple[LLMOperationRecord, ...]:
        """Return active-or-pending candidates matching one exact name.

        The method intentionally returns all candidates.  A caller must treat zero or
        multiple matches as unattributed instead of guessing.
        """

        operations: list[LLMOperationRecord] = []
        for state in self._states_for_alias(processor_name):
            operation = state.active or state.metrics_pending
            if operation is not None:
                operations.append(operation)
        return tuple(operations)

    def set_pending_input(self, processor: Any, value: Mapping[str, Any]) -> None:
        """Store destination-scoped context for the processor's next operation.

        Empty mappings are retained as explicit clears rather than treated as absent.
        """

        state = self._state(processor, create=True)
        assert state is not None
        state.pending_input = copy.deepcopy(dict(value))

    def merge_pending_input(self, processor: Any, value: Mapping[str, Any]) -> None:
        """Merge context fragments for one destination's next invocation."""
        state = self._state(processor, create=True)
        assert state is not None
        current = {} if state.pending_input is _UNSET else dict(state.pending_input)
        current.update(copy.deepcopy(dict(value)))
        state.pending_input = current

    def pending_input_for(self, processor: Any) -> Optional[dict[str, Any]]:
        """Return a defensive copy of one processor's pending context."""
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
        """Start a new invocation for one exact processor instance.

        Starting the processor's next invocation settles its prior metrics-pending
        record.  An already active invocation is not silently overwritten because
        Pipecat supplies no identifier with which to untangle same-source overlap.
        """

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
        public_label = _public_label(display_name)
        operation_id = f"llm-{public_label}-{state.ordinal}-{state.invocation_sequence}"

        pending_input_was_set = state.pending_input is not _UNSET
        pending_input = (
            copy.deepcopy(state.pending_input) if pending_input_was_set else None
        )
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
            pending_input=pending_input,
            pending_input_was_set=pending_input_was_set,
            response_started_at=(
                time.monotonic() if started_at is None else started_at
            ),
            start_frame_id=start_frame_id,
            settled_predecessor_at_start=settled_predecessor,
        )
        state.active = operation
        self._operations[operation.operation_id] = operation
        return operation

    def get_active(self, processor: Any) -> Optional[LLMOperationRecord]:
        """Return only this processor's active operation."""

        state = self._state(processor, create=False)
        return state.active if state is not None else None

    def get_metrics_target(self, processor: Any) -> Optional[LLMOperationRecord]:
        """Return this processor's active or logically complete operation."""

        state = self._state(processor, create=False)
        if state is None:
            return None
        return state.active or state.metrics_pending

    def get_by_operation_id(self, operation_id: str) -> Optional[LLMOperationRecord]:
        """Return a live correlation record by its public operation ID."""

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
        """Logically complete an operation and retain it for late metrics."""

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
        """Record one deduplicated metric item.

        ``canonical_policy`` is deliberately explicit because snapshots, deltas, and
        first-event measurements have different replacement semantics.
        """

        self._assert_owned(operation)
        if canonical_policy not in _CANONICAL_POLICIES:
            raise ValueError("canonical_policy must be one of: first, replace, none")

        copied_value = copy.deepcopy(value)
        metric_fingerprint = fingerprint or make_metric_fingerprint(
            operation.operation_id,
            family,
            copied_value,
            unit=unit,
            model=model,
            channel=channel,
            native_type=native_type,
        )
        if metric_fingerprint in operation.metric_fingerprints:
            return False

        observation = LLMMetricObservation(
            family=family,
            value=copied_value,
            unit=unit,
            model=model,
            channel=channel,
            native_type=native_type,
            fingerprint=metric_fingerprint,
            observed_at=observed_at,
        )
        operation.metric_fingerprints.add(metric_fingerprint)
        operation.raw_metrics.append(observation)
        if canonical_policy == "replace":
            operation.canonical_metrics[family] = observation
        elif canonical_policy == "first":
            operation.canonical_metrics.setdefault(family, observation)
        return True

    def settle(self, processor: Any) -> Optional[LLMAccountingRecord]:
        """Discard this processor's pending correlation state into the ledger."""

        state = self._state(processor, create=False)
        return self._settle_state(state) if state is not None else None

    def settle_all(self) -> tuple[LLMAccountingRecord, ...]:
        """Settle every logically complete record in deterministic order."""

        settled: list[LLMAccountingRecord] = []
        for state in self._ordered_processor_states():
            accounting = self._settle_state(state)
            if accounting is not None:
                settled.append(accounting)
        return tuple(settled)

    def clear(self) -> None:
        """Drop all session state after trace finalization."""
        self._processors.clear()
        self._aliases.clear()
        self._operations.clear()
        self._ledger.clear()
        self._next_processor_ordinal = 0

    def sum_metric_fields(self, family: str, fields: Iterable[str]) -> dict[str, float]:
        """Sum numeric canonical fields from unique settled operations."""

        requested_fields = tuple(fields)
        totals = dict.fromkeys(requested_fields, 0.0)
        for accounting in self._ledger:
            metric = accounting.metric(family)
            if metric is None or not isinstance(metric.value, Mapping):
                continue
            for name in requested_fields:
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
            if state.processor is not processor:  # pragma: no cover - strong ref guard
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
        object_keys = self._aliases.get(str(processor_name), set())
        states = (self._processors[key] for key in object_keys)
        return tuple(sorted(states, key=lambda state: state.ordinal))

    def _ordered_processor_states(self) -> tuple[_ProcessorState, ...]:
        return tuple(sorted(self._processors.values(), key=lambda state: state.ordinal))

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

    payload = {
        "operation_id": operation_id,
        "family": family,
        "value": _stable_metric_value(value),
        "unit": unit,
        "model": model,
        "channel": channel,
        "native_type": native_type,
    }
    encoded = json.dumps(
        payload,
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
