"""Tests for exact-processor LLM invocation correlation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from noveum_trace.integrations.pipecat._llm_operation_registry import (
    LLMOperationAlreadyActiveError,
    LLMOperationRegistry,
)


class _EqualUnhashableProcessor:
    __hash__ = None

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EqualUnhashableProcessor)


def test_parallel_same_provider_processors_keep_distinct_state() -> None:
    registry = LLMOperationRegistry()
    first = _EqualUnhashableProcessor("OpenAILLMService")
    second = _EqualUnhashableProcessor("OpenAILLMService")

    first_op = registry.start(first, span=object(), provider="openai", started_at=1.0)
    second_op = registry.start(second, span=object(), provider="openai", started_at=2.0)
    first_op.output_chunks.append("main")
    second_op.output_chunks.append("evaluation")

    assert registry.get_active(first) is first_op
    assert registry.get_active(second) is second_op
    assert first_op.operation_id != second_op.operation_id
    assert first_op.processor_key != second_op.processor_key
    assert first_op.output_text == "main"
    assert second_op.output_text == "evaluation"
    assert len(registry.operations_for_alias("OpenAILLMService")) == 2


def test_completed_operation_accepts_late_metric_without_reopening_span() -> None:
    registry = LLMOperationRegistry()
    processor = _EqualUnhashableProcessor("AnthropicLLMService")
    span = object()
    operation = registry.start(processor, span=span, started_at=10.0)
    registry.complete(processor, logical_end_at=11.5)

    assert operation.phase == "metrics_pending"
    assert registry.get_active(processor) is None
    assert registry.get_metrics_target(processor) is operation
    assert registry.record_metric(
        operation,
        family="llm_usage",
        value={"prompt_tokens": 2, "completion_tokens": 3},
        unit="tokens",
    )
    accounting = registry.settle(processor)
    assert accounting is not None
    assert accounting.logical_end_at == 11.5
    assert accounting.metric("llm_usage") is not None
    assert not hasattr(accounting, "source_processor")
    assert operation.phase == "settled"


def test_start_settles_previous_record_and_increments_sequence() -> None:
    registry = LLMOperationRegistry()
    processor = _EqualUnhashableProcessor("AnthropicLLMService")
    first = registry.start(processor, span=object(), started_at=1.0)
    registry.complete(processor, logical_end_at=2.0)
    second = registry.start(processor, span=object(), started_at=3.0)

    assert first.phase == "settled"
    assert second.invocation_sequence == 2
    assert registry.ledger[0].operation_id == first.operation_id


def test_same_processor_overlap_is_rejected_instead_of_mixed() -> None:
    registry = LLMOperationRegistry()
    processor = _EqualUnhashableProcessor("OpenAILLMService")
    active = registry.start(processor, span=object())
    with pytest.raises(LLMOperationAlreadyActiveError):
        registry.start(processor, span=object())
    assert registry.get_active(processor) is active


def test_pending_input_preserves_absent_set_and_explicit_clear() -> None:
    registry = LLMOperationRegistry()
    processor = _EqualUnhashableProcessor("OpenAILLMService")
    first = registry.start(processor, span=object())
    assert first.pending_input is None
    assert first.pending_input_was_set is False
    registry.complete(processor)

    registry.set_pending_input(processor, {})
    second = registry.start(processor, span=object())
    assert second.pending_input == {}
    assert second.pending_input_was_set is True


def test_metric_deduplication_and_revised_canonical_snapshot() -> None:
    registry = LLMOperationRegistry()
    processor = _EqualUnhashableProcessor("OpenAILLMService")
    operation = registry.start(processor, span=object())
    original = {"prompt_tokens": 4, "completion_tokens": 1}
    assert registry.record_metric(operation, family="llm_usage", value=original)
    assert not registry.record_metric(operation, family="llm_usage", value=original)
    assert registry.record_metric(
        operation,
        family="llm_usage",
        value={"prompt_tokens": 4, "completion_tokens": 2},
    )
    original["prompt_tokens"] = 99
    assert len(operation.raw_metrics) == 2
    assert operation.canonical_metrics["llm_usage"].value == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
    }


def test_settled_ledger_is_copied_and_totals_use_unique_operations() -> None:
    registry = LLMOperationRegistry()
    processors = [
        _EqualUnhashableProcessor("OpenAILLMService"),
        _EqualUnhashableProcessor("OpenAILLMService"),
    ]
    operations = [registry.start(item, span=object()) for item in processors]
    for index, (processor, operation) in enumerate(
        zip(processors, operations), start=1
    ):
        registry.record_metric(
            operation,
            family="llm_usage",
            value={"prompt_tokens": index, "completion_tokens": index + 1},
        )
        registry.complete(processor)
    registry.settle_all()

    assert registry.sum_metric_fields(
        "llm_usage", ("prompt_tokens", "completion_tokens")
    ) == {"prompt_tokens": 3.0, "completion_tokens": 5.0}
    operations[0].canonical_metrics["llm_usage"].value["prompt_tokens"] = 1000
    assert registry.ledger[0].metric("llm_usage").value["prompt_tokens"] == 1
    with pytest.raises(FrozenInstanceError):
        registry.ledger[0].operation_id = "changed"  # type: ignore[misc]
