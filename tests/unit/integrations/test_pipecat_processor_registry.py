"""Tests for exact Pipecat processor identity and role resolution."""

from types import SimpleNamespace

from noveum_trace.integrations.pipecat._processor_registry import (
    PROCESSOR_ROLE_LLM,
    PROCESSOR_ROLE_TTS,
    ProcessorRegistry,
)


def test_same_name_processors_remain_distinct_and_ambiguous_by_alias() -> None:
    registry = ProcessorRegistry()
    first = SimpleNamespace(name="OpenAILLMService")
    second = SimpleNamespace(name="OpenAILLMService")
    registry.set_explicit_role(first, PROCESSOR_ROLE_LLM)
    registry.set_explicit_role(second, PROCESSOR_ROLE_LLM)

    assert registry.get(first) != registry.get(second)
    assert len(registry.records_for_name("OpenAILLMService", PROCESSOR_ROLE_LLM)) == 2
    assert (
        len(
            registry.resolve_metric_processors(
                None, "OpenAILLMService", PROCESSOR_ROLE_LLM
            )
        )
        == 2
    )
    assert registry.resolve_metric_processors(
        first, "OpenAILLMService", PROCESSOR_ROLE_LLM
    ) == [registry.get(first)]


def test_one_processor_can_have_llm_and_tts_capabilities() -> None:
    registry = ProcessorRegistry()
    realtime = SimpleNamespace(name="realtime")

    registry.set_explicit_role(realtime, PROCESSOR_ROLE_LLM)
    record = registry.set_explicit_role(realtime, PROCESSOR_ROLE_TTS)

    assert record.has_role(PROCESSOR_ROLE_LLM)
    assert record.has_role(PROCESSOR_ROLE_TTS)
    assert registry.resolve_metric_processors(
        realtime, "realtime", PROCESSOR_ROLE_TTS
    ) == [record]
