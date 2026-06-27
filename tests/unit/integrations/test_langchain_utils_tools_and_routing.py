"""
Pure-function unit tests for langchain_utils tool extraction and routing builders.

Targets gaps in:
- extract_tool_calls_from_response (modern/legacy/additive/callback/bad-JSON)
- _tool_to_dict / _convert_tools_to_dict_list (OpenAI-dict strategy + filtering)
- extract_available_tools (metadata-over-serialized priority)
- build_routing_attributes (per-tool score expansion, raise paths)
- build_langgraph_attributes (empty guard, step==0 inclusion)
- extract_langgraph_metadata (step==0 flips flag, graph_name does not, path node)

No client, no network -- all pure-function behavior.
"""

import json

import pytest

# Skip all tests if LangChain is not available
try:
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from noveum_trace.integrations.langchain.langchain_utils import (
        _convert_tools_to_dict_list,
        _tool_to_dict,
        build_langgraph_attributes,
        build_routing_attributes,
        extract_available_tools,
        extract_langgraph_metadata,
        extract_tool_calls_from_response,
    )

    LANGCHAIN_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    LANGCHAIN_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="LangChain not available"
)


def _llm_result(message):
    """Wrap a single message in an LLMResult with one ChatGeneration."""
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_extract_tool_calls_modern_legacy_additive_and_callback():
    # Modern format: AIMessage.tool_calls -> entry returned + callback fired
    modern_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}],
    )
    captured = []
    modern = extract_tool_calls_from_response(
        _llm_result(modern_msg),
        tool_call_id_callback=lambda cid, rid: captured.append((cid, rid)),
        run_id="run-42",
    )
    assert modern == [{"name": "search", "args": {"q": "x"}, "id": "call_1"}]
    # Callback wiring (populates _tool_call_id_to_llm in real handler)
    assert captured == [("call_1", "run-42")]

    # Legacy format: additional_kwargs.function_call -> id None, args from JSON
    legacy_msg = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {"name": "legacy_fn", "arguments": json.dumps({"a": 2})}
        },
    )
    legacy = extract_tool_calls_from_response(_llm_result(legacy_msg))
    assert legacy == [{"name": "legacy_fn", "args": {"a": 2}, "id": None}]

    # Both present on one message -> additive (modern entry first, legacy second)
    both_msg = AIMessage(
        content="",
        tool_calls=[{"name": "modern_fn", "args": {}, "id": "call_2"}],
        additional_kwargs={"function_call": {"name": "legacy_fn", "arguments": "{}"}},
    )
    both = extract_tool_calls_from_response(_llm_result(both_msg))
    assert len(both) == 2
    assert both[0] == {"name": "modern_fn", "args": {}, "id": "call_2"}
    assert both[1] == {"name": "legacy_fn", "args": {}, "id": None}

    # Invalid legacy JSON is swallowed (skipped, no raise -> no entry)
    bad_msg = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {"name": "legacy_fn", "arguments": "not valid json"}
        },
    )
    bad = extract_tool_calls_from_response(_llm_result(bad_msg))
    assert bad == []


def test_tool_to_dict_strategies_and_convert_filters_unknown():
    # Strategy 3: OpenAI function-calling dict -> name from function, args_schema
    # from parameters.
    openai_tool = {
        "type": "function",
        "function": {"name": "f", "parameters": {"type": "object", "x": 1}},
    }
    converted = _tool_to_dict(openai_tool)
    assert converted["name"] == "f"
    assert converted["args_schema"] == {"type": "object", "x": 1}
    # description falls back when absent
    assert converted["description"] == "No description"

    # _convert_tools_to_dict_list drops unknown-named and name-less tools
    result = _convert_tools_to_dict_list(
        [{"name": "a"}, {"name": "unknown"}, {"no": "name"}]
    )
    assert len(result) == 1
    assert result[0]["name"] == "a"

    # A single (non-list) tool is wrapped and processed
    single = _convert_tools_to_dict_list({"name": "solo"})
    assert len(single) == 1
    assert single[0]["name"] == "solo"


def test_extract_available_tools_metadata_wins_over_serialized():
    serialized = {"kwargs": {"tools": [{"name": "validB"}]}}
    metadata = {"noveum": {"available_tools": [{"name": "validA"}]}}

    # Priority 1: metadata wins over serialized
    both = extract_available_tools(serialized, metadata)
    assert len(both) == 1
    assert both[0]["name"] == "validA"

    # Priority 2: only serialized present -> serialized tool
    only_serialized = extract_available_tools(serialized, None)
    assert len(only_serialized) == 1
    assert only_serialized[0]["name"] == "validB"

    # Neither -> empty
    assert extract_available_tools(None, None) == []


def test_build_routing_attributes_optional_and_raise_paths():
    attrs = build_routing_attributes(
        {
            "source_node": "a",
            "target_node": "b",
            "tool_scores": {"x": 1, "y": 2},
            "alternatives": ["p", "q"],
            "custom_field": 123,
        }
    )
    assert attrs["routing.type"] == "conditional_edge"
    assert attrs["routing.source_node"] == "a"
    assert attrs["routing.target_node"] == "b"
    # Per-tool score expansion as floats
    assert attrs["routing.score.x"] == 1.0
    assert attrs["routing.score.y"] == 2.0
    # Alternatives count
    assert attrs["routing.alternatives_count"] == 2
    # Arbitrary key -> routing.<key> str-coerced
    assert attrs["routing.custom_field"] == "123"

    # NOTE: documents current behavior; possible bug: float(confidence) raises on
    # a non-numeric string (unwrapped float() call) and aborts the whole builder.
    with pytest.raises(ValueError):
        build_routing_attributes({"confidence": "high"})

    # float(score) likewise raises on a non-numeric tool_scores value.
    with pytest.raises(ValueError):
        build_routing_attributes({"tool_scores": {"x": "not-a-number"}})


def test_build_langgraph_attributes_empty_and_step_zero():
    # Not a graph -> empty dict
    assert build_langgraph_attributes({"is_langgraph": False}) == {}

    # is_langgraph True with step==0 -> step included (is-not-None inclusion),
    # is_graph True, None fields omitted.
    md = {
        "is_langgraph": True,
        "step": 0,
        "node_name": None,
        "graph_name": None,
        "checkpoint_ns": None,
        "execution_type": None,
    }
    attrs = build_langgraph_attributes(md)
    assert attrs["langgraph.is_graph"] is True
    assert attrs["langgraph.step"] == 0
    # None-valued fields are omitted entirely
    assert "langgraph.node_name" not in attrs
    assert "langgraph.graph_name" not in attrs
    assert "langgraph.checkpoint_ns" not in attrs
    assert "langgraph.execution_type" not in attrs


def test_extract_langgraph_metadata_step_zero_and_graph_name_no_flip():
    # step==0 flips is_langgraph True (0 is not None) with execution_type unknown
    step_zero = extract_langgraph_metadata({"langgraph_step": 0}, None, None)
    assert step_zero["step"] == 0
    assert step_zero["is_langgraph"] is True
    assert step_zero["execution_type"] == "unknown"

    # graph_name alone does NOT flip is_langgraph
    graph_only = extract_langgraph_metadata({"langgraph_graph_name": "G"}, None, None)
    assert graph_only["graph_name"] == "G"
    assert graph_only["is_langgraph"] is False
    assert graph_only["execution_type"] is None

    # langgraph_path: first non-"__" part selected as node_name
    path = extract_langgraph_metadata(
        {"langgraph_path": ("__pregel_pull", "myNode")}, None, None
    )
    assert path["node_name"] == "myNode"
    assert path["is_langgraph"] is True
    assert path["execution_type"] == "node"
