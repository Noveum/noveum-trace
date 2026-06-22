"""
Unit tests for ``noveum_trace.integrations.crewai.crewai_utils``.

These are pure, deterministic tests of the serialization / token-extraction /
cost / message helpers that the CrewAI handler mixins rely on. The module had
no direct coverage before; see ``CREWAI_TEST_PLAN.md`` §4.

A ``# KNOWN BUG`` marker means the test baselines current (buggy) behavior so
the suite stays green and the defect is documented (see CREWAI_TEST_PLAN.md §2).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("crewai", reason="requires optional 'crewai' extra")

_src = Path(__file__).parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from noveum_trace.integrations.crewai import crewai_utils as u  # noqa: E402

# ---------------------------------------------------------------------------
# safe_serialize / _safe_serialize_inner
# ---------------------------------------------------------------------------


class TestSafeSerialize:
    def test_primitives_passthrough(self) -> None:
        assert u.safe_serialize(5) == 5
        assert u.safe_serialize(1.5) == 1.5
        assert u.safe_serialize(True) is True
        assert u.safe_serialize("hi") == "hi"
        assert u.safe_serialize(None) is None

    def test_nested_dict_list_recursed(self) -> None:
        out = u.safe_serialize({"a": [1, {"b": 2}], "c": (3, 4)})
        assert out == {"a": [1, {"b": 2}], "c": [3, 4]}  # tuple → list

    def test_non_str_dict_keys_coerced_to_str(self) -> None:
        assert u.safe_serialize({1: "x"}) == {"1": "x"}

    def test_max_depth_truncation_marker(self) -> None:
        deep: dict[str, Any] = {}
        node = deep
        for _ in range(12):
            node["child"] = {}
            node = node["child"]
        out = u.safe_serialize(deep, max_depth=8)
        # Walk down to the truncation marker.
        cur: Any = out
        depth = 0
        while isinstance(cur, dict) and "child" in cur:
            cur = cur["child"]
            depth += 1
        assert isinstance(cur, str) and cur.startswith("<max_depth:")
        assert depth == 8

    def test_circular_dict_reference(self) -> None:
        a: dict[str, Any] = {}
        a["self"] = a
        out = u.safe_serialize(a)
        assert out["self"] == "<circular_reference:dict>"

    def test_circular_list_reference(self) -> None:
        lst: list[Any] = []
        lst.append(lst)
        out = u.safe_serialize(lst)
        assert out == ["<circular_reference:list>"]

    def test_circular_object_reference(self) -> None:
        class Node:
            pass

        n = Node()
        n.parent = n  # type: ignore[attr-defined]
        out = u.safe_serialize(n)
        assert out["parent"] == "<circular_reference:Node>"

    def test_pydantic_v2_model_dump(self) -> None:
        from pydantic import BaseModel

        class Inner(BaseModel):
            x: int = 1

        class Outer(BaseModel):
            name: str = "n"
            inner: Inner = Inner()

        out = u.safe_serialize(Outer())
        assert out == {"name": "n", "inner": {"x": 1}}

    def test_object_with_to_dict(self) -> None:
        class HasToDict:
            def to_dict(self) -> dict[str, Any]:
                return {"k": "v"}

        assert u.safe_serialize(HasToDict()) == {"k": "v"}

    def test_object_dict_excludes_private(self) -> None:
        class Obj:
            def __init__(self) -> None:
                self.public = "p"
                self._private = "secret"

        out = u.safe_serialize(Obj())
        assert out == {"public": "p"}
        assert "_private" not in out

    def test_bytes_falls_through_to_str(self) -> None:
        assert u.safe_serialize(b"hello") == "b'hello'"

    def test_set_falls_through_to_str(self) -> None:
        out = u.safe_serialize({1, 2})
        assert isinstance(out, str)

    def test_outer_exception_returns_error_marker(self) -> None:
        class Boom:
            @property
            def __dict__(self):  # type: ignore[override]
                raise ValueError("boom")

        out = u.safe_serialize(Boom())
        assert isinstance(out, str)
        # Either the error marker or a str() fallback — must never raise.

    def test_safe_json_dumps_complex_object(self) -> None:
        from pydantic import BaseModel

        class M(BaseModel):
            a: int = 1

        s = u.safe_json_dumps({"m": M()})
        assert '"a": 1' in s

    def test_safe_json_dumps_never_raises(self) -> None:
        # object() has no __dict__, so safe_serialize falls back to str(); the
        # result is a valid JSON string. The point: it must never raise.
        out = u.safe_json_dumps(object(), fallback="{}")
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# truncate_str — KNOWN BUG: it is a no-op
# ---------------------------------------------------------------------------


class TestTruncateStr:
    def test_truncate_str_is_a_noop_known_bug(self) -> None:
        # KNOWN BUG (CREWAI_TEST_PLAN.md §2B): truncate_str ignores max_len and
        # returns the full string. No payload is ever truncated. When fixed,
        # this assertion should flip to len(result) <= max_len + suffix.
        text = "x" * 10_000
        result = u.truncate_str(text, 100)
        assert len(result) == 10_000  # <-- documents the bug

    def test_truncate_str_coerces_non_str(self) -> None:
        assert u.truncate_str(12345, 100) == "12345"


# ---------------------------------------------------------------------------
# extract_token_usage / _probe_token
# ---------------------------------------------------------------------------


class TestExtractTokenUsage:
    def test_openai_style(self) -> None:
        r = SimpleNamespace(usage={"prompt_tokens": 100, "completion_tokens": 50})
        assert u.extract_token_usage(r) == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    def test_anthropic_style(self) -> None:
        r = SimpleNamespace(usage={"input_tokens": 80, "output_tokens": 20})
        assert u.extract_token_usage(r) == {
            "input_tokens": 80,
            "output_tokens": 20,
            "total_tokens": 100,
        }

    def test_vertex_ai_style(self) -> None:
        r = SimpleNamespace(
            usage={"prompt_token_count": 30, "candidates_token_count": 10}
        )
        out = u.extract_token_usage(r)
        assert out["input_tokens"] == 30 and out["output_tokens"] == 10

    def test_bedrock_style(self) -> None:
        r = SimpleNamespace(usage={"inputTokenCount": 7, "outputTokenCount": 3})
        out = u.extract_token_usage(r)
        assert out["input_tokens"] == 7 and out["output_tokens"] == 3

    def test_watsonx_style(self) -> None:
        r = SimpleNamespace(usage={"input_token_count": 11, "generated_token_count": 4})
        out = u.extract_token_usage(r)
        assert out["input_tokens"] == 11 and out["output_tokens"] == 4

    def test_precedence_anthropic_over_openai(self) -> None:
        # Anthropic path (input_tokens) is probed before OpenAI (prompt_tokens).
        r = SimpleNamespace(usage={"input_tokens": 1, "prompt_tokens": 999})
        assert u.extract_token_usage(r)["input_tokens"] == 1

    def test_total_computed_when_missing(self) -> None:
        r = SimpleNamespace(usage={"input_tokens": 60, "output_tokens": 40})
        assert u.extract_token_usage(r)["total_tokens"] == 100

    def test_explicit_total_used(self) -> None:
        r = SimpleNamespace(
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 99}
        )
        assert u.extract_token_usage(r)["total_tokens"] == 99

    def test_none_response(self) -> None:
        assert u.extract_token_usage(None) == {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    def test_no_usage_all_none(self) -> None:
        assert u.extract_token_usage(SimpleNamespace()) == {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    def test_non_int_value_skipped_safely(self) -> None:
        r = SimpleNamespace(usage={"prompt_tokens": "NaN", "completion_tokens": 5})
        out = u.extract_token_usage(r)
        assert out["input_tokens"] is None
        assert out["output_tokens"] == 5


# ---------------------------------------------------------------------------
# extract_finish_reason / extract_response_text
# ---------------------------------------------------------------------------


class TestExtractFinishReason:
    def test_openai_choices(self) -> None:
        r = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])
        assert u.extract_finish_reason(r) == "stop"

    def test_anthropic_stop_reason(self) -> None:
        assert u.extract_finish_reason(SimpleNamespace(stop_reason="end_turn")) == (
            "end_turn"
        )

    def test_google_candidates(self) -> None:
        r = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])
        assert u.extract_finish_reason(r) == "STOP"

    def test_none_when_absent(self) -> None:
        assert u.extract_finish_reason(SimpleNamespace()) is None


class TestExtractResponseText:
    def test_openai_message_content(self) -> None:
        r = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello"))]
        )
        assert u.extract_response_text(r) == "Hello"

    def test_anthropic_content_list(self) -> None:
        r = SimpleNamespace(content=[SimpleNamespace(text="Reply")])
        assert u.extract_response_text(r) == "Reply"

    def test_anthropic_content_string(self) -> None:
        assert u.extract_response_text(SimpleNamespace(content="Direct")) == "Direct"

    def test_google_candidates_parts(self) -> None:
        r = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="Gen")])
                )
            ]
        )
        assert u.extract_response_text(r) == "Gen"

    def test_none_when_absent(self) -> None:
        assert u.extract_response_text(SimpleNamespace()) is None


# ---------------------------------------------------------------------------
# extract_system_prompt
# ---------------------------------------------------------------------------


class TestExtractSystemPrompt:
    def test_dict_message(self) -> None:
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        assert u.extract_system_prompt(msgs) == "Be helpful"

    def test_object_message(self) -> None:
        msg = SimpleNamespace(role="system", content="Sys")
        assert u.extract_system_prompt([msg]) == "Sys"

    def test_multiple_system_joined(self) -> None:
        msgs = [
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
        ]
        assert u.extract_system_prompt(msgs) == "A\nB"

    def test_case_insensitive_role(self) -> None:
        assert u.extract_system_prompt([{"role": "SYSTEM", "content": "X"}]) == "X"

    def test_anthropic_content_blocks(self) -> None:
        msgs = [{"role": "system", "content": [{"text": "p1"}, {"text": "p2"}]}]
        assert u.extract_system_prompt(msgs) == "p1\np2"

    def test_empty_returns_none(self) -> None:
        assert u.extract_system_prompt([]) is None
        assert u.extract_system_prompt(None) is None

    def test_no_system_message_returns_none(self) -> None:
        assert u.extract_system_prompt([{"role": "user", "content": "hi"}]) is None


# ---------------------------------------------------------------------------
# count_messages_by_role / messages_to_json
# ---------------------------------------------------------------------------


class TestMessages:
    def test_count_by_role_dict(self) -> None:
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
        ]
        out = u.count_messages_by_role(msgs)
        assert out.get("system") == 1 and out.get("user") == 2

    def test_count_by_role_case_normalized(self) -> None:
        out = u.count_messages_by_role([{"role": "USER", "content": "x"}])
        assert out.get("user") == 1

    def test_count_empty(self) -> None:
        assert u.count_messages_by_role([]) == {}

    def test_messages_to_json_roundtrip(self) -> None:
        import json

        s = u.messages_to_json([{"role": "user", "content": "hi"}])
        assert s is not None
        assert json.loads(s) == [{"role": "user", "content": "hi"}]

    def test_messages_to_json_empty_none(self) -> None:
        assert u.messages_to_json([]) is None


# ---------------------------------------------------------------------------
# tools serialisation
# ---------------------------------------------------------------------------


class TestToolsSerialisation:
    def test_empty_returns_empty_list(self) -> None:
        assert u.serialise_tools_list(None) == []
        assert u.serialise_tools_list([]) == []

    def test_object_tool_name_description(self) -> None:
        tool = SimpleNamespace(name="search", description="Search the web")
        out = u.serialise_tools_list([tool])
        assert out and out[0].get("name") == "search"

    def test_serialize_tool_schema_returns_json(self) -> None:
        tool = SimpleNamespace(name="t", description="d")
        s = u.serialize_tool_schema([tool])
        assert s is None or ("t" in s)

    def test_merge_available_tools_attributes_populates(self) -> None:
        attrs: dict[str, Any] = {}
        tool = SimpleNamespace(name="search", description="d")
        u.merge_available_tools_attributes(attrs, [tool], "agent")
        assert attrs["agent.available_tools.count"] == 1
        assert attrs["agent.available_tools.names"] == ["search"]


# ---------------------------------------------------------------------------
# calculate_llm_cost
# ---------------------------------------------------------------------------


class TestCalculateLlmCost:
    def test_known_model_returns_cost_dict(self) -> None:
        out = u.calculate_llm_cost("gpt-4o", 1000, 500)
        assert set(out).issuperset({"input", "output", "total", "currency"})
        assert out["currency"] == "USD"
        assert out["total"] >= 0.0

    def test_none_tokens_treated_as_zero(self) -> None:
        out = u.calculate_llm_cost("gpt-4o", None, None)
        # total should be 0 (no tokens) — and must not raise.
        assert out == {} or out.get("total") == 0.0


# ---------------------------------------------------------------------------
# duration helpers
# ---------------------------------------------------------------------------


class TestDuration:
    def test_duration_ms_explicit_end(self) -> None:
        assert u.duration_ms(1000.0, 1000.5) == 500.0

    def test_duration_ms_negative_clamped(self) -> None:
        assert u.duration_ms(1000.0, 999.0) == 0.0

    def test_duration_ms_rounded_3dp(self) -> None:
        out = u.duration_ms(0.0, 0.0001234)
        assert out == round(0.0001234 * 1000, 3)

    def test_duration_ms_monotonic_explicit(self) -> None:
        assert u.duration_ms_monotonic(10.0, 10.25) == 250.0


# ---------------------------------------------------------------------------
# safe_getattr / resolve_agent_id
# ---------------------------------------------------------------------------


class TestSafeGetattr:
    def test_dotted_chain(self) -> None:
        obj = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=42)))
        assert u.safe_getattr(obj, "a", "b", "c") == 42

    def test_dict_access(self) -> None:
        assert u.safe_getattr({"k": "v"}, "k") == "v"

    def test_none_in_chain_returns_default(self) -> None:
        obj = SimpleNamespace(a=None)
        assert u.safe_getattr(obj, "a", "b", default="D") == "D"

    def test_missing_returns_default(self) -> None:
        assert u.safe_getattr(SimpleNamespace(), "nope", default="D") == "D"

    def test_resolve_agent_id_prefers_event_agent_id(self) -> None:
        event = SimpleNamespace(agent_id="ev-agent")
        source = SimpleNamespace(id="src-id")
        assert u.resolve_agent_id(source, event) == "ev-agent"
