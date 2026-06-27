"""
Unit tests for noveum_trace.integrations.langchain.message_utils.

Pure-function coverage of the message-parsing helpers that were previously
untested: message_to_dict, is_langchain_message, parse_messages_list,
process_chain_inputs_outputs, and extract_images_from_messages.

These tests use only CI-safe langchain_core fakes/messages -- no client,
no network -- and assert the *actual* behavior of the functions so future
regressions fail loudly.
"""

import pytest

# Skip all tests if LangChain (core) is not available.
try:
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    from noveum_trace.integrations.langchain.message_utils import (
        extract_images_from_messages,
        is_langchain_message,
        message_to_dict,
        parse_messages_list,
        process_chain_inputs_outputs,
    )

    LANGCHAIN_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    LANGCHAIN_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="LangChain not available"
)


def test_message_to_dict_pydantic_and_manual_and_error():
    """model_dump path, manual .content path, and exception fallback path."""
    # Strategy 1: Pydantic v2 model_dump (real HumanMessage).
    pyd = message_to_dict(HumanMessage("hi"))
    assert isinstance(pyd, dict)
    assert pyd["content"] == "hi"
    # model_dump emits the full pydantic field set, including type.
    assert pyd["type"] == "human"
    assert "error" not in pyd

    # Manual branch: object exposing only .content (no model_dump/dict).
    class FakeOnlyContent:
        content = "foo"

    manual = message_to_dict(FakeOnlyContent())
    assert manual == {"type": "FakeOnlyContent", "content": "foo"}
    # No id/name attributes -> those keys are not added.
    assert "id" not in manual
    assert "name" not in manual

    # Exception fallback: model_dump raises -> dict with an 'error' key.
    class FakeRaises:
        content = "bar"

        def model_dump(self):
            raise RuntimeError("boom")

    errd = message_to_dict(FakeRaises())
    assert errd["type"] == "FakeRaises"
    assert errd["content"] == "bar"
    assert errd["error"] == "boom"


def test_is_langchain_message_name_and_module_guard():
    """True for real messages; False for non-messages and same-named fakes."""
    assert is_langchain_message(HumanMessage("x")) is True
    assert is_langchain_message(AIMessage("x")) is True
    assert is_langchain_message(SystemMessage("x")) is True

    assert is_langchain_message({}) is False
    assert is_langchain_message("x") is False
    assert is_langchain_message(None) is False

    # A class literally named BaseMessage but defined in this (non-langchain)
    # module must NOT be treated as a LangChain message: the function guards
    # on cls.__module__.startswith("langchain").
    class BaseMessage:  # noqa: N801 - intentionally mirrors the real name
        pass

    class CustomMsg(BaseMessage):
        pass

    assert is_langchain_message(CustomMsg()) is False


def test_parse_messages_list_buckets_and_toolcall_dedup():
    """Routing into messages/tool_calls/tool_results with AI tool_call dedup."""
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"q": "hi"}, "id": "call_1"}],
    )
    msgs = [
        SystemMessage("sys"),
        HumanMessage("hi"),
        ai,
        ToolMessage(content="result", tool_call_id="call_1"),
    ]

    result = parse_messages_list(msgs)

    # system + human + ai all land in 'messages'.
    assert len(result["messages"]) == 3
    types = [m.get("type") for m in result["messages"]]
    assert types == ["system", "human", "ai"]

    # The AI message's tool_calls are de-duped to [] in the message entry.
    ai_entry = next(m for m in result["messages"] if m.get("type") == "ai")
    assert ai_entry["tool_calls"] == []

    # The extracted tool call lives in the dedicated bucket.
    assert result["tool_calls"] == [
        {"name": "search", "args": {"q": "hi"}, "id": "call_1"}
    ]

    # The ToolMessage lands in tool_results, not messages.
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0].get("type") == "tool"


def test_process_chain_inputs_outputs_message_list_vs_scalar_and_empty():
    """Message-list expansion, scalar stringify, empty-list and first-elem guards."""
    # A list of messages expands into the three structured sub-keys; the
    # original key is dropped and the .messages value is itself a list.
    expanded = process_chain_inputs_outputs(
        {"messages": [HumanMessage("hi"), AIMessage("yo")]}
    )
    assert set(expanded.keys()) == {
        "messages.messages",
        "messages.tool_calls",
        "messages.tool_results",
    }
    assert isinstance(expanded["messages.messages"], list)
    assert "messages" not in expanded

    # Scalar value is stringified.
    scalar = process_chain_inputs_outputs({"iteration": 3})
    assert scalar == {"iteration": "3"}

    # Empty list fails the `value and ...` guard -> falls to else -> str([]).
    empty = process_chain_inputs_outputs({"x": []})
    assert empty == {"x": "[]"}

    # A non-empty list whose first element is NOT a message is stringified
    # wholesale (only the first element is inspected for message-ness).
    mixed = process_chain_inputs_outputs({"y": ["hello", HumanMessage("hi")]})
    assert set(mixed.keys()) == {"y"}
    assert mixed["y"] == str(["hello", HumanMessage("hi")])


def test_extract_images_from_messages_dict_string_url_and_dedup():
    """image_url as dict vs string, dedup, plain text, and non-dict url skip."""
    # image_url given as a dict {"url": ...}.
    dict_batch = [
        {"content": [{"type": "image_url", "image_url": {"url": "https://x/a.png"}}]}
    ]
    assert extract_images_from_messages([dict_batch]) == ["https://x/a.png"]

    # image_url given as a bare string.
    str_batch = [{"content": [{"type": "image_url", "image_url": "https://x/b.png"}]}]
    assert extract_images_from_messages([str_batch]) == ["https://x/b.png"]

    # The same URL appearing twice is deduplicated.
    dup_batch = [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "https://x/d.png"}},
                {"type": "image_url", "image_url": {"url": "https://x/d.png"}},
            ]
        }
    ]
    deduped = extract_images_from_messages([dup_batch])
    assert deduped == ["https://x/d.png"]
    assert len(deduped) == 1

    # Plain-string content yields no images.
    plain_batch = [{"content": "just text"}]
    assert extract_images_from_messages([plain_batch]) == []

    # Non-dict, non-str image_url (int) is skipped (url resolves to None).
    int_batch = [{"content": [{"type": "image_url", "image_url": 123}]}]
    assert extract_images_from_messages([int_batch]) == []


def test_message_to_dict_noncallable_model_dump_guard():
    """A non-callable model_dump attribute skips strategy 1 (callable() guard)."""

    class FakeNonCallableModelDump:
        content = "cc"
        # model_dump exists but is NOT callable -> the callable() guard must
        # skip strategy 1 and fall through to the manual .content branch.
        model_dump = 5

    result = message_to_dict(FakeNonCallableModelDump())
    assert result == {"type": "FakeNonCallableModelDump", "content": "cc"}
    assert "error" not in result
