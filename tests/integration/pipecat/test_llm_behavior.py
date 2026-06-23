"""
LLM subsystem (§D, LLM-1..12) — value-asserting regression tests per PIPECAT_TEST_PLAN.md.

Covers ``_handlers_llm._LLMHandlersMixin`` (one ``pipecat.llm`` span per response
cycle): settings flush + NOT_GIVEN/empty filtering, ``llm.input``/``tools``/
``tool_choice`` stash flush, message append-vs-replace, thought blocks, function-call
results/dedupe/pre-span exclusion, plus the metrics token+cost path
(``MetricsFrame`` -> ``LLMUsageMetricsData``) and the capture/opt-out gates.

These drive REAL pipecat frames against a REAL ``noveum_trace`` ``Trace`` and assert
concrete attribute *values* (names, JSON, parenting, status, lists), never mere
key-presence or mock-was-called.
"""

from __future__ import annotations

import json
import types

import pytest

pytest.importorskip("pipecat.frames.frames")

from noveum_trace.utils.llm_utils import estimate_cost  # noqa: E402


# --------------------------------------------------------------------------- #
# Local harness (does not touch conftest)                                      #
# --------------------------------------------------------------------------- #
def _ff():
    from pipecat.frames import frames as _frames

    return _frames


def _new_obs(*, capture_text: bool = True, capture_function_calls: bool = True):
    """Real observer wired to a real Trace + open ``pipecat.turn`` span.

    External turn tracking is on so handlers never auto-open a turn — the turn
    here is the genuine fold target whose ``span_id`` the LLM span parents to.
    Returns ``(observer, trace, turn_span)``.
    """
    from noveum_trace.core.trace import Trace
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    trace = Trace(name="pipecat.conversation")
    turn = trace.create_span(name="pipecat.turn", attributes={"turn.number": 1})
    obs = NoveumTraceObserver(
        capture_text=capture_text, capture_function_calls=capture_function_calls
    )
    obs._trace = trace
    obs._current_turn_span = turn
    obs._using_external_turn_tracking = True
    return obs, trace, turn


def _data(frame, source=None):
    """The dispatch payload: ``data.frame`` and ``data.source``."""
    return types.SimpleNamespace(frame=frame, source=source)


# --------------------------------------------------------------------------- #
# LLM-1 — settings flush with NOT_GIVEN/empty filtering + overwrite order      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_llm_settings_flush_filters_sentinels_and_overwrites_max_tokens() -> None:
    # Guards: NOT_GIVEN/empty filtering + the max_completion_tokens->llm.max_tokens overwrite order.
    obs, _trace, turn = _new_obs()
    settings = types.SimpleNamespace(
        model="gpt-4o",
        system_instruction="be brief",
        temperature=0.7,
        top_p="NOT_GIVEN",  # OpenAI sentinel repr -> filtered
        frequency_penalty={},  # empty dict -> filtered
        max_tokens=256,
        max_completion_tokens=512,  # OpenAI alias overwrites llm.max_tokens
    )
    source = types.SimpleNamespace(_settings=settings)

    await obs._handle_llm_response_start(
        _data(_ff().LLMFullResponseStartFrame(), source)
    )

    span = obs._active_llm_span
    assert span.name == "pipecat.llm"
    assert span.parent_span_id == turn.span_id
    assert span.attributes["llm.model"] == "gpt-4o"
    assert span.attributes["llm.system_prompt"] == "be brief"
    assert span.attributes["llm.temperature"] == 0.7
    assert "llm.top_p" not in span.attributes
    assert "llm.frequency_penalty" not in span.attributes
    assert span.attributes["llm.max_tokens"] == 512


# --------------------------------------------------------------------------- #
# LLM-2 — input/tools/tool_choice stash flush asserts real JSON values         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_llm_context_stash_flushes_real_json_values_and_resets() -> None:
    # Guards: stash merge+flush+reset values for llm.input/tools/tool_choice.
    obs, _trace, _turn = _new_obs()
    ctx = types.SimpleNamespace(
        get_messages=lambda: [{"role": "user", "content": "hi"}], tools=None
    )
    tool = {"type": "function", "function": {"name": "f", "parameters": {}}}

    await obs._handle_llm_context(_data(_ff().LLMContextFrame(context=ctx)))
    await obs._handle_llm_set_tools(_data(_ff().LLMSetToolsFrame(tools=[tool])))
    await obs._handle_llm_set_tool_choice(
        _data(_ff().LLMSetToolChoiceFrame(tool_choice="auto"))
    )
    await obs._handle_llm_response_start(
        _data(_ff().LLMFullResponseStartFrame(), types.SimpleNamespace(_settings=None))
    )

    span = obs._active_llm_span
    assert json.loads(span.attributes["llm.input"]) == [
        {"role": "user", "content": "hi"}
    ]
    assert json.loads(span.attributes["llm.tools"]) == [tool]
    assert span.attributes["llm.tool_choice"] == json.dumps("auto")
    assert obs._pending_llm_context == {}


# --------------------------------------------------------------------------- #
# LLM-3 — LLMMessagesAppendFrame appends rather than replaces                   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_llm_messages_append_extends_stashed_messages() -> None:
    # Guards: merge_appended_messages_json append-vs-replace ordering.
    obs, _trace, _turn = _new_obs()

    await obs._handle_llm_messages_replace(
        _data(_ff().LLMMessagesUpdateFrame(messages=[{"role": "user", "content": "a"}]))
    )
    await obs._handle_llm_messages_append(
        _data(
            _ff().LLMMessagesAppendFrame(
                messages=[{"role": "assistant", "content": "b"}]
            )
        )
    )
    await obs._handle_llm_response_start(
        _data(_ff().LLMFullResponseStartFrame(), types.SimpleNamespace(_settings=None))
    )

    assert json.loads(obs._active_llm_span.attributes["llm.input"]) == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


# --------------------------------------------------------------------------- #
# LLM-4 — thought blocks pinned to exact list values + ordering                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_llm_thoughts_pinned_to_exact_values_and_ordering() -> None:
    # Guards: per-block concatenation, block boundaries, None-signature -> ''.
    obs, trace, turn = _new_obs()
    span = trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)
    obs._active_llm_span = span

    ff = _ff()
    await obs._handle_llm_thought_start(types.SimpleNamespace())
    await obs._handle_llm_thought_text(_data(ff.LLMThoughtTextFrame(text="a1")))
    await obs._handle_llm_thought_end(_data(ff.LLMThoughtEndFrame(signature="s1")))
    await obs._handle_llm_thought_start(types.SimpleNamespace())
    await obs._handle_llm_thought_text(_data(ff.LLMThoughtTextFrame(text="b1")))
    await obs._handle_llm_thought_text(_data(ff.LLMThoughtTextFrame(text="b2")))
    await obs._handle_llm_thought_end(_data(ff.LLMThoughtEndFrame(signature=None)))
    await obs._handle_llm_response_end(types.SimpleNamespace())

    assert span.attributes["llm.thoughts"] == ["a1", "b1b2"]
    assert span.attributes["llm.thought_signatures"] == ["s1", ""]


# --------------------------------------------------------------------------- #
# LLM-5 — defensive flush of an unclosed thought block on response end          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_llm_unclosed_thought_flushed_on_response_end() -> None:
    # Guards: the unclosed-thought fallback flush in _handle_llm_response_end.
    obs, trace, turn = _new_obs()
    span = trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)
    obs._active_llm_span = span

    ff = _ff()
    await obs._handle_llm_thought_start(types.SimpleNamespace())
    await obs._handle_llm_thought_text(_data(ff.LLMThoughtTextFrame(text="partial")))
    # No ThoughtEnd before the response ends.
    await obs._handle_llm_response_end(types.SimpleNamespace())

    assert span.attributes["llm.thoughts"] == ["partial"]
    assert span.attributes["llm.thought_signatures"] == [""]


# --------------------------------------------------------------------------- #
# LLM-6 — function-call result dict carries full content                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_function_call_result_dict_carries_full_content() -> None:
    # Guards: in-progress->result move + full field extraction (name/args/result/run_llm).
    obs, trace, turn = _new_obs()
    span = trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)
    obs._active_llm_span = span

    ff = _ff()
    await obs._handle_function_call_start(
        _data(
            ff.FunctionCallInProgressFrame(
                function_name="get_weather",
                tool_call_id="c1",
                arguments='{"city":"SF"}',
            )
        )
    )
    await obs._handle_function_call_result(
        _data(
            ff.FunctionCallResultFrame(
                function_name="get_weather",
                tool_call_id="c1",
                arguments='{"city":"SF"}',
                result="sunny",
                run_llm=True,
            )
        )
    )
    await obs._handle_llm_response_end(types.SimpleNamespace())

    assert span.attributes["llm.function_call_results"] == [
        {
            "tool_call_id": "c1",
            "name": "get_weather",
            "arguments": '{"city":"SF"}',
            "result": "sunny",
            "run_llm": True,
        }
    ]
    # Call moved into results — it must NOT also appear in llm.function_calls.
    assert "llm.function_calls" not in span.attributes


# --------------------------------------------------------------------------- #
# LLM-7 — duplicate FunctionCallInProgressFrame deduped by tool_call_id         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_duplicate_function_call_in_progress_deduped_by_tool_call_id() -> None:
    # Guards: handler-level tool_call_id dedupe (upstream+downstream double-push).
    obs, trace, turn = _new_obs()
    span = trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)
    obs._active_llm_span = span

    ff = _ff()
    # Two DISTINCT frame objects sharing tool_call_id — frame-id dedupe would not
    # catch these, so this proves the handler's own tool_call_id dedupe.
    f1 = ff.FunctionCallInProgressFrame(
        function_name="dup", tool_call_id="d1", arguments="{}"
    )
    f2 = ff.FunctionCallInProgressFrame(
        function_name="dup", tool_call_id="d1", arguments="{}"
    )
    assert f1.id != f2.id
    await obs._handle_function_call_start(_data(f1))
    await obs._handle_function_call_start(_data(f2))
    await obs._handle_llm_response_end(types.SimpleNamespace())

    assert span.attributes["llm.function_calls"] == [
        {"tool_call_id": "d1", "name": "dup", "arguments": "{}"}
    ]


# --------------------------------------------------------------------------- #
# LLM-8 — pre-span function call is NOT double-counted on span 2                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_pre_span_function_call_not_double_counted_on_span2() -> None:
    # Guards: the _pre_span_function_call_ids exclusion (no double-count across spans).
    obs, _trace, _turn = _new_obs()
    source = types.SimpleNamespace(_settings=None)
    ff = _ff()

    # Span 1: open then close.
    await obs._handle_llm_response_start(_data(ff.LLMFullResponseStartFrame(), source))
    span1 = obs._active_llm_span
    await obs._handle_llm_response_end(types.SimpleNamespace())

    # Function-call frame arrives between span 1 close and span 2 open -> backref write.
    await obs._handle_function_call_start(
        _data(
            ff.FunctionCallInProgressFrame(
                function_name="late", tool_call_id="late1", arguments="{}"
            )
        )
    )

    # Span 2: open then close (no result fired).
    await obs._handle_llm_response_start(_data(ff.LLMFullResponseStartFrame(), source))
    span2 = obs._active_llm_span
    await obs._handle_llm_response_end(types.SimpleNamespace())

    assert span1 is not span2
    assert span1.attributes["llm.function_calls"] == [
        {"tool_call_id": "late1", "name": "late", "arguments": "{}"}
    ]
    # span2 must not re-list the pre-span call.
    assert "llm.function_calls" not in span2.attributes


# --------------------------------------------------------------------------- #
# LLM-9 — token usage + cost via MetricsFrame -> LLMUsageMetricsData            #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_metrics_token_usage_and_cost_pinned() -> None:
    # Guards: token->span mapping + the cost computation (the live MetricsFrame path).
    pytest.importorskip("pipecat.metrics.metrics")
    from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData

    obs, trace, turn = _new_obs()
    span = trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)
    obs._active_llm_span = span

    usage = LLMTokenUsage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000)
    metrics_data = LLMUsageMetricsData(
        processor="llm", model="gpt-4o-mini", value=usage
    )
    frame = _ff().MetricsFrame(data=[metrics_data])
    await obs._handle_metrics(_data(frame))

    assert span.attributes["llm.input_tokens"] == 1000
    assert span.attributes["llm.output_tokens"] == 1000
    assert span.attributes["llm.total_tokens"] == 2000
    assert span.attributes["llm.model"] == "gpt-4o-mini"
    # Derive expected costs from the pricing table rather than hard-coding
    # literals, so intentional price-table updates don't break this test.
    expected = estimate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=1000)
    assert span.attributes["llm.cost.input"] == pytest.approx(expected["input_cost"])
    assert span.attributes["llm.cost.output"] == pytest.approx(expected["output_cost"])
    assert span.attributes["llm.cost.total"] == pytest.approx(expected["total_cost"])
    assert span.attributes["llm.cost.currency"] == "USD"
    assert obs._metrics_accumulator["total_input_tokens"] == 1000
    assert obs._metrics_accumulator["total_output_tokens"] == 1000
    assert obs._metrics_accumulator["total_cost"] == pytest.approx(0.003)


# --------------------------------------------------------------------------- #
# LLM-10 — orphan LLM span under external tracking + no open turn (XFAIL)       #
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="Issue 1 (PIPECAT_SPAN_HIERARCHY_ISSUES.md): under external turn tracking "
    "with no open turn, the pipecat.llm span is orphaned (parent_span_id is None); it "
    "SHOULD attach to the last turn — flips to xpass when the _last_turn_span fallback lands.",
)
@pytest.mark.asyncio
async def test_llm_span_orphaned_under_external_tracking_xfail() -> None:
    # Guards: the LLM face of the orphan-parenting bug (Issue 1).
    from noveum_trace.core.trace import Trace
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    trace = Trace(name="pipecat.conversation")
    obs = NoveumTraceObserver(capture_text=True, capture_function_calls=True)
    obs._trace = trace
    obs._using_external_turn_tracking = True
    obs._current_turn_span = None

    await obs._handle_llm_response_start(
        _data(_ff().LLMFullResponseStartFrame(), types.SimpleNamespace(_settings=None))
    )

    # DESIRED behavior: the span should be parented under a turn. It currently is
    # not (parent_span_id is None), so this xfails until the fallback lands.
    assert obs._active_llm_span.parent_span_id is not None


@pytest.mark.asyncio
async def test_llm_orphan_does_not_auto_open_turn() -> None:
    # Guards: the `and not self._using_external_turn_tracking` branch — no turn auto-opened.
    from noveum_trace.core.trace import Trace
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    trace = Trace(name="pipecat.conversation")
    obs = NoveumTraceObserver(capture_text=True, capture_function_calls=True)
    obs._trace = trace
    obs._using_external_turn_tracking = True
    obs._current_turn_span = None

    await obs._handle_llm_response_start(
        _data(_ff().LLMFullResponseStartFrame(), types.SimpleNamespace(_settings=None))
    )

    # No new turn span was opened; the only span is the LLM span itself.
    assert obs._current_turn_span is None
    assert [s.name for s in trace.spans] == ["pipecat.llm"]


# --------------------------------------------------------------------------- #
# LLM-11 — capture_text / capture_function_calls opt-out gates                  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_capture_text_false_suppresses_output_and_thoughts() -> None:
    # Guards: the capture_text privacy gate (no llm.output / llm.thoughts; buffers empty).
    obs, trace, turn = _new_obs(capture_text=False)
    span = trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)
    obs._active_llm_span = span

    ff = _ff()
    await obs._handle_llm_text(_data(ff.LLMTextFrame(text="hi")))
    await obs._handle_llm_thought_start(types.SimpleNamespace())
    await obs._handle_llm_thought_text(_data(ff.LLMThoughtTextFrame(text="t")))
    await obs._handle_llm_thought_end(_data(ff.LLMThoughtEndFrame(signature="s")))
    await obs._handle_llm_response_end(types.SimpleNamespace())

    assert "llm.output" not in span.attributes
    assert "llm.thoughts" not in span.attributes
    assert obs._llm_text_buffer == []
    assert obs._llm_thoughts_list == []


@pytest.mark.asyncio
async def test_capture_function_calls_false_suppresses_call_attributes() -> None:
    # Guards: the capture_function_calls opt-out gate (no call attrs; pending empty).
    obs, trace, turn = _new_obs(capture_function_calls=False)
    span = trace.create_span(name="pipecat.llm", parent_span_id=turn.span_id)
    obs._active_llm_span = span

    ff = _ff()
    await obs._handle_function_call_start(
        _data(
            ff.FunctionCallInProgressFrame(
                function_name="x", tool_call_id="x1", arguments="{}"
            )
        )
    )
    await obs._handle_function_call_result(
        _data(
            ff.FunctionCallResultFrame(
                function_name="x", tool_call_id="x1", arguments="{}", result="r"
            )
        )
    )
    await obs._handle_llm_response_end(types.SimpleNamespace())

    assert "llm.function_calls" not in span.attributes
    assert "llm.function_call_results" not in span.attributes
    assert obs._pending_function_calls == {}


# --------------------------------------------------------------------------- #
# LLM-12 — context-summary error short-circuit + numeric coercion              #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_summary_result_error_short_circuits_text() -> None:
    # Guards: the error branch — llm.summary.error set, llm.summary.text suppressed.
    obs, _trace, turn = _new_obs()

    await obs._handle_llm_summary_result(
        _data(
            _ff().LLMContextSummaryResultFrame(
                request_id="r1",
                summary="ignored",
                last_summarized_index=3,
                error="boom",
            )
        )
    )

    assert turn.attributes["llm.summary.error"] == "boom"
    assert "llm.summary.text" not in turn.attributes


@pytest.mark.asyncio
async def test_summary_request_numeric_coercion_on_turn() -> None:
    # Guards: numeric coercion of summary-request fields onto the turn target.
    obs, _trace, turn = _new_obs()
    ctx = types.SimpleNamespace(get_messages=lambda: [], tools=None)

    req = _ff().LLMContextSummaryRequestFrame(
        request_id="r2",
        context=ctx,
        min_messages_to_keep=2,
        target_context_tokens=8000,
        summarization_prompt="p",
        summarization_timeout=30.0,
    )
    await obs._handle_llm_summary_request(_data(req))

    min_keep = turn.attributes["llm.summary.request.min_messages_to_keep"]
    tgt = turn.attributes["llm.summary.request.target_context_tokens"]
    timeout = turn.attributes["llm.summary.request.summarization_timeout_sec"]
    assert min_keep == 2 and isinstance(min_keep, int)
    assert tgt == 8000 and isinstance(tgt, int)
    assert timeout == 30.0 and isinstance(timeout, float)
