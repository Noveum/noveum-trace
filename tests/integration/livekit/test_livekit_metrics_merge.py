"""
LLM metrics extraction / merge logic (Section F of LIVEKIT_TEST_PLAN.md).

Pure, synchronous manager logic. The token/cost *summation* and multi-model
dedup are high-value regression targets and were previously untested.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("livekit.agents")

from noveum_trace.integrations.livekit.livekit_session import (  # noqa: E402
    _LiveKitTracingManager,
)
from noveum_trace.integrations.livekit.livekit_utils import (  # noqa: E402
    update_speech_span_with_chat_items,
)


def _manager():
    return _LiveKitTracingManager(session=SimpleNamespace())


# --------------------------------------------------------------------------- #
# F1 — field mapping + unit conversions (seconds -> ms)
# --------------------------------------------------------------------------- #
def test_build_llm_metrics_maps_fields_and_converts_units():
    metrics = SimpleNamespace(
        type="llm_metrics",
        speech_id="sp1",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        ttft=0.2,
        tokens_per_second=25.0,
        duration=1.0,
        request_id="r1",
        cancelled=False,
        metadata=SimpleNamespace(model_name="gpt-4o", model_provider="openai"),
    )
    out = _manager()._build_llm_metrics(metrics)

    assert out["llm.input_tokens"] == 10
    assert out["llm.output_tokens"] == 5
    assert out["llm.total_tokens"] == 15
    assert out["llm.model"] == "gpt-4o"
    assert out["llm.provider"] == "openai"
    assert out["llm.time_to_first_token_ms"] == 200.0  # 0.2s -> ms
    assert out["llm.tokens_per_second"] == 25.0
    assert out["llm.latency_ms"] == 1000.0  # 1.0s -> ms
    assert out["llm.request_id"] == "r1"
    assert out["llm.cancelled"] is False
    assert "llm.cost.total" in out  # model + input tokens -> cost computed


# --------------------------------------------------------------------------- #
# F2 — cost only added when model + input tokens are present
# --------------------------------------------------------------------------- #
def test_cost_only_added_with_model_and_input_tokens():
    mgr = _manager()
    with_model = {"llm.model": "gpt-4o", "llm.input_tokens": 10, "llm.output_tokens": 5}
    mgr._add_llm_cost_metrics(with_model)
    assert "llm.cost.total" in with_model

    no_model = {"llm.input_tokens": 10}
    mgr._add_llm_cost_metrics(no_model)
    assert "llm.cost.total" not in no_model


# --------------------------------------------------------------------------- #
# F3 — extraction guards
# --------------------------------------------------------------------------- #
def test_extract_returns_none_for_invalid_events():
    mgr = _manager()
    assert mgr._extract_llm_metrics_from_event(SimpleNamespace(metrics=None)) is None
    assert (
        mgr._extract_llm_metrics_from_event(
            SimpleNamespace(metrics=SimpleNamespace(type="tts_metrics"))
        )
        is None
    )
    assert (
        mgr._extract_llm_metrics_from_event(
            SimpleNamespace(metrics=SimpleNamespace(type="llm_metrics", speech_id=None))
        )
        is None
    )


def test_extract_returns_speech_id_and_metrics_for_valid_event():
    """Positive case: a valid llm_metrics event yields (speech_id, metrics dict).
    Without this, disabling extraction entirely would slip past the guard tests."""
    ev = SimpleNamespace(
        metrics=SimpleNamespace(
            type="llm_metrics",
            speech_id="sp1",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            metadata=SimpleNamespace(model_name="gpt-4o", model_provider="openai"),
        )
    )
    result = _manager()._extract_llm_metrics_from_event(ev)
    assert result is not None
    speech_id, metrics = result
    assert speech_id == "sp1"
    assert metrics["llm.input_tokens"] == 10
    assert metrics["llm.total_tokens"] == 15


# --------------------------------------------------------------------------- #
# F4 — store + merge: token/cost sums and multi-model dedup
# --------------------------------------------------------------------------- #
def test_store_and_merge_sums_tokens_and_dedups_models():
    """Guards: repeated metrics for the same speech accumulate tokens/cost and
    record distinct models in ``llm.models`` while keeping the first as primary.
    (Mutation-tested: breaking the summation flips this red.)"""
    mgr = _manager()
    mgr._store_llm_metrics(
        "sp1",
        {
            "llm.input_tokens": 10,
            "llm.output_tokens": 5,
            "llm.total_tokens": 15,
            "llm.model": "model-a",
            "llm.cost.total": 0.1,
        },
    )
    merged = mgr._store_llm_metrics(
        "sp1",
        {
            "llm.input_tokens": 20,
            "llm.output_tokens": 10,
            "llm.total_tokens": 30,
            "llm.model": "model-b",
            "llm.cost.total": 0.2,
        },
    )

    assert merged["llm.input_tokens"] == 30
    assert merged["llm.output_tokens"] == 15
    assert merged["llm.total_tokens"] == 45
    assert round(merged["llm.cost.total"], 6) == 0.3
    assert merged["llm.model"] == "model-a"  # primary preserved
    assert merged["llm.models"] == ["model-a", "model-b"]
    # state persisted on the manager
    assert mgr._pending_llm_metrics["sp1"]["llm.total_tokens"] == 45

    # repeating an already-seen model must NOT grow llm.models (dedup branch)
    merged2 = mgr._store_llm_metrics(
        "sp1",
        {"llm.input_tokens": 1, "llm.total_tokens": 1, "llm.model": "model-a"},
    )
    assert merged2["llm.models"] == ["model-a", "model-b"]


# --------------------------------------------------------------------------- #
# F5 — update tracked speech span
# --------------------------------------------------------------------------- #
def test_update_speech_span_with_metrics(lk_trace, lk_client):
    mgr = _manager()
    span = lk_client.start_span("livekit.speech_created", attributes={})
    mgr._speech_spans["sp1"] = span

    mgr._update_speech_span_with_metrics("sp1", {"llm.total_tokens": 42})
    assert span.attributes["llm.total_tokens"] == 42

    # untracked speech id is a safe no-op
    mgr._update_speech_span_with_metrics("nope", {"llm.total_tokens": 1})


# --------------------------------------------------------------------------- #
# F6 — correlation: pending metrics flushed onto span when speech completes
# --------------------------------------------------------------------------- #
async def test_pending_metrics_merged_on_speech_completion(lk_trace, lk_client):
    """Guards: metrics that arrived before playout finished are flushed onto the
    speech span and tracking dicts are cleaned (no leak)."""
    mgr = _manager()
    span = lk_client.start_span("livekit.speech_created", attributes={})
    mgr._speech_spans["sp1"] = span
    mgr._pending_llm_metrics["sp1"] = {"llm.total_tokens": 99, "llm.model": "gpt-4o"}

    handle = SimpleNamespace(id="sp1", chat_items=[], wait_for_playout=AsyncMock())
    await update_speech_span_with_chat_items(handle, span, mgr)

    # waiting for playout before flushing IS the ordering contract
    handle.wait_for_playout.assert_awaited()
    assert span.attributes["llm.total_tokens"] == 99
    assert "sp1" not in mgr._speech_spans
    assert "sp1" not in mgr._pending_llm_metrics


async def test_chat_items_serialized_onto_speech_span(lk_trace, lk_client):
    """Guards: completed speech chat_items are serialized onto the span."""
    mgr = _manager()
    span = lk_client.start_span("livekit.speech_created", attributes={})
    mgr._speech_spans["sp1"] = span
    msg = SimpleNamespace(type="message", role="user", content="hi", interrupted=False)
    handle = SimpleNamespace(id="sp1", chat_items=[msg], wait_for_playout=AsyncMock())

    await update_speech_span_with_chat_items(handle, span, mgr)

    assert span.attributes["speech.chat_items.count"] == 1
    assert span.attributes["speech.messages"][0]["content"] == "hi"


async def test_tracking_cleaned_when_playout_raises(lk_trace, lk_client):
    """Guards: an exception while waiting for playout must still clean both
    tracking dicts (no memory leak)."""
    mgr = _manager()
    span = lk_client.start_span("livekit.speech_created", attributes={})
    mgr._speech_spans["sp1"] = span
    mgr._pending_llm_metrics["sp1"] = {"llm.total_tokens": 1}
    handle = SimpleNamespace(
        id="sp1",
        chat_items=[],
        wait_for_playout=AsyncMock(side_effect=RuntimeError("x")),
    )

    await update_speech_span_with_chat_items(handle, span, mgr)  # must not raise

    assert "sp1" not in mgr._speech_spans
    assert "sp1" not in mgr._pending_llm_metrics


# --------------------------------------------------------------------------- #
# End-to-end: the live _on_metrics_collected handler wires extract->store->update
# --------------------------------------------------------------------------- #
async def test_on_metrics_collected_updates_tracked_speech_span(lk_trace, lk_client):
    """Guards: the real handler path correctly threads metrics from the event
    onto the tracked speech span (a miswire between stages would be invisible to
    the isolated unit tests)."""
    mgr = _manager()
    speech_span = lk_client.start_span("livekit.speech_created", attributes={})
    mgr._speech_spans["sp1"] = speech_span

    ev = SimpleNamespace(
        metrics=SimpleNamespace(
            type="llm_metrics",
            speech_id="sp1",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            metadata=SimpleNamespace(model_name="gpt-4o", model_provider="openai"),
        )
    )

    before = set(asyncio.all_tasks())
    mgr._on_metrics_collected(ev)
    # drain only the task(s) this handler scheduled (deterministic; no sleep) --
    # awaiting every pending task could pull in unrelated background tasks.
    created = [
        t
        for t in (set(asyncio.all_tasks()) - before)
        if t is not asyncio.current_task()
    ]
    if created:
        await asyncio.gather(*created)

    assert speech_span.attributes["llm.input_tokens"] == 10
    assert speech_span.attributes["llm.total_tokens"] == 15
    assert speech_span.attributes["llm.model"] == "gpt-4o"
