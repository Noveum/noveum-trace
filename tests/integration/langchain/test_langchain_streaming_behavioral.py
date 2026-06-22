"""Behavioral tests for the LangChain handler's streaming TTFT path.

These tests inspect the *real* spans produced by ``NoveumTraceCallbackHandler``
during (and around) streaming responses.  Unlike the mock-interaction tests
elsewhere, the real-runnable test drives an actual multi-token
``FakeListChatModel.stream(...)`` and asserts that time-to-first-token metrics
land exactly once on a captured span; the direct-callback tests pin down the
two edge branches of ``on_llm_new_token`` that a real stream can never reach
(an unknown run id, and a registered span whose ``start_time`` is falsy).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from ._helpers import (
    LANGCHAIN_AVAILABLE,
    attrs,
    find_span,
    get_exported_spans,
)

pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="LangChain not available"
)


def test_streaming_ttft_recorded_once_real(client_with_mocked_transport):
    """A real multi-token chat stream records TTFT metrics exactly once.

    ``FakeListChatModel(responses=["hello world"]).stream(...)`` emits many
    per-character chunks, firing ``on_llm_new_token`` repeatedly.  TTFT must be
    recorded only on the *first* token: the captured span should carry
    ``llm.streaming`` True, a single ``llm.first_token_time``, and a
    non-negative ``llm.time_to_first_token_ms``.
    """
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import HumanMessage

    import noveum_trace
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    client = client_with_mocked_transport
    handler = NoveumTraceCallbackHandler()

    model = FakeListChatModel(responses=["hello world"])
    chunks = list(model.stream([HumanMessage("hi")], config={"callbacks": [handler]}))
    # Sanity: this really is a multi-token stream (per-character chunks),
    # so on_llm_new_token fired more than once.
    assert len(chunks) > 1

    noveum_trace.flush()

    # The first-token tracking set must be drained after on_llm_end.
    assert handler._first_token_received == set()

    span = find_span(client, predicate=lambda s: attrs(s).get("llm.streaming") is True)
    a = attrs(span)

    # Streaming markers recorded.
    assert a.get("llm.streaming") is True
    assert a.get("llm.first_token_time") is not None
    # ISO-8601 timestamp string (recorded once, not a list).
    assert isinstance(a["llm.first_token_time"], str)
    assert "T" in a["llm.first_token_time"]

    # TTFT present and non-negative (do not assert an exact timing value).
    assert "llm.time_to_first_token_ms" in a
    assert a["llm.time_to_first_token_ms"] >= 0

    # This is the LLM span for the fake chat model.
    assert span.name == "llm.fake_chat_models"


def test_new_token_unknown_run_id_discards_for_retry(client_with_mocked_transport):
    """An unknown run id is discarded from tracking so a later token can record.

    Calling ``on_llm_new_token`` for a run with no registered span must leave
    the run id absent from ``_first_token_received`` (discard-for-retry).  Once
    a span is registered, a subsequent token call then records TTFT -- proving
    the first call did not poison future recording for that run id.
    """
    from noveum_trace.core.span import Span
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    handler = NoveumTraceCallbackHandler()
    run_id = uuid4()

    # First token with no span registered: discarded, nothing recorded.
    handler.on_llm_new_token("tok", run_id=run_id)
    assert run_id not in handler._first_token_received

    # Now register a real span and retry; TTFT gets recorded this time.
    span = Span(name="llm.retry", trace_id="trace-retry")
    span.start_time = datetime.now(timezone.utc)
    handler._set_run(run_id, span)

    handler.on_llm_new_token("tok", run_id=run_id)

    assert run_id in handler._first_token_received
    a = attrs(span)
    assert a.get("llm.streaming") is True
    assert a.get("llm.first_token_time") is not None
    assert "llm.time_to_first_token_ms" in a
    assert a["llm.time_to_first_token_ms"] >= 0

    # No trace was finalized in this direct-callback test.
    assert get_exported_spans(client_with_mocked_transport) == []


def test_new_token_missing_start_time_omits_ttft(client_with_mocked_transport):
    """A registered span with a falsy ``start_time`` records streaming markers
    but omits the TTFT duration.

    When ``span.start_time`` is falsy, ``on_llm_new_token`` still sets
    ``llm.first_token_time`` and ``llm.streaming`` True, but cannot compute a
    duration, so ``llm.time_to_first_token_ms`` must be absent.
    """
    from noveum_trace.core.span import Span
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    handler = NoveumTraceCallbackHandler()
    run_id = uuid4()

    span = Span(name="llm.no_start", trace_id="trace-no-start")
    span.start_time = None  # falsy -> ttft computation skipped
    handler._set_run(run_id, span)

    handler.on_llm_new_token("tok", run_id=run_id)

    # Span was found, so the run id remains tracked (not discarded).
    assert run_id in handler._first_token_received

    a = attrs(span)
    # Streaming markers still set despite missing start_time.
    assert a.get("llm.streaming") is True
    assert a.get("llm.first_token_time") is not None
    assert isinstance(a["llm.first_token_time"], str)
    # TTFT duration omitted because start_time was falsy.
    assert "llm.time_to_first_token_ms" not in a
