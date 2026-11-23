"""Traces larger than the request limit are sent in chunks the API can merge."""

import json
from unittest.mock import Mock, patch

import pytest

from noveum_trace.core.config import Config, ConfigurationError
from noveum_trace.transport.http_transport import HttpTransport
from noveum_trace.transport.trace_chunking import (
    REQUEST_ENVELOPE_BYTES,
    TraceTooLargeError,
    encoded_size,
    plan_trace_requests,
    split_trace,
)
from noveum_trace.utils.exceptions import TransportError

LIMIT = 64 * 1024


def make_trace(trace_id: str, span_count: int, span_bytes: int = 1000) -> dict:
    return {
        "trace_id": trace_id,
        "name": "novasynth_phone.agent_session",
        "start_time": "2026-11-23T15:04:40.487Z",
        "end_time": "2026-11-23T15:35:04.868Z",
        "duration_ms": 1_824_381.0,
        "status": "ok",
        "span_count": span_count,
        "error_count": 0,
        "project": "novasynth-phone",
        "environment": "production",
        "sdk": {"name": "noveum-trace-python", "version": "1.6.0"},
        "attributes": {"llm.system_prompt": "ROLE: You are the CALLER"},
        "metadata": {"tags": {}, "custom_attributes": {}},
        "spans": [
            {
                "span_id": f"{trace_id}-s{index}",
                "trace_id": trace_id,
                "name": "llm.chat",
                "attributes": {"llm.response": "x" * span_bytes},
            }
            for index in range(span_count)
        ],
    }


def request_size(traces: list[dict]) -> int:
    return encoded_size({"traces": traces, "timestamp": 1795446075.6784413})


class TestSplitTrace:
    def test_a_trace_that_fits_is_returned_unchanged(self):
        trace = make_trace("t-small", 10)

        assert split_trace(trace, LIMIT) == [trace]

    def test_a_large_trace_becomes_chunks_that_each_fit(self):
        trace = make_trace("t-call", 300)

        chunks = split_trace(trace, LIMIT)

        assert len(chunks) > 1
        for chunk in chunks:
            assert request_size([chunk]) <= LIMIT

    def test_chunks_keep_every_span_once_and_in_order(self):
        trace = make_trace("t-call", 300)

        chunks = split_trace(trace, LIMIT)

        span_ids = [span["span_id"] for chunk in chunks for span in chunk["spans"]]
        assert span_ids == [span["span_id"] for span in trace["spans"]]

    def test_every_chunk_carries_the_same_trace_fields_and_the_total_span_count(self):
        trace = make_trace("t-call", 300)

        chunks = split_trace(trace, LIMIT)

        headers = [
            {k: v for k, v in chunk.items() if k not in ("spans", "chunk")}
            for chunk in chunks
        ]
        assert all(header == headers[0] for header in headers)
        assert headers[0]["span_count"] == 300
        assert headers[0]["trace_id"] == "t-call"
        assert [chunk["chunk"] for chunk in chunks] == [
            {"index": index, "count": len(chunks)} for index in range(len(chunks))
        ]

    def test_the_total_span_count_is_taken_from_the_spans(self):
        trace = {**make_trace("t-call", 300), "span_count": 12}

        chunks = split_trace(trace, LIMIT)

        assert {chunk["span_count"] for chunk in chunks} == {300}

    def test_a_span_larger_than_the_budget_gets_a_chunk_of_its_own(self):
        trace = make_trace("t-call", 3)
        trace["spans"][1]["attributes"]["llm.response"] = "y" * (2 * LIMIT)

        chunks = split_trace(trace, LIMIT)

        assert [len(chunk["spans"]) for chunk in chunks] == [1, 1, 1]
        assert chunks[1]["spans"][0]["span_id"] == "t-call-s1"

    def test_a_trace_whose_fields_alone_do_not_fit_is_refused(self):
        trace = make_trace("t-call", 3)
        trace["attributes"]["conversation.history"] = "z" * LIMIT

        with pytest.raises(TraceTooLargeError):
            split_trace(trace, LIMIT)

    def test_too_large_is_a_transport_error(self):
        assert issubclass(TraceTooLargeError, TransportError)


class TestPlanTraceRequests:
    def test_small_traces_share_one_request_in_order(self):
        traces = [make_trace(f"t-{index}", 2) for index in range(5)]

        assert plan_trace_requests(traces, LIMIT) == [traces]

    def test_small_traces_are_spread_over_requests_that_each_fit(self):
        traces = [make_trace(f"t-{index}", 20) for index in range(10)]

        requests = plan_trace_requests(traces, LIMIT)

        assert len(requests) > 1
        assert [t for request in requests for t in request] == traces
        for request in requests:
            assert request_size(request) <= LIMIT

    def test_a_large_trace_is_sent_alone_one_chunk_per_request_last_chunk_last(self):
        before = make_trace("t-before", 2)
        large = make_trace("t-call", 300)
        after = make_trace("t-after", 2)

        requests = plan_trace_requests([before, large, after], LIMIT)

        assert requests[0] == [before]
        assert requests[-1] == [after]
        chunk_requests = requests[1:-1]
        assert all(len(request) == 1 for request in chunk_requests)
        indexes = [request[0]["chunk"]["index"] for request in chunk_requests]
        assert indexes == list(range(len(chunk_requests)))

    def test_the_envelope_allowance_covers_the_request_wrapper(self):
        assert request_size([]) <= REQUEST_ENVELOPE_BYTES


def transport_with(max_request_bytes: int, statuses: list[int]) -> HttpTransport:
    config = Config.create(api_key="test-key", project="p")
    config.transport.max_request_bytes = max_request_bytes
    with patch("noveum_trace.transport.http_transport.BatchProcessor"):
        transport = HttpTransport(config)
    responses = [Mock(status_code=status, text="{}") for status in statuses]
    transport.session.post = Mock(side_effect=responses)
    return transport


def posted_traces(transport: HttpTransport) -> list[list[dict]]:
    return [
        call.kwargs["json"]["traces"] for call in transport.session.post.call_args_list
    ]


class TestHttpTransportSendsChunks:
    def test_a_large_trace_is_posted_as_chunks_in_order(self):
        trace = make_trace("t-call", 300)
        expected = split_trace(trace, LIMIT)
        transport = transport_with(LIMIT, [200] * len(expected))

        transport._send_trace_batch([trace])

        bodies = posted_traces(transport)
        assert bodies == [[chunk] for chunk in expected]
        for call in transport.session.post.call_args_list:
            assert len(json.dumps(call.kwargs["json"]).encode()) <= LIMIT

    def test_a_failed_chunk_stops_the_batch_and_raises(self):
        trace = make_trace("t-call", 300)
        chunk_count = len(split_trace(trace, LIMIT))
        transport = transport_with(LIMIT, [200, 413] + [200] * chunk_count)

        with pytest.raises(TransportError):
            transport._send_trace_batch([trace])

        assert transport.session.post.call_count == 2

    def test_a_batch_that_fits_is_one_request_as_before(self):
        traces = [make_trace("t-a", 2), make_trace("t-b", 2)]
        transport = transport_with(LIMIT, [200])

        transport._send_trace_batch(traces)

        assert posted_traces(transport) == [traces]
        assert all("chunk" not in trace for trace in posted_traces(transport)[0])

    def test_the_default_limit_leaves_room_under_the_api_limit(self):
        config = Config.create(api_key="test-key", project="p")

        assert config.transport.max_request_bytes == 8 * 1024 * 1024
        assert config.to_dict()["transport"]["max_request_bytes"] == 8 * 1024 * 1024

    def test_the_limit_is_read_from_a_config_dict(self):
        config = Config.from_dict(
            {"api_key": "k", "transport": {"max_request_bytes": 1_048_576}}
        )

        assert config.transport.max_request_bytes == 1_048_576

    def test_a_limit_too_small_to_hold_a_trace_is_refused(self):
        with pytest.raises(ConfigurationError):
            Config.from_dict({"api_key": "k", "transport": {"max_request_bytes": 100}})
