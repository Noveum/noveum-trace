"""Traces larger than the request limit are sent in chunks the API can merge."""

import json
import time
from unittest.mock import Mock, patch

import pytest

from noveum_trace.core.config import Config, ConfigurationError
from noveum_trace.transport.http_transport import HttpTransport
from noveum_trace.transport.trace_chunking import (
    _SPAN_SEPARATOR_BYTES,
    MAX_TRACE_CHUNKS,
    REQUEST_ENVELOPE_BYTES,
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

    def test_a_trace_whose_fields_alone_do_not_fit_is_sent_whole(self):
        trace = make_trace("t-call", 3)
        trace["attributes"]["conversation.history"] = "z" * LIMIT

        assert split_trace(trace, LIMIT) == [trace]

    def test_the_platform_chunk_cap_is_1000(self):
        assert MAX_TRACE_CHUNKS == 1000

    def test_a_trace_needing_exactly_1000_chunks_is_chunked(self):
        trace = make_trace("t-max", 1000, span_bytes=LIMIT)

        chunks = split_trace(trace, LIMIT)

        assert len(chunks) == 1000
        assert chunks[-1]["chunk"] == {"index": 999, "count": 1000}

    def test_a_trace_needing_more_than_1000_chunks_is_sent_whole(self):
        trace = make_trace("t-huge", 1001, span_bytes=LIMIT)

        assert split_trace(trace, LIMIT) == [trace]

    def test_a_trace_over_the_chunk_cap_does_not_take_others_with_it(self):
        small = make_trace("t-small", 2)
        huge = make_trace("t-huge", 1001, span_bytes=LIMIT)

        assert plan_trace_requests([small, huge, small], LIMIT) == [
            [small],
            [huge],
            [small],
        ]

    def test_a_trace_exactly_at_the_limit_is_not_split(self):
        trace = make_trace("t-edge", 10)
        size = encoded_size(trace) + REQUEST_ENVELOPE_BYTES

        assert split_trace(trace, size) == [trace]
        assert len(split_trace(trace, size - 1)) > 1

    def test_chunks_of_non_ascii_content_fit_once_escaped(self):
        trace = make_trace("t-hindi", 300)
        for span in trace["spans"]:
            span["attributes"]["llm.response"] = 'नमस्ते, आपका खाता 🙂 "quoted" ' * 40

        chunks = split_trace(trace, LIMIT)

        assert len(chunks) > 1
        for chunk in chunks:
            body = json.dumps({"traces": [chunk], "timestamp": time.time()})
            assert len(body.encode("utf-8")) <= LIMIT


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

    def test_packing_stops_at_the_limit_exactly(self):
        first, second = make_trace("t-a", 5), make_trace("t-b", 5)
        packed = (
            REQUEST_ENVELOPE_BYTES
            + encoded_size(first)
            + encoded_size(second)
            + 2 * _SPAN_SEPARATOR_BYTES
        )

        assert plan_trace_requests([first, second], packed) == [[first, second]]
        assert plan_trace_requests([first, second], packed - 1) == [
            [first],
            [second],
        ]

    def test_an_oversized_trace_is_sent_alone_and_does_not_take_others_with_it(self):
        small = make_trace("t-small", 2)
        oversized = make_trace("t-big", 2)
        oversized["attributes"]["conversation.history"] = "z" * LIMIT

        assert plan_trace_requests([small, oversized, small], LIMIT) == [
            [small],
            [oversized],
            [small],
        ]


def transport_with(
    max_request_bytes: int, statuses: list[int], **config_args
) -> HttpTransport:
    config = Config.create(api_key="test-key", project="p", **config_args)
    config.transport.max_request_bytes = max_request_bytes
    with patch("noveum_trace.transport.http_transport.BatchProcessor"):
        transport = HttpTransport(config)
    responses = [Mock(status_code=status, text="{}") for status in statuses]
    transport.session.post = Mock(side_effect=responses)
    return transport


def posted_urls(transport: HttpTransport) -> list[str]:
    return [call.args[0] for call in transport.session.post.call_args_list]


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
        assert all(url.endswith("/v1/traces/chunks") for url in posted_urls(transport))
        for call in transport.session.post.call_args_list:
            assert len(json.dumps(call.kwargs["json"]).encode()) <= LIMIT

    def test_a_failed_chunk_skips_the_rest_of_its_trace_but_not_other_traces(self):
        trace = make_trace("t-call", 300)
        after = make_trace("t-after", 2)
        chunk_count = len(split_trace(trace, LIMIT))
        transport = transport_with(LIMIT, [200, 413] + [200] * chunk_count)

        with pytest.raises(TransportError):
            transport._send_trace_batch([trace, after])

        bodies = posted_traces(transport)
        assert [body[0]["trace_id"] for body in bodies] == [
            "t-call",
            "t-call",
            "t-after",
        ]
        assert posted_urls(transport)[-1].endswith("/v1/traces")

    def test_a_batch_that_fits_is_one_request_as_before(self):
        traces = [make_trace("t-a", 2), make_trace("t-b", 2)]
        transport = transport_with(LIMIT, [200])

        transport._send_trace_batch(traces)

        assert posted_traces(transport) == [traces]
        assert all("chunk" not in trace for trace in posted_traces(transport)[0])
        assert posted_urls(transport)[0].endswith("/v1/traces")

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
            Config.from_dict(
                {"api_key": "k", "transport": {"max_request_bytes": 65_535}}
            )
        assert (
            Config.from_dict(
                {"api_key": "k", "transport": {"max_request_bytes": 65_536}}
            ).transport.max_request_bytes
            == 65_536
        )

    def test_the_synchronous_send_chunks_a_large_trace_too(self):
        trace = make_trace("t-call", 300)
        expected = split_trace(trace, LIMIT)
        transport = transport_with(LIMIT, [200] * len(expected))
        transport._format_trace_for_export = Mock(return_value=trace)

        transport.send_trace_now(Mock())

        assert posted_traces(transport) == [[chunk] for chunk in expected]
        assert all(url.endswith("/v1/traces/chunks") for url in posted_urls(transport))

    def test_the_first_error_of_a_batch_is_the_one_raised(self):
        first, second = make_trace("t-first", 2), make_trace("t-second", 2)
        first["attributes"]["conversation.history"] = "a" * LIMIT
        transport = transport_with(LIMIT, [429, 403])

        with pytest.raises(TransportError, match="Rate limit exceeded"):
            transport._send_trace_batch([first, second])

        assert transport.session.post.call_count == 2

    def test_dev_mode_saves_a_synchronously_sent_chunked_trace_whole(self, tmp_path):
        trace = make_trace("t-call", 300)
        chunk_count = len(split_trace(trace, LIMIT))
        transport = transport_with(
            LIMIT, [200] * chunk_count, dev_mode=True, dev_traces_dir=str(tmp_path)
        )
        transport._format_trace_for_export = Mock(return_value=trace)

        transport.send_trace_now(Mock())

        saved = json.loads((tmp_path / "t-call.json").read_text(encoding="utf-8"))
        assert len(saved["spans"]) == 300

    def test_dev_mode_saves_a_chunked_trace_whole(self, tmp_path):
        trace = make_trace("t-call", 300)
        chunk_count = len(split_trace(trace, LIMIT))
        transport = transport_with(
            LIMIT, [200] * chunk_count, dev_mode=True, dev_traces_dir=str(tmp_path)
        )

        transport._send_trace_batch([trace])

        saved = json.loads((tmp_path / "t-call.json").read_text(encoding="utf-8"))
        assert len(saved["spans"]) == 300
        assert "chunk" not in saved
