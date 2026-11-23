"""
Split traces into requests that fit under the ingestion body limit.

The Noveum API rejects a request body above its limit (50 MiB for
``/api/v1/traces``) with HTTP 413, and the trace is lost. A long voice call can
produce a trace above that. A trace whose serialized size exceeds
``max_request_bytes`` is therefore sent as several requests ("chunks"): every
chunk repeats the trace fields with the total ``span_count``, carries its own
share of the spans, is marked with ``chunk = {"index": i, "count": n}`` and is
sent to ``/v1/traces/chunks``, which older platforms do not have, so they
answer 404 instead of storing a chunk as a whole trace.
The platform stores each chunk's spans under the same trace and runs its
post-ingest work once, when the last chunk has arrived.

Traces that fit are left untouched and packed together into as few requests
as the limit allows, so small traces are sent exactly as before.
"""

from __future__ import annotations

import json
from typing import Any

MAX_TRACE_CHUNKS = 1_000
REQUEST_ENVELOPE_BYTES = 256
_SPAN_SEPARATOR_BYTES = 2


def encoded_size(value: Any) -> int:
    """Bytes ``value`` takes in a request body, serialized as the HTTP client does."""
    return len(json.dumps(value, default=str).encode("utf-8"))


def split_trace(trace: dict[str, Any], max_request_bytes: int) -> list[dict[str, Any]]:
    """Return ``trace`` as one or more traces that each fit in a request.

    A trace that fits is returned unchanged. A larger one is returned as
    chunks in send order. A single span larger than the budget gets a chunk of
    its own, which the API may still reject; nothing is dropped here. A trace
    whose fields alone do not fit, or that would need more than
    ``MAX_TRACE_CHUNKS`` chunks, is returned whole, as before chunking: the
    API's own limit is far above the default here and may still accept it.
    """
    spans = list(trace.get("spans") or [])
    if (
        encoded_size(trace) + REQUEST_ENVELOPE_BYTES <= max_request_bytes
        or len(spans) <= 1
    ):
        return [trace]

    header = {key: value for key, value in trace.items() if key != "spans"}
    header["span_count"] = len(spans)
    header_size = encoded_size(
        {
            **header,
            "spans": [],
            "chunk": {"index": MAX_TRACE_CHUNKS, "count": MAX_TRACE_CHUNKS},
        }
    )
    budget = max_request_bytes - REQUEST_ENVELOPE_BYTES - header_size
    if budget <= 0:
        return [trace]

    groups: list[list[Any]] = []
    current: list[Any] = []
    current_size = 0
    for span in spans:
        span_size = encoded_size(span) + _SPAN_SEPARATOR_BYTES
        if current and current_size + span_size > budget:
            groups.append(current)
            current, current_size = [], 0
        current.append(span)
        current_size += span_size
    groups.append(current)

    if len(groups) > MAX_TRACE_CHUNKS:
        return [trace]
    return [
        {**header, "spans": group, "chunk": {"index": index, "count": len(groups)}}
        for index, group in enumerate(groups)
    ]


def plan_trace_requests(
    traces: list[dict[str, Any]], max_request_bytes: int
) -> list[list[dict[str, Any]]]:
    """Group ``traces`` into request bodies that each fit under the limit.

    Traces that fit are packed together in their original order. A trace that
    has to be chunked is sent as its own requests, one chunk per request, with
    the last chunk sent last. A trace too large to fit even alone is sent in a
    request of its own, so it cannot take other traces down with it.
    """
    requests: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    pending_size = REQUEST_ENVELOPE_BYTES

    def flush() -> None:
        nonlocal pending, pending_size
        if pending:
            requests.append(pending)
            pending, pending_size = [], REQUEST_ENVELOPE_BYTES

    for trace in traces:
        pieces = split_trace(trace, max_request_bytes)
        if len(pieces) > 1:
            flush()
            requests.extend([piece] for piece in pieces)
            continue
        trace_size = encoded_size(trace) + _SPAN_SEPARATOR_BYTES
        if pending and pending_size + trace_size > max_request_bytes:
            flush()
        pending.append(trace)
        pending_size += trace_size
        if pending_size > max_request_bytes:
            flush()
    flush()
    return requests
