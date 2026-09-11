"""Test-only Anthropic client compatibility; all requests still traverse Guard."""

from __future__ import annotations

import importlib

import anthropic
import httpx

from noveum_trace.guard.transport import NoveumAsyncTransport, NoveumTransport

# Use the HTTP package actually used by this SDK, even when both are installed.
_native = importlib.import_module(
    next(
        cls.__module__.split(".")[0]
        for cls in anthropic.DefaultHttpxClient.__mro__
        if cls.__module__.split(".")[0] in {"httpx", "httpx2"}
    )
)


def _request(request, content):
    return httpx.Request(
        request.method,
        str(request.url),
        headers=request.headers.raw,
        content=content,
        extensions=request.extensions,
    )


class _SyncStream(_native.SyncByteStream):
    def __init__(self, response):
        self.response = response

    def __iter__(self):
        if self.response.is_stream_consumed:
            yield self.response.content
        else:
            yield from self.response.stream

    def close(self):
        self.response.close()


class _AsyncStream(_native.AsyncByteStream):
    def __init__(self, response):
        self.response = response

    async def __aiter__(self):
        if self.response.is_stream_consumed:
            yield self.response.content
        else:
            async for chunk in self.response.stream:
                yield chunk

    async def aclose(self):
        await self.response.aclose()


def _response(response, stream):
    headers = response.headers.copy()
    if response.is_stream_consumed:
        # Guard has already read/decoded non-streaming responses.
        headers.pop("content-encoding", None)
        headers.pop("content-length", None)
    return _native.Response(
        response.status_code,
        headers=headers.raw,
        stream=stream(response),
        extensions=response.extensions,
    )


class _SyncTransport(_native.BaseTransport):
    def __init__(self, guard):
        self.guard = guard

    def handle_request(self, request):
        response = self.guard.handle_request(_request(request, request.read()))
        return _response(response, _SyncStream)

    def close(self):
        self.guard.close()


class _AsyncTransport(_native.AsyncBaseTransport):
    def __init__(self, guard):
        self.guard = guard

    async def handle_async_request(self, request):
        response = await self.guard.handle_async_request(
            _request(request, await request.aread())
        )
        return _response(response, _AsyncStream)

    async def aclose(self):
        await self.guard.aclose()


def http_client(engine, context, *, inner=None):
    """Supply the SDK's native sync client with the real Guard transport."""
    guard = NoveumTransport(engine, context, inner=inner)
    transport = guard if _native is httpx else _SyncTransport(guard)
    return anthropic.DefaultHttpxClient(transport=transport)


def async_http_client(engine, context, *, inner=None):
    """Supply the SDK's native async client with the real Guard transport."""
    guard = NoveumAsyncTransport(engine, context, inner=inner)
    transport = guard if _native is httpx else _AsyncTransport(guard)
    return anthropic.DefaultAsyncHttpxClient(transport=transport)
