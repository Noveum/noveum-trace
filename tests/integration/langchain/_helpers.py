"""Shared helpers for *behavioral* LangChain integration tests.

Unlike the older ``tests/**/test_langchain_*`` suites — which patch
``noveum_trace.get_client`` with a bare ``Mock`` and assert on mock
interactions — the tests in this package inspect the **real** spans that the
handler produces.  They rely on the repo-wide ``client_with_mocked_transport``
fixture (see ``tests/conftest.py``), which installs a real ``NoveumClient`` as
the global client with a ``Mock(spec=HttpTransport)`` transport.  Because
``NoveumTraceCallbackHandler.__init__`` calls ``get_client()``, a handler
constructed *inside* such a test captures that inspectable client, and every
finished trace is recorded on ``client.transport.export_trace``.

Typical usage::

    def test_something(client_with_mocked_transport):
        client = client_with_mocked_transport
        handler = NoveumTraceCallbackHandler()
        RunnableLambda(lambda x: x).invoke(1, config={"callbacks": [handler]})
        import noveum_trace
        noveum_trace.flush()
        spans = get_exported_spans(client)
        ...
"""

from __future__ import annotations

from typing import Any, Callable, Optional

try:  # pragma: no cover - import guard mirrors the rest of the suite
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    NoveumTraceCallbackHandler = None  # type: ignore[assignment,misc]
    LANGCHAIN_AVAILABLE = False


def get_exported_traces(client: Any) -> list:
    """Return every ``Trace`` object handed to ``transport.export_trace``."""
    traces = []
    for call in client.transport.export_trace.call_args_list:
        trace = call.args[0] if call.args else call.kwargs.get("trace")
        if trace is not None:
            traces.append(trace)
    return traces


def get_exported_spans(client: Any) -> list:
    """Return a flat list of every span across all exported traces."""
    spans: list = []
    for trace in get_exported_traces(client):
        spans.extend(getattr(trace, "spans", None) or [])
    return spans


def span_status(span: Any) -> Optional[str]:
    """Return the string value of a span's status (``"ok"``/``"error"``/``"unset"``)."""
    status = getattr(span, "status", None)
    return getattr(status, "value", status)


def attrs(span: Any) -> dict:
    """Return a span's attributes dict (never ``None``)."""
    return getattr(span, "attributes", None) or {}


def find_span(
    client: Any,
    *,
    name: Optional[str] = None,
    predicate: Optional[Callable[[Any], bool]] = None,
) -> Any:
    """Return the single exported span matching ``name`` and/or ``predicate``.

    Raises ``AssertionError`` if zero or more than one span matches, so tests
    fail loudly on ambiguity instead of silently picking the wrong span.
    """
    matches = []
    for span in get_exported_spans(client):
        if name is not None and getattr(span, "name", None) != name:
            continue
        if predicate is not None and not predicate(span):
            continue
        matches.append(span)
    assert len(matches) == 1, (
        f"expected exactly 1 span matching name={name!r}/predicate, "
        f"found {len(matches)}: {[getattr(s, 'name', None) for s in matches]}"
    )
    return matches[0]
