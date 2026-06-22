"""
Fixtures for LiveKit behavioral tests.

``lk_client`` / ``lk_trace`` give a REAL ``NoveumClient`` (transport mocked) plus
a REAL, non-noop ``Trace`` set as current — so tests inspect the actual
``trace.spans`` the integration produces (names, attribute values, parents,
status) instead of asserting "a mock was called".
"""

from __future__ import annotations

import pytest


@pytest.fixture
def lk_client(client_with_mocked_transport):
    """Real client with mocked transport, registered as the global client.

    Delegates to the repo-wide ``client_with_mocked_transport`` fixture so
    ``noveum_trace.get_client()`` resolves to it inside the integration code.
    """
    return client_with_mocked_transport


@pytest.fixture
def lk_trace(lk_client):
    """Start a real trace, set it as the current trace, and yield it.

    ``client.start_trace`` already sets the trace as current (default config has
    ``tracing.enabled=True`` and ``sample_rate=1.0`` so it is a real trace, not a
    no-op). We additionally clear the current span so each test starts clean.
    """
    from noveum_trace.core.context import set_current_span, set_current_trace

    trace = lk_client.start_trace("livekit.test_session")
    set_current_trace(trace)
    set_current_span(None)
    try:
        yield trace
    finally:
        set_current_trace(None)
        set_current_span(None)
