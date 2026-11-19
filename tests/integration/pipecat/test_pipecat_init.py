"""Tests for noveum_trace.integrations.pipecat public API."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import types

import pytest


def test_pipecat_observer_import_without_pipecat_extra() -> None:
    """Importing setup_pipecat_tracing must not require pipecat-ai to be installed.

    noveum-trace declares pipecat-ai as an optional extra. The module-level import
    must succeed in a base install so that the integrations/__init__.py try/except
    guard works as intended — transports.py wraps every pipecat import in its own
    try/except and falls back to ImportError-raising stubs.
    """

    # Remove any already-cached pipecat integration modules so we can reload them
    # with the pipecat package itself blocked from being importable.
    pipecat_noveum_mods = [
        k
        for k in list(sys.modules)
        if k.startswith("noveum_trace.integrations.pipecat")
    ]
    for mod in pipecat_noveum_mods:
        del sys.modules[mod]

    # Block pipecat itself so we simulate an environment without pipecat-ai.
    pipecat_mods = [
        k for k in list(sys.modules) if k == "pipecat" or k.startswith("pipecat.")
    ]
    saved = {k: sys.modules.pop(k) for k in pipecat_mods}

    types.ModuleType("pipecat")

    class _BlockPipecat:
        def find_spec(self, fullname: str, path: object, target: object = None) -> None:  # type: ignore[override]
            if fullname == "pipecat" or fullname.startswith("pipecat."):
                raise ModuleNotFoundError(f"No module named {fullname!r}")
            return None

    blocker = _BlockPipecat()
    sys.meta_path.insert(0, blocker)
    try:
        from noveum_trace.integrations.pipecat import (  # noqa: F401
            setup_pipecat_tracing,
        )

        assert callable(setup_pipecat_tracing)
    finally:
        sys.meta_path.remove(blocker)
        # Restore original pipecat modules (if they were present before)
        sys.modules.update(saved)
        # Remove any pipecat integration modules loaded with stubs so later tests
        # can re-import them against the real pipecat.
        for k in list(sys.modules):
            if k.startswith("noveum_trace.integrations.pipecat"):
                del sys.modules[k]


def test_setup_pipecat_tracing_returns_observer() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat import (
        NoveumTraceObserver,
        setup_pipecat_tracing,
    )

    obs = setup_pipecat_tracing(record_audio=True)
    assert isinstance(obs, NoveumTraceObserver)
    assert obs._record_audio is True


def test_pipecat_all_exports() -> None:
    import noveum_trace.integrations.pipecat as m

    assert set(m.__all__) == {
        "NoveumTraceObserver",
        "NoveumPipecatTracer",
        "NoveumCustomSpanProcessor",
        "setup_pipecat_tracing",
        "NoveumRawAudioTapMixin",
        "NoveumDailyTransport",
        "NoveumLiveKitTransport",
        "NoveumSmallWebRTCTransport",
        "NoveumFastAPIWebsocketTransport",
        "NoveumWebsocketServerTransport",
        "NoveumWebsocketClientTransport",
        "NoveumLocalAudioTransport",
        "NoveumTkTransport",
        "NoveumTavusTransport",
        "NoveumHeyGenTransport",
        "NoveumLemonSliceTransport",
    }


def test_setup_pipecat_tracing_forwards_kwargs_to_observer() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat import setup_pipecat_tracing

    obs = setup_pipecat_tracing(
        record_audio=False,
        record_raw_input_audio=False,
        trace_name_prefix="custom",
    )
    assert obs._record_audio is False
    assert obs._record_raw_input_audio is False
    assert obs._trace_name_prefix == "custom"


def test_setup_pipecat_tracing_rejects_api_key() -> None:
    pytest.importorskip("pipecat.observers.base_observer")

    from noveum_trace.integrations.pipecat import setup_pipecat_tracing

    with pytest.raises(TypeError, match="api_key"):
        setup_pipecat_tracing(api_key="x")  # type: ignore[call-arg]


def test_import_noveum_trace_does_not_load_pipecat_transports() -> None:
    """``import noveum_trace`` must not try every Pipecat transport.

    The package is imported whenever pipecat-ai is installed, and Pipecat logs an
    ERROR for each transport whose optional extra is missing. Runs in a fresh
    interpreter because this session has already imported the transports.
    """
    pytest.importorskip("pipecat")

    script = textwrap.dedent("""
        import json
        import sys

        from loguru import logger

        logger.remove()
        errors = []
        logger.add(lambda message: errors.append(message.record["name"]), level="ERROR")

        import noveum_trace  # noqa: F401

        print(json.dumps({
            "transports_loaded": (
                "noveum_trace.integrations.pipecat.transports" in sys.modules
            ),
            "pipecat_errors": [name for name in errors if name.startswith("pipecat.")],
        }))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {"transports_loaded": False, "pipecat_errors": []}


def test_transport_wrappers_load_on_first_access() -> None:
    pytest.importorskip("pipecat")

    import noveum_trace.integrations.pipecat as m
    from noveum_trace.integrations.pipecat import transports

    assert m.NoveumSmallWebRTCTransport is transports.NoveumSmallWebRTCTransport
    assert m.NoveumRawAudioTapMixin is transports.NoveumRawAudioTapMixin
    with pytest.raises(AttributeError):
        m.NoveumNotATransport  # noqa: B018
