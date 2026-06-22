"""Unit tests for Pipecat Noveum transport wrappers."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _require(modname: str) -> Any:
    """Import ``modname`` or skip — robust to a missing *optional backend*.

    ``pytest.importorskip`` only skips when the requested module itself is
    absent. pytest>=9.1 re-raises (rather than skips) when importing the
    pipecat transport module fails on a *deeper* missing dependency — e.g. the
    optional ``daily`` / ``fastapi`` / webrtc backend — which makes these
    optional-transport tests ERROR on installs without that extra (notably the
    pipecat-0.0.x matrix leg, where transport backends are not installed).
    Treat any ImportError as "backend unavailable -> skip cleanly".
    """
    try:
        return importlib.import_module(modname)
    except ImportError as exc:  # 1.x style: a deep backend ModuleNotFoundError
        pytest.skip(f"{modname} unavailable (optional backend missing): {exc}")
    except Exception as exc:
        # 0.0.x style: pipecat re-raises a missing transport backend as a
        # *generic* Exception ("Missing module: No module named 'daily'"), not
        # ImportError (mirrors transports.py's own `except Exception`). Skip on
        # those; re-raise anything that is not a missing-backend signal.
        msg = str(exc)
        if any(s in msg for s in ("Missing module", "No module named", "pip install")):
            pytest.skip(f"{modname} unavailable (optional backend missing): {exc}")
        raise


@pytest.fixture
def ff():
    pytest.importorskip("pipecat.frames.frames")
    from pipecat.frames import frames as _ff

    return _ff


def test_capture_raw_input_audio_snapshots_bytes(ff) -> None:
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=True, record_raw_input_audio=True)
    obs._pipeline_has_stt = True
    buf = bytearray(b"\x01\x02\x03\x04")
    frame = ff.InputAudioRawFrame(audio=buf, sample_rate=16000, num_channels=1)
    obs.capture_raw_input_audio(frame)
    buf[:] = b"\xff\xff\xff\xff"
    assert len(obs._stt_raw_audio_buffer) == 1
    assert obs._stt_raw_audio_buffer[0].audio == b"\x01\x02\x03\x04"
    assert obs._stt_raw_audio_buffer[0].audio is not buf


def test_capture_raw_input_audio_noop_without_stt(ff) -> None:
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(record_audio=True, record_raw_input_audio=True)
    obs._pipeline_has_stt = False
    frame = ff.InputAudioRawFrame(audio=b"\x01", sample_rate=16000, num_channels=1)
    obs.capture_raw_input_audio(frame)
    assert obs._stt_raw_audio_buffer == []


@pytest.mark.asyncio
async def test_mixin_push_audio_frame_calls_observer(ff) -> None:
    from noveum_trace.integrations.pipecat.transports import NoveumRawAudioTapMixin

    observer = MagicMock()
    frame = ff.InputAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)

    class _Base:
        push_audio_frame = AsyncMock()

    class _Tap(NoveumRawAudioTapMixin, _Base):
        def __init__(self) -> None:
            self._noveum_observer = observer

    tap = _Tap()
    await tap.push_audio_frame(frame)
    observer.capture_raw_input_audio.assert_called_once_with(frame)
    _Base.push_audio_frame.assert_awaited_once_with(frame)


@pytest.mark.asyncio
async def test_mixin_push_audio_frame_forwards_when_observer_raises(ff) -> None:
    from noveum_trace.integrations.pipecat.transports import NoveumRawAudioTapMixin

    observer = MagicMock()
    observer.capture_raw_input_audio.side_effect = RuntimeError("capture failed")
    frame = ff.InputAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)

    class _Base:
        push_audio_frame = AsyncMock()

    class _Tap(NoveumRawAudioTapMixin, _Base):
        def __init__(self) -> None:
            self._noveum_observer = observer

    tap = _Tap()
    await tap.push_audio_frame(frame)
    observer.capture_raw_input_audio.assert_called_once_with(frame)
    _Base.push_audio_frame.assert_awaited_once_with(frame)


def test_mro_guard_raises_on_bad_order() -> None:
    pytest.importorskip("pipecat.transports.base_input")
    from pipecat.transports.base_input import BaseInputTransport

    from noveum_trace.integrations.pipecat.transports import NoveumRawAudioTapMixin

    with pytest.raises(TypeError, match="must precede BaseInputTransport"):

        class Bad(BaseInputTransport, NoveumRawAudioTapMixin):  # type: ignore[misc]
            pass


def test_local_audio_stub_raises_import_error_when_backend_unavailable() -> None:
    from noveum_trace.integrations.pipecat import transports as transports_mod

    stub = transports_mod.NoveumLocalAudioTransport
    # The stub set by _unavailable_transport has __name__ == "NoveumLocalAudioTransport"
    # (no leading underscore); the real wrapper class is "_NoveumLocalAudioTransport".
    if stub.__name__ == "NoveumLocalAudioTransport":
        with pytest.raises(ImportError, match="pyaudio"):
            stub()
    else:
        pytest.skip("pyaudio available — real NoveumLocalAudioTransport is in use")


def test_wrap_input_sets_observer_on_instance() -> None:
    _require("pipecat.transports.daily.transport")
    from pipecat.transports.daily.transport import DailyInputTransport

    from noveum_trace.integrations.pipecat.transports import _wrap_input

    Wrapped = _wrap_input(DailyInputTransport)
    assert Wrapped.__name__ == "_NoveumDailyInputTransport"


def test_transports_module_exports_all_symbols() -> None:
    import noveum_trace.integrations.pipecat.transports as transports_mod

    for name in transports_mod.__all__:
        assert hasattr(transports_mod, name), name


# ---------------------------------------------------------------------------
# Composite wrapper instantiation — observer wiring under real Pipecat classes
# ---------------------------------------------------------------------------


def test_noveum_websocket_server_transport_input_sets_observer() -> None:
    pytest.importorskip("pipecat.transports.websocket.server")
    from pipecat.transports.websocket.server import WebsocketServerParams

    from noveum_trace.integrations.pipecat.transports import (
        NoveumWebsocketServerTransport,
    )

    observer = MagicMock()
    t = NoveumWebsocketServerTransport(
        WebsocketServerParams(), noveum_observer=observer
    )
    inp = t.input()
    assert inp._noveum_observer is observer


def test_noveum_websocket_client_transport_input_sets_observer() -> None:
    pytest.importorskip("pipecat.transports.websocket.client")

    from noveum_trace.integrations.pipecat.transports import (
        NoveumWebsocketClientTransport,
    )

    observer = MagicMock()
    t = NoveumWebsocketClientTransport("ws://localhost:9999", noveum_observer=observer)
    inp = t.input()
    assert inp._noveum_observer is observer


def test_noveum_fastapi_websocket_transport_input_sets_observer() -> None:
    _require("fastapi")
    _require("pipecat.transports.websocket.fastapi")
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

    from noveum_trace.integrations.pipecat.transports import (
        NoveumFastAPIWebsocketTransport,
    )

    observer = MagicMock()
    mock_websocket = MagicMock()
    t = NoveumFastAPIWebsocketTransport(
        mock_websocket, FastAPIWebsocketParams(), noveum_observer=observer
    )
    inp = t.input()
    assert inp._noveum_observer is observer


def test_noveum_smallwebrtc_transport_input_sets_observer() -> None:
    _require("pipecat.transports.smallwebrtc.transport")
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

    from noveum_trace.integrations.pipecat.transports import NoveumSmallWebRTCTransport

    observer = MagicMock()
    mock_conn = MagicMock(spec=SmallWebRTCConnection)
    t = NoveumSmallWebRTCTransport(
        mock_conn, TransportParams(), noveum_observer=observer
    )
    inp = t.input()
    assert inp._noveum_observer is observer


def test_noveum_daily_transport_input_sets_observer(monkeypatch) -> None:
    _require("pipecat.transports.daily.transport")
    import pipecat.transports.daily.transport as daily_mod
    from pipecat.transports.daily.transport import DailyParams

    from noveum_trace.integrations.pipecat.transports import NoveumDailyTransport

    monkeypatch.setattr(daily_mod, "DailyTransportClient", lambda *a, **kw: MagicMock())
    observer = MagicMock()
    t = NoveumDailyTransport(
        "https://example.daily.co/test",
        None,
        "bot",
        params=DailyParams(),
        noveum_observer=observer,
    )
    inp = t.input()
    assert inp._noveum_observer is observer


def test_input_is_idempotent() -> None:
    """Calling .input() twice must return the same object (lazy-init guard)."""
    pytest.importorskip("pipecat.transports.websocket.server")
    from pipecat.transports.websocket.server import WebsocketServerParams

    from noveum_trace.integrations.pipecat.transports import (
        NoveumWebsocketServerTransport,
    )

    t = NoveumWebsocketServerTransport(
        WebsocketServerParams(), noveum_observer=MagicMock()
    )
    assert t.input() is t.input()


def test_input_with_no_observer_propagates_none() -> None:
    """Constructing without noveum_observer must set _noveum_observer=None on the input."""
    pytest.importorskip("pipecat.transports.websocket.server")
    from pipecat.transports.websocket.server import WebsocketServerParams

    from noveum_trace.integrations.pipecat.transports import (
        NoveumWebsocketServerTransport,
    )

    t = NoveumWebsocketServerTransport(WebsocketServerParams())
    inp = t.input()
    assert inp._noveum_observer is None


def test_missing_backend_does_not_hide_other_transports(monkeypatch) -> None:
    pytest.importorskip("pipecat")
    import builtins

    real_import = builtins.__import__

    def _import(name: str, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        if name == "pipecat.transports.heygen.transport":
            raise ImportError("simulated missing heygen backend")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    import importlib

    import noveum_trace.integrations.pipecat.transports as transports_mod

    importlib.reload(transports_mod)

    _require("pipecat.transports.daily.transport")
    from pipecat.transports.daily.transport import DailyTransport

    assert issubclass(transports_mod.NoveumDailyTransport, DailyTransport)

    with pytest.raises(ImportError, match="HeyGen"):
        transports_mod.NoveumHeyGenTransport()

    importlib.reload(
        importlib.import_module("noveum_trace.integrations.pipecat.transports")
    )
