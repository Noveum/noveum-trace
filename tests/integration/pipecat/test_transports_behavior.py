"""
Transports subsystem (§H, TP-1..7) — value-asserting regression tests per
PIPECAT_TEST_PLAN.md.

Covers the ``Noveum*Transport`` composites, the ``NoveumRawAudioTapMixin``
pre-filter raw-audio tee, the ``_wrap_input`` factory, and the import-stub
fallback pattern. These transports **emit no spans**; the contract under test is
that pre-filter input audio is teed into the observer buffer *before* being
forwarded downstream, that downstream errors propagate, that capture is a
genuine no-op when recording is disabled, and that a missing/broken backend
degrades to an ``ImportError``-raising stub.

Tests that ``importlib.reload`` the transports module reload it again in
teardown so no stub class leaks into the rest of the session.
"""

from __future__ import annotations

import asyncio
import builtins
import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pipecat.frames.frames")


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #
def _input_audio_frame(audio: bytes = b"\x01\x02") -> Any:
    from pipecat.frames.frames import InputAudioRawFrame

    return InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)


def _stt_observer(
    *, record_audio: bool = True, record_raw_input_audio: bool = True
) -> Any:
    """Real observer with an STT pipeline detected (so capture is live)."""
    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    obs = NoveumTraceObserver(
        record_audio=record_audio, record_raw_input_audio=record_raw_input_audio
    )
    obs._pipeline_has_stt = True
    return obs


def _reload_transports_clean(purge_substr: str) -> Any:
    """Reload the transports module from a clean state.

    Reload-based tests (here and in ``test_pipecat_transports.py``) can leave the
    transports module holding a stubbed/broken backend class in ``sys.modules``.
    Purging every cached submodule whose name contains ``purge_substr`` before the
    reload guarantees a clean baseline, so this test is order-independent and does
    not itself leak a stub into the rest of the session.
    """
    import noveum_trace.integrations.pipecat.transports as T

    for key in [k for k in list(sys.modules) if purge_substr in k]:
        del sys.modules[key]
    importlib.reload(T)
    return T


# --------------------------------------------------------------------------- #
# TP-1 — import-stub fires for a NON-ImportError dep failure                    #
# --------------------------------------------------------------------------- #
def test_heygen_stub_on_non_importerror_dep_failure(monkeypatch: Any) -> None:
    # Guards: narrowing the deliberate `except Exception` to `except ImportError`,
    # which would crash the whole transports module on pipecat's generic-Exception
    # missing-dep error (TP-1).
    real_import = builtins.__import__

    # Establish a clean baseline first: a prior reload-based test (here or in
    # test_pipecat_transports.py) may have left the module holding a stub/broken
    # heygen, which would otherwise make `original_name` order-dependent.
    T = _reload_transports_clean("heygen")
    # Baseline target works whether heygen is installed (real `_Noveum…`
    # composite) or absent (already a stub) — the injected RuntimeError, not
    # ambient availability, is what drives the stub under test.
    original_name = T.NoveumHeyGenTransport.__name__

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == "pipecat.transports.heygen.transport":
            raise RuntimeError("simulated non-ImportError dep failure")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        importlib.reload(T)
        # A RuntimeError on import still degrades to the stub class only because
        # the source catches `except Exception`, not `except ImportError`.
        assert T.NoveumHeyGenTransport.__name__ == "NoveumHeyGenTransport"
        with pytest.raises(ImportError, match="HeyGen"):
            T.NoveumHeyGenTransport()
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        T = _reload_transports_clean("heygen")
    # After a clean teardown reload the baseline class is restored.
    assert T.NoveumHeyGenTransport.__name__ == original_name


# --------------------------------------------------------------------------- #
# TP-2 — super().push_audio_frame exception PROPAGATES                          #
# --------------------------------------------------------------------------- #
def test_push_audio_frame_propagates_downstream_error() -> None:
    # Guards: widening the tap's try/except to wrap the super() call, which would
    # silently eat real pipeline/transport errors. Capture must run BEFORE the
    # propagating super() call (TP-2).
    from noveum_trace.integrations.pipecat.transports import NoveumRawAudioTapMixin

    class _Base:
        async def push_audio_frame(self, frame: Any) -> None:
            raise RuntimeError("pipeline boom")

    class _Tap(NoveumRawAudioTapMixin, _Base):
        pass

    obs = _stt_observer()
    tap = _Tap()
    tap._noveum_observer = obs

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="pipeline boom"):
            await tap.push_audio_frame(_input_audio_frame(b"\xaa\xbb"))

    asyncio.run(_run())
    # Capture happened before the propagating super() call — exactly one snapshot.
    assert len(obs._stt_raw_audio_buffer) == 1
    assert obs._stt_raw_audio_buffer[0].audio == b"\xaa\xbb"


# --------------------------------------------------------------------------- #
# TP-3 — end-to-end raw-audio tap through a REAL wrapped transport              #
# --------------------------------------------------------------------------- #
def test_real_websocket_transport_tees_raw_audio_to_observer() -> None:
    # Guards: composite input() wiring / _wrap_input MRO breaking so the tap
    # silently stops — invisible to the `inp._noveum_observer is observer`
    # identity check (TP-3).
    pytest.importorskip("pipecat.transports.websocket.server")
    from pipecat.transports.websocket.server import (
        WebsocketServerInputTransport,
        WebsocketServerParams,
    )

    import noveum_trace.integrations.pipecat.transports as T

    obs = _stt_observer()
    transport = T.NoveumWebsocketServerTransport(
        WebsocketServerParams(), noveum_observer=obs
    )
    inp = transport.input()
    assert isinstance(inp, WebsocketServerInputTransport)
    assert isinstance(inp, T.NoveumRawAudioTapMixin)

    pushed_frame = _input_audio_frame(b"\x01\x02")

    async def _run() -> None:
        await inp.push_audio_frame(pushed_frame)

    asyncio.run(_run())

    buf = obs._stt_raw_audio_buffer
    assert len(buf) == 1
    assert buf[0].audio == b"\x01\x02"
    assert buf[0].sample_rate == 16000
    assert buf[0].num_channels == 1
    # The buffered frame is a distinct snapshot InputAudioRawFrame, not the
    # original pushed frame.
    from pipecat.frames.frames import InputAudioRawFrame

    assert isinstance(buf[0], InputAudioRawFrame)
    assert buf[0] is not pushed_frame


# --------------------------------------------------------------------------- #
# TP-4 — input() returns the real input transport AND the tap mixin            #
# --------------------------------------------------------------------------- #
def _ws_server_composite() -> tuple[Any, Any]:
    pytest.importorskip("pipecat.transports.websocket.server")
    from pipecat.transports.websocket.server import (
        WebsocketServerInputTransport,
        WebsocketServerParams,
    )

    import noveum_trace.integrations.pipecat.transports as T

    obs = MagicMock()
    transport = T.NoveumWebsocketServerTransport(
        WebsocketServerParams(), noveum_observer=obs
    )
    return transport, (WebsocketServerInputTransport, obs)


def _ws_client_composite() -> tuple[Any, Any]:
    pytest.importorskip("pipecat.transports.websocket.client")
    from pipecat.transports.websocket.client import (
        WebsocketClientInputTransport,
        WebsocketClientParams,
    )

    import noveum_trace.integrations.pipecat.transports as T

    obs = MagicMock()
    transport = T.NoveumWebsocketClientTransport(
        "ws://example", params=WebsocketClientParams(), noveum_observer=obs
    )
    return transport, (WebsocketClientInputTransport, obs)


def _fastapi_composite() -> tuple[Any, Any]:
    # The real `fastapi` backend dep — on 0.0.x the pipecat fastapi module
    # imports-but-is-broken when fastapi is absent (it catches the ImportError
    # and only logs), so `importorskip` on the pipecat module is not enough to
    # skip; gate on the actual backend package instead.
    pytest.importorskip("fastapi")
    pytest.importorskip("pipecat.transports.websocket.fastapi")
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketInputTransport,
        FastAPIWebsocketParams,
    )

    import noveum_trace.integrations.pipecat.transports as T

    obs = MagicMock()
    transport = T.NoveumFastAPIWebsocketTransport(
        MagicMock(), params=FastAPIWebsocketParams(), noveum_observer=obs
    )
    return transport, (FastAPIWebsocketInputTransport, obs)


@pytest.mark.parametrize(
    "builder",
    [_ws_server_composite, _ws_client_composite, _fastapi_composite],
    ids=["websocket_server", "websocket_client", "fastapi_websocket"],
)
def test_input_is_real_transport_and_tap_with_wired_observer(builder: Any) -> None:
    # Guards: hand-copied per-transport input() arg-shape drift dropping the real
    # input subclass / not mixing in the tap / not wiring the observer (TP-4).
    import noveum_trace.integrations.pipecat.transports as T

    transport, (real_input_cls, obs) = builder()
    inp = transport.input()
    assert isinstance(inp, real_input_cls)
    assert isinstance(inp, T.NoveumRawAudioTapMixin)
    assert inp._noveum_observer is obs


# --------------------------------------------------------------------------- #
# TP-5 — LocalAudio stub path forced via import injection                      #
# --------------------------------------------------------------------------- #
def test_local_audio_stub_on_pyaudio_importerror(monkeypatch: Any) -> None:
    # Guards: the LocalAudio stub path fires when pyaudio is genuinely missing,
    # forced via import injection rather than relying on ambient pyaudio absence
    # (TP-5). Replaces a test that skipped whenever pyaudio was installed.
    import noveum_trace.integrations.pipecat.transports as T

    original_name = T.NoveumLocalAudioTransport.__name__

    real_import = builtins.__import__

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == "pyaudio":
            raise ImportError("No module named 'pyaudio'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        importlib.reload(T)
        assert T.NoveumLocalAudioTransport.__name__ == "NoveumLocalAudioTransport"
        with pytest.raises(ImportError, match="pyaudio"):
            T.NoveumLocalAudioTransport()
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(T)
    assert T.NoveumLocalAudioTransport.__name__ == original_name


def test_local_audio_non_importerror_does_not_degrade(monkeypatch: Any) -> None:
    # Guards: surfaces (pins) the lone catch-inconsistency — the LocalAudio branch
    # uses `except ImportError`, so a NON-ImportError pyaudio failure is NOT caught
    # and the whole module import raises, unlike every other branch's
    # `except Exception` graceful-stub fallback (TP-5).
    import noveum_trace.integrations.pipecat.transports as T

    original_name = T.NoveumLocalAudioTransport.__name__

    real_import = builtins.__import__

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == "pyaudio":
            raise RuntimeError("simulated non-ImportError pyaudio failure")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        # The LocalAudio branch catches only ImportError, so a RuntimeError from
        # importing pyaudio propagates out of the module reload itself.
        with pytest.raises(RuntimeError, match="non-ImportError pyaudio failure"):
            importlib.reload(T)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(T)
    assert T.NoveumLocalAudioTransport.__name__ == original_name


# --------------------------------------------------------------------------- #
# TP-6 — capture is a no-op when recording flags are off                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("record_audio", "record_raw_input_audio"),
    [(False, True), (True, False)],
    ids=["record_audio_off", "record_raw_input_audio_off"],
)
def test_capture_is_noop_when_recording_disabled(
    record_audio: bool, record_raw_input_audio: bool
) -> None:
    # Guards: a regression that captures/buffers raw input audio when the customer
    # disabled recording (privacy/compliance) — buffer must stay empty (TP-6).
    obs = _stt_observer(
        record_audio=record_audio, record_raw_input_audio=record_raw_input_audio
    )
    obs.capture_raw_input_audio(_input_audio_frame(b"\x01\x02\x03"))
    assert obs._stt_raw_audio_buffer == []


# --------------------------------------------------------------------------- #
# TP-7 — _wrap_input names the input subclass and mixes in the tap             #
# --------------------------------------------------------------------------- #
def test_wrap_input_names_subclass_and_mixes_in_tap() -> None:
    # Guards: _wrap_input not prepending the mixin (tap inert) while __name__
    # stays correct; and not wiring the observer onto the instance (TP-7).
    pytest.importorskip("pipecat.transports.websocket.server")
    from pipecat.transports.websocket.server import WebsocketServerInputTransport

    import noveum_trace.integrations.pipecat.transports as T

    wrapped = T._wrap_input(WebsocketServerInputTransport)
    assert wrapped.__name__ == "_NoveumWebsocketServerInputTransport"
    assert issubclass(wrapped, T.NoveumRawAudioTapMixin)
    assert issubclass(wrapped, WebsocketServerInputTransport)
    # The mixin must precede the real base so its push_audio_frame tap wins in MRO.
    mro_names = [c.__name__ for c in wrapped.__mro__]
    assert mro_names.index("NoveumRawAudioTapMixin") < mro_names.index(
        "WebsocketServerInputTransport"
    )

    # _Wrapped.__init__ stores noveum_observer before delegating to the real
    # base constructor — exercise that wiring path directly with mocked base args.
    obs = MagicMock()
    instance = wrapped(
        MagicMock(),  # transport
        MagicMock(),  # host
        MagicMock(),  # port
        MagicMock(),  # params
        MagicMock(),  # callbacks
        name="ws-input",
        noveum_observer=obs,
    )
    assert instance._noveum_observer is obs
    assert isinstance(instance, T.NoveumRawAudioTapMixin)
    assert isinstance(instance, WebsocketServerInputTransport)
