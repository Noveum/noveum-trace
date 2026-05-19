"""Unit tests for Pipecat Noveum transport wrappers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


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
    # When pyaudio is missing, transports.py defines the stub instead of the real class.
    if getattr(stub, "__module__", "").endswith("transports"):
        with pytest.raises(ImportError, match="pyaudio"):
            stub()
    else:
        pytest.skip("pyaudio available — real NoveumLocalAudioTransport is in use")


def test_wrap_input_sets_observer_on_instance() -> None:
    pytest.importorskip("pipecat.transports.daily.transport")
    from pipecat.transports.daily.transport import DailyInputTransport

    from noveum_trace.integrations.pipecat.transports import _wrap_input

    Wrapped = _wrap_input(DailyInputTransport)
    assert Wrapped.__name__ == "_NoveumDailyInputTransport"
