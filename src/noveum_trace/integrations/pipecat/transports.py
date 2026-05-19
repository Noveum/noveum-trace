"""
Pipecat transport wrappers for pre-filter (raw) input audio capture.

Use ``Noveum*Transport`` composites instead of stock pipecat transports and pass
``noveum_observer=`` from :class:`~noveum_trace.integrations.pipecat.NoveumTraceObserver`.
"""

from __future__ import annotations

from typing import Any, Optional

from pipecat.transports.daily.transport import (
    DailyInputTransport,
    DailyTransport,
)
from pipecat.transports.heygen.transport import HeyGenInputTransport, HeyGenTransport
from pipecat.transports.lemonslice.transport import (
    LemonSliceInputTransport,
    LemonSliceTransport,
)
from pipecat.transports.livekit.transport import (
    LiveKitInputTransport,
    LiveKitTransport,
)
from pipecat.transports.smallwebrtc.transport import (
    SmallWebRTCInputTransport,
    SmallWebRTCTransport,
)
from pipecat.transports.tavus.transport import TavusInputTransport, TavusTransport
from pipecat.transports.websocket.client import (
    WebsocketClientInputTransport,
    WebsocketClientTransport,
)
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketInputTransport,
    FastAPIWebsocketTransport,
)
from pipecat.transports.websocket.server import (
    WebsocketServerInputTransport,
    WebsocketServerTransport,
)


class _NoveumLazyInputTransportMixin:
    """Annotations for pipecat transport lazy ``_input`` (parent attrs are untyped)."""

    _input: Any
    _input_name: str
    _noveum_observer: Any


class NoveumRawAudioTapMixin:
    """
    Tee pre-filter input audio to a :class:`~NoveumTraceObserver`.

    Must appear **before** ``BaseInputTransport`` in the MRO. Apply via
    ``class TappedFoo(NoveumRawAudioTapMixin, FooInputTransport): pass``.
    """

    _noveum_observer: Any

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is NoveumRawAudioTapMixin:
            return
        mro_names = [c.__name__ for c in cls.__mro__]
        if "NoveumRawAudioTapMixin" in mro_names and "BaseInputTransport" in mro_names:
            mixin_idx = mro_names.index("NoveumRawAudioTapMixin")
            base_idx = mro_names.index("BaseInputTransport")
            if mixin_idx > base_idx:
                raise TypeError(
                    f"{cls.__name__}: NoveumRawAudioTapMixin must precede "
                    "BaseInputTransport in the MRO (put the mixin first)"
                )

    async def push_audio_frame(self, frame: Any) -> None:
        observer = getattr(self, "_noveum_observer", None)
        if observer is not None:
            observer.capture_raw_input_audio(frame)
        await super().push_audio_frame(frame)  # type: ignore[misc]


def _wrap_input(base: type[Any]) -> type[Any]:
    """Build a private input transport subclass with raw-audio tap."""

    class _Wrapped(NoveumRawAudioTapMixin, base):
        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
            self._noveum_observer = noveum_observer
            super().__init__(*args, **kwargs)

    _Wrapped.__name__ = f"_Noveum{base.__name__}"
    _Wrapped.__qualname__ = f"_Noveum{base.__name__}"
    return _Wrapped


_NoveumDailyInputTransport = _wrap_input(DailyInputTransport)
_NoveumLiveKitInputTransport = _wrap_input(LiveKitInputTransport)
_NoveumSmallWebRTCInputTransport = _wrap_input(SmallWebRTCInputTransport)
_NoveumFastAPIWebsocketInputTransport = _wrap_input(FastAPIWebsocketInputTransport)
_NoveumWebsocketServerInputTransport = _wrap_input(WebsocketServerInputTransport)
_NoveumWebsocketClientInputTransport = _wrap_input(WebsocketClientInputTransport)
_NoveumTavusInputTransport = _wrap_input(TavusInputTransport)
_NoveumHeyGenInputTransport = _wrap_input(HeyGenInputTransport)
_NoveumLemonSliceInputTransport = _wrap_input(LemonSliceInputTransport)


class NoveumDailyTransport(_NoveumLazyInputTransportMixin, DailyTransport):
    """Daily transport with pre-filter input audio capture for Noveum tracing."""

    def __init__(self, *args: Any, noveum_observer: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._noveum_observer = noveum_observer

    def input(self) -> Any:
        if not self._input:
            self._input = _NoveumDailyInputTransport(
                self,
                self._client,
                self._params,
                name=self._input_name,
                noveum_observer=self._noveum_observer,
            )
        return self._input


class NoveumLiveKitTransport(_NoveumLazyInputTransportMixin, LiveKitTransport):
    """LiveKit transport with pre-filter input audio capture for Noveum tracing."""

    def __init__(self, *args: Any, noveum_observer: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._noveum_observer = noveum_observer

    def input(self) -> Any:
        if not self._input:
            self._input = _NoveumLiveKitInputTransport(
                self,
                self._client,
                self._params,
                name=self._input_name,
                noveum_observer=self._noveum_observer,
            )
        return self._input


class NoveumSmallWebRTCTransport(_NoveumLazyInputTransportMixin, SmallWebRTCTransport):
    """SmallWebRTC transport with pre-filter input audio capture for Noveum tracing."""

    def __init__(self, *args: Any, noveum_observer: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._noveum_observer = noveum_observer

    def input(self) -> Any:
        if not self._input:
            self._input = _NoveumSmallWebRTCInputTransport(
                self._client,
                self._params,
                name=self._input_name,
                noveum_observer=self._noveum_observer,
            )
        return self._input


class NoveumFastAPIWebsocketTransport(
    _NoveumLazyInputTransportMixin, FastAPIWebsocketTransport
):
    """FastAPI WebSocket transport with pre-filter input audio capture."""

    def __init__(
        self,
        *args: Any,
        noveum_observer: Any = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            input_name=input_name,
            output_name=output_name,
            **kwargs,
        )
        self._noveum_observer = noveum_observer
        self._input = _NoveumFastAPIWebsocketInputTransport(
            self,
            self._client,
            self._params,
            name=self._input_name,
            noveum_observer=self._noveum_observer,
        )

    def input(self) -> Any:
        return self._input


class NoveumWebsocketServerTransport(
    _NoveumLazyInputTransportMixin, WebsocketServerTransport
):
    """WebSocket server transport with pre-filter input audio capture."""

    def __init__(self, *args: Any, noveum_observer: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._noveum_observer = noveum_observer

    def input(self) -> Any:
        if not self._input:
            self._input = _NoveumWebsocketServerInputTransport(
                self,
                self._host,
                self._port,
                self._params,
                self._callbacks,
                name=self._input_name,
                noveum_observer=self._noveum_observer,
            )
        return self._input


class NoveumWebsocketClientTransport(
    _NoveumLazyInputTransportMixin, WebsocketClientTransport
):
    """WebSocket client transport with pre-filter input audio capture."""

    def __init__(self, *args: Any, noveum_observer: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._noveum_observer = noveum_observer

    def input(self) -> Any:
        if not self._input:
            self._input = _NoveumWebsocketClientInputTransport(
                self,
                self._session,
                self._params,
                noveum_observer=self._noveum_observer,
            )
        return self._input


class NoveumTavusTransport(_NoveumLazyInputTransportMixin, TavusTransport):
    """Tavus transport with pre-filter input audio capture."""

    def __init__(self, *args: Any, noveum_observer: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._noveum_observer = noveum_observer

    def input(self) -> Any:
        if not self._input:
            self._input = _NoveumTavusInputTransport(
                client=self._client,
                params=self._params,
                noveum_observer=self._noveum_observer,
            )
        return self._input


class NoveumHeyGenTransport(_NoveumLazyInputTransportMixin, HeyGenTransport):
    """HeyGen transport with pre-filter input audio capture."""

    def __init__(self, *args: Any, noveum_observer: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._noveum_observer = noveum_observer

    def input(self) -> Any:
        if not self._input:
            self._input = _NoveumHeyGenInputTransport(
                client=self._client,
                params=self._params,
                noveum_observer=self._noveum_observer,
            )
        return self._input


class NoveumLemonSliceTransport(_NoveumLazyInputTransportMixin, LemonSliceTransport):
    """LemonSlice transport with pre-filter input audio capture."""

    def __init__(self, *args: Any, noveum_observer: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._noveum_observer = noveum_observer

    def input(self) -> Any:
        if not self._input:
            self._input = _NoveumLemonSliceInputTransport(
                client=self._client,
                params=self._params,
                noveum_observer=self._noveum_observer,
            )
        return self._input


# Optional local transports (pyaudio / tkinter).
# Probe deps before importing pipecat local modules — pipecat re-raises missing
# deps as generic Exception, which would bypass `except ImportError`.
try:
    import pyaudio  # type: ignore[import-untyped]  # noqa: F401
    from pipecat.transports.local.audio import (
        LocalAudioInputTransport,
        LocalAudioTransport,
    )

    _NoveumLocalAudioInputTransport = _wrap_input(LocalAudioInputTransport)

    class NoveumLocalAudioTransport(
        _NoveumLazyInputTransportMixin, LocalAudioTransport
    ):
        """Local audio transport with pre-filter input audio capture."""

        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
            super().__init__(*args, **kwargs)
            self._noveum_observer = noveum_observer

        def input(self) -> Any:
            if not self._input:
                self._input = _NoveumLocalAudioInputTransport(
                    self._pyaudio,
                    self._params,
                    noveum_observer=self._noveum_observer,
                )
            return self._input

except ImportError:

    class NoveumLocalAudioTransport:  # type: ignore[no-redef]
        """Stub when pyaudio / pipecat local audio extras are unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "NoveumLocalAudioTransport requires pyaudio. "
                "Install it via `pip install pyaudio` or use the "
                "pipecat extras that include it."
            ) from None


try:
    import tkinter  # noqa: F401

    from pipecat.transports.local.tk import (
        TkInputTransport,
        TkTransport,
    )

    _NoveumTkInputTransport = _wrap_input(TkInputTransport)

    class NoveumTkTransport(_NoveumLazyInputTransportMixin, TkTransport):
        """Tkinter local transport with pre-filter input audio capture."""

        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
            super().__init__(*args, **kwargs)
            self._noveum_observer = noveum_observer

        def input(self) -> Any:
            if not self._input:
                self._input = _NoveumTkInputTransport(
                    self._pyaudio,
                    self._params,
                    noveum_observer=self._noveum_observer,
                )
            return self._input

except ImportError:

    class NoveumTkTransport:  # type: ignore[no-redef]
        """Stub when tkinter / pipecat local tk extras are unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "NoveumTkTransport requires tkinter. "
                "Install a Python build with tkinter support, or use the "
                "pipecat extras that include it."
            ) from None


__all__ = [
    "NoveumRawAudioTapMixin",
    "NoveumDailyTransport",
    "NoveumLiveKitTransport",
    "NoveumSmallWebRTCTransport",
    "NoveumFastAPIWebsocketTransport",
    "NoveumWebsocketServerTransport",
    "NoveumWebsocketClientTransport",
    "NoveumTavusTransport",
    "NoveumHeyGenTransport",
    "NoveumLemonSliceTransport",
    "NoveumLocalAudioTransport",
    "NoveumTkTransport",
]
