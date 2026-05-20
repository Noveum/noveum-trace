"""
Pipecat transport wrappers for pre-filter (raw) input audio capture.

Use ``Noveum*Transport`` composites instead of stock pipecat transports and pass
``noveum_observer=`` from :class:`~noveum_trace.integrations.pipecat.NoveumTraceObserver`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
            try:
                observer.capture_raw_input_audio(frame)
            except Exception:
                logger.debug(
                    "Raw input audio capture failed; continuing pipeline",
                    exc_info=True,
                )
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


def _unavailable_transport(name: str, message: str) -> type[Any]:
    """Return a stub class that raises ``ImportError`` on instantiation."""

    class _Stub:
        """Stub when Pipecat transport dependencies are unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(message) from None

    _Stub.__name__ = name
    _Stub.__qualname__ = name
    _Stub.__doc__ = f"Stub when {name} dependencies are unavailable."
    return _Stub


NoveumDailyTransport: type[Any]
NoveumLiveKitTransport: type[Any]
NoveumSmallWebRTCTransport: type[Any]
NoveumFastAPIWebsocketTransport: type[Any]
NoveumWebsocketServerTransport: type[Any]
NoveumWebsocketClientTransport: type[Any]
NoveumTavusTransport: type[Any]
NoveumHeyGenTransport: type[Any]
NoveumLemonSliceTransport: type[Any]
NoveumLocalAudioTransport: type[Any]
NoveumTkTransport: type[Any]


try:
    from pipecat.transports.daily.transport import (
        DailyInputTransport,
        DailyTransport,
    )

    _NoveumDailyInputTransport = _wrap_input(DailyInputTransport)

    class _NoveumDailyTransport(_NoveumLazyInputTransportMixin, DailyTransport):
        """Daily transport with pre-filter input audio capture for Noveum tracing."""

        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
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

    NoveumDailyTransport = _NoveumDailyTransport

# Catch Exception (not just ImportError): pipecat re-raises missing transport
# deps as generic Exception, which would otherwise bypass ImportError handling.
except Exception:
    NoveumDailyTransport = _unavailable_transport(
        "NoveumDailyTransport",
        "NoveumDailyTransport requires Pipecat Daily transport dependencies. "
        "Install the relevant pipecat-ai extras and retry.",
    )


try:
    from pipecat.transports.livekit.transport import (
        LiveKitInputTransport,
        LiveKitTransport,
    )

    _NoveumLiveKitInputTransport = _wrap_input(LiveKitInputTransport)

    class _NoveumLiveKitTransport(_NoveumLazyInputTransportMixin, LiveKitTransport):
        """LiveKit transport with pre-filter input audio capture for Noveum tracing."""

        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
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

    NoveumLiveKitTransport = _NoveumLiveKitTransport

except Exception:
    NoveumLiveKitTransport = _unavailable_transport(
        "NoveumLiveKitTransport",
        "NoveumLiveKitTransport requires Pipecat LiveKit transport dependencies. "
        "Install the relevant pipecat-ai extras and retry.",
    )


try:
    from pipecat.transports.smallwebrtc.transport import (
        SmallWebRTCInputTransport,
        SmallWebRTCTransport,
    )

    _NoveumSmallWebRTCInputTransport = _wrap_input(SmallWebRTCInputTransport)

    class _NoveumSmallWebRTCTransport(
        _NoveumLazyInputTransportMixin, SmallWebRTCTransport
    ):
        """SmallWebRTC transport with pre-filter input audio capture for Noveum tracing."""

        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
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

    NoveumSmallWebRTCTransport = _NoveumSmallWebRTCTransport

except Exception:
    NoveumSmallWebRTCTransport = _unavailable_transport(
        "NoveumSmallWebRTCTransport",
        "NoveumSmallWebRTCTransport requires Pipecat SmallWebRTC transport dependencies. "
        "Install the relevant pipecat-ai extras and retry.",
    )


try:
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketInputTransport,
        FastAPIWebsocketTransport,
    )

    _NoveumFastAPIWebsocketInputTransport = _wrap_input(FastAPIWebsocketInputTransport)

    class _NoveumFastAPIWebsocketTransport(
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

    NoveumFastAPIWebsocketTransport = _NoveumFastAPIWebsocketTransport

except Exception:
    NoveumFastAPIWebsocketTransport = _unavailable_transport(
        "NoveumFastAPIWebsocketTransport",
        "NoveumFastAPIWebsocketTransport requires Pipecat FastAPI WebSocket "
        "transport dependencies. Install the relevant pipecat-ai extras and retry.",
    )


try:
    from pipecat.transports.websocket.server import (
        WebsocketServerInputTransport,
        WebsocketServerTransport,
    )

    _NoveumWebsocketServerInputTransport = _wrap_input(WebsocketServerInputTransport)

    class _NoveumWebsocketServerTransport(
        _NoveumLazyInputTransportMixin, WebsocketServerTransport
    ):
        """WebSocket server transport with pre-filter input audio capture."""

        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
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

    NoveumWebsocketServerTransport = _NoveumWebsocketServerTransport

except Exception:
    NoveumWebsocketServerTransport = _unavailable_transport(
        "NoveumWebsocketServerTransport",
        "NoveumWebsocketServerTransport requires Pipecat WebSocket server "
        "transport dependencies. Install the relevant pipecat-ai extras and retry.",
    )


try:
    from pipecat.transports.websocket.client import (
        WebsocketClientInputTransport,
        WebsocketClientTransport,
    )

    _NoveumWebsocketClientInputTransport = _wrap_input(WebsocketClientInputTransport)

    class _NoveumWebsocketClientTransport(
        _NoveumLazyInputTransportMixin, WebsocketClientTransport
    ):
        """WebSocket client transport with pre-filter input audio capture."""

        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
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

    NoveumWebsocketClientTransport = _NoveumWebsocketClientTransport

except Exception:
    NoveumWebsocketClientTransport = _unavailable_transport(
        "NoveumWebsocketClientTransport",
        "NoveumWebsocketClientTransport requires Pipecat WebSocket client "
        "transport dependencies. Install the relevant pipecat-ai extras and retry.",
    )


try:
    from pipecat.transports.tavus.transport import TavusInputTransport, TavusTransport

    _NoveumTavusInputTransport = _wrap_input(TavusInputTransport)

    class _NoveumTavusTransport(_NoveumLazyInputTransportMixin, TavusTransport):
        """Tavus transport with pre-filter input audio capture."""

        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
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

    NoveumTavusTransport = _NoveumTavusTransport

except Exception:
    NoveumTavusTransport = _unavailable_transport(
        "NoveumTavusTransport",
        "NoveumTavusTransport requires Pipecat Tavus transport dependencies. "
        "Install the relevant pipecat-ai extras and retry.",
    )


try:
    from pipecat.transports.heygen.transport import (
        HeyGenInputTransport,
        HeyGenTransport,
    )

    _NoveumHeyGenInputTransport = _wrap_input(HeyGenInputTransport)

    class _NoveumHeyGenTransport(_NoveumLazyInputTransportMixin, HeyGenTransport):
        """HeyGen transport with pre-filter input audio capture."""

        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
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

    NoveumHeyGenTransport = _NoveumHeyGenTransport

except Exception:
    NoveumHeyGenTransport = _unavailable_transport(
        "NoveumHeyGenTransport",
        "NoveumHeyGenTransport requires Pipecat HeyGen transport dependencies. "
        "Install the relevant pipecat-ai extras and retry.",
    )


try:
    from pipecat.transports.lemonslice.transport import (
        LemonSliceInputTransport,
        LemonSliceTransport,
    )

    _NoveumLemonSliceInputTransport = _wrap_input(LemonSliceInputTransport)

    class _NoveumLemonSliceTransport(
        _NoveumLazyInputTransportMixin, LemonSliceTransport
    ):
        """LemonSlice transport with pre-filter input audio capture."""

        def __init__(
            self, *args: Any, noveum_observer: Any = None, **kwargs: Any
        ) -> None:
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

    NoveumLemonSliceTransport = _NoveumLemonSliceTransport

except Exception:
    NoveumLemonSliceTransport = _unavailable_transport(
        "NoveumLemonSliceTransport",
        "NoveumLemonSliceTransport requires Pipecat LemonSlice transport dependencies. "
        "Install the relevant pipecat-ai extras and retry.",
    )


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

    class _NoveumLocalAudioTransport(
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

    NoveumLocalAudioTransport = _NoveumLocalAudioTransport

except ImportError:
    NoveumLocalAudioTransport = _unavailable_transport(
        "NoveumLocalAudioTransport",
        "NoveumLocalAudioTransport requires pyaudio. "
        "Install it via `pip install pyaudio` or use the "
        "pipecat extras that include it.",
    )


try:
    import tkinter  # noqa: F401

    from pipecat.transports.local.tk import (
        TkInputTransport,
        TkTransport,
    )

    _NoveumTkInputTransport = _wrap_input(TkInputTransport)

    class _NoveumTkTransport(_NoveumLazyInputTransportMixin, TkTransport):
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

    NoveumTkTransport = _NoveumTkTransport

except ImportError:
    NoveumTkTransport = _unavailable_transport(
        "NoveumTkTransport",
        "NoveumTkTransport requires tkinter. "
        "Install a Python build with tkinter support, or use the "
        "pipecat extras that include it.",
    )


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
