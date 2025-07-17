"""
Exception classes for Noveum Trace SDK.

This module defines custom exception classes used throughout the SDK.
"""


class NoveumTraceError(Exception):
    """Base exception class for Noveum Trace SDK."""

    pass


class ConfigurationError(NoveumTraceError):
    """Raised when there's a configuration error."""

    pass


class TransportError(NoveumTraceError):
    """Raised when there's a transport/network error."""

    pass


class TracingError(NoveumTraceError):
    """Raised when there's an error in tracing operations."""

    pass


class AuthenticationError(NoveumTraceError):
    """Raised when authentication fails."""

    pass


class ValidationError(NoveumTraceError):
    """Raised when data validation fails."""

    pass


class InitializationError(NoveumTraceError):
    """Raised when SDK initialization fails."""

    pass


class InstrumentationError(NoveumTraceError):
    """Raised when instrumentation fails."""

    pass
