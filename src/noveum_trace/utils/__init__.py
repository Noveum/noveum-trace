"""
Utility modules for the Noveum Trace SDK.
"""

from .exceptions import (
    NoveumTracingError,
    ConfigurationError,
    ValidationError,
    NetworkError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    SerializationError,
    InstrumentationError,
    SinkError,
    ElasticsearchError,
    NoveumAPIError,
    ContextError,
    SpanError,
    SamplingError,
    ResourceError,
    TimeoutError,
    configuration_error,
    validation_error,
    network_error,
    sink_error,
)

__all__ = [
    "NoveumTracingError",
    "ConfigurationError",
    "ValidationError", 
    "NetworkError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "SerializationError",
    "InstrumentationError",
    "SinkError",
    "ElasticsearchError",
    "NoveumAPIError",
    "ContextError",
    "SpanError",
    "SamplingError",
    "ResourceError",
    "TimeoutError",
    "configuration_error",
    "validation_error",
    "network_error",
    "sink_error",
]

