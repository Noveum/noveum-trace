"""
Exception hierarchy for the Noveum Trace SDK.
"""

from typing import Optional, Any, Dict


class NoveumTracingError(Exception):
    """Base exception for all Noveum Trace SDK errors.
    
    This is the root exception class that all other SDK exceptions inherit from.
    It provides common functionality for error handling and reporting.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional additional details about the error
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        parts = [self.message]
        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }


class ConfigurationError(NoveumTracingError):
    """Raised when there are configuration-related errors."""
    pass


class ValidationError(NoveumTracingError):
    """Raised when data validation fails."""
    pass


class NetworkError(NoveumTracingError):
    """Raised when network operations fail."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        """Initialize network error with additional HTTP details."""
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_body = response_body
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with network-specific details."""
        result = super().to_dict()
        result.update({
            "status_code": self.status_code,
            "response_body": self.response_body,
        })
        return result


class AuthenticationError(NetworkError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(NetworkError):
    """Raised when authorization fails."""
    pass


class RateLimitError(NetworkError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize rate limit error."""
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with rate limit details."""
        result = super().to_dict()
        result["retry_after"] = self.retry_after
        return result


class SerializationError(NoveumTracingError):
    """Raised when serialization or deserialization fails."""
    pass


class InstrumentationError(NoveumTracingError):
    """Raised when instrumentation operations fail."""
    pass


class SinkError(NoveumTracingError):
    """Raised when sink operations fail."""
    
    def __init__(
        self,
        message: str,
        sink_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize sink error."""
        super().__init__(message, **kwargs)
        self.sink_name = sink_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sink details."""
        result = super().to_dict()
        result["sink_name"] = self.sink_name
        return result


class ElasticsearchError(SinkError):
    """Raised when Elasticsearch operations fail."""
    pass


class NoveumAPIError(SinkError):
    """Raised when Noveum API operations fail."""
    pass


class ContextError(NoveumTracingError):
    """Raised when context operations fail."""
    pass


class SpanError(NoveumTracingError):
    """Raised when span operations fail."""
    pass


class SamplingError(NoveumTracingError):
    """Raised when sampling operations fail."""
    pass


class ResourceError(NoveumTracingError):
    """Raised when resource management fails."""
    pass


class TimeoutError(NoveumTracingError):
    """Raised when operations timeout."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        """Initialize timeout error."""
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with timeout details."""
        result = super().to_dict()
        result["timeout_seconds"] = self.timeout_seconds
        return result


# Convenience functions for common error scenarios
def configuration_error(message: str, **kwargs) -> ConfigurationError:
    """Create a configuration error with standard formatting."""
    return ConfigurationError(f"Configuration Error: {message}", **kwargs)


def validation_error(field: str, value: Any, reason: str, **kwargs) -> ValidationError:
    """Create a validation error with standard formatting."""
    message = f"Validation failed for field '{field}' with value '{value}': {reason}"
    return ValidationError(message, **kwargs)


def network_error(operation: str, details: str, **kwargs) -> NetworkError:
    """Create a network error with standard formatting."""
    message = f"Network operation '{operation}' failed: {details}"
    return NetworkError(message, **kwargs)


def sink_error(sink_name: str, operation: str, details: str, **kwargs) -> SinkError:
    """Create a sink error with standard formatting."""
    message = f"Sink '{sink_name}' operation '{operation}' failed: {details}"
    return SinkError(message, sink_name=sink_name, **kwargs)

