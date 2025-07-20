"""
Logging utilities for Noveum Trace SDK.

This module provides centralized logging configuration and utilities
for the SDK, with support for debug logging via environment variables.
"""

import logging
import os
import sys
from typing import Any, Optional


class SafeStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """StreamHandler that silently ignores ValueError during shutdown."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Format the record
            msg = self.format(record)
            stream = self.stream

            # Try to write to stream, ignoring closed file errors
            try:
                stream.write(msg + self.terminator)
                self.flush()
            except (ValueError, OSError, AttributeError):
                # Silently ignore logging errors during shutdown
                pass
        except Exception:
            # Fallback: ignore all exceptions during emit
            pass


# SDK-specific logger names
SDK_LOGGER_NAME = "noveum_trace"
SDK_LOGGERS = [
    "noveum_trace",
    "noveum_trace.core",
    "noveum_trace.core.client",
    "noveum_trace.core.trace",
    "noveum_trace.core.span",
    "noveum_trace.transport",
    "noveum_trace.transport.http_transport",
    "noveum_trace.transport.batch_processor",
    "noveum_trace.decorators",
    "noveum_trace.integrations",
]


def get_log_level_from_env() -> int:
    """
    Get the log level from environment variables.

    Checks the following environment variables in order:
    1. NOVEUM_LOG_LEVEL - specific log level (DEBUG, INFO, WARNING, ERROR)
    2. NOVEUM_DEBUG - if set to true/1/yes, enables DEBUG level
    3. NOVEUM_VERBOSE - if set to true/1/yes, enables DEBUG level

    Returns:
        Log level integer (logging.DEBUG, logging.INFO, etc.)
    """
    # Check explicit log level first
    log_level_str = os.getenv("NOVEUM_LOG_LEVEL", "").upper()
    if log_level_str:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if log_level_str in level_map:
            return level_map[log_level_str]

    # Check debug flags
    debug_flags = ["NOVEUM_DEBUG", "NOVEUM_VERBOSE", "NOVEUM_DEBUG_LOGS"]
    for flag in debug_flags:
        value = os.getenv(flag, "").lower()
        if value in ("true", "1", "yes", "on"):
            return logging.DEBUG

    # Default to INFO for SDK loggers
    return logging.INFO


def setup_sdk_logging(force_level: Optional[int] = None) -> None:
    """
    Set up logging for the Noveum Trace SDK.

    Args:
        force_level: Force a specific log level, overriding environment variables
    """
    # Determine log level
    if force_level is not None:
        log_level = force_level
    else:
        log_level = get_log_level_from_env()

    # Set up formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure SDK loggers
    for logger_name in SDK_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add console handler if none exists or if we're in debug mode
        if not logger.handlers or log_level == logging.DEBUG:
            handler = SafeStreamHandler(sys.stderr)
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False


def get_sdk_logger(name: str) -> logging.Logger:
    """
    Get a logger for SDK components.

    Args:
        name: Logger name (will be prefixed with 'noveum_trace.')

    Returns:
        Configured logger instance
    """
    if not name.startswith("noveum_trace"):
        name = f"noveum_trace.{name}"

    logger = logging.getLogger(name)

    # Ensure the logger is configured
    if not logger.handlers:
        setup_sdk_logging()

    return logger


def log_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return get_log_level_from_env() == logging.DEBUG


def log_trace_flow(logger: logging.Logger, message: str, **kwargs: Any) -> None:
    """
    Log trace flow information with consistent formatting.

    Args:
        logger: Logger instance
        message: Log message
        **kwargs: Additional context to log
    """
    if kwargs:
        context_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        full_message = f"{message} ({context_str})"
    else:
        full_message = message

    logger.debug(f"🔄 TRACE_FLOW: {full_message}")


def log_http_request(
    logger: logging.Logger, method: str, url: str, **kwargs: Any
) -> None:
    """
    Log HTTP request details.

    Args:
        logger: Logger instance
        method: HTTP method
        url: Request URL
        **kwargs: Additional request details
    """
    logger.debug(f"🌐 HTTP_REQUEST: {method} {url}")
    for key, value in kwargs.items():
        logger.debug(f"    {key}: {value}")


def log_http_response(
    logger: logging.Logger, status_code: int, url: str, **kwargs: Any
) -> None:
    """
    Log HTTP response details.

    Args:
        logger: Logger instance
        status_code: HTTP status code
        url: Request URL
        **kwargs: Additional response details
    """
    status_emoji = "✅" if 200 <= status_code < 300 else "❌"
    logger.debug(f"{status_emoji} HTTP_RESPONSE: {status_code} from {url}")
    for key, value in kwargs.items():
        logger.debug(f"    {key}: {value}")


def log_error_always(
    logger: logging.Logger, message: str, exc_info: Any = None, **kwargs: Any
) -> None:
    """
    Log an error that should always be shown regardless of log level.

    Args:
        logger: Logger instance
        message: Error message
        exc_info: Exception info to include
        **kwargs: Additional error context
    """
    # Force log level to ERROR temporarily if needed
    original_level = logger.level
    if logger.level > logging.ERROR:
        logger.setLevel(logging.ERROR)

    try:
        if kwargs:
            context_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            full_message = f"{message} ({context_str})"
        else:
            full_message = message

        logger.error(f"❌ {full_message}", exc_info=exc_info)
    finally:
        # Restore original log level
        logger.setLevel(original_level)


# Initialize SDK logging on import
setup_sdk_logging()
