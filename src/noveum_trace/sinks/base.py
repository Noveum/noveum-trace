"""
Base sink classes for the Noveum Trace SDK.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from noveum_trace.types import SpanData
from noveum_trace.utils.exceptions import ConfigurationError, SinkError

logger = logging.getLogger(__name__)


class SinkStatus(Enum):
    """Enumeration of sink status values."""

    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTDOWN = "shutdown"


@dataclass
class SinkMetrics:
    """Metrics for sink performance monitoring."""

    spans_sent: int = 0
    spans_failed: int = 0
    batches_sent: int = 0
    batches_failed: int = 0
    total_latency_ms: float = 0.0
    last_success_time: Optional[float] = None
    last_error_time: Optional[float] = None
    last_error_message: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        total = self.spans_sent + self.spans_failed
        if total == 0:
            return 100.0
        return (self.spans_sent / total) * 100.0

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency per batch."""
        if self.batches_sent == 0:
            return 0.0
        return self.total_latency_ms / self.batches_sent

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "spans_sent": self.spans_sent,
            "spans_failed": self.spans_failed,
            "batches_sent": self.batches_sent,
            "batches_failed": self.batches_failed,
            "success_rate": self.success_rate,
            "average_latency_ms": self.average_latency_ms,
            "last_success_time": self.last_success_time,
            "last_error_time": self.last_error_time,
            "last_error_message": self.last_error_message,
        }


@dataclass
class SinkConfig:
    """Base configuration for sinks."""

    name: str
    enabled: bool = True
    max_retries: int = 3
    retry_delay_ms: int = 1000
    timeout_ms: int = 30000
    max_batch_size: int = 100
    compression: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the sink configuration."""
        if not self.name:
            raise ConfigurationError("Sink name cannot be empty")

        if self.max_retries < 0:
            raise ConfigurationError("max_retries must be non-negative")

        if self.retry_delay_ms <= 0:
            raise ConfigurationError("retry_delay_ms must be positive")

        if self.timeout_ms <= 0:
            raise ConfigurationError("timeout_ms must be positive")

        if self.max_batch_size <= 0:
            raise ConfigurationError("max_batch_size must be positive")


class BaseSink(ABC):
    """Abstract base class for all sinks."""

    def __init__(self, config: SinkConfig):
        """Initialize the sink."""
        config.validate()
        self._config = config
        self._status = SinkStatus.INITIALIZING
        self._metrics = SinkMetrics()
        self._initialized = False

        # Initialize the sink
        try:
            self._initialize()
            self._status = SinkStatus.HEALTHY
            self._initialized = True
            logger.info(f"Sink '{self.name}' initialized successfully")
        except Exception as e:
            self._status = SinkStatus.UNHEALTHY
            self._metrics.last_error_time = time.time()
            self._metrics.last_error_message = str(e)
            logger.error(f"Failed to initialize sink '{self.name}': {e}")
            raise SinkError(
                f"Sink initialization failed: {e}", sink_name=self.name
            ) from e

    @property
    def name(self) -> str:
        """Get the sink name."""
        return self._config.name

    @property
    def config(self) -> SinkConfig:
        """Get the sink configuration."""
        return self._config

    @property
    def status(self) -> SinkStatus:
        """Get the current sink status."""
        return self._status

    @property
    def metrics(self) -> SinkMetrics:
        """Get the sink metrics."""
        return self._metrics

    @property
    def is_healthy(self) -> bool:
        """Check if the sink is healthy."""
        return self._status in (SinkStatus.HEALTHY, SinkStatus.DEGRADED)

    @property
    def is_enabled(self) -> bool:
        """Check if the sink is enabled."""
        return self._config.enabled and self._initialized

    def export(self, spans: List[SpanData]) -> None:
        """Export spans (alias for send_batch for compatibility)."""
        self.send_batch(spans)

    def send_batch(self, spans: List[SpanData]) -> None:
        """Send a batch of spans to the sink."""
        if not self.is_enabled:
            logger.debug(f"Sink '{self.name}' is disabled, skipping batch")
            return

        if not spans:
            logger.debug(f"Empty batch sent to sink '{self.name}', skipping")
            return

        # Limit batch size
        if len(spans) > self._config.max_batch_size:
            logger.warning(
                f"Batch size ({len(spans)}) exceeds max_batch_size "
                f"({self._config.max_batch_size}) for sink '{self.name}', truncating"
            )
            spans = spans[: self._config.max_batch_size]

        start_time = time.time()

        try:
            # Attempt to send with retries
            self._send_with_retries(spans)

            # Update success metrics
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            self._metrics.spans_sent += len(spans)
            self._metrics.batches_sent += 1
            self._metrics.total_latency_ms += latency_ms
            self._metrics.last_success_time = end_time

            # Update status to healthy if it was degraded
            if self._status == SinkStatus.DEGRADED:
                self._status = SinkStatus.HEALTHY
                logger.info(f"Sink '{self.name}' recovered to healthy status")

            logger.debug(
                f"Successfully sent {len(spans)} spans to sink '{self.name}' in {latency_ms:.2f}ms"
            )

        except Exception as e:
            # Update failure metrics
            self._metrics.spans_failed += len(spans)
            self._metrics.batches_failed += 1
            self._metrics.last_error_time = time.time()
            self._metrics.last_error_message = str(e)

            # Update status based on error severity
            if self._status == SinkStatus.HEALTHY:
                self._status = SinkStatus.DEGRADED
                logger.warning(f"Sink '{self.name}' degraded due to error: {e}")
            elif self._status == SinkStatus.DEGRADED:
                self._status = SinkStatus.UNHEALTHY
                logger.error(
                    f"Sink '{self.name}' marked unhealthy due to repeated errors: {e}"
                )

            # Re-raise the exception
            raise SinkError(
                f"Failed to send batch to sink '{self.name}': {e}", sink_name=self.name
            ) from e

    def health_check(self) -> bool:
        """Perform a health check on the sink."""
        try:
            result = self._health_check()
            if result:
                if self._status == SinkStatus.UNHEALTHY:
                    self._status = SinkStatus.DEGRADED
                    logger.info(
                        f"Sink '{self.name}' health check passed, status upgraded to degraded"
                    )
                elif self._status == SinkStatus.DEGRADED:
                    self._status = SinkStatus.HEALTHY
                    logger.info(
                        f"Sink '{self.name}' health check passed, status upgraded to healthy"
                    )
            return result
        except Exception as e:
            logger.error(f"Health check failed for sink '{self.name}': {e}")
            self._status = SinkStatus.UNHEALTHY
            return False

    def shutdown(self) -> None:
        """Shutdown the sink."""
        if self._status == SinkStatus.SHUTDOWN:
            return

        logger.info(f"Shutting down sink '{self.name}'")

        try:
            self._shutdown()
            self._status = SinkStatus.SHUTDOWN
            logger.info(f"Sink '{self.name}' shutdown successfully")
        except Exception as e:
            logger.error(f"Error during sink '{self.name}' shutdown: {e}")
            self._status = SinkStatus.SHUTDOWN

    def _send_with_retries(self, spans: List[SpanData]) -> None:
        """Send spans with retry logic."""
        last_exception: Optional[Exception] = None

        for attempt in range(self._config.max_retries + 1):
            try:
                self._send_batch(spans)
                return  # Success
            except Exception as e:
                last_exception = e

                if attempt < self._config.max_retries:
                    delay_ms = self._config.retry_delay_ms * (
                        2**attempt
                    )  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1} failed for sink '{self.name}': {e}. "
                        f"Retrying in {delay_ms}ms"
                    )
                    time.sleep(delay_ms / 1000.0)
                else:
                    logger.error(
                        f"All {self._config.max_retries + 1} attempts failed for sink '{self.name}'"
                    )

        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Sink '{self.name}' failed without specific exception")

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the sink. Subclasses must implement this."""
        pass

    @abstractmethod
    def _send_batch(self, spans: List[SpanData]) -> None:
        """Send a batch of spans. Subclasses must implement this."""
        pass

    def _health_check(self) -> bool:
        """Perform sink-specific health check. Subclasses can override this."""
        return True

    @abstractmethod
    def _shutdown(self) -> None:
        """Perform sink-specific shutdown. Subclasses can override this."""
        pass

    def __repr__(self) -> str:
        """Return string representation of the sink."""
        return f"{self.__class__.__name__}(name='{self.name}', status='{self.status.value}')"
