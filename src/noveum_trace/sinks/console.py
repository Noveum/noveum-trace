"""
Console sink for the Noveum Trace SDK.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from noveum_trace.types import SpanData, SpanStatus

from .base import BaseSink, SinkConfig

logger = logging.getLogger(__name__)


@dataclass
class ConsoleSinkConfig(SinkConfig):
    """Configuration for console sink."""

    # Output settings
    pretty_print: bool = True
    include_metadata: bool = False
    max_content_length: int = 200

    # Filtering
    min_duration_ms: float = 0.0
    include_successful_only: bool = False


class ConsoleSink(BaseSink):
    """Console sink that prints spans to stdout for debugging."""

    def __init__(self, config: Optional[ConsoleSinkConfig] = None) -> None:
        """Initialize console sink."""
        if config is None:
            config = ConsoleSinkConfig(name="console-sink")

        super().__init__(config)
        self._config: ConsoleSinkConfig = config

    def _initialize(self) -> None:
        """Initialize console sink."""
        logger.info("Console sink initialized")

    def _send_batch(self, spans: List[SpanData]) -> None:
        """Print spans to console."""
        for span in spans:
            self._print_span(span)

    def _print_span(self, span: SpanData) -> None:
        """Print a single span to console."""
        # Apply filters
        if (
            span.duration_ms is not None
            and span.duration_ms < self._config.min_duration_ms
        ):
            return

        if self._config.include_successful_only and span.status.value != "ok":
            return

        # Format span for console output
        if self._config.pretty_print:
            self._print_pretty_span(span)
        else:
            self._print_json_span(span)

    def _print_pretty_span(self, span: SpanData) -> None:
        """Print span in human-readable format."""
        # Header
        status_icon = "✅" if span.status == SpanStatus.OK else "❌"
        print(f"\n{status_icon} {span.name} ({span.duration_ms:.1f}ms)")
        print(f"   Span ID: {span.span_id}")
        print(f"   Trace ID: {span.trace_id}")

        # Attributes
        if span.attributes:
            print("   Attributes:")
            for key, value in span.attributes.items():
                if key.startswith("gen_ai."):
                    print(f"     {key}: {value}")

        # Events (truncated)
        if span.events:
            print("   Events:")
            for event in span.events[:3]:  # Show first 3 events
                event_name = event.get("name", "unknown")
                print(f"     - {event_name}")

                # Show content if available
                if "attributes" in event:
                    attrs = event["attributes"]
                    for attr_key in ["gen_ai.prompt", "gen_ai.completion"]:
                        if attr_key in attrs:
                            content = str(attrs[attr_key])
                            if len(content) > self._config.max_content_length:
                                content = (
                                    content[: self._config.max_content_length] + "..."
                                )
                            print(f"       {attr_key}: {content}")

        # Error details
        if span.status == SpanStatus.ERROR and "status.description" in span.attributes:
            print(f"   Error: {span.attributes['status.description']}")

    def _print_json_span(self, span: SpanData) -> None:
        """Print span as JSON."""
        span_dict = span.to_dict()

        # Remove metadata if not requested
        if not self._config.include_metadata:
            span_dict.pop("_sink_metadata", None)

        print(json.dumps(span_dict, indent=2))

    def _health_check(self) -> bool:
        """Console sink is always healthy."""
        return True

    def _shutdown(self) -> None:
        """Shutdown console sink."""
        logger.info("Console sink shutdown")


# Add to base module for easy import
def create_console_sink() -> ConsoleSink:
    """Create a console sink with default configuration."""
    return ConsoleSink(ConsoleSinkConfig(name="console-sink"))
