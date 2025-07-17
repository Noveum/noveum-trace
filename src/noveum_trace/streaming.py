"""
Streaming support for Noveum Trace SDK.

This module provides specialized components for tracing streaming LLM responses,
where tokens are received incrementally rather than all at once.
"""

import threading
import time
from collections.abc import Iterator
from types import TracebackType
from typing import Any, Callable, Generic, Optional, TypeVar

from noveum_trace.context_managers import trace_llm
from noveum_trace.utils.exceptions import NoveumTraceError

T = TypeVar("T")


class StreamingSpanManager:
    """
    Manager for tracing streaming LLM responses.

    This class handles the lifecycle of a span for streaming responses,
    updating metrics as tokens arrive and finalizing when complete.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        operation: str = "streaming",
        capture_tokens: bool = True,
        max_tokens_stored: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the streaming span manager.

        Args:
            model: LLM model name
            provider: LLM provider (openai, anthropic, etc.)
            operation: Operation type for the span
            capture_tokens: Whether to capture individual tokens
            max_tokens_stored: Maximum number of tokens to store for debugging
            **kwargs: Additional span attributes
        """
        self.model = model
        self.provider = provider
        self.operation = operation
        self.capture_tokens = capture_tokens
        self.max_tokens_stored = max_tokens_stored
        self.span_attributes = kwargs

        # Token tracking
        self.tokens_received = 0
        self.tokens_stored: list[str] = []
        self.total_characters = 0
        self.start_time = time.time()
        self.first_token_time: Optional[float] = None
        self.last_token_time: Optional[float] = None

        # Thread safety
        self.lock = threading.Lock()
        self.is_finished = False

        # Span management
        self.span: Optional[Any] = None
        self.span_context: Optional[Any] = None

    def __enter__(self) -> "StreamingSpanManager":
        """Start the streaming span."""
        self.span_context = trace_llm(
            model=self.model,
            provider=self.provider,
            operation=self.operation,
            **self.span_attributes,
        )
        self.span = self.span_context.__enter__()

        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Finish the streaming span."""
        with self.lock:
            if not self.is_finished:
                self.finish_streaming()

        # Let the span context manager handle the exception
        if self.span_context:
            return self.span_context.__exit__(exc_type, exc_val, exc_tb)

    def add_token(self, token: str) -> None:
        """
        Add a token to the stream.

        Args:
            token: Token string to add
        """
        if self.is_finished:
            raise NoveumTraceError("Cannot add token to finished stream")

        with self.lock:
            current_time = time.time()
            self.tokens_received += 1
            self.total_characters += len(token)

            # Track timing
            if self.first_token_time is None:
                self.first_token_time = current_time
            self.last_token_time = current_time

            # Store token if enabled
            if self.capture_tokens and len(self.tokens_stored) < self.max_tokens_stored:
                self.tokens_stored.append(token)

            # Update span attributes
            if self.span:
                self.span.set_attributes(
                    {
                        "streaming.tokens_received": self.tokens_received,
                        "streaming.total_characters": self.total_characters,
                        "streaming.time_to_first_token": (
                            self.first_token_time - self.start_time
                            if self.first_token_time
                            else None
                        ),
                        "streaming.last_token_time": self.last_token_time,
                    }
                )

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Add metadata to the span.

        Args:
            metadata: Metadata dictionary to add
        """
        if self.span:
            self.span.set_attributes(metadata)

    def finish_streaming(self) -> None:
        """Finish the streaming operation."""
        if self.is_finished:
            return

        with self.lock:
            end_time = time.time()
            total_duration = end_time - self.start_time

            # Calculate streaming metrics
            tokens_per_second = (
                self.tokens_received / total_duration if total_duration > 0 else 0
            )
            chars_per_second = (
                self.total_characters / total_duration if total_duration > 0 else 0
            )

            # Update final span attributes
            if self.span:
                final_attributes: dict[str, Any] = {
                    "streaming.tokens_received": self.tokens_received,
                    "streaming.total_characters": self.total_characters,
                    "streaming.duration": total_duration,
                    "streaming.tokens_per_second": tokens_per_second,
                    "streaming.chars_per_second": chars_per_second,
                    "streaming.is_finished": True,
                }

                # Add timing metrics if available
                if self.first_token_time:
                    final_attributes["streaming.time_to_first_token"] = (
                        self.first_token_time - self.start_time
                    )

                if self.last_token_time:
                    final_attributes["streaming.time_to_last_token"] = (
                        self.last_token_time - self.start_time
                    )

                # Add stored tokens if enabled
                if self.capture_tokens and self.tokens_stored:
                    final_attributes["streaming.sample_tokens"] = self.tokens_stored

                self.span.set_attributes(final_attributes)

            self.is_finished = True

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current streaming metrics.

        Returns:
            Dictionary with streaming metrics
        """
        with self.lock:
            current_time = time.time()
            duration = current_time - self.start_time

            return {
                "tokens_received": self.tokens_received,
                "total_characters": self.total_characters,
                "duration": duration,
                "tokens_per_second": (
                    self.tokens_received / duration if duration > 0 else 0
                ),
                "chars_per_second": (
                    self.total_characters / duration if duration > 0 else 0
                ),
                "time_to_first_token": (
                    self.first_token_time - self.start_time
                    if self.first_token_time
                    else None
                ),
                "is_finished": self.is_finished,
            }


class TracedStreamWrapper(Generic[T]):
    """
    Wrapper for stream iterators that adds tracing.

    This class wraps a stream iterator and automatically traces
    tokens as they are yielded.
    """

    def __init__(
        self,
        stream_iterator: Iterator[T],
        model: str,
        provider: str,
        operation: str = "streaming",
        token_extractor: Optional[Callable[[T], str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the traced stream wrapper.

        Args:
            stream_iterator: Original stream iterator
            model: LLM model name
            provider: LLM provider (openai, anthropic, etc.)
            operation: Operation type for the span
            token_extractor: Function to extract token string from stream item
            **kwargs: Additional span attributes
        """
        self.stream_iterator = stream_iterator
        self.token_extractor = token_extractor or self._default_token_extractor
        self.manager = StreamingSpanManager(
            model=model,
            provider=provider,
            operation=operation,
            **kwargs,
        )

    def _default_token_extractor(self, item: T) -> str:
        """
        Default token extractor.

        Args:
            item: Stream item

        Returns:
            Token string
        """
        # Try common patterns for extracting tokens
        if hasattr(item, "choices") and item.choices:
            # OpenAI-style response
            choice = item.choices[0]
            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                return choice.delta.content or ""
            elif hasattr(choice, "text"):
                return choice.text or ""

        elif hasattr(item, "delta") and hasattr(item.delta, "text"):
            # Anthropic-style response
            return item.delta.text or ""

        elif hasattr(item, "token"):
            # Direct token attribute
            return item.token

        elif isinstance(item, str):
            # Direct string
            return item

        # Fallback to string representation
        return str(item)

    def __iter__(self) -> Iterator[T]:
        """Return the iterator."""
        return self

    def __next__(self) -> T:
        """Get the next item from the stream."""
        try:
            item = next(self.stream_iterator)

            # Extract token and add to manager
            token = self.token_extractor(item)
            if token:
                self.manager.add_token(token)

            return item
        except StopIteration:
            # Stream is finished
            self.manager.finish_streaming()
            raise

    def __enter__(self) -> "TracedStreamWrapper[T]":
        """Enter the context."""
        self.manager.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the context."""
        self.manager.__exit__(exc_type, exc_val, exc_tb)


def trace_streaming(
    stream_iterator: Iterator[T],
    model: str,
    provider: str,
    operation: str = "streaming",
    token_extractor: Optional[Callable[[T], str]] = None,
    **kwargs: Any,
) -> TracedStreamWrapper[T]:
    """
    Create a traced wrapper for a stream iterator.

    Args:
        stream_iterator: Original stream iterator
        model: LLM model name
        provider: LLM provider (openai, anthropic, etc.)
        operation: Operation type for the span
        token_extractor: Function to extract token string from stream item
        **kwargs: Additional span attributes

    Returns:
        TracedStreamWrapper that can be used as a context manager

    Example:
        # Trace an OpenAI stream
        stream = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        with trace_streaming(stream, "gpt-4", "openai") as traced_stream:
            for chunk in traced_stream:
                print(chunk.choices[0].delta.content, end="")
    """
    return TracedStreamWrapper(
        stream_iterator=stream_iterator,
        model=model,
        provider=provider,
        operation=operation,
        token_extractor=token_extractor,
        **kwargs,
    )


def create_openai_streaming_callback(
    model: str, **kwargs: Any
) -> Callable[[Iterator[Any]], Iterator[Any]]:
    """
    Create a callback for OpenAI streaming that adds tracing.

    Args:
        model: OpenAI model name
        **kwargs: Additional span attributes

    Returns:
        Callback function that can be used to wrap OpenAI streams

    Example:
        callback = create_openai_streaming_callback("gpt-4")

        # Use with OpenAI streaming
        stream = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        traced_stream = callback(stream)
        for chunk in traced_stream:
            print(chunk.choices[0].delta.content, end="")
    """

    def callback(stream: Iterator[Any]) -> Iterator[Any]:
        """Callback that wraps the stream with tracing."""
        return trace_streaming(
            stream_iterator=stream,
            model=model,
            provider="openai",
            operation="chat_completion_streaming",
            **kwargs,
        )

    return callback


def create_anthropic_streaming_callback(
    model: str, **kwargs: Any
) -> Callable[[Iterator[Any]], Iterator[Any]]:
    """
    Create a callback for Anthropic streaming that adds tracing.

    Args:
        model: Anthropic model name
        **kwargs: Additional span attributes

    Returns:
        Callback function that can be used to wrap Anthropic streams

    Example:
        callback = create_anthropic_streaming_callback("claude-3-opus")

        # Use with Anthropic streaming
        stream = anthropic.messages.create(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        traced_stream = callback(stream)
        for chunk in traced_stream:
            print(chunk.delta.text, end="")
    """

    def callback(stream: Iterator[Any]) -> Iterator[Any]:
        """Callback that wraps the stream with tracing."""
        return trace_streaming(
            stream_iterator=stream,
            model=model,
            provider="anthropic",
            operation="message_streaming",
            **kwargs,
        )

    return callback


class StreamingContext:
    """
    Context manager for streaming operations.

    This is a simplified interface for streaming tracing that
    handles the setup and teardown automatically.
    """

    def __init__(
        self, model: str, provider: str, operation: str = "streaming", **kwargs: Any
    ) -> None:
        """
        Initialize the streaming context manager.

        Args:
            model: LLM model name
            provider: LLM provider (openai, anthropic, etc.)
            operation: Operation type for the span
            **kwargs: Additional span attributes
        """
        self.manager = StreamingSpanManager(
            model=model,
            provider=provider,
            operation=operation,
            **kwargs,
        )

    def __enter__(self) -> StreamingSpanManager:
        """Enter the context and start tracing."""
        return self.manager.__enter__()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the context and finish tracing."""
        return self.manager.__exit__(exc_type, exc_val, exc_tb)


def streaming_llm(
    model: str, provider: str, operation: str = "streaming", **kwargs: Any
) -> StreamingContext:
    """
    Create a context manager for streaming LLM operations.

    This is a convenience function that creates a StreamingContext for
    tracing streaming LLM responses where tokens are received incrementally.

    Args:
        model: LLM model name
        provider: LLM provider (openai, anthropic, etc.)
        operation: Operation type for the span
        **kwargs: Additional span attributes

    Returns:
        StreamingContext that can be used as a context manager

    Example:
        with streaming_llm(model="gpt-4", provider="openai") as stream_manager:
            # Process streaming response
            for chunk in stream:
                token = chunk.choices[0].delta.content
                stream_manager.add_token(token)
                print(token, end="")
    """
    return StreamingContext(
        model=model,
        provider=provider,
        operation=operation,
        **kwargs,
    )
