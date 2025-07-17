"""
Thread management for conversation tracking.

This module provides classes for managing conversation threads,
including context management and thread-specific tracing.
"""

import time
import uuid
from collections.abc import Generator
from types import TracebackType
from typing import Any, Optional

from noveum_trace.context_managers import trace_llm
from noveum_trace.core.context import get_current_span


class ThreadContext:
    """
    Context manager for conversation threads.

    This class provides a way to track and manage conversation threads,
    including automatic message tracking and thread-specific attributes.
    """

    def __init__(
        self,
        thread_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize a new thread context.

        Args:
            thread_id: Unique identifier for the thread
            name: Human-readable name for the thread
            metadata: Additional metadata for the thread
        """
        self.thread_id = thread_id or str(uuid.uuid4())
        self.name = name or f"thread_{self.thread_id[:8]}"
        self.metadata = metadata or {}

        self.messages: list[dict[str, Any]] = []
        self.turn_count = 0
        self.created_at = time.time()
        self.last_updated_at = self.created_at

        self.span = None
        self.trace = None

    def __enter__(self) -> "ThreadContext":
        """Enter the thread context."""
        # Create a new span for this thread
        current_span = get_current_span()

        from noveum_trace.core.client import get_client

        client = get_client()
        self.span = client.create_span(
            name=f"thread.{self.name}",
            span_type="thread",
            parent_span=current_span,
            attributes={
                "thread.id": self.thread_id,
                "thread.name": self.name,
                "thread.created_at": self.created_at,
                "thread.metadata": self.metadata,
            },
        ).__enter__()

        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the thread context."""
        if self.span:
            # Update final thread metrics
            self.span.set_attributes(
                {
                    "thread.message_count": len(self.messages),
                    "thread.turn_count": self.turn_count,
                    "thread.duration": time.time() - self.created_at,
                    "thread.last_updated_at": self.last_updated_at,
                }
            )

            # Record any exception
            if exc_type:
                self.span.record_exception(exc_val)
                self.span.set_status("error", str(exc_val))

            self.span.__exit__(exc_type, exc_val, exc_tb)

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Add a message to the thread.

        Args:
            role: Role of the message sender (user, assistant, system)
            content: Content of the message
            metadata: Additional metadata for the message
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        self.messages.append(message)
        self.last_updated_at = time.time()

        # Update turn count for user messages
        if role == "user":
            self.turn_count += 1

        # Update span attributes if available
        if self.span:
            self.span.set_attributes(
                {
                    "thread.message_count": len(self.messages),
                    "thread.turn_count": self.turn_count,
                    "thread.last_message_role": role,
                    "thread.last_updated_at": self.last_updated_at,
                }
            )

    def get_messages(
        self,
        limit: Optional[int] = None,
        role_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get messages from the thread.

        Args:
            limit: Maximum number of messages to return
            role_filter: Filter messages by role

        Returns:
            List of message dictionaries
        """
        messages = self.messages

        if role_filter:
            messages = [msg for msg in messages if msg["role"] == role_filter]

        if limit:
            messages = messages[-limit:]

        return messages

    def get_context_summary(self) -> dict[str, Any]:
        """
        Get a summary of the thread context.

        Returns:
            Dictionary with thread summary information
        """
        return {
            "thread_id": self.thread_id,
            "name": self.name,
            "message_count": len(self.messages),
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "duration": time.time() - self.created_at,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert thread context to dictionary."""
        return {
            "thread_id": self.thread_id,
            "name": self.name,
            "metadata": self.metadata,
            "messages": self.messages,
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThreadContext":
        """Create thread context from dictionary."""
        thread = cls(
            thread_id=data["thread_id"],
            name=data["name"],
            metadata=data.get("metadata", {}),
        )
        thread.messages = data.get("messages", [])
        thread.turn_count = data.get("turn_count", 0)
        thread.created_at = data.get("created_at", time.time())
        thread.last_updated_at = data.get("last_updated_at", thread.created_at)
        return thread


class ThreadManager:
    """
    Manager for conversation threads.

    This class provides a centralized way to create and manage
    multiple conversation threads.
    """

    def __init__(self) -> None:
        """Initialize the thread manager."""
        self.threads: dict[str, ThreadContext] = {}

    def create_thread(
        self,
        thread_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ThreadContext:
        """
        Create a new conversation thread.

        Args:
            thread_id: Unique identifier for the thread
            name: Human-readable name for the thread
            metadata: Additional metadata for the thread

        Returns:
            New ThreadContext instance
        """
        thread = ThreadContext(
            thread_id=thread_id,
            name=name,
            metadata=metadata,
        )

        self.threads[thread.thread_id] = thread
        return thread

    def get_thread(self, thread_id: str) -> Optional[ThreadContext]:
        """
        Get a thread by ID.

        Args:
            thread_id: Thread identifier

        Returns:
            ThreadContext if found, None otherwise
        """
        return self.threads.get(thread_id)

    def get_all_threads(self) -> dict[str, ThreadContext]:
        """
        Get all threads.

        Returns:
            Dictionary of thread_id -> ThreadContext
        """
        return self.threads.copy()

    def remove_thread(self, thread_id: str) -> bool:
        """
        Remove a thread by ID.

        Args:
            thread_id: Thread identifier

        Returns:
            True if thread was removed, False if not found
        """
        if thread_id in self.threads:
            del self.threads[thread_id]
            return True
        return False

    def clear_threads(self) -> None:
        """Clear all threads."""
        self.threads.clear()

    def get_thread_summary(self) -> dict[str, Any]:
        """
        Get a summary of all threads.

        Returns:
            Dictionary with thread manager summary
        """
        return {
            "total_threads": len(self.threads),
            "threads": {
                thread_id: thread.get_context_summary()
                for thread_id, thread in self.threads.items()
            },
        }


# Global thread manager instance
_thread_manager = ThreadManager()


def get_thread_manager() -> ThreadManager:
    """Get the global thread manager instance."""
    return _thread_manager


def create_thread(
    thread_id: Optional[str] = None,
    name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ThreadContext:
    """
    Create a new conversation thread.

    Args:
        thread_id: Unique identifier for the thread
        name: Human-readable name for the thread
        metadata: Additional metadata for the thread

    Returns:
        New ThreadContext instance
    """
    return _thread_manager.create_thread(thread_id, name, metadata)


def get_thread(thread_id: str) -> Optional[ThreadContext]:
    """
    Get a thread by ID.

    Args:
        thread_id: Thread identifier

    Returns:
        ThreadContext if found, None otherwise
    """
    return _thread_manager.get_thread(thread_id)


def delete_thread(thread_id: str) -> bool:
    """
    Delete a thread by ID.

    Args:
        thread_id: Thread identifier

    Returns:
        True if thread was deleted, False if not found
    """
    return _thread_manager.remove_thread(thread_id)


def list_threads() -> dict[str, ThreadContext]:
    """
    List all threads.

    Returns:
        Dictionary of thread_id -> ThreadContext
    """
    return _thread_manager.get_all_threads()


# Context manager for thread-specific tracing


def thread_context(
    thread_id: Optional[str] = None,
    name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ThreadContext:
    """
    Context manager for thread-specific tracing.

    Args:
        thread_id: Unique identifier for the thread
        name: Human-readable name for the thread
        metadata: Additional metadata for the thread

    Returns:
        ThreadContext instance

    Example:
        with thread_context(name="customer_support") as thread:
            thread.add_message("user", "Hello, I need help")

            # LLM call within thread context
            with trace_llm(model="gpt-4") as llm_span:
                response = llm_client.chat.completions.create(...)
                thread.add_message("assistant", response.choices[0].message.content)
    """
    return ThreadContext(thread_id, name, metadata)


# Specialized tracing for thread-based LLM calls


def trace_thread_llm(
    thread: ThreadContext,
    model: str,
    provider: str,
    operation: str = "thread_message",
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """
    Context manager for tracing LLM calls in a thread.

    This context manager combines thread context with LLM tracing,
    automatically linking the LLM call to the thread.

    Args:
        thread: Thread context to associate with the LLM call
        model: Model name
        provider: Provider name
        operation: Operation name
        **kwargs: Additional attributes for the LLM span

    Yields:
        LLM span context

    Example:
        with thread_context(name="conversation") as thread:
            thread.add_message("user", "What is AI?")

            with trace_thread_llm(thread, "gpt-4", "openai") as llm_span:
                response = llm_client.chat.completions.create(...)
                thread.add_message("assistant", response.choices[0].message.content)
    """
    # Add thread context to LLM attributes
    thread_attributes = {
        "thread.id": thread.thread_id,
        "thread.name": thread.name,
        "thread.message_count": len(thread.messages),
        "thread.turn_count": thread.turn_count,
    }

    # Merge with provided attributes
    attributes = {**thread_attributes, **kwargs}

    # Use the LLM tracing context manager
    with trace_llm(
        model=model,
        provider=provider,
        operation=operation,
        **attributes,
    ) as llm_span:
        # Update thread metadata
        thread.last_updated_at = time.time()

        yield llm_span
