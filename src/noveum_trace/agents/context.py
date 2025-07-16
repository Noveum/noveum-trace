"""
Context management for multi-agent tracing.
"""

import logging
import threading
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from noveum_trace.core.tracer import NoveumTracer

    from .agent import Agent

logger = logging.getLogger(__name__)

# Context variables for async support
_current_agent: ContextVar[Optional["Agent"]] = ContextVar(
    "current_agent", default=None
)

# Thread-local storage for sync support
_thread_local = threading.local()


class AgentContext:
    """Context manager for agent-scoped operations."""

    def __init__(self, agent: "Agent") -> None:
        """Initialize agent context."""
        self._agent = agent
        self._previous_agent: Optional[Agent] = None

    def __enter__(self) -> "Agent":
        """Enter the agent context."""
        self._previous_agent = get_current_agent()
        set_current_agent(self._agent)
        return self._agent

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the agent context."""
        set_current_agent(self._previous_agent)

        # Record any exception
        if exc_type is not None:
            logger.warning(
                f"Exception in agent context for '{self._agent.name}': {exc_val}"
            )


class AsyncAgentContext:
    """Async context manager for agent operations."""

    def __init__(self, agent: "Agent") -> None:
        """Initialize async agent context."""
        self._agent = agent
        self._previous_agent: Optional[Agent] = None

    async def __aenter__(self) -> "Agent":
        """Enter the async agent context."""
        self._previous_agent = get_current_agent()
        set_current_agent(self._agent)
        return self._agent

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async agent context."""
        set_current_agent(self._previous_agent)

        # Record any exception
        if exc_type is not None:
            logger.warning(
                f"Exception in async agent context for '{self._agent.name}': {exc_val}"
            )


def get_current_agent() -> Optional["Agent"]:
    """Get the currently active agent."""
    # Try async context first
    try:
        agent = _current_agent.get()
        if agent is not None:
            return agent
    except LookupError:
        # No async context available
        pass

    # Fall back to thread-local storage
    if hasattr(_thread_local, "current_agent"):
        return _thread_local.current_agent

    return None


def set_current_agent(agent: Optional["Agent"]) -> None:
    """Set the currently active agent."""
    # Set in async context
    _current_agent.set(agent)

    # Also set in thread-local for sync compatibility
    _thread_local.current_agent = agent

    if agent:
        logger.debug(f"Set current agent to '{agent.name}'")
    else:
        logger.debug("Cleared current agent")


def get_current_agent_name() -> Optional[str]:
    """Get the name of the currently active agent."""
    agent = get_current_agent()
    return agent.name if agent else None


def get_current_tracer() -> Optional["NoveumTracer"]:
    """Get the current tracer from the current agent."""
    agent = get_current_agent()
    return agent.tracer if agent else None


def with_agent(agent: "Agent") -> Callable:
    """Decorator to run a function with a specific agent context."""

    def decorator(func: Callable) -> Callable:
        if callable(func):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with AgentContext(agent):
                    return func(*args, **kwargs)

            return wrapper
        else:
            raise TypeError("Expected callable, got {type(func)}")

    return decorator


def with_async_agent(agent: "Agent") -> Callable:
    """Decorator to run an async function with a specific agent context."""

    def decorator(func: Callable) -> Callable:
        if callable(func):

            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                async with AsyncAgentContext(agent):
                    return await func(*args, **kwargs)

            return wrapper
        else:
            raise TypeError("Expected callable, got {type(func)}")

    return decorator


def clear_agent_context() -> None:
    """Clear the current agent context."""
    set_current_agent(None)
