"""
Agent class for managing individual AI agents and their lifecycle.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from noveum_trace.core.tracer import NoveumTracer, TracerConfig
from noveum_trace.types import CustomHeaders
from noveum_trace.utils.exceptions import (
    ConfigurationError,
    ValidationError,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""

    # Agent identity
    name: str
    agent_type: str = "generic"  # Changed from 'type' to 'agent_type'
    id: Optional[str] = None  # Added id field
    description: Optional[str] = None
    version: str = "1.0.0"

    # Agent capabilities and metadata
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    # Tracing configuration (inherits from project defaults)
    custom_headers: Optional[CustomHeaders] = None
    sampling_rate: Optional[float] = None
    capture_llm_content: Optional[bool] = None

    # Agent-specific settings
    max_concurrent_traces: int = 100
    trace_retention_hours: int = 24
    enable_metrics: bool = True
    enable_evaluation: bool = True

    # Relationship configuration
    parent_agent: Optional[str] = None
    child_agents: Set[str] = field(default_factory=set)

    def validate(self) -> None:
        """Validate agent configuration."""
        if not self.name:
            raise ValidationError("Agent name must be a non-empty string")

        if self.sampling_rate is not None and not (0.0 <= self.sampling_rate <= 1.0):
            raise ValidationError("Sampling rate must be between 0.0 and 1.0")

        if self.max_concurrent_traces <= 0:
            raise ConfigurationError("max_concurrent_traces must be positive")

        if self.trace_retention_hours <= 0:
            raise ConfigurationError("trace_retention_hours must be positive")


class Agent:
    """Represents an individual agent in a multi-agent system."""

    def __init__(
        self, config: AgentConfig, project_tracer_config: Optional[TracerConfig] = None
    ):
        """Initialize the agent."""
        config.validate()
        self._config = config
        self._agent_id = config.id or str(uuid.uuid4())
        self._created_at = datetime.now(timezone.utc)
        self._active_traces: Dict[str, Any] = {}
        self._trace_count = 0
        self._lock = threading.RLock()

        # Create agent-specific tracer configuration
        self._tracer_config = self._create_tracer_config(project_tracer_config)
        self._tracer = NoveumTracer(self._tracer_config)

        # Agent state
        self._is_active = True
        self._last_activity = self._created_at

        logger.info(f"Agent '{self.name}' initialized with ID {self._agent_id}")

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._config.name

    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self._agent_id

    @property
    def agent_type(self) -> str:
        """Get agent type."""
        return self._config.agent_type

    @property
    def config(self) -> AgentConfig:
        """Get agent configuration."""
        return self._config

    @property
    def tracer(self) -> NoveumTracer:
        """Get agent's tracer."""
        return self._tracer

    @property
    def is_active(self) -> bool:
        """Check if agent is active."""
        return self._is_active

    @property
    def trace_count(self) -> int:
        """Get number of traces created by this agent."""
        return self._trace_count

    @property
    def active_trace_count(self) -> int:
        """Get number of active traces for this agent."""
        with self._lock:
            return len(self._active_traces)

    def add_capability(self, capability: str) -> None:
        """Add a capability to the agent."""
        self._config.capabilities.add(capability)
        logger.debug(f"Added capability '{capability}' to agent '{self.name}'")

    def remove_capability(self, capability: str) -> None:
        """Remove a capability from the agent."""
        self._config.capabilities.discard(capability)
        logger.debug(f"Removed capability '{capability}' from agent '{self.name}'")

    def add_tag(self, tag: str) -> None:
        """Add a tag to the agent."""
        self._config.tags.add(tag)
        logger.debug(f"Added tag '{tag}' to agent '{self.name}'")

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the agent."""
        self._config.tags.discard(tag)
        logger.debug(f"Removed tag '{tag}' from agent '{self.name}'")

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for the agent."""
        self._config.metadata[key] = value
        logger.debug(f"Set metadata '{key}' to '{value}' for agent '{self.name}'")

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata for the agent."""
        return self._config.metadata.get(key, default)

    def has_capability(self, capability: str) -> bool:
        """Check if the agent has a specific capability."""
        return capability in self._config.capabilities

    def has_tag(self, tag: str) -> bool:
        """Check if the agent has a specific tag."""
        return tag in self._config.tags

    def get_active_traces(self) -> Dict[str, Any]:
        """Get active traces for the agent."""
        with self._lock:
            return self._active_traces.copy()

    def register_trace(self, trace_id: str, trace_data: Dict[str, Any]) -> None:
        """Register a trace with the agent."""
        with self._lock:
            self._active_traces[trace_id] = trace_data
            self._trace_count += 1
            self._last_activity = datetime.now(timezone.utc)
        logger.debug(f"Registered trace '{trace_id}' with agent '{self.name}'")

    def unregister_trace(self, trace_id: str) -> None:
        """Unregister a trace from the agent."""
        with self._lock:
            if trace_id in self._active_traces:
                del self._active_traces[trace_id]
                self._last_activity = datetime.now(timezone.utc)
        logger.debug(f"Unregistered trace '{trace_id}' from agent '{self.name}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "agent_id": self._agent_id,
            "name": self.name,
            "agent_type": self.agent_type,
            "description": self._config.description,
            "version": self._config.version,
            "capabilities": list(self._config.capabilities),
            "tags": list(self._config.tags),
            "metadata": self._config.metadata.copy(),
            "is_active": self._is_active,
            "trace_count": self._trace_count,
            "active_trace_count": len(self._active_traces),
            "created_at": self._created_at.isoformat(),
            "last_activity": self._last_activity.isoformat(),
            "child_agents": list(self._config.child_agents),
            "parent_agent": self._config.parent_agent,
            "sampling_rate": self._config.sampling_rate,
            "enable_metrics": self._config.enable_metrics,
            "enable_evaluation": self._config.enable_evaluation,
        }

    def activate(self) -> None:
        """Activate the agent."""
        with self._lock:
            self._is_active = True
            self._last_activity = datetime.now(timezone.utc)
        logger.info(f"Agent '{self.name}' activated")

    def deactivate(self) -> None:
        """Deactivate the agent."""
        with self._lock:
            self._is_active = False
        logger.info(f"Agent '{self.name}' deactivated")

    def add_child_agent(self, child_agent_name: str) -> None:
        """Add a child agent."""
        self._config.child_agents.add(child_agent_name)
        logger.debug(f"Added child agent '{child_agent_name}' to '{self.name}'")

    def remove_child_agent(self, child_agent_name: str) -> None:
        """Remove a child agent."""
        self._config.child_agents.discard(child_agent_name)
        logger.debug(f"Removed child agent '{child_agent_name}' from '{self.name}'")

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        with self._lock:
            return {
                "agent_id": self._agent_id,
                "name": self.name,
                "type": self.agent_type,
                "is_active": self._is_active,
                "trace_count": self._trace_count,
                "active_traces": len(self._active_traces),
                "created_at": self._created_at.isoformat(),
                "last_activity": self._last_activity.isoformat(),
                "capabilities": list(self._config.capabilities),
                "tags": list(self._config.tags),
                "child_agents": list(self._config.child_agents),
            }

    def shutdown(self) -> None:
        """Shutdown the agent."""
        logger.info(f"Shutting down agent '{self.name}'")

        with self._lock:
            self._is_active = False
            self._active_traces.clear()

        # Shutdown the tracer
        if self._tracer:
            self._tracer.shutdown()

        logger.info(f"Agent '{self.name}' shutdown complete")

    def _create_tracer_config(
        self, project_config: Optional[TracerConfig]
    ) -> TracerConfig:
        """Create agent-specific tracer configuration."""
        if project_config:
            # Inherit from project configuration
            config = TracerConfig(
                project_id=project_config.project_id,
                project_name=project_config.project_name,
                org_id=project_config.org_id,
                user_id=project_config.user_id,
                session_id=project_config.session_id,
                environment=project_config.environment,
                custom_headers=project_config.custom_headers,
                sampling_rate=project_config.sampling_rate,
                max_queue_size=project_config.max_queue_size,
                batch_size=project_config.batch_size,
                batch_timeout_ms=project_config.batch_timeout_ms,
                max_spans_per_trace=project_config.max_spans_per_trace,
                sinks=project_config.sinks,
            )
        else:
            # Create default config with required project_id
            config = TracerConfig(
                project_id="default-agent-project", project_name="Agent Project"
            )

        # Apply agent-specific overrides
        if self._config.custom_headers:
            # Create agent-specific headers
            agent_headers = CustomHeaders(
                project_id=config.project_id,
                org_id=self._config.custom_headers.org_id,
                user_id=self._config.custom_headers.user_id,
                session_id=self._config.custom_headers.session_id,
                additional_headers={
                    **self._config.custom_headers.additional_headers,
                    "agent.name": self.name,
                    "agent.id": self._agent_id,
                    "agent.type": self.agent_type,
                },
            )
            config.custom_headers = agent_headers.to_dict()

        # Apply other agent-specific overrides
        if self._config.sampling_rate is not None:
            config.sampling_rate = self._config.sampling_rate

        if self._config.capture_llm_content is not None:
            config.capture_content = self._config.capture_llm_content

        return config

    def __repr__(self) -> str:
        """Return string representation of the agent."""
        return f"Agent(name='{self.name}', id='{self._agent_id}', type='{self.agent_type}')"
