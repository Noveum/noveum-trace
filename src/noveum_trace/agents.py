"""
Advanced agent interaction tracing for Noveum Trace SDK.

This module provides specialized components for tracing complex agent interactions,
including multi-agent systems, agent workflows, and agent graphs.
"""

import os
import threading
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Callable, Optional

from noveum_trace.context_managers import trace_agent, trace_operation
from noveum_trace.utils.exceptions import NoveumTraceError


class AgentNode:
    """
    Context manager for tracing individual agent execution.

    This class provides a way to trace agent interactions, including
    agent-to-agent communication and task execution.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: str = "generic",
        capabilities: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize an agent node.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (task, conversational, etc.)
            capabilities: List of agent capabilities
            metadata: Additional metadata for the agent
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type
        self.capabilities = capabilities or []
        self.metadata = metadata or {}

        self.messages: list[dict[str, Any]] = []
        self.tasks: list[dict[str, Any]] = []
        self.interactions: list[dict[str, Any]] = []
        self.created_at = time.time()
        self.last_updated_at = self.created_at

        self.span: Optional[Any] = None
        self.span_context: Optional[Any] = None
        self.is_active = False

    def __enter__(self) -> "AgentNode":
        """Enter the agent context."""
        self.is_active = True

        # Create a span for the agent
        self.span_context = trace_agent(
            agent_type=self.agent_type,
            capabilities=self.capabilities,
            attributes={
                "agent.id": self.agent_id,
                "agent.type": self.agent_type,
                "agent.capabilities": self.capabilities,
                "agent.created_at": self.created_at,
                "agent.metadata": self.metadata,
            },
        )
        self.span = self.span_context.__enter__()

        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the agent context."""
        if self.span:
            # Update final agent metrics
            self.span.set_attributes(
                {
                    "agent.message_count": len(self.messages),
                    "agent.task_count": len(self.tasks),
                    "agent.interaction_count": len(self.interactions),
                    "agent.duration": time.time() - self.created_at,
                    "agent.last_updated_at": self.last_updated_at,
                }
            )

        # Let the agent context manager handle the exception
        if self.span_context:
            result = self.span_context.__exit__(exc_type, exc_val, exc_tb)
            self.is_active = False
            return result

    def add_message(
        self, content: str, role: str, metadata: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Add a message to the agent.

        Args:
            content: Message content
            role: Message role (input, output, internal)
            metadata: Additional message metadata

        Returns:
            Message object with metadata
        """
        if not self.is_active:
            raise NoveumTraceError("Cannot add message to inactive agent")

        message_id = f"msg_{uuid.uuid4().hex}"
        timestamp = time.time()

        message = {
            "id": message_id,
            "agent_id": self.agent_id,
            "content": content,
            "role": role,
            "created_at": timestamp,
            "metadata": metadata or {},
        }

        self.messages.append(message)
        self.last_updated_at = timestamp

        # Update agent metrics
        if self.span:
            self.span.set_attributes(
                {
                    "agent.message_count": len(self.messages),
                    "agent.last_updated_at": timestamp,
                    "agent.last_message_role": role,
                }
            )

        return message

    def add_task(
        self,
        task_id: str,
        description: str,
        status: str = "pending",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Add a task to the agent.

        Args:
            task_id: Task identifier
            description: Task description
            status: Task status (pending, in_progress, completed, failed)
            metadata: Additional task metadata

        Returns:
            Task object with metadata
        """
        if not self.is_active:
            raise NoveumTraceError("Cannot add task to inactive agent")

        timestamp = time.time()
        task = {
            "id": task_id,
            "agent_id": self.agent_id,
            "description": description,
            "status": status,
            "created_at": timestamp,
            "updated_at": timestamp,
            "metadata": metadata or {},
        }

        self.tasks.append(task)
        self.last_updated_at = timestamp

        # Update agent metrics
        if self.span:
            self.span.set_attributes(
                {
                    "agent.task_count": len(self.tasks),
                    "agent.last_updated_at": timestamp,
                    "agent.last_task_status": status,
                }
            )

        return task

    def update_task(
        self, task_id: str, status: str, metadata: Optional[dict[str, Any]] = None
    ) -> Optional[dict[str, Any]]:
        """
        Update a task status.

        Args:
            task_id: Task identifier
            status: New task status
            metadata: Additional metadata to merge

        Returns:
            Updated task object or None if not found
        """
        if not self.is_active:
            raise NoveumTraceError("Cannot update task for inactive agent")

        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = status
                task["updated_at"] = time.time()
                if metadata:
                    task["metadata"].update(metadata)

                self.last_updated_at = task["updated_at"]

                # Update agent metrics
                if self.span:
                    self.span.set_attributes(
                        {
                            "agent.last_updated_at": self.last_updated_at,
                            "agent.last_task_status": status,
                        }
                    )

                return task

        return None

    def add_interaction(
        self,
        target_agent_id: str,
        interaction_type: str,
        content: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Record an interaction with another agent.

        Args:
            target_agent_id: ID of the target agent
            interaction_type: Type of interaction (message, task_delegation, etc.)
            content: Interaction content
            metadata: Additional interaction metadata

        Returns:
            Interaction object with metadata
        """
        if not self.is_active:
            raise NoveumTraceError("Cannot add interaction to inactive agent")

        interaction_id = f"int_{uuid.uuid4().hex}"
        timestamp = time.time()

        interaction = {
            "id": interaction_id,
            "source_agent_id": self.agent_id,
            "target_agent_id": target_agent_id,
            "interaction_type": interaction_type,
            "content": content,
            "created_at": timestamp,
            "metadata": metadata or {},
        }

        self.interactions.append(interaction)
        self.last_updated_at = timestamp

        # Update agent metrics
        if self.span:
            self.span.set_attributes(
                {
                    "agent.interaction_count": len(self.interactions),
                    "agent.last_updated_at": timestamp,
                    "agent.last_interaction_type": interaction_type,
                    "agent.last_interaction_target": target_agent_id,
                }
            )

        return interaction


class AgentGraph:
    """
    Context manager for tracing agent graphs and multi-agent interactions.

    This class provides a way to trace complex agent interactions,
    including agent graphs and multi-agent workflows.
    """

    def __init__(
        self,
        graph_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize an agent graph.

        Args:
            graph_id: Unique identifier for the graph
            name: Human-readable name for the graph
            metadata: Additional metadata for the graph
        """
        self.graph_id = graph_id or str(uuid.uuid4())
        self.name = name or f"graph_{self.graph_id[:8]}"
        self.metadata = metadata or {}

        self.agents: dict[str, AgentNode] = {}
        self.connections: list[dict[str, Any]] = []
        self.created_at = time.time()
        self.last_updated_at = self.created_at

        self.span: Optional[Any] = None
        self.span_context: Optional[Any] = None
        self.is_active = False

    def __enter__(self) -> "AgentGraph":
        """Enter the agent graph context."""
        self.is_active = True

        # Create a span for the agent graph
        self.span_context = trace_operation(
            operation_name=f"agent_graph_{self.name}",
            attributes={
                "agent_graph.id": self.graph_id,
                "agent_graph.name": self.name,
                "agent_graph.created_at": self.created_at,
                "agent_graph.metadata": self.metadata,
            },
        )
        self.span = self.span_context.__enter__()

        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the agent graph context."""
        if self.span:
            # Update final graph metrics
            self.span.set_attributes(
                {
                    "agent_graph.node_count": len(self.agents),
                    "agent_graph.connection_count": len(self.connections),
                    "agent_graph.duration": time.time() - self.created_at,
                    "agent_graph.last_updated_at": self.last_updated_at,
                }
            )

        # Let the graph context manager handle the exception
        if self.span_context:
            result = self.span_context.__exit__(exc_type, exc_val, exc_tb)
            self.is_active = False
            return result

    def add_agent(
        self,
        agent_id: str,
        agent_type: str = "generic",
        capabilities: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentNode:
        """
        Add an agent to the graph.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent
            capabilities: List of agent capabilities
            metadata: Additional agent metadata

        Returns:
            AgentNode instance
        """
        if not self.is_active:
            raise NoveumTraceError("Cannot add agent to inactive graph")

        agent = AgentNode(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities,
            metadata=metadata,
        )

        self.agents[agent_id] = agent
        self.last_updated_at = time.time()

        # Update graph metrics
        if self.span:
            self.span.set_attributes(
                {
                    "agent_graph.agent_count": len(self.agents),
                    "agent_graph.last_updated_at": self.last_updated_at,
                }
            )

        return agent

    def add_connection(
        self,
        source_agent_id: str,
        target_agent_id: str,
        connection_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Add a connection between agents.

        Args:
            source_agent_id: ID of the source agent
            target_agent_id: ID of the target agent
            connection_type: Type of connection (communication, delegation, etc.)
            metadata: Additional connection metadata

        Returns:
            Connection object with metadata
        """
        if not self.is_active:
            raise NoveumTraceError("Cannot add connection to inactive graph")

        connection_id = f"conn_{uuid.uuid4().hex}"
        timestamp = time.time()

        connection = {
            "id": connection_id,
            "graph_id": self.graph_id,
            "source_agent_id": source_agent_id,
            "target_agent_id": target_agent_id,
            "connection_type": connection_type,
            "created_at": timestamp,
            "metadata": metadata or {},
        }

        self.connections.append(connection)
        self.last_updated_at = timestamp

        # Update graph metrics
        if self.span:
            self.span.set_attributes(
                {
                    "agent_graph.connection_count": len(self.connections),
                    "agent_graph.last_updated_at": timestamp,
                    "agent_graph.last_connection_type": connection_type,
                }
            )

        return connection

    def get_agent(self, agent_id: str) -> Optional[AgentNode]:
        """
        Get an agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentNode if found, None otherwise
        """
        return self.agents.get(agent_id)

    def get_connections(self, agent_id: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Get connections, optionally filtered by agent.

        Args:
            agent_id: Filter connections by agent ID

        Returns:
            List of connection objects
        """
        if agent_id:
            return [
                conn
                for conn in self.connections
                if conn["source_agent_id"] == agent_id
                or conn["target_agent_id"] == agent_id
            ]
        return self.connections.copy()


class AgentWorkflow:
    """
    Context manager for tracing agent workflows.

    This class provides a way to trace complex agent workflows,
    including sequential and parallel task execution.
    """

    def __init__(
        self,
        workflow_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize an agent workflow.

        Args:
            workflow_id: Unique identifier for the workflow
            name: Human-readable name for the workflow
            metadata: Additional metadata for the workflow
        """
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.name = name or f"workflow_{self.workflow_id[:8]}"
        self.metadata = metadata or {}

        self.tasks: list[dict[str, Any]] = []
        self.dependencies: list[dict[str, Any]] = []
        self.created_at = time.time()
        self.last_updated_at = self.created_at

        self.span: Optional[Any] = None
        self.span_context: Optional[Any] = None
        self.is_active = False

    def __enter__(self) -> "AgentWorkflow":
        """Enter the workflow context."""
        self.is_active = True

        # Create a span for the workflow
        self.span_context = trace_operation(
            operation_name=f"agent_workflow_{self.name}",
            attributes={
                "agent_workflow.id": self.workflow_id,
                "agent_workflow.name": self.name,
                "agent_workflow.created_at": self.created_at,
                "agent_workflow.metadata": self.metadata,
            },
        )
        self.span = self.span_context.__enter__()

        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the workflow context."""
        if self.span:
            # Calculate workflow metrics
            completed_tasks = [t for t in self.tasks if t.get("status") == "completed"]
            failed_tasks = [t for t in self.tasks if t.get("status") == "failed"]

            self.span.set_attributes(
                {
                    "agent_workflow.task_count": len(self.tasks),
                    "agent_workflow.completed_tasks": len(completed_tasks),
                    "agent_workflow.failed_tasks": len(failed_tasks),
                    "agent_workflow.success_rate": (
                        len(completed_tasks) / len(self.tasks) if self.tasks else 0
                    ),
                    "agent_workflow.duration": time.time() - self.created_at,
                    "agent_workflow.last_updated_at": self.last_updated_at,
                }
            )

        # Let the workflow context manager handle the exception
        if self.span_context:
            result = self.span_context.__exit__(exc_type, exc_val, exc_tb)
            self.is_active = False
            return result

    def add_task(
        self,
        task_id: str,
        agent_id: str,
        description: str,
        dependencies: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Add a task to the workflow.

        Args:
            task_id: Task identifier
            agent_id: ID of the agent responsible for the task
            description: Task description
            dependencies: List of task IDs this task depends on
            metadata: Additional task metadata

        Returns:
            Task object with metadata
        """
        if not self.is_active:
            raise NoveumTraceError("Cannot add task to inactive workflow")

        timestamp = time.time()
        task = {
            "id": task_id,
            "workflow_id": self.workflow_id,
            "agent_id": agent_id,
            "description": description,
            "status": "pending",
            "created_at": timestamp,
            "updated_at": timestamp,
            "metadata": metadata or {},
        }

        self.tasks.append(task)
        self.last_updated_at = timestamp

        # Add dependencies if specified
        if dependencies:
            for dep_id in dependencies:
                dependency = {
                    "id": f"dep_{uuid.uuid4().hex}",
                    "workflow_id": self.workflow_id,
                    "task_id": task_id,
                    "depends_on": dep_id,
                    "created_at": timestamp,
                }
                self.dependencies.append(dependency)

        # Update workflow metrics
        if self.span:
            self.span.set_attributes(
                {
                    "agent_workflow.task_count": len(self.tasks),
                    "agent_workflow.last_updated_at": timestamp,
                    "agent_workflow.last_task_agent": agent_id,
                }
            )

        return task

    def update_task_status(
        self, task_id: str, status: str, metadata: Optional[dict[str, Any]] = None
    ) -> Optional[dict[str, Any]]:
        """
        Update a task status.

        Args:
            task_id: Task identifier
            status: New task status
            metadata: Additional metadata to merge

        Returns:
            Updated task object or None if not found
        """
        if not self.is_active:
            raise NoveumTraceError("Cannot update task for inactive workflow")

        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = status
                task["updated_at"] = time.time()
                if metadata:
                    task["metadata"].update(metadata)

                self.last_updated_at = task["updated_at"]

                # Update workflow metrics
                if self.span:
                    completed_tasks = [
                        t for t in self.tasks if t.get("status") == "completed"
                    ]
                    self.span.set_attributes(
                        {
                            "agent_workflow.completed_tasks": len(completed_tasks),
                            "agent_workflow.last_updated_at": self.last_updated_at,
                            "agent_workflow.last_task_status": status,
                        }
                    )

                return task

        return None

    def get_ready_tasks(self) -> list[dict[str, Any]]:
        """
        Get tasks that are ready to execute (no pending dependencies).

        Returns:
            List of ready task objects
        """
        ready_tasks = []

        for task in self.tasks:
            if task["status"] != "pending":
                continue

            # Check if all dependencies are completed
            task_deps = [
                dep for dep in self.dependencies if dep["task_id"] == task["id"]
            ]

            if not task_deps:
                # No dependencies, task is ready
                ready_tasks.append(task)
                continue

            # Check if all dependencies are completed
            all_deps_completed = True
            for dep in task_deps:
                dep_task = next(
                    (t for t in self.tasks if t["id"] == dep["depends_on"]), None
                )
                if not dep_task or dep_task["status"] != "completed":
                    all_deps_completed = False
                    break

            if all_deps_completed:
                ready_tasks.append(task)

        return ready_tasks

    def get_workflow_status(self) -> dict[str, Any]:
        """
        Get the current workflow status.

        Returns:
            Dictionary with workflow status information
        """
        pending_tasks = [t for t in self.tasks if t.get("status") == "pending"]
        in_progress_tasks = [t for t in self.tasks if t.get("status") == "in_progress"]
        completed_tasks = [t for t in self.tasks if t.get("status") == "completed"]
        failed_tasks = [t for t in self.tasks if t.get("status") == "failed"]

        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "total_tasks": len(self.tasks),
            "pending_tasks": len(pending_tasks),
            "in_progress_tasks": len(in_progress_tasks),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": (
                len(completed_tasks) / len(self.tasks) if self.tasks else 0
            ),
            "is_complete": len(pending_tasks) == 0 and len(in_progress_tasks) == 0,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
        }


# Global agent tracking
_agent_graphs: dict[str, AgentGraph] = {}
_agent_workflows: dict[str, AgentWorkflow] = {}


def create_agent_node(
    agent_id: Optional[str] = None,
    agent_type: str = "generic",
    capabilities: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> AgentNode:
    """
    Create a new agent node.

    Args:
        agent_id: Unique identifier for the agent
        agent_type: Type of agent
        capabilities: List of agent capabilities
        metadata: Additional agent metadata

    Returns:
        AgentNode instance
    """
    return AgentNode(
        agent_id=agent_id,
        agent_type=agent_type,
        capabilities=capabilities,
        metadata=metadata,
    )


# Global agent registry
_agents: dict[str, AgentNode] = {}


def create_agent(
    agent_id: Optional[str] = None,
    agent_type: str = "generic",
    capabilities: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> AgentNode:
    """
    Create a new agent and register it globally.

    Args:
        agent_id: Unique identifier for the agent
        agent_type: Type of agent
        capabilities: List of agent capabilities
        metadata: Additional agent metadata

    Returns:
        AgentNode instance
    """
    agent = create_agent_node(
        agent_id=agent_id,
        agent_type=agent_type,
        capabilities=capabilities,
        metadata=metadata,
    )
    _agents[agent.agent_id] = agent
    return agent


def get_agent(agent_id: str) -> Optional[AgentNode]:
    """
    Get an agent by ID.

    Args:
        agent_id: Agent identifier

    Returns:
        AgentNode if found, None otherwise
    """
    return _agents.get(agent_id)


def create_agent_graph(
    graph_id: Optional[str] = None,
    name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> AgentGraph:
    """
    Create a new agent graph.

    Args:
        graph_id: Unique identifier for the graph
        name: Human-readable name for the graph
        metadata: Additional metadata for the graph

    Returns:
        AgentGraph instance
    """
    graph = AgentGraph(graph_id=graph_id, name=name, metadata=metadata)
    _agent_graphs[graph.graph_id] = graph
    return graph


def create_agent_workflow(
    workflow_id: Optional[str] = None,
    name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> AgentWorkflow:
    """
    Create a new agent workflow.

    Args:
        workflow_id: Unique identifier for the workflow
        name: Human-readable name for the workflow
        metadata: Additional metadata for the workflow

    Returns:
        AgentWorkflow instance
    """
    workflow = AgentWorkflow(workflow_id=workflow_id, name=name, metadata=metadata)
    _agent_workflows[workflow.workflow_id] = workflow
    return workflow


def get_agent_graph(graph_id: str) -> Optional[AgentGraph]:
    """
    Get an agent graph by ID.

    Args:
        graph_id: Graph identifier

    Returns:
        AgentGraph if found, None otherwise
    """
    return _agent_graphs.get(graph_id)


def get_agent_workflow(workflow_id: str) -> Optional[AgentWorkflow]:
    """
    Get an agent workflow by ID.

    Args:
        workflow_id: Workflow identifier

    Returns:
        AgentWorkflow if found, None otherwise
    """
    return _agent_workflows.get(workflow_id)


def list_agent_graphs() -> list[dict[str, Any]]:
    """
    List all agent graphs.

    Returns:
        List of graph metadata
    """
    return [
        {
            "id": graph.graph_id,
            "name": graph.name,
            "agent_count": len(graph.agents),
            "connection_count": len(graph.connections),
            "created_at": graph.created_at,
            "last_updated_at": graph.last_updated_at,
            "is_active": graph.is_active,
        }
        for graph in _agent_graphs.values()
    ]


def list_agent_workflows() -> list[dict[str, Any]]:
    """
    List all agent workflows.

    Returns:
        List of workflow metadata
    """
    return [
        {
            "id": workflow.workflow_id,
            "name": workflow.name,
            "task_count": len(workflow.tasks),
            "created_at": workflow.created_at,
            "last_updated_at": workflow.last_updated_at,
            "is_active": workflow.is_active,
        }
        for workflow in _agent_workflows.values()
    ]


# Specialized tracing for agent operations


def trace_agent_operation(
    agent: AgentNode, operation: str, **kwargs: Any
) -> Generator[Any, None, None]:
    """
    Context manager for tracing agent operations.

    This context manager combines agent context with operation tracing,
    automatically linking the operation to the agent.

    Args:
        agent: Agent node
        operation: Operation being performed
        **kwargs: Additional span attributes

    Returns:
        Context manager for the operation

    Example:
        agent = create_agent_node(agent_type="research_agent")

        with agent:
            with trace_agent_operation(agent, "web_search") as span:
                # Perform web search
                results = search_web(query)

                # Add results to agent
                agent.add_message(str(results), "output")

                # Add metrics to span
                span.set_attributes({
                    "search.query": query,
                    "search.results_count": len(results),
                    "search.duration": search_time
                })
    """
    # Combine agent attributes with operation attributes
    attributes = {
        "agent.id": agent.agent_id,
        "agent.type": agent.agent_type,
        "agent.capabilities": agent.capabilities,
        "agent.operation": operation,
        **kwargs.get("attributes", {}),
    }

    kwargs["attributes"] = attributes

    # Create operation span with agent context
    with trace_operation(operation_name=f"agent_{operation}", **kwargs) as span:
        yield span


# =============================================================================
# REGISTRY CLEANUP MECHANISMS
# =============================================================================

# Registry size limits (can be configured via environment variables)
MAX_AGENTS = int(os.getenv("NOVEUM_MAX_AGENTS", "1000"))
MAX_AGENT_GRAPHS = int(os.getenv("NOVEUM_MAX_AGENT_GRAPHS", "100"))
MAX_AGENT_WORKFLOWS = int(os.getenv("NOVEUM_MAX_AGENT_WORKFLOWS", "100"))

# TTL tracking for automatic cleanup
_agent_timestamps: dict[str, float] = {}
_graph_timestamps: dict[str, float] = {}
_workflow_timestamps: dict[str, float] = {}

# Cleanup lock for thread safety
_cleanup_lock = threading.Lock()


def cleanup_agent(agent_id: str) -> bool:
    """
    Remove a specific agent from the global registry.

    Args:
        agent_id: Agent identifier to remove

    Returns:
        True if agent was removed, False if not found
    """
    with _cleanup_lock:
        removed = agent_id in _agents
        if removed:
            del _agents[agent_id]
            _agent_timestamps.pop(agent_id, None)
        return removed


def cleanup_agent_graph(graph_id: str) -> bool:
    """
    Remove a specific agent graph from the global registry.

    Args:
        graph_id: Graph identifier to remove

    Returns:
        True if graph was removed, False if not found
    """
    with _cleanup_lock:
        removed = graph_id in _agent_graphs
        if removed:
            del _agent_graphs[graph_id]
            _graph_timestamps.pop(graph_id, None)
        return removed


def cleanup_agent_workflow(workflow_id: str) -> bool:
    """
    Remove a specific agent workflow from the global registry.

    Args:
        workflow_id: Workflow identifier to remove

    Returns:
        True if workflow was removed, False if not found
    """
    with _cleanup_lock:
        removed = workflow_id in _agent_workflows
        if removed:
            del _agent_workflows[workflow_id]
            _workflow_timestamps.pop(workflow_id, None)
        return removed


def cleanup_all_registries() -> dict[str, int]:
    """
    Clear all global registries.

    Returns:
        Dictionary with counts of cleared items
    """
    with _cleanup_lock:
        counts = {
            "agents": len(_agents),
            "graphs": len(_agent_graphs),
            "workflows": len(_agent_workflows),
        }

        _agents.clear()
        _agent_graphs.clear()
        _agent_workflows.clear()
        _agent_timestamps.clear()
        _graph_timestamps.clear()
        _workflow_timestamps.clear()

        return counts


def cleanup_by_ttl(ttl_seconds: float = 3600) -> dict[str, int]:
    """
    Clean up entries older than the specified TTL.

    Args:
        ttl_seconds: Time-to-live in seconds (default: 1 hour)

    Returns:
        Dictionary with counts of cleaned items
    """
    import time

    current_time = time.time()
    cutoff_time = current_time - ttl_seconds

    with _cleanup_lock:
        # Clean up expired agents
        expired_agents = [
            agent_id
            for agent_id, timestamp in _agent_timestamps.items()
            if timestamp < cutoff_time
        ]
        for agent_id in expired_agents:
            _agents.pop(agent_id, None)
            _agent_timestamps.pop(agent_id, None)

        # Clean up expired graphs
        expired_graphs = [
            graph_id
            for graph_id, timestamp in _graph_timestamps.items()
            if timestamp < cutoff_time
        ]
        for graph_id in expired_graphs:
            _agent_graphs.pop(graph_id, None)
            _graph_timestamps.pop(graph_id, None)

        # Clean up expired workflows
        expired_workflows = [
            workflow_id
            for workflow_id, timestamp in _workflow_timestamps.items()
            if timestamp < cutoff_time
        ]
        for workflow_id in expired_workflows:
            _agent_workflows.pop(workflow_id, None)
            _workflow_timestamps.pop(workflow_id, None)

        return {
            "agents": len(expired_agents),
            "graphs": len(expired_graphs),
            "workflows": len(expired_workflows),
        }


def enforce_size_limits() -> dict[str, int]:
    """
    Enforce size limits on registries by removing oldest entries.

    Returns:
        Dictionary with counts of evicted items
    """
    with _cleanup_lock:
        evicted_counts = {"agents": 0, "graphs": 0, "workflows": 0}

        # Enforce agent limit
        if len(_agents) > MAX_AGENTS:
            # Sort by timestamp and remove oldest
            sorted_agents = sorted(_agent_timestamps.items(), key=lambda x: x[1])
            num_to_remove = len(_agents) - MAX_AGENTS
            for agent_id, _ in sorted_agents[:num_to_remove]:
                _agents.pop(agent_id, None)
                _agent_timestamps.pop(agent_id, None)
                evicted_counts["agents"] += 1

        # Enforce graph limit
        if len(_agent_graphs) > MAX_AGENT_GRAPHS:
            sorted_graphs = sorted(_graph_timestamps.items(), key=lambda x: x[1])
            num_to_remove = len(_agent_graphs) - MAX_AGENT_GRAPHS
            for graph_id, _ in sorted_graphs[:num_to_remove]:
                _agent_graphs.pop(graph_id, None)
                _graph_timestamps.pop(graph_id, None)
                evicted_counts["graphs"] += 1

        # Enforce workflow limit
        if len(_agent_workflows) > MAX_AGENT_WORKFLOWS:
            sorted_workflows = sorted(_workflow_timestamps.items(), key=lambda x: x[1])
            num_to_remove = len(_agent_workflows) - MAX_AGENT_WORKFLOWS
            for workflow_id, _ in sorted_workflows[:num_to_remove]:
                _agent_workflows.pop(workflow_id, None)
                _workflow_timestamps.pop(workflow_id, None)
                evicted_counts["workflows"] += 1

        return evicted_counts


def get_registry_stats() -> dict[str, Any]:
    """
    Get statistics about the current state of registries.

    Returns:
        Dictionary with registry statistics
    """
    with _cleanup_lock:
        return {
            "agents": {
                "count": len(_agents),
                "max_limit": MAX_AGENTS,
                "utilization": len(_agents) / MAX_AGENTS * 100,
            },
            "graphs": {
                "count": len(_agent_graphs),
                "max_limit": MAX_AGENT_GRAPHS,
                "utilization": len(_agent_graphs) / MAX_AGENT_GRAPHS * 100,
            },
            "workflows": {
                "count": len(_agent_workflows),
                "max_limit": MAX_AGENT_WORKFLOWS,
                "utilization": len(_agent_workflows) / MAX_AGENT_WORKFLOWS * 100,
            },
        }


@contextmanager
def temporary_agent_context() -> Generator[None, None, None]:
    """
    Context manager that automatically cleans up agents created within it.

    Example:
        with temporary_agent_context():
            agent = create_agent("temp_agent")
            # Agent will be automatically cleaned up when exiting context
    """
    agents_before = set(_agents.keys())
    graphs_before = set(_agent_graphs.keys())
    workflows_before = set(_agent_workflows.keys())

    try:
        yield
    finally:
        # Clean up any agents, graphs, or workflows created in this context
        with _cleanup_lock:
            agents_to_remove = set(_agents.keys()) - agents_before
            graphs_to_remove = set(_agent_graphs.keys()) - graphs_before
            workflows_to_remove = set(_agent_workflows.keys()) - workflows_before

            for agent_id in agents_to_remove:
                _agents.pop(agent_id, None)
                _agent_timestamps.pop(agent_id, None)

            for graph_id in graphs_to_remove:
                _agent_graphs.pop(graph_id, None)
                _graph_timestamps.pop(graph_id, None)

            for workflow_id in workflows_to_remove:
                _agent_workflows.pop(workflow_id, None)
                _workflow_timestamps.pop(workflow_id, None)


# Update the create functions to track timestamps and enforce limits
def _update_agent_create(
    original_create_func: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrapper to add timestamp tracking and size enforcement to create functions."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original_create_func(*args, **kwargs)

        # Track timestamp for TTL cleanup
        import time

        current_time = time.time()

        if hasattr(result, "agent_id"):
            _agent_timestamps[result.agent_id] = current_time
        elif hasattr(result, "graph_id"):
            _graph_timestamps[result.graph_id] = current_time
        elif hasattr(result, "workflow_id"):
            _workflow_timestamps[result.workflow_id] = current_time

        # Enforce size limits
        enforce_size_limits()

        return result

    return wrapper


# Apply wrappers to existing create functions
create_agent = _update_agent_create(create_agent)
create_agent_graph = _update_agent_create(create_agent_graph)
create_agent_workflow = _update_agent_create(create_agent_workflow)
