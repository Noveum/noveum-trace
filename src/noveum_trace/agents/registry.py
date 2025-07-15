"""
Agent registry for managing multiple agents in a project.
"""

import threading
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime, timezone
import logging

from .agent import Agent, AgentConfig
from ..core.tracer import TracerConfig
from ..utils.exceptions import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for managing multiple agents within a project."""
    
    def __init__(self, project_tracer_config: Optional[TracerConfig] = None):
        """Initialize the agent registry."""
        self._agents: Dict[str, Agent] = {}
        self._project_tracer_config = project_tracer_config
        self._lock = threading.RLock()
        self._event_handlers: Dict[str, List[Callable]] = {
            'agent_registered': [],
            'agent_deregistered': [],
            'agent_activated': [],
            'agent_deactivated': [],
        }
        
        logger.info("AgentRegistry initialized")
    
    def register_agent(
        self,
        config: AgentConfig,
        replace_existing: bool = False
    ) -> Agent:
        """Register a new agent with the registry."""
        with self._lock:
            if config.name in self._agents:
                if not replace_existing:
                    raise ValidationError(
                        f"Agent with name '{config.name}' already exists"
                    )
                else:
                    # Deactivate existing agent
                    self._agents[config.name].deactivate()
                    logger.info(f"Replacing existing agent '{config.name}'")
            
            # Create new agent
            agent = Agent(config, self._project_tracer_config)
            self._agents[config.name] = agent
            
            # Handle parent-child relationships
            if config.parent_agent:
                parent = self._agents.get(config.parent_agent)
                if parent:
                    parent.config.child_agents.add(config.name)
                else:
                    logger.warning(
                        f"Parent agent '{config.parent_agent}' not found for agent '{config.name}'"
                    )
            
            # Add this agent as child to specified children
            for child_name in config.child_agents:
                child = self._agents.get(child_name)
                if child:
                    child.config.parent_agent = config.name
                else:
                    logger.warning(
                        f"Child agent '{child_name}' not found for agent '{config.name}'"
                    )
            
            logger.info(f"Agent '{config.name}' registered successfully")
            self._emit_event('agent_registered', agent)
            
            return agent
    
    def deregister_agent(self, name: str) -> bool:
        """Deregister an agent from the registry."""
        with self._lock:
            agent = self._agents.get(name)
            if not agent:
                logger.warning(f"Agent '{name}' not found for deregistration")
                return False
            
            # Handle parent-child relationships
            if agent.config.parent_agent:
                parent = self._agents.get(agent.config.parent_agent)
                if parent:
                    parent.config.child_agents.discard(name)
            
            # Update child agents
            for child_name in agent.config.child_agents:
                child = self._agents.get(child_name)
                if child:
                    child.config.parent_agent = None
            
            # Deactivate and remove agent
            agent.deactivate()
            del self._agents[name]
            
            logger.info(f"Agent '{name}' deregistered successfully")
            self._emit_event('agent_deregistered', agent)
            
            return True
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        with self._lock:
            return self._agents.get(name)
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        with self._lock:
            for agent in self._agents.values():
                if agent.agent_id == agent_id:
                    return agent
            return None
    
    def list_agents(
        self,
        agent_type: Optional[str] = None,
        active_only: bool = True,
        has_capability: Optional[str] = None,
        has_tag: Optional[str] = None
    ) -> List[Agent]:
        """List agents with optional filtering."""
        with self._lock:
            agents = list(self._agents.values())
            
            # Apply filters
            if active_only:
                agents = [a for a in agents if a.is_active]
            
            if agent_type:
                agents = [a for a in agents if a.agent_type == agent_type]
            
            if has_capability:
                agents = [a for a in agents if a.has_capability(has_capability)]
            
            if has_tag:
                agents = [a for a in agents if a.has_tag(has_tag)]
            
            return agents
    
    def get_agent_names(self, active_only: bool = True) -> List[str]:
        """Get list of agent names."""
        agents = self.list_agents(active_only=active_only)
        return [agent.name for agent in agents]
    
    def get_agent_types(self, active_only: bool = True) -> Set[str]:
        """Get set of unique agent types."""
        agents = self.list_agents(active_only=active_only)
        return {agent.agent_type for agent in agents}
    
    def get_agents_by_type(self, agent_type: str, active_only: bool = True) -> List[Agent]:
        """Get all agents of a specific type."""
        return self.list_agents(agent_type=agent_type, active_only=active_only)
    
    def get_child_agents(self, parent_name: str) -> List[Agent]:
        """Get all child agents of a parent agent."""
        parent = self.get_agent(parent_name)
        if not parent:
            return []
        
        children = []
        for child_name in parent.config.child_agents:
            child = self.get_agent(child_name)
            if child:
                children.append(child)
        
        return children
    
    def get_parent_agent(self, child_name: str) -> Optional[Agent]:
        """Get the parent agent of a child agent."""
        child = self.get_agent(child_name)
        if not child or not child.config.parent_agent:
            return None
        
        return self.get_agent(child.config.parent_agent)
    
    def activate_agent(self, name: str) -> bool:
        """Activate a deactivated agent."""
        with self._lock:
            agent = self._agents.get(name)
            if not agent:
                logger.warning(f"Agent '{name}' not found for activation")
                return False
            
            if agent.is_active:
                logger.debug(f"Agent '{name}' is already active")
                return True
            
            # Reactivate agent (would need to recreate tracer)
            # For now, just mark as active
            agent._is_active = True
            agent.update_activity()
            
            logger.info(f"Agent '{name}' activated")
            self._emit_event('agent_activated', agent)
            
            return True
    
    def deactivate_agent(self, name: str) -> bool:
        """Deactivate an agent without removing it."""
        with self._lock:
            agent = self._agents.get(name)
            if not agent:
                logger.warning(f"Agent '{name}' not found for deactivation")
                return False
            
            if not agent.is_active:
                logger.debug(f"Agent '{name}' is already inactive")
                return True
            
            agent.deactivate()
            
            logger.info(f"Agent '{name}' deactivated")
            self._emit_event('agent_deactivated', agent)
            
            return True
    
    def clear_inactive_agents(self) -> int:
        """Remove all inactive agents from the registry."""
        with self._lock:
            inactive_agents = [
                name for name, agent in self._agents.items()
                if not agent.is_active
            ]
            
            for name in inactive_agents:
                del self._agents[name]
            
            count = len(inactive_agents)
            if count > 0:
                logger.info(f"Cleared {count} inactive agents")
            
            return count
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent registry."""
        with self._lock:
            agents = list(self._agents.values())
            active_agents = [a for a in agents if a.is_active]
            
            stats = {
                "total_agents": len(agents),
                "active_agents": len(active_agents),
                "inactive_agents": len(agents) - len(active_agents),
                "agent_types": len(self.get_agent_types()),
                "total_traces": sum(a.trace_count for a in agents),
                "active_traces": sum(a.active_trace_count for a in active_agents),
            }
            
            # Agent type breakdown
            type_counts = {}
            for agent in agents:
                type_counts[agent.agent_type] = type_counts.get(agent.agent_type, 0) + 1
            stats["agent_type_counts"] = type_counts
            
            return stats
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler for registry events."""
        if event_type not in self._event_handlers:
            raise ValidationError(f"Unknown event type: {event_type}")
        
        self._event_handlers[event_type].append(handler)
        logger.debug(f"Added event handler for '{event_type}'")
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> bool:
        """Remove an event handler."""
        if event_type not in self._event_handlers:
            return False
        
        try:
            self._event_handlers[event_type].remove(handler)
            logger.debug(f"Removed event handler for '{event_type}'")
            return True
        except ValueError:
            return False
    
    def shutdown(self) -> None:
        """Shutdown the registry and all agents."""
        with self._lock:
            logger.info("Shutting down AgentRegistry")
            
            # Deactivate all agents
            for agent in self._agents.values():
                try:
                    agent.deactivate()
                except Exception as e:
                    logger.error(f"Error deactivating agent '{agent.name}': {e}")
            
            # Clear registry
            self._agents.clear()
            self._event_handlers.clear()
            
            logger.info("AgentRegistry shutdown complete")
    
    def _emit_event(self, event_type: str, agent: Agent) -> None:
        """Emit an event to registered handlers."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(agent)
            except Exception as e:
                logger.error(f"Error in event handler for '{event_type}': {e}")
    
    def __len__(self) -> int:
        """Get number of registered agents."""
        return len(self._agents)
    
    def __contains__(self, name: str) -> bool:
        """Check if agent is registered."""
        return name in self._agents
    
    def __iter__(self):
        """Iterate over registered agents."""
        return iter(self._agents.values())


# Global registry instance
_global_registry: Optional[AgentRegistry] = None
_registry_lock = threading.Lock()


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _global_registry
    
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = AgentRegistry()
    
    return _global_registry


def set_agent_registry(registry: AgentRegistry) -> None:
    """Set the global agent registry instance."""
    global _global_registry
    
    with _registry_lock:
        if _global_registry is not None:
            _global_registry.shutdown()
        _global_registry = registry


def create_agent_registry(project_tracer_config: Optional[TracerConfig] = None) -> AgentRegistry:
    """Create a new agent registry with optional project configuration."""
    return AgentRegistry(project_tracer_config)

