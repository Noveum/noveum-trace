"""
Unit tests for multi-agent functionality.
"""

import pytest
import threading
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from noveum_trace.agents import (
    Agent, AgentConfig, AgentRegistry, get_agent_registry,
    AgentContext, get_current_agent, set_current_agent,
    trace, observe, llm_trace, update_current_span
)
from noveum_trace.core.tracer import TracerConfig
from noveum_trace.types import CustomHeaders, SpanKind
from noveum_trace.utils.exceptions import ValidationError, ConfigurationError


class TestAgentConfig:
    """Test AgentConfig class."""
    
    def test_agent_config_creation(self):
        """Test basic agent config creation."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        assert config.name == "test-agent"
        assert config.agent_type == "llm"
        assert config.version == "1.0.0"
        assert config.max_concurrent_traces == 100
        assert config.enable_metrics is True
    
    def test_agent_config_with_custom_headers(self):
        """Test agent config with custom headers."""
        headers = CustomHeaders(
            project_id="test-project",
            org_id="test-org",
            additional_headers={"custom": "value"}
        )
        config = AgentConfig(
            name="test-agent",
            agent_type="llm",
            custom_headers=headers
        )
        assert config.custom_headers == headers
    
    def test_agent_config_validation_success(self):
        """Test successful agent config validation."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        config.validate()  # Should not raise
    
    def test_agent_config_validation_empty_name(self):
        """Test agent config validation with empty name."""
        config = AgentConfig(name="", agent_type="llm")
        with pytest.raises(ValidationError, match="Agent name must be a non-empty string"):
            config.validate()
    
    def test_agent_config_validation_invalid_sampling_rate(self):
        """Test agent config validation with invalid sampling rate."""
        config = AgentConfig(name="test-agent", agent_type="llm", sampling_rate=1.5)
        with pytest.raises(ValidationError, match="Sampling rate must be between 0.0 and 1.0"):
            config.validate()
    
    def test_agent_config_with_capabilities_and_tags(self):
        """Test agent config with capabilities and tags."""
        config = AgentConfig(
            name="test-agent",
            agent_type="llm",
            capabilities={"chat", "completion"},
            tags={"production", "v2"}
        )
        assert "chat" in config.capabilities
        assert "completion" in config.capabilities
        assert "production" in config.tags
        assert "v2" in config.tags


class TestAgent:
    """Test Agent class."""
    
    def test_agent_creation(self):
        """Test basic agent creation."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        agent = Agent(config)
        
        assert agent.name == "test-agent"
        assert agent.agent_type == "llm"
        assert agent.is_active is True
        assert agent.trace_count == 0
        assert agent.active_trace_count == 0
        assert agent.tracer is not None
    
    def test_agent_with_project_tracer_config(self):
        """Test agent creation with project tracer config."""
        project_config = TracerConfig(
            project_name="test-project",
            sampling_rate=0.5
        )
        agent_config = AgentConfig(name="test-agent", agent_type="llm")
        agent = Agent(agent_config, project_config)
        
        assert agent.tracer.config.project_name == "test-project"
        assert agent.tracer.config.sampling_rate == 0.5
    
    def test_agent_capabilities_management(self):
        """Test agent capabilities management."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        agent = Agent(config)
        
        # Add capabilities
        agent.add_capability("chat")
        agent.add_capability("completion")
        assert agent.has_capability("chat")
        assert agent.has_capability("completion")
        assert not agent.has_capability("embedding")
        
        # Remove capability
        agent.remove_capability("chat")
        assert not agent.has_capability("chat")
        assert agent.has_capability("completion")
    
    def test_agent_tags_management(self):
        """Test agent tags management."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        agent = Agent(config)
        
        # Add tags
        agent.add_tag("production")
        agent.add_tag("v2")
        assert agent.has_tag("production")
        assert agent.has_tag("v2")
        assert not agent.has_tag("development")
        
        # Remove tag
        agent.remove_tag("production")
        assert not agent.has_tag("production")
        assert agent.has_tag("v2")
    
    def test_agent_metadata_management(self):
        """Test agent metadata management."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        agent = Agent(config)
        
        # Set metadata
        agent.set_metadata("version", "1.2.3")
        agent.set_metadata("author", "test-user")
        
        assert agent.get_metadata("version") == "1.2.3"
        assert agent.get_metadata("author") == "test-user"
        assert agent.get_metadata("nonexistent") is None
        assert agent.get_metadata("nonexistent", "default") == "default"
    
    def test_agent_trace_registration(self):
        """Test agent trace registration."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        agent = Agent(config)
        
        # Register traces
        agent.register_trace("trace-1", {"data": "test1"})
        agent.register_trace("trace-2", {"data": "test2"})
        
        assert agent.active_trace_count == 2
        assert agent.trace_count == 2
        
        active_traces = agent.get_active_traces()
        assert "trace-1" in active_traces
        assert "trace-2" in active_traces
        
        # Unregister trace
        agent.unregister_trace("trace-1")
        assert agent.active_trace_count == 1
        assert agent.trace_count == 2  # Total count doesn't decrease
    
    def test_agent_deactivation(self):
        """Test agent deactivation."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        agent = Agent(config)
        
        assert agent.is_active is True
        
        agent.deactivate()
        assert agent.is_active is False
    
    def test_agent_to_dict(self):
        """Test agent serialization to dictionary."""
        config = AgentConfig(
            name="test-agent",
            agent_type="llm",
            description="Test agent",
            capabilities={"chat"},
            tags={"test"}
        )
        agent = Agent(config)
        
        agent_dict = agent.to_dict()
        
        assert agent_dict["name"] == "test-agent"
        assert agent_dict["agent_type"] == "llm"
        assert agent_dict["description"] == "Test agent"
        assert "chat" in agent_dict["capabilities"]
        assert "test" in agent_dict["tags"]
        assert agent_dict["is_active"] is True
        assert "agent_id" in agent_dict
        assert "created_at" in agent_dict


class TestAgentRegistry:
    """Test AgentRegistry class."""
    
    def setup_method(self):
        """Set up test method."""
        self.registry = AgentRegistry()
    
    def teardown_method(self):
        """Tear down test method."""
        self.registry.shutdown()
    
    def test_agent_registration(self):
        """Test agent registration."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        agent = self.registry.register_agent(config)
        
        assert agent.name == "test-agent"
        assert len(self.registry) == 1
        assert "test-agent" in self.registry
        
        # Get agent
        retrieved_agent = self.registry.get_agent("test-agent")
        assert retrieved_agent == agent
    
    def test_agent_registration_duplicate_name(self):
        """Test agent registration with duplicate name."""
        config1 = AgentConfig(name="test-agent", agent_type="llm")
        config2 = AgentConfig(name="test-agent", agent_type="chat")
        
        self.registry.register_agent(config1)
        
        # Should raise error for duplicate name
        with pytest.raises(ValidationError, match="Agent with name 'test-agent' already exists"):
            self.registry.register_agent(config2)
        
        # Should succeed with replace_existing=True
        agent2 = self.registry.register_agent(config2, replace_existing=True)
        assert agent2.agent_type == "chat"
    
    def test_agent_deregistration(self):
        """Test agent deregistration."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        self.registry.register_agent(config)
        
        assert len(self.registry) == 1
        
        success = self.registry.deregister_agent("test-agent")
        assert success is True
        assert len(self.registry) == 0
        assert "test-agent" not in self.registry
        
        # Deregistering non-existent agent should return False
        success = self.registry.deregister_agent("nonexistent")
        assert success is False
    
    def test_list_agents_filtering(self):
        """Test agent listing with filtering."""
        # Register multiple agents
        configs = [
            AgentConfig(name="llm-agent-1", agent_type="llm"),
            AgentConfig(name="llm-agent-2", agent_type="llm"),
            AgentConfig(name="chat-agent", agent_type="chat"),
        ]
        
        for config in configs:
            agent = self.registry.register_agent(config)
            if config.name == "llm-agent-1":
                agent.add_capability("completion")
                agent.add_tag("production")
        
        # Test filtering by type
        llm_agents = self.registry.list_agents(agent_type="llm")
        assert len(llm_agents) == 2
        
        chat_agents = self.registry.list_agents(agent_type="chat")
        assert len(chat_agents) == 1
        
        # Test filtering by capability
        completion_agents = self.registry.list_agents(has_capability="completion")
        assert len(completion_agents) == 1
        assert completion_agents[0].name == "llm-agent-1"
        
        # Test filtering by tag
        prod_agents = self.registry.list_agents(has_tag="production")
        assert len(prod_agents) == 1
        assert prod_agents[0].name == "llm-agent-1"
    
    def test_parent_child_relationships(self):
        """Test parent-child agent relationships."""
        parent_config = AgentConfig(name="parent-agent", agent_type="coordinator")
        child_config = AgentConfig(
            name="child-agent",
            agent_type="worker",
            parent_agent="parent-agent"
        )
        
        # Register parent first
        parent_agent = self.registry.register_agent(parent_config)
        child_agent = self.registry.register_agent(child_config)
        
        # Check relationships
        assert child_agent.config.parent_agent == "parent-agent"
        assert "child-agent" in parent_agent.config.child_agents
        
        # Test registry methods
        children = self.registry.get_child_agents("parent-agent")
        assert len(children) == 1
        assert children[0] == child_agent
        
        parent = self.registry.get_parent_agent("child-agent")
        assert parent == parent_agent
    
    def test_registry_stats(self):
        """Test registry statistics."""
        # Register agents
        configs = [
            AgentConfig(name="agent-1", agent_type="llm"),
            AgentConfig(name="agent-2", agent_type="llm"),
            AgentConfig(name="agent-3", agent_type="chat"),
        ]
        
        agents = []
        for config in configs:
            agent = self.registry.register_agent(config)
            agents.append(agent)
        
        # Deactivate one agent
        agents[0].deactivate()
        
        stats = self.registry.get_registry_stats()
        
        assert stats["total_agents"] == 3
        assert stats["active_agents"] == 2
        assert stats["inactive_agents"] == 1
        assert stats["agent_types"] == 2
        assert stats["agent_type_counts"]["llm"] == 2
        assert stats["agent_type_counts"]["chat"] == 1


class TestAgentContext:
    """Test agent context management."""
    
    def test_agent_context_manager(self):
        """Test agent context manager."""
        config = AgentConfig(name="test-agent", agent_type="llm")
        agent = Agent(config)
        
        # Initially no current agent
        assert get_current_agent() is None
        
        # Use context manager
        with AgentContext(agent):
            assert get_current_agent() == agent
        
        # Context should be restored
        assert get_current_agent() is None
    
    def test_nested_agent_contexts(self):
        """Test nested agent contexts."""
        config1 = AgentConfig(name="agent-1", agent_type="llm")
        config2 = AgentConfig(name="agent-2", agent_type="chat")
        agent1 = Agent(config1)
        agent2 = Agent(config2)
        
        with AgentContext(agent1):
            assert get_current_agent() == agent1
            
            with AgentContext(agent2):
                assert get_current_agent() == agent2
            
            # Should restore to agent1
            assert get_current_agent() == agent1
        
        # Should restore to None
        assert get_current_agent() is None
    
    @pytest.mark.asyncio
    async def test_async_agent_context(self):
        """Test async agent context manager."""
        from noveum_trace.agents.context import AsyncAgentContext
        
        config = AgentConfig(name="test-agent", agent_type="llm")
        agent = Agent(config)
        
        # Initially no current agent
        assert get_current_agent() is None
        
        # Use async context manager
        async with AsyncAgentContext(agent):
            assert get_current_agent() == agent
        
        # Context should be restored
        assert get_current_agent() is None


class TestAgentDecorators:
    """Test agent-aware decorators."""
    
    def setup_method(self):
        """Set up test method."""
        self.registry = get_agent_registry()
        self.config = AgentConfig(name="test-agent", agent_type="llm")
        self.agent = self.registry.register_agent(self.config)
    
    def teardown_method(self):
        """Tear down test method."""
        self.registry.shutdown()
    
    def test_trace_decorator_basic(self):
        """Test basic @trace decorator."""
        @trace
        def test_function():
            return "result"
        
        with AgentContext(self.agent):
            result = test_function()
        
        assert result == "result"
    
    def test_trace_decorator_with_args(self):
        """Test @trace decorator with arguments."""
        @trace(name="custom_operation", capture_args=True)
        def test_function(arg1, arg2="default"):
            return f"{arg1}-{arg2}"
        
        with AgentContext(self.agent):
            result = test_function("test", arg2="value")
        
        assert result == "test-value"
    
    @pytest.mark.asyncio
    async def test_trace_decorator_async(self):
        """Test @trace decorator with async function."""
        @trace
        async def async_test_function():
            await asyncio.sleep(0.01)
            return "async_result"
        
        async with AsyncAgentContext(self.agent):
            result = await async_test_function()
        
        assert result == "async_result"
    
    def test_observe_decorator_basic(self):
        """Test basic @observe decorator."""
        @observe
        def test_component():
            return "component_result"
        
        with AgentContext(self.agent):
            result = test_component()
        
        assert result == "component_result"
    
    def test_observe_decorator_with_metrics(self):
        """Test @observe decorator with metrics."""
        @observe(metrics=["latency", "accuracy"])
        def test_component():
            return "component_result"
        
        with AgentContext(self.agent):
            result = test_component()
        
        assert result == "component_result"
    
    def test_llm_trace_decorator_basic(self):
        """Test basic @llm_trace decorator."""
        @llm_trace(model="gpt-4", operation="chat")
        def test_llm_call():
            # Mock LLM response
            class MockResponse:
                def __init__(self):
                    self.choices = [MockChoice()]
                    self.usage = MockUsage()
                    self.model = "gpt-4"
                    self.id = "test-id"
            
            class MockChoice:
                def __init__(self):
                    self.message = MockMessage()
                    self.finish_reason = "stop"
            
            class MockMessage:
                def __init__(self):
                    self.content = "Test response"
            
            class MockUsage:
                def __init__(self):
                    self.prompt_tokens = 10
                    self.completion_tokens = 20
                    self.total_tokens = 30
            
            return MockResponse()
        
        with AgentContext(self.agent):
            result = test_llm_call()
        
        assert result.choices[0].message.content == "Test response"
    
    def test_update_current_span(self):
        """Test update_current_span function."""
        @trace
        def test_function():
            update_current_span(
                input="test_input",
                output="test_output",
                metadata={"model": "gpt-4"},
                custom_attr="custom_value"
            )
            return "result"
        
        with AgentContext(self.agent):
            result = test_function()
        
        assert result == "result"


class TestThreadSafety:
    """Test thread safety of multi-agent functionality."""
    
    def test_concurrent_agent_registration(self):
        """Test concurrent agent registration."""
        registry = AgentRegistry()
        results = []
        errors = []
        
        def register_agent(agent_id):
            try:
                config = AgentConfig(name=f"agent-{agent_id}", agent_type="llm")
                agent = registry.register_agent(config)
                results.append(agent)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_agent, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0
        assert len(results) == 10
        assert len(registry) == 10
        
        registry.shutdown()
    
    def test_concurrent_context_switching(self):
        """Test concurrent agent context switching."""
        registry = AgentRegistry()
        
        # Create agents
        agents = []
        for i in range(5):
            config = AgentConfig(name=f"agent-{i}", agent_type="llm")
            agent = registry.register_agent(config)
            agents.append(agent)
        
        results = []
        
        def context_test(agent):
            with AgentContext(agent):
                current = get_current_agent()
                results.append(current.name if current else None)
        
        # Create multiple threads with different agents
        threads = []
        for agent in agents:
            thread = threading.Thread(target=context_test, args=(agent,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results - each thread should see its own agent
        assert len(results) == 5
        expected_names = {f"agent-{i}" for i in range(5)}
        actual_names = set(results)
        assert actual_names == expected_names
        
        registry.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])

