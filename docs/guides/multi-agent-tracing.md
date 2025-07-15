# Multi-Agent Tracing Guide

The Noveum Trace SDK provides comprehensive support for multi-agent systems, enabling you to trace, monitor, and analyze complex AI applications with multiple specialized agents working together.

## Overview

Multi-agent systems present unique observability challenges that traditional single-service tracing cannot adequately address. The Noveum Trace SDK's multi-agent capabilities provide:

- **Agent Registry**: Centralized management of multiple agents
- **Agent-Scoped Contexts**: Proper isolation and correlation between agents
- **Simplified Decorators**: Easy-to-use decorators similar to DeepEval, Phoenix, and Braintrust
- **Cross-Agent Correlation**: Track workflows that span multiple agents
- **Hierarchical Relationships**: Support for parent-child agent structures

## Quick Start

### 1. Basic Multi-Agent Setup

```python
import noveum_trace
from noveum_trace import Agent, AgentConfig, AgentContext

# Initialize the SDK
noveum_trace.init(project_name="my-multi-agent-app")

# Create agent configurations
coordinator_config = AgentConfig(
    name="coordinator",
    agent_type="coordinator",
    description="Coordinates workflow between agents"
)

worker_config = AgentConfig(
    name="worker",
    agent_type="worker", 
    description="Processes individual tasks",
    parent_agent="coordinator"
)

# Register agents
registry = noveum_trace.get_agent_registry()
coordinator = registry.register_agent(coordinator_config)
worker = registry.register_agent(worker_config)
```

### 2. Using Simplified Decorators

```python
from noveum_trace import trace, observe, llm_trace

# Simple function tracing
@trace
def process_data(data):
    return transform(data)

# Component-level observability (DeepEval-style)
@observe(metrics=["accuracy", "latency"])
def llm_component(prompt):
    response = call_llm(prompt)
    update_current_span(
        input=prompt,
        output=response,
        metadata={"model": "gpt-4"}
    )
    return response

# LLM operation tracing
@llm_trace(model="gpt-4", operation="chat")
async def chat_completion(messages):
    return await openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
```

### 3. Agent-Aware Execution

```python
# Execute with specific agent context
with AgentContext(coordinator):
    result = coordinate_workflow(task)
    
    # Switch to worker context
    with AgentContext(worker):
        processed = process_task(subtask)
```

## Agent Management

### Agent Configuration

The `AgentConfig` class provides comprehensive configuration options for agents:

```python
from noveum_trace import AgentConfig
from noveum_trace.types import CustomHeaders

config = AgentConfig(
    name="my-agent",
    agent_type="llm_processor",
    description="Processes LLM requests",
    version="1.2.0",
    
    # Capabilities and metadata
    capabilities={"chat", "completion", "embedding"},
    tags={"production", "v2"},
    metadata={"team": "ai-platform", "owner": "john@company.com"},
    
    # Tracing configuration
    custom_headers=CustomHeaders(
        project_id="my-project",
        org_id="my-org",
        additional_headers={"environment": "production"}
    ),
    sampling_rate=0.8,
    capture_llm_content=True,
    
    # Agent-specific settings
    max_concurrent_traces=200,
    trace_retention_hours=48,
    enable_metrics=True,
    enable_evaluation=True,
    
    # Relationships
    parent_agent="coordinator",
    child_agents={"worker-1", "worker-2"}
)
```

### Agent Registry Operations

The `AgentRegistry` provides centralized agent management:

```python
from noveum_trace import get_agent_registry

registry = get_agent_registry()

# Register agents
agent = registry.register_agent(config)

# Query agents
all_agents = registry.list_agents()
llm_agents = registry.list_agents(agent_type="llm_processor")
prod_agents = registry.list_agents(has_tag="production")

# Agent relationships
children = registry.get_child_agents("coordinator")
parent = registry.get_parent_agent("worker-1")

# Registry statistics
stats = registry.get_registry_stats()
print(f"Total agents: {stats['total_agents']}")
print(f"Active traces: {stats['active_traces']}")
```

## Simplified Decorators

### @trace Decorator

The `@trace` decorator provides the simplest interface for function tracing:

```python
from noveum_trace import trace, SpanKind

# Basic usage
@trace
def simple_function():
    return "result"

# Advanced configuration
@trace(
    name="custom_operation",
    kind=SpanKind.CLIENT,
    attributes={"service": "external-api"},
    capture_args=True,
    capture_result=True,
    agent="specific-agent"  # Use specific agent
)
def advanced_function(arg1, arg2="default"):
    return f"{arg1}-{arg2}"

# Async support
@trace
async def async_function():
    await asyncio.sleep(0.1)
    return "async_result"
```

### @observe Decorator (DeepEval-inspired)

The `@observe` decorator focuses on component-level observability:

```python
from noveum_trace import observe, update_current_span

@observe(metrics=["accuracy", "latency", "throughput"])
def data_processor(data):
    # Process data
    result = transform(data)
    
    # Update span with component information
    update_current_span(
        input=data,
        output=result,
        metadata={
            "processing_time_ms": 150,
            "accuracy_score": 0.95,
            "items_processed": len(data)
        }
    )
    
    return result

@observe(
    name="llm_component",
    metrics=["response_quality", "token_efficiency"],
    capture_input=True,
    capture_output=True
)
async def llm_component(prompt, model="gpt-4"):
    response = await call_llm(prompt, model)
    
    update_current_span(
        metadata={
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "quality_score": 0.88
        }
    )
    
    return response
```

### @llm_trace Decorator

The `@llm_trace` decorator specializes in LLM operation tracing:

```python
from noveum_trace import llm_trace

@llm_trace(model="gpt-4", operation="chat", ai_system="openai")
def chat_completion(messages, temperature=0.7):
    return openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature
    )

@llm_trace(
    name="custom_llm_operation",
    model="claude-3",
    operation="completion",
    ai_system="anthropic",
    capture_content=True
)
async def anthropic_completion(prompt, max_tokens=100):
    return await anthropic_client.completions.create(
        model="claude-3",
        prompt=prompt,
        max_tokens=max_tokens
    )

# Automatic parameter extraction
@llm_trace(operation="embedding")
def create_embedding(text, model="text-embedding-ada-002"):
    # Model and operation type are automatically extracted
    return openai_client.embeddings.create(
        model=model,
        input=text
    )
```

### Dynamic Span Updates

Use `update_current_span` to add information to the current span:

```python
from noveum_trace import update_current_span

@trace
def complex_operation(data):
    # Add input information
    update_current_span(
        input=data,
        metadata={"operation_type": "batch_processing"}
    )
    
    # Process data
    result = process(data)
    
    # Add output and metrics
    update_current_span(
        output=result,
        metadata={
            "processing_time_ms": 250,
            "items_processed": len(data),
            "success_rate": 0.98
        },
        custom_metric=42
    )
    
    return result
```

## Multi-Agent Workflows

### Simple Multi-Agent Workflow

```python
from noveum_trace import AgentContext, trace

@trace(name="coordinate_workflow")
def coordinate_workflow(task):
    # Coordinator processes the task
    subtasks = split_task(task)
    results = []
    
    # Process with different agents
    for subtask in subtasks:
        if subtask.type == "analysis":
            with AgentContext(analyzer_agent):
                result = analyze_data(subtask.data)
        elif subtask.type == "generation":
            with AgentContext(generator_agent):
                result = generate_content(subtask.prompt)
        
        results.append(result)
    
    return combine_results(results)
```

### Async Multi-Agent Workflow

```python
from noveum_trace.agents.context import AsyncAgentContext

@trace(name="async_multi_agent_workflow")
async def async_workflow(tasks):
    results = []
    
    # Process tasks concurrently with different agents
    async with AsyncAgentContext(orchestrator_agent):
        concurrent_tasks = []
        
        for task in tasks:
            if task.requires_llm:
                async with AsyncAgentContext(llm_agent):
                    concurrent_tasks.append(process_llm_task(task))
            else:
                async with AsyncAgentContext(data_agent):
                    concurrent_tasks.append(process_data_task(task))
        
        results = await asyncio.gather(*concurrent_tasks)
    
    return results
```

### Hierarchical Agent Workflow

```python
@trace(name="hierarchical_workflow")
def hierarchical_workflow(project):
    # Manager level
    with AgentContext(manager_agent):
        project_plan = create_project_plan(project)
        
        # Team lead level
        with AgentContext(team_lead_agent):
            tasks = break_down_project(project_plan)
            
            # Developer level
            results = []
            for task in tasks:
                with AgentContext(developer_agent):
                    result = implement_task(task)
                    results.append(result)
            
            integration_result = integrate_results(results)
        
        final_result = finalize_project(integration_result)
    
    return final_result
```

## Agent Communication

### Message-Based Communication

```python
@trace(name="send_message")
def send_message(message, target_agent):
    current_agent = get_current_agent()
    
    message_data = {
        "content": message,
        "sender": current_agent.name,
        "target": target_agent,
        "timestamp": time.time()
    }
    
    # Send through message queue
    message_queue.put(message_data)
    
    update_current_span(
        output=message_data,
        metadata={"communication_type": "async_message"}
    )
    
    return message_data

@trace(name="receive_messages")
def receive_messages():
    current_agent = get_current_agent()
    messages = message_queue.get_messages_for(current_agent.name)
    
    update_current_span(
        output=messages,
        metadata={"messages_received": len(messages)}
    )
    
    return messages
```

### Context Propagation

```python
@trace(name="cross_agent_operation")
def cross_agent_operation(data):
    # Process with first agent
    with AgentContext(agent1):
        intermediate = process_step1(data)
        
        # Pass context to second agent
        trace_context = get_current_trace()
        
    # Continue with second agent (context is propagated)
    with AgentContext(agent2):
        # Context is automatically available
        final_result = process_step2(intermediate)
    
    return final_result
```

## Best Practices

### 1. Agent Organization

- Use descriptive agent names and types
- Organize agents by capability or domain
- Establish clear parent-child relationships
- Use tags for grouping and filtering

### 2. Decorator Usage

- Use `@trace` for simple function tracing
- Use `@observe` for component-level monitoring
- Use `@llm_trace` for LLM operations
- Combine decorators for comprehensive coverage

### 3. Context Management

- Always use `AgentContext` for agent-specific operations
- Keep agent contexts as narrow as possible
- Use async contexts for async operations
- Properly handle context switching

### 4. Performance Optimization

- Configure appropriate sampling rates per agent
- Use batching for high-throughput scenarios
- Monitor agent resource usage
- Implement proper cleanup procedures

### 5. Monitoring and Debugging

- Use registry statistics for system health
- Monitor cross-agent communication patterns
- Track agent performance metrics
- Implement proper error handling

## Advanced Features

### Custom Agent Types

```python
class CustomAgent(Agent):
    def __init__(self, config, custom_param):
        super().__init__(config)
        self.custom_param = custom_param
    
    def custom_method(self):
        # Custom agent behavior
        pass

# Register custom agent
custom_config = AgentConfig(name="custom", agent_type="custom")
custom_agent = CustomAgent(custom_config, custom_param="value")
registry.register_agent(custom_agent)
```

### Event Handlers

```python
def on_agent_registered(agent):
    print(f"Agent {agent.name} registered")

def on_agent_deregistered(agent):
    print(f"Agent {agent.name} deregistered")

# Add event handlers
registry.add_event_handler('agent_registered', on_agent_registered)
registry.add_event_handler('agent_deregistered', on_agent_deregistered)
```

### Agent Metrics

```python
@observe(metrics=["custom_metric"])
def monitored_function():
    # Function with custom metrics
    update_current_span(
        metadata={
            "custom_metric": calculate_metric(),
            "performance_score": 0.95
        }
    )
```

## Troubleshooting

### Common Issues

1. **Agent Not Found**: Ensure agent is registered before use
2. **Context Not Available**: Use proper `AgentContext` wrapper
3. **Memory Leaks**: Properly shutdown agents and registry
4. **Performance Issues**: Adjust sampling rates and batch sizes

### Debugging Tips

- Use console sink for real-time trace viewing
- Check registry statistics regularly
- Monitor agent activity timestamps
- Validate agent configurations

### Error Handling

```python
try:
    with AgentContext(agent):
        result = risky_operation()
except Exception as e:
    # Error is automatically recorded in span
    logger.error(f"Operation failed: {e}")
    raise
```

## Migration Guide

### From Single-Agent to Multi-Agent

1. **Identify Components**: Break down application into logical agents
2. **Create Agent Configs**: Define agent configurations
3. **Update Decorators**: Replace old decorators with new simplified ones
4. **Add Context Management**: Wrap operations with `AgentContext`
5. **Test Thoroughly**: Verify trace correlation and performance

### From Other SDKs

#### From DeepEval

```python
# DeepEval
@observe(metrics=[correctness])
def component():
    update_current_span(test_case=LLMTestCase(...))

# Noveum Trace
@observe(metrics=["correctness"])
def component():
    update_current_span(
        input=input_data,
        output=output_data,
        metadata={"correctness_score": 0.95}
    )
```

#### From Phoenix/Braintrust

```python
# Phoenix
@tracer.start_as_current_span("operation")
def operation():
    pass

# Noveum Trace
@trace(name="operation")
def operation():
    pass
```

## Examples

See the `examples/` directory for complete working examples:

- `multi_agent_chatbot.py`: Multi-agent chatbot system
- `simple_decorators_demo.py`: Decorator patterns demonstration
- `hierarchical_agents.py`: Hierarchical agent structures
- `async_workflows.py`: Async multi-agent workflows

## API Reference

For detailed API documentation, see:

- [Agent API Reference](../api/agents.md)
- [Decorator API Reference](../api/decorators.md)
- [Registry API Reference](../api/registry.md)
- [Context API Reference](../api/context.md)

