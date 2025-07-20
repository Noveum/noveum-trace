# Integration Tests

This directory contains comprehensive integration tests for the Noveum Trace SDK. Tests are organized into two main categories based on their approach to external dependencies.

## Directory Structure

### `mock_endpoint/` - Mock Endpoint Tests

Tests that mock HTTP endpoints and don't make real API calls to external services. These tests use various mocking strategies to simulate endpoint behavior and validate trace data export.

**Test categories:**

- Base configuration and endpoint switching
- Decorator functionality with mocked backends
- HTTP transport and URL construction
- OpenTelemetry integration with mocked collectors
- Endpoint validation and error handling

### `end_to_end/` - End-to-End Tests

Tests that make real API calls to external services like LLM providers (OpenAI, Anthropic) while still mocking the Noveum endpoint for trace capture. These tests validate complete workflows with actual external dependencies.

**Test categories:**

- Real LLM API integration scenarios
- Multi-turn conversation workflows
- Function calling and tool usage
- Agent orchestration with real services
- Streaming response handling

## Quick Start

### 1. Run Integration Tests

```bash
# Run all integration tests with localhost endpoint
pytest tests/integration/

# Run only mock endpoint tests (no external API calls)
pytest tests/integration/mock_endpoint/

# Run only end-to-end tests (requires API keys)
pytest tests/integration/end_to_end/

# Run with verbose output to see test details
pytest -v -s tests/integration/

# Run specific test categories
pytest -m integration tests/integration/
pytest -m llm tests/integration/
pytest -m agent tests/integration/
```

## Test Categories

### Mock Endpoint Tests (`mock_endpoint/`)

#### Base Configuration Tests (`test_base_configuration.py`)

Tests core configuration functionality with different endpoints:

```bash
# Test endpoint switching
pytest tests/integration/mock_endpoint/test_base_configuration.py::TestBaseConfiguration::test_endpoint_switching

# Test health checks
pytest tests/integration/mock_endpoint/test_base_configuration.py::TestBaseConfiguration::test_health_check_localhost
```

**Key scenarios:**

- Localhost vs production endpoint configuration
- Endpoint validation and error handling
- Health check functionality
- Configuration persistence across operations

#### Decorator Integration Tests (`test_decorator_integrations.py`)

Tests all decorator types with realistic usage patterns:

```bash
# Test @trace decorator
pytest tests/integration/mock_endpoint/test_decorator_integrations.py::TestTraceDecoratorIntegration

# Test @trace_llm decorator
pytest tests/integration/mock_endpoint/test_decorator_integrations.py::TestLLMDecoratorIntegration

# Test @trace_agent decorator
pytest tests/integration/mock_endpoint/test_decorator_integrations.py::TestAgentDecoratorIntegration

# Test @trace_tool decorator
pytest tests/integration/mock_endpoint/test_decorator_integrations.py::TestToolDecoratorIntegration

# Test decorator composition
pytest tests/integration/mock_endpoint/test_decorator_integrations.py::TestDecoratorComposition
```

**Key scenarios:**

- Individual decorator functionality
- Decorator composition and nesting
- Error handling across decorators
- Attribute and metadata capture

#### OpenTelemetry Integration Tests (`test_opentelemetry_integration.py`)

Tests OpenTelemetry auto-instrumentation and integration:

```bash
# Test OTel auto-instrumentation
pytest tests/integration/mock_endpoint/test_opentelemetry_integration.py::TestOTelAutoInstrumentation

# Test framework integration
pytest tests/integration/mock_endpoint/test_opentelemetry_integration.py::TestFrameworkIntegration

# Test dual export
pytest tests/integration/mock_endpoint/test_opentelemetry_integration.py::TestDualExport
```

**Key scenarios:**

- Auto-instrumentation of popular frameworks
- Custom instrumentation patterns
- Dual export (OTel collector + Noveum endpoint)
- High-throughput and sampling scenarios

### End-to-End Tests (`end_to_end/`)

#### Real LLM Scenarios (`test_real_llm_scenarios.py`)

Tests with actual LLM provider APIs:

```bash
# Test OpenAI integration (requires OPENAI_API_KEY)
pytest tests/integration/end_to_end/test_real_llm_scenarios.py::TestRealLLMScenarios::test_openai_chat_completion

# Test Anthropic integration (requires ANTHROPIC_API_KEY)
pytest tests/integration/end_to_end/test_real_llm_scenarios.py::TestRealLLMScenarios::test_anthropic_chat_completion

# Test function calling scenarios
pytest tests/integration/end_to_end/test_real_llm_scenarios.py::TestRealLLMScenarios::test_function_calling_scenario

# Test multi-agent workflows
pytest tests/integration/end_to_end/test_real_llm_scenarios.py::TestRealLLMScenarios::test_multi_agent_research_workflow
```

**Key scenarios:**

- Real OpenAI and Anthropic API calls
- Function calling and tool usage
- Multi-turn conversations with memory
- Complex agent orchestration workflows
- Concurrent agent processing

## Endpoint Configuration

### Environment Variables

Configure test endpoints using environment variables:

```bash
# Development (default)
export NOVEUM_TEST_ENDPOINT="http://localhost:3000"

# Production
export NOVEUM_TEST_ENDPOINT="https://api.noveum.ai"

# Custom/Staging
export NOVEUM_TEST_ENDPOINT="https://staging.noveum.ai"

# Required for end-to-end tests
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
```

### Running Tests with Different Endpoints

```bash
# Test with localhost (mock server) - mock endpoint tests
NOVEUM_TEST_ENDPOINT="http://localhost:3000" pytest tests/integration/mock_endpoint/

# Test with production endpoint - mock endpoint tests
NOVEUM_TEST_ENDPOINT="https://api.noveum.ai" pytest tests/integration/mock_endpoint/test_base_configuration.py

# Test with real LLM APIs - end-to-end tests
OPENAI_API_KEY="sk-..." ANTHROPIC_API_KEY="sk-ant-..." pytest tests/integration/end_to_end/

# Test specific endpoint switching
NOVEUM_TEST_ENDPOINT="https://custom.endpoint.com" pytest tests/integration/mock_endpoint/test_base_configuration.py::TestBaseConfiguration::test_custom_endpoint_configuration
```

## Test Configuration

Integration tests use the endpoint configured via environment variables or test fixtures. Tests can run against:

- Local development servers
- Staging environments
- Production endpoints (with appropriate credentials)

### Inspecting Test Data

```bash
# View server statistics
curl http://localhost:3000/api/v1/stats

# View received traces
curl http://localhost:3000/api/v1/traces

# Clear data between test runs
curl -X POST http://localhost:3000/api/v1/clear
```

## Test Markers

Use pytest markers to run specific test categories:

```bash
# Integration tests
pytest -m integration

# LLM-specific tests
pytest -m llm

# Agent workflow tests
pytest -m agent

# OpenTelemetry tests
pytest -m opentelemetry

# Async support tests
pytest -m async_support

# Comprehensive workflow tests
pytest -m comprehensive
```

## Writing New Integration Tests

### 1. Choose the Right Directory

**Use `mock_endpoint/` when:**

- Testing SDK functionality without external dependencies
- Validating trace data structure and export
- Testing configuration and error handling
- Performance and load testing

**Use `end_to_end/` when:**

- Testing real integration with LLM providers
- Validating complete workflows with external services
- Testing streaming responses and real-time scenarios
- Demonstrating production-like usage patterns

### 2. Use Configurable Endpoints

```python
import os
import pytest
import noveum_trace

def test_my_feature(clean_noveum_client):
    endpoint = os.environ.get('NOVEUM_TEST_ENDPOINT', 'http://localhost:3000')
    noveum_trace.init(
        project="test-project",
        api_key="test-key",
        endpoint=endpoint
    )
    # Your test logic here
```

### 3. Add Appropriate Markers

```python
@pytest.mark.integration
@pytest.mark.llm
def test_llm_integration():
    # Test LLM integration scenarios
    pass

@pytest.mark.integration
@pytest.mark.agent
def test_agent_workflow():
    # Test agent workflow scenarios
    pass
```

### 4. Test Real Scenarios

```python
def test_realistic_conversation():
    # Use actual message structures
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]

    # Test actual conversation patterns
    # Include error handling and edge cases
    # Verify trace data structure and content
```

### 5. Validate Endpoint Usage

```python
def test_endpoint_validation(endpoint_capture):
    # Your test logic

    # Verify correct endpoint was used
    requests = endpoint_capture.get_requests('/api/v1/trace')
    assert len(requests) >= 1

    # Verify trace data structure
    trace_data = requests[0]['json_data']
    assert 'trace_id' in trace_data
    assert 'project' in trace_data
```

## API Key Requirements

### Mock Endpoint Tests

- **No API keys required** - these tests mock all external endpoints
- Can run completely offline
- Fast execution, suitable for CI/CD

### End-to-End Tests

- **Require valid API keys** for external services
- Set these environment variables:
  - `OPENAI_API_KEY` - For OpenAI tests
  - `ANTHROPIC_API_KEY` - For Anthropic tests
- Tests will be skipped if API keys are missing
- May incur small API costs

## Troubleshooting

### Common Issues

1. **Mock server not running** (for mock_endpoint tests)

   ```bash
   # Ensure endpoint is accessible
   curl -X GET $NOVEUM_TEST_ENDPOINT/health
   ```

2. **Missing API keys** (for end_to_end tests)

   ```bash
   # Set required API keys
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

3. **Import errors after reorganization**

   ```bash
   # Clear Python cache
   find . -name "*.pyc" -delete
   find . -name "__pycache__" -type d -exec rm -rf {} +
   ```

4. **Port conflicts**
   ```bash
   # Check if endpoint is accessible
   curl -X GET $NOVEUM_TEST_ENDPOINT/health
   # Use different endpoint if needed
   NOVEUM_TEST_ENDPOINT="http://localhost:3001" pytest tests/integration/
   ```

### Running Specific Test Categories

```bash
# Only mock endpoint tests (fast, no external dependencies)
pytest tests/integration/mock_endpoint/ -v

# Only end-to-end tests (requires API keys)
pytest tests/integration/end_to_end/ -v

# Skip end-to-end tests if API keys missing
pytest tests/integration/ -k "not test_real_llm_scenarios"

# Run with coverage for mock endpoint tests
pytest tests/integration/mock_endpoint/ --cov=noveum_trace --cov-report=html
```
