"""
Configuration for integration tests.

This file contains pytest fixtures and configuration specific to integration tests.
"""

import os

import pytest


def pytest_configure(config):
    """Configure pytest for integration tests."""
    # Register custom markers
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "llm: marks tests as LLM integration tests")
    config.addinivalue_line(
        "markers", "openai: marks tests as OpenAI integration tests"
    )
    config.addinivalue_line(
        "markers", "anthropic: marks tests as Anthropic integration tests"
    )
    config.addinivalue_line("markers", "agent: marks tests as agent integration tests")
    config.addinivalue_line("markers", "tool: marks tests as tool integration tests")
    config.addinivalue_line(
        "markers", "retrieval: marks tests as retrieval integration tests"
    )
    config.addinivalue_line(
        "markers", "workflow: marks tests as workflow integration tests"
    )
    config.addinivalue_line(
        "markers", "error_handling: marks tests as error handling integration tests"
    )
    config.addinivalue_line(
        "markers", "async_support: marks tests as async support integration tests"
    )
    config.addinivalue_line(
        "markers", "comprehensive: marks tests as comprehensive integration tests"
    )


@pytest.fixture(scope="session")
def integration_test_env():
    """Set up integration test environment."""
    # Ensure we have a clean test environment
    test_env = {
        "NOVEUM_API_KEY": "test-integration-key",
        "NOVEUM_PROJECT": "integration-test-project",
        "NOVEUM_ENVIRONMENT": "test",
    }

    # Store original env vars
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield test_env

    # Restore original env vars
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture(scope="function")
def clean_noveum_client():
    """Ensure clean Noveum client state for each test."""
    import noveum_trace

    # Clear any existing client
    noveum_trace._client = None

    yield

    # Clean up after test
    if hasattr(noveum_trace, "_client") and noveum_trace._client:
        try:
            noveum_trace._client.shutdown()
        except Exception:
            pass
        finally:
            noveum_trace._client = None


@pytest.fixture(scope="function")
def skip_on_missing_api_keys():
    """Skip tests if required API keys are missing."""
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = []

    for key in required_keys:
        if not os.environ.get(key):
            missing_keys.append(key)

    if missing_keys:
        pytest.skip(f"Missing required API keys: {', '.join(missing_keys)}")
