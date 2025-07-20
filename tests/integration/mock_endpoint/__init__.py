"""
Mock endpoint integration tests.

This package contains integration tests that mock HTTP endpoints and don't make
real API calls to external services. These tests use various mocking strategies
to simulate endpoint behavior and validate trace data export.

Test categories:
- Base configuration and endpoint switching
- Decorator functionality with mocked backends
- HTTP transport and URL construction
- OpenTelemetry integration with mocked collectors
- Endpoint validation and error handling
"""
