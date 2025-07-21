"""
End-to-end integration tests.

This package contains integration tests that make real API calls to external
services like LLM providers (OpenAI, Anthropic) while still mocking the Noveum
endpoint for trace capture. These tests validate complete workflows with actual
external dependencies.

Test categories:
- Real LLM API integration scenarios
- Multi-turn conversation workflows
- Function calling and tool usage
- Agent orchestration with real services
- Streaming response handling

Note: These tests require valid API keys in environment variables:
- OPENAI_API_KEY for OpenAI tests
- ANTHROPIC_API_KEY for Anthropic tests
"""
