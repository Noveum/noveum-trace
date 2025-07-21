# Contributing to Noveum Trace SDK

Thank you for your interest in contributing to the Noveum Trace SDK! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip for dependency management
- Git for version control

### Setting up the Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Noveum/noveum-trace.git
   cd noveum-trace
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

## Project Structure

```
noveum_trace/
├── core/              # Core tracing functionality
│   ├── client.py      # Main client class
│   ├── config.py      # Configuration management
│   ├── context.py     # Context management
│   ├── span.py        # Span implementation
│   └── trace.py       # Trace implementation
├── decorators/        # Decorator-based API
│   ├── base.py        # Base trace decorator
│   ├── llm.py         # LLM-specific decorator
│   ├── agent.py       # Agent-specific decorator
│   ├── tool.py        # Tool-specific decorator
│   └── retrieval.py   # Retrieval-specific decorator
├── transport/         # Transport layer
│   ├── http_transport.py    # HTTP transport
│   └── batch_processor.py   # Batch processing
├── integrations/      # Framework integrations
│   └── openai.py      # OpenAI integration
└── utils/             # Utility modules
    ├── exceptions.py   # Custom exceptions
    ├── llm_utils.py    # LLM utilities
    └── pii_redaction.py # PII redaction
```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public functions and classes
- Maximum line length: 88 characters

### Testing

- Write unit tests for all new functionality
- Aim for high test coverage
- Use pytest for testing framework
- Place tests in the `tests/` directory

### Documentation

- Update docstrings for any changed functionality
- Add examples to docstrings where helpful
- Update README.md if adding new features
- Follow Google-style docstring format

## Adding New Features

### Adding a New Decorator

1. Create a new file in `noveum_trace/decorators/` or extend existing ones
2. Follow the pattern established by existing decorators
3. Add comprehensive tests
4. Update the `__init__.py` file to export the new decorator
5. Add documentation and examples

### Adding a New Integration

1. Create a new file in `noveum_trace/integrations/`
2. Implement integration following existing patterns
3. Add integration tests
4. Document any special requirements or limitations

### Adding Utility Functions

1. Add functions to appropriate module in `noveum_trace/utils/`
2. Ensure functions are well-documented and tested
3. Consider if the function should be part of the public API

## Testing

### Running Tests

#### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run with coverage
pytest --cov=noveum_trace tests/unit/

# Run specific test file
pytest tests/unit/core/test_client.py

# Run with verbose output
pytest -v tests/unit/
```

#### Integration Tests

Integration tests verify end-to-end functionality with configurable endpoints and real-world scenarios.

```bash
# Run all integration tests
pytest tests/integration/

# Run integration tests with specific markers
pytest -m integration tests/integration/
pytest -m llm tests/integration/
pytest -m agent tests/integration/
pytest -m opentelemetry tests/integration/

# Run specific integration test files
pytest tests/integration/test_base_configuration.py
pytest tests/integration/test_real_llm_scenarios.py
pytest tests/integration/test_decorator_integrations.py
pytest tests/integration/test_opentelemetry_integration.py

# Run with verbose output to see detailed test execution
pytest -v -s tests/integration/
```

#### Configuring Endpoints for Integration Tests

Integration tests support configurable endpoints for different environments:

**Environment Variables:**
```bash
# For localhost development (default)
export NOVEUM_TEST_ENDPOINT="http://localhost:3000"

# For production testing
export NOVEUM_TEST_ENDPOINT="https://api.noveum.ai"

# For custom endpoint testing
export NOVEUM_TEST_ENDPOINT="https://your-custom-endpoint.com"

# Optional: Set API keys for real LLM provider testing
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

**Running Tests with Specific Endpoints:**

```bash
# Test with localhost endpoint (development)
NOVEUM_TEST_ENDPOINT="http://localhost:3000" pytest tests/integration/test_base_configuration.py

# Test with production endpoint
NOVEUM_TEST_ENDPOINT="https://api.noveum.ai" pytest tests/integration/test_base_configuration.py

# Test real LLM scenarios with actual API keys
OPENAI_API_KEY="sk-..." ANTHROPIC_API_KEY="sk-ant-..." pytest tests/integration/test_real_llm_scenarios.py

# Test all integration scenarios with custom endpoint
NOVEUM_TEST_ENDPOINT="https://staging.noveum.ai" pytest tests/integration/
```

**Test Categories:**

- **Base Configuration** (`test_base_configuration.py`): Endpoint switching, health checks, configuration persistence
- **Decorator Integration** (`test_decorator_integrations.py`): All decorators (@trace, @trace_llm, @trace_agent, @trace_tool)
- **Real LLM Scenarios** (`test_real_llm_scenarios.py`): Multi-turn conversations, function calling, multi-agent workflows
- **OpenTelemetry Integration** (`test_opentelemetry_integration.py`): Auto-instrumentation, framework integration, dual export

**Setting Up Local Test Server:**

For comprehensive integration testing, you can set up a local mock server:

```bash
# Simple HTTP server for endpoint testing
python -m http.server 3000
```

### Test Structure

- **Unit tests** (`tests/unit/`): Test individual functions and classes
- **Integration tests** (`tests/integration/`): Test end-to-end workflows with configurable endpoints
- **Performance tests** (`tests/performance/`): Test performance and load scenarios
- **E2E tests** (`tests/e2e/`): Test complete user workflows

### Writing Integration Tests

When adding new integration tests:

1. **Use configurable endpoints**:
   ```python
   def test_my_feature(clean_noveum_client):
       endpoint = os.environ.get('NOVEUM_TEST_ENDPOINT', 'http://localhost:3000')
       noveum_trace.init(
           project="test-project",
           api_key="test-key",
           endpoint=endpoint
       )
   ```

2. **Test real scenarios**:
   ```python
   @pytest.mark.integration
   @pytest.mark.llm
   def test_realistic_conversation():
       # Test actual conversation patterns
       # Use real message structures
       # Include error handling and edge cases
   ```

3. **Mock external services appropriately**:
   ```python
   # Mock LLM providers but preserve request/response structure
   # Mock databases but test actual query patterns
   # Mock HTTP calls but validate endpoint usage
   ```

## Release Process

### Prerequisites for Releases

- Ensure you have maintainer access to the repository
- Install commitizen: `pip install commitizen`
- Make sure all changes are merged to main branch
- Ensure all tests pass

### Creating a New Release

1. **Switch to main branch and pull latest changes:**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Run the test suite to ensure everything works:**
   ```bash
   pytest
   ```

3. **Use commitizen to bump version and generate changelog:**
   ```bash
   # Automatically determine version bump (patch/minor/major)
   cz bump

   # Or specify version type explicitly
   cz bump --increment PATCH   # for bug fixes
   cz bump --increment MINOR   # for new features
   cz bump --increment MAJOR   # for breaking changes
   ```

4. **Push the release with tags:**
   ```bash
   git push origin main --follow-tags
   ```

5. **Create a GitHub release (optional):**
   - Go to GitHub releases page
   - Click "Create a new release"
   - Select the newly created tag
   - Use the generated changelog as release notes

### Making Conventional Commits

When making commits, use commitizen for conventional commit messages:

```bash
# Interactive commit message creation
cz commit

# Or use conventional commit format manually:
git commit -m "feat: add new tracing decorator"
git commit -m "fix: resolve span context issue"
git commit -m "docs: update API documentation"
```

### Commit Types

- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the development guidelines
3. **Write or update tests** for your changes
4. **Update documentation** as needed
5. **Run the test suite** and ensure all tests pass
6. **Submit a pull request** with a clear description

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] No breaking changes (or clearly documented)

## Code Review Process

- All submissions require review from maintainers
- Reviews focus on correctness, performance, and maintainability
- Address feedback promptly and professionally

## Getting Help

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: Check the README and examples
- **Email**: Contact the maintainers at support@noveum.ai

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (Apache License 2.0).
