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

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=noveum_trace

# Run specific test file
pytest tests/test_decorators.py

# Run with verbose output
pytest -v
```

### Test Structure

- Unit tests: Test individual functions and classes
- Integration tests: Test interactions between components
- Mock external dependencies appropriately

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
