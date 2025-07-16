# Contributing to Noveum Trace SDK

We welcome contributions to the Noveum Trace SDK! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## Getting Started

### Development Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Noveum/noveum-trace.git
   cd noveum-trace
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e .[dev]
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards outlined below
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and linting**
   ```bash
   # Run tests
   pytest

   # Run linting
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add streaming LLM support
fix: resolve context propagation issue
docs: update installation instructions
test: add unit tests for span creation
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use [mypy](https://mypy.readthedocs.io/) for type checking

### Code Quality

- **Type Hints**: All public functions must have type hints
- **Docstrings**: All public classes and functions must have docstrings
- **Error Handling**: Use appropriate exception types from `utils.exceptions`
- **Logging**: Use structured logging with appropriate levels
- **Testing**: Maintain >90% test coverage

### Architecture Guidelines

- **Single Responsibility**: Each class/function should have one clear purpose
- **Dependency Injection**: Use dependency injection for testability
- **Interface Segregation**: Keep interfaces focused and minimal
- **Open/Closed Principle**: Design for extension without modification

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── conftest.py     # Shared test fixtures
```

### Writing Tests

- Use `pytest` for all tests
- Follow the AAA pattern (Arrange, Act, Assert)
- Use descriptive test names
- Mock external dependencies
- Test both success and failure scenarios

Example test:
```python
def test_span_creation_with_attributes():
    # Arrange
    tracer = NoveumTracer()
    attributes = {"user.id": "123", "operation": "test"}

    # Act
    span = tracer.start_span("test_span", attributes=attributes)

    # Assert
    assert span.name == "test_span"
    assert span.get_attribute("user.id") == "123"
    assert span.get_attribute("operation") == "test"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=noveum_trace --cov-report=html

# Run specific test file
pytest tests/unit/test_span.py

# Run tests matching pattern
pytest -k "test_span"

# Run integration tests
pytest tests/integration/
```

## Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples for all public APIs
- Update docstrings when changing function signatures
- Add type information to all parameters and return values

### Building Documentation

```bash
# Install documentation dependencies
pip install .[docs]

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Update documentation if needed
3. Add changelog entry if applicable
4. Verify code coverage meets requirements
5. Run pre-commit hooks

### Pull Request Template

When creating a pull request, please include:

- **Description**: Clear description of changes
- **Type**: Feature, bug fix, documentation, etc.
- **Testing**: How the changes were tested
- **Breaking Changes**: Any breaking changes
- **Related Issues**: Link to related issues

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Verify tests cover new functionality
4. **Documentation**: Ensure documentation is updated

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version in `src/noveum_trace/__init__.py`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and publish to PyPI
5. Update documentation

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time chat with the community
- **Email**: team@noveum.ai for private inquiries

### Issue Templates

When reporting bugs or requesting features, please use the appropriate issue template and provide:

- **Bug Reports**: Steps to reproduce, expected vs actual behavior, environment details
- **Feature Requests**: Use case, proposed solution, alternatives considered

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing to Noveum Trace SDK!
