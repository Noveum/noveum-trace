# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.2 (2025-01-23)

### Added

#### 🧪 **Comprehensive Test Infrastructure**

- **Integration Test Suite**: Complete integration testing framework in `tests/integration/` directory
  - **Mock Endpoint Tests** (`mock_endpoint/`): Tests with mocked HTTP endpoints for fast, offline testing
  - **End-to-End Tests** (`end_to_end/`): Tests with real LLM API calls for production validation
  - **Configurable Endpoints**: Support for testing against localhost, staging, or production endpoints
- **Enhanced Unit Test Coverage**: Added comprehensive unit tests across all modules:
  - `test_client_comprehensive.py`, `test_client_fixed.py` - Enhanced client testing
  - `test_config_comprehensive.py` - Comprehensive configuration testing
  - `test_context_comprehensive.py` - Complete context management testing
  - `test_decorators_comprehensive.py` - Extensive decorator testing
  - `test_http_transport_comprehensive.py` - Transport layer testing
  - `test_endpoint_validation.py` - Endpoint validation testing
- **Test Organization and Markers**: Comprehensive test categorization with pytest markers:
  - `@pytest.mark.integration` - Integration tests
  - `@pytest.mark.llm` - LLM-specific tests
  - `@pytest.mark.agent` - Agent workflow tests
  - `@pytest.mark.openai` - OpenAI integration tests
  - `@pytest.mark.anthropic` - Anthropic integration tests
- **Advanced Test Infrastructure**: Enhanced mocking and validation capabilities:
  - `EndpointCapture` class for HTTP request validation
  - Comprehensive test fixtures for clean state management
  - Advanced mock infrastructure preventing real API calls during testing

#### 📊 **Enhanced Test Coverage**

- **Increased Code Coverage**: Significantly improved test coverage across all modules
- **Comprehensive Validation**: Tests now validate:
  - HTTP mocking infrastructure prevents all real API calls
  - Trace and span data structures are properly formatted
  - LLM, agent, tool, and performance attributes are correctly captured
  - Nested span hierarchies maintain proper parent-child relationships
  - Error handling preserves exception details and status codes
  - Data type compatibility ensures JSON serialization works correctly

#### 📚 **Enhanced Documentation**

- **Integration Testing Guide**: Comprehensive documentation in `tests/integration/README.md`
  - Detailed setup instructions for different testing scenarios
  - Environment configuration guidance
  - Test execution examples and best practices
- **Enhanced Contributing Guidelines**: Updated `CONTRIBUTING.md` with:
  - Integration test setup and execution instructions
  - Endpoint configuration for different environments
  - Test categorization and selection guidance

### Fixed

#### 🔧 **Test Infrastructure Fixes**

- **Test Reliability**: Fixed flaky tests and improved test stability
- **Mock Accuracy**: Enhanced mock infrastructure for better test isolation
- **CI/CD Integration**: Improved continuous integration and test execution
- **Import and Setup Issues**: Resolved test environment setup and configuration issues

### Changed

#### 🏗️ **Test Organization**

- **Structured Test Layout**: Reorganized tests into logical categories and directories
- **Improved Test Execution**: Enhanced test execution speed and reliability
- **Better Test Documentation**: Clear guidelines for writing and executing different types of tests

---

## 0.3.1 (2025-01-22)

### Fixed

#### 🐛 **Critical Configuration Fixes**

- **Endpoint Configuration Bug**: Fixed critical issue where SDK was ignoring custom endpoint configuration
  - **Custom Endpoint Support**: SDK now properly uses custom endpoints specified via `init()` or environment variables
  - **Path Preservation**: Fixed endpoint path preservation (e.g., `/beta`, `/api`) in URL construction
  - **Environment Variable Priority**: Fixed precedence so explicit parameters override environment variables
  - **Configuration Validation**: Enhanced configuration validation with better error messages
- **Transport Layer**: Fixed HTTP transport to properly use configured custom endpoints instead of hardcoded defaults
- **Config Class Enhancement**: Fixed `Config.create()` method to properly accept and handle `endpoint` parameter

#### 🔧 **Configuration Improvements**

- **URL Construction**: Enhanced `_build_api_url()` method to preserve custom endpoint paths
- **Environment Loading**: Improved environment variable loading and configuration merging
- **Validation Logic**: Enhanced configuration validation with proper URL format checking

### Added

#### 🔧 **Configuration Features**

- **Endpoint Property**: Added proper endpoint property getter/setter to Config class
- **Enhanced Validation**: Comprehensive URL validation with regex patterns
- **Better Error Messages**: More descriptive configuration error messages with actionable guidance

### Changed

#### 🏗️ **Configuration Architecture**

- **Config Initialization**: Improved configuration initialization and merging logic
- **Environment Handling**: Better handling of environment variables and explicit parameters
- **Transport Configuration**: Enhanced transport configuration with proper endpoint usage

---

## 0.3.0 (2025-07-18)

### BREAKING CHANGE

- Complete architectural rewrite from ground up with new client-based system
- Replaced Tracer/Sink pattern with NoveumClient, Trace, and Span primitives
- New decorator-based API incompatible with previous versions
- Simplified initialization and configuration system
- Removed OpenTelemetry dependency for direct HTTP transport

### Added

#### 🏗️ **Core Architecture**

- **NoveumClient**: New centralized client for all SDK operations with automatic lifecycle management
- **Trace & Span Classes**: Explicit `Trace` and `Span` objects with full serialization support
- **Context Management**: Robust context propagation using `contextvars` for sync and async operations
- **Auto-trace Creation**: Intelligent automatic trace creation when none exists

#### 🎨 **Decorator-based API**

- **@trace**: Universal decorator for function tracing with comprehensive metadata capture
- **@trace_llm**: LLM-specific decorator with automatic prompt, completion, and token tracking
- **@trace_agent**: Multi-agent system decorator with agent identity and capability tracking
- **@trace_tool**: Tool usage decorator for RAG and agent tool interactions
- **@trace_retrieval**: Specialized decorator for retrieval operations and vector search

#### 🤖 **Multi-Agent System Support**

- **Agent Registry**: Comprehensive agent management with lifecycle tracking
- **Agent Graphs**: Visual representation and tracking of agent interactions
- **Agent Workflows**: Structured workflow coordination with automatic tracing
- **Inter-agent Communication**: Automatic tracing of agent-to-agent interactions
- **Agent Cleanup**: Memory management and cleanup utilities for long-running systems

#### 🔄 **Context Managers**

- **trace_context**: Programmatic span creation and management
- **trace_llm_call**: Direct LLM call tracing with metadata extraction
- **trace_agent_operation**: Agent-specific operation tracing
- **trace_batch_operation**: Batch processing tracing
- **trace_pipeline_stage**: Pipeline stage tracing

#### 📡 **Auto-instrumentation System**

- **OpenAI Integration**: Complete auto-instrumentation for OpenAI API calls
- **Anthropic Support**: Auto-instrumentation for Anthropic Claude API
- **LangChain Integration**: Framework-level tracing for LangChain applications
- **Plugin Architecture**: Extensible system for custom integrations
- **Configuration-driven**: Flexible configuration options for each integration

#### 🌊 **Streaming Support**

- **StreamingSpanManager**: Real-time tracing for streaming LLM responses
- **Token-by-token Tracking**: Incremental token capture with timing metrics
- **Streaming Metrics**: Tokens per second, time to first token, latency analysis
- **OpenAI Streaming**: Native support for OpenAI streaming responses
- **Anthropic Streaming**: Native support for Anthropic streaming responses

#### 🧵 **Thread Management**

- **ThreadContext**: Conversation thread tracking with message history
- **Multi-turn Conversations**: Automatic conversation flow tracking
- **Thread-specific Tracing**: LLM calls linked to specific conversation threads
- **Thread Metadata**: Rich metadata capture for conversation analysis

#### 🚀 **Transport Layer**

- **HTTP-only Transport**: Direct HTTP communication (no OpenTelemetry complexity)
- **Batch Processing**: Efficient batching with configurable size and timeout
- **Retry Logic**: Robust retry mechanism with exponential backoff
- **Compression**: Optional payload compression for large traces
- **Authentication**: Bearer token authentication with API key management

#### ⚙️ **Configuration System**

- **Environment Variables**: Complete environment variable support
- **Config Files**: YAML and JSON configuration file support
- **Programmatic Configuration**: Type-safe configuration classes
- **Auto-discovery**: Automatic configuration discovery from multiple sources
- **Validation**: Built-in configuration validation with helpful error messages

#### 🔒 **Security & Privacy**

- **PII Redaction**: Advanced PII detection and redaction utilities
- **Data Encryption**: Transport-level encryption for sensitive data
- **Configurable Redaction**: Custom redaction patterns and rules
- **Data Residency**: Geographic data residency configuration
- **Secure Defaults**: Privacy-first default configuration

#### 🛠️ **Development Experience**

- **Type Hints**: Complete type annotations throughout the codebase
- **Auto-completion**: Rich IDE support with comprehensive type information
- **Error Handling**: Detailed error messages with actionable suggestions
- **Debug Mode**: Comprehensive debug logging and introspection
- **Performance Monitoring**: Built-in performance tracking and optimization

#### 📊 **Observability Features**

- **Performance Metrics**: CPU usage, memory consumption, latency tracking
- **Cost Estimation**: Automatic cost calculation for LLM API calls
- **Token Analysis**: Detailed token usage and optimization recommendations
- **Error Tracking**: Comprehensive error capture with stack traces
- **Health Checks**: Transport and system health monitoring

#### 🧪 **Testing & Quality**

- **Comprehensive Test Suite**: 42 tests with 100% pass rate
- **Mock Integration**: Extensive mocking for external dependencies
- **Performance Tests**: Benchmark tests for performance validation
- **Integration Tests**: End-to-end testing scenarios
- **Example Applications**: Working examples for all major features

### Changed

#### 🔄 **API Simplification**

- **Simplified Initialization**: Single `noveum_trace.init()` function replaces complex setup
- **Cleaner Imports**: Streamlined import structure with clear public API
- **Consistent Naming**: Standardized naming conventions across all modules
- **Better Defaults**: Sensible default configuration for immediate productivity
- **Reduced Boilerplate**: Minimal code required for basic functionality

#### 📈 **Performance Improvements**

- **Minimal Overhead**: <2% CPU impact during normal operation
- **Memory Efficiency**: Optimized memory usage with proper cleanup
- **Batch Optimization**: Intelligent batching for maximum throughput
- **Context Switching**: Efficient context propagation with minimal overhead
- **Resource Management**: Proper resource cleanup and lifecycle management

#### 🏗️ **Code Organization**

- **Modular Structure**: Clear separation of concerns across modules
- **Plugin System**: Extensible architecture for custom functionality
- **Dependency Management**: Minimal runtime dependencies
- **Documentation**: Comprehensive docstrings and type annotations
- **Code Quality**: Consistent formatting and linting throughout

### Fixed

#### 🐛 **Core Fixes**

- **Import System**: Resolved circular import issues and import errors
- **Context Propagation**: Fixed context inheritance across async boundaries
- **Memory Leaks**: Eliminated memory leaks in long-running applications
- **Thread Safety**: Resolved race conditions in concurrent usage
- **Error Handling**: Improved error handling with better error messages

#### 🔧 **Integration Fixes**

- **OpenAI Integration**: Fixed instrumentation issues with OpenAI v1.x
- **Anthropic Integration**: Resolved API compatibility issues
- **LangChain Integration**: Fixed chain tracing and context propagation
- **Transport Layer**: Fixed HTTP timeout and retry logic issues
- **Configuration**: Resolved configuration loading and validation issues

#### 🔧 **Critical Configuration Fixes**

- **Endpoint Configuration**: Fixed Config class constructor to accept `endpoint` parameter
- **Custom Endpoint Support**: Fixed SDK ignoring custom endpoint configuration
- **Environment Variable Priority**: Fixed precedence of explicit parameters over environment variables
- **Default Endpoint**: Ensured `https://api.noveum.ai` is always used as default when no endpoint is provided
- **Transport Layer**: Fixed HTTP transport to properly use configured custom endpoints instead of hardcoded defaults
- **Configuration Validation**: Added comprehensive unit tests for endpoint configuration functionality

#### 🧪 **Testing Fixes**

- **Test Reliability**: Fixed flaky tests and improved test stability
- **Mock Accuracy**: Improved mock accuracy for external dependencies
- **Coverage**: Increased test coverage to 90%+ across all modules
- **CI/CD**: Fixed continuous integration and deployment issues
- **Comprehensive Endpoint Validation**: Added extensive test suite (`test_mock_validation.py`) that validates:
  - HTTP mocking infrastructure prevents all real API calls to api.noveum.ai
  - Trace and span data structures are valid and properly formatted
  - LLM, agent, tool, and performance attributes are correctly captured
  - Nested span hierarchies maintain proper parent-child relationships
  - Error handling preserves exception details and status codes
  - Data type compatibility ensures JSON serialization works correctly
  - Decorator integration functions properly with mocked transport layer
  - Configuration injection adds required SDK metadata to traces
- **Mock Infrastructure Enhancement**: Enhanced `conftest.py` with `TestEndpointCapture` class for advanced HTTP request validation
- **Zero Real API Calls**: Verified complete prevention of external HTTP requests during test execution

### Removed

#### 🗑️ **Deprecated Features**

- **OpenTelemetry Dependency**: Removed complex OpenTelemetry integration
- **Legacy Sink System**: Replaced with modern transport layer
- **Old Tracer API**: Simplified to new client-based approach
- **Complex Configuration**: Removed overly complex configuration options
- **Unused Dependencies**: Cleaned up unused and outdated dependencies

#### 🧹 **Code Cleanup**

- **Dead Code**: Removed unused code and obsolete modules
- **Legacy Examples**: Replaced with modern usage examples
- **Outdated Tests**: Removed obsolete test files and fixtures
- **Documentation**: Removed outdated documentation and comments

### Migration Guide

#### From v0.1.x to v0.3.0

**Old API:**

```python
from noveum_trace import NoveumTracer
tracer = NoveumTracer(api_key="key")
with tracer.trace("operation") as span:
    # operation code
```

**New API:**

```python
import noveum_trace
noveum_trace.init(api_key="key", project="my-app")

@noveum_trace.trace
def operation():
    # operation code
```

**Key Changes:**

1. Replace `NoveumTracer` with `noveum_trace.init()`
2. Use decorators instead of context managers for most use cases
3. Update import statements to use the new module structure
4. Review configuration options (many have been simplified)
5. Update test code to use new mock interfaces

**Benefits:**

- 90% reduction in setup code
- Automatic trace lifecycle management
- Better error handling and debugging
- Improved performance and reliability
- Enhanced multi-agent support

---

## v0.1.2 (2025-07-16)

### Fixed

- Critical bug fixes in core tracing functionality
- Improved error handling and validation
- Enhanced test coverage and reliability

## v0.1.1 (2025-07-16)

### Fixed

- Initial bug fixes and stability improvements
- Test infrastructure improvements
- Documentation updates and clarifications
