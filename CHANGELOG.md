# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-16

### Added
- Initial release of noveum-trace SDK
- OpenTelemetry-compliant tracing for LLM applications
- Automatic instrumentation for OpenAI and Anthropic APIs
- Multi-agent workflow support
- Real-time evaluation integration with Noveum.ai
- Multiple sink implementations (console, file, Elasticsearch, Noveum)
- Comprehensive test suite with unit and integration tests
- Complete documentation and examples

### Changed
- Updated build system to use setuptools instead of hatchling
- Migrated to Apache-2.0 license
- Enhanced project configuration with comprehensive tooling

### Fixed
- Resolved import system issues in agents module
- Fixed test initialization parameters
- Corrected span and tracer implementation issues
- Enhanced error handling and validation
