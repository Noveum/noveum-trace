#!/usr/bin/env python3
"""
Development setup script for Noveum Trace SDK.

This script helps set up the development environment and
verifies that all components are properly structured.
"""

import sys
from pathlib import Path


def check_file_structure():
    """Check that all required files and directories exist."""

    base_dir = Path(__file__).parent

    required_files = [
        "pyproject.toml",
        "README.md",
        "CONTRIBUTING.md",
        ".gitignore",
        "src/noveum_trace/__init__.py",
        "src/noveum_trace/core/__init__.py",
        "src/noveum_trace/core/client.py",
        "src/noveum_trace/core/config.py",
        "src/noveum_trace/core/context.py",
        "src/noveum_trace/core/span.py",
        "src/noveum_trace/core/trace.py",
        "src/noveum_trace/decorators/__init__.py",
        "src/noveum_trace/decorators/base.py",
        "src/noveum_trace/decorators/llm.py",
        "src/noveum_trace/decorators/agent.py",
        "src/noveum_trace/decorators/tool.py",
        "src/noveum_trace/decorators/retrieval.py",
        "src/noveum_trace/transport/__init__.py",
        "src/noveum_trace/transport/http_transport.py",
        "src/noveum_trace/transport/batch_processor.py",
        "src/noveum_trace/utils/__init__.py",
        "src/noveum_trace/utils/exceptions.py",
        "src/noveum_trace/utils/llm_utils.py",
        "src/noveum_trace/utils/pii_redaction.py",
        "src/noveum_trace/integrations/__init__.py",
        "src/noveum_trace/integrations/openai.py",
        "examples/basic_usage.py",
        "tests/__init__.py",
        "tests/conftest.py",
    ]

    required_dirs = [
        "src",
        "src/noveum_trace",
        "src/noveum_trace/core",
        "src/noveum_trace/decorators",
        "src/noveum_trace/transport",
        "src/noveum_trace/utils",
        "src/noveum_trace/integrations",
        "src/noveum_trace/plugins",
        "tests",
        "tests/unit",
        "tests/e2e",
        "tests/performance",
        "examples",
    ]

    print("Checking file structure...")

    missing_files = []
    for file_path in required_files:
        full_path = base_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    missing_dirs = []
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)

    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")

    if missing_dirs:
        print("❌ Missing directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")

    if not missing_files and not missing_dirs:
        print("✅ All required files and directories are present!")
        return True

    return False


def check_imports():
    """Check that basic imports work."""

    print("\nChecking imports...")

    try:
        # Add the src directory to Python path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))

        # Test basic imports
        import noveum_trace  # noqa: F401

        print("✅ Main package imports successfully")

        from noveum_trace import trace_llm_call, trace_operation  # noqa: F401

        print("✅ Context manager imports successful")

        from noveum_trace.core.config import Config  # noqa: F401

        print("✅ Core modules import successfully")

        from noveum_trace.utils.exceptions import NoveumTraceError  # noqa: F401

        print("✅ Utility modules import successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def print_summary():
    """Print a summary of the SDK structure."""

    print("\n" + "=" * 60)
    print("NOVEUM TRACE SDK - DEVELOPMENT SETUP COMPLETE")
    print("=" * 60)

    print("\n📦 Package Structure:")
    print("├── src/")
    print("│   └── noveum_trace/       # Main package")
    print("│       ├── core/           # Core tracing functionality")
    print("│       ├── decorators/     # Decorator-based API")
    print("│       ├── transport/      # HTTP transport layer")
    print("│       ├── utils/          # Utility functions")
    print("│       ├── integrations/   # Framework integrations")
    print("│       └── plugins/        # Plugin system")
    print("├── tests/                  # Test suite")
    print("│   ├── unit/              # Unit tests")
    print("│   ├── e2e/               # End-to-end tests")
    print("│   └── performance/       # Performance tests")
    print("├── examples/               # Usage examples")
    print("└── docs/                   # Documentation")

    print("\n🚀 Next Steps:")
    print("1. Install development dependencies:")
    print("   pip install -e '.[dev]'")
    print("\n2. Run the example:")
    print("   python examples/basic_usage.py")
    print("\n3. Run tests:")
    print("   pytest tests/")
    print("\n4. Run linting:")
    print("   ruff check src/ tests/")
    print("   black --check src/ tests/")
    print("   isort --check-only src/ tests/")
    print("   mypy src/noveum_trace/")
    print("\n5. Start developing!")
    print("   - Extend context managers or core tracing in src/noveum_trace/")
    print("   - Add new integrations in src/noveum_trace/integrations/")
    print("   - Add tests in tests/")

    print("\n📚 Key Features Implemented:")
    print("✅ Modular architecture with clear separation of concerns")
    print(
        "✅ Context-manager API (trace_llm_call, trace_operation, trace_agent_operation, etc.)"
    )
    print("✅ HTTP transport with batching and retry logic")
    print("✅ Comprehensive configuration system")
    print("✅ Framework integration support")
    print("✅ PII redaction and security features")
    print("✅ Multi-agent system tracing")
    print("✅ Auto-instrumentation capabilities")
    print("✅ Multiple output sinks (console, file, cloud)")
    print("✅ Extensible design for future enhancements")

    print("\n🔧 Development Tools:")
    print("- Code formatting with Black")
    print("- Import sorting with isort")
    print("- Linting with Ruff")
    print("- Type checking with mypy")
    print("- Security scanning with bandit")
    print("- Pre-commit hooks for code quality")
    print("- Comprehensive test structure")
    print("- Type hints throughout")
    print("- Detailed documentation")
    print("- Contributing guidelines")


def main():
    """Main setup function."""

    print("Noveum Trace SDK - Development Setup")
    print("=" * 40)

    structure_ok = check_file_structure()
    imports_ok = check_imports()

    if structure_ok and imports_ok:
        print("\n🎉 Setup verification completed successfully!")
        print_summary()
    else:
        print("\n❌ Setup verification failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
