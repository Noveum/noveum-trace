"""
Unit tests for LangChain code tracing functionality.

Tests the code tracing features:
- extract_call_site_info()
- extract_function_definition_info()
- Code attributes in spans
"""

from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

# Skip all tests if LangChain is not available
try:
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler
    from noveum_trace.integrations.langchain_utils import (
        extract_call_site_info,
        extract_function_definition_info,
        _find_project_root,
        _make_path_relative,
        _is_library_directory,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestCodeTracingUtils:
    """Test code tracing utility functions."""

    def test_extract_call_site_info_finds_user_code(self):
        """Test that extract_call_site_info finds user code frame."""

        def user_function():
            return extract_call_site_info(skip_frames=1)

        result = user_function()

        assert result is not None
        assert "code.file" in result
        assert "code.line" in result
        assert "code.function" in result
        # The function name will be the test function, not user_function
        # because skip_frames=1 skips extract_call_site_info, then finds the test
        assert "code.module" in result

    def test_extract_call_site_info_skips_library_code(self):
        """Test that library code is skipped."""
        result = extract_call_site_info(skip_frames=0)

        if result:
            file_path = result.get("code.file", "")
            assert "site-packages" not in file_path
            assert "venv" not in file_path

    def test_extract_function_definition_info(self):
        """Test function definition info extraction."""

        def test_function():
            """A test function."""
            return 42

        result = extract_function_definition_info(test_function)

        assert result is not None
        assert "function.definition.file" in result
        assert "function.definition.start_line" in result
        assert "function.definition.end_line" in result
        assert result["function.definition.start_line"] > 0
        assert (
            result["function.definition.end_line"]
            >= result["function.definition.start_line"]
        )

    def test_extract_function_definition_info_builtin(self):
        """Test that builtin functions return None."""
        result = extract_function_definition_info(len)
        assert result is None

    def test_is_library_directory(self):
        """Test library directory detection."""
        assert _is_library_directory(Path("/usr/lib/python3.9/site-packages"))
        assert _is_library_directory(Path("/path/to/venv/lib/python3.9"))
        assert not _is_library_directory(Path("/path/to/project/src"))

    def test_find_project_root(self):
        """Test project root detection."""
        current_file = __file__
        project_root = _find_project_root(current_file)

        if project_root:
            assert not _is_library_directory(project_root)

    def test_make_path_relative(self):
        """Test path relative conversion."""
        current_file = __file__
        relative_path = _make_path_relative(current_file)

        assert relative_path is not None
        assert not Path(relative_path).is_absolute()
        assert Path(current_file).name in relative_path


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestCodeTracingInSpans:
    """Test code tracing attributes in spans."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_span = Mock()
            mock_span.span_id = "test-span-id"
            mock_client.start_span.return_value = mock_span
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_on_llm_start_includes_code_attributes(self, handler):
        """Test that on_llm_start includes code attributes."""
        run_id = uuid4()
        serialized = {
            "name": "ChatOpenAI",
            "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
            "kwargs": {"model": "gpt-4"},
        }
        prompts = ["Hello, world!"]

        def user_code():
            handler.on_llm_start(serialized, prompts, run_id=run_id)

        user_code()

        handler._client.start_span.assert_called_once()
        call_args = handler._client.start_span.call_args
        attributes = call_args.kwargs.get("attributes", {})

        assert (
            "code.file" in attributes
            or "code.line" in attributes
            or "code.function" in attributes
        )

    def test_on_tool_start_includes_code_attributes(self, handler):
        """Test that on_tool_start includes code attributes."""
        run_id = uuid4()
        serialized = {
            "name": "test_tool",
            "id": ["langchain", "tools", "test_tool"],
        }

        def user_code():
            handler.on_tool_start(serialized, "test input", run_id=run_id)

        user_code()

        handler._client.start_span.assert_called_once()
        call_args = handler._client.start_span.call_args
        attributes = call_args.kwargs.get("attributes", {})

        assert (
            "code.file" in attributes
            or "code.line" in attributes
            or "code.function" in attributes
        )
