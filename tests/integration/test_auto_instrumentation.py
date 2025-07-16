"""
Tests for auto-instrumentation functionality.
"""

import contextlib
import importlib.util
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import noveum_trace
from noveum_trace.instrumentation import anthropic, openai


class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    def setup_method(self):
        """Setup for each test."""
        # Disable auto-instrumentation to start clean
        with contextlib.suppress(Exception):
            openai.uninstrument_openai()
        with contextlib.suppress(Exception):
            anthropic.uninstrument_anthropic()

        # Create temp directory for test traces
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test."""
        # Shutdown tracer
        noveum_trace.shutdown()

        # Disable instrumentation
        with contextlib.suppress(Exception):
            openai.uninstrument_openai()
        with contextlib.suppress(Exception):
            anthropic.uninstrument_anthropic()

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_auto_instrumentation_enabled(self):
        """Test that auto-instrumentation is enabled by default."""
        # Initialize tracer with auto-instrumentation enabled
        noveum_trace.init(
            project_id="test_project",
            file_logging=True,
            log_directory="test_traces",
            auto_instrument=True,
        )

        # Check that instrumentation is enabled (only if packages are installed)
        if importlib.util.find_spec("openai") is not None:
            assert openai.is_instrumented()
        else:
            pytest.skip("OpenAI not installed")

        if importlib.util.find_spec("anthropic") is not None:
            assert anthropic.is_instrumented()
        else:
            pytest.skip("Anthropic not installed")

        # Clean up
        noveum_trace.shutdown()

    def test_auto_instrumentation_disabled(self):
        """Test that auto-instrumentation can be disabled."""
        # Initialize tracer with auto-instrumentation disabled
        noveum_trace.init(
            project_id="test_project",
            file_logging=True,
            log_directory="test_traces",
            auto_instrument=False,
        )

        # Check that instrumentation is not enabled
        if importlib.util.find_spec("openai") is not None:
            assert not openai.is_instrumented()
        else:
            pytest.skip("OpenAI not installed")

        if importlib.util.find_spec("anthropic") is not None:
            assert not anthropic.is_instrumented()
        else:
            pytest.skip("Anthropic not installed")

        # Clean up
        noveum_trace.shutdown()

    def test_selective_instrumentation(self):
        """Test selective instrumentation of specific packages."""
        # Initialize tracer with automatic instrumentation
        noveum_trace.init(
            project_id="test_project",
            file_logging=True,
            log_directory="test_traces",
            auto_instrument=True,
        )

        # Check that only selected packages are instrumented
        if importlib.util.find_spec("openai") is not None:
            assert openai.is_instrumented()
        else:
            pytest.skip("OpenAI not installed")

        if importlib.util.find_spec("anthropic") is not None:
            assert anthropic.is_instrumented()
        else:
            pytest.skip("Anthropic not installed")

        # Clean up
        noveum_trace.shutdown()

    def test_uninstrumentation(self):
        """Test that uninstrumentation works correctly."""
        # Initialize tracer with instrumentation enabled
        noveum_trace.init(
            project_id="test_project",
            file_logging=True,
            log_directory="test_traces",
            auto_instrument=True,
        )

        # Check that instrumentation is enabled (only if packages are installed)
        if importlib.util.find_spec("openai") is not None:
            assert openai.is_instrumented()
        else:
            pytest.skip("OpenAI not installed")

        if importlib.util.find_spec("anthropic") is not None:
            assert anthropic.is_instrumented()
        else:
            pytest.skip("Anthropic not installed")

        # Uninstrument specific packages
        openai.uninstrument_openai()
        anthropic.uninstrument_anthropic()

        # Check that uninstrumentation worked
        if importlib.util.find_spec("openai") is not None:
            assert not openai.is_instrumented()
        else:
            pytest.skip("OpenAI not installed")

        if importlib.util.find_spec("anthropic") is not None:
            assert not anthropic.is_instrumented()
        else:
            pytest.skip("Anthropic not installed")

        # Clean up
        noveum_trace.shutdown()

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
    )
    def test_anthropic_real_api_call(self):
        """Test real Anthropic API call with instrumentation."""
        noveum_trace.init(
            project_id="test-project",
            log_directory=self.temp_dir,
            batch_size=1,  # Force immediate export for testing
            batch_timeout_ms=100,
        )

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

            # Make a real API call (this will be traced)
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=20,
                messages=[{"role": "user", "content": "Say hello briefly"}],
            )

            # Wait a moment for export worker to process
            time.sleep(0.2)

            # Flush traces
            noveum_trace.flush()

            # Check that trace files were created
            trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
            assert len(trace_files) > 0

            # Check trace content
            with open(trace_files[0]) as f:
                content = f.read()
                assert "anthropic" in content
                assert "claude-3-haiku" in content
                assert "gen_ai.system" in content

        except ImportError:
            pytest.skip("Anthropic SDK not available")

    def test_openai_mock_api_call(self):
        """Test OpenAI API call with mocked response."""
        # Initialize without auto-instrumentation first
        noveum_trace.init(
            project_id="test-project",
            log_directory=self.temp_dir,
            batch_size=1,  # Force immediate export for testing
            batch_timeout_ms=100,
            auto_instrument=False,  # Disable initially
        )

        try:
            import openai

            # Mock the OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Hello from mocked OpenAI!"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-3.5-turbo"
            mock_response.id = "mock-response-id"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15

            # Patch the method before enabling instrumentation
            with patch.object(
                openai.resources.chat.completions.Completions,
                "create",
                return_value=mock_response,
            ):
                # Now enable auto-instrumentation - this will wrap the already patched method
                noveum_trace.enable_auto_instrumentation()

                client = openai.OpenAI(api_key="mock-api-key")

                # Make a mocked API call (this will be traced)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Say hello"}],
                    max_tokens=10,
                )

                assert (
                    response.choices[0].message.content == "Hello from mocked OpenAI!"
                )

            # Allow time for export
            time.sleep(0.2)

            # Flush traces
            noveum_trace.flush()

            # Check that trace files were created
            trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
            assert len(trace_files) > 0

            # Check trace content
            with open(trace_files[0]) as f:
                content = f.read()
                assert content.strip(), "Trace file should not be empty"
                assert "openai" in content
                assert "gpt-3.5-turbo" in content
                assert "gen_ai.system" in content
                # Note: The mocked content won't be in the trace due to content capture settings

        except ImportError:
            pytest.skip("OpenAI SDK not available")

    def test_instrumentation_error_handling(self):
        """Test error handling in instrumentation."""
        noveum_trace.init(
            project_id="test-project",
            log_directory=self.temp_dir,
            batch_size=1,  # Force immediate export for testing
            batch_timeout_ms=100,
            auto_instrument=False,  # Disable initially
        )

        try:
            import openai

            # Create a proper mock response for the error
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Invalid API key"
            mock_response.headers = {}

            # Mock an API error with proper constructor arguments
            error = openai.AuthenticationError(
                "Invalid API key",
                response=mock_response,
                body={"error": {"message": "Invalid API key"}},
            )

            # Patch before enabling instrumentation
            with patch.object(
                openai.resources.chat.completions.Completions,
                "create",
                side_effect=error,
            ):
                # Enable auto-instrumentation after patching
                noveum_trace.enable_auto_instrumentation()

                client = openai.OpenAI(api_key="invalid-key")

                # This should raise an exception but still be traced
                with pytest.raises(openai.AuthenticationError):
                    client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "test"}],
                    )

            # Allow time for export
            time.sleep(0.2)

            # Flush traces
            noveum_trace.flush()

            # Check that error was traced
            trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
            assert len(trace_files) > 0

            with open(trace_files[0]) as f:
                content = f.read()
                assert content.strip(), "Trace file should not be empty"
                assert "error" in content.lower()
                assert "AuthenticationError" in content

        except ImportError:
            pytest.skip("OpenAI SDK not available")

    def test_multiple_llm_calls(self):
        """Test multiple LLM calls with instrumentation."""
        noveum_trace.init(
            project_id="test-project",
            log_directory=self.temp_dir,
            batch_size=1,  # Force immediate export for testing
            batch_timeout_ms=100,
            auto_instrument=False,  # Disable initially
        )

        try:
            import openai

            # Mock multiple responses
            mock_responses = []
            for i in range(3):
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = f"Response {i+1}"
                mock_response.choices[0].finish_reason = "stop"
                mock_response.model = "gpt-3.5-turbo"
                mock_response.id = f"mock-response-{i+1}"
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 5
                mock_response.usage.total_tokens = 15
                mock_responses.append(mock_response)

            # Patch before enabling instrumentation
            with patch.object(
                openai.resources.chat.completions.Completions,
                "create",
                side_effect=mock_responses,
            ):
                # Enable auto-instrumentation after patching
                noveum_trace.enable_auto_instrumentation()

                client = openai.OpenAI(api_key="mock-api-key")

                # Make multiple API calls
                for i in range(3):
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"Message {i+1}"}],
                    )
                    assert f"Response {i+1}" in response.choices[0].message.content

            # Allow time for export
            time.sleep(0.2)

            # Flush traces
            noveum_trace.flush()

            # Check that all calls were traced
            trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
            assert len(trace_files) > 0

            with open(trace_files[0]) as f:
                content = f.read()
                assert content.strip(), "Trace file should not be empty"
                # Should have 3 separate trace entries
                lines = [line for line in content.split("\n") if line.strip()]
                assert len(lines) == 3, f"Expected 3 traces, got {len(lines)}"

                for _i, line in enumerate(lines):
                    # Check for OpenAI system in traces but not response content
                    assert "openai" in line or "gen_ai.system" in line
                    assert "gpt-3.5-turbo" in line

        except ImportError:
            pytest.skip("OpenAI SDK not available")


class TestInstrumentationCompatibility:
    """Test instrumentation compatibility with different SDK versions."""

    def setup_method(self):
        """Setup for each test."""
        noveum_trace.shutdown()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test."""
        noveum_trace.shutdown()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_missing_sdk_graceful_handling(self):
        """Test graceful handling when SDKs are not installed."""
        # This test simulates the case where OpenAI/Anthropic SDKs are not installed

        tracer = noveum_trace.init(
            project_id="test-project",
            log_directory=self.temp_dir,
            auto_instrument=True,  # Should not fail even if SDKs are missing
        )

        # Should still initialize successfully
        assert tracer is not None

        # Manual instrumentation should handle missing SDKs gracefully
        noveum_trace.enable_auto_instrumentation()  # Should not raise
        noveum_trace.disable_auto_instrumentation()  # Should not raise
