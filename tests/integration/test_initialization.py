"""
Integration tests for SDK initialization and configuration.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

import noveum_trace
from noveum_trace.utils.exceptions import SinkError


class TestInitialization:
    """Test SDK initialization patterns."""

    def setup_method(self):
        """Setup for each test."""
        # Ensure clean state
        noveum_trace.shutdown()

        # Create temporary directory for traces
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test."""
        noveum_trace.shutdown()

        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_simple_init(self):
        """Test simple initialization."""
        # Temporarily unset NOVEUM_API_KEY to test only file sink
        import os

        original_api_key = os.environ.get("NOVEUM_API_KEY")
        if "NOVEUM_API_KEY" in os.environ:
            del os.environ["NOVEUM_API_KEY"]

        try:
            tracer = noveum_trace.init(
                project_id="test-project", log_directory=self.temp_dir
            )

            assert tracer is not None
            assert tracer.config.project_id == "test-project"
            assert len(tracer.config.sinks) == 1  # File sink only
            assert tracer.config.sinks[0].name == "file-sink"

            # Check that global tracer is set
            current_tracer = noveum_trace.get_tracer()
            assert current_tracer is tracer
        finally:
            # Restore original API key
            if original_api_key:
                os.environ["NOVEUM_API_KEY"] = original_api_key

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        tracer = noveum_trace.init(
            api_key="test-api-key",
            project_id="test-project",
            log_directory=self.temp_dir,
        )

        assert tracer is not None
        assert len(tracer.config.sinks) == 2  # File + Noveum sinks

    def test_init_from_env_vars(self):
        """Test initialization from environment variables."""
        # Set environment variables
        os.environ["NOVEUM_API_KEY"] = "env-api-key"
        os.environ["NOVEUM_PROJECT_ID"] = "env-project"

        try:
            tracer = noveum_trace.init(log_directory=self.temp_dir)

            assert tracer is not None
            assert len(tracer.config.sinks) == 2  # File + Noveum sinks

        finally:
            # Clean up environment variables
            del os.environ["NOVEUM_API_KEY"]
            del os.environ["NOVEUM_PROJECT_ID"]

    def test_context_manager(self):
        """Test context manager pattern."""
        with noveum_trace.init(
            project_id="context-test", log_directory=self.temp_dir
        ) as tracer:
            assert tracer is not None
            assert noveum_trace.get_tracer() is tracer

        # After context exit, tracer should be cleaned up
        assert noveum_trace.get_tracer() is None

    def test_multiple_init_calls(self):
        """Test multiple initialization calls."""
        tracer1 = noveum_trace.init(project_id="service1", log_directory=self.temp_dir)

        tracer2 = noveum_trace.init(project_id="service2", log_directory=self.temp_dir)

        # Second init should replace the first
        assert tracer2 is not tracer1
        assert noveum_trace.get_tracer() is tracer2

    def test_file_logging_disabled(self):
        """Test initialization with file logging disabled."""
        tracer = noveum_trace.init(project_id="no-file-service", file_logging=False)

        assert tracer is not None
        assert len(tracer.config.sinks) == 1  # Console sink only

    def test_auto_instrumentation_disabled(self):
        """Test initialization with auto-instrumentation disabled."""
        tracer = noveum_trace.init(
            project_id="no-auto-service",
            log_directory=self.temp_dir,
            auto_instrument=False,
        )

        assert tracer is not None

        # Check that auto-instrumentation is not enabled
        from noveum_trace.instrumentation import anthropic, openai

        assert not openai.is_instrumented()
        assert not anthropic.is_instrumented()

    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        tracer = noveum_trace.init(
            project_id="custom-service",
            environment="testing",
            log_directory=self.temp_dir,
            capture_content=False,
            batch_size=5,
            batch_timeout_ms=500,
            max_file_size_mb=50,
            max_files=5,
        )

        assert tracer is not None
        assert tracer.config.project_id == "custom-service"
        assert tracer.config.environment == "testing"
        assert tracer.config.capture_content is False
        assert tracer.config.batch_size == 5
        assert tracer.config.batch_timeout_ms == 500

    def test_flush_and_shutdown(self):
        """Test flush and shutdown operations."""
        tracer = noveum_trace.init(project_id="flush-test", log_directory=self.temp_dir)

        # Create a test span
        with tracer.start_span("test-span") as span:
            span.set_attribute("test.key", "test.value")

        # Flush should work without errors
        noveum_trace.flush(timeout_ms=1000)

        # Shutdown should work without errors
        noveum_trace.shutdown()

        # After shutdown, tracer should be None
        assert noveum_trace.get_tracer() is None


class TestFileLogging:
    """Test file logging functionality."""

    def setup_method(self):
        """Setup for each test."""
        noveum_trace.shutdown()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test."""
        noveum_trace.shutdown()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_trace_files_created(self):
        """Test that trace files are created."""
        tracer = noveum_trace.init(
            project_id="file-test",
            log_directory=self.temp_dir,
            batch_size=1,  # Force immediate export for testing
            batch_timeout_ms=100,
        )

        # Create a test span
        with tracer.start_span("test-span") as span:
            span.set_attribute("test.key", "test.value")

        # Allow time for export worker to process the span
        time.sleep(0.2)

        # Flush to ensure file is written
        noveum_trace.flush()

        # Check that trace files exist
        trace_dir = Path(self.temp_dir)
        trace_files = list(trace_dir.glob("traces_*.jsonl"))
        assert len(trace_files) > 0

        # Check file content
        with open(trace_files[0]) as f:
            content = f.read()
            assert content.strip(), "Trace file should not be empty"
            assert "test-span" in content
            assert "test.key" in content

    def test_custom_log_directory(self):
        """Test custom log directory."""
        custom_dir = os.path.join(self.temp_dir, "custom", "traces")

        tracer = noveum_trace.init(
            project_id="custom-dir-test",
            log_directory=custom_dir,
            batch_size=1,  # Force immediate export for testing
            batch_timeout_ms=100,
        )

        # Create a test span
        with tracer.start_span("test-span"):
            pass

        # Allow time for export
        time.sleep(0.2)
        noveum_trace.flush()

        # Check that custom directory was created and contains files
        assert os.path.exists(custom_dir)
        trace_files = list(Path(custom_dir).glob("traces_*.jsonl"))
        assert len(trace_files) > 0

        # Verify file has content
        with open(trace_files[0]) as f:
            content = f.read()
            assert content.strip(), "Trace file should not be empty"


class TestErrorHandling:
    """Test error handling in initialization."""

    def setup_method(self):
        """Setup for each test."""
        noveum_trace.shutdown()

    def teardown_method(self):
        """Cleanup after each test."""
        noveum_trace.shutdown()

    def test_invalid_log_directory(self):
        """Test handling of invalid log directory."""
        # Try to use a file as a directory (should fail gracefully)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            # This should now raise a SinkError instead of silently deleting the file
            noveum_trace.init(
                project_id="invalid-dir-test",
                log_directory=temp_file_path,  # This is a file, not a directory
            )

            # Should not reach here - the above should raise an exception
            raise AssertionError(
                "Expected SinkError to be raised for invalid log directory"
            )

        except SinkError as e:
            # This is the expected behavior - FileSink should raise an error
            # instead of silently deleting the file
            assert "points to an existing file" in str(e)
            print(f"✅ Correctly raised SinkError: {e}")

            # Verify the original file was not deleted
            assert os.path.exists(temp_file_path), "Original file should not be deleted"

        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except (OSError, PermissionError):
                # On some systems, we might not be able to delete immediately
                # This is acceptable for the test
                pass

    def test_missing_project_id_with_api_key(self):
        """Test that missing project_id raises error even with API key."""
        with pytest.raises(
            noveum_trace.ConfigurationError, match="project_id is required"
        ):
            noveum_trace.init(
                api_key="test-api-key",
                # project_id is missing
                file_logging=False,
            )

        # If we get here, the exception should have been raised
        # No further assertions needed since we're testing the exception
