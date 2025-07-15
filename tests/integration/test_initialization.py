"""
Integration tests for SDK initialization and configuration.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path

import noveum_trace
from noveum_trace.core.tracer import get_current_tracer


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
        tracer = noveum_trace.init(
            service_name="test-service",
            log_directory=self.temp_dir
        )
        
        assert tracer is not None
        assert tracer.config.service_name == "test-service"
        assert len(tracer.config.sinks) == 1  # File sink
        
        # Check that global tracer is set
        current_tracer = noveum_trace.get_tracer()
        assert current_tracer is tracer
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        tracer = noveum_trace.init(
            api_key="test-api-key",
            project_id="test-project",
            service_name="test-service",
            log_directory=self.temp_dir
        )
        
        assert tracer is not None
        assert len(tracer.config.sinks) == 2  # File + Noveum sinks
    
    def test_init_from_env_vars(self):
        """Test initialization from environment variables."""
        # Set environment variables
        os.environ["NOVEUM_API_KEY"] = "env-api-key"
        os.environ["NOVEUM_PROJECT_ID"] = "env-project"
        
        try:
            tracer = noveum_trace.init(
                service_name="env-test-service",
                log_directory=self.temp_dir
            )
            
            assert tracer is not None
            assert len(tracer.config.sinks) == 2  # File + Noveum sinks
            
        finally:
            # Clean up environment variables
            del os.environ["NOVEUM_API_KEY"]
            del os.environ["NOVEUM_PROJECT_ID"]
    
    def test_context_manager(self):
        """Test context manager pattern."""
        with noveum_trace.NoveumTrace(
            service_name="context-test",
            log_directory=self.temp_dir
        ) as tracer:
            assert tracer is not None
            assert noveum_trace.get_tracer() is tracer
        
        # After context exit, tracer should be cleaned up
        assert noveum_trace.get_tracer() is None
    
    def test_multiple_init_calls(self):
        """Test multiple initialization calls."""
        tracer1 = noveum_trace.init(
            service_name="service1",
            log_directory=self.temp_dir
        )
        
        tracer2 = noveum_trace.init(
            service_name="service2", 
            log_directory=self.temp_dir
        )
        
        # Second init should replace the first
        assert tracer2 is not tracer1
        assert noveum_trace.get_tracer() is tracer2
    
    def test_file_logging_disabled(self):
        """Test initialization with file logging disabled."""
        tracer = noveum_trace.init(
            service_name="no-file-service",
            file_logging=False
        )
        
        assert tracer is not None
        assert len(tracer.config.sinks) == 1  # Console sink only
    
    def test_auto_instrumentation_disabled(self):
        """Test initialization with auto-instrumentation disabled."""
        tracer = noveum_trace.init(
            service_name="no-auto-service",
            log_directory=self.temp_dir,
            auto_instrument=False
        )
        
        assert tracer is not None
        
        # Check that auto-instrumentation is not enabled
        from noveum_trace.instrumentation import openai, anthropic
        assert not openai.is_instrumented()
        assert not anthropic.is_instrumented()
    
    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        tracer = noveum_trace.init(
            service_name="custom-service",
            environment="testing",
            log_directory=self.temp_dir,
            capture_content=False,
            batch_size=5,
            batch_timeout_ms=500,
            max_file_size_mb=50,
            max_files=5
        )
        
        assert tracer is not None
        assert tracer.config.service_name == "custom-service"
        assert tracer.config.environment == "testing"
        assert tracer.config.capture_llm_content is False
        assert tracer.config.batch_size == 5
        assert tracer.config.batch_timeout_ms == 500
    
    def test_flush_and_shutdown(self):
        """Test flush and shutdown operations."""
        tracer = noveum_trace.init(
            service_name="flush-test",
            log_directory=self.temp_dir
        )
        
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
            service_name="file-test",
            log_directory=self.temp_dir
        )
        
        # Create a test span
        with tracer.start_span("test-span") as span:
            span.set_attribute("test.key", "test.value")
        
        # Flush to ensure file is written
        noveum_trace.flush()
        
        # Check that trace files exist
        trace_dir = Path(self.temp_dir)
        trace_files = list(trace_dir.glob("traces_*.jsonl"))
        assert len(trace_files) > 0
        
        # Check file content
        with open(trace_files[0], 'r') as f:
            content = f.read()
            assert "test-span" in content
            assert "test.key" in content
    
    def test_custom_log_directory(self):
        """Test custom log directory."""
        custom_dir = os.path.join(self.temp_dir, "custom", "traces")
        
        tracer = noveum_trace.init(
            service_name="custom-dir-test",
            log_directory=custom_dir
        )
        
        # Create a test span
        with tracer.start_span("test-span"):
            pass
        
        noveum_trace.flush()
        
        # Check that custom directory was created and contains files
        assert os.path.exists(custom_dir)
        trace_files = list(Path(custom_dir).glob("traces_*.jsonl"))
        assert len(trace_files) > 0


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
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        
        try:
            # This should not raise an exception, but should log a warning
            tracer = noveum_trace.init(
                service_name="invalid-dir-test",
                log_directory=temp_file.name  # This is a file, not a directory
            )
            
            # Should fall back to console sink
            assert tracer is not None
            
        finally:
            os.unlink(temp_file.name)
    
    def test_missing_project_id_with_api_key(self):
        """Test handling of API key without project ID."""
        tracer = noveum_trace.init(
            api_key="test-api-key",
            # project_id is missing
            service_name="missing-project-test",
            file_logging=False
        )
        
        # Should still initialize but only with console sink
        assert tracer is not None
        assert len(tracer.config.sinks) == 1  # Console sink only

