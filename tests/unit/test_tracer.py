"""
Unit tests for NoveumTracer class.
"""

import pytest
import time
from noveum_trace import NoveumTracer, TracerConfig
from noveum_trace.types import SpanKind


class TestNoveumTracer:
    """Test cases for NoveumTracer class."""
    
    def test_tracer_creation(self, tracer_config):
        """Test basic tracer creation."""
        tracer = NoveumTracer(tracer_config)
        
        assert tracer.config.project_name == "test-project"
        assert tracer.is_recording is True
        
        tracer.shutdown()
    
    def test_start_span(self, tracer):
        """Test starting a span."""
        span = tracer.start_span("test_operation")
        
        assert span.name == "test_operation"
        assert span.kind == SpanKind.INTERNAL
        assert span.is_recording is True
        
        span.end()
    
    def test_start_span_with_attributes(self, tracer):
        """Test starting a span with attributes."""
        attributes = {"user.id": "123", "operation": "test"}
        span = tracer.start_span("test_operation", attributes=attributes)
        
        assert span.get_attribute("user.id") == "123"
        assert span.get_attribute("operation") == "test"
        
        span.end()
    
    def test_span_hierarchy(self, tracer):
        """Test parent-child span relationships."""
        parent_span = tracer.start_span("parent_operation")
        
        with parent_span:
            child_span = tracer.start_span("child_operation")
            assert child_span.parent_span_id == parent_span.span_id
            child_span.end()
    
    def test_tracer_decorator(self, tracer):
        """Test tracer decorator functionality."""
        @tracer.trace_function("decorated_function")
        def test_function():
            return "result"
        
        result = test_function()
        assert result == "result"
    
    def test_flush(self, tracer, mock_sink):
        """Test flushing spans."""
        span = tracer.start_span("test_operation")
        span.end()
        
        # Give some time for background processing
        time.sleep(0.2)
        
        success = tracer.flush(timeout_ms=1000)
        assert success is True
    
    def test_shutdown(self, tracer_config):
        """Test tracer shutdown."""
        tracer = NoveumTracer(tracer_config)
        
        span = tracer.start_span("test_operation")
        span.end()
        
        success = tracer.shutdown(timeout_ms=1000)
        assert success is True
        assert tracer.is_recording is False

