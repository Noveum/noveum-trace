"""
Integration tests for auto-instrumentation functionality.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

import noveum_trace
from noveum_trace.instrumentation import openai, anthropic


class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        noveum_trace.shutdown()
        self.temp_dir = tempfile.mkdtemp()
        
        # Disable auto-instrumentation to start clean
        try:
            openai.uninstrument_openai()
        except:
            pass
        try:
            anthropic.uninstrument_anthropic()
        except:
            pass
    
    def teardown_method(self):
        """Cleanup after each test."""
        noveum_trace.shutdown()
        
        # Disable instrumentation
        try:
            openai.uninstrument_openai()
        except:
            pass
        try:
            anthropic.uninstrument_anthropic()
        except:
            pass
        
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_auto_instrumentation_enabled_by_default(self):
        """Test that auto-instrumentation is enabled by default."""
        tracer = noveum_trace.init(
            service_name="auto-instrument-test",
            log_directory=self.temp_dir
        )
        
        # Check that instrumentation is enabled
        assert openai.is_instrumented()
        assert anthropic.is_instrumented()
    
    def test_auto_instrumentation_can_be_disabled(self):
        """Test that auto-instrumentation can be disabled."""
        tracer = noveum_trace.init(
            service_name="no-auto-instrument-test",
            log_directory=self.temp_dir,
            auto_instrument=False
        )
        
        # Check that instrumentation is not enabled
        assert not openai.is_instrumented()
        assert not anthropic.is_instrumented()
    
    def test_manual_instrumentation_control(self):
        """Test manual control of instrumentation."""
        tracer = noveum_trace.init(
            service_name="manual-instrument-test",
            log_directory=self.temp_dir,
            auto_instrument=False
        )
        
        # Manually enable instrumentation
        noveum_trace.enable_auto_instrumentation()
        
        assert openai.is_instrumented()
        assert anthropic.is_instrumented()
        
        # Manually disable instrumentation
        noveum_trace.disable_auto_instrumentation()
        
        assert not openai.is_instrumented()
        assert not anthropic.is_instrumented()
    
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_anthropic_real_api_call(self):
        """Test real Anthropic API call with instrumentation."""
        tracer = noveum_trace.init(
            service_name="anthropic-real-test",
            log_directory=self.temp_dir
        )
        
        try:
            import anthropic
            
            client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            
            # Make a real API call (this will be traced)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=20,
                messages=[
                    {"role": "user", "content": "Say hello briefly"}
                ]
            )
            
            # Flush traces
            noveum_trace.flush()
            
            # Check that trace files were created
            trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
            assert len(trace_files) > 0
            
            # Check trace content
            with open(trace_files[0], 'r') as f:
                content = f.read()
                assert "anthropic" in content
                assert "claude-3-haiku" in content
                assert "gen_ai.system" in content
                
        except ImportError:
            pytest.skip("Anthropic SDK not available")
    
    def test_openai_mock_api_call(self):
        """Test OpenAI API call with mocked response."""
        tracer = noveum_trace.init(
            service_name="openai-mock-test",
            log_directory=self.temp_dir
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
            
            with patch.object(
                openai.resources.chat.completions.Completions,
                'create',
                return_value=mock_response
            ):
                client = openai.OpenAI(api_key="mock-api-key")
                
                # Make a mocked API call (this will be traced)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Say hello"}
                    ],
                    max_tokens=10
                )
                
                assert response.choices[0].message.content == "Hello from mocked OpenAI!"
            
            # Flush traces
            noveum_trace.flush()
            
            # Check that trace files were created
            trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
            assert len(trace_files) > 0
            
            # Check trace content
            with open(trace_files[0], 'r') as f:
                content = f.read()
                assert "openai" in content
                assert "gpt-3.5-turbo" in content
                assert "gen_ai.system" in content
                assert "Hello from mocked OpenAI!" in content
                
        except ImportError:
            pytest.skip("OpenAI SDK not available")
    
    def test_instrumentation_error_handling(self):
        """Test error handling in instrumentation."""
        tracer = noveum_trace.init(
            service_name="error-handling-test",
            log_directory=self.temp_dir
        )
        
        try:
            import openai
            
            # Mock an API error
            with patch.object(
                openai.resources.chat.completions.Completions,
                'create',
                side_effect=openai.AuthenticationError("Invalid API key")
            ):
                client = openai.OpenAI(api_key="invalid-key")
                
                # This should raise an exception but still be traced
                with pytest.raises(openai.AuthenticationError):
                    client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "test"}]
                    )
            
            # Flush traces
            noveum_trace.flush()
            
            # Check that error was traced
            trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
            assert len(trace_files) > 0
            
            with open(trace_files[0], 'r') as f:
                content = f.read()
                assert "error" in content
                assert "AuthenticationError" in content
                
        except ImportError:
            pytest.skip("OpenAI SDK not available")
    
    def test_multiple_llm_calls(self):
        """Test multiple LLM calls with instrumentation."""
        tracer = noveum_trace.init(
            service_name="multiple-calls-test",
            log_directory=self.temp_dir
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
            
            with patch.object(
                openai.resources.chat.completions.Completions,
                'create',
                side_effect=mock_responses
            ):
                client = openai.OpenAI(api_key="mock-api-key")
                
                # Make multiple API calls
                for i in range(3):
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"Message {i+1}"}]
                    )
                    assert f"Response {i+1}" in response.choices[0].message.content
            
            # Flush traces
            noveum_trace.flush()
            
            # Check that all calls were traced
            trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
            assert len(trace_files) > 0
            
            with open(trace_files[0], 'r') as f:
                content = f.read()
                # Should have 3 separate trace entries
                lines = [line for line in content.split('\n') if line.strip()]
                assert len(lines) == 3
                
                for i, line in enumerate(lines):
                    assert f"Response {i+1}" in line
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
            service_name="missing-sdk-test",
            log_directory=self.temp_dir,
            auto_instrument=True  # Should not fail even if SDKs are missing
        )
        
        # Should still initialize successfully
        assert tracer is not None
        
        # Manual instrumentation should handle missing SDKs gracefully
        noveum_trace.enable_auto_instrumentation()  # Should not raise
        noveum_trace.disable_auto_instrumentation()  # Should not raise

