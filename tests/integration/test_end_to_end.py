"""
End-to-end integration tests for complete tracing workflows.
"""

import os
import tempfile
import shutil
import json
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

import noveum_trace


class TestEndToEndTracing:
    """Test complete end-to-end tracing workflows."""
    
    def setup_method(self):
        """Setup for each test."""
        noveum_trace.shutdown()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after each test."""
        noveum_trace.shutdown()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_llm_workflow(self):
        """Test complete LLM workflow with tracing."""
        # Initialize with file logging
        tracer = noveum_trace.init(
            service_name="e2e-llm-workflow",
            log_directory=self.temp_dir,
            capture_content=True
        )
        
        # Simulate a complete LLM application workflow
        with tracer.start_span("llm_application") as app_span:
            app_span.set_attribute("app.version", "1.0.0")
            app_span.set_attribute("app.user_id", "user123")
            
            # Preprocessing step
            with tracer.start_span("preprocessing") as prep_span:
                prep_span.set_attribute("preprocessing.input_length", 100)
                time.sleep(0.01)  # Simulate processing time
                prep_span.set_attribute("preprocessing.output_length", 95)
            
            # LLM call (simulated)
            with tracer.start_span("llm_call") as llm_span:
                llm_span.set_attribute("gen_ai.system", "openai")
                llm_span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
                llm_span.set_attribute("gen_ai.operation.name", "chat")
                
                # Add input event
                llm_span.add_event("gen_ai.content.prompt", {
                    "gen_ai.prompt": "What is the capital of France?"
                })
                
                time.sleep(0.05)  # Simulate LLM latency
                
                # Add output event
                llm_span.add_event("gen_ai.content.completion", {
                    "gen_ai.completion": "The capital of France is Paris."
                })
                
                llm_span.set_attribute("gen_ai.usage.input_tokens", 8)
                llm_span.set_attribute("gen_ai.usage.output_tokens", 7)
                llm_span.set_attribute("gen_ai.usage.total_tokens", 15)
            
            # Postprocessing step
            with tracer.start_span("postprocessing") as post_span:
                post_span.set_attribute("postprocessing.filter_applied", True)
                time.sleep(0.01)  # Simulate processing time
        
        # Flush and verify traces
        noveum_trace.flush()
        
        # Check trace files
        trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
        assert len(trace_files) > 0
        
        # Parse and verify trace content
        traces = []
        with open(trace_files[0], 'r') as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
        
        # Should have 4 spans: app, preprocessing, llm_call, postprocessing
        assert len(traces) == 4
        
        # Verify span hierarchy and content
        span_names = [trace['name'] for trace in traces]
        assert "llm_application" in span_names
        assert "preprocessing" in span_names
        assert "llm_call" in span_names
        assert "postprocessing" in span_names
        
        # Verify LLM span has correct attributes
        llm_trace = next(trace for trace in traces if trace['name'] == 'llm_call')
        assert llm_trace['attributes']['gen_ai.system'] == 'openai'
        assert llm_trace['attributes']['gen_ai.request.model'] == 'gpt-3.5-turbo'
        assert llm_trace['attributes']['gen_ai.usage.total_tokens'] == 15
        
        # Verify events
        assert len(llm_trace['events']) == 2
        event_names = [event['name'] for event in llm_trace['events']]
        assert "gen_ai.content.prompt" in event_names
        assert "gen_ai.content.completion" in event_names
    
    def test_error_handling_workflow(self):
        """Test error handling in tracing workflow."""
        tracer = noveum_trace.init(
            service_name="e2e-error-handling",
            log_directory=self.temp_dir
        )
        
        # Simulate workflow with errors
        with tracer.start_span("error_workflow") as app_span:
            try:
                with tracer.start_span("failing_operation") as fail_span:
                    fail_span.set_attribute("operation.type", "llm_call")
                    
                    # Simulate an error
                    raise ValueError("Simulated LLM API error")
                    
            except ValueError as e:
                app_span.record_exception(e)
                app_span.set_status("error", str(e))
        
        noveum_trace.flush()
        
        # Verify error was captured
        trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
        assert len(trace_files) > 0
        
        with open(trace_files[0], 'r') as f:
            content = f.read()
            assert "error" in content
            assert "ValueError" in content
            assert "Simulated LLM API error" in content
    
    def test_concurrent_tracing(self):
        """Test concurrent tracing operations."""
        import threading
        import queue
        
        tracer = noveum_trace.init(
            service_name="e2e-concurrent",
            log_directory=self.temp_dir
        )
        
        results = queue.Queue()
        
        def worker(worker_id):
            """Worker function that creates traces."""
            try:
                with tracer.start_span(f"worker_{worker_id}") as span:
                    span.set_attribute("worker.id", worker_id)
                    time.sleep(0.01)  # Simulate work
                    span.set_attribute("worker.status", "completed")
                results.put(f"worker_{worker_id}_success")
            except Exception as e:
                results.put(f"worker_{worker_id}_error: {e}")
        
        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        worker_results = []
        while not results.empty():
            worker_results.append(results.get())
        
        assert len(worker_results) == 5
        assert all("success" in result for result in worker_results)
        
        noveum_trace.flush()
        
        # Verify all worker spans were captured
        trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
        assert len(trace_files) > 0
        
        traces = []
        with open(trace_files[0], 'r') as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
        
        # Should have 5 worker spans
        worker_spans = [trace for trace in traces if trace['name'].startswith('worker_')]
        assert len(worker_spans) == 5
        
        # Verify each worker has unique ID
        worker_ids = [span['attributes']['worker.id'] for span in worker_spans]
        assert len(set(worker_ids)) == 5  # All unique
    
    def test_streaming_simulation(self):
        """Test simulation of streaming LLM responses."""
        tracer = noveum_trace.init(
            service_name="e2e-streaming",
            log_directory=self.temp_dir
        )
        
        # Simulate streaming LLM call
        with tracer.start_span("streaming_llm_call") as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.request.model", "gpt-3.5-turbo")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("llm.streaming", True)
            
            # Add prompt event
            span.add_event("gen_ai.content.prompt", {
                "gen_ai.prompt": "Write a short story about AI"
            })
            
            # Simulate streaming chunks
            chunks = [
                "Once upon a time,",
                " there was an AI",
                " that learned to dream.",
                " The end."
            ]
            
            full_response = ""
            for i, chunk in enumerate(chunks):
                time.sleep(0.01)  # Simulate streaming delay
                full_response += chunk
                
                # Add chunk event
                span.add_event("gen_ai.content.chunk", {
                    "gen_ai.completion.chunk": chunk,
                    "chunk.index": i,
                    "chunk.timestamp": time.time()
                })
            
            # Add final completion event
            span.add_event("gen_ai.content.completion", {
                "gen_ai.completion": full_response
            })
            
            span.set_attribute("gen_ai.usage.input_tokens", 10)
            span.set_attribute("gen_ai.usage.output_tokens", 15)
            span.set_attribute("llm.chunks_count", len(chunks))
        
        noveum_trace.flush()
        
        # Verify streaming trace
        trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
        assert len(trace_files) > 0
        
        with open(trace_files[0], 'r') as f:
            trace = json.loads(f.read().strip())
        
        assert trace['attributes']['llm.streaming'] is True
        assert trace['attributes']['llm.chunks_count'] == 4
        
        # Should have prompt + 4 chunks + completion = 6 events
        assert len(trace['events']) == 6
        
        # Verify chunk events
        chunk_events = [event for event in trace['events'] if event['name'] == 'gen_ai.content.chunk']
        assert len(chunk_events) == 4
    
    def test_batch_processing_workflow(self):
        """Test batch processing workflow with multiple LLM calls."""
        tracer = noveum_trace.init(
            service_name="e2e-batch-processing",
            log_directory=self.temp_dir,
            batch_size=3,  # Small batch size for testing
            batch_timeout_ms=100
        )
        
        # Simulate batch processing
        with tracer.start_span("batch_processor") as batch_span:
            batch_span.set_attribute("batch.size", 5)
            
            for i in range(5):
                with tracer.start_span(f"process_item_{i}") as item_span:
                    item_span.set_attribute("item.id", i)
                    item_span.set_attribute("item.type", "text_generation")
                    
                    # Simulate LLM call for each item
                    with tracer.start_span(f"llm_call_{i}") as llm_span:
                        llm_span.set_attribute("gen_ai.system", "anthropic")
                        llm_span.set_attribute("gen_ai.request.model", "claude-3-haiku")
                        
                        time.sleep(0.01)  # Simulate processing
                        
                        llm_span.set_attribute("gen_ai.usage.input_tokens", 20)
                        llm_span.set_attribute("gen_ai.usage.output_tokens", 30)
        
        # Wait for batching and flush
        time.sleep(0.2)  # Allow batch timeout
        noveum_trace.flush()
        
        # Verify all spans were captured
        trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
        assert len(trace_files) > 0
        
        traces = []
        with open(trace_files[0], 'r') as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
        
        # Should have: 1 batch + 5 items + 5 LLM calls = 11 spans
        assert len(traces) == 11
        
        # Verify span types
        span_names = [trace['name'] for trace in traces]
        assert "batch_processor" in span_names
        assert sum(1 for name in span_names if name.startswith("process_item_")) == 5
        assert sum(1 for name in span_names if name.startswith("llm_call_")) == 5


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    def setup_method(self):
        """Setup for each test."""
        noveum_trace.shutdown()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after each test."""
        noveum_trace.shutdown()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_high_volume_tracing(self):
        """Test tracing with high volume of spans."""
        tracer = noveum_trace.init(
            service_name="e2e-high-volume",
            log_directory=self.temp_dir,
            batch_size=50,
            batch_timeout_ms=100
        )
        
        start_time = time.time()
        
        # Create many spans quickly
        for i in range(100):
            with tracer.start_span(f"high_volume_span_{i}") as span:
                span.set_attribute("span.index", i)
                span.set_attribute("gen_ai.system", "test")
        
        end_time = time.time()
        
        # Should complete quickly (< 1 second for 100 spans)
        duration = end_time - start_time
        assert duration < 1.0, f"High volume tracing took too long: {duration}s"
        
        noveum_trace.flush()
        
        # Verify all spans were captured
        trace_files = list(Path(self.temp_dir).glob("traces_*.jsonl"))
        assert len(trace_files) > 0
        
        traces = []
        with open(trace_files[0], 'r') as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
        
        assert len(traces) == 100
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during tracing."""
        import psutil
        import gc
        
        tracer = noveum_trace.init(
            service_name="e2e-memory-test",
            log_directory=self.temp_dir
        )
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create many spans and flush regularly
        for batch in range(10):
            for i in range(50):
                with tracer.start_span(f"memory_test_span_{batch}_{i}") as span:
                    span.set_attribute("batch", batch)
                    span.set_attribute("index", i)
            
            noveum_trace.flush()
            gc.collect()  # Force garbage collection
        
        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB for this test)
        assert memory_increase < 50 * 1024 * 1024, f"Memory usage increased by {memory_increase / 1024 / 1024:.1f}MB"

