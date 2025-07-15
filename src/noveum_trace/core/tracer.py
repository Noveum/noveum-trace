"""
Main tracer implementation for the Noveum Trace SDK.

This tracer provides OpenTelemetry-compatible tracing with enhanced features
for LLM applications, multi-agent systems, and tool calls.
"""

import asyncio
import threading
import queue
import time
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
import logging
from dataclasses import dataclass, field
from contextlib import contextmanager

from .span import Span
from .context import (
    TraceContext, get_current_span, get_current_trace, 
    set_current_trace, SpanContext, AsyncSpanContext
)
from ..types import (
    SpanKind, SpanData, LLMRequest, LLMResponse, CustomHeaders,
    TraceID, SpanID, OperationType, AISystem
)
from ..sinks.base import BaseSink
from ..utils.exceptions import (
    ConfigurationError, ValidationError, NoveumTracingError
)

logger = logging.getLogger(__name__)


@dataclass
class TracerConfig:
    """Configuration for the Noveum tracer."""
    
    # Project configuration
    project_id: str
    project_name: str = "unknown-project"
    org_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    environment: str = "development"
    
    # Sinks configuration
    sinks: List[BaseSink] = field(default_factory=list)
    
    # Custom headers
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    # Sampling configuration
    sampling_rate: float = 1.0  # 1.0 = 100% sampling
    max_spans_per_trace: int = 1000
    
    # Performance configuration
    max_queue_size: int = 10000
    batch_size: int = 100
    batch_timeout_ms: int = 5000
    max_export_timeout_ms: int = 30000
    
    # Content capture configuration
    capture_content: bool = True
    max_attribute_length: int = 1024
    max_content_length: int = 10000
    
    # Feature flags
    enable_auto_instrumentation: bool = True
    enable_metrics: bool = True
    enable_events: bool = True


class NoveumTracer:
    """
    Main tracer class for the Noveum Trace SDK.
    
    This tracer provides OpenTelemetry-compatible tracing with enhanced features
    for LLM applications, multi-agent systems, and tool calls.
    """
    
    def __init__(self, config: TracerConfig):
        """Initialize the tracer with configuration."""
        self.config = config
        self._validate_config()
        
        # Initialize internal state
        self._active_traces: Dict[str, TraceContext] = {}
        self._span_queue = queue.Queue(maxsize=config.max_queue_size)
        self._shutdown_event = threading.Event()
        self._export_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Start export thread
        self._start_export_thread()
        
        logger.info(f"NoveumTracer initialized for project: {config.project_id}")
    
    def _validate_config(self):
        """Validate tracer configuration."""
        if not self.config.project_id:
            raise ConfigurationError("project_id is required")
        
        if not (0.0 <= self.config.sampling_rate <= 1.0):
            raise ConfigurationError("sampling_rate must be between 0.0 and 1.0")
        
        if not self.config.sinks:
            logger.warning("No sinks configured - traces will not be exported")
    
    def _start_export_thread(self):
        """Start the background export thread."""
        self._export_thread = threading.Thread(
            target=self._export_worker,
            name="noveum-trace-exporter",
            daemon=True
        )
        self._export_thread.start()
    
    def _export_worker(self):
        """Background worker that exports spans to sinks."""
        batch = []
        last_export = time.time()
        
        while not self._shutdown_event.is_set():
            try:
                # Try to get a span with timeout
                try:
                    span_data = self._span_queue.get(timeout=1.0)
                    batch.append(span_data)
                except queue.Empty:
                    # Check if we should export partial batch
                    if batch and (time.time() - last_export) > (self.config.batch_timeout_ms / 1000):
                        self._export_batch(batch)
                        batch = []
                        last_export = time.time()
                    continue
                
                # Export batch if full or timeout reached
                should_export = (
                    len(batch) >= self.config.batch_size or
                    (time.time() - last_export) > (self.config.batch_timeout_ms / 1000)
                )
                
                if should_export:
                    self._export_batch(batch)
                    batch = []
                    last_export = time.time()
                
            except Exception as e:
                logger.error(f"Error in export worker: {e}")
        
        # Export remaining spans on shutdown
        if batch:
            self._export_batch(batch)
    
    def _export_batch(self, spans: List[SpanData]):
        """Export a batch of spans to all configured sinks."""
        if not spans:
            return
        
        for sink in self.config.sinks:
            try:
                sink.export(spans)
            except Exception as e:
                logger.error(f"Error exporting to sink {sink.__class__.__name__}: {e}")
    
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[Union[Span, SpanData]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        operation_type: Optional[OperationType] = None,
        ai_system: Optional[AISystem] = None
    ) -> Span:
        """
        Start a new span.
        
        Args:
            name: Span name
            kind: Span kind (INTERNAL, CLIENT, SERVER, etc.)
            parent: Parent span or span data
            attributes: Initial span attributes
            start_time: Span start time (defaults to now)
            operation_type: Type of operation being traced
            ai_system: AI system for LLM operations
        
        Returns:
            New Span instance
        """
        # Check sampling
        if not self._should_sample():
            return Span.create_no_op_span()
        
        # Get or create trace context
        current_trace = get_current_trace()
        if current_trace is None:
            trace_id = TraceID.generate()
            current_trace = TraceContext(trace_id=trace_id)
            set_current_trace(current_trace)
        else:
            trace_id = current_trace.trace_id
        
        # Determine parent span ID
        parent_span_id = None
        if parent:
            if isinstance(parent, Span):
                parent_span_id = parent.span_id
            elif isinstance(parent, SpanData):
                parent_span_id = parent.span_id
        else:
            current_span = get_current_span()
            if current_span:
                parent_span_id = current_span.span_id
        
        # Create span data
        span_data = SpanData(
            trace_id=trace_id,
            span_id=SpanID.generate(),
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=start_time or datetime.now(timezone.utc),
            attributes=attributes or {},
            project_id=self.config.project_id,
            project_name=self.config.project_name,
            org_id=self.config.org_id,
            user_id=self.config.user_id,
            session_id=self.config.session_id,
            environment=self.config.environment
        )
        
        # Add operation type and AI system
        if operation_type:
            span_data.set_attribute("operation.type", operation_type.value)
        if ai_system:
            span_data.set_attribute("ai.system", ai_system.value)
        
        # Create span
        span = Span(span_data, self)
        
        # Add to trace context
        current_trace.add_span(span)
        
        logger.debug(f"Started span: {name} (trace_id={trace_id}, span_id={span_data.span_id})")
        
        return span
    
    @contextmanager
    def start_span_context(self, *args, **kwargs):
        """Context manager for automatic span lifecycle management."""
        span = self.start_span(*args, **kwargs)
        try:
            with SpanContext(span):
                yield span
        finally:
            span.end()
    
    def _should_sample(self) -> bool:
        """Determine if a trace should be sampled."""
        import random
        return random.random() < self.config.sampling_rate
    
    def export_span(self, span_data: SpanData):
        """Export a completed span."""
        try:
            self._span_queue.put_nowait(span_data)
        except queue.Full:
            logger.warning("Span queue full, dropping span")
    
    def flush(self, timeout_ms: int = 5000) -> bool:
        """
        Flush all pending spans.
        
        Args:
            timeout_ms: Maximum time to wait for flush
        
        Returns:
            True if all spans were flushed successfully
        """
        start_time = time.time()
        timeout_seconds = timeout_ms / 1000
        
        # Wait for queue to empty
        while not self._span_queue.empty():
            if time.time() - start_time > timeout_seconds:
                logger.warning("Flush timeout reached")
                return False
            time.sleep(0.1)
        
        # Flush all sinks
        for sink in self.config.sinks:
            try:
                if hasattr(sink, 'flush'):
                    sink.flush()
            except Exception as e:
                logger.error(f"Error flushing sink {sink.__class__.__name__}: {e}")
        
        return True
    
    def shutdown(self):
        """Shutdown the tracer and cleanup resources."""
        logger.info("Shutting down NoveumTracer")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for export thread to finish
        if self._export_thread and self._export_thread.is_alive():
            self._export_thread.join(timeout=5.0)
        
        # Shutdown all sinks
        for sink in self.config.sinks:
            try:
                if hasattr(sink, 'shutdown'):
                    sink.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down sink {sink.__class__.__name__}: {e}")
        
        logger.info("NoveumTracer shutdown complete")
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return get_current_span()
    
    def get_current_trace(self) -> Optional[TraceContext]:
        """Get the current trace context."""
        return get_current_trace()
    
    def create_llm_span(
        self,
        name: str,
        model: str,
        operation: str = "chat",
        ai_system: AISystem = AISystem.OPENAI,
        request: Optional[LLMRequest] = None,
        **kwargs
    ) -> Span:
        """
        Create a span specifically for LLM operations.
        
        Args:
            name: Span name
            model: Model name
            operation: Operation type (chat, completion, embedding)
            ai_system: AI system/provider
            request: LLM request data
            **kwargs: Additional span arguments
        
        Returns:
            New Span configured for LLM tracing
        """
        attributes = kwargs.get('attributes', {})
        attributes.update({
            "llm.model": model,
            "llm.operation": operation,
            "gen_ai.system": ai_system.value,
            "gen_ai.operation.name": operation,
            "gen_ai.request.model": model
        })
        
        span = self.start_span(
            name=name,
            kind=SpanKind.CLIENT,
            attributes=attributes,
            operation_type=OperationType.LLM_CHAT if operation == "chat" else OperationType.LLM_COMPLETION,
            ai_system=ai_system,
            **{k: v for k, v in kwargs.items() if k != 'attributes'}
        )
        
        # Add LLM request data
        if request:
            span.span_data.llm_request = request
        
        return span
    
    def create_agent_span(
        self,
        name: str,
        agent_name: str,
        agent_type: str,
        agent_id: str,
        **kwargs
    ) -> Span:
        """
        Create a span specifically for agent operations.
        
        Args:
            name: Span name
            agent_name: Agent name
            agent_type: Agent type
            agent_id: Agent ID
            **kwargs: Additional span arguments
        
        Returns:
            New Span configured for agent tracing
        """
        from ..types import AgentInfo
        
        attributes = kwargs.get('attributes', {})
        attributes.update({
            "agent.name": agent_name,
            "agent.type": agent_type,
            "agent.id": agent_id
        })
        
        span = self.start_span(
            name=name,
            kind=SpanKind.INTERNAL,
            attributes=attributes,
            operation_type=OperationType.AGENT_TASK,
            **{k: v for k, v in kwargs.items() if k != 'attributes'}
        )
        
        # Add agent info
        span.span_data.agent_info = AgentInfo(
            name=agent_name,
            type=agent_type,
            id=agent_id
        )
        
        return span
    
    def create_tool_call_span(
        self,
        name: str,
        tool_name: str,
        tool_id: str,
        arguments: Dict[str, Any],
        **kwargs
    ) -> Span:
        """
        Create a span specifically for tool calls.
        
        Args:
            name: Span name
            tool_name: Tool/function name
            tool_id: Tool call ID
            arguments: Tool arguments
            **kwargs: Additional span arguments
        
        Returns:
            New Span configured for tool call tracing
        """
        from ..types import ToolCall
        
        attributes = kwargs.get('attributes', {})
        attributes.update({
            "tool.name": tool_name,
            "tool.id": tool_id,
            "tool.arguments": str(arguments)
        })
        
        span = self.start_span(
            name=name,
            kind=SpanKind.CLIENT,
            attributes=attributes,
            operation_type=OperationType.AGENT_TOOL_CALL,
            **{k: v for k, v in kwargs.items() if k != 'attributes'}
        )
        
        # Add tool call info
        tool_call = ToolCall(
            id=tool_id,
            name=tool_name,
            arguments=arguments
        )
        span.span_data.tool_calls.append(tool_call)
        
        return span

    @property
    def is_recording(self) -> bool:
        """Check if the tracer is recording."""
        return self._is_recording

    def trace_function(self, name: str):
        """Decorator to trace a function."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                span = self.start_span(name)
                try:
                    result = func(*args, **kwargs)
                    span.set_status(SpanStatus.OK)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(SpanStatus.ERROR)
                    raise
                finally:
                    span.end()
            return wrapper
        return decorator

    def shutdown(self, timeout_ms: int = 5000) -> bool:
        """Shutdown the tracer."""
        logger.info("Shutting down tracer...")
        
        # Stop recording
        self._is_recording = False
        
        # Flush any remaining spans
        success = self.flush(timeout_ms)
        
        # Stop the worker thread
        if self._worker_thread and self._worker_thread.is_alive():
            self._shutdown_event.set()
            self._worker_thread.join(timeout=timeout_ms / 1000.0)
        
        # Shutdown all sinks
        for sink in self._sinks:
            try:
                sink.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down sink {sink.name}: {e}")
        
        logger.info("Tracer shutdown complete")
        return success


# Global tracer instance
_current_tracer: Optional[NoveumTracer] = None
_tracer_lock = threading.RLock()


def set_current_tracer(tracer: Optional[NoveumTracer]):
    """Set the global current tracer."""
    global _current_tracer
    with _tracer_lock:
        _current_tracer = tracer


def get_current_tracer() -> Optional[NoveumTracer]:
    """Get the global current tracer."""
    with _tracer_lock:
        return _current_tracer


# Export main classes
__all__ = [
    "TracerConfig",
    "NoveumTracer",
    "set_current_tracer",
    "get_current_tracer",
]

