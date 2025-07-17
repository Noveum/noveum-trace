"""
Context managers for inline tracing in Noveum Trace SDK.

This module provides context managers that allow tracing specific operations
within functions without requiring decorators on the entire function.
"""

import functools
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Optional

from noveum_trace.core.context import get_current_trace
from noveum_trace.core.span import Span


class TraceContextManager:
    """Base context manager for tracing operations."""

    def __init__(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
        auto_finish: bool = True,
    ):
        self.name = name
        self.attributes = attributes or {}
        self.tags = tags or {}
        self.auto_finish = auto_finish
        self.span: Optional[Span] = None
        self.client = None
        self.auto_trace = None

    def __enter__(self) -> Span:
        """Enter the context and start a span."""
        from noveum_trace import get_client, is_initialized

        if not is_initialized():
            # Return a no-op span if not initialized
            return NoOpSpan()

        self.client = get_client()
        if self.client is None:
            return NoOpSpan()

        trace = get_current_trace()

        # Auto-create trace if none exists
        if trace is None:
            self.auto_trace = self.client.start_trace(f"auto_trace_{self.name}")
            trace = self.auto_trace

        # Create span
        self.span = self.client.start_span(name=self.name, attributes=self.attributes)

        # Add tags if provided
        if self.tags:
            for key, value in self.tags.items():
                self.span.add_tag(key, value)

        return self.span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and finish the span."""
        if self.span and self.client and self.auto_finish:
            if exc_type is not None:
                # Record exception if one occurred
                self.span.record_exception(exc_val)
                self.span.set_status("error", str(exc_val))
            else:
                self.span.set_status("ok")

            self.client.finish_span(self.span)

        # Clean up auto-created trace
        if self.auto_trace and self.client:
            self.client.finish_trace(self.auto_trace)


class LLMContextManager(TraceContextManager):
    """Context manager specifically for LLM operations."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        operation: Optional[str] = None,
        capture_inputs: bool = True,
        capture_outputs: bool = True,
        **kwargs: Any,
    ):
        name = f"llm.{operation}" if operation else "llm_call"

        attributes = {
            "llm.model": model,
            "llm.provider": provider,
            "llm.operation": operation or "unknown",
            **kwargs.get("attributes", {}),
        }

        super().__init__(name=name, attributes=attributes, **kwargs)
        self.capture_inputs = capture_inputs
        self.capture_outputs = capture_outputs

    def set_input_attributes(self, **attributes: Any) -> None:
        """Set input-related attributes."""
        if self.span and self.capture_inputs:
            input_attrs = {f"llm.input.{k}": v for k, v in attributes.items()}
            self.span.set_attributes(input_attrs)

    def set_output_attributes(self, **attributes: Any) -> None:
        """Set output-related attributes."""
        if self.span and self.capture_outputs:
            output_attrs = {f"llm.output.{k}": v for k, v in attributes.items()}
            self.span.set_attributes(output_attrs)

    def set_usage_attributes(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> None:
        """Set usage-related attributes."""
        if self.span:
            usage_attrs: dict[str, Any] = {}
            if input_tokens is not None:
                usage_attrs["llm.input_tokens"] = input_tokens
            if output_tokens is not None:
                usage_attrs["llm.output_tokens"] = output_tokens
            if total_tokens is not None:
                usage_attrs["llm.total_tokens"] = total_tokens
            if cost is not None:
                usage_attrs["llm.cost"] = cost

            self.span.set_attributes(usage_attrs)


class AgentContextManager(TraceContextManager):
    """Context manager for agent operations."""

    def __init__(
        self,
        agent_type: Optional[str] = None,
        operation: Optional[str] = None,
        capabilities: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        name = f"agent.{operation}" if operation else "agent_operation"

        attributes = {
            "agent.type": agent_type,
            "agent.operation": operation or "unknown",
            "agent.capabilities": capabilities,
            **kwargs.get("attributes", {}),
        }

        super().__init__(name=name, attributes=attributes, **kwargs)


class OperationContextManager(TraceContextManager):
    """Generic context manager for any operation."""

    def __init__(self, operation_name: str, **kwargs: Any) -> None:
        super().__init__(name=operation_name, **kwargs)


class NoOpSpan:
    """No-operation span for when tracing is not initialized."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def add_tag(self, key: str, value: str) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        pass


# Convenience functions for creating context managers


def trace_llm(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    operation: Optional[str] = None,
    **kwargs: Any,
) -> LLMContextManager:
    """
    Create a context manager for tracing LLM operations.

    Args:
        model: LLM model name
        provider: LLM provider (openai, anthropic, etc.)
        operation: Specific operation being performed
        **kwargs: Additional attributes or configuration

    Returns:
        LLMContextManager instance

    Example:
        with trace_llm(model="gpt-4", provider="openai") as span:
            response = client.chat.completions.create(...)
            span.set_usage_attributes(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
    """
    return LLMContextManager(
        model=model, provider=provider, operation=operation, **kwargs
    )


def trace_agent(
    agent_type: Optional[str] = None,
    operation: Optional[str] = None,
    capabilities: Optional[list[str]] = None,
    **kwargs: Any,
) -> AgentContextManager:
    """
    Create a context manager for tracing agent operations.

    Args:
        agent_type: Type of agent (conversational, task, etc.)
        operation: Specific operation being performed
        capabilities: List of agent capabilities
        **kwargs: Additional attributes or configuration

    Returns:
        AgentContextManager instance

    Example:
        with trace_agent(agent_type="task_agent", operation="planning") as span:
            plan = agent.create_plan(task)
            span.set_attributes({
                "plan.steps": len(plan.steps),
                "plan.estimated_duration": plan.duration
            })
    """
    return AgentContextManager(
        agent_type=agent_type, operation=operation, capabilities=capabilities, **kwargs
    )


def trace_operation(
    operation_name: str,
    attributes: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> OperationContextManager:
    """
    Create a context manager for tracing generic operations.

    Args:
        operation_name: Name of the operation
        attributes: Operation attributes
        tags: Operation tags
        **kwargs: Additional configuration

    Returns:
        OperationContextManager instance

    Example:
        with trace_operation("database_query", {"query.table": "users"}) as span:
            results = db.query("SELECT * FROM users")
            span.set_attributes({
                "query.results_count": len(results),
                "query.duration_ms": query_duration
            })
    """
    return OperationContextManager(
        operation_name=operation_name, attributes=attributes, tags=tags, **kwargs
    )


# Advanced context managers for specific use cases


@contextmanager
def trace_batch_operation(
    operation_name: str, batch_size: int, attributes: Optional[dict[str, Any]] = None
) -> Any:
    """
    Context manager for tracing batch operations.

    Args:
        operation_name: Name of the batch operation
        batch_size: Size of the batch
        attributes: Additional attributes

    Example:
        with trace_batch_operation("batch_llm_calls", len(queries)) as span:
            results = []
            for i, query in enumerate(queries):
                with trace_llm(model="gpt-4") as llm_span:
                    result = process_query(query)
                    results.append(result)

            span.set_attributes({
                "batch.successful": len([r for r in results if r]),
                "batch.failed": len([r for r in results if not r])
            })
    """
    batch_attributes = {
        "batch.size": batch_size,
        "batch.operation": operation_name,
        **(attributes or {}),
    }

    with trace_operation(f"batch.{operation_name}", batch_attributes) as span:
        span.batch_results = []
        yield span


@contextmanager
def trace_pipeline_stage(
    stage_name: str,
    pipeline_id: Optional[str] = None,
    stage_index: Optional[int] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Context manager for tracing pipeline stages.

    Args:
        stage_name: Name of the pipeline stage
        pipeline_id: Identifier for the pipeline
        stage_index: Index of this stage in the pipeline
        attributes: Additional attributes

    Example:
        pipeline_id = "data_processing_pipeline"

        with trace_pipeline_stage("data_extraction", pipeline_id, 0) as span:
            raw_data = extract_data()
            span.set_attribute("data.records_extracted", len(raw_data))

        with trace_pipeline_stage("data_transformation", pipeline_id, 1) as span:
            processed_data = transform_data(raw_data)
            span.set_attribute("data.records_processed", len(processed_data))
    """
    stage_attributes = {
        "pipeline.id": pipeline_id,
        "pipeline.stage": stage_name,
        "pipeline.stage_index": stage_index,
        **(attributes or {}),
    }

    with trace_operation(f"pipeline.{stage_name}", stage_attributes) as span:
        yield span


# Utility functions for working with context managers


def create_child_span(
    parent_span: "Span", name: str, attributes: Optional[dict[str, Any]] = None
) -> AbstractContextManager["Span"]:
    """
    Create a child span context manager.

    Args:
        parent_span: Parent span
        name: Child span name
        attributes: Child span attributes

    Returns:
        Context manager for the child span
    """

    @contextmanager
    def child_span_context() -> Any:
        from noveum_trace import get_client

        client = get_client()
        child_span = client.start_span(
            name=name, parent_span_id=parent_span.span_id, attributes=attributes or {}
        )

        try:
            yield child_span
        except Exception as e:
            child_span.record_exception(e)
            child_span.set_status("error", str(e))
            raise
        else:
            child_span.set_status("ok")
        finally:
            client.finish_span(child_span)

    return child_span_context()


def trace_function_calls(
    func: Any,
    span_name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Decorator that uses context managers internally to trace function calls.

    Args:
        func: Function to trace
        span_name: Custom span name
        attributes: Additional attributes

    Returns:
        Traced function

    Example:
        # Trace an existing function
        traced_func = trace_function_calls(existing_function, "custom_operation")
        result = traced_func(arg1, arg2)
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        name = span_name or f"function.{func.__name__}"
        func_attributes = {
            "function.name": func.__name__,
            "function.module": func.__module__,
            "function.args_count": len(args),
            "function.kwargs_count": len(kwargs),
            **(attributes or {}),
        }

        with trace_operation(name, func_attributes) as span:
            try:
                result = func(*args, **kwargs)
                span.set_attribute("function.success", True)
                return result
            except Exception:
                span.set_attribute("function.success", False)
                raise

    return wrapper
