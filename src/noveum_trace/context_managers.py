"""
Context managers for inline tracing in Noveum Trace SDK.

This module provides context managers that allow tracing specific operations
within functions without requiring decorators on the entire function.
"""

import functools
import inspect
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Callable, Optional, Union

from noveum_trace.core.context import attach_context_to_span, get_current_trace
from noveum_trace.core.span import Span, SpanStatus
from noveum_trace.decorators.base import _serialize_value
from noveum_trace.utils.llm_utils import (
    extract_llm_metadata,
    estimate_cost,
)


class TraceContextManager:
    """Base context manager for tracing operations."""

    def __init__(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
        auto_finish: bool = True,
        capture_args: bool = False,
    ):
        self.name = name
        self.attributes = attributes or {}
        self.tags = tags or {}
        self.auto_finish = auto_finish
        self.capture_args = capture_args
        self.span: Optional[Span] = None
        self.client: Optional[Any] = None
        self.auto_trace: Optional[Any] = None

    def __enter__(self) -> Union[Span, "NoOpSpan"]:
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
                self.span.set_attribute(f"tag.{key}", value)

        # Attach context attributes to span
        attach_context_to_span(self.span)

        return self.span

    def capture_function_args(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """
        Capture function arguments and set them as span attributes.

        This method should be called when the context manager is used to wrap
        a function call, to automatically capture the function's arguments.

        Args:
            func: The function being called
            *args: Positional arguments passed to the function
            **kwargs: Keyword arguments passed to the function
        """
        if not self.span or not self.capture_args:
            return

        try:
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Capture individual arguments
            for param_name, param_value in bound_args.arguments.items():
                serialized_value = _serialize_value(param_value)
                self.span.set_attribute(f"input.{param_name}", serialized_value)

            # Also capture args and kwargs as structured data
            if args:
                serialized_args = _serialize_value(args)
                self.span.set_attribute("input.args", serialized_args)

            if kwargs:
                serialized_kwargs = _serialize_value(kwargs)
                self.span.set_attribute("input.kwargs", serialized_kwargs)

        except Exception as e:
            # If argument capture fails, at least record the error
            self.span.set_attribute("input.capture_error", str(e))
            # Fallback: capture raw args/kwargs as strings
            try:
                if args:
                    self.span.set_attribute("input.args", _serialize_value(args))
                if kwargs:
                    self.span.set_attribute("input.kwargs", _serialize_value(kwargs))
            except Exception:
                pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and finish the span."""
        if self.span and self.client and self.auto_finish:
            if exc_type is not None:
                # Record exception if one occurred
                self.span.record_exception(exc_val)
                self.span.set_status(SpanStatus.ERROR, str(exc_val))
            else:
                self.span.set_status(SpanStatus.OK)

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

    def __enter__(self) -> "LLMContextManager":
        """Enter the context and start a span, returning self for method access."""
        # Call parent to set up span
        span = super().__enter__()
        # Return self so users can call capture_response() and other methods
        return self

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
        """Set usage-related attributes and automatically calculate costs."""
        if self.span:
            usage_attrs: dict[str, Any] = {}
            if input_tokens is not None:
                usage_attrs["llm.input_tokens"] = input_tokens
            if output_tokens is not None:
                usage_attrs["llm.output_tokens"] = output_tokens
            if total_tokens is not None:
                usage_attrs["llm.total_tokens"] = total_tokens
            elif input_tokens is not None and output_tokens is not None:
                # Calculate total if not provided
                usage_attrs["llm.total_tokens"] = input_tokens + output_tokens

            # Automatically calculate costs if tokens are provided and cost not explicitly set
            if cost is None and (input_tokens is not None or output_tokens is not None):
                model = self.span.attributes.get("llm.model")
                if model:
                    try:
                        cost_info = estimate_cost(
                            model=model,
                            input_tokens=input_tokens or 0,
                            output_tokens=output_tokens or 0,
                        )
                        usage_attrs["llm.cost.input"] = cost_info.get("input_cost", 0)
                        usage_attrs["llm.cost.output"] = cost_info.get("output_cost", 0)
                        usage_attrs["llm.cost.total"] = cost_info.get("total_cost", 0)
                        usage_attrs["llm.cost.currency"] = cost_info.get("currency", "USD")
                    except Exception:
                        # If cost calculation fails, just set the single cost value if provided
                        pass
            elif cost is not None:
                # Legacy single cost value
                usage_attrs["llm.cost"] = cost

            self.span.set_attributes(usage_attrs)

    def capture_response(self, response: Any) -> None:
        """
        Automatically extract metadata, token usage, and costs from LLM response.

        This method extracts all relevant information from LLM response objects
        (OpenAI, Anthropic, Google, etc.) and sets span attributes matching
        the LangChain integration format.

        Args:
            response: LLM response object (OpenAI, Anthropic, Google, etc.)

        Example:
            with trace_llm(model="gpt-4", operation="ocr_cleaning") as span:
                response = client.chat.completions.create(...)
                span.capture_response(response)  # Automatically extracts tokens, costs, etc.
        """
        if not self.span:
            return

        try:
            # Extract metadata from response using the utility function
            metadata = extract_llm_metadata(response)

            # Prepare attributes to set
            attributes_to_set: dict[str, Any] = {}

            # Update model and provider if extracted from response
            if "llm.model" in metadata:
                attributes_to_set["llm.model"] = metadata["llm.model"]
            if "llm.provider" in metadata:
                attributes_to_set["llm.provider"] = metadata["llm.provider"]

            # Extract and flatten token usage (matching LangChain format)
            input_tokens = (
                metadata.get("llm.usage.input_tokens")
                or metadata.get("llm.usage.prompt_tokens")
            )
            output_tokens = (
                metadata.get("llm.usage.output_tokens")
                or metadata.get("llm.usage.completion_tokens")
            )
            total_tokens = metadata.get("llm.usage.total_tokens")

            # Set token usage attributes (flattened format like LangChain)
            if input_tokens is not None:
                attributes_to_set["llm.input_tokens"] = input_tokens
            if output_tokens is not None:
                attributes_to_set["llm.output_tokens"] = output_tokens
            if total_tokens is not None:
                attributes_to_set["llm.total_tokens"] = total_tokens
            elif input_tokens is not None and output_tokens is not None:
                attributes_to_set["llm.total_tokens"] = input_tokens + output_tokens

            # Calculate costs if we have tokens and model
            model = attributes_to_set.get("llm.model") or self.span.attributes.get(
                "llm.model"
            )
            if model and (input_tokens is not None or output_tokens is not None):
                try:
                    cost_info = estimate_cost(
                        model=model,
                        input_tokens=input_tokens or 0,
                        output_tokens=output_tokens or 0,
                    )
                    attributes_to_set["llm.cost.input"] = cost_info.get("input_cost", 0)
                    attributes_to_set["llm.cost.output"] = cost_info.get("output_cost", 0)
                    attributes_to_set["llm.cost.total"] = cost_info.get("total_cost", 0)
                    attributes_to_set["llm.cost.currency"] = cost_info.get("currency", "USD")
                except Exception:
                    # If cost calculation fails, continue without cost info
                    pass

            # Add other metadata attributes
            if "llm.context_window" in metadata:
                attributes_to_set["llm.context_window"] = metadata["llm.context_window"]
            if "llm.max_output_tokens" in metadata:
                attributes_to_set["llm.max_output_tokens"] = metadata[
                    "llm.max_output_tokens"
                ]
            if "llm.finish_reason" in metadata:
                attributes_to_set["llm.finish_reason"] = metadata["llm.finish_reason"]
            if "llm.system_fingerprint" in metadata:
                attributes_to_set["llm.system_fingerprint"] = metadata[
                    "llm.system_fingerprint"
                ]
            if "llm.created" in metadata:
                attributes_to_set["llm.created"] = metadata["llm.created"]

            # Set all attributes at once
            if attributes_to_set:
                self.span.set_attributes(attributes_to_set)

        except Exception:
            # If extraction fails, silently continue
            # This ensures the context manager doesn't break user code
            pass


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

        # Extract attributes from kwargs first to avoid parameter conflict
        incoming_attributes = kwargs.pop("attributes", {})

        attributes = {
            "agent.type": agent_type,
            "agent.operation": operation or "unknown",
            "agent.capabilities": capabilities,
            **incoming_attributes,
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

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any, message: Optional[str] = None) -> None:
        pass

    def capture_response(self, response: Any) -> None:
        """No-op version of capture_response for when tracing is not initialized."""
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
            # Automatically extract tokens, costs, and model info
            span.capture_response(response)
            
        # Or manually set usage attributes (costs calculated automatically)
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
    capture_args: bool = False,
    **kwargs: Any,
) -> OperationContextManager:
    """
    Create a context manager for tracing generic operations.

    Args:
        operation_name: Name of the operation
        attributes: Operation attributes
        tags: Operation tags
        capture_args: Whether to capture function arguments if used as decorator (default: False)
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
        operation_name=operation_name,
        attributes=attributes,
        tags=tags,
        capture_args=capture_args,
        **kwargs,
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
        span.set_attribute("batch.results", [])
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
            child_span.set_status(SpanStatus.ERROR, str(e))
            raise
        else:
            child_span.set_status(SpanStatus.OK)
        finally:
            client.finish_span(child_span)

    return child_span_context()


def trace_function_calls(
    func: Optional[Any] = None,
    *,
    span_name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
    capture_args: bool = True,
) -> Any:
    """
    Decorator that uses context managers internally to trace function calls.

    Can be used as:
        @trace_function_calls
        def my_func():
            pass

        @trace_function_calls(capture_args=False)
        def my_func():
            pass

    Args:
        func: Function to trace (when used as @trace_function_calls)
        span_name: Custom span name
        attributes: Additional attributes
        capture_args: Whether to capture function arguments (default: True)

    Returns:
        Traced function or decorator

    Example:
        # Trace an existing function
        traced_func = trace_function_calls(existing_function, span_name="custom_operation")
        result = traced_func(arg1, arg2)
    """
    # Handle being called with keyword arguments only
    if func is None:
        # Called as @trace_function_calls(capture_args=True)
        def decorator(f: Any) -> Any:
            return trace_function_calls(
                f, span_name=span_name, attributes=attributes, capture_args=capture_args
            )

        return decorator

    # Handle being called with function as first argument
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

        # Create context manager with capture_args enabled
        context_mgr = trace_operation(name, func_attributes, capture_args=capture_args)

        with context_mgr as span:
            # Capture function arguments if enabled
            if capture_args:
                context_mgr.capture_function_args(func, *args, **kwargs)

            try:
                result = func(*args, **kwargs)
                span.set_attribute("function.success", True)

                # Capture result if it's a simple type
                try:
                    serialized_result = _serialize_value(result)
                    span.set_attribute("output.result", serialized_result)
                except Exception:
                    pass  # Silently ignore result serialization errors

                return result
            except Exception:
                span.set_attribute("function.success", False)
                raise

    return wrapper
