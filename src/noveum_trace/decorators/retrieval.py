"""
Retrieval-specific trace decorator for Noveum Trace SDK.

This module provides the @trace_retrieval decorator for tracing
retrieval operations in RAG (Retrieval-Augmented Generation) systems.
"""

import functools
import inspect
from typing import Any, Callable, Optional, Union

from noveum_trace.core.context import attach_context_to_span
from noveum_trace.core.span import SpanStatus
from noveum_trace.decorators.base import _serialize_value


def trace_retrieval(
    func: Optional[Callable[..., Any]] = None,
    *,
    retrieval_type: str = "search",
    name: Optional[str] = None,
    index_name: Optional[str] = None,
    capture_query: bool = True,
    capture_results: bool = True,
    capture_scores: bool = True,
    capture_metadata: bool = True,
    max_results: int = 50,
    metadata: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, str]] = None,
) -> Union[Callable[..., Any], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """
    Decorator to add retrieval-specific tracing to functions.

    This decorator automatically captures retrieval operation metadata
    including queries, results, relevance scores, and retrieval context.

    Args:
        func: Function to decorate (when used as @trace_retrieval)
        retrieval_type: Type of retrieval (e.g., "vector_search", "keyword_search")
        name: Custom span name (defaults to function name)
        index_name: Name of the index being searched
        capture_query: Whether to capture the search query
        capture_results: Whether to capture retrieval results
        capture_scores: Whether to capture relevance scores
        capture_metadata: Whether to capture result metadata
        max_results: Maximum number of results to capture (default: 50)
        metadata: Additional metadata to attach to the span
        tags: Tags to add to the span

    Returns:
        Decorated function or decorator

    Example:
        >>> @trace_retrieval(index_name="documents")
        >>> def search_documents(query: str, top_k: int = 5) -> list:
        ...     # Vector search implementation
        ...     return search_results

        >>> @trace_retrieval(
        ...     retrieval_type="hybrid_search",
        ...     capture_scores=True,
        ...     capture_metadata=True,
        ...     max_results=100
        ... )
        >>> def hybrid_search(query: str, filters: dict) -> list:
        ...     # Hybrid search implementation
        ...     return results
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from noveum_trace import get_client, is_initialized
            from noveum_trace.core.context import get_current_trace

            if not is_initialized():
                return func(*args, **kwargs)

            client = get_client()

            # Auto-create trace if none exists
            auto_created_trace = False
            current_trace = get_current_trace()
            if current_trace is None:
                auto_created_trace = True
                current_trace = client.start_trace(
                    name=f"auto_trace_{func_name}",
                    attributes={
                        "auto_created": True,
                        "function": func_name,
                        "type": "retrieval_operation",
                    },
                )

            # Create span attributes
            attributes = {
                "function.name": func_name,
                "function.module": func.__module__,
                "function.type": "retrieval_operation",
                "retrieval.type": retrieval_type,
                "retrieval.operation": func_name,
            }

            if index_name:
                attributes["retrieval.index_name"] = index_name

            # Add metadata and tags
            if metadata:
                for key, value in metadata.items():
                    attributes[f"metadata.{key}"] = value
            if tags:
                for key, value in tags.items():
                    attributes[f"tag.{key}"] = value

            # Capture retrieval parameters
            try:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Extract retrieval parameters
                retrieval_params = _extract_retrieval_params(bound_args.arguments)
                attributes.update(retrieval_params)

                # Capture query if enabled
                if capture_query:
                    query = _extract_query(bound_args.arguments)
                    if query:
                        attributes["retrieval.query"] = query

            except Exception as e:
                attributes["retrieval.extraction_error"] = str(e)

            # Start the span
            span = client.start_span(
                name=f"retrieval:{retrieval_type}:{func_name}",
                attributes=attributes,
            )

            attach_context_to_span(span)

            try:
                result = func(*args, **kwargs)

                # Capture retrieval results
                if capture_results or capture_scores or capture_metadata:
                    try:
                        result_data = _extract_retrieval_results(
                            result,
                            capture_results=capture_results,
                            capture_scores=capture_scores,
                            capture_metadata=capture_metadata,
                            max_results=max_results,
                        )
                        span.set_attributes(result_data)
                    except Exception as e:
                        span.set_attribute("retrieval.result_extraction_error", str(e))

                span.set_status(SpanStatus.OK)
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(SpanStatus.ERROR, str(e))
                span.set_attributes(
                    {
                        "retrieval.error.type": type(e).__name__,
                        "retrieval.error.message": str(e),
                        "retrieval.type": retrieval_type,
                    }
                )
                raise

            finally:
                client.finish_span(span)

                # Finish auto-created trace
                if auto_created_trace and current_trace:
                    client.finish_trace(current_trace)

        wrapper._noveum_traced = True  # type: ignore
        wrapper._noveum_trace_type = "retrieval"  # type: ignore
        wrapper._noveum_retrieval_type = retrieval_type  # type: ignore

        return wrapper

    # Handle both @trace_retrieval and @trace_retrieval() usage
    if func is None:
        # Called as @trace_retrieval() with arguments
        return decorator
    else:
        # Called as @trace_retrieval without arguments
        return decorator(func)


def _extract_retrieval_params(arguments: dict[str, Any]) -> dict[str, Any]:
    """Extract retrieval parameters from function arguments."""
    params = {}

    # Common retrieval parameter names
    param_mappings = {
        "top_k": "retrieval.top_k",
        "limit": "retrieval.limit",
        "threshold": "retrieval.threshold",
        "filters": "retrieval.filters",
        "embedding_model": "retrieval.embedding_model",
        "similarity_metric": "retrieval.similarity_metric",
    }

    for arg_name, arg_value in arguments.items():
        if arg_name in param_mappings:
            params[param_mappings[arg_name]] = _serialize_value(arg_value)

    return params


def _extract_query(arguments: dict[str, Any]) -> Optional[str]:
    """Extract query from function arguments."""
    # Common query parameter names
    query_params = ["query", "search_query", "text", "question", "prompt"]

    for param_name, param_value in arguments.items():
        if param_name.lower() in query_params:
            return _serialize_value(param_value)

    return None


def _extract_retrieval_results(
    result: Any,
    capture_results: bool = True,
    capture_scores: bool = True,
    capture_metadata: bool = True,
    max_results: int = 50,
) -> dict[str, Any]:
    """
    Extract retrieval result metadata.

    Args:
        result: The retrieval results to extract metadata from
        capture_results: Whether to capture the actual result content
        capture_scores: Whether to capture relevance scores
        capture_metadata: Whether to capture result metadata
        max_results: Maximum number of results to capture (default: 50, prevents oversized traces)

    Returns:
        Dictionary containing extracted metadata
    """
    result_data: dict[str, Any] = {}

    # Basic result information
    result_data["retrieval.result_count"] = _get_result_count(result)
    result_data["retrieval.result_type"] = type(result).__name__

    if isinstance(result, list):
        # Handle list of results
        if capture_results and result:
            # Capture results with configurable limit to prevent performance issues
            result_data["retrieval.sample_results"] = [
                _serialize_value(r) for r in result[:max_results]
            ]
            if len(result) > max_results:
                result_data["retrieval.results_truncated"] = True
                result_data["retrieval.total_results"] = len(result)
            else:
                result_data["retrieval.results_truncated"] = False

        if capture_scores:
            scores = _extract_scores_from_list(result)
            if scores:
                result_data["retrieval.scores"] = scores
                result_data["retrieval.max_score"] = max(scores)
                result_data["retrieval.min_score"] = min(scores)
                result_data["retrieval.avg_score"] = sum(scores) / len(scores)

        if capture_metadata:
            metadata_list = _extract_metadata_from_list(result)
            if metadata_list:
                result_data["retrieval.result_metadata"] = metadata_list

    elif isinstance(result, dict):
        # Handle dictionary result
        if "results" in result:
            result_data["retrieval.result_count"] = len(result["results"])
            if capture_results:
                # Capture results with configurable limit to prevent performance issues
                result_data["retrieval.sample_results"] = [
                    _serialize_value(r) for r in result["results"][:max_results]
                ]
                if len(result["results"]) > max_results:
                    result_data["retrieval.results_truncated"] = True
                    result_data["retrieval.total_results"] = len(result["results"])
                else:
                    result_data["retrieval.results_truncated"] = False

        if capture_scores and "scores" in result:
            scores = result["scores"]
            if scores is not None:
                result_data["retrieval.scores"] = scores
                if scores:
                    result_data["retrieval.max_score"] = max(scores)
                    result_data["retrieval.min_score"] = min(scores)
                    result_data["retrieval.avg_score"] = sum(scores) / len(scores)

        if capture_metadata and "metadata" in result:
            result_data["retrieval.result_metadata"] = _serialize_value(
                result["metadata"]
            )

    return result_data


def _get_result_count(result: Any) -> int:
    """Get the count of results."""
    if isinstance(result, list):
        return len(result)
    elif isinstance(result, dict) and "results" in result:
        return len(result["results"])
    elif hasattr(result, "__len__"):
        try:
            return len(result)
        except (TypeError, AttributeError):
            return 0
    else:
        return 1 if result is not None else 0


def _extract_scores_from_list(results: list[Any]) -> Optional[list[float]]:
    """Extract relevance scores from a list of results."""
    scores = []

    for result in results:
        if isinstance(result, dict):
            # Look for common score field names
            for score_field in ["score", "relevance", "similarity", "confidence"]:
                if score_field in result:
                    try:
                        scores.append(float(result[score_field]))
                        break
                    except (ValueError, TypeError):
                        continue
        elif isinstance(result, tuple) and len(result) >= 2:
            # Handle (document, score) tuples
            try:
                scores.append(float(result[1]))
            except (ValueError, TypeError):
                continue

    return scores if scores else None


def _extract_metadata_from_list(results: list[Any]) -> Optional[list[dict[str, Any]]]:
    """Extract metadata from a list of results."""
    metadata_list = []

    for result in results:
        if isinstance(result, dict):
            # Extract metadata fields
            metadata = {}
            for key, value in result.items():
                if key.startswith("metadata") or key in [
                    "source",
                    "document_id",
                    "chunk_id",
                ]:
                    metadata[key] = _serialize_value(value)

            if metadata:
                metadata_list.append(metadata)

    return metadata_list if metadata_list else None
