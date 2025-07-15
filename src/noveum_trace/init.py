"""
Simplified initialization API for Noveum Trace SDK.

This module provides easy-to-use initialization functions that match
the patterns used by leading competitors like DeepEval, Phoenix, and Braintrust.
"""

import os
from typing import Optional, Dict, Any, List, Union
import logging

from .core.tracer import NoveumTracer, TracerConfig, set_current_tracer
from .sinks.file import FileSink, FileSinkConfig
from .sinks.noveum import NoveumSink, NoveumConfig
from .sinks.console import ConsoleSink, ConsoleSinkConfig
from .sinks.base import BaseSink
from .instrumentation import openai, anthropic
from .utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Global tracer instance
_global_tracer: Optional[NoveumTracer] = None


def init(
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    project_name: Optional[str] = None,
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    environment: Optional[str] = None,
    file_logging: bool = True,
    log_directory: str = "./traces",
    auto_instrument: bool = True,
    capture_content: bool = True,
    custom_headers: Optional[Dict[str, str]] = None,
    # Elasticsearch configuration
    elasticsearch_hosts: Optional[List[str]] = None,
    elasticsearch_username: Optional[str] = None,
    elasticsearch_password: Optional[str] = None,
    elasticsearch_api_key: Optional[str] = None,
    elasticsearch_cloud_id: Optional[str] = None,
    elasticsearch_index_prefix: str = "noveum-traces",
    **kwargs
) -> NoveumTracer:
    """
    Initialize Noveum Trace SDK with simple configuration.
    
    This is the main entry point for the SDK, designed to be as simple as:
    
    ```python
    import noveum_trace
    noveum_trace.init(project_id="proj_123")
    ```
    
    Args:
        api_key: Noveum API key (can also be set via NOVEUM_API_KEY env var)
        project_id: Project identifier for organizing traces (required)
        project_name: Human-readable project name
        org_id: Organization identifier
        user_id: User identifier for the current session
        session_id: Session identifier for grouping related traces
        environment: Environment name (dev, staging, prod, etc.)
        file_logging: Whether to enable local file logging
        log_directory: Directory for local log files
        auto_instrument: Whether to automatically instrument LLM libraries
        capture_content: Whether to capture request/response content
        custom_headers: Additional headers to send with traces
        elasticsearch_hosts: List of Elasticsearch hosts (e.g., ["localhost:9200"])
        elasticsearch_username: Elasticsearch username for authentication
        elasticsearch_password: Elasticsearch password for authentication
        elasticsearch_api_key: Elasticsearch API key for authentication
        elasticsearch_cloud_id: Elasticsearch Cloud ID for Elastic Cloud
        elasticsearch_index_prefix: Prefix for Elasticsearch indices
        **kwargs: Additional configuration options
    
    Returns:
        NoveumTracer: Configured tracer instance
        
    Raises:
        ConfigurationError: If required configuration is missing
    """
    global _global_tracer
    
    # Get API key from environment if not provided
    if not api_key:
        api_key = os.getenv("NOVEUM_API_KEY")
    
    # Get project_id from environment if not provided
    if not project_id:
        project_id = os.getenv("NOVEUM_PROJECT_ID")
    
    # Project ID is required for proper trace organization
    if not project_id:
        raise ConfigurationError(
            "project_id is required. Set it via init(project_id='...') or NOVEUM_PROJECT_ID env var"
        )
    
    # Get other IDs from environment if not provided
    if not org_id:
        org_id = os.getenv("NOVEUM_ORG_ID")
    if not user_id:
        user_id = os.getenv("NOVEUM_USER_ID")
    if not session_id:
        session_id = os.getenv("NOVEUM_SESSION_ID")
    if not environment:
        environment = os.getenv("NOVEUM_ENVIRONMENT", "development")
    if not project_name:
        project_name = project_id
    
    # Build custom headers with project and org info
    headers = custom_headers or {}
    headers.update({
        "X-Noveum-Project-ID": project_id,
    })
    
    if org_id:
        headers["X-Noveum-Org-ID"] = org_id
    if user_id:
        headers["X-Noveum-User-ID"] = user_id
    if session_id:
        headers["X-Noveum-Session-ID"] = session_id
    if environment:
        headers["X-Noveum-Environment"] = environment
    
    # Create sinks
    sinks = []
    
    # File sink (always enabled for local development)
    if file_logging:
        file_config = FileSinkConfig(
            directory=log_directory,
            max_file_size_mb=100,
            max_files=10,
            compress_old_files=True
        )
        sinks.append(FileSink(file_config))
    
    # Noveum cloud sink (if API key provided)
    if api_key:
        noveum_config = NoveumConfig(
            name="noveum-sink",
            api_key=api_key,
            project_id=project_id,
            max_batch_size=100,
            max_retries=3
        )
        sinks.append(NoveumSink(noveum_config))
    
    # Elasticsearch sink (if hosts provided)
    if elasticsearch_hosts:
        try:
            from .sinks.elasticsearch import ElasticsearchSink, ElasticsearchConfig
            
            es_config = ElasticsearchConfig(
                name="elasticsearch-sink",
                hosts=elasticsearch_hosts,
                username=elasticsearch_username,
                password=elasticsearch_password,
                api_key=elasticsearch_api_key,
                cloud_id=elasticsearch_cloud_id,
                index_prefix=elasticsearch_index_prefix
            )
            sinks.append(ElasticsearchSink(es_config))
            logger.info(f"Elasticsearch sink configured with hosts: {elasticsearch_hosts}")
        except ImportError:
            logger.warning("Elasticsearch sink requested but elasticsearch package not installed. Install with: pip install elasticsearch")
        except Exception as e:
            logger.error(f"Failed to configure Elasticsearch sink: {e}")
    
    # Add console sink if no other sinks
    if not sinks:
        console_config = ConsoleSinkConfig(
            name="console-sink",
            pretty_print=True
        )
        sinks.append(ConsoleSink(console_config))
    
    # Create tracer configuration
    config = TracerConfig(
        project_id=project_id,
        project_name=project_name,
        org_id=org_id,
        user_id=user_id,
        session_id=session_id,
        environment=environment,
        sinks=sinks,
        capture_content=capture_content,
        custom_headers=headers,
        **kwargs
    )
    
    # Create and configure tracer
    tracer = NoveumTracer(config)
    _global_tracer = tracer
    set_current_tracer(tracer)
    
    # Enable auto-instrumentation if requested
    if auto_instrument:
        enable_auto_instrumentation()
    
    logger.info(f"Noveum Trace initialized for project: {project_id}")
    return tracer


def enable_auto_instrumentation() -> None:
    """Enable auto-instrumentation for supported LLM libraries."""
    instrumented = []
    
    try:
        openai.instrument_openai()
        instrumented.append("OpenAI")
    except Exception as e:
        logger.debug(f"OpenAI auto-instrumentation not available: {e}")
    
    try:
        anthropic.instrument_anthropic()
        instrumented.append("Anthropic")
    except Exception as e:
        logger.debug(f"Anthropic auto-instrumentation not available: {e}")
    
    if instrumented:
        logger.info(f"Auto-instrumentation enabled for: {', '.join(instrumented)}")
    else:
        logger.warning("No LLM libraries found for auto-instrumentation")


def disable_auto_instrumentation() -> None:
    """Disable auto-instrumentation for all LLM libraries."""
    try:
        openai.uninstrument_openai()
    except:
        pass
    
    try:
        anthropic.uninstrument_anthropic()
    except:
        pass
    
    logger.info("Auto-instrumentation disabled")


def get_tracer() -> Optional[NoveumTracer]:
    """Get the current global tracer instance."""
    return _global_tracer


def shutdown() -> None:
    """Shutdown the global tracer and cleanup resources."""
    global _global_tracer
    
    if _global_tracer:
        disable_auto_instrumentation()
        _global_tracer.shutdown()
        _global_tracer = None
        set_current_tracer(None)
        logger.info("Noveum Trace SDK shutdown complete")


def flush(timeout_ms: int = 5000) -> None:
    """Flush all pending traces."""
    if _global_tracer:
        _global_tracer.flush(timeout_ms=timeout_ms)


# Context manager for easy setup/teardown
class NoveumTrace:
    """Context manager for Noveum Trace SDK."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tracer = None
    
    def __enter__(self):
        self.tracer = init(**self.kwargs)
        return self.tracer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        shutdown()


# Convenience functions that match competitor patterns
def configure(**kwargs) -> NoveumTracer:
    """Alternative name for init() to match some competitor patterns."""
    return init(**kwargs)


def setup(**kwargs) -> NoveumTracer:
    """Alternative name for init() to match some competitor patterns."""
    return init(**kwargs)


# Export commonly used classes for advanced users
__all__ = [
    "init",
    "configure", 
    "setup",
    "enable_auto_instrumentation",
    "disable_auto_instrumentation",
    "get_tracer",
    "shutdown",
    "flush",
    "NoveumTrace",
]

