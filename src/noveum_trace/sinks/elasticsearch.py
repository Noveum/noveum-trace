"""
Elasticsearch sink for the Noveum Trace SDK.
"""

import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import ElasticsearchException, ConnectionError, RequestError
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    Elasticsearch = None
    ElasticsearchException = Exception
    ConnectionError = Exception
    RequestError = Exception

from .base import BaseSink, SinkConfig
from ..types import SpanData
from ..utils.exceptions import ElasticsearchError, ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ElasticsearchConfig(SinkConfig):
    """Configuration for Elasticsearch sink."""
    
    # Connection settings
    hosts: List[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    cloud_id: Optional[str] = None
    
    # Index settings
    index_prefix: str = "noveum-traces"
    index_template_name: str = "noveum-traces-template"
    create_index_template: bool = True
    
    # Performance settings
    bulk_chunk_size: int = 500
    bulk_max_chunk_bytes: int = 10 * 1024 * 1024  # 10MB
    refresh: Union[bool, str] = False  # 'wait_for' for immediate visibility
    
    # SSL/TLS settings
    use_ssl: bool = False
    verify_certs: bool = True
    ca_certs: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Call parent validation if it exists
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        
        if not ELASTICSEARCH_AVAILABLE:
            raise ConfigurationError(
                "Elasticsearch is not available. Install with: pip install elasticsearch"
            )
        
        if not self.hosts:
            self.hosts = ["localhost:9200"]
        
        if not self.index_prefix:
            raise ConfigurationError("index_prefix cannot be empty")
    
    def get_elasticsearch_config(self) -> Dict[str, Any]:
        """Get Elasticsearch client configuration."""
        config = {
            "hosts": self.hosts,
            "timeout": self.timeout_ms / 1000.0,
            "max_retries": 0,  # We handle retries ourselves
            "retry_on_timeout": False,
        }
        
        # Authentication
        if self.username and self.password:
            config["http_auth"] = (self.username, self.password)
        elif self.api_key:
            config["api_key"] = self.api_key
        
        # Cloud configuration
        if self.cloud_id:
            config["cloud_id"] = self.cloud_id
        
        # SSL/TLS configuration
        if self.use_ssl:
            config["use_ssl"] = True
            config["verify_certs"] = self.verify_certs
            
            if self.ca_certs:
                config["ca_certs"] = self.ca_certs
            if self.client_cert:
                config["client_cert"] = self.client_cert
            if self.client_key:
                config["client_key"] = self.client_key
        
        return config


class ElasticsearchSink(BaseSink):
    """Elasticsearch sink for storing trace data."""
    
    def __init__(self, config: ElasticsearchConfig):
        """Initialize Elasticsearch sink."""
        if not isinstance(config, ElasticsearchConfig):
            raise ConfigurationError("ElasticsearchSink requires ElasticsearchConfig")
        
        self._es_config = config
        self._client = None
        
        super().__init__(config)
    
    def _initialize(self) -> None:
        """Initialize Elasticsearch client and index template."""
        # Create Elasticsearch client
        es_config = self._es_config.get_elasticsearch_config()
        self._client = Elasticsearch(**es_config)
        
        # Test connection
        try:
            info = self._client.info()
            logger.info(f"Connected to Elasticsearch cluster: {info['cluster_name']} (version {info['version']['number']})")
        except Exception as e:
            raise ElasticsearchError(f"Failed to connect to Elasticsearch: {e}", sink_name=self.name)
        
        # Create index template if requested
        if self._es_config.create_index_template:
            self._create_index_template()
    
    def _send_batch(self, spans: List[SpanData]) -> None:
        """Send a batch of spans to Elasticsearch."""
        if not spans:
            return
        
        # Prepare bulk operations
        operations = []
        current_date = datetime.now().strftime("%Y.%m.%d")
        index_name = f"{self._es_config.index_prefix}-{current_date}"
        
        for span in spans:
            # Index operation
            operations.append({
                "index": {
                    "_index": index_name,
                    "_id": span.span_id,
                }
            })
            
            # Document data
            doc = self._prepare_document(span)
            operations.append(doc)
        
        # Send bulk request
        try:
            response = self._client.bulk(
                operations=operations,
                refresh=self._es_config.refresh,
                timeout=f"{self._es_config.timeout_ms}ms",
            )
            
            # Check for errors in bulk response
            if response.get("errors"):
                error_count = 0
                for item in response.get("items", []):
                    if "index" in item and item["index"].get("status", 200) >= 400:
                        error_count += 1
                        logger.error(f"Elasticsearch indexing error: {item['index'].get('error', 'Unknown error')}")
                
                if error_count > 0:
                    raise ElasticsearchError(
                        f"Bulk indexing failed for {error_count} out of {len(spans)} spans",
                        sink_name=self.name
                    )
            
            logger.debug(f"Successfully indexed {len(spans)} spans to Elasticsearch index '{index_name}'")
            
        except ElasticsearchException as e:
            raise ElasticsearchError(f"Elasticsearch bulk operation failed: {e}", sink_name=self.name) from e
    
    def _health_check(self) -> bool:
        """Perform Elasticsearch health check."""
        try:
            # Check cluster health
            health = self._client.cluster.health(timeout="5s")
            status = health.get("status", "red")
            
            if status == "red":
                logger.warning(f"Elasticsearch cluster health is RED")
                return False
            elif status == "yellow":
                logger.info(f"Elasticsearch cluster health is YELLOW")
                return True
            else:  # green
                return True
                
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False
    
    def _shutdown(self) -> None:
        """Shutdown Elasticsearch client."""
        if self._client:
            try:
                self._client.close()
                logger.info("Elasticsearch client closed")
            except Exception as e:
                logger.error(f"Error closing Elasticsearch client: {e}")
    
    def _create_index_template(self) -> None:
        """Create index template for trace data."""
        template_name = self._es_config.index_template_name
        index_pattern = f"{self._es_config.index_prefix}-*"
        
        template = {
            "index_patterns": [index_pattern],
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index.refresh_interval": "5s",
                    "index.codec": "best_compression",
                },
                "mappings": {
                    "properties": {
                        "span_id": {"type": "keyword"},
                        "trace_id": {"type": "keyword"},
                        "parent_span_id": {"type": "keyword"},
                        "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "kind": {"type": "keyword"},
                        "status": {"type": "keyword"},
                        "start_time": {"type": "date"},
                        "end_time": {"type": "date"},
                        "duration_ms": {"type": "float"},
                        "attributes": {
                            "type": "object",
                            "dynamic": True,
                        },
                        "events": {
                            "type": "nested",
                            "properties": {
                                "name": {"type": "keyword"},
                                "timestamp": {"type": "date"},
                                "attributes": {"type": "object", "dynamic": True},
                            }
                        },
                        "links": {
                            "type": "nested",
                            "properties": {
                                "span_id": {"type": "keyword"},
                                "trace_id": {"type": "keyword"},
                                "attributes": {"type": "object", "dynamic": True},
                            }
                        },
                        "resource": {
                            "type": "object",
                            "dynamic": True,
                        },
                        # LLM-specific fields
                        "llm": {
                            "properties": {
                                "operation_type": {"type": "keyword"},
                                "ai_system": {"type": "keyword"},
                                "model": {"type": "keyword"},
                                "input_tokens": {"type": "integer"},
                                "output_tokens": {"type": "integer"},
                                "total_tokens": {"type": "integer"},
                                "latency_ms": {"type": "float"},
                                "time_to_first_token_ms": {"type": "float"},
                            }
                        }
                    }
                }
            },
            "priority": 100,
            "version": 1,
            "_meta": {
                "description": "Template for Noveum trace data",
                "created_by": "noveum-trace-sdk",
            }
        }
        
        try:
            # Check if template exists
            if self._client.indices.exists_index_template(name=template_name):
                logger.info(f"Index template '{template_name}' already exists")
                return
            
            # Create template
            self._client.indices.put_index_template(
                name=template_name,
                body=template,
            )
            
            logger.info(f"Created Elasticsearch index template '{template_name}' for pattern '{index_pattern}'")
            
        except Exception as e:
            logger.warning(f"Failed to create index template: {e}")
            # Don't fail initialization if template creation fails
    
    def _prepare_document(self, span: SpanData) -> Dict[str, Any]:
        """Prepare span data for Elasticsearch indexing."""
        doc = span.to_dict()
        
        # Extract LLM-specific attributes for better querying
        llm_data = {}
        attributes = doc.get("attributes", {})
        
        # Extract OpenTelemetry semantic convention attributes
        for key, value in attributes.items():
            if key.startswith("gen_ai."):
                llm_key = key.replace("gen_ai.", "").replace(".", "_")
                llm_data[llm_key] = value
            elif key.startswith("llm."):
                llm_key = key.replace("llm.", "")
                llm_data[llm_key] = value
        
        if llm_data:
            doc["llm"] = llm_data
        
        # Ensure timestamps are properly formatted
        for time_field in ["start_time", "end_time"]:
            if time_field in doc and doc[time_field]:
                # Ensure ISO format with timezone
                if isinstance(doc[time_field], str):
                    try:
                        dt = datetime.fromisoformat(doc[time_field].replace('Z', '+00:00'))
                        doc[time_field] = dt.isoformat()
                    except ValueError:
                        pass  # Keep original format if parsing fails
        
        # Add indexing metadata
        doc["@timestamp"] = doc.get("start_time") or datetime.now().isoformat()
        doc["@version"] = "1"
        
        return doc

