# Configuration Guide

This guide covers all configuration options available in the Noveum Trace SDK.

## Initialization Methods

### 1. Simple Initialization

```python
import noveum_trace

# Minimal configuration
noveum_trace.init(service_name="my-app")
```

### 2. Detailed Configuration

```python
import noveum_trace

noveum_trace.init(
    service_name="my-application",
    environment="production",
    service_version="1.2.3",
    log_directory="./traces",
    capture_content=True,
    batch_size=100,
    batch_timeout_ms=1000,
    enable_console_output=False,
    project_id="my-project-id",
    api_key="your-api-key"
)
```

### 3. Configuration with Custom Sinks

```python
import noveum_trace
from noveum_trace.sinks import FileSink, ElasticsearchSink

# Create custom sinks
file_sink = FileSink({
    "directory": "./custom_traces",
    "name": "custom-file-sink",
    "max_file_size_mb": 100,
    "compression": True
})

es_sink = ElasticsearchSink({
    "hosts": ["localhost:9200"],
    "index_name": "traces",
    "name": "elasticsearch-sink"
})

noveum_trace.init(
    service_name="custom-app",
    sinks=[file_sink, es_sink]
)
```

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `service_name` | str | **Required** | Name of your service/application |
| `environment` | str | `"development"` | Environment (dev, staging, prod) |
| `service_version` | str | `"unknown"` | Version of your service |

### Tracing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capture_content` | bool | `True` | Capture LLM inputs/outputs |
| `batch_size` | int | `100` | Number of spans to batch |
| `batch_timeout_ms` | int | `1000` | Batch timeout in milliseconds |
| `max_span_attributes` | int | `128` | Maximum attributes per span |
| `max_span_events` | int | `128` | Maximum events per span |

### Sink Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_directory` | str | `None` | Directory for file logging |
| `enable_console_output` | bool | `True` | Enable console sink |
| `sinks` | List | `None` | Custom sink instances |

### Noveum.ai Integration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_id` | str | `None` | Noveum.ai project ID |
| `api_key` | str | `None` | Noveum.ai API key |
| `endpoint` | str | `"https://api.noveum.ai"` | Noveum.ai endpoint |

## Environment Variables

You can also configure the SDK using environment variables:

```bash
# Core configuration
export NOVEUM_SERVICE_NAME="my-app"
export NOVEUM_ENVIRONMENT="production"
export NOVEUM_SERVICE_VERSION="1.0.0"

# Tracing configuration
export NOVEUM_CAPTURE_CONTENT="true"
export NOVEUM_BATCH_SIZE="50"
export NOVEUM_BATCH_TIMEOUT_MS="500"

# File logging
export NOVEUM_LOG_DIRECTORY="./traces"

# Noveum.ai integration
export NOVEUM_PROJECT_ID="my-project"
export NOVEUM_API_KEY="your-api-key"
export NOVEUM_ENDPOINT="https://api.noveum.ai"
```

## Configuration File

Create a `noveum_trace.yaml` file:

```yaml
# Core settings
service_name: "my-application"
environment: "production"
service_version: "1.2.3"

# Tracing settings
capture_content: true
batch_size: 100
batch_timeout_ms: 1000
max_span_attributes: 128
max_span_events: 128

# File logging
log_directory: "./traces"
enable_console_output: false

# Noveum.ai integration
project_id: "my-project-id"
api_key: "${NOVEUM_API_KEY}"  # Use environment variable
endpoint: "https://api.noveum.ai"

# Sink configurations
sinks:
  - type: "file"
    name: "primary-file-sink"
    directory: "./traces"
    max_file_size_mb: 100
    compression: true
    
  - type: "elasticsearch"
    name: "es-sink"
    hosts: ["localhost:9200"]
    index_name: "traces"
    username: "elastic"
    password: "${ES_PASSWORD}"
```

Load configuration from file:

```python
import noveum_trace

# Load from YAML file
noveum_trace.init_from_config("noveum_trace.yaml")
```

## Sink Configuration

### File Sink

```python
from noveum_trace.sinks import FileSink

file_sink = FileSink({
    "name": "file-sink",
    "directory": "./traces",
    "max_file_size_mb": 100,
    "max_files": 10,
    "compression": True,
    "file_format": "jsonl"
})
```

### Console Sink

```python
from noveum_trace.sinks import ConsoleSink

console_sink = ConsoleSink({
    "name": "console-sink",
    "format": "json",  # or "pretty"
    "level": "info"
})
```

### Elasticsearch Sink

```python
from noveum_trace.sinks import ElasticsearchSink

es_sink = ElasticsearchSink({
    "name": "elasticsearch-sink",
    "hosts": ["localhost:9200"],
    "index_name": "traces",
    "username": "elastic",
    "password": "password",
    "verify_certs": True,
    "timeout": 30
})
```

### Noveum.ai Sink

```python
from noveum_trace.sinks import NoveumSink

noveum_sink = NoveumSink({
    "name": "noveum-sink",
    "project_id": "my-project",
    "api_key": "your-api-key",
    "endpoint": "https://api.noveum.ai",
    "timeout": 30,
    "retry_attempts": 3
})
```

## Performance Tuning

### Batch Configuration

For high-throughput applications:

```python
noveum_trace.init(
    service_name="high-throughput-app",
    batch_size=500,        # Larger batches
    batch_timeout_ms=2000, # Longer timeout
    capture_content=False  # Disable content capture for performance
)
```

### Memory Optimization

For memory-constrained environments:

```python
noveum_trace.init(
    service_name="memory-optimized-app",
    batch_size=50,           # Smaller batches
    batch_timeout_ms=500,    # Shorter timeout
    max_span_attributes=64,  # Fewer attributes
    max_span_events=32       # Fewer events
)
```

## Security Configuration

### Content Filtering

```python
noveum_trace.init(
    service_name="secure-app",
    capture_content=True,
    content_filter={
        "exclude_patterns": [
            r"password",
            r"api[_-]?key",
            r"secret",
            r"token"
        ],
        "max_content_length": 1000
    }
)
```

### TLS Configuration

```python
from noveum_trace.sinks import NoveumSink

noveum_sink = NoveumSink({
    "name": "secure-noveum-sink",
    "project_id": "my-project",
    "api_key": "your-api-key",
    "endpoint": "https://api.noveum.ai",
    "verify_ssl": True,
    "ssl_cert_path": "/path/to/cert.pem"
})
```

## Development vs Production

### Development Configuration

```python
noveum_trace.init(
    service_name="my-app",
    environment="development",
    enable_console_output=True,
    capture_content=True,
    batch_size=10,
    batch_timeout_ms=100
)
```

### Production Configuration

```python
noveum_trace.init(
    service_name="my-app",
    environment="production",
    enable_console_output=False,
    capture_content=False,  # For performance
    batch_size=200,
    batch_timeout_ms=2000,
    log_directory="/var/log/traces"
)
```

## Troubleshooting

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

noveum_trace.init(
    service_name="debug-app",
    enable_console_output=True
)
```

### Health Checks

Check SDK status:

```python
import noveum_trace

# Check if initialized
if noveum_trace.is_initialized():
    print("SDK is initialized")

# Get tracer stats
tracer = noveum_trace.get_tracer()
stats = tracer.get_stats()
print(f"Spans created: {stats['spans_created']}")
print(f"Spans exported: {stats['spans_exported']}")
```

## Best Practices

1. **Use environment-specific configuration**
2. **Set appropriate batch sizes for your workload**
3. **Disable content capture in production for performance**
4. **Use structured logging for better observability**
5. **Monitor SDK performance and adjust settings**
6. **Implement proper error handling**
7. **Use meaningful service and span names**

