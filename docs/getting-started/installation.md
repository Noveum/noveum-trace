# Installation Guide

This guide will help you install and set up the Noveum Trace SDK in your Python environment.

## Requirements

- Python 3.8 or higher
- pip package manager

## Installation Methods

### 1. Install from PyPI (Recommended)

```bash
pip install noveum-trace
```

### 2. Install from Source

```bash
git clone https://github.com/Noveum/noveum-trace.git
cd noveum-trace
pip install -e .
```

### 3. Development Installation

For development and contributing:

```bash
git clone https://github.com/Noveum/noveum-trace.git
cd noveum-trace
pip install -e ".[dev]"
```

## Verify Installation

After installation, verify that the SDK is working correctly:

```python
import noveum_trace

# Initialize with console output for testing
noveum_trace.init(service_name="test-app")

# Create a simple trace
with noveum_trace.get_tracer().start_span("test-span") as span:
    span.set_attribute("test.attribute", "test-value")
    print("✅ Noveum Trace SDK is working!")

# Shutdown
noveum_trace.shutdown()
```

## Optional Dependencies

### For Elasticsearch Integration

```bash
pip install noveum-trace[elasticsearch]
```

### For OpenTelemetry Integration

```bash
pip install noveum-trace[otel]
```

### For All Features

```bash
pip install noveum-trace[all]
```

## Environment Setup

### API Keys

If you plan to use the Noveum.ai sink (coming soon), set your API key:

```bash
export NOVEUM_API_KEY="your-api-key-here"
```

### Configuration File

Create a configuration file `noveum_trace.yaml`:

```yaml
service_name: "my-application"
environment: "production"
log_directory: "./traces"
capture_content: true
batch_size: 100
batch_timeout_ms: 1000
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you have Python 3.8+ and the package is installed correctly
2. **Permission Error**: Ensure write permissions for the log directory
3. **Network Error**: Check firewall settings if using remote sinks

### Getting Help

- Check the [FAQ](../guides/faq.md)
- Review [examples](../examples/)
- Open an issue on [GitHub](https://github.com/Noveum/noveum-trace/issues)

## Next Steps

- Read the [Quick Start Guide](quickstart.md)
- Explore [Configuration Options](../guides/configuration.md)
- Try the [Examples](../examples/)
