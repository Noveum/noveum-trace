# Noveum API Backend Specification

## 🎯 Executive Summary

This document provides a comprehensive specification for the Noveum API backend (`api.noveum.ai`) that will receive traces from the Noveum Trace SDK, store them in Elasticsearch, and provide dataset APIs for NovaEval integration. The backend serves as the central hub for LLM application observability, dataset creation, and evaluation workflows.

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Data Flow Overview](#data-flow-overview)
3. [API Endpoints Specification](#api-endpoints-specification)
4. [Elasticsearch Schema Design](#elasticsearch-schema-design)
5. [Authentication & Authorization](#authentication--authorization)
6. [Dataset API for NovaEval](#dataset-api-for-novaeval)
7. [Performance & Scalability](#performance--scalability)
8. [Monitoring & Observability](#monitoring--observability)
9. [SDK Integration Updates](#sdk-integration-updates)
10. [Implementation Roadmap](#implementation-roadmap)

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Noveum Trace  │    │   Noveum API    │    │  Elasticsearch  │
│      SDK        │───▶│    Backend      │───▶│    Cluster      │
│                 │    │  (api.noveum.ai)│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Dataset API   │    │    NovaEval     │
                       │   (HF-like)     │───▶│   Framework     │
                       └─────────────────┘    └─────────────────┘
```

### Core Components

1. **Trace Ingestion Service** - Receives and validates traces from SDK
2. **Data Processing Pipeline** - Transforms and enriches trace data
3. **Elasticsearch Storage** - Stores traces with optimized indexing
4. **Dataset Generation Service** - Creates evaluation datasets from traces
5. **Dataset API** - HuggingFace-compatible dataset serving
6. **Authentication Service** - API key management and authorization
7. **Analytics Engine** - Real-time metrics and insights

## 🔄 Data Flow Overview

### 1. Trace Ingestion Flow

```
SDK Batch Request → Authentication → Validation → Processing → Elasticsearch → Response
```

### 2. Dataset Creation Flow

```
Elasticsearch Query → Data Aggregation → Format Conversion → Dataset Storage → API Serving
```

### 3. NovaEval Integration Flow

```
NovaEval Request → Dataset API → Data Retrieval → Format Conversion → Evaluation Framework
```

## 🔌 API Endpoints Specification

### Base Configuration

- **Base URL**: `https://api.noveum.ai`
- **API Version**: `v1`
- **Content-Type**: `application/json`
- **Authentication**: Bearer token (API key)

### 1. Health & Status Endpoints

#### `GET /health`
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-16T20:30:00Z",
  "version": "1.0.0",
  "services": {
    "elasticsearch": "healthy",
    "redis": "healthy",
    "database": "healthy"
  }
}
```

#### `GET /v1/status`
Detailed system status for authenticated users.

**Headers:**
```
Authorization: Bearer <api_key>
```

**Response:**
```json
{
  "status": "operational",
  "timestamp": "2025-07-16T20:30:00Z",
  "project_id": "proj_abc123",
  "usage": {
    "traces_today": 1250,
    "spans_today": 8420,
    "storage_used_mb": 156.7,
    "rate_limit_remaining": 4750
  },
  "quotas": {
    "traces_per_day": 10000,
    "storage_limit_gb": 5.0,
    "rate_limit_per_minute": 100
  }
}
```

### 2. Trace Ingestion Endpoints

#### `POST /v1/trace`
Submit a single trace (used by SDK for immediate sends).

**Headers:**
```
Authorization: Bearer <api_key>
Content-Type: application/json
```

**Request Body:**
```json
{
  "trace_id": "trace_abc123",
  "name": "user_query_processing",
  "start_time": "2025-07-16T20:25:00.000Z",
  "end_time": "2025-07-16T20:25:03.450Z",
  "duration_ms": 3450.0,
  "status": "ok",
  "status_message": null,
  "span_count": 5,
  "error_count": 0,
  "project": "my-llm-app",
  "environment": "production",
  "sdk": {
    "name": "noveum-trace-python",
    "version": "0.1.0"
  },
  "attributes": {
    "user_id": "user_456",
    "session_id": "session_789",
    "model_provider": "openai",
    "total_tokens": 1250,
    "total_cost": 0.0125
  },
  "metadata": {
    "user_id": "user_456",
    "session_id": "session_789",
    "request_id": "req_xyz789",
    "tags": {
      "environment": "production",
      "feature": "chat"
    },
    "custom_attributes": {
      "customer_tier": "premium",
      "region": "us-east-1"
    }
  },
  "spans": [
    {
      "span_id": "span_001",
      "trace_id": "trace_abc123",
      "parent_span_id": null,
      "name": "llm_call",
      "start_time": "2025-07-16T20:25:00.100Z",
      "end_time": "2025-07-16T20:25:02.800Z",
      "duration_ms": 2700.0,
      "status": "ok",
      "status_message": null,
      "attributes": {
        "llm.provider": "openai",
        "llm.model": "gpt-4",
        "llm.input_tokens": 850,
        "llm.output_tokens": 400,
        "llm.total_tokens": 1250,
        "llm.cost": 0.0125,
        "llm.temperature": 0.7,
        "llm.prompt": "What are the benefits of renewable energy?",
        "llm.response": "Renewable energy offers numerous benefits..."
      },
      "events": [
        {
          "name": "request_sent",
          "timestamp": "2025-07-16T20:25:00.150Z",
          "attributes": {
            "request_size_bytes": 1024
          }
        },
        {
          "name": "response_received",
          "timestamp": "2025-07-16T20:25:02.750Z",
          "attributes": {
            "response_size_bytes": 2048
          }
        }
      ],
      "links": []
    }
  ]
}
```

**Response (Success):**
```json
{
  "success": true,
  "trace_id": "trace_abc123",
  "message": "Trace ingested successfully",
  "timestamp": "2025-07-16T20:30:00Z",
  "processing_time_ms": 45.2
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid trace format",
    "details": {
      "field": "spans[0].start_time",
      "issue": "Invalid ISO 8601 timestamp format"
    }
  },
  "timestamp": "2025-07-16T20:30:00Z"
}
```

#### `POST /v1/traces`
Submit multiple traces in a batch (primary SDK endpoint).

**Headers:**
```
Authorization: Bearer <api_key>
Content-Type: application/json
```

**Request Body:**
```json
{
  "traces": [
    {
      // Trace object (same format as single trace)
    },
    {
      // Additional trace objects...
    }
  ],
  "timestamp": 1721160600.0,
  "batch_metadata": {
    "sdk_version": "0.1.0",
    "batch_size": 5,
    "compression": false
  }
}
```

**Response (Success):**
```json
{
  "success": true,
  "ingested_count": 5,
  "failed_count": 0,
  "message": "Batch ingested successfully",
  "timestamp": "2025-07-16T20:30:00Z",
  "processing_time_ms": 123.7,
  "trace_ids": [
    "trace_abc123",
    "trace_def456",
    "trace_ghi789"
  ]
}
```

**Response (Partial Success):**
```json
{
  "success": true,
  "ingested_count": 3,
  "failed_count": 2,
  "message": "Batch partially ingested",
  "timestamp": "2025-07-16T20:30:00Z",
  "processing_time_ms": 156.3,
  "successful_traces": [
    "trace_abc123",
    "trace_def456",
    "trace_ghi789"
  ],
  "failed_traces": [
    {
      "trace_id": "trace_jkl012",
      "error": "Invalid span hierarchy"
    },
    {
      "trace_id": "trace_mno345",
      "error": "Missing required fields"
    }
  ]
}
```

### 3. Query & Analytics Endpoints

#### `GET /v1/traces`
Query traces with filtering and pagination.

**Headers:**
```
Authorization: Bearer <api_key>
```

**Query Parameters:**
```
project: string (optional) - Filter by project
environment: string (optional) - Filter by environment
start_time: ISO8601 (optional) - Start time filter
end_time: ISO8601 (optional) - End time filter
status: string (optional) - Filter by status (ok, error, timeout)
user_id: string (optional) - Filter by user ID
session_id: string (optional) - Filter by session ID
tags: string (optional) - Filter by tags (comma-separated)
limit: integer (optional, default: 50, max: 1000)
offset: integer (optional, default: 0)
sort: string (optional, default: start_time:desc)
```

**Response:**
```json
{
  "traces": [
    {
      // Full trace objects
    }
  ],
  "pagination": {
    "total": 1250,
    "limit": 50,
    "offset": 0,
    "has_more": true
  },
  "filters_applied": {
    "project": "my-llm-app",
    "environment": "production",
    "start_time": "2025-07-16T00:00:00Z"
  }
}
```

#### `GET /v1/traces/{trace_id}`
Get a specific trace by ID.

**Response:**
```json
{
  "trace": {
    // Full trace object with all spans
  },
  "analytics": {
    "total_duration_ms": 3450.0,
    "span_breakdown": {
      "llm_calls": 2,
      "tool_calls": 1,
      "agent_decisions": 2
    },
    "cost_breakdown": {
      "total_cost": 0.0125,
      "openai_cost": 0.0125
    }
  }
}
```

#### `GET /v1/analytics/summary`
Get analytics summary for a project.

**Query Parameters:**
```
project: string (required)
time_range: string (optional, default: 24h) - 1h, 24h, 7d, 30d
granularity: string (optional, default: hour) - minute, hour, day
```

**Response:**
```json
{
  "summary": {
    "total_traces": 1250,
    "total_spans": 8420,
    "total_errors": 23,
    "error_rate": 0.018,
    "avg_duration_ms": 2340.5,
    "total_cost": 15.67,
    "unique_users": 456,
    "unique_sessions": 789
  },
  "time_series": [
    {
      "timestamp": "2025-07-16T19:00:00Z",
      "traces": 52,
      "errors": 1,
      "avg_duration_ms": 2100.3,
      "cost": 0.65
    }
  ],
  "top_errors": [
    {
      "error_type": "OpenAI API Error",
      "count": 12,
      "percentage": 52.2
    }
  ]
}
```

### 4. Dataset Management Endpoints

#### `POST /v1/datasets`
Create a new dataset from traces.

**Request Body:**
```json
{
  "name": "customer_support_conversations",
  "description": "Customer support conversations for evaluation",
  "query": {
    "project": "customer-support",
    "environment": "production",
    "start_time": "2025-07-01T00:00:00Z",
    "end_time": "2025-07-16T23:59:59Z",
    "tags": ["customer_support", "resolved"],
    "min_spans": 3,
    "status": "ok"
  },
  "format": "novaeval",
  "config": {
    "include_pii": false,
    "max_samples": 1000,
    "sample_strategy": "random",
    "validation_split": 0.2
  },
  "metadata": {
    "created_by": "user_123",
    "purpose": "model_evaluation",
    "tags": ["v1.0", "production"]
  }
}
```

**Response:**
```json
{
  "dataset_id": "dataset_abc123",
  "name": "customer_support_conversations",
  "status": "processing",
  "estimated_completion": "2025-07-16T20:35:00Z",
  "sample_count": null,
  "created_at": "2025-07-16T20:30:00Z"
}
```

#### `GET /v1/datasets`
List all datasets for a project.

**Response:**
```json
{
  "datasets": [
    {
      "dataset_id": "dataset_abc123",
      "name": "customer_support_conversations",
      "description": "Customer support conversations for evaluation",
      "status": "ready",
      "sample_count": 856,
      "format": "novaeval",
      "created_at": "2025-07-16T20:30:00Z",
      "updated_at": "2025-07-16T20:33:45Z",
      "size_mb": 12.4,
      "metadata": {
        "created_by": "user_123",
        "purpose": "model_evaluation",
        "tags": ["v1.0", "production"]
      }
    }
  ],
  "pagination": {
    "total": 15,
    "limit": 50,
    "offset": 0
  }
}
```

#### `GET /v1/datasets/{dataset_id}`
Get dataset details and metadata.

**Response:**
```json
{
  "dataset_id": "dataset_abc123",
  "name": "customer_support_conversations",
  "description": "Customer support conversations for evaluation",
  "status": "ready",
  "sample_count": 856,
  "format": "novaeval",
  "created_at": "2025-07-16T20:30:00Z",
  "updated_at": "2025-07-16T20:33:45Z",
  "size_mb": 12.4,
  "schema": {
    "fields": [
      {
        "name": "input",
        "type": "string",
        "description": "User input/query"
      },
      {
        "name": "output",
        "type": "string",
        "description": "Model response"
      },
      {
        "name": "context",
        "type": "array",
        "description": "Retrieved context documents"
      },
      {
        "name": "metadata",
        "type": "object",
        "description": "Additional trace metadata"
      }
    ]
  },
  "statistics": {
    "avg_input_length": 156.7,
    "avg_output_length": 342.1,
    "unique_users": 234,
    "date_range": {
      "start": "2025-07-01T00:00:00Z",
      "end": "2025-07-16T23:59:59Z"
    }
  },
  "query_used": {
    "project": "customer-support",
    "environment": "production",
    "tags": ["customer_support", "resolved"]
  }
}
```

### 5. HuggingFace-Compatible Dataset API

#### `GET /v1/datasets/{dataset_id}/data`
Get dataset samples in HuggingFace-compatible format.

**Query Parameters:**
```
split: string (optional, default: train) - train, validation, test
offset: integer (optional, default: 0)
length: integer (optional, default: 100, max: 10000)
format: string (optional, default: json) - json, jsonl, parquet
```

**Response (JSON):**
```json
{
  "data": [
    {
      "input": "How do I reset my password?",
      "output": "To reset your password, please follow these steps: 1. Go to the login page...",
      "context": [
        "Password reset documentation: Users can reset passwords through...",
        "Security policy: Password resets require email verification..."
      ],
      "metadata": {
        "trace_id": "trace_abc123",
        "user_id": "user_456",
        "timestamp": "2025-07-15T14:30:00Z",
        "model": "gpt-4",
        "tokens": 245,
        "cost": 0.0049,
        "duration_ms": 1850,
        "satisfaction_score": 4.5
      }
    }
  ],
  "pagination": {
    "offset": 0,
    "length": 100,
    "total": 856,
    "has_more": true
  },
  "dataset_info": {
    "dataset_id": "dataset_abc123",
    "name": "customer_support_conversations",
    "split": "train",
    "version": "1.0"
  }
}
```

#### `GET /v1/datasets/{dataset_id}/info`
Get dataset info in HuggingFace format.

**Response:**
```json
{
  "dataset_name": "customer_support_conversations",
  "description": "Customer support conversations for evaluation",
  "citation": "Noveum Dataset: customer_support_conversations (2025)",
  "homepage": "https://api.noveum.ai/datasets/dataset_abc123",
  "license": "Apache-2.0",
  "features": {
    "input": {
      "dtype": "string",
      "_type": "Value"
    },
    "output": {
      "dtype": "string",
      "_type": "Value"
    },
    "context": {
      "feature": {
        "dtype": "string",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    "metadata": {
      "_type": "Value",
      "dtype": "string"
    }
  },
  "splits": {
    "train": {
      "name": "train",
      "num_bytes": 8945632,
      "num_examples": 685,
      "dataset_name": "customer_support_conversations"
    },
    "validation": {
      "name": "validation",
      "num_bytes": 2236408,
      "num_examples": 171,
      "dataset_name": "customer_support_conversations"
    }
  },
  "download_size": 11182040,
  "dataset_size": 11182040,
  "size_in_bytes": 11182040
}
```

## 🗄️ Elasticsearch Schema Design

### Index Strategy

**Index Pattern**: `noveum-traces-{YYYY-MM}`
- Monthly indices for efficient time-based queries
- Automatic rollover and lifecycle management
- Hot/warm/cold tier optimization

### Trace Document Schema

```json
{
  "mappings": {
    "properties": {
      "trace_id": {
        "type": "keyword",
        "index": true
      },
      "name": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "start_time": {
        "type": "date",
        "format": "strict_date_optional_time"
      },
      "end_time": {
        "type": "date",
        "format": "strict_date_optional_time"
      },
      "duration_ms": {
        "type": "float"
      },
      "status": {
        "type": "keyword"
      },
      "status_message": {
        "type": "text"
      },
      "span_count": {
        "type": "integer"
      },
      "error_count": {
        "type": "integer"
      },
      "project": {
        "type": "keyword"
      },
      "environment": {
        "type": "keyword"
      },
      "sdk": {
        "properties": {
          "name": {
            "type": "keyword"
          },
          "version": {
            "type": "keyword"
          }
        }
      },
      "attributes": {
        "type": "object",
        "dynamic": true
      },
      "metadata": {
        "properties": {
          "user_id": {
            "type": "keyword"
          },
          "session_id": {
            "type": "keyword"
          },
          "request_id": {
            "type": "keyword"
          },
          "tags": {
            "type": "object",
            "dynamic": true
          },
          "custom_attributes": {
            "type": "object",
            "dynamic": true
          }
        }
      },
      "spans": {
        "type": "nested",
        "properties": {
          "span_id": {
            "type": "keyword"
          },
          "parent_span_id": {
            "type": "keyword"
          },
          "name": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword"
              }
            }
          },
          "start_time": {
            "type": "date"
          },
          "end_time": {
            "type": "date"
          },
          "duration_ms": {
            "type": "float"
          },
          "status": {
            "type": "keyword"
          },
          "attributes": {
            "type": "object",
            "dynamic": true,
            "properties": {
              "llm.provider": {
                "type": "keyword"
              },
              "llm.model": {
                "type": "keyword"
              },
              "llm.input_tokens": {
                "type": "integer"
              },
              "llm.output_tokens": {
                "type": "integer"
              },
              "llm.total_tokens": {
                "type": "integer"
              },
              "llm.cost": {
                "type": "float"
              },
              "llm.temperature": {
                "type": "float"
              }
            }
          },
          "events": {
            "type": "nested",
            "properties": {
              "name": {
                "type": "keyword"
              },
              "timestamp": {
                "type": "date"
              },
              "attributes": {
                "type": "object",
                "dynamic": true
              }
            }
          }
        }
      },
      "ingested_at": {
        "type": "date"
      },
      "processed_at": {
        "type": "date"
      }
    }
  },
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "refresh_interval": "5s",
    "index.lifecycle.name": "noveum-traces-policy",
    "index.lifecycle.rollover_alias": "noveum-traces"
  }
}
```

### Index Lifecycle Management

```json
{
  "policy": {
    "phases": {
      "hot": {
        "actions": {
          "rollover": {
            "max_size": "10GB",
            "max_age": "30d"
          },
          "set_priority": {
            "priority": 100
          }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "allocate": {
            "number_of_replicas": 0
          },
          "set_priority": {
            "priority": 50
          }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "allocate": {
            "number_of_replicas": 0
          },
          "set_priority": {
            "priority": 0
          }
        }
      },
      "delete": {
        "min_age": "365d"
      }
    }
  }
}
```

## 🔐 Authentication & Authorization

### API Key Management

**API Key Format**: `nv_<environment>_<32_char_random>`
- `nv_prod_abc123...` for production
- `nv_dev_xyz789...` for development
- `nv_test_def456...` for testing

### Authentication Flow

1. **API Key Validation**
   - Validate key format and existence
   - Check key status (active, suspended, expired)
   - Rate limit validation

2. **Project Authorization**
   - Extract project from API key
   - Validate project access permissions
   - Apply project-specific quotas

3. **Request Authorization**
   - Validate endpoint access permissions
   - Apply rate limiting per key/project
   - Log access for audit trail

### Rate Limiting

**Default Limits:**
- **Trace Ingestion**: 100 requests/minute, 10,000 traces/day
- **Query API**: 1,000 requests/hour
- **Dataset API**: 100 requests/hour

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1721161200
X-RateLimit-Retry-After: 60
```

## 📊 Dataset API for NovaEval

### Dataset Format Conversion

**From Traces to NovaEval Format:**

```python
# Input: Elasticsearch trace document
{
  "trace_id": "trace_abc123",
  "spans": [
    {
      "name": "llm_call",
      "attributes": {
        "llm.prompt": "What is renewable energy?",
        "llm.response": "Renewable energy is...",
        "llm.model": "gpt-4",
        "llm.tokens": 245
      }
    }
  ]
}

# Output: NovaEval dataset sample
{
  "input": "What is renewable energy?",
  "output": "Renewable energy is...",
  "metadata": {
    "model": "gpt-4",
    "tokens": 245,
    "trace_id": "trace_abc123",
    "timestamp": "2025-07-16T20:30:00Z"
  }
}
```

### NovaEval Integration Points

1. **Dataset Discovery**
   ```python
   # NovaEval can discover Noveum datasets
   from novaeval.datasets import load_dataset

   dataset = load_dataset("noveum://customer_support_conversations")
   ```

2. **Streaming Support**
   ```python
   # Large datasets can be streamed
   dataset = load_dataset(
       "noveum://large_dataset",
       streaming=True,
       split="train[:1000]"
   )
   ```

3. **Custom Evaluation Metrics**
   ```python
   # Noveum-specific metrics
   from novaeval.scorers import NoveumCostScorer, NoveumLatencyScorer

   evaluator = Evaluator(
       dataset=dataset,
       models=[model],
       scorers=[
           AccuracyScorer(),
           NoveumCostScorer(),  # Cost per evaluation
           NoveumLatencyScorer()  # Response time analysis
       ]
   )
   ```

### Dataset Types Supported

1. **Conversational Datasets**
   - Multi-turn conversations
   - Context-aware responses
   - Agent workflow traces

2. **RAG Evaluation Datasets**
   - Query-context-response triplets
   - Retrieval quality metrics
   - Context relevance scores

3. **Code Generation Datasets**
   - Programming problems
   - Code completion tasks
   - Execution results

4. **Agent Workflow Datasets**
   - Multi-step reasoning
   - Tool usage patterns
   - Decision trees

## ⚡ Performance & Scalability

### Ingestion Performance

**Target Metrics:**
- **Throughput**: 10,000 traces/second peak
- **Latency**: <100ms p95 for batch ingestion
- **Availability**: 99.9% uptime SLA

**Scaling Strategy:**
- Horizontal scaling with load balancers
- Async processing with message queues
- Elasticsearch cluster auto-scaling
- CDN for dataset API responses

### Storage Optimization

**Compression:**
- Gzip compression for API responses
- Elasticsearch field compression
- Parquet format for large datasets

**Caching:**
- Redis for frequently accessed data
- CDN for static dataset files
- Query result caching (5-minute TTL)

### Database Performance

**Elasticsearch Optimization:**
- Index templates with proper mappings
- Shard allocation based on time ranges
- Aggregation caching for analytics
- Search result pagination

## 📈 Monitoring & Observability

### Application Metrics

**Key Metrics to Track:**
- Ingestion rate (traces/second)
- Processing latency (p50, p95, p99)
- Error rates by endpoint
- Storage utilization
- Query performance
- Dataset generation time

### Health Checks

**Service Health:**
- Elasticsearch cluster health
- Redis connectivity
- Database connections
- External API dependencies

**Data Quality:**
- Trace validation errors
- Schema compliance rates
- Data completeness scores
- Duplicate detection

### Alerting

**Critical Alerts:**
- Ingestion pipeline failures
- High error rates (>5%)
- Storage capacity warnings (>80%)
- Query performance degradation

**Warning Alerts:**
- Rate limit approaching
- Unusual traffic patterns
- Data quality issues
- Long-running dataset generation

## 🔧 SDK Integration Updates

### Required SDK Enhancements

#### 1. Enhanced Error Handling

```python
# Current SDK enhancement needed
class HttpTransport:
    def _send_batch(self, traces):
        try:
            response = self.session.post(url, json=payload)

            # Enhanced error handling
            if response.status_code == 413:
                # Payload too large - split batch
                return self._split_and_retry(traces)
            elif response.status_code == 429:
                # Rate limited - implement backoff
                return self._handle_rate_limit(response)
            elif response.status_code >= 500:
                # Server error - retry with exponential backoff
                return self._retry_with_backoff(traces)

        except requests.exceptions.Timeout:
            # Implement timeout handling
            return self._handle_timeout(traces)
```

#### 2. Compression Support

```python
# Add compression to SDK
class HttpTransport:
    def _compress_payload(self, payload):
        if self.config.transport.compression:
            import gzip
            import json

            json_data = json.dumps(payload).encode('utf-8')
            compressed = gzip.compress(json_data)

            # Only use compression if beneficial
            if len(compressed) < len(json_data) * 0.8:
                return {
                    "compressed": True,
                    "data": base64.b64encode(compressed).decode('utf-8')
                }

        return payload
```

#### 3. Batch Size Optimization

```python
# Dynamic batch sizing
class BatchProcessor:
    def __init__(self):
        self.optimal_batch_size = 50
        self.last_response_time = None

    def adjust_batch_size(self, response_time, success_rate):
        if success_rate > 0.95 and response_time < 100:
            # Increase batch size
            self.optimal_batch_size = min(200, self.optimal_batch_size * 1.2)
        elif success_rate < 0.9 or response_time > 500:
            # Decrease batch size
            self.optimal_batch_size = max(10, self.optimal_batch_size * 0.8)
```

#### 4. Metadata Enhancement

```python
# Enhanced metadata collection
class Trace:
    def __init__(self, name, **kwargs):
        # Add automatic metadata collection
        self.attributes.update({
            "runtime.platform": platform.platform(),
            "runtime.python_version": platform.python_version(),
            "process.pid": os.getpid(),
            "host.name": socket.gethostname(),
            "deployment.environment": os.getenv("DEPLOYMENT_ENV", "unknown")
        })
```

### Configuration Updates

```python
# Enhanced configuration options
@dataclass
class TransportConfig:
    endpoint: str = "https://api.noveum.ai"
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_backoff: float = 1.0
    batch_size: int = 50
    max_queue_size: int = 1000
    compression: bool = True
    compression_threshold: float = 0.8  # Only compress if >20% reduction
    adaptive_batching: bool = True
    max_batch_size: int = 200
    min_batch_size: int = 10
```

## 🗺️ Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-4)

**Week 1-2: Basic API Setup**
- [ ] Set up FastAPI/Flask application
- [ ] Implement basic authentication
- [ ] Create health check endpoints
- [ ] Set up Elasticsearch cluster
- [ ] Design initial index schema

**Week 3-4: Trace Ingestion**
- [ ] Implement `/v1/trace` endpoint
- [ ] Implement `/v1/traces` batch endpoint
- [ ] Add request validation
- [ ] Set up async processing pipeline
- [ ] Implement basic error handling

### Phase 2: Data Processing & Storage (Weeks 5-8)

**Week 5-6: Elasticsearch Integration**
- [ ] Implement trace document indexing
- [ ] Set up index lifecycle management
- [ ] Add search and query capabilities
- [ ] Implement data retention policies

**Week 7-8: Analytics & Querying**
- [ ] Implement `/v1/traces` query endpoint
- [ ] Add analytics aggregations
- [ ] Create summary statistics
- [ ] Implement real-time metrics

### Phase 3: Dataset Management (Weeks 9-12)

**Week 9-10: Dataset Creation**
- [ ] Implement dataset generation from traces
- [ ] Add format conversion (NovaEval, HF)
- [ ] Create dataset storage system
- [ ] Implement dataset metadata management

**Week 11-12: Dataset API**
- [ ] Implement HuggingFace-compatible API
- [ ] Add streaming support for large datasets
- [ ] Implement dataset versioning
- [ ] Add dataset sharing capabilities

### Phase 4: NovaEval Integration (Weeks 13-16)

**Week 13-14: Integration Layer**
- [ ] Create NovaEval dataset loader
- [ ] Implement custom evaluation metrics
- [ ] Add Noveum-specific scorers
- [ ] Test end-to-end integration

**Week 15-16: Advanced Features**
- [ ] Implement real-time evaluation
- [ ] Add automated dataset updates
- [ ] Create evaluation pipelines
- [ ] Add result visualization

### Phase 5: Production Readiness (Weeks 17-20)

**Week 17-18: Performance & Scaling**
- [ ] Implement caching layers
- [ ] Add load balancing
- [ ] Optimize database queries
- [ ] Add compression support

**Week 19-20: Monitoring & Security**
- [ ] Set up comprehensive monitoring
- [ ] Implement security hardening
- [ ] Add audit logging
- [ ] Create operational runbooks

## 📋 Success Metrics

### Technical Metrics
- **Ingestion Latency**: <100ms p95
- **Query Response Time**: <500ms p95
- **Uptime**: >99.9%
- **Data Loss Rate**: <0.01%

### Business Metrics
- **Dataset Creation Time**: <5 minutes for 1000 samples
- **NovaEval Integration**: Seamless dataset loading
- **User Adoption**: SDK usage growth
- **Cost Efficiency**: <$0.001 per trace processed

## 🎯 Conclusion

This specification provides a comprehensive blueprint for building the Noveum API backend that will serve as the central hub for LLM application observability and evaluation. The system is designed to:

1. **Scale efficiently** with growing trace volumes
2. **Integrate seamlessly** with NovaEval for evaluation workflows
3. **Provide rich analytics** for application insights
4. **Support diverse use cases** from development to production

The phased implementation approach ensures rapid time-to-value while building toward a robust, production-ready platform that can support the entire LLM application lifecycle from development through evaluation and optimization.

---

**Document Version**: 1.0
**Last Updated**: 2025-07-16
**Next Review**: 2025-08-16
