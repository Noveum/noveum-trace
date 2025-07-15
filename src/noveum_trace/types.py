"""
Type definitions for the Noveum Trace SDK.

This module contains all the data structures and enums used throughout the SDK.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass, field
import json


class SpanKind(Enum):
    """OpenTelemetry-compatible span kinds."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Span status values."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class OperationType(Enum):
    """Types of operations that can be traced."""
    LLM_CHAT = "llm.chat"
    LLM_COMPLETION = "llm.completion"
    LLM_EMBEDDING = "llm.embedding"
    AGENT_TASK = "agent.task"
    AGENT_TOOL_CALL = "agent.tool_call"
    FUNCTION_CALL = "function.call"
    RETRIEVAL = "retrieval"
    CUSTOM = "custom"
    
    # Backward compatibility aliases
    CHAT = LLM_CHAT
    COMPLETION = LLM_COMPLETION
    EMBEDDING = LLM_EMBEDDING


class AISystem(Enum):
    """AI systems/providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class TraceID:
    """Unique trace identifier with proper formatting."""
    
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"TraceID('{self.value}')"
    
    @classmethod
    def generate(cls) -> 'TraceID':
        """Generate a new trace ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'TraceID':
        """Create trace ID from string value."""
        return cls(value=value)


@dataclass
class SpanID:
    """Unique span identifier with proper formatting."""
    
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"SpanID('{self.value}')"
    
    @classmethod
    def generate(cls) -> 'SpanID':
        """Generate a new span ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'SpanID':
        """Create span ID from string value."""
        return cls(value=value)


@dataclass
class Message:
    """Represents a message in an LLM conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "role": self.role,
            "content": self.content
        }
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class TokenUsage:
    """Token usage information for LLM calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary representation."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class LLMRequest:
    """Represents an LLM request."""
    model: str
    messages: List[Message] = field(default_factory=list)
    
    # Legacy parameters for backward compatibility
    operation_type: Optional[OperationType] = None
    ai_system: Optional[AISystem] = None
    prompt: Optional[str] = None
    
    # Standard parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Post-initialization to handle prompt conversion."""
        # Convert prompt to messages if provided
        if self.prompt and not self.messages:
            self.messages = [Message(role="user", content=self.prompt)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
            "stream": self.stream
        }
        
        # Add optional parameters if present
        optional_fields = [
            "temperature", "max_tokens", "top_p", "frequency_penalty",
            "presence_penalty", "stop", "tools", "tool_choice"
        ]
        
        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                result[field] = value
        
        return result


@dataclass
class LLMResponse:
    """Represents an LLM response."""
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[TokenUsage] = None
    created: Optional[int] = None
    system_fingerprint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "model": self.model,
            "choices": self.choices
        }
        
        if self.usage:
            result["usage"] = self.usage.to_dict()
        if self.created:
            result["created"] = self.created
        if self.system_fingerprint:
            result["system_fingerprint"] = self.system_fingerprint
        
        return result


@dataclass
class ToolCall:
    """Represents a tool/function call."""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments
        }
        
        if self.result is not None:
            result["result"] = self.result
        if self.error:
            result["error"] = self.error
        if self.duration_ms:
            result["duration_ms"] = self.duration_ms
        
        return result


@dataclass
class AgentInfo:
    """Information about an agent."""
    name: str
    type: str
    id: str
    capabilities: Optional[Set[str]] = None
    tags: Optional[Set[str]] = None
    parent_agent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "type": self.type,
            "id": self.id
        }
        
        if self.capabilities:
            result["capabilities"] = list(self.capabilities)
        if self.tags:
            result["tags"] = list(self.tags)
        if self.parent_agent_id:
            result["parent_agent_id"] = self.parent_agent_id
        
        return result


@dataclass
class SpanData:
    """Complete span data structure with all OpenTelemetry fields."""
    
    # Core identifiers
    trace_id: TraceID
    span_id: SpanID
    parent_span_id: Optional[SpanID] = None
    
    # Basic span information
    name: str = "unknown"
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None
    
    # Timing information
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Attributes and metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    
    # Project and organization context
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    org_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    environment: Optional[str] = None
    
    # Agent context
    agent_info: Optional[AgentInfo] = None
    
    # LLM-specific data
    llm_request: Optional[LLMRequest] = None
    llm_response: Optional[LLMResponse] = None
    
    # Tool call data
    tool_calls: List[ToolCall] = field(default_factory=list)
    
    # Custom data
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        result = {
            "trace_id": str(self.trace_id),
            "span_id": str(self.span_id),
            "name": self.name,
            "kind": self.kind.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "attributes": self.attributes,
            "events": self.events,
            "links": self.links
        }
        
        # Add optional fields
        if self.parent_span_id:
            result["parent_span_id"] = str(self.parent_span_id)
        if self.status_message:
            result["status_message"] = self.status_message
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        if self.duration_ms:
            result["duration_ms"] = self.duration_ms
        
        # Add context fields
        context_fields = [
            "project_id", "project_name", "org_id", "user_id", 
            "session_id", "environment"
        ]
        for field in context_fields:
            value = getattr(self, field)
            if value:
                result[field] = value
        
        # Add agent info
        if self.agent_info:
            result["agent"] = self.agent_info.to_dict()
        
        # Add LLM data
        if self.llm_request:
            result["llm_request"] = self.llm_request.to_dict()
        if self.llm_response:
            result["llm_response"] = self.llm_response.to_dict()
        
        # Add tool calls
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        
        # Add custom data
        if self.custom_data:
            result["custom_data"] = self.custom_data
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        event = {
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {}
        }
        self.events.append(event)
    
    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None):
        """Set span status."""
        self.status = status
        if message:
            self.status_message = message


@dataclass
class CustomHeaders:
    """Custom headers for trace data."""
    project_id: str
    org_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_headers: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for HTTP headers."""
        headers = {
            "X-Noveum-Project-ID": self.project_id
        }
        
        if self.org_id:
            headers["X-Noveum-Org-ID"] = self.org_id
        if self.user_id:
            headers["X-Noveum-User-ID"] = self.user_id
        if self.session_id:
            headers["X-Noveum-Session-ID"] = self.session_id
        
        # Add additional headers
        headers.update(self.additional_headers)
        
        return headers


# Export all types
__all__ = [
    "SpanKind",
    "SpanStatus", 
    "OperationType",
    "AISystem",
    "TraceID",
    "SpanID",
    "Message",
    "TokenUsage",
    "LLMRequest",
    "LLMResponse",
    "ToolCall",
    "AgentInfo",
    "SpanData",
    "CustomHeaders",
]

