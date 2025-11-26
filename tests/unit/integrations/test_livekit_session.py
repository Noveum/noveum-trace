"""
Unit tests for LiveKit session tracing.

Tests the session tracing functionality in livekit_session.py:
- setup_livekit_tracing
- _LiveKitTracingManager
- Event serialization functions
- Event handlers
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Skip all tests if LiveKit is not available
try:
    from noveum_trace.integrations.livekit.livekit_session import (
        _LiveKitTracingManager,
        _create_event_span,
        _serialize_chat_items,
        _serialize_event_data,
        _serialize_value,
        setup_livekit_tracing,
    )

    LIVEKIT_SESSION_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    LIVEKIT_SESSION_AVAILABLE = False


@pytest.fixture
def mock_session():
    """Create a mock AgentSession."""
    session = Mock()
    session.start = AsyncMock(return_value=None)
    session.on = Mock()
    session.off = Mock()
    return session


@pytest.fixture
def mock_trace():
    """Create a mock trace."""
    trace = Mock()
    trace.trace_id = "test_trace_123"
    trace.span_id = "test_span_456"
    trace.create_span = Mock(return_value=Mock())
    return trace


@pytest.fixture
def mock_client():
    """Create a mock noveum client."""
    client = Mock()
    mock_span = Mock()
    mock_span.span_id = "test_span_789"
    mock_span.attributes = {}
    client.start_span = Mock(return_value=mock_span)
    client.finish_span = Mock()
    client.start_trace = Mock(return_value=Mock())
    client.finish_trace = Mock()
    return client


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestSerializeEventData:
    """Test _serialize_event_data function."""

    def test_serialize_event_data_none(self):
        """Test serialization of None."""
        result = _serialize_event_data(None)
        assert result == {}

    def test_serialize_event_data_dict(self):
        """Test serialization of dictionary."""
        event = {"key1": "value1", "key2": 42, "key3": True}
        result = _serialize_event_data(event)

        assert result["key1"] == "value1"
        assert result["key2"] == 42
        assert result["key3"] is True

    def test_serialize_event_data_with_prefix(self):
        """Test serialization with prefix."""
        event = {"key": "value"}
        result = _serialize_event_data(event, prefix="event")

        assert result["event.key"] == "value"

    def test_serialize_event_data_dataclass(self):
        """Test serialization of dataclass."""

        @dataclass
        class TestEvent:
            field1: str
            field2: int

        event = TestEvent(field1="test", field2=123)
        result = _serialize_event_data(event)

        assert result["field1"] == "test"
        assert result["field2"] == 123

    def test_serialize_event_data_pydantic_v2(self):
        """Test serialization of Pydantic v2 model."""
        event = Mock()
        event.model_dump = Mock(return_value={"key": "value"})

        result = _serialize_event_data(event)

        assert result["key"] == "value"
        event.model_dump.assert_called_once()

    def test_serialize_event_data_pydantic_v1(self):
        """Test serialization of Pydantic v1 model."""
        event = Mock()
        del event.model_dump  # Remove v2 method
        event.dict = Mock(return_value={"key": "value"})

        result = _serialize_event_data(event)

        assert result["key"] == "value"
        event.dict.assert_called_once()

    def test_serialize_event_data_nested_dict(self):
        """Test serialization of nested dictionary."""
        event = {"outer": {"inner": "value"}}
        result = _serialize_event_data(event)

        assert "outer.inner" in result
        assert result["outer.inner"] == "value"

    def test_serialize_event_data_list(self):
        """Test serialization of list."""
        event = {"items": ["item1", "item2", 3]}
        result = _serialize_event_data(event)

        assert result["items[0]"] == "item1"
        assert result["items[1]"] == "item2"
        assert result["items[2]"] == 3

    def test_serialize_event_data_filters_none(self):
        """Test that None values are filtered out."""
        event = {"key1": "value", "key2": None}
        result = _serialize_event_data(event)

        assert "key1" in result
        assert "key2" not in result

    def test_serialize_event_data_filters_private_attrs(self):
        """Test that private attributes (starting with _) are filtered."""
        event = Mock()
        event.__dict__ = {"public": "value", "_private": "hidden"}

        result = _serialize_event_data(event)

        assert "public" in result
        assert "_private" not in result

    def test_serialize_event_data_fallback_to_string(self):
        """Test fallback to string conversion."""
        event = 42  # Not a dict, dataclass, or object with __dict__

        result = _serialize_event_data(event)

        assert "value" in result or "42" in str(result.values())


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestSerializeValue:
    """Test _serialize_value function."""

    def test_serialize_value_none(self):
        """Test serialization of None."""
        result = _serialize_value(None)
        assert result is None

    def test_serialize_value_primitive(self):
        """Test serialization of primitive types."""
        assert _serialize_value("string") == "string"
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value(True) is True

    def test_serialize_value_dict(self):
        """Test serialization of dictionary."""
        value = {"key1": "value1", "key2": {"nested": "value2"}}
        result = _serialize_value(value, prefix="test")

        assert result["test.key1"] == "value1"
        assert result["test.key2.nested"] == "value2"

    def test_serialize_value_list(self):
        """Test serialization of list."""
        value = ["item1", "item2", 3]
        result = _serialize_value(value, prefix="test")

        assert result["test[0]"] == "item1"
        assert result["test[1]"] == "item2"
        assert result["test[2]"] == 3

    def test_serialize_value_tuple(self):
        """Test serialization of tuple."""
        value = ("item1", "item2")
        result = _serialize_value(value)

        assert "[0]" in result or "item1" in str(result.values())
        assert "[1]" in result or "item2" in str(result.values())


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestSerializeChatItems:
    """Test _serialize_chat_items function."""

    def test_serialize_chat_items_empty(self):
        """Test serialization of empty chat items."""
        result = _serialize_chat_items([])
        assert result == {}

    def test_serialize_chat_items_messages(self):
        """Test serialization of chat messages."""
        message1 = Mock()
        message1.type = "message"
        message1.role = "user"
        message1.text_content = "Hello"

        message2 = Mock()
        message2.type = "message"
        message2.role = "assistant"
        message2.content = "Hi there"

        result = _serialize_chat_items([message1, message2])

        assert result["speech.chat_items.count"] == 2
        assert "speech.messages" in result
        assert len(result["speech.messages"]) == 2

    def test_serialize_chat_items_function_calls(self):
        """Test serialization of function calls."""
        func_call = Mock()
        func_call.type = "function_call"
        func_call.name = "get_weather"
        func_call.arguments = '{"city": "NYC"}'

        result = _serialize_chat_items([func_call])

        assert result["speech.chat_items.count"] == 1
        assert "speech.function_calls" in result
        assert len(result["speech.function_calls"]) == 1

    def test_serialize_chat_items_function_outputs(self):
        """Test serialization of function outputs."""
        func_output = Mock()
        func_output.type = "function_call_output"
        func_output.name = "get_weather"
        func_output.output = "Sunny, 72F"
        func_output.is_error = False

        result = _serialize_chat_items([func_output])

        assert result["speech.chat_items.count"] == 1
        assert "speech.function_outputs" in result
        assert len(result["speech.function_outputs"]) == 1

    def test_serialize_chat_items_infers_type_from_attributes(self):
        """Test that item type is inferred from attributes if type is missing."""
        message = Mock()
        del message.type  # Remove type attribute
        message.content = "Hello"
        message.role = "user"

        result = _serialize_chat_items([message])

        assert "speech.messages" in result


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestCreateEventSpan:
    """Test _create_event_span function."""

    @patch("noveum_trace.integrations.livekit.livekit_session.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_session.get_current_span")
    def test_create_event_span_with_trace(
        self, mock_get_span, mock_get_client, mock_get_trace, mock_trace, mock_client
    ):
        """Test span creation when trace exists."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client
        mock_get_span.return_value = None  # No current span

        # Create event as a simple dict-like object
        event = type("Event", (), {"field": "value"})()

        span = _create_event_span("test_event", event)

        assert span is not None
        mock_client.start_span.assert_called_once()
        mock_client.finish_span.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_session.get_current_trace")
    def test_create_event_span_without_trace(self, mock_get_trace):
        """Test span creation when no trace exists."""
        mock_get_trace.return_value = None

        event = Mock()
        span = _create_event_span("test_event", event)

        assert span is None

    @patch("noveum_trace.integrations.livekit.livekit_session.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_session.get_current_span")
    def test_create_event_span_with_parent(
        self, mock_get_span, mock_get_client, mock_get_trace, mock_trace, mock_client
    ):
        """Test span creation with parent span."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client

        mock_parent_span = Mock()
        mock_parent_span.span_id = "parent_123"
        mock_parent_span.name = "parent_span"
        mock_get_span.return_value = mock_parent_span

        # Create event as a simple object
        event = type("Event", (), {})()

        span = _create_event_span("test_event", event)

        assert span is not None
        # Check that parent_span_id was passed
        call_args = mock_client.start_span.call_args
        assert call_args is not None
        # Verify parent_span_id was used (check both kwargs and args)
        if call_args.kwargs:
            assert call_args.kwargs.get(
                "parent_span_id"
            ) == "parent_123" or "parent_span_id" in str(call_args)


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestLiveKitTracingManager:
    """Test _LiveKitTracingManager class."""

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_init(self, mock_session):
        """Test manager initialization."""
        manager = _LiveKitTracingManager(session=mock_session)

        assert manager.session == mock_session
        assert manager.enabled is True
        assert manager.trace_name_prefix == "livekit"
        assert manager._wrapped is False

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_init_with_custom_prefix(self, mock_session):
        """Test manager initialization with custom prefix."""
        manager = _LiveKitTracingManager(
            session=mock_session, trace_name_prefix="custom"
        )

        assert manager.trace_name_prefix == "custom"

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_init_disabled(self, mock_session):
        """Test manager initialization with tracing disabled."""
        manager = _LiveKitTracingManager(session=mock_session, enabled=False)

        assert manager.enabled is False

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_wrap_start_method(self, mock_session):
        """Test wrapping of session.start() method."""
        manager = _LiveKitTracingManager(session=mock_session)
        manager._wrap_start_method()

        assert manager._wrapped is True
        assert manager.session.start != manager._original_start

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_session.set_current_trace")
    @patch("noveum_trace.integrations.livekit.livekit_session.asyncio.sleep", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_wrapped_start_creates_trace(
        self, mock_sleep, mock_set_trace, mock_get_client, mock_session, mock_client
    ):
        """Test that wrapped start() creates a trace."""
        manager = _LiveKitTracingManager(session=mock_session)
        manager._wrap_start_method()

        mock_agent = Mock()
        mock_agent.label = "TestAgent"
        mock_trace = Mock()
        mock_client.start_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client
        # Ensure original start is async
        manager._original_start = AsyncMock(return_value=None)

        await manager.session.start(mock_agent)

        mock_client.start_trace.assert_called_once()
        mock_set_trace.assert_called_once_with(mock_trace)
        assert manager._trace == mock_trace

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_wrapped_start_disabled(self, mock_session):
        """Test that wrapped start() doesn't create trace when disabled."""
        manager = _LiveKitTracingManager(session=mock_session, enabled=False)
        manager._wrap_start_method()

        # Store reference to original start before it gets wrapped
        original_start = manager._original_start
        # Ensure it's async
        original_start = AsyncMock(return_value=None)
        manager._original_start = original_start
        mock_agent = Mock()

        await manager.session.start(mock_agent)

        # Should call original start directly
        original_start.assert_called_once_with(mock_agent)

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_register_agent_session_handlers(self, mock_session):
        """Test registration of AgentSession event handlers."""
        manager = _LiveKitTracingManager(session=mock_session)
        manager._register_agent_session_handlers()

        # Check that session.on was called for each handler
        assert mock_session.on.call_count > 0
        assert len(manager._event_handlers) > 0

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_register_realtime_handlers(self, mock_session):
        """Test registration of RealtimeSession event handlers."""
        manager = _LiveKitTracingManager(session=mock_session)

        mock_realtime_session = Mock()
        mock_realtime_session.on = Mock()

        # Set _realtime_session before registering (as _setup_realtime_handlers does)
        manager._realtime_session = mock_realtime_session
        manager._register_realtime_handlers(mock_realtime_session)

        assert mock_realtime_session.on.call_count > 0
        assert len(manager._realtime_handlers) > 0
        assert manager._realtime_session == mock_realtime_session

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session._create_event_span")
    @pytest.mark.asyncio
    async def test_event_handler_creates_span(self, mock_create_span, mock_session):
        """Test that event handler creates span."""
        manager = _LiveKitTracingManager(session=mock_session)

        mock_span = Mock()
        mock_create_span.return_value = mock_span

        handler = manager._create_async_handler("test_event")
        mock_event = Mock()

        handler(mock_event)

        # Wait for async task to complete
        await asyncio.sleep(0.01)

        mock_create_span.assert_called_once_with(
            "test_event", mock_event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_cleanup(self, mock_session):
        """Test cleanup removes handlers and restores original start."""
        manager = _LiveKitTracingManager(session=mock_session)
        manager._wrap_start_method()
        manager._register_agent_session_handlers()

        original_start = manager._original_start

        manager.cleanup()

        # Check that handlers were removed
        assert mock_session.off.call_count > 0
        # Check that start method was restored
        assert manager.session.start == original_start
        assert manager._wrapped is False


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestSetupLiveKitTracing:
    """Test setup_livekit_tracing function."""

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_setup_livekit_tracing(self, mock_session):
        """Test basic setup of LiveKit tracing."""
        manager = setup_livekit_tracing(mock_session)

        assert isinstance(manager, _LiveKitTracingManager)
        assert manager.session == mock_session
        assert manager._wrapped is True
        assert mock_session.on.call_count > 0  # Handlers registered

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_setup_livekit_tracing_with_custom_prefix(self, mock_session):
        """Test setup with custom trace name prefix."""
        manager = setup_livekit_tracing(mock_session, trace_name_prefix="custom")

        assert manager.trace_name_prefix == "custom"

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_setup_livekit_tracing_disabled(self, mock_session):
        """Test setup with tracing disabled."""
        manager = setup_livekit_tracing(mock_session, enabled=False)

        assert manager.enabled is False
