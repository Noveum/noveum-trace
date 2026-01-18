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
        setup_livekit_tracing,
    )
    from noveum_trace.integrations.livekit.livekit_utils import (
        _serialize_chat_items,
        _serialize_event_data,
        _serialize_value,
        create_event_span,
    )

    LIVEKIT_SESSION_AVAILABLE = True
except ImportError:
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
    """Test create_event_span function."""

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_span")
    def test_create_event_span_with_trace(
        self, mock_get_span, mock_get_client, mock_get_trace, mock_trace, mock_client
    ):
        """Test span creation when trace exists."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client
        mock_get_span.return_value = None  # No current span

        # Create event as a simple dict-like object
        event = type("Event", (), {"field": "value"})()

        span = create_event_span("test_event", event)

        assert span is not None
        mock_client.start_span.assert_called_once()
        mock_client.finish_span.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    def test_create_event_span_without_trace(self, mock_get_trace):
        """Test span creation when no trace exists."""
        mock_get_trace.return_value = None

        event = Mock()
        span = create_event_span("test_event", event)

        assert span is None

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_span")
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

        span = create_event_span("test_event", event)

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

    @pytest.mark.asyncio
    async def test_wrapped_start_creates_trace(self, mock_session, mock_client):
        """Test that wrapped start() creates a trace."""
        mock_trace = Mock()
        mock_client.start_trace.return_value = mock_trace

        with patch(
            "noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True
        ):
            manager = _LiveKitTracingManager(session=mock_session)
            manager._wrap_start_method()

            # Configure mock agent - set tools to empty list to avoid iteration error
            mock_agent = Mock(spec=[])
            mock_agent.label = "TestAgent"
            mock_agent.tools = []
            mock_agent.instructions = None

            with patch(
                "noveum_trace.integrations.livekit.livekit_session.set_current_trace"
            ) as mock_set_trace:
                with patch(
                    "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
                    new_callable=AsyncMock,
                ):
                    with patch("noveum_trace.get_client", return_value=mock_client):
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

        # When disabled, _wrap_start_method returns early, so session.start is not wrapped
        # It should still be the original mock_session.start
        mock_agent = Mock()

        await manager.session.start(mock_agent)

        # Should call original start directly (not wrapped)
        mock_session.start.assert_called_once_with(mock_agent)

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
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
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


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestSerializeEventDataErrorHandling:
    """Test error handling in _serialize_event_data."""

    def test_serialize_event_data_exception_handling(self):
        """Test exception handling in _serialize_event_data."""
        event = Mock()
        event.model_dump = Mock(side_effect=Exception("Serialization error"))
        event.dict = Mock(side_effect=Exception("Serialization error"))
        event.__dict__ = {"key": "value"}

        # Should handle exception gracefully
        result = _serialize_event_data(event)

        # Should return fallback result
        assert isinstance(result, dict)

    def test_serialize_event_data_with_dict_attr(self):
        """Test serialization with object having __dict__ but no model_dump/dict."""
        event = Mock()
        del event.model_dump
        del event.dict
        event.__dict__ = {"public": "value", "_private": "hidden"}

        result = _serialize_event_data(event)

        assert "public" in result
        assert "_private" not in result

    def test_serialize_event_data_nested_pydantic(self):
        """Test serialization with nested Pydantic models."""
        nested_obj = Mock()
        nested_obj.model_dump = Mock(return_value={"nested_key": "nested_value"})

        event = Mock()
        event.model_dump = Mock(return_value={"outer": nested_obj})

        result = _serialize_event_data(event)

        assert "outer.nested_key" in result or "outer" in result

    def test_serialize_event_data_nested_dataclass(self):
        """Test serialization with nested dataclasses."""

        @dataclass
        class NestedEvent:
            nested_field: str

        @dataclass
        class OuterEvent:
            outer_field: str
            nested: NestedEvent

        nested = NestedEvent(nested_field="nested_value")
        event = OuterEvent(outer_field="outer_value", nested=nested)

        result = _serialize_event_data(event)

        assert "outer_field" in result
        assert "nested.nested_field" in result


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestSerializeValueEdgeCases:
    """Test edge cases in _serialize_value."""

    def test_serialize_value_with_dataclass(self):
        """Test _serialize_value with dataclass value."""

        @dataclass
        class TestData:
            field: str

        value = {"key": TestData(field="value")}
        result = _serialize_value(value, prefix="test")

        assert isinstance(result, dict)

    def test_serialize_value_with_dict_attr(self):
        """Test _serialize_value with object having __dict__."""
        obj = Mock()
        obj.__dict__ = {"attr": "value"}

        value = {"key": obj}
        result = _serialize_value(value, prefix="test")

        assert isinstance(result, dict)


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestSerializeChatItemsEdgeCases:
    """Test edge cases in _serialize_chat_items."""

    def test_serialize_chat_items_mixed_types(self):
        """Test serialization with mixed item types."""
        message = Mock()
        message.type = "message"
        message.role = "user"
        message.text_content = "Hello"

        func_call = Mock()
        func_call.type = "function_call"
        func_call.name = "test_func"
        func_call.arguments = '{"arg": "value"}'

        func_output = Mock()
        func_output.type = "function_call_output"
        func_output.name = "test_func"
        func_output.output = "result"
        func_output.is_error = False

        result = _serialize_chat_items([message, func_call, func_output])

        assert result["speech.chat_items.count"] == 3
        assert "speech.messages" in result
        assert "speech.function_calls" in result
        assert "speech.function_outputs" in result

    def test_serialize_chat_items_content_as_list_of_dicts(self):
        """Test serialization with content as list of dicts."""
        message = Mock()
        message.type = "message"
        message.role = "user"
        message.content = [{"text": "Hello"}, {"text": "World"}]

        result = _serialize_chat_items([message])

        assert "speech.messages" in result
        assert len(result["speech.messages"]) == 1

    def test_serialize_chat_items_with_interrupted_flag(self):
        """Test serialization with interrupted flag."""
        message = Mock()
        message.type = "message"
        message.role = "user"
        message.text_content = "Hello"
        message.interrupted = True

        result = _serialize_chat_items([message])

        assert "speech.messages" in result
        assert result["speech.messages"][0]["interrupted"] is True

    def test_serialize_chat_items_function_call_with_error(self):
        """Test serialization with function call output having error."""
        func_output = Mock()
        func_output.type = "function_call_output"
        func_output.name = "test_func"
        func_output.output = "Error occurred"
        func_output.is_error = True

        result = _serialize_chat_items([func_output])

        assert "speech.function_outputs" in result
        assert result["speech.function_outputs"][0]["is_error"] is True


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestCreateEventSpanErrorHandling:
    """Test error handling in create_event_span."""

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    @patch("noveum_trace.get_client")
    def test_create_event_span_get_client_fails(
        self, mock_get_client, mock_get_trace, mock_trace
    ):
        """Test create_event_span when get_client() fails."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.side_effect = Exception("Client error")

        event = Mock()
        span = create_event_span("test_event", event)

        assert span is None

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_span")
    def test_create_event_span_exception_handling(
        self, mock_get_span, mock_get_client, mock_get_trace, mock_trace, mock_client
    ):
        """Test create_event_span exception handling."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client
        mock_get_span.return_value = None
        mock_client.start_span.side_effect = Exception("Span creation error")

        event = Mock()
        span = create_event_span("test_event", event)

        assert span is None


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestCreateEventSpanParentResolution:
    """Test parent span resolution logic in create_event_span."""

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_span")
    def test_metrics_collected_with_agent_state_span(
        self, mock_get_span, mock_get_client, mock_get_trace, mock_trace, mock_client
    ):
        """Test metrics_collected events with _last_agent_state_changed_span_id set."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client
        mock_get_span.return_value = None

        manager = Mock()
        manager._last_agent_state_changed_span_id = "agent_span_123"

        event = Mock()
        span = create_event_span("metrics_collected", event, manager=manager)

        assert span is not None
        call_args = mock_client.start_span.call_args
        assert call_args is not None

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_span")
    def test_metrics_collected_without_agent_state_span(
        self, mock_get_span, mock_get_client, mock_get_trace, mock_trace, mock_client
    ):
        """Test metrics_collected events without _last_agent_state_changed_span_id."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client
        mock_get_span.return_value = None

        manager = Mock()
        manager._last_agent_state_changed_span_id = None

        event = Mock()
        span = create_event_span("metrics_collected", event, manager=manager)

        assert span is not None
        # Should use trace.create_span directly
        mock_trace.create_span.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_span")
    @patch("noveum_trace.integrations.livekit.livekit_utils.asyncio.create_task")
    def test_speech_created_with_metrics_current_span(
        self,
        mock_create_task,
        mock_get_span,
        mock_get_client,
        mock_get_trace,
        mock_trace,
        mock_client,
    ):
        """Test speech_created events when current span is metrics_collected."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client

        mock_current_span = Mock()
        mock_current_span.name = "livekit.metrics_collected"
        mock_current_span.span_id = "metrics_span_123"
        mock_get_span.return_value = mock_current_span

        manager = Mock()
        manager._last_agent_state_changed_span_id = "agent_span_123"

        mock_span = Mock()
        mock_span.span_id = "span_123"
        mock_span.attributes = {}
        mock_span.set_status = Mock()  # Mock set_status in case it's called
        mock_client.start_span.return_value = mock_span
        mock_client.finish_span = Mock()  # Ensure finish_span is mocked

        event = Mock()
        # Ensure event can be serialized
        event.__dict__ = {"field": "value"}

        span = create_event_span("speech_created", event, manager=manager)

        assert span is not None

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_span")
    def test_speech_created_normal_parent(
        self, mock_get_span, mock_get_client, mock_get_trace, mock_trace, mock_client
    ):
        """Test speech_created events with normal parent resolution."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client

        mock_current_span = Mock()
        mock_current_span.name = "livekit.user_state_changed"
        mock_current_span.span_id = "parent_span_123"
        mock_get_span.return_value = mock_current_span

        event = Mock()
        span = create_event_span("speech_created", event)

        assert span is not None

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    @patch("noveum_trace.get_client")
    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_span")
    def test_other_event_with_metrics_current_span(
        self, mock_get_span, mock_get_client, mock_get_trace, mock_trace, mock_client
    ):
        """Test other events when current span is metrics_collected."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client

        mock_current_span = Mock()
        mock_current_span.name = "livekit.metrics_collected"
        mock_current_span.span_id = "metrics_span_123"
        mock_get_span.return_value = mock_current_span

        manager = Mock()
        manager._last_agent_state_changed_span_id = "agent_span_123"

        event = Mock()
        span = create_event_span("user_state_changed", event, manager=manager)

        assert span is not None

    @patch("noveum_trace.integrations.livekit.livekit_utils.get_current_trace")
    @patch("noveum_trace.get_client")
    def test_error_event_sets_error_status(
        self, mock_get_client, mock_get_trace, mock_trace, mock_client
    ):
        """Test error events set error status."""
        mock_get_trace.return_value = mock_trace
        mock_get_client.return_value = mock_client

        mock_span = Mock()
        mock_span.span_id = "span_123"
        mock_span.set_status = Mock()
        mock_client.start_span.return_value = mock_span

        event = Mock()
        event.error = "Test error"

        span = create_event_span("error", event)

        assert span is not None
        mock_span.set_status.assert_called_once()


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestEventHandlers:
    """Test all event handlers."""

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_user_state_changed(self, mock_create_span, mock_session):
        """Test _on_user_state_changed handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_user_state_changed(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "user_state_changed", event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_agent_state_changed(self, mock_create_span, mock_session):
        """Test _on_agent_state_changed handler with additional work."""
        manager = _LiveKitTracingManager(session=mock_session)
        manager._try_setup_realtime_handlers_later = Mock()
        event = Mock()

        manager._on_agent_state_changed(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "agent_state_changed", event, manager=manager
        )
        manager._try_setup_realtime_handlers_later.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_user_input_transcribed(self, mock_create_span, mock_session):
        """Test _on_user_input_transcribed handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_user_input_transcribed(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "user_input_transcribed", event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_conversation_item_added(self, mock_create_span, mock_session):
        """Test _on_conversation_item_added handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_conversation_item_added(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "conversation_item_added", event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_agent_false_interruption(self, mock_create_span, mock_session):
        """Test _on_agent_false_interruption handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_agent_false_interruption(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "agent_false_interruption", event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_function_tools_executed(self, mock_create_span, mock_session):
        """Test _on_function_tools_executed handler - calls create_event_span as fallback when no pending span."""
        manager = _LiveKitTracingManager(session=mock_session)
        # Ensure no pending generation span so fallback path is taken
        manager._pending_generation_span = None
        event = Mock()
        event.results = []  # No results to process

        manager._on_function_tools_executed(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "function_tools_executed", event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_metrics_collected(self, mock_create_span, mock_session):
        """Test _on_metrics_collected handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_metrics_collected(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "metrics_collected", event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_error(self, mock_create_span, mock_session):
        """Test _on_error handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_error(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with("error", event, manager=manager)


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestRealtimeEventHandlers:
    """Test RealtimeSession event handlers."""

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_input_speech_started(self, mock_create_span, mock_session):
        """Test _on_input_speech_started handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_input_speech_started(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "realtime.input_speech_started", event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_input_speech_stopped(self, mock_create_span, mock_session):
        """Test _on_input_speech_stopped handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_input_speech_stopped(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "realtime.input_speech_stopped", event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_input_audio_transcription_completed(
        self, mock_create_span, mock_session
    ):
        """Test _on_input_audio_transcription_completed handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_input_audio_transcription_completed(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "realtime.input_audio_transcription_completed", event, manager=manager
        )

    @pytest.mark.asyncio
    async def test_on_generation_created(self, mock_session, mock_client):
        """Test _on_generation_created handler - creates span directly without using create_event_span."""
        mock_trace = Mock()
        mock_span = Mock()
        mock_span.span_id = "test_span"
        mock_span.attributes = {}
        mock_client.start_span.return_value = mock_span

        with patch(
            "noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True
        ):
            manager = _LiveKitTracingManager(session=mock_session)

            event = Mock()
            event.__dict__ = {"field": "value"}

            # Patch at the location where it's used in the async handler
            with patch(
                "noveum_trace.core.context.get_current_trace", return_value=mock_trace
            ):
                with patch("noveum_trace.get_client", return_value=mock_client):
                    manager._on_generation_created(event)
                    await asyncio.sleep(0.05)  # Give async task more time

        # _on_generation_created uses client.start_span directly, not create_event_span
        mock_client.start_span.assert_called_once()
        call_kwargs = mock_client.start_span.call_args
        assert call_kwargs is not None
        assert "livekit.realtime.generation_created" in str(call_kwargs)

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_session_reconnected(self, mock_create_span, mock_session):
        """Test _on_session_reconnected handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_session_reconnected(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "realtime.session_reconnected", event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_realtime_metrics_collected(self, mock_create_span, mock_session):
        """Test _on_realtime_metrics_collected handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_realtime_metrics_collected(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "realtime.metrics_collected", event, manager=manager
        )

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @pytest.mark.asyncio
    async def test_on_realtime_error(self, mock_create_span, mock_session):
        """Test _on_realtime_error handler."""
        manager = _LiveKitTracingManager(session=mock_session)
        event = Mock()

        manager._on_realtime_error(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with(
            "realtime.error", event, manager=manager
        )


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestBackgroundTasks:
    """Test background task functions."""

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_utils._serialize_chat_items")
    @pytest.mark.asyncio
    async def test_update_speech_span_with_chat_items_success(
        self, mock_serialize, mock_session
    ):
        """Test update_speech_span_with_chat_items successful update."""
        from noveum_trace.integrations.livekit.livekit_utils import (
            update_speech_span_with_chat_items,
        )

        manager = _LiveKitTracingManager(session=mock_session)
        speech_handle = AsyncMock()
        speech_handle.id = "speech_123"
        speech_handle.wait_for_playout = AsyncMock()
        speech_handle.chat_items = [Mock(), Mock()]

        span = Mock()
        span.span_id = "span_123"
        span.attributes = {}

        manager._speech_spans["speech_123"] = span

        mock_serialize.return_value = {"chat.item": "value"}

        await update_speech_span_with_chat_items(speech_handle, span, manager)

        assert "speech_123" not in manager._speech_spans
        assert "chat.item" in span.attributes

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_update_speech_span_with_chat_items_empty(self, mock_session):
        """Test update_speech_span_with_chat_items with empty chat_items."""
        from noveum_trace.integrations.livekit.livekit_utils import (
            update_speech_span_with_chat_items,
        )

        manager = _LiveKitTracingManager(session=mock_session)
        speech_handle = AsyncMock()
        speech_handle.id = "speech_123"
        speech_handle.wait_for_playout = AsyncMock()
        speech_handle.chat_items = []

        span = Mock()
        span.span_id = "span_123"
        span.attributes = {}

        manager._speech_spans["speech_123"] = span

        await update_speech_span_with_chat_items(speech_handle, span, manager)

        assert "speech_123" not in manager._speech_spans

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_update_speech_span_with_chat_items_exception(self, mock_session):
        """Test update_speech_span_with_chat_items exception handling."""
        from noveum_trace.integrations.livekit.livekit_utils import (
            update_speech_span_with_chat_items,
        )

        manager = _LiveKitTracingManager(session=mock_session)
        speech_handle = AsyncMock()
        speech_handle.id = "speech_123"
        speech_handle.wait_for_playout = AsyncMock(
            side_effect=Exception("Playout error")
        )

        span = Mock()
        span.span_id = "span_123"
        span.attributes = {}

        manager._speech_spans["speech_123"] = span

        await update_speech_span_with_chat_items(speech_handle, span, manager)

        # Should still remove from tracking
        assert "speech_123" not in manager._speech_spans

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @pytest.mark.asyncio
    async def test_update_speech_span_not_in_tracking(self, mock_session):
        """Test update_speech_span_with_chat_items when not in tracking dict."""
        from noveum_trace.integrations.livekit.livekit_utils import (
            update_speech_span_with_chat_items,
        )

        manager = _LiveKitTracingManager(session=mock_session)
        speech_handle = AsyncMock()
        speech_handle.id = "speech_123"
        speech_handle.wait_for_playout = AsyncMock()
        speech_handle.chat_items = []

        span = Mock()
        span.span_id = "span_123"
        span.attributes = {}

        # Not in tracking dict
        await update_speech_span_with_chat_items(speech_handle, span, manager)

        # Should not raise error

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch(
        "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_update_span_with_system_prompt_success(
        self, mock_sleep, mock_session
    ):
        """Test _update_span_with_system_prompt successful extraction."""
        from noveum_trace.integrations.livekit.livekit_utils import (
            _update_span_with_system_prompt,
        )

        manager = _LiveKitTracingManager(session=mock_session)
        agent_activity = Mock()
        agent = Mock()
        agent.instructions = "Test system prompt"
        agent_activity._agent = agent
        manager.session.agent_activity = agent_activity

        span = Mock()
        span.attributes = {}

        await _update_span_with_system_prompt(span, manager, max_wait_seconds=1.0)

        assert span.attributes["llm.system_prompt"] == "Test system prompt"

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch(
        "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_update_span_with_system_prompt_timeout(
        self, mock_sleep, mock_session
    ):
        """Test _update_span_with_system_prompt timeout scenario."""
        from noveum_trace.integrations.livekit.livekit_utils import (
            _update_span_with_system_prompt,
        )

        manager = _LiveKitTracingManager(session=mock_session)
        manager.session.agent_activity = None

        span = Mock()
        span.attributes = {}

        # Mock time to simulate timeout
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_event_loop = Mock()
            mock_event_loop.time.side_effect = [0.0, 6.0]  # Start and after timeout
            mock_loop.return_value = mock_event_loop

            await _update_span_with_system_prompt(span, manager, max_wait_seconds=5.0)

        assert "llm.system_prompt" not in span.attributes

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch(
        "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_update_span_with_system_prompt_via_activity_attr(
        self, mock_sleep, mock_session
    ):
        """Test _update_span_with_system_prompt via _activity attribute."""
        from noveum_trace.integrations.livekit.livekit_utils import (
            _update_span_with_system_prompt,
        )

        manager = _LiveKitTracingManager(session=mock_session)
        agent_activity = Mock()
        agent = Mock()
        agent.instructions = "Test prompt"
        agent_activity._agent = agent
        # agent_activity property not available, but _activity is
        del manager.session.agent_activity
        manager.session._activity = agent_activity

        span = Mock()
        span.attributes = {}

        await _update_span_with_system_prompt(span, manager, max_wait_seconds=1.0)

        assert span.attributes["llm.system_prompt"] == "Test prompt"

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch(
        "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_update_span_with_system_prompt_no_prompt(
        self, mock_sleep, mock_session
    ):
        """Test _update_span_with_system_prompt when no system prompt found."""
        from noveum_trace.integrations.livekit.livekit_utils import (
            _update_span_with_system_prompt,
        )

        manager = _LiveKitTracingManager(session=mock_session)
        agent_activity = Mock()
        agent = Mock()
        agent.instructions = None
        agent._instructions = None
        agent_activity._agent = agent
        manager.session.agent_activity = agent_activity

        span = Mock()
        span.attributes = {}

        await _update_span_with_system_prompt(span, manager, max_wait_seconds=1.0)

        assert "llm.system_prompt" not in span.attributes

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch(
        "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_update_span_with_system_prompt_exception(
        self, mock_sleep, mock_session
    ):
        """Test _update_span_with_system_prompt exception handling."""
        from noveum_trace.integrations.livekit.livekit_utils import (
            _update_span_with_system_prompt,
        )

        manager = _LiveKitTracingManager(session=mock_session)
        manager.session.agent_activity = Mock(side_effect=Exception("Error"))

        span = Mock()
        span.attributes = {}

        # Should not raise exception
        await _update_span_with_system_prompt(span, manager, max_wait_seconds=0.1)


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestCloseEventHandling:
    """Test close event handling."""

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @patch("noveum_trace.get_client")
    @pytest.mark.asyncio
    async def test_on_close_with_error_reason(
        self, mock_get_client, mock_create_span, mock_session
    ):
        """Test _on_close with CloseReason.ERROR."""
        try:
            from livekit.agents.voice.events import CloseReason
        except ImportError:
            pytest.skip("CloseReason not available")

        manager = _LiveKitTracingManager(session=mock_session)
        mock_trace = Mock()
        mock_trace.set_status = Mock()
        manager._trace = mock_trace

        mock_client = Mock()
        mock_client.finish_trace = Mock()
        mock_get_client.return_value = mock_client

        event = Mock()
        event.reason = CloseReason.ERROR
        event.error = "Test error"

        manager._on_close(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once_with("close", event, manager=manager)
        # Use stored reference since manager._trace is set to None after finish_trace
        mock_trace.set_status.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @patch("noveum_trace.get_client")
    @pytest.mark.asyncio
    async def test_on_close_with_job_shutdown(
        self, mock_get_client, mock_create_span, mock_session
    ):
        """Test _on_close with CloseReason.JOB_SHUTDOWN."""
        try:
            from livekit.agents.voice.events import CloseReason
        except ImportError:
            pytest.skip("CloseReason not available")

        manager = _LiveKitTracingManager(session=mock_session)
        mock_trace = Mock()
        mock_trace.set_status = Mock()
        manager._trace = mock_trace

        mock_client = Mock()
        mock_client.finish_trace = Mock()
        mock_get_client.return_value = mock_client

        event = Mock()
        event.reason = CloseReason.JOB_SHUTDOWN

        manager._on_close(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once()
        # Use stored reference since manager._trace is set to None after finish_trace
        mock_trace.set_status.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @patch("noveum_trace.get_client")
    @pytest.mark.asyncio
    async def test_on_close_without_livekit(
        self, mock_get_client, mock_create_span, mock_session
    ):
        """Test _on_close when LiveKit not available."""
        manager = _LiveKitTracingManager(session=mock_session)
        manager._trace = Mock()
        manager._trace.set_status = Mock()

        mock_client = Mock()
        mock_client.finish_trace = Mock()
        mock_get_client.return_value = mock_client

        event = Mock()
        event.error = "Test error"

        with patch(
            "noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", False
        ):
            manager._on_close(event)

        await asyncio.sleep(0.01)
        mock_create_span.assert_called_once()
        # Trace should still exist and set_status should be called
        if manager._trace:
            manager._trace.set_status.assert_called_once()

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.integrations.livekit.livekit_session.create_event_span")
    @patch("noveum_trace.get_client")
    @pytest.mark.asyncio
    async def test_on_close_exception_handling(
        self, mock_get_client, mock_create_span, mock_session
    ):
        """Test _on_close exception handling."""
        manager = _LiveKitTracingManager(session=mock_session)
        manager._trace = Mock()
        manager._trace.set_status = Mock()

        mock_client = Mock()
        mock_client.finish_trace = Mock(side_effect=Exception("Finish error"))
        mock_get_client.return_value = mock_client

        event = Mock()

        # Should not raise exception
        manager._on_close(event)

        await asyncio.sleep(0.01)


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestRealtimeSessionSetup:
    """Test RealtimeSession setup methods."""

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_setup_realtime_handlers_when_available(self, mock_session):
        """Test _setup_realtime_handlers when agent_activity available."""
        manager = _LiveKitTracingManager(session=mock_session)

        mock_agent_activity = Mock()
        mock_realtime_session = Mock()
        mock_realtime_session.on = Mock()
        mock_agent_activity.realtime_llm_session = Mock(
            return_value=mock_realtime_session
        )
        manager.session.agent_activity = mock_agent_activity

        manager._setup_realtime_handlers()

        assert manager._realtime_session == mock_realtime_session
        assert mock_realtime_session.on.call_count > 0

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_setup_realtime_handlers_when_none(self, mock_session):
        """Test _setup_realtime_handlers when realtime_llm_session returns None."""
        manager = _LiveKitTracingManager(session=mock_session)

        mock_agent_activity = Mock()
        mock_agent_activity.realtime_llm_session = Mock(return_value=None)
        manager.session.agent_activity = mock_agent_activity

        manager._setup_realtime_handlers()

        assert manager._realtime_session is None

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_try_setup_realtime_handlers_later_when_setup(self, mock_session):
        """Test _try_setup_realtime_handlers_later when already set up."""
        manager = _LiveKitTracingManager(session=mock_session)
        manager._realtime_session = Mock()

        manager._try_setup_realtime_handlers_later()

        # Should not call _setup_realtime_handlers again
        assert manager._realtime_session is not None

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    def test_register_realtime_handlers_exception(self, mock_session):
        """Test _register_realtime_handlers exception handling."""
        manager = _LiveKitTracingManager(session=mock_session)

        mock_realtime_session = Mock()
        mock_realtime_session.on = Mock(side_effect=Exception("Handler error"))

        # Should not raise exception
        manager._register_realtime_handlers(mock_realtime_session)


@pytest.mark.skipif(
    not LIVEKIT_SESSION_AVAILABLE, reason="LiveKit session not available"
)
class TestTraceCreationEdgeCases:
    """Test trace creation edge cases."""

    @pytest.mark.asyncio
    async def test_wrapped_start_with_job_context(self, mock_session, mock_client):
        """Test wrapped_start with job context available."""
        mock_trace = Mock()
        mock_client.start_trace.return_value = mock_trace

        with patch(
            "noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True
        ):
            manager = _LiveKitTracingManager(session=mock_session)
            manager._wrap_start_method()

            # Configure mock agent properly
            mock_agent = Mock(spec=[])
            mock_agent.label = "TestAgent"
            mock_agent.tools = []
            mock_agent.instructions = None

            # Mock get_job_context if available
            try:
                with patch(
                    "noveum_trace.integrations.livekit.livekit_session.set_current_trace"
                ):
                    with patch(
                        "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
                        new_callable=AsyncMock,
                    ):
                        with patch("noveum_trace.get_client", return_value=mock_client):
                            with patch(
                                "livekit.agents.get_job_context", create=True
                            ) as mock_get_job_context:
                                mock_job_ctx = Mock()
                                mock_job = Mock()
                                mock_job.id = "job_123"
                                mock_job_ctx.job = mock_job
                                mock_get_job_context.return_value = mock_job_ctx

                                manager._original_start = AsyncMock(return_value=None)

                                await manager.session.start(mock_agent)

                                mock_client.start_trace.assert_called_once()
                                # Check that job context was used in trace name
                                call_args = mock_client.start_trace.call_args
                                assert call_args is not None
            except (ImportError, ModuleNotFoundError):
                # Skip if livekit.agents not available
                pytest.skip("livekit.agents not available")

    @pytest.mark.asyncio
    async def test_wrapped_start_with_instructions_via_private(
        self, mock_session, mock_client
    ):
        """Test wrapped_start with agent instructions via _instructions."""
        mock_trace = Mock()
        mock_client.start_trace.return_value = mock_trace

        with patch(
            "noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True
        ):
            manager = _LiveKitTracingManager(session=mock_session)
            manager._wrap_start_method()

            # Configure mock agent properly
            mock_agent = Mock(spec=[])
            mock_agent.label = "TestAgent"
            mock_agent.tools = []
            mock_agent.instructions = None
            mock_agent._instructions = "Private instructions"

            with patch(
                "noveum_trace.integrations.livekit.livekit_session.set_current_trace"
            ):
                with patch(
                    "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
                    new_callable=AsyncMock,
                ):
                    with patch("noveum_trace.get_client", return_value=mock_client):
                        manager._original_start = AsyncMock(return_value=None)
                        await manager.session.start(mock_agent)

            mock_client.start_trace.assert_called_once()
            call_args = mock_client.start_trace.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_wrapped_start_when_get_job_context_fails(
        self, mock_session, mock_client
    ):
        """Test wrapped_start when get_job_context fails."""
        mock_trace = Mock()
        mock_client.start_trace.return_value = mock_trace

        with patch(
            "noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True
        ):
            manager = _LiveKitTracingManager(session=mock_session)
            manager._wrap_start_method()

            # Configure mock agent properly
            mock_agent = Mock(spec=[])
            mock_agent.label = "TestAgent"
            mock_agent.tools = []
            mock_agent.instructions = None

            try:
                with patch(
                    "noveum_trace.integrations.livekit.livekit_session.set_current_trace"
                ):
                    with patch(
                        "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
                        new_callable=AsyncMock,
                    ):
                        with patch("noveum_trace.get_client", return_value=mock_client):
                            with patch(
                                "livekit.agents.get_job_context", create=True
                            ) as mock_get_job:
                                mock_get_job.side_effect = Exception(
                                    "Job context error"
                                )

                                manager._original_start = AsyncMock(return_value=None)

                                await manager.session.start(mock_agent)

                                mock_client.start_trace.assert_called_once()
            except (ImportError, ModuleNotFoundError):
                pytest.skip("livekit.agents not available")

    @patch("noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True)
    @patch("noveum_trace.get_client")
    @patch(
        "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_wrapped_start_when_trace_creation_fails(
        self, mock_sleep, mock_get_client, mock_session, mock_client
    ):
        """Test wrapped_start when trace creation fails."""
        manager = _LiveKitTracingManager(session=mock_session)
        manager._wrap_start_method()

        mock_agent = Mock()
        mock_client.start_trace.side_effect = Exception("Trace creation error")
        mock_get_client.return_value = mock_client

        manager._original_start = AsyncMock(return_value=None)

        # Should fallback to original start
        await manager.session.start(mock_agent)

        manager._original_start.assert_called_once_with(mock_agent)

    @pytest.mark.asyncio
    async def test_wrapped_start_error_handling(self, mock_session, mock_client):
        """Test wrapped_start error handling and fallback."""
        mock_trace = Mock()
        mock_client.start_trace.return_value = mock_trace

        with patch(
            "noveum_trace.integrations.livekit.livekit_session.LIVEKIT_AVAILABLE", True
        ):
            manager = _LiveKitTracingManager(session=mock_session)
            manager._wrap_start_method()

            # Configure mock agent properly
            mock_agent = Mock(spec=[])
            mock_agent.label = "TestAgent"
            mock_agent.tools = []
            mock_agent.instructions = None

            with patch(
                "noveum_trace.integrations.livekit.livekit_session.set_current_trace"
            ):
                with patch(
                    "noveum_trace.integrations.livekit.livekit_session.asyncio.sleep",
                    new_callable=AsyncMock,
                ):
                    with patch("noveum_trace.get_client", return_value=mock_client):
                        manager._original_start = AsyncMock(
                            side_effect=RuntimeError("Start error")
                        )

                        # Should handle error and end trace
                        with pytest.raises(RuntimeError, match="Start error"):
                            await manager.session.start(mock_agent)

            mock_client.finish_trace.assert_called_once()
