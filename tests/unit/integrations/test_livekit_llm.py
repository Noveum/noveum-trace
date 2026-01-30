"""Unit tests for LiveKit LLM wrapper."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from noveum_trace.integrations.livekit.livekit_llm import (
    LIVEKIT_AVAILABLE,
    LiveKitLLMWrapper,
    _WrappedLLMStream,
)


class AsyncStreamMock:
    """Mock async stream that wraps an async generator and adds aclose."""

    def __init__(self, generator):
        self._generator = generator
        self._aclose = AsyncMock()

    def __anext__(self):
        return self._generator.__anext__()

    async def aclose(self):
        await self._aclose()


@pytest.fixture
def mock_base_llm():
    """Create a mock base LLM."""
    llm = Mock()
    llm.model = "gpt-4"
    llm.provider = "openai"
    llm.label = "openai.LLM"
    llm.on = Mock()
    llm.off = Mock()
    llm.prewarm = Mock()
    llm.aclose = AsyncMock()
    return llm


@pytest.fixture
def mock_chat_context():
    """Create a mock chat context."""
    if not LIVEKIT_AVAILABLE:
        pytest.skip("LiveKit not available")

    from livekit.agents.llm import ChatContext

    ctx = ChatContext()
    ctx.add_message(role="system", content="You are a helpful assistant.")
    ctx.add_message(role="user", content="Hello, how are you?")
    return ctx


@pytest.fixture
def mock_tools():
    """Create mock tools."""
    tool1 = Mock()
    tool1.name = "get_weather"
    tool1.description = "Get weather information"
    tool1.args_schema = {
        "type": "object",
        "properties": {"location": {"type": "string"}},
    }

    tool2 = Mock()
    tool2.name = "calculate"
    tool2.description = "Perform calculations"
    tool2.args_schema = {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
    }

    return [tool1, tool2]


class TestLiveKitLLMWrapper:
    """Test LiveKitLLMWrapper class."""

    def test_initialization(self, mock_base_llm):
        """Test wrapper initialization."""
        wrapper = LiveKitLLMWrapper(
            llm=mock_base_llm,
            session_id="test_session",
            job_context={"job_id": "test_job"},
        )

        assert wrapper._base_llm == mock_base_llm
        assert wrapper._session_id == "test_session"
        assert wrapper._job_context == {"job_id": "test_job"}

        # Check event forwarding was set up
        if LIVEKIT_AVAILABLE:
            assert mock_base_llm.on.call_count == 2

    def test_initialization_without_job_context(self, mock_base_llm):
        """Test wrapper initialization without job context."""
        wrapper = LiveKitLLMWrapper(
            llm=mock_base_llm,
            session_id="test_session",
        )

        assert wrapper._job_context == {}

    def test_model_property(self, mock_base_llm):
        """Test model property delegation."""
        wrapper = LiveKitLLMWrapper(
            llm=mock_base_llm,
            session_id="test_session",
        )

        assert wrapper.model == "gpt-4"

    def test_provider_property(self, mock_base_llm):
        """Test provider property delegation."""
        wrapper = LiveKitLLMWrapper(
            llm=mock_base_llm,
            session_id="test_session",
        )

        assert wrapper.provider == "openai"

    def test_label_property(self, mock_base_llm):
        """Test label property delegation."""
        wrapper = LiveKitLLMWrapper(
            llm=mock_base_llm,
            session_id="test_session",
        )

        assert wrapper.label == "openai.LLM"

    def test_prewarm(self, mock_base_llm):
        """Test prewarm delegation."""
        wrapper = LiveKitLLMWrapper(
            llm=mock_base_llm,
            session_id="test_session",
        )

        wrapper.prewarm()
        mock_base_llm.prewarm.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose(self, mock_base_llm):
        """Test aclose with event handler cleanup."""
        wrapper = LiveKitLLMWrapper(
            llm=mock_base_llm,
            session_id="test_session",
        )

        await wrapper.aclose()

        if LIVEKIT_AVAILABLE:
            # Check event handlers were unregistered
            assert mock_base_llm.off.call_count >= 2
        mock_base_llm.aclose.assert_called_once()

    def test_chat_returns_wrapped_stream(self, mock_base_llm, mock_chat_context):
        """Test that chat() returns a wrapped stream."""
        if not LIVEKIT_AVAILABLE:
            pytest.skip("LiveKit not available")

        mock_stream = Mock()
        mock_base_llm.chat = Mock(return_value=mock_stream)

        wrapper = LiveKitLLMWrapper(
            llm=mock_base_llm,
            session_id="test_session",
        )

        result = wrapper.chat(chat_ctx=mock_chat_context, tools=[])

        assert isinstance(result, _WrappedLLMStream)
        assert result._base_stream == mock_stream
        mock_base_llm.chat.assert_called_once()


class TestWrappedLLMStream:
    """Test _WrappedLLMStream class."""

    @pytest.fixture
    def mock_wrapper(self, mock_base_llm):
        """Create a mock wrapper."""
        wrapper = LiveKitLLMWrapper(
            llm=mock_base_llm,
            session_id="test_session",
            job_context={"job_id": "test_job"},
        )
        return wrapper

    @pytest.fixture
    def create_mock_chunk(self):
        """Factory for creating mock ChatChunk objects."""
        if not LIVEKIT_AVAILABLE:
            pytest.skip("LiveKit not available")

        from livekit.agents.llm import ChatChunk, ChoiceDelta

        def _create(
            content=None,
            role=None,
            tool_calls=None,
            usage=None,
            chunk_id="test_chunk_id",
        ):
            delta = None
            if content or role or tool_calls:
                delta = ChoiceDelta(
                    content=content,
                    role=role,
                    tool_calls=tool_calls or [],
                )

            return ChatChunk(
                id=chunk_id,
                delta=delta,
                usage=usage,
            )

        return _create

    @pytest.mark.asyncio
    async def test_chunk_buffering(
        self, mock_wrapper, mock_chat_context, create_mock_chunk
    ):
        """Test that chunks are buffered correctly."""
        if not LIVEKIT_AVAILABLE:
            pytest.skip("LiveKit not available")

        # Create mock base stream that yields chunks
        chunks = [
            create_mock_chunk(content="Hello"),
            create_mock_chunk(content=" world"),
            create_mock_chunk(content="!"),
        ]

        async def chunk_generator():
            for chunk in chunks:
                yield chunk

        mock_base_stream = AsyncStreamMock(chunk_generator())

        wrapped_stream = _WrappedLLMStream(
            base_stream=mock_base_stream,
            llm_wrapper=mock_wrapper,
            chat_ctx=mock_chat_context,
            tools=[],
            kwargs={},
        )

        # Consume stream
        collected = []
        with patch("noveum_trace.integrations.livekit.livekit_llm.get_current_trace"):
            try:
                async for chunk in wrapped_stream:
                    collected.append(chunk)
            except StopAsyncIteration:
                pass

        # Check buffering
        assert len(wrapped_stream._buffered_chunks) == 3
        assert wrapped_stream._response_content == "Hello world!"

    @pytest.mark.asyncio
    async def test_ttft_measurement(
        self, mock_wrapper, mock_chat_context, create_mock_chunk
    ):
        """Test time to first token measurement."""
        if not LIVEKIT_AVAILABLE:
            pytest.skip("LiveKit not available")

        # First chunk has content (starts TTFT)
        chunks = [
            create_mock_chunk(content="Hello"),
            create_mock_chunk(content=" world"),
        ]

        async def chunk_generator():
            await asyncio.sleep(0.1)  # Simulate delay
            for chunk in chunks:
                yield chunk

        mock_base_stream = AsyncStreamMock(chunk_generator())

        wrapped_stream = _WrappedLLMStream(
            base_stream=mock_base_stream,
            llm_wrapper=mock_wrapper,
            chat_ctx=mock_chat_context,
            tools=[],
            kwargs={},
        )

        # Consume first chunk
        with patch("noveum_trace.integrations.livekit.livekit_llm.get_current_trace"):
            try:
                await wrapped_stream.__anext__()
                assert wrapped_stream._ttft is not None
                assert wrapped_stream._ttft > 0
            except StopAsyncIteration:
                pass

    @pytest.mark.asyncio
    async def test_usage_capture(
        self, mock_wrapper, mock_chat_context, create_mock_chunk
    ):
        """Test that usage information is captured."""
        if not LIVEKIT_AVAILABLE:
            pytest.skip("LiveKit not available")

        from livekit.agents.llm import CompletionUsage

        usage = CompletionUsage(
            completion_tokens=10,
            prompt_tokens=20,
            prompt_cached_tokens=5,
            total_tokens=30,
        )

        chunks = [
            create_mock_chunk(content="Hello"),
            create_mock_chunk(content=" world", usage=usage),
        ]

        async def chunk_generator():
            for chunk in chunks:
                yield chunk

        mock_base_stream = AsyncStreamMock(chunk_generator())

        wrapped_stream = _WrappedLLMStream(
            base_stream=mock_base_stream,
            llm_wrapper=mock_wrapper,
            chat_ctx=mock_chat_context,
            tools=[],
            kwargs={},
        )

        # Consume stream
        with patch("noveum_trace.integrations.livekit.livekit_llm.get_current_trace"):
            try:
                async for _ in wrapped_stream:
                    pass
            except StopAsyncIteration:
                pass

        # Check usage was captured
        assert wrapped_stream._usage == usage
        assert wrapped_stream._usage.completion_tokens == 10
        assert wrapped_stream._usage.prompt_tokens == 20

    @pytest.mark.asyncio
    async def test_tool_calls_capture(
        self, mock_wrapper, mock_chat_context, create_mock_chunk
    ):
        """Test that tool calls are captured."""
        if not LIVEKIT_AVAILABLE:
            pytest.skip("LiveKit not available")

        from livekit.agents.llm import FunctionToolCall

        tool_call = FunctionToolCall(
            type="function",
            name="get_weather",
            arguments='{"location": "Paris"}',
            call_id="call_123",
        )

        chunks = [
            create_mock_chunk(content="Let me check"),
            create_mock_chunk(tool_calls=[tool_call]),
        ]

        async def chunk_generator():
            for chunk in chunks:
                yield chunk

        mock_base_stream = AsyncStreamMock(chunk_generator())

        wrapped_stream = _WrappedLLMStream(
            base_stream=mock_base_stream,
            llm_wrapper=mock_wrapper,
            chat_ctx=mock_chat_context,
            tools=[],
            kwargs={},
        )

        # Consume stream
        with patch("noveum_trace.integrations.livekit.livekit_llm.get_current_trace"):
            try:
                async for _ in wrapped_stream:
                    pass
            except StopAsyncIteration:
                pass

        # Check tool calls were captured
        assert len(wrapped_stream._tool_calls) == 1
        assert wrapped_stream._tool_calls[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_span_creation_on_stream_end(
        self, mock_wrapper, mock_chat_context, create_mock_chunk
    ):
        """Test that span is created when stream ends."""
        if not LIVEKIT_AVAILABLE:
            pytest.skip("LiveKit not available")

        chunks = [create_mock_chunk(content="Test")]

        async def chunk_generator():
            for chunk in chunks:
                yield chunk

        mock_base_stream = AsyncStreamMock(chunk_generator())

        wrapped_stream = _WrappedLLMStream(
            base_stream=mock_base_stream,
            llm_wrapper=mock_wrapper,
            chat_ctx=mock_chat_context,
            tools=[],
            kwargs={},
        )

        # Mock trace and client
        mock_trace = Mock()
        mock_client = Mock()
        mock_span = Mock()
        mock_client.start_span = Mock(return_value=mock_span)
        mock_client.finish_span = Mock()

        with patch(
            "noveum_trace.integrations.livekit.livekit_llm.get_current_trace",
            return_value=mock_trace,
        ):
            with patch(
                "noveum_trace.get_client",
                return_value=mock_client,
            ):
                # Consume stream
                try:
                    async for _ in wrapped_stream:
                        pass
                except StopAsyncIteration:
                    pass

        # Verify span was created and finished
        assert mock_client.start_span.called
        assert mock_client.finish_span.called
        assert wrapped_stream._span_created

    @pytest.mark.asyncio
    async def test_error_handling(
        self, mock_wrapper, mock_chat_context, create_mock_chunk
    ):
        """Test error handling during streaming."""
        if not LIVEKIT_AVAILABLE:
            pytest.skip("LiveKit not available")

        chunks = [create_mock_chunk(content="Test")]

        async def chunk_generator():
            for chunk in chunks:
                yield chunk
            raise RuntimeError("Test error")

        mock_base_stream = AsyncStreamMock(chunk_generator())

        wrapped_stream = _WrappedLLMStream(
            base_stream=mock_base_stream,
            llm_wrapper=mock_wrapper,
            chat_ctx=mock_chat_context,
            tools=[],
            kwargs={},
        )

        # Mock trace and client
        mock_trace = Mock()
        mock_client = Mock()
        mock_span = Mock()
        mock_client.start_span = Mock(return_value=mock_span)
        mock_client.finish_span = Mock()

        with patch(
            "noveum_trace.integrations.livekit.livekit_llm.get_current_trace",
            return_value=mock_trace,
        ):
            with patch(
                "noveum_trace.get_client",
                return_value=mock_client,
            ):
                # Consume stream - should raise error
                with pytest.raises(RuntimeError, match="Test error"):
                    async for _ in wrapped_stream:
                        pass

        # Error flag should be set
        assert wrapped_stream._had_error

    @pytest.mark.asyncio
    async def test_aclose_creates_span_if_not_created(
        self, mock_wrapper, mock_chat_context
    ):
        """Test that aclose creates span if not already created."""
        if not LIVEKIT_AVAILABLE:
            pytest.skip("LiveKit not available")

        # Create empty async generator
        async def empty_generator():
            return
            yield  # Make it a generator

        mock_base_stream = AsyncStreamMock(empty_generator())

        wrapped_stream = _WrappedLLMStream(
            base_stream=mock_base_stream,
            llm_wrapper=mock_wrapper,
            chat_ctx=mock_chat_context,
            tools=[],
            kwargs={},
        )

        # Mock trace and client
        mock_trace = Mock()
        mock_client = Mock()
        mock_span = Mock()
        mock_client.start_span = Mock(return_value=mock_span)
        mock_client.finish_span = Mock()

        with patch(
            "noveum_trace.integrations.livekit.livekit_llm.get_current_trace",
            return_value=mock_trace,
        ):
            with patch(
                "noveum_trace.get_client",
                return_value=mock_client,
            ):
                await wrapped_stream.aclose()

        # Span should be created
        assert wrapped_stream._span_created
        assert mock_client.start_span.called
