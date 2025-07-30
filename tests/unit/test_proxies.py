"""Unit tests for proxy classes."""

from unittest.mock import Mock, patch

import pytest

from noveum_trace.proxies import (
    TracedChatCompletions,
    TracedCompletions,
    TracedEmbeddings,
    TracedTranscriptions,
)


class TestTracedCompletions:
    """Test suite for traced completions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.mock_original_completions = Mock()
        self.trace_config = {"capture_inputs": True, "capture_outputs": True}

        self.proxy = TracedCompletions(
            self.mock_original_completions, self.trace_config
        )

    @patch("noveum_trace.proxies.trace_llm")
    def test_create_with_basic_parameters(self, mock_trace_llm):
        """Test create method with basic parameters."""
        # Setup mock span
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)
        mock_trace_llm.return_value = mock_span

        # Mock response
        mock_response = Mock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_response.choices = [Mock(message=Mock(content="Test response"))]

        self.mock_original_completions.create.return_value = mock_response

        # Call proxy
        response = self.proxy.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test"}]
        )

        # Verify response
        assert response == mock_response

        # Verify trace_llm was called
        mock_trace_llm.assert_called_once_with(model="gpt-3.5-turbo", provider="openai")

        # Verify attributes were set
        mock_span.set_attribute.assert_any_call(
            "llm.messages", str([{"role": "user", "content": "Test"}])
        )
        mock_span.set_attributes.assert_called_with(
            {
                "llm.input_tokens": 10,
                "llm.output_tokens": 20,
                "llm.total_tokens": 30,
            }
        )

    @patch("noveum_trace.proxies.trace_llm")
    def test_create_with_capture_disabled(self, mock_trace_llm):
        """Test create method with capture flags disabled."""
        # Setup mock span
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)
        mock_trace_llm.return_value = mock_span

        # Create proxy with capture disabled
        trace_config = {"capture_inputs": False, "capture_outputs": False}
        proxy = TracedCompletions(self.mock_original_completions, trace_config)

        # Mock response
        mock_response = Mock()
        mock_response.model = "gpt-3.5-turbo"
        self.mock_original_completions.create.return_value = mock_response

        # Call proxy
        proxy.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Secret data"}]
        )

        # Verify sensitive data was not captured
        mock_span.set_attribute.assert_not_called()
        mock_span.set_attributes.assert_not_called()

    def test_getattr_delegation(self):
        """Test that unknown attributes are delegated to original."""
        # Add attribute to mock
        self.mock_original_completions.custom_attr = "test_value"

        # Access through proxy
        assert self.proxy.custom_attr == "test_value"


class TestTracedChatCompletions:
    """Test suite for traced chat completions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.mock_original_chat = Mock()
        self.trace_config = {"capture_inputs": True, "capture_outputs": True}

        self.proxy = TracedChatCompletions(self.mock_original_chat, self.trace_config)

    def test_completions_delegation(self):
        """Test that completions are properly delegated."""
        # TracedChatCompletions should have a completions attribute
        assert hasattr(self.proxy, "completions")
        # The completions should be a TracedCompletions instance
        from noveum_trace.proxies import TracedCompletions

        assert isinstance(self.proxy.completions, TracedCompletions)


class TestTracedEmbeddings:
    """Test suite for traced embeddings."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.mock_original_embeddings = Mock()
        self.trace_config = {"capture_inputs": True}

        self.proxy = TracedEmbeddings(self.mock_original_embeddings, self.trace_config)

    @patch("noveum_trace.proxies.trace_llm")
    def test_create_single_embedding(self, mock_trace_llm):
        """Test create method with single input."""
        # Setup mock span
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)
        mock_trace_llm.return_value = mock_span

        # Mock response
        mock_response = Mock()
        mock_response.model = "text-embedding-ada-002"
        mock_response.usage = Mock(prompt_tokens=5, total_tokens=5)
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

        self.mock_original_embeddings.create.return_value = mock_response

        # Call proxy
        response = self.proxy.create(model="text-embedding-ada-002", input="Test text")

        # Verify response
        assert response == mock_response

        # Verify trace was created with correct operation
        mock_trace_llm.assert_called_once_with(
            model="text-embedding-ada-002", provider="openai", operation="embeddings"
        )

        # Verify attributes were set (called twice - input and output)
        assert mock_span.set_attributes.call_count == 2
        # Check first call (input attributes)
        input_call_args = mock_span.set_attributes.call_args_list[0][0][0]
        assert input_call_args["llm.input_count"] == 1
        assert input_call_args["llm.input_type"] == "string"

    @patch("noveum_trace.proxies.trace_llm")
    def test_create_batch_embeddings(self, mock_trace_llm):
        """Test create method with batch input."""
        # Setup mock span
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)
        mock_trace_llm.return_value = mock_span

        # Mock response
        mock_response = Mock()
        mock_response.model = "text-embedding-ada-002"
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4]),
            Mock(embedding=[0.5, 0.6]),
        ]

        self.mock_original_embeddings.create.return_value = mock_response

        # Call proxy with list input
        input_texts = ["Text 1", "Text 2", "Text 3"]
        self.proxy.create(model="text-embedding-ada-002", input=input_texts)

        # Verify batch attributes (called twice - input and output)
        assert mock_span.set_attributes.call_count == 2
        # Check first call (input attributes)
        input_call_args = mock_span.set_attributes.call_args_list[0][0][0]
        assert input_call_args["llm.input_count"] == 3
        assert input_call_args["llm.input_type"] == "list"


class TestTracedTranscriptions:
    """Test suite for traced transcriptions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.mock_original_transcriptions = Mock()
        self.trace_config = {"capture_inputs": True, "capture_outputs": True}

        self.proxy = TracedTranscriptions(
            self.mock_original_transcriptions, self.trace_config
        )

    @patch("noveum_trace.proxies.trace_llm")
    def test_create_transcription(self, mock_trace_llm):
        """Test create method for transcription."""
        # Setup mock span
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)
        mock_trace_llm.return_value = mock_span

        # Mock response
        mock_response = Mock()
        mock_response.text = "This is the transcribed text"

        self.mock_original_transcriptions.create.return_value = mock_response

        # Mock file
        mock_file = Mock()
        mock_file.name = "audio.mp3"

        # Call proxy
        response = self.proxy.create(model="whisper-1", file=mock_file, language="en")

        # Verify response
        assert response == mock_response

        # Verify trace was created
        mock_trace_llm.assert_called_once_with(
            model="whisper-1", provider="openai", operation="speech_to_text"
        )

        # Verify attributes were set
        mock_span.set_attributes.assert_called_once()
        mock_span.set_attribute.assert_called_once_with(
            "audio.transcription", "This is the transcribed text"
        )

    @patch("noveum_trace.proxies.trace_llm")
    def test_create_with_additional_params(self, mock_trace_llm):
        """Test create with additional parameters."""
        # Setup mock span
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)
        mock_trace_llm.return_value = mock_span

        mock_response = Mock()
        mock_response.text = "Transcribed"
        self.mock_original_transcriptions.create.return_value = mock_response

        mock_file = Mock()
        mock_file.name = "speech.wav"

        # Call with additional params
        self.proxy.create(
            model="whisper-1",
            file=mock_file,
            prompt="Previous context",
            temperature=0.5,
            response_format="json",
        )

        # Verify attributes were set (response_format should be captured)
        mock_span.set_attributes.assert_called_once()
        # The response_format is captured in set_attributes, not individual set_attribute calls
        call_args = mock_span.set_attributes.call_args[0][0]
        assert call_args["audio.response_format"] == "json"

    def test_getattr_delegation(self):
        """Test attribute delegation."""
        self.mock_original_transcriptions.custom_method = Mock(return_value="result")

        result = self.proxy.custom_method("arg")

        assert result == "result"
        self.mock_original_transcriptions.custom_method.assert_called_once_with("arg")
