"""
Proxy objects for tracing existing instances without modification.

This module provides proxy classes that wrap existing objects to add
tracing capabilities without requiring code changes.
"""

import functools
from typing import Any, Callable, Optional

from noveum_trace.context_managers import trace_agent, trace_llm


class TracedOpenAIClient:
    """
    Proxy for OpenAI client that automatically traces all calls.

    This class wraps an existing OpenAI client instance and adds
    tracing to all API calls without modifying the original code.
    """

    def __init__(
        self, original_client: Any, trace_config: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize the traced OpenAI client.

        Args:
            original_client: Original OpenAI client instance
            trace_config: Configuration options for tracing
        """
        self._original_client = original_client
        self._trace_config: dict[str, Any] = trace_config or {}

        # Create traced versions of nested objects
        self.chat = TracedChatCompletions(original_client.chat, trace_config)

        # Add other API objects if they exist
        if hasattr(original_client, "embeddings"):
            self.embeddings = TracedEmbeddings(original_client.embeddings, trace_config)

        if hasattr(original_client, "images"):
            self.images = TracedImages(original_client.images, trace_config)

        if hasattr(original_client, "audio"):
            self.audio = TracedAudio(original_client.audio, trace_config)

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to original client."""
        return getattr(self._original_client, name)


class TracedChatCompletions:
    """Traced version of chat completions."""

    def __init__(
        self, original_chat: Any, trace_config: Optional[dict[str, Any]]
    ) -> None:
        self._original_chat = original_chat
        self._trace_config: dict[str, Any] = trace_config or {}
        self.completions = TracedCompletions(original_chat.completions, trace_config)

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to original chat object."""
        return getattr(self._original_chat, name)


class TracedCompletions:
    """Traced version of completions."""

    def __init__(
        self, original_completions: Any, trace_config: Optional[dict[str, Any]]
    ) -> None:
        self._original_completions = original_completions
        self._trace_config: dict[str, Any] = trace_config or {}

    def create(self, **kwargs: Any) -> Any:
        """Traced version of create method."""
        model = kwargs.get("model", "unknown")

        with trace_llm(model=model, provider="openai") as span:
            # Capture input attributes
            if self._trace_config.get("capture_inputs", True):
                messages = kwargs.get("messages", [])
                max_length = self._trace_config.get("max_input_length", 2000)
                span.set_attribute("llm.messages", str(messages)[:max_length])

            # Make the actual call
            response = self._original_completions.create(**kwargs)

            # Capture output attributes
            if self._trace_config.get("capture_outputs", True):
                if hasattr(response, "usage"):
                    span.set_attributes(
                        {
                            "llm.input_tokens": response.usage.prompt_tokens,
                            "llm.output_tokens": response.usage.completion_tokens,
                            "llm.total_tokens": response.usage.total_tokens,
                        }
                    )

                if hasattr(response, "choices") and response.choices:
                    max_output = self._trace_config.get("max_output_length", 2000)
                    content = response.choices[0].message.content
                    span.set_attribute("llm.response", content[:max_output])

            return response

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to original completions object."""
        return getattr(self._original_completions, name)


class TracedEmbeddings:
    """Traced version of embeddings."""

    def __init__(
        self, original_embeddings: Any, trace_config: Optional[dict[str, Any]]
    ) -> None:
        self._original_embeddings = original_embeddings
        self._trace_config: dict[str, Any] = trace_config or {}

    def create(self, **kwargs: Any) -> Any:
        """Traced version of create method."""
        model = kwargs.get("model", "unknown")

        with trace_llm(model=model, provider="openai", operation="embeddings") as span:
            # Capture input attributes
            if self._trace_config.get("capture_inputs", True):
                input_data = kwargs.get("input", [])
                if isinstance(input_data, list):
                    span.set_attributes(
                        {"llm.input_count": len(input_data), "llm.input_type": "list"}
                    )
                else:
                    span.set_attributes(
                        {
                            "llm.input_count": 1,
                            "llm.input_type": "string",
                            "llm.input_length": len(str(input_data)),
                        }
                    )

            # Make the actual call
            response = self._original_embeddings.create(**kwargs)

            # Capture output attributes
            if self._trace_config.get("capture_outputs", True):
                if hasattr(response, "usage"):
                    span.set_attribute("llm.total_tokens", response.usage.total_tokens)

                if hasattr(response, "data") and response.data:
                    span.set_attributes(
                        {
                            "llm.embeddings_count": len(response.data),
                            "llm.embedding_dimensions": (
                                len(response.data[0].embedding) if response.data else 0
                            ),
                        }
                    )

            return response

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to original embeddings object."""
        return getattr(self._original_embeddings, name)


class TracedImages:
    """Traced version of images."""

    def __init__(
        self, original_images: Any, trace_config: Optional[dict[str, Any]]
    ) -> None:
        self._original_images = original_images
        self._trace_config: dict[str, Any] = trace_config or {}

    def generate(self, **kwargs: Any) -> Any:
        """Traced version of generate method."""
        model = kwargs.get("model", "dall-e-2")

        with trace_llm(
            model=model, provider="openai", operation="image_generation"
        ) as span:
            # Capture input attributes
            if self._trace_config.get("capture_inputs", True):
                prompt = kwargs.get("prompt", "")
                span.set_attributes(
                    {
                        "image.prompt": prompt[:500],  # Truncate prompt
                        "image.size": kwargs.get("size", "unknown"),
                        "image.quality": kwargs.get("quality", "standard"),
                        "image.n": kwargs.get("n", 1),
                    }
                )

            # Make the actual call
            response = self._original_images.generate(**kwargs)

            # Capture output attributes
            if self._trace_config.get("capture_outputs", True):
                if hasattr(response, "data") and response.data:
                    span.set_attributes(
                        {
                            "image.generated_count": len(response.data),
                            "image.urls": [
                                img.url for img in response.data if hasattr(img, "url")
                            ],
                        }
                    )

            return response

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to original images object."""
        return getattr(self._original_images, name)


class TracedAudio:
    """Traced version of audio."""

    def __init__(
        self, original_audio: Any, trace_config: Optional[dict[str, Any]]
    ) -> None:
        self._original_audio = original_audio
        self._trace_config: dict[str, Any] = trace_config or {}

        # Add speech sub-object if it exists
        if hasattr(original_audio, "speech"):
            self.speech = TracedSpeech(original_audio.speech, trace_config)

        # Add transcriptions sub-object if it exists
        if hasattr(original_audio, "transcriptions"):
            self.transcriptions = TracedTranscriptions(
                original_audio.transcriptions, trace_config
            )

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to original audio object."""
        return getattr(self._original_audio, name)


class TracedSpeech:
    """Traced version of speech."""

    def __init__(
        self, original_speech: Any, trace_config: Optional[dict[str, Any]]
    ) -> None:
        self._original_speech = original_speech
        self._trace_config: dict[str, Any] = trace_config or {}

    def create(self, **kwargs: Any) -> Any:
        """Traced version of create method."""
        model = kwargs.get("model", "tts-1")

        with trace_llm(
            model=model, provider="openai", operation="text_to_speech"
        ) as span:
            # Capture input attributes
            if self._trace_config.get("capture_inputs", True):
                input_text = kwargs.get("input", "")
                span.set_attributes(
                    {
                        "audio.input_text": input_text[:500],  # Truncate text
                        "audio.voice": kwargs.get("voice", "unknown"),
                        "audio.response_format": kwargs.get("response_format", "mp3"),
                    }
                )

            # Make the actual call
            response = self._original_speech.create(**kwargs)

            # Capture output attributes
            if self._trace_config.get("capture_outputs", True):
                # Audio response doesn't have much metadata to capture
                span.set_attribute("audio.generated", True)

            return response

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to original speech object."""
        return getattr(self._original_speech, name)


class TracedTranscriptions:
    """Traced version of transcriptions."""

    def __init__(
        self, original_transcriptions: Any, trace_config: Optional[dict[str, Any]]
    ) -> None:
        self._original_transcriptions = original_transcriptions
        self._trace_config: dict[str, Any] = trace_config or {}

    def create(self, **kwargs: Any) -> Any:
        """Traced version of create method."""
        model = kwargs.get("model", "whisper-1")

        with trace_llm(
            model=model, provider="openai", operation="speech_to_text"
        ) as span:
            # Capture input attributes
            if self._trace_config.get("capture_inputs", True):
                file = kwargs.get("file", None)
                span.set_attributes(
                    {
                        "audio.file_provided": file is not None,
                        "audio.language": kwargs.get("language", "en"),
                        "audio.response_format": kwargs.get("response_format", "json"),
                    }
                )

            # Make the actual call
            response = self._original_transcriptions.create(**kwargs)

            # Capture output attributes
            if self._trace_config.get("capture_outputs", True):
                if hasattr(response, "text"):
                    max_output = self._trace_config.get("max_output_length", 2000)
                    span.set_attribute(
                        "audio.transcription", response.text[:max_output]
                    )

            return response

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to original transcriptions object."""
        return getattr(self._original_transcriptions, name)


class TracedAgentProxy:
    """
    Proxy for existing agent instances to add tracing.

    This class wraps an existing agent instance and adds tracing
    to all method calls without modifying the original code.
    """

    def __init__(
        self,
        agent: Any,
        agent_type: str = "unknown",
        capabilities: Optional[list[str]] = None,
        trace_config: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the traced agent proxy.

        Args:
            agent: Original agent instance
            agent_type: Type of agent (conversational, task, etc.)
            capabilities: List of agent capabilities
            trace_config: Configuration options for tracing
        """
        self._agent = agent
        self._agent_type = agent_type
        self._capabilities = capabilities or []
        self._trace_config: dict[str, Any] = trace_config or {}

    def __getattr__(self, name: str) -> Any:
        """
        Get attribute from the original agent.

        If the attribute is a callable method, wrap it with tracing.
        """
        attr = getattr(self._agent, name)

        # If it's a callable method, wrap it with tracing
        if callable(attr):
            return self._wrap_method(attr, name)

        return attr

    def _wrap_method(
        self, method: Callable[..., Any], method_name: str
    ) -> Callable[..., Any]:
        """Wrap a method with tracing."""

        @functools.wraps(method)
        def traced_method(*args: Any, **kwargs: Any) -> Any:
            # Skip tracing for special methods
            if method_name.startswith("__") and method_name.endswith("__"):
                return method(*args, **kwargs)

            # Skip tracing for methods in the ignore list
            if method_name in self._trace_config.get("ignore_methods", []):
                return method(*args, **kwargs)

            with trace_agent(
                agent_type=self._agent_type,
                operation=method_name,
                capabilities=self._capabilities,
            ) as span:
                try:
                    # Capture input attributes if configured
                    if self._trace_config.get("capture_inputs", True):
                        span.set_attributes(
                            {
                                "agent.method": method_name,
                                "agent.args_count": len(args),
                                "agent.kwargs_count": len(kwargs),
                            }
                        )

                        # Capture specific arguments if configured
                        if self._trace_config.get("capture_args", False) and args:
                            max_length = self._trace_config.get("max_input_length", 500)
                            for i, arg in enumerate(args[:3]):  # Limit to first 3 args
                                span.set_attribute(
                                    f"agent.arg{i}", str(arg)[:max_length]
                                )

                    # Call the original method
                    result = method(*args, **kwargs)

                    # Capture output attributes if configured
                    if self._trace_config.get("capture_outputs", True):
                        max_length = self._trace_config.get("max_output_length", 500)
                        span.set_attribute("agent.result", str(result)[:max_length])

                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status("error", str(e))
                    raise

        return traced_method


class TracedLangChainLLM:
    """
    Proxy for LangChain LLM instances to add tracing.

    This class wraps an existing LangChain LLM instance and adds
    tracing to all calls without modifying the original code.
    """

    def __init__(self, llm: Any, trace_config: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the traced LangChain LLM proxy.

        Args:
            llm: Original LangChain LLM instance
            trace_config: Configuration options for tracing
        """
        self._llm = llm
        self._trace_config: dict[str, Any] = trace_config or {}

        # Extract model information if available
        self._model = getattr(llm, "model_name", None) or getattr(
            llm, "model", "unknown"
        )
        self._provider = self._detect_provider(llm)

    def _detect_provider(self, llm: Any) -> str:
        """Detect the provider from the LLM instance."""
        llm_class = llm.__class__.__name__.lower()

        if "openai" in llm_class:
            return "openai"
        elif "anthropic" in llm_class:
            return "anthropic"
        elif "huggingface" in llm_class:
            return "huggingface"
        elif "cohere" in llm_class:
            return "cohere"
        elif "azure" in llm_class:
            return "azure"
        else:
            return "unknown"

    def __call__(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        """Trace calls to the LLM."""
        with trace_llm(
            model=self._model, provider=self._provider, operation="langchain_llm_call"
        ) as span:
            try:
                # Capture input attributes
                if self._trace_config.get("capture_inputs", True):
                    max_length = self._trace_config.get("max_input_length", 2000)
                    span.set_attributes(
                        {
                            "llm.prompt": str(prompt)[:max_length],
                            "llm.args_count": len(args),
                            "llm.kwargs_count": len(kwargs),
                        }
                    )

                # Call the original LLM
                result = self._llm(prompt, *args, **kwargs)

                # Capture output attributes
                if self._trace_config.get("capture_outputs", True):
                    max_length = self._trace_config.get("max_output_length", 2000)
                    span.set_attribute("llm.response", str(result)[:max_length])

                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status("error", str(e))
                raise

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to original LLM."""
        return getattr(self._llm, name)


# Utility functions for creating proxies


def create_traced_openai_client(
    original_client: Any, trace_config: Optional[dict[str, Any]] = None
) -> TracedOpenAIClient:
    """
    Create a traced OpenAI client from an existing client.

    Args:
        original_client: Original OpenAI client instance
        trace_config: Configuration options for tracing

    Returns:
        TracedOpenAIClient instance

    Example:
        from openai import OpenAI

        # Create original client
        original_client = OpenAI()

        # Create traced client
        traced_client = create_traced_openai_client(original_client)

        # Use traced client normally - all calls are traced
        response = traced_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """
    return TracedOpenAIClient(original_client, trace_config)


def create_traced_agent(
    agent: Any,
    agent_type: str = "unknown",
    capabilities: Optional[list[str]] = None,
    trace_config: Optional[dict[str, Any]] = None,
) -> TracedAgentProxy:
    """
    Create a traced agent proxy from an existing agent.

    Args:
        agent: Original agent instance
        agent_type: Type of agent (conversational, task, etc.)
        capabilities: List of agent capabilities
        trace_config: Configuration options for tracing

    Returns:
        TracedAgentProxy instance

    Example:
        # Create original agent
        original_agent = SomeAgentClass()

        # Create traced agent
        traced_agent = create_traced_agent(
            original_agent,
            agent_type="research_agent",
            capabilities=["web_search", "summarization"]
        )

        # Use traced agent normally - all method calls are traced
        result = traced_agent.process_query("What is AI?")
    """
    return TracedAgentProxy(agent, agent_type, capabilities, trace_config)


def create_traced_langchain_llm(
    llm: Any, trace_config: Optional[dict[str, Any]] = None
) -> TracedLangChainLLM:
    """
    Create a traced LangChain LLM from an existing LLM.

    Args:
        llm: Original LangChain LLM instance
        trace_config: Configuration options for tracing

    Returns:
        TracedLangChainLLM instance

    Example:
        from langchain.llms import OpenAI

        # Create original LLM
        original_llm = OpenAI(temperature=0.7)

        # Create traced LLM
        traced_llm = create_traced_langchain_llm(original_llm)

        # Use traced LLM normally - all calls are traced
        result = traced_llm("What is AI?")
    """
    return TracedLangChainLLM(llm, trace_config)
