"""
Unit tests for LLM utility functions.

This module tests the comprehensive model registry, token counting,
cost estimation, and model validation functionality.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from noveum_trace.utils.llm_utils import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    ModelInfo,
    estimate_cost,
    estimate_token_count,
    extract_llm_metadata,
    get_model_info,
    get_supported_models,
    normalize_model_name,
    validate_model_compatibility,
)


class TestModelRegistry:
    """Test the comprehensive model registry."""

    def test_model_registry_completeness(self):
        """Test that the model registry contains expected models."""
        # Test OpenAI models
        assert "gpt-4.1" in MODEL_REGISTRY
        assert "gpt-4o" in MODEL_REGISTRY
        assert "gpt-3.5-turbo" in MODEL_REGISTRY

        # Test multi-provider support
        assert "gemini-2.5-flash" in MODEL_REGISTRY
        assert "claude-3.5-sonnet" in MODEL_REGISTRY  # Use actual model name

        # Verify minimum number of models
        assert len(MODEL_REGISTRY) >= 30

    def test_model_info_structure(self):
        """Test that model info has correct structure."""
        model_info = MODEL_REGISTRY["gpt-4.1"]

        assert isinstance(model_info, ModelInfo)
        assert model_info.provider == "openai"
        assert model_info.name == "gpt-4.1"
        assert isinstance(model_info.context_window, int)
        assert isinstance(model_info.max_output_tokens, int)
        assert isinstance(model_info.input_cost_per_1m, float)
        assert isinstance(model_info.output_cost_per_1m, float)
        assert isinstance(model_info.supports_vision, bool)
        assert isinstance(model_info.supports_function_calling, bool)

    def test_model_aliases(self):
        """Test that model aliases work correctly."""
        # Test that aliases exist
        assert isinstance(MODEL_ALIASES, dict)
        assert len(MODEL_ALIASES) > 0

        # Test specific alias functionality
        assert "claude-3.5-sonnet-20241022" in MODEL_ALIASES
        assert MODEL_ALIASES["claude-3.5-sonnet-20241022"] == "claude-3.5-sonnet"

    def test_normalize_model_name(self):
        """Test model name normalization."""
        # Test direct names
        assert normalize_model_name("gpt-4.1") == "gpt-4.1"
        assert normalize_model_name("gemini-2.5-flash") == "gemini-2.5-flash"

        # Test case normalization
        assert normalize_model_name("GPT-4.1") == "gpt-4.1"

        # Test alias resolution through normalization
        normalized = normalize_model_name("claude-3.5-sonnet-20241022")
        # Should normalize to the canonical form
        assert normalized in ["claude-3.5-sonnet", "claude-3.5-sonnet-20241022"]


class TestModelValidation:
    """Test model validation and compatibility checking."""

    def test_get_model_info_valid(self):
        """Test getting model info for valid models."""
        info = get_model_info("gpt-4.1")
        assert info is not None
        assert info.provider == "openai"
        assert info.context_window == 1047576

    def test_get_model_info_invalid(self):
        """Test getting model info for invalid models."""
        info = get_model_info("nonexistent-model")
        assert info is None

    def test_get_model_info_with_alias(self):
        """Test getting model info using aliases."""
        # Test that we can get model info through aliases
        info = get_model_info("claude-3.5-sonnet-20241022")
        assert info is not None
        # Should resolve to the canonical model
        assert info.name in ["claude-3.5-sonnet", "claude-3.5-sonnet-20241022"]

    def test_validate_model_compatibility_valid(self):
        """Test validation for supported models."""
        messages = [{"role": "user", "content": "Hello"}]
        result = validate_model_compatibility("gpt-4.1", messages)
        assert result["valid"] is True
        assert result["model_info"] is not None

    def test_validate_model_compatibility_invalid(self):
        """Test validation for unsupported models."""
        messages = [{"role": "user", "content": "Hello"}]
        result = validate_model_compatibility("nonexistent-model", messages)
        assert result["valid"] is False
        assert len(result["warnings"]) > 0

    def test_validate_model_compatibility_with_suggestions(self):
        """Test validation provides intelligent suggestions."""
        messages = [{"role": "user", "content": "Hello"}]
        result = validate_model_compatibility("gpt-unknown", messages)
        assert result["valid"] is False
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0

    def test_get_supported_models(self):
        """Test getting supported models by provider."""
        openai_models = get_supported_models("openai")
        assert "gpt-4.1" in openai_models
        assert "gpt-4o" in openai_models
        assert len(openai_models) >= 5

        all_models = get_supported_models()
        assert len(all_models) >= 30


class TestTokenCounting:
    """Test token counting and estimation."""

    def test_estimate_token_count_basic(self):
        """Test basic token counting."""
        text = "Hello, world!"
        count = estimate_token_count(text)
        assert isinstance(count, int)
        assert count > 0
        assert count < 10  # Should be a small number

    def test_estimate_token_count_long_text(self):
        """Test token counting for longer text."""
        text = "This is a longer text that should have more tokens. " * 10
        count = estimate_token_count(text)
        assert count > 50  # Should be more tokens

    def test_estimate_token_count_json(self):
        """Test token counting for JSON content."""
        json_text = '{"key": "value", "number": 42, "nested": {"array": [1, 2, 3]}}'
        count = estimate_token_count(json_text)
        assert count > 10

    def test_estimate_token_count_code(self):
        """Test token counting for code content."""
        code_text = """
def hello_world():
    print("Hello, world!")
    return True
"""
        count = estimate_token_count(code_text)
        assert count > 10

    def test_estimate_token_count_messages(self):
        """Test token counting for message arrays."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]
        count = estimate_token_count(messages)
        assert count > 20  # Should include message overhead

    def test_estimate_token_count_uses_provider_tokenizer(self, monkeypatch):
        """Ensure provider-specific tokenizers are consulted when available."""

        def _mock_count(*_, **__):
            return 42

        monkeypatch.setattr(
            "noveum_trace.utils.tokenizers.count_tokens", _mock_count
        )

        count = estimate_token_count("hello world", model="gpt-4o")
        assert count == 42

    def test_estimate_token_count_fallback_when_tokenizer_missing(self, monkeypatch):
        """Fallback heuristics should be used when provider tokenizers return None."""

        def _mock_count(*_, **__):
            return None

        monkeypatch.setattr(
            "noveum_trace.utils.tokenizers.count_tokens", _mock_count
        )

        count = estimate_token_count("hello world", model="gpt-4o")
        assert isinstance(count, int)
        assert count > 0


class TestCostEstimation:
    """Test cost estimation functionality."""

    def test_estimate_cost_basic(self):
        """Test basic cost estimation."""
        cost = estimate_cost("gpt-4.1", input_tokens=1000, output_tokens=500)
        assert isinstance(cost, dict)
        assert "total_cost" in cost
        assert "input_cost" in cost
        assert "output_cost" in cost
        assert cost["total_cost"] > 0

    def test_estimate_cost_accuracy(self):
        """Test cost estimation accuracy for known pricing."""
        # GPT-4.1: $3.00 input, $12.00 output per 1M tokens (Oct 29 2025)
        cost = estimate_cost("gpt-4.1", input_tokens=1000000, output_tokens=1000000)

        assert abs(cost["input_cost"] - 3.00) < 0.01
        assert abs(cost["output_cost"] - 12.00) < 0.01
        assert abs(cost["total_cost"] - 15.00) < 0.01

    def test_estimate_cost_different_models(self):
        """Test cost estimation for different models."""
        cost_gpt4 = estimate_cost("gpt-4.1", input_tokens=10000, output_tokens=5000)
        cost_gpt35 = estimate_cost(
            "gpt-3.5-turbo", input_tokens=10000, output_tokens=5000
        )

        # GPT-4.1 should be more expensive
        assert cost_gpt4["total_cost"] > cost_gpt35["total_cost"]

    def test_estimate_cost_invalid_model(self):
        """Test cost estimation for invalid model."""
        cost = estimate_cost("nonexistent-model", input_tokens=1000, output_tokens=500)
        # Should use fallback pricing, not return 0
        assert cost["total_cost"] > 0
        assert cost["provider"] == "unknown"

    def test_estimate_cost_zero_tokens(self):
        """Test cost estimation with zero tokens."""
        cost = estimate_cost("gpt-4.1", input_tokens=0, output_tokens=0)
        assert cost["total_cost"] == 0.0
        assert cost["input_cost"] == 0.0
        assert cost["output_cost"] == 0.0

    def test_estimate_cost_pricing_details(self):
        """Test that cost estimation includes pricing details."""
        cost = estimate_cost("gpt-4.1", input_tokens=10000, output_tokens=5000)

        assert "input_cost_per_1m" in cost
        assert "output_cost_per_1m" in cost
        assert cost["input_cost_per_1m"] == 3.00
        assert cost["output_cost_per_1m"] == 12.00


class TestMetadataExtraction:
    """Test LLM metadata extraction."""

    def test_extract_llm_metadata_basic(self):
        """Test basic metadata extraction."""
        # Create a mock response object
        mock_response = MagicMock()
        mock_response.model = "gpt-4.1"
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 500
        mock_response.usage.total_tokens = 1500

        metadata = extract_llm_metadata(mock_response)

        assert isinstance(metadata, dict)
        # The exact structure depends on implementation

    def test_extract_llm_metadata_no_usage(self):
        """Test metadata extraction without usage info."""
        mock_response = MagicMock()
        mock_response.model = "gpt-4.1"
        mock_response.usage = None

        metadata = extract_llm_metadata(mock_response)
        assert isinstance(metadata, dict)

    def test_extract_llm_metadata_gemini_usage_metadata(self):
        """Gemini responses expose usage via usage_metadata."""

        response = SimpleNamespace(
            model="gemini-1.5-pro",
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=4,
                total_token_count=16,
            ),
        )

        metadata = extract_llm_metadata(response)

        assert metadata.get("llm.usage.prompt_tokens") == 12
        assert metadata.get("llm.usage.output_tokens") == 4
        assert metadata.get("llm.usage.total_tokens") == 16
        assert metadata.get("llm.provider") == "google"

    def test_extract_llm_metadata_cohere_meta_tokens(self):
        """Cohere responses surface counts inside the meta.tokens structure."""

        response = SimpleNamespace(
            model="command-r",
            meta={"tokens": {"input_tokens": 7, "output_tokens": 3}},
        )

        metadata = extract_llm_metadata(response)

        assert metadata.get("llm.usage.input_tokens") == 7
        assert metadata.get("llm.usage.output_tokens") == 3
        assert metadata.get("llm.usage.total_tokens") == 10


class TestProxyServiceDetection:
    """Test proxy service detection functionality."""

    def test_detect_multi_provider_proxy(self):
        """Test detection of multi-provider proxy services."""
        allowed_models = ["gemini-2.5-flash", "gpt-4.1-mini", "gpt-4.1-nano"]

        # Check if we can detect this as a proxy service
        openai_models = [
            m
            for m in allowed_models
            if get_model_info(m) and get_model_info(m).provider == "openai"
        ]
        google_models = [
            m
            for m in allowed_models
            if get_model_info(m) and get_model_info(m).provider == "google"
        ]

        assert len(openai_models) >= 1  # Should find OpenAI models
        assert len(google_models) >= 1  # Should find Google models

    def test_cost_comparison_proxy_models(self):
        """Test cost comparison for proxy service models."""
        models_to_compare = ["gemini-2.5-flash", "gpt-4.1-mini"]

        costs = {}
        for model in models_to_compare:
            if get_model_info(model):
                cost = estimate_cost(model, input_tokens=10000, output_tokens=5000)
                costs[model] = cost["total_cost"]

        assert len(costs) >= 1  # At least one model should have cost info

    def test_model_suggestion_algorithm(self):
        """Test the model suggestion algorithm for proxy services."""
        # Test with Issue #3 scenario
        messages = [{"role": "user", "content": "Hello"}]
        result = validate_model_compatibility("gpt-4.1-unknown", messages)

        assert result["valid"] is False
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_input(self):
        """Test handling of empty inputs."""
        # Empty string might return 1 token (empty token)
        count = estimate_token_count("")
        assert isinstance(count, int)
        assert count >= 0

        cost = estimate_cost("gpt-4.1", input_tokens=0, output_tokens=0)
        assert cost["total_cost"] == 0.0

    def test_none_input(self):
        """Test handling of None inputs."""
        count = estimate_token_count(None)
        assert isinstance(count, int)
        assert count >= 0

        assert get_model_info(None) is None

    def test_invalid_content_type(self):
        """Test handling of invalid content types."""
        count = estimate_token_count(12345)  # Number instead of string
        assert isinstance(count, int)
        assert count >= 0

    def test_very_large_token_counts(self):
        """Test handling of very large token counts."""
        cost = estimate_cost(
            "gpt-4.1", input_tokens=1000000000, output_tokens=500000000
        )
        assert cost["total_cost"] > 1000  # Should be very expensive
        assert isinstance(cost["total_cost"], float)

    def test_case_insensitive_model_names(self):
        """Test that model names are case insensitive."""
        info1 = get_model_info("gpt-4.1")
        info2 = get_model_info("GPT-4.1")
        info3 = get_model_info("Gpt-4.1")

        assert info1 is not None
        assert info2 is not None
        assert info3 is not None
        assert info1.name == info2.name == info3.name
