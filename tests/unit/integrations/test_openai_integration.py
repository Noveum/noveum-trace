"""
Unit tests for OpenAI integration with enhanced error handling.

This module tests the improved OpenAI integration with proxy service detection,
enhanced error handling, model validation, and comprehensive diagnostics.
"""

from unittest.mock import MagicMock


class TestModelValidation:
    """Test model validation and compatibility checking."""

    def test_model_registry_access(self):
        """Test that we can access the model registry."""
        from noveum_trace.utils.llm_utils import MODEL_REGISTRY, get_model_info

        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0

        # Test basic model info retrieval
        info = get_model_info("gpt-4.1")
        assert info is not None
        assert info.provider == "openai"

    def test_cost_estimation_basic(self):
        """Test basic cost estimation functionality."""
        from noveum_trace.utils.llm_utils import estimate_cost

        cost = estimate_cost("gpt-4.1", input_tokens=1000, output_tokens=500)
        assert isinstance(cost, dict)
        assert "total_cost" in cost
        assert cost["total_cost"] > 0

    def test_token_counting_basic(self):
        """Test basic token counting functionality."""
        from noveum_trace.utils.llm_utils import estimate_token_count

        count = estimate_token_count("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    def test_model_compatibility_validation(self):
        """Test model compatibility validation."""
        from noveum_trace.utils.llm_utils import validate_model_compatibility

        messages = [{"role": "user", "content": "Hello"}]

        # Test valid model
        result = validate_model_compatibility("gpt-4.1", messages)
        assert isinstance(result, dict)
        assert "valid" in result
        assert result["valid"] is True

        # Test invalid model
        result = validate_model_compatibility("nonexistent-model", messages)
        assert result["valid"] is False


class TestProxyServiceDetection:
    """Test proxy service detection functionality."""

    def test_detect_multi_provider_models(self):
        """Test detection of multi-provider model configurations."""
        from noveum_trace.utils.llm_utils import get_model_info

        # Test Issue #3 scenario models
        test_models = ["gemini-2.5-flash", "gpt-4.1-mini"]

        model_providers = {}
        for model in test_models:
            info = get_model_info(model)
            if info:
                model_providers[model] = info.provider

        # Should have models from different providers
        assert len(set(model_providers.values())) > 1

    def test_cost_comparison_for_proxy_models(self):
        """Test cost comparison for proxy service models."""
        from noveum_trace.utils.llm_utils import estimate_cost, get_model_info

        models = ["gemini-2.5-flash", "gpt-4.1-mini"]
        costs = {}

        for model in models:
            if get_model_info(model):
                cost_info = estimate_cost(model, input_tokens=10000, output_tokens=5000)
                costs[model] = cost_info["total_cost"]

        # Verify we can calculate costs for multiple models
        assert len(costs) >= 1

    def test_model_suggestion_algorithm(self):
        """Test the model suggestion algorithm."""
        from noveum_trace.utils.llm_utils import validate_model_compatibility

        messages = [{"role": "user", "content": "Hello"}]

        # Test with unknown model that should trigger suggestions
        result = validate_model_compatibility("gpt-unknown-model", messages)

        assert result["valid"] is False
        assert "suggestions" in result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_and_none_inputs(self):
        """Test handling of edge case inputs."""
        from noveum_trace.utils.llm_utils import estimate_token_count, get_model_info

        # Test empty string
        count = estimate_token_count("")
        assert isinstance(count, int)
        assert count >= 0

        # Test None input
        count = estimate_token_count(None)
        assert isinstance(count, int)
        assert count >= 0

        # Test None model name
        info = get_model_info(None)
        assert info is None

    def test_invalid_model_fallback(self):
        """Test fallback behavior for invalid models."""
        from noveum_trace.utils.llm_utils import estimate_cost

        # Test with nonexistent model
        cost = estimate_cost("totally-fake-model", input_tokens=1000, output_tokens=500)

        # Should use fallback pricing
        assert cost["total_cost"] > 0
        assert cost["provider"] == "unknown"

    def test_case_insensitive_model_names(self):
        """Test that model names are handled case-insensitively."""
        from noveum_trace.utils.llm_utils import get_model_info

        info1 = get_model_info("gpt-4.1")
        info2 = get_model_info("GPT-4.1")
        info3 = get_model_info("Gpt-4.1")

        # All should return valid info (case insensitive)
        assert info1 is not None
        assert info2 is not None
        assert info3 is not None


class TestPerformanceMetrics:
    """Test performance-related functionality."""

    def test_large_token_counts(self):
        """Test handling of very large token counts."""
        from noveum_trace.utils.llm_utils import estimate_cost

        # Test with very large numbers
        cost = estimate_cost("gpt-4.1", input_tokens=1000000, output_tokens=500000)

        assert cost["total_cost"] > 1.0  # Should be expensive
        assert isinstance(cost["total_cost"], float)

    def test_zero_token_counts(self):
        """Test handling of zero token counts."""
        from noveum_trace.utils.llm_utils import estimate_cost

        cost = estimate_cost("gpt-4.1", input_tokens=0, output_tokens=0)

        assert cost["total_cost"] == 0.0
        assert cost["input_cost"] == 0.0
        assert cost["output_cost"] == 0.0


class TestModelRegistryIntegrity:
    """Test the integrity and completeness of the model registry."""

    def test_model_registry_structure(self):
        """Test that the model registry has proper structure."""
        from noveum_trace.utils.llm_utils import MODEL_REGISTRY, ModelInfo

        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0

        # Test that all entries are ModelInfo objects
        for model_name, model_info in MODEL_REGISTRY.items():
            assert isinstance(model_info, ModelInfo)
            assert isinstance(model_name, str)
            assert model_info.provider
            assert model_info.name
            assert model_info.context_window > 0
            assert model_info.input_cost_per_1m >= 0
            assert model_info.output_cost_per_1m >= 0

    def test_openai_models_present(self):
        """Test that OpenAI models are present in registry."""
        from noveum_trace.utils.llm_utils import MODEL_REGISTRY

        openai_models = [
            name for name, info in MODEL_REGISTRY.items() if info.provider == "openai"
        ]

        assert len(openai_models) > 0
        assert "gpt-4.1" in openai_models

    def test_multi_provider_support(self):
        """Test that multiple providers are supported."""
        from noveum_trace.utils.llm_utils import MODEL_REGISTRY

        providers = {info.provider for info in MODEL_REGISTRY.values()}

        # Should have multiple providers
        assert len(providers) > 1
        assert "openai" in providers


class TestUtilityFunctions:
    """Test various utility functions."""

    def test_normalize_model_name(self):
        """Test model name normalization."""
        from noveum_trace.utils.llm_utils import normalize_model_name

        # Test basic normalization
        assert normalize_model_name("GPT-4.1") == "gpt-4.1"
        assert normalize_model_name("Claude-3") == "claude-3"
        assert normalize_model_name("GEMINI-PRO") == "gemini-pro"

    def test_get_supported_models(self):
        """Test getting supported models."""
        from noveum_trace.utils.llm_utils import get_supported_models

        # Test all models
        all_models = get_supported_models()
        assert isinstance(all_models, list)
        assert len(all_models) > 0

        # Test OpenAI models
        openai_models = get_supported_models("openai")
        assert isinstance(openai_models, list)
        assert len(openai_models) > 0

        # Check that OpenAI models follow expected patterns
        # Should include GPT models and reasoning models (o1, o3, o4, etc.)
        for model in openai_models:
            is_valid_openai = (
                "gpt" in model.lower()
                or model.lower().startswith("o")
                and any(char.isdigit() for char in model)
            )
            assert is_valid_openai, f"Unexpected OpenAI model pattern: {model}"

    def test_metadata_extraction(self):
        """Test LLM metadata extraction."""
        from noveum_trace.utils.llm_utils import extract_llm_metadata

        # Create a mock response
        mock_response = MagicMock()
        mock_response.model = "gpt-4.1"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        metadata = extract_llm_metadata(mock_response)
        assert isinstance(metadata, dict)
