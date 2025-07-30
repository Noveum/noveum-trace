"""Unit tests for auto instrumentation functionality."""

from unittest.mock import Mock

import pytest

from noveum_trace.auto_instrument import (
    OpenAIInstrumentation,
    create_production_config,
    get_default_config,
)


class TestAutoInstrumentConfig:
    """Test auto instrumentation configuration functions."""

    def test_get_default_config_openai(self):
        """Test default configuration for OpenAI."""
        config = get_default_config("openai")

        assert config == {
            "capture_inputs": True,
            "capture_outputs": True,
            "calculate_cost": False,
        }

    def test_get_default_config_anthropic(self):
        """Test default configuration for Anthropic."""
        config = get_default_config("anthropic")

        assert config == {
            "capture_inputs": True,
            "capture_outputs": True,
        }

    def test_get_default_config_langchain(self):
        """Test default configuration for LangChain."""
        config = get_default_config("langchain")

        assert config == {
            "capture_inputs": True,
            "capture_outputs": True,
            "trace_chains": True,
        }

    def test_get_default_config_unknown_library(self):
        """Test default configuration for unknown library."""
        config = get_default_config("unknown_library")

        assert config == {}

    @pytest.mark.parametrize(
        "library",
        [
            "openai",
            "anthropic",
            "langchain",
            "some_other_lib",
            "",
            None,
        ],
    )
    def test_get_default_config_returns_dict(self, library):
        """Test that get_default_config always returns a dictionary."""
        config = get_default_config(library)

        assert isinstance(config, dict)

    def test_create_production_config(self):
        """Test production configuration creation."""
        prod_config = create_production_config()

        expected = {
            "openai": {
                "capture_inputs": False,
                "capture_outputs": True,
                "calculate_cost": True,
            },
            "anthropic": {
                "capture_inputs": False,
                "capture_outputs": True,
            },
            "langchain": {
                "capture_inputs": False,
                "capture_outputs": True,
                "trace_chains": False,
            },
        }

        assert prod_config == expected

    def test_production_config_reduces_overhead(self):
        """Test that production config reduces overhead compared to defaults."""
        prod_config = create_production_config()

        # Verify capture_inputs is disabled in production
        assert prod_config["openai"]["capture_inputs"] is False
        assert prod_config["anthropic"]["capture_inputs"] is False
        assert prod_config["langchain"]["capture_inputs"] is False

        # Verify trace_chains is disabled for langchain
        assert prod_config["langchain"]["trace_chains"] is False

        # Verify cost calculation is enabled for openai
        assert prod_config["openai"]["calculate_cost"] is True


class TestOpenAIInstrumentationCostCalculation:
    """Test cost calculation functionality in OpenAIInstrumentation."""

    @pytest.fixture
    def openai_instrumentation(self):
        """Create OpenAIInstrumentation instance."""
        config = {"calculate_cost": True}
        return OpenAIInstrumentation(config)

    def test_calculate_openai_cost_gpt35(self, openai_instrumentation):
        """Test cost calculation for GPT-3.5 model."""
        usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        cost = openai_instrumentation._calculate_openai_cost("gpt-3.5-turbo", usage)

        # GPT-3.5: $0.002/1K total tokens (based on actual implementation)
        expected_cost = 150 * 0.002 / 1000
        assert cost == pytest.approx(expected_cost, rel=1e-6)

    def test_calculate_openai_cost_gpt4(self, openai_instrumentation):
        """Test cost calculation for GPT-4 model."""
        usage = Mock(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        cost = openai_instrumentation._calculate_openai_cost("gpt-4", usage)

        # GPT-4: $0.03/1K total tokens
        expected_cost = 300 * 0.03 / 1000
        assert cost == pytest.approx(expected_cost, rel=1e-6)

    def test_calculate_openai_cost_gpt4_turbo(self, openai_instrumentation):
        """Test cost calculation for GPT-4 Turbo model."""
        usage = Mock(prompt_tokens=500, completion_tokens=200, total_tokens=700)

        cost = openai_instrumentation._calculate_openai_cost("gpt-4-turbo", usage)

        # GPT-4 Turbo: $0.01/1K total tokens
        expected_cost = 700 * 0.01 / 1000
        assert cost == pytest.approx(expected_cost, rel=1e-6)

    def test_calculate_openai_cost_unknown_model(self, openai_instrumentation):
        """Test cost calculation for unknown model uses default rate."""
        usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        cost = openai_instrumentation._calculate_openai_cost("unknown-model", usage)

        # Unknown models default to $0.002/1K total tokens
        expected_cost = 150 * 0.002 / 1000
        assert cost == pytest.approx(expected_cost, rel=1e-6)

    @pytest.mark.parametrize(
        "model,total_tokens,expected_rate",
        [
            ("gpt-3.5-turbo", 1500, 0.002),  # Known model
            ("gpt-4", 1500, 0.03),  # Known model
            ("gpt-4-turbo", 1500, 0.01),  # Known model
            ("unknown-model", 1500, 0.002),  # Default rate for unknown
        ],
    )
    def test_calculate_openai_cost_variants(
        self, openai_instrumentation, model, total_tokens, expected_rate
    ):
        """Test cost calculation for model variants."""
        usage = Mock(
            prompt_tokens=1000, completion_tokens=500, total_tokens=total_tokens
        )

        cost = openai_instrumentation._calculate_openai_cost(model, usage)

        expected_cost = total_tokens * expected_rate / 1000
        assert cost == pytest.approx(expected_cost, rel=1e-6)

    def test_calculate_openai_cost_edge_cases(self, openai_instrumentation):
        """Test cost calculation edge cases."""
        # Zero tokens
        usage = Mock(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        cost = openai_instrumentation._calculate_openai_cost("gpt-3.5-turbo", usage)
        assert cost == 0.0

        # Very large token counts
        usage = Mock(
            prompt_tokens=1000000, completion_tokens=500000, total_tokens=1500000
        )
        cost = openai_instrumentation._calculate_openai_cost("gpt-4", usage)
        expected = 1500000 * 0.03 / 1000  # Based on total_tokens
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_calculate_openai_cost_with_missing_usage_attrs(
        self, openai_instrumentation
    ):
        """Test cost calculation when usage object has missing attributes."""
        # Usage without total_tokens
        usage = Mock(spec=[])
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        # No total_tokens attribute

        cost = openai_instrumentation._calculate_openai_cost("gpt-3.5-turbo", usage)
        # Should handle gracefully, returning 0 when total_tokens is missing
        assert cost == 0.0
