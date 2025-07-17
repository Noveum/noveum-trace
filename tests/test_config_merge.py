"""
Tests for configuration merging functionality.

This module tests the deep merging of configuration dictionaries
to ensure nested settings are preserved correctly.
"""

import os
from unittest.mock import patch

from noveum_trace.core.config import configure, get_config


class TestConfigMerging:
    """Test configuration merging functionality."""

    def setup_method(self):
        """Reset configuration before each test."""
        import noveum_trace.core.config as config_module

        config_module._config = None

    def teardown_method(self):
        """Clean up after each test."""
        import noveum_trace.core.config as config_module

        config_module._config = None

    def test_deep_merge_preserves_nested_settings(self):
        """Test that deep merge preserves nested settings from environment."""
        with patch.dict(
            os.environ,
            {
                "NOVEUM_API_KEY": "env-key",
                "NOVEUM_PROJECT": "env-project",
                "NOVEUM_ENDPOINT": "https://env.example.com/api/v1",
            },
            clear=True,
        ):

            # Configure with partial nested config
            config_data = {
                "transport": {
                    "timeout": 60,
                    "batch_size": 200,
                }
            }

            configure(config_data)
            config = get_config()

            # Environment settings should be preserved
            assert config.api_key == "env-key"
            assert config.project == "env-project"
            assert config.transport.endpoint == "https://env.example.com/api/v1"

            # Override settings should be applied
            assert config.transport.timeout == 60
            assert config.transport.batch_size == 200

            # Default settings should remain unchanged
            assert config.transport.retry_attempts == 3  # Default value

    def test_nested_config_overrides_work(self):
        """Test that nested config overrides work correctly."""
        with patch.dict(
            os.environ,
            {
                "NOVEUM_API_KEY": "env-key",
                "NOVEUM_PROJECT": "env-project",
            },
            clear=True,
        ):

            config_data = {
                "tracing": {
                    "enabled": False,
                    "sample_rate": 0.5,
                },
                "security": {
                    "redact_pii": True,
                },
            }

            configure(config_data)
            config = get_config()

            # Environment settings preserved
            assert config.api_key == "env-key"
            assert config.project == "env-project"

            # Nested overrides applied
            assert config.tracing.enabled is False
            assert config.tracing.sample_rate == 0.5
            assert config.security.redact_pii is True

            # Other nested settings remain default
            assert config.tracing.capture_errors is True  # Default
            assert config.security.encrypt_data is True  # Default

    def test_top_level_endpoint_precedence(self):
        """Test that top-level endpoint overrides transport.endpoint consistently."""
        with patch.dict(
            os.environ,
            {
                "NOVEUM_API_KEY": "env-key",
                "NOVEUM_PROJECT": "env-project",
            },
            clear=True,
        ):

            config_data = {
                "endpoint": "https://top-level.example.com/api/v1",
                "transport": {
                    "timeout": 60,
                    "endpoint": "https://transport.example.com/api/v1",
                },
            }

            configure(config_data)
            config = get_config()

            # Top-level endpoint should win
            assert config.transport.endpoint == "https://top-level.example.com/api/v1"

            # Other transport settings should be preserved
            assert config.transport.timeout == 60

    def test_deep_merge_with_multiple_levels(self):
        """Test deep merge with multiple levels of nesting."""
        with patch.dict(
            os.environ,
            {
                "NOVEUM_API_KEY": "env-key",
            },
            clear=True,
        ):

            config_data = {
                "integrations": {
                    "langchain": {
                        "enabled": True,
                        "auto_trace": True,
                    },
                    "openai": {
                        "enabled": True,
                    },
                }
            }

            configure(config_data)
            config = get_config()

            # Environment settings preserved
            assert config.api_key == "env-key"

            # Nested integration overrides applied
            assert config.integrations.langchain["enabled"] is True
            assert config.integrations.langchain["auto_trace"] is True
            assert config.integrations.openai["enabled"] is True

            # Other integrations remain default
            assert config.integrations.anthropic["enabled"] is False

    def test_merge_with_none_values(self):
        """Test that None values are handled correctly in merge."""
        with patch.dict(
            os.environ,
            {
                "NOVEUM_API_KEY": "env-key",
                "NOVEUM_PROJECT": "env-project",
            },
            clear=True,
        ):

            config_data = {
                "api_key": None,  # Should not override env value
                "project": "override-project",  # Should override env value
                "transport": {
                    "endpoint": None,  # Should not override default
                    "timeout": 45,  # Should override default
                },
            }

            configure(config_data)
            config = get_config()

            # None values should not override
            assert config.api_key == "env-key"
            assert config.transport.endpoint == "https://api.noveum.ai"  # Default

            # Non-None values should override
            assert config.project == "override-project"
            assert config.transport.timeout == 45

    def test_empty_nested_dict_merge(self):
        """Test merging with empty nested dictionaries."""
        with patch.dict(
            os.environ,
            {
                "NOVEUM_API_KEY": "env-key",
            },
            clear=True,
        ):

            config_data = {
                "transport": {},  # Empty dict should not clear existing values
                "tracing": {
                    "sample_rate": 0.8,
                },
            }

            configure(config_data)
            config = get_config()

            # Empty transport dict should not clear defaults
            assert config.transport.endpoint == "https://api.noveum.ai"
            assert config.transport.timeout == 30

            # Non-empty nested dict should apply overrides
            assert config.tracing.sample_rate == 0.8
            assert config.tracing.enabled is True  # Default preserved
