"""
Comprehensive tests for configuration management functionality.

This module provides extensive test coverage for the configuration system,
including all configuration classes, loading mechanisms, and edge cases.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml

from noveum_trace.core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT,
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_MAX_SPANS_PER_TRACE,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_TIMEOUT,
    Config,
    IntegrationConfig,
    SecurityConfig,
    TracingConfig,
    TransportConfig,
    _deep_merge_dicts,
    _load_from_environment,
    _load_from_file,
    configure,
    get_config,
)
from noveum_trace.utils.exceptions import ConfigurationError


class TestTracingConfig:
    """Test TracingConfig class."""

    def test_tracing_config_defaults(self):
        """Test TracingConfig default values."""
        config = TracingConfig()

        assert config.enabled is True
        assert config.sample_rate == 1.0
        assert config.max_spans_per_trace == DEFAULT_MAX_SPANS_PER_TRACE
        assert config.capture_errors is True
        assert config.capture_stack_traces is False
        assert config.capture_performance is False

    def test_tracing_config_custom_values(self):
        """Test TracingConfig with custom values."""
        config = TracingConfig(
            enabled=False,
            sample_rate=0.5,
            max_spans_per_trace=500,
            capture_errors=False,
            capture_stack_traces=True,
            capture_performance=True,
        )

        assert config.enabled is False
        assert config.sample_rate == 0.5
        assert config.max_spans_per_trace == 500
        assert config.capture_errors is False
        assert config.capture_stack_traces is True
        assert config.capture_performance is True


class TestTransportConfig:
    """Test TransportConfig class."""

    def test_transport_config_defaults(self):
        """Test TransportConfig default values."""
        config = TransportConfig()

        assert config.endpoint == DEFAULT_ENDPOINT
        assert config.timeout == DEFAULT_TIMEOUT
        assert config.retry_attempts == DEFAULT_RETRY_ATTEMPTS
        assert config.retry_backoff == 1.0
        assert config.batch_size == DEFAULT_BATCH_SIZE
        assert config.batch_timeout == DEFAULT_BATCH_TIMEOUT
        assert config.max_queue_size == DEFAULT_MAX_QUEUE_SIZE
        assert config.compression is True

    def test_transport_config_custom_values(self):
        """Test TransportConfig with custom values."""
        config = TransportConfig(
            endpoint="https://custom.api.com",
            timeout=60,
            retry_attempts=5,
            retry_backoff=2.0,
            batch_size=200,
            batch_timeout=10.0,
            max_queue_size=2000,
            compression=False,
        )

        assert config.endpoint == "https://custom.api.com"
        assert config.timeout == 60
        assert config.retry_attempts == 5
        assert config.retry_backoff == 2.0
        assert config.batch_size == 200
        assert config.batch_timeout == 10.0
        assert config.max_queue_size == 2000
        assert config.compression is False


class TestSecurityConfig:
    """Test SecurityConfig class."""

    def test_security_config_defaults(self):
        """Test SecurityConfig default values."""
        config = SecurityConfig()

        assert config.redact_pii is False
        assert config.custom_redaction_patterns == []
        assert config.encrypt_data is True
        assert config.data_residency is None

    def test_security_config_custom_values(self):
        """Test SecurityConfig with custom values."""
        patterns = ["email", "phone"]
        config = SecurityConfig(
            redact_pii=True,
            custom_redaction_patterns=patterns,
            encrypt_data=False,
            data_residency="EU",
        )

        assert config.redact_pii is True
        assert config.custom_redaction_patterns == patterns
        assert config.encrypt_data is False
        assert config.data_residency == "EU"


class TestIntegrationConfig:
    """Test IntegrationConfig class."""

    def test_integration_config_defaults(self):
        """Test IntegrationConfig default values."""
        config = IntegrationConfig()

        assert config.langchain == {"enabled": False}
        assert config.llamaindex == {"enabled": False}
        assert config.openai == {"enabled": False}
        assert config.anthropic == {"enabled": False}

    def test_integration_config_custom_values(self):
        """Test IntegrationConfig with custom values."""
        config = IntegrationConfig(
            langchain={"enabled": True, "auto_trace": True},
            llamaindex={"enabled": True},
            openai={"enabled": True, "api_key": "test"},
            anthropic={"enabled": False},
        )

        assert config.langchain == {"enabled": True, "auto_trace": True}
        assert config.llamaindex == {"enabled": True}
        assert config.openai == {"enabled": True, "api_key": "test"}
        assert config.anthropic == {"enabled": False}


class TestConfig:
    """Test main Config class."""

    def test_config_defaults(self):
        """Test Config default values."""
        config = Config()

        assert config.project is None
        assert config.api_key is None
        assert config.environment == "development"
        assert config.debug is False
        assert config.log_level == "INFO"
        assert isinstance(config.tracing, TracingConfig)
        assert isinstance(config.transport, TransportConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.integrations, IntegrationConfig)

    def test_config_custom_values(self):
        """Test Config with custom values."""
        tracing = TracingConfig(enabled=False)
        transport = TransportConfig(endpoint="https://custom.api.com")
        security = SecurityConfig(redact_pii=True)
        integrations = IntegrationConfig(openai={"enabled": True})

        config = Config(
            project="test-project",
            api_key="test-key",
            environment="production",
            debug=True,
            log_level="DEBUG",
            tracing=tracing,
            transport=transport,
            security=security,
            integrations=integrations,
        )

        assert config.project == "test-project"
        assert config.api_key == "test-key"
        assert config.environment == "production"
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.tracing == tracing
        assert config.transport == transport
        assert config.security == security
        assert config.integrations == integrations

    def test_config_endpoint_property(self):
        """Test Config endpoint property."""
        config = Config()

        # Default endpoint
        assert config.endpoint == DEFAULT_ENDPOINT

        # Set endpoint
        config.endpoint = "https://custom.api.com"
        assert config.endpoint == "https://custom.api.com"
        assert config.transport.endpoint == "https://custom.api.com"

    def test_config_validation_valid(self):
        """Test Config validation with valid values."""
        # Should not raise any exceptions
        Config(
            tracing=TracingConfig(sample_rate=0.5, max_spans_per_trace=100),
            transport=TransportConfig(timeout=30),
        )

    def test_config_validation_invalid_sample_rate_low(self):
        """Test Config validation with invalid sample rate (too low)."""
        with pytest.raises(ConfigurationError, match="Invalid sample_rate: -0.1"):
            Config(tracing=TracingConfig(sample_rate=-0.1))

    def test_config_validation_invalid_sample_rate_high(self):
        """Test Config validation with invalid sample rate (too high)."""
        with pytest.raises(ConfigurationError, match="Invalid sample_rate: 1.5"):
            Config(tracing=TracingConfig(sample_rate=1.5))

    def test_config_validation_invalid_max_spans(self):
        """Test Config validation with invalid max spans."""
        with pytest.raises(ConfigurationError, match="Invalid max_spans_per_trace: -1"):
            Config(tracing=TracingConfig(max_spans_per_trace=-1))

    def test_config_validation_invalid_timeout(self):
        """Test Config validation with invalid timeout."""
        with pytest.raises(ConfigurationError, match="Invalid timeout: -1"):
            Config(transport=TransportConfig(timeout=-1))

    def test_config_create_basic(self):
        """Test Config.create() method with basic parameters."""
        config = Config.create(
            project="test-project",
            api_key="test-key",
            environment="production",
            debug=True,
            log_level="DEBUG",
        )

        assert config.project == "test-project"
        assert config.api_key == "test-key"
        assert config.environment == "production"
        assert config.debug is True
        assert config.log_level == "DEBUG"

    def test_config_create_with_endpoint(self):
        """Test Config.create() method with endpoint."""
        config = Config.create(endpoint="https://custom.api.com")

        assert config.endpoint == "https://custom.api.com"
        assert config.transport.endpoint == "https://custom.api.com"

    def test_config_create_with_components(self):
        """Test Config.create() method with component configurations."""
        tracing = TracingConfig(enabled=False)
        transport = TransportConfig(timeout=60)
        security = SecurityConfig(redact_pii=True)
        integrations = IntegrationConfig(openai={"enabled": True})

        config = Config.create(
            tracing=tracing,
            transport=transport,
            security=security,
            integrations=integrations,
        )

        assert config.tracing == tracing
        assert config.transport == transport
        assert config.security == security
        assert config.integrations == integrations

    def test_config_to_dict(self):
        """Test Config.to_dict() method."""
        config = Config(
            project="test-project",
            api_key="test-key",
            environment="production",
            debug=True,
            log_level="DEBUG",
        )

        data = config.to_dict()

        assert data["project"] == "test-project"
        assert data["api_key"] == "test-key"
        assert data["environment"] == "production"
        assert data["debug"] is True
        assert data["log_level"] == "DEBUG"
        assert "tracing" in data
        assert "transport" in data
        assert "security" in data
        assert "integrations" in data

    def test_config_from_dict_basic(self):
        """Test Config.from_dict() method with basic data."""
        data = {
            "project": "test-project",
            "api_key": "test-key",
            "environment": "production",
            "debug": True,
            "log_level": "DEBUG",
        }

        config = Config.from_dict(data)

        assert config.project == "test-project"
        assert config.api_key == "test-key"
        assert config.environment == "production"
        assert config.debug is True
        assert config.log_level == "DEBUG"

    def test_config_from_dict_with_endpoint(self):
        """Test Config.from_dict() method with endpoint."""
        data = {"endpoint": "https://custom.api.com"}

        config = Config.from_dict(data)

        assert config.endpoint == "https://custom.api.com"

    def test_config_from_dict_with_components(self):
        """Test Config.from_dict() method with component configurations."""
        data = {
            "tracing": {
                "enabled": False,
                "sample_rate": 0.5,
                "max_spans_per_trace": 500,
            },
            "transport": {
                "endpoint": "https://custom.api.com",
                "timeout": 60,
                "retry_attempts": 5,
            },
            "security": {
                "redact_pii": True,
                "custom_redaction_patterns": ["email", "phone"],
            },
            "integrations": {
                "openai": {"enabled": True},
                "langchain": {"enabled": True, "auto_trace": True},
            },
        }

        config = Config.from_dict(data)

        assert config.tracing.enabled is False
        assert config.tracing.sample_rate == 0.5
        assert config.tracing.max_spans_per_trace == 500
        assert config.transport.endpoint == "https://custom.api.com"
        assert config.transport.timeout == 60
        assert config.transport.retry_attempts == 5
        assert config.security.redact_pii is True
        assert config.security.custom_redaction_patterns == ["email", "phone"]
        assert config.integrations.openai == {"enabled": True}
        assert config.integrations.langchain == {"enabled": True, "auto_trace": True}

    def test_config_from_dict_with_invalid_component_types(self):
        """Test Config.from_dict() method with invalid component types."""
        data = {
            "tracing": "invalid",
            "transport": "invalid",
            "security": "invalid",
            "integrations": "invalid",
        }

        config = Config.from_dict(data)

        # Should use default configurations for invalid types
        assert isinstance(config.tracing, TracingConfig)
        assert isinstance(config.transport, TransportConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.integrations, IntegrationConfig)

    def test_config_from_dict_endpoint_precedence(self):
        """Test Config.from_dict() endpoint precedence."""
        data = {
            "endpoint": "https://top-level.com",
            "transport": {"endpoint": "https://transport-level.com"},
        }

        config = Config.from_dict(data)

        # Top-level endpoint should take precedence
        assert config.endpoint == "https://top-level.com"


class TestDeepMergeDicts:
    """Test deep merge dictionary functionality."""

    def test_deep_merge_basic(self):
        """Test basic deep merge functionality."""
        base = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}

        result = _deep_merge_dicts(base, update)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        """Test deep merge with nested dictionaries."""
        base = {"level1": {"level2": {"a": 1, "b": 2}}}
        update = {"level1": {"level2": {"b": 3, "c": 4}}}

        result = _deep_merge_dicts(base, update)

        expected = {"level1": {"level2": {"a": 1, "b": 3, "c": 4}}}

        assert result == expected

    def test_deep_merge_skip_none_values(self):
        """Test deep merge skips None values."""
        base = {"a": 1, "b": 2}
        update = {"a": None, "b": 3, "c": None}

        result = _deep_merge_dicts(base, update)

        assert result == {"a": 1, "b": 3}

    def test_deep_merge_mixed_types(self):
        """Test deep merge with mixed types."""
        base = {"dict": {"a": 1}, "list": [1, 2], "str": "base"}
        update = {"dict": {"b": 2}, "list": [3, 4], "str": "update"}

        result = _deep_merge_dicts(base, update)

        expected = {
            "dict": {"a": 1, "b": 2},
            "list": [3, 4],  # Lists are replaced, not merged
            "str": "update",
        }

        assert result == expected

    def test_deep_merge_preserves_original(self):
        """Test deep merge preserves original dictionaries."""
        base = {"a": 1, "b": {"c": 2}}
        update = {"b": {"d": 3}}

        result = _deep_merge_dicts(base, update)

        # Original dictionaries should be unchanged
        assert base == {"a": 1, "b": {"c": 2}}
        assert update == {"b": {"d": 3}}
        assert result == {"a": 1, "b": {"c": 2, "d": 3}}


class TestConfigurationLoading:
    """Test configuration loading functionality."""

    def test_load_from_environment_empty(self):
        """Test loading from environment with no variables set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("noveum_trace.core.config.os.path.exists", return_value=False):
                config = _load_from_environment()

                assert config.project is None
                assert config.api_key is None
                assert config.environment == "development"
                assert config.debug is False

    def test_load_from_environment_with_variables(self):
        """Test loading from environment with variables set."""
        env_vars = {
            "NOVEUM_PROJECT": "env-project",
            "NOVEUM_API_KEY": "env-key",
            "NOVEUM_ENVIRONMENT": "production",
            "NOVEUM_ENDPOINT": "https://env.api.com",
            "NOVEUM_DEBUG": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with patch("noveum_trace.core.config.os.path.exists", return_value=False):
                config = _load_from_environment()

                assert config.project == "env-project"
                assert config.api_key == "env-key"
                assert config.environment == "production"
                assert config.endpoint == "https://env.api.com"
                assert config.debug is True

    def test_load_from_environment_debug_values(self):
        """Test loading debug values from environment."""
        debug_values = ["true", "1", "yes", "TRUE", "True"]

        for debug_value in debug_values:
            with patch.dict(os.environ, {"NOVEUM_DEBUG": debug_value}, clear=True):
                with patch(
                    "noveum_trace.core.config.os.path.exists", return_value=False
                ):
                    config = _load_from_environment()
                    assert config.debug is True

        false_values = ["false", "0", "no", "FALSE", "False", ""]

        for debug_value in false_values:
            with patch.dict(os.environ, {"NOVEUM_DEBUG": debug_value}, clear=True):
                with patch(
                    "noveum_trace.core.config.os.path.exists", return_value=False
                ):
                    config = _load_from_environment()
                    assert config.debug is False

    def test_load_from_environment_with_config_file(self):
        """Test loading from environment with config file present."""
        file_config = Config(project="file-project", api_key="file-key")

        with patch.dict(os.environ, {"NOVEUM_PROJECT": "env-project"}, clear=True):
            with patch("noveum_trace.core.config.os.path.exists", return_value=True):
                with patch(
                    "noveum_trace.core.config._load_from_file", return_value=file_config
                ):
                    config = _load_from_environment()

                    # Environment variable should override file config
                    assert config.project == "env-project"
                    assert config.api_key == "file-key"

    def test_load_from_file_yaml(self):
        """Test loading from YAML file."""
        yaml_content = """
        project: yaml-project
        api_key: yaml-key
        environment: production
        tracing:
          enabled: false
          sample_rate: 0.5
        transport:
          endpoint: https://yaml.api.com
          timeout: 60
        """

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("noveum_trace.core.config.Path.exists", return_value=True):
                config = _load_from_file("test.yaml")

                assert config.project == "yaml-project"
                assert config.api_key == "yaml-key"
                assert config.environment == "production"
                assert config.tracing.enabled is False
                assert config.tracing.sample_rate == 0.5
                assert config.transport.endpoint == "https://yaml.api.com"
                assert config.transport.timeout == 60

    def test_load_from_file_json(self):
        """Test loading from JSON file."""
        json_content = {
            "project": "json-project",
            "api_key": "json-key",
            "environment": "production",
            "tracing": {"enabled": False, "sample_rate": 0.5},
            "transport": {"endpoint": "https://json.api.com", "timeout": 60},
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(json_content))):
            with patch("noveum_trace.core.config.Path.exists", return_value=True):
                config = _load_from_file("test.json")

                assert config.project == "json-project"
                assert config.api_key == "json-key"
                assert config.environment == "production"
                assert config.tracing.enabled is False
                assert config.tracing.sample_rate == 0.5
                assert config.transport.endpoint == "https://json.api.com"
                assert config.transport.timeout == 60

    def test_load_from_file_not_found(self):
        """Test loading from non-existent file."""
        with patch("noveum_trace.core.config.Path.exists", return_value=False):
            with pytest.raises(
                ConfigurationError, match="Configuration file not found"
            ):
                _load_from_file("nonexistent.yaml")

    def test_load_from_file_unsupported_format(self):
        """Test loading from unsupported file format."""
        with patch("noveum_trace.core.config.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="")):
                with pytest.raises(
                    ConfigurationError, match="Unsupported configuration file format"
                ):
                    _load_from_file("test.txt")

    def test_load_from_file_invalid_yaml(self):
        """Test loading from invalid YAML file."""
        invalid_yaml = "invalid: yaml: content:"

        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with patch("noveum_trace.core.config.Path.exists", return_value=True):
                with pytest.raises(
                    ConfigurationError, match="Failed to parse configuration file"
                ):
                    _load_from_file("test.yaml")

    def test_load_from_file_invalid_json(self):
        """Test loading from invalid JSON file."""
        invalid_json = '{"invalid": json content}'

        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with patch("noveum_trace.core.config.Path.exists", return_value=True):
                with pytest.raises(
                    ConfigurationError, match="Failed to parse configuration file"
                ):
                    _load_from_file("test.json")

    def test_load_from_file_empty_file(self):
        """Test loading from empty file."""
        with patch("builtins.open", mock_open(read_data="")):
            with patch("noveum_trace.core.config.Path.exists", return_value=True):
                config = _load_from_file("test.yaml")

                # Should create default config
                assert isinstance(config, Config)
                assert config.project is None

    def test_load_from_file_io_error(self):
        """Test loading from file with IO error."""
        with patch("noveum_trace.core.config.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=OSError("File read error")):
                with pytest.raises(
                    ConfigurationError, match="Failed to load configuration file"
                ):
                    _load_from_file("test.yaml")


class TestConfigurationModule:
    """Test configuration module functions."""

    def teardown_method(self):
        """Clean up after each test."""
        import noveum_trace.core.config as config_module

        config_module._config = None

    def test_configure_with_none(self):
        """Test configure with None."""
        with patch("noveum_trace.core.config._load_from_environment") as mock_load:
            mock_config = Mock(spec=Config)
            mock_load.return_value = mock_config

            configure(None)

            assert get_config() == mock_config

    def test_configure_with_empty_dict(self):
        """Test configure with empty dictionary."""
        with patch("noveum_trace.core.config._load_from_environment") as mock_load:
            mock_config = Mock(spec=Config)
            mock_load.return_value = mock_config

            configure({})

            assert get_config() == mock_config

    def test_configure_with_config_instance(self):
        """Test configure with Config instance."""
        config = Config(project="test-project")

        configure(config)

        assert get_config() == config

    def test_configure_with_dict(self):
        """Test configure with dictionary."""
        config_data = {
            "project": "test-project",
            "api_key": "test-key",
            "environment": "production",
        }

        with patch("noveum_trace.core.config._load_from_environment") as mock_load:
            mock_env_config = Config()
            mock_load.return_value = mock_env_config

            configure(config_data)

            config = get_config()
            assert config.project == "test-project"
            assert config.api_key == "test-key"
            assert config.environment == "production"

    def test_configure_with_dict_endpoint_override(self):
        """Test configure with dictionary containing endpoint override."""
        config_data = {
            "endpoint": "https://custom.api.com",
            "transport": {"endpoint": "https://transport.api.com"},
        }

        with patch("noveum_trace.core.config._load_from_environment") as mock_load:
            mock_env_config = Config()
            mock_load.return_value = mock_env_config

            configure(config_data)

            config = get_config()
            # Top-level endpoint should override transport endpoint
            assert config.endpoint == "https://custom.api.com"

    def test_configure_with_file_path_string(self):
        """Test configure with file path string."""
        mock_config = Config(project="file-project")

        with patch("noveum_trace.core.config._load_from_file") as mock_load:
            mock_load.return_value = mock_config

            configure("test.yaml")

            assert get_config() == mock_config
            mock_load.assert_called_once_with("test.yaml")

    def test_configure_with_file_path_object(self):
        """Test configure with Path object."""
        mock_config = Config(project="file-project")
        file_path = Path("test.yaml")

        with patch("noveum_trace.core.config._load_from_file") as mock_load:
            mock_load.return_value = mock_config

            configure(file_path)

            assert get_config() == mock_config
            mock_load.assert_called_once_with(file_path)

    def test_configure_with_invalid_type(self):
        """Test configure with invalid type."""
        with pytest.raises(ConfigurationError, match="Invalid configuration type"):
            configure(123)

    def test_get_config_auto_configure(self):
        """Test get_config auto-configures when not set."""
        with patch("noveum_trace.core.config._load_from_environment") as mock_load:
            mock_config = Mock(spec=Config)
            mock_load.return_value = mock_config

            config = get_config()

            assert config == mock_config
            mock_load.assert_called_once()

    def test_get_config_existing(self):
        """Test get_config returns existing configuration."""
        existing_config = Config(project="existing-project")
        configure(existing_config)

        config = get_config()

        assert config == existing_config


class TestConfigurationIntegration:
    """Integration tests for configuration functionality."""

    def teardown_method(self):
        """Clean up after each test."""
        import noveum_trace.core.config as config_module

        config_module._config = None

    def test_complete_configuration_workflow(self):
        """Test complete configuration workflow."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "project": "file-project",
                    "api_key": "file-key",
                    "tracing": {"enabled": False, "sample_rate": 0.5},
                },
                f,
            )
            temp_file = f.name

        try:
            # Load from file
            configure(temp_file)
            config = get_config()

            assert config.project == "file-project"
            assert config.api_key == "file-key"
            assert config.tracing.enabled is False
            assert config.tracing.sample_rate == 0.5

            # Convert to dict and back
            config_dict = config.to_dict()
            new_config = Config.from_dict(config_dict)

            assert new_config.project == config.project
            assert new_config.api_key == config.api_key
            assert new_config.tracing.enabled == config.tracing.enabled
            assert new_config.tracing.sample_rate == config.tracing.sample_rate

        finally:
            # Clean up
            os.unlink(temp_file)

    def test_environment_override_workflow(self):
        """Test environment variable override workflow."""
        # Set environment variables
        env_vars = {
            "NOVEUM_PROJECT": "env-project",
            "NOVEUM_API_KEY": "env-key",
            "NOVEUM_ENVIRONMENT": "production",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with patch("noveum_trace.core.config.os.path.exists", return_value=False):
                # Configure should merge environment variables
                configure({"project": "dict-project", "environment": "development"})

                config = get_config()

                # Dict values should be used since configure was called with explicit values
                assert config.project == "dict-project"
                assert config.environment == "development"
                # API key should come from environment since not in dict
                assert config.api_key == "env-key"

    def test_nested_configuration_merge(self):
        """Test nested configuration merging."""
        # Start with environment config
        with patch("noveum_trace.core.config._load_from_environment") as mock_load:
            env_config = Config(
                project="env-project",
                tracing=TracingConfig(enabled=True, sample_rate=1.0),
                transport=TransportConfig(endpoint="https://env.api.com", timeout=30),
            )
            mock_load.return_value = env_config

            # Configure with partial override
            configure(
                {
                    "tracing": {"enabled": False, "capture_performance": True},
                    "transport": {"timeout": 60},
                }
            )

            config = get_config()

            # Should merge nested configurations
            assert config.project == "env-project"  # From environment
            assert config.tracing.enabled is False  # From dict
            assert config.tracing.sample_rate == 1.0  # From environment
            assert config.tracing.capture_performance is True  # From dict
            assert (
                config.transport.endpoint == "https://env.api.com"
            )  # From environment
            assert config.transport.timeout == 60  # From dict

    def test_configuration_validation_integration(self):
        """Test configuration validation in integration scenarios."""
        # Valid configuration should work
        configure(
            {
                "tracing": {"sample_rate": 0.5, "max_spans_per_trace": 100},
                "transport": {"timeout": 30},
            }
        )

        config = get_config()
        assert config.tracing.sample_rate == 0.5

        # Invalid configuration should raise error
        with pytest.raises(ConfigurationError):
            # Create invalid config directly to trigger validation
            Config(tracing=TracingConfig(sample_rate=2.0))  # Invalid: > 1.0

    def test_multiple_configuration_calls(self):
        """Test multiple configuration calls."""
        # First configuration
        configure({"project": "first-project"})
        config1 = get_config()
        assert config1.project == "first-project"

        # Second configuration should replace first
        configure({"project": "second-project"})
        config2 = get_config()
        assert config2.project == "second-project"

        # Should be different instances
        assert config1 is not config2
