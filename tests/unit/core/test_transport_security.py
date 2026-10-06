"""Insecure transport is refused unless explicitly opted in."""

from unittest.mock import patch

import pytest

from noveum_trace.core.config import Config, TransportConfig, get_config
from noveum_trace.transport.http_transport import HttpTransport
from noveum_trace.utils.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def no_env_opt_in(monkeypatch):
    # conftest opts every test in; these tests check the default.
    monkeypatch.delenv("NOVEUM_ALLOW_INSECURE_TRANSPORT", raising=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"endpoint": "http://example.com/api"},
        {"endpoint": "http://localhost:8080"},
        {"endpoint": "http://127.0.0.1:8080"},
        {"transport": TransportConfig(ssl_verify=False)},
    ],
)
def test_insecure_refused_by_default(kwargs):
    with pytest.raises(ConfigurationError, match="Insecure transport"):
        Config.create(**kwargs)


def test_credentials_in_url_refused_without_echoing_them():
    with pytest.raises(ConfigurationError) as exc:
        Config.create(endpoint="https://user:s3cret@api.example.com")
    assert "s3cret" not in str(exc.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"endpoint": "https://api.example.com"},
        {"transport": TransportConfig(ssl_verify=False, ca_bundle="/ca.pem")},
        {
            "transport": TransportConfig(
                endpoint="http://localhost:8080", allow_insecure_transport=True
            )
        },
        {"transport": TransportConfig(ssl_verify=False, allow_insecure_transport=True)},
    ],
)
def test_secure_or_opted_in_allowed(kwargs):
    Config.create(**kwargs)


def test_opt_in_survives_dict_round_trip():
    config = Config.create(
        transport=TransportConfig(ssl_verify=False, allow_insecure_transport=True)
    )
    assert Config.from_dict(config.to_dict()).transport.allow_insecure_transport


def test_env_var_opt_in(monkeypatch):
    from noveum_trace.core import config as config_module

    monkeypatch.setenv("NOVEUM_ALLOW_INSECURE_TRANSPORT", "true")
    Config.create(endpoint="http://localhost:8080")

    monkeypatch.setenv("NOVEUM_ENDPOINT", "http://localhost:8080")
    monkeypatch.setattr(config_module, "_config", None)
    assert get_config().transport.endpoint == "http://localhost:8080"


@pytest.mark.disable_transport_mocking
def test_redirects_are_never_followed():
    with patch("noveum_trace.transport.http_transport.BatchProcessor"):
        transport = HttpTransport(Config.create(api_key="k"))
    assert transport.session.max_redirects == 0
