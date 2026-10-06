"""Insecure transport is refused unless explicitly opted in."""

from unittest.mock import Mock, patch

import pytest

from noveum_trace.core.config import Config, TransportConfig, get_config
from noveum_trace.transport.http_transport import HttpTransport
from noveum_trace.utils.exceptions import ConfigurationError, TransportError


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


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:s3cret@api.example.com",
        "ftp://user:s3cret@api.example.com",  # would fail the scheme check
        "https://user:s3cr^t@api.example.com",  # would fail the format check
    ],
)
def test_credentials_in_url_refused_without_echoing_them(endpoint):
    with pytest.raises(ConfigurationError, match="credentials") as exc:
        Config.create(endpoint=endpoint)
    assert "s3cr" not in str(exc.value)


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


def test_string_false_from_config_file_does_not_opt_in():
    with pytest.raises(ConfigurationError, match="Insecure transport"):
        Config.from_dict(
            {
                "transport": {
                    "endpoint": "http://example.com",
                    "allow_insecure_transport": "false",
                }
            }
        )


def test_endpoint_setter_validates_and_keeps_old_value():
    config = Config.create()
    with pytest.raises(ConfigurationError, match="Insecure transport"):
        config.endpoint = "http://example.com"
    assert config.endpoint.startswith("https://")


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


@pytest.mark.parametrize("path", ["batch", "image"])
def test_redirect_response_is_not_treated_as_delivered(path):
    with patch("noveum_trace.transport.http_transport.BatchProcessor"):
        transport = HttpTransport(Config.create(api_key="k"))
    response = Mock(status_code=302, text="", headers={})  # no Location header
    transport.session.post = Mock(return_value=response)
    with pytest.raises(TransportError, match="unexpected status 302"):
        if path == "batch":
            transport._send_trace_batch([{"trace_id": "t"}])
        else:
            transport._send_single_image(
                {"image_uuid": "i", "image_data": b"x", "metadata": {}}
            )
