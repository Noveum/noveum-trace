"""The API key must never appear in SDK log output, even with debug logging on."""

import io
import logging
from unittest.mock import Mock, patch

from noveum_trace.core.config import Config, SecurityConfig
from noveum_trace.transport import http_transport
from noveum_trace.transport.http_transport import HttpTransport

KEY = "nv_CANARY_KEY_9f2c"


def test_debug_logs_never_contain_api_key(monkeypatch):
    monkeypatch.setenv("NOVEUM_DEBUG", "true")
    # SDK loggers don't propagate, so attach directly to the transport logger.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    log = http_transport.logger
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        config = Config.create(
            api_key=KEY, project="p", endpoint="https://api.test.com"
        )
        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)
        response = Mock(status_code=200, text="{}", headers={"Set-Cookie": KEY})
        transport.session.post = Mock(return_value=response)
        transport._send_trace_batch([{"trace_id": "t1", "name": "n", "spans": []}])
        transport._send_request({"trace_id": "t2", "name": "n", "spans": []})
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)

    output = buf.getvalue()
    assert "HTTP_REQUEST" in output  # debug logging actually ran
    assert KEY not in output


def test_config_repr_hides_secrets():
    config = Config.create(
        api_key=KEY,
        security=SecurityConfig(pii_enabled=True, pii_salt="salt_CANARY"),
    )
    text = repr(config)
    assert KEY not in text
    assert "salt_CANARY" not in text
