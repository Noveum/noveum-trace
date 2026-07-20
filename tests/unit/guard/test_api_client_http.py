"""Unit tests for HttpGuardAPIClient — the server-authoritative Guard backend.

The backend is exercised through a fake ``httpx.Client`` injected via
monkeypatch, so these tests assert the exact HTTP contract (URLs, request
bodies, response mapping) without a network.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from noveum_trace.guard.api_client_http import HttpGuardAPIClient
from noveum_trace.guard.exceptions import GuardBackendUnavailable


def _call_id() -> str:
    return str(uuid.uuid4())


class _Resp:
    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}

    def json(self):
        return self._json


class _FakeClient:
    """Records every get/post so tests can assert the HTTP contract."""

    calls: list[dict] = []

    def __init__(self, *, get_resp=None, get_exc=None, post_exc=None):
        self._get_resp = get_resp
        self._get_exc = get_exc
        self._post_exc = post_exc

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url, headers=None, params=None):
        _FakeClient.calls.append(
            {"method": "GET", "url": url, "headers": headers, "params": params}
        )
        if self._get_exc is not None:
            raise self._get_exc
        return self._get_resp

    def post(self, url, headers=None, params=None, json=None):
        _FakeClient.calls.append(
            {
                "method": "POST",
                "url": url,
                "headers": headers,
                "params": params,
                "json": json,
            }
        )
        if self._post_exc is not None:
            raise self._post_exc
        return _Resp(202, {"success": True})


def _patch(monkeypatch, **kwargs) -> None:
    _FakeClient.calls = []
    monkeypatch.setattr(httpx, "Client", lambda **kw: _FakeClient(**kwargs))


# ---------------------------------------------------------------------------
# get_state — GET /policies/state
# ---------------------------------------------------------------------------


class TestGetState:
    def test_reads_the_requested_window(self, monkeypatch):
        _patch(
            monkeypatch,
            get_resp=_Resp(200, {"cost": {"30d_rolling": 1247.81, "1d_rolling": 12.0}}),
        )
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        assert api.get_state("proj", "30d_rolling") == {"spend": 1247.81}

    def test_missing_window_reads_zero(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(200, {"cost": {}}))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        assert api.get_state("proj", "7d_rolling") == {"spend": 0.0}

    def test_hits_project_state_url(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(200, {"cost": {}}))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        api.get_state("proj-1")
        call = _FakeClient.calls[0]
        assert call["url"] == "https://api.noveum.ai/v1/projects/proj-1/policies/state"
        assert call["headers"]["Authorization"] == "Bearer k"

    def test_org_slug_sent_as_query_param(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(200, {"cost": {}}))
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", organization_slug="acme"
        )
        api.get_state("proj")
        assert _FakeClient.calls[0]["params"] == {"organizationSlug": "acme"}

    def test_non_200_raises(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(500))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        with pytest.raises(GuardBackendUnavailable):
            api.get_state("proj")

    def test_network_error_raises(self, monkeypatch):
        _patch(monkeypatch, get_exc=httpx.ConnectError("boom"))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        with pytest.raises(GuardBackendUnavailable):
            api.get_state("proj")

    def test_strips_api_suffix_from_base_url(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(200, {"cost": {}}))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai/api")
        api.get_state("proj")
        assert _FakeClient.calls[0]["url"].startswith(
            "https://api.noveum.ai/v1/projects/proj"
        )


# ---------------------------------------------------------------------------
# report_usage — batched POST /policies/usage
# ---------------------------------------------------------------------------


class TestReportUsage:
    def test_event_has_backend_shape(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        cid = _call_id()
        api.report_usage(cid, "proj", 0.0123, "gpt-4o", 800, 200)
        api.close()  # drains

        post = [c for c in _FakeClient.calls if c["method"] == "POST"][0]
        assert post["url"] == "https://api.noveum.ai/v1/projects/proj/policies/usage"
        event = post["json"][0]
        assert event["model"] == "gpt-4o"
        assert event["inputTokens"] == 800
        assert event["outputTokens"] == 200
        assert event["costUsd"] == 0.0123
        assert event["requestCount"] == 1
        assert event["eventId"] == cid  # idempotency key = call_id
        assert event["timestamp"].endswith("Z")

    def test_auto_flush_when_batch_full(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k",
            base_url="https://api.noveum.ai",
            flush_interval=3600,
            batch_max=2,
        )
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")
        assert [c for c in _FakeClient.calls if c["method"] == "POST"] == []
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")  # hits batch_max → flush
        posts = [c for c in _FakeClient.calls if c["method"] == "POST"]
        assert len(posts) == 1
        assert len(posts[0]["json"]) == 2
        api.close()

    def test_events_grouped_by_scope(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        api.report_usage(_call_id(), "proj-a", 1.0, "gpt-4o")
        api.report_usage(_call_id(), "proj-b", 1.0, "gpt-4o")
        api.close()
        posts = [c for c in _FakeClient.calls if c["method"] == "POST"]
        urls = {c["url"] for c in posts}
        assert urls == {
            "https://api.noveum.ai/v1/projects/proj-a/policies/usage",
            "https://api.noveum.ai/v1/projects/proj-b/policies/usage",
        }

    def test_negative_cost_is_dropped(self, monkeypatch):
        _patch(monkeypatch)
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        api.report_usage(_call_id(), "proj", -1.0, "gpt-4o")
        api.close()
        assert [c for c in _FakeClient.calls if c["method"] == "POST"] == []

    def test_push_failure_is_swallowed(self, monkeypatch):
        _patch(monkeypatch, post_exc=httpx.ConnectError("down"))
        api = HttpGuardAPIClient(
            api_key="k", base_url="https://api.noveum.ai", flush_interval=3600
        )
        api.report_usage(_call_id(), "proj", 1.0, "gpt-4o")
        api.close()  # must not raise


# ---------------------------------------------------------------------------
# reserve / reconcile — unsupported by the HTTP backend
# ---------------------------------------------------------------------------


class TestReservationUnsupported:
    def test_supports_reservation_is_false(self):
        api = HttpGuardAPIClient(api_key="k")
        assert api.supports_reservation is False

    def test_reserve_raises(self):
        api = HttpGuardAPIClient(api_key="k")
        with pytest.raises(NotImplementedError):
            api.reserve(_call_id(), "proj", 1.0, 100.0)

    def test_reconcile_raises(self):
        api = HttpGuardAPIClient(api_key="k")
        with pytest.raises(NotImplementedError):
            api.reconcile(_call_id(), "proj", 1.0)


# ---------------------------------------------------------------------------
# fetch_remote_policies — GET /policies/effective + shape mapping
# ---------------------------------------------------------------------------


class TestFetchRemotePolicies:
    def test_maps_cost_cap_shape(self, monkeypatch):
        _patch(
            monkeypatch,
            get_resp=_Resp(
                200,
                [
                    {
                        "policyId": "p1",
                        "name": "Monthly budget",
                        "type": "COST_CAP",
                        "enabled": True,
                        "failClosed": True,
                        "config": {
                            "window": "30d_rolling",
                            "maxUsd": 1500,
                            "action": "BLOCK",
                        },
                    }
                ],
            ),
        )
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        result = api.fetch_remote_policies("proj")
        assert result == [
            {
                "type": "cost_cap",
                "name": "Monthly budget",
                "fail_closed": True,
                "max_usd": 1500,
                "window": "30d_rolling",
            }
        ]

    def test_hits_effective_url(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(200, []))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        api.fetch_remote_policies("proj")
        assert (
            _FakeClient.calls[0]["url"]
            == "https://api.noveum.ai/v1/projects/proj/policies/effective"
        )

    def test_disabled_policy_skipped(self, monkeypatch):
        _patch(
            monkeypatch,
            get_resp=_Resp(
                200,
                [
                    {
                        "name": "off",
                        "type": "COST_CAP",
                        "enabled": False,
                        "config": {"maxUsd": 10, "window": "1d_rolling"},
                    }
                ],
            ),
        )
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        assert api.fetch_remote_policies("proj") == []

    def test_unknown_type_skipped(self, monkeypatch):
        _patch(
            monkeypatch,
            get_resp=_Resp(
                200,
                [{"name": "x", "type": "SOMETHING_NEW", "enabled": True, "config": {}}],
            ),
        )
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        assert api.fetch_remote_policies("proj") == []

    def test_non_200_raises(self, monkeypatch):
        _patch(monkeypatch, get_resp=_Resp(403))
        api = HttpGuardAPIClient(api_key="k", base_url="https://api.noveum.ai")
        with pytest.raises(GuardBackendUnavailable):
            api.fetch_remote_policies("proj")
