"""Unit tests for the client-initiated NovaSynth call loop.

Driven through a fake ``httpx.Client`` and a fake clock, so the queued /
ready / expired / dial-failure paths are exercised with no network and no
real sleeping — the tests assert on ``clock.sleeps`` rather than wall time.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from noveum_trace import novasynth
from noveum_trace.novasynth import CallQueue
from noveum_trace.utils.exceptions import ConfigurationError

BASE = "https://api.noveum.ai/api"
BULK = f"{BASE}/v1/novasynth/runs/bulk"


class _Resp:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}

    def json(self):
        return self._json


class _FakeClient:
    """Records every get/post. Queued responses are consumed in order and the
    last one repeats, so a loop keeps seeing the same state."""

    calls: list = []
    get_resps: list = []
    post_resps: list = []
    post_exc = None

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @staticmethod
    def _next(queue, default):
        if not queue:
            return default
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def get(self, url, headers=None, params=None):
        _FakeClient.calls.append(
            {"method": "GET", "url": url, "headers": headers, "params": params}
        )
        return _FakeClient._next(_FakeClient.get_resps, _Resp(200, []))

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
        if _FakeClient.post_exc is not None:
            raise _FakeClient.post_exc
        return _FakeClient._next(_FakeClient.post_resps, _Resp(202))


class _Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(novasynth, "time", c)
    return c


@pytest.fixture
def logs(caplog, monkeypatch):
    """The SDK sets ``propagate = False`` on the ``noveum_trace`` logger, which
    stops records before they reach caplog's root-attached handler."""
    monkeypatch.setattr(logging.getLogger("noveum_trace"), "propagate", True)
    caplog.set_level(logging.WARNING, logger=novasynth._log.name)
    return caplog


def _patch(monkeypatch, get_resps=None, post_resps=None, post_exc=None):
    _FakeClient.calls = []
    _FakeClient.get_resps = list(get_resps or [])
    _FakeClient.post_resps = list(post_resps or [])
    _FakeClient.post_exc = post_exc
    monkeypatch.setattr(httpx, "Client", lambda **kw: _FakeClient(**kw))


def _gets():
    return [c for c in _FakeClient.calls if c["method"] == "GET"]


def _posts():
    return [c for c in _FakeClient.calls if c["method"] == "POST"]


def _row(run_id, status, **extra):
    row = {
        "runId": run_id,
        "status": status,
        "dialNumber": None,
        "dialWindowOpensAt": None,
        "dialWindowClosesAt": None,
        "dialWindowSeconds": None,
        "persona": {"id": "p_1", "name": "Asha Menon"},
        "scenario": {"id": "s_1", "name": "Delayed refund"},
        "agentVariables": {"user_id": "U-42"},
        "traceId": None,
        "result": None,
    }
    row.update(extra)
    return row


def _ready(run_id, number="+918065481242", seconds=180, **extra):
    return _row(
        run_id,
        "ready_to_dial",
        dialNumber=number,
        dialWindowSeconds=seconds,
        **extra,
    )


def _queue(run_ids, **kwargs):
    kwargs.setdefault("api_key", "k")
    kwargs.setdefault("base_url", BASE)
    return CallQueue(run_ids, **kwargs)


# --- construction -----------------------------------------------------------


def test_missing_api_key_raises_rather_than_faking_a_batch():
    with pytest.raises(ConfigurationError):
        CallQueue(["a"], api_key="", base_url=BASE)


def test_credentials_fall_back_to_sdk_config(monkeypatch):
    class _Cfg:
        api_key = "from-config"
        endpoint = "https://example.test/api/"

    monkeypatch.setattr(novasynth, "get_config", lambda: _Cfg())
    q = CallQueue(["a"])
    assert q.api_key == "from-config"
    assert q.base_url == "https://example.test/api"  # trailing slash trimmed


# --- the bulk poll ----------------------------------------------------------


def test_one_bulk_get_with_comma_joined_run_ids(monkeypatch, clock):
    _patch(
        monkeypatch,
        get_resps=[_Resp(200, [_row("a", "completed"), _row("b", "completed")])],
    )
    q = _queue(["a", "b"], organization_slug="acme")
    assert list(q.iter_calls()) == []
    assert len(_gets()) == 1
    get = _gets()[0]
    assert get["url"] == BULK
    assert get["headers"] == {"Authorization": "Bearer k"}
    assert get["params"] == {"organizationSlug": "acme", "run_ids": "a,b"}


def test_only_ready_to_dial_runs_are_yielded(monkeypatch, clock):
    _patch(
        monkeypatch,
        get_resps=[
            _Resp(200, [_row("a", "queued"), _row("b", "arming")]),
            _Resp(200, [_ready("a"), _row("b", "arming")]),
            _Resp(200, [_row("a", "completed"), _row("b", "completed")]),
        ],
    )
    q = _queue(["a", "b"])
    seen = [call.run_id for call in q.iter_calls()]
    assert seen == ["a"]


def test_ready_run_carries_the_agent_facing_half_only(monkeypatch, clock):
    _patch(monkeypatch, get_resps=[_Resp(200, [_ready("a")]), _Resp(200, [])])
    q = _queue(["a"])
    call = next(q.iter_calls())
    assert call.dial_number == "+918065481242"
    assert call.agent_variables == {"user_id": "U-42"}
    assert call.persona == {"id": "p_1", "name": "Asha Menon"}


def test_unknown_status_is_not_terminal(monkeypatch, clock):
    """An older SDK must keep polling a platform state it has never heard of."""
    _patch(
        monkeypatch,
        get_resps=[
            _Resp(200, [_row("a", "warming_up_v2")]),
            _Resp(200, [_row("a", "completed")]),
        ],
    )
    q = _queue(["a"])
    assert list(q.iter_calls()) == []
    assert len(_gets()) == 2  # it did not give up on the unknown status
    assert q.statuses == {"a": "completed"}


def test_poll_backs_off_once_the_call_is_up(monkeypatch, clock):
    _patch(
        monkeypatch,
        get_resps=[
            _Resp(200, [_row("a", "in_call")]),
            _Resp(200, [_row("a", "completed")]),
        ],
    )
    q = _queue(["a"])
    list(q.iter_calls())
    assert clock.sleeps == [10.0]


def test_poll_failure_is_not_fatal(monkeypatch, clock):
    _patch(
        monkeypatch,
        get_resps=[_Resp(503), _Resp(200, [_row("a", "completed")])],
    )
    q = _queue(["a"])
    assert list(q.iter_calls()) == []
    assert q.statuses == {"a": "completed"}


# --- the dial window --------------------------------------------------------


def test_seconds_remaining_uses_the_local_anchor_not_server_wall_time(
    monkeypatch, clock
):
    _patch(
        monkeypatch,
        get_resps=[
            _Resp(
                200,
                [
                    _ready(
                        "a",
                        seconds=180,
                        # Deliberately nonsense timestamps: they are for display
                        # and support, never for arithmetic.
                        dialWindowOpensAt="2000-01-01T00:00:00Z",
                        dialWindowClosesAt="2000-01-01T00:03:00Z",
                    )
                ],
            )
        ],
    )
    q = _queue(["a"])
    call = next(q.iter_calls())
    assert call.seconds_remaining == 180.0
    clock.sleep(30)
    assert call.seconds_remaining == 150.0


def test_run_whose_window_closed_is_skipped_and_recorded(monkeypatch, clock, logs):
    _patch(
        monkeypatch,
        get_resps=[
            _Resp(200, [_ready("a"), _ready("b")]),
            _Resp(200, [_row("a", "completed"), _row("b", "expired")]),
        ],
    )
    q = _queue(["a", "b"])
    seen = []
    for call in q.iter_calls():
        seen.append(call.run_id)
        clock.sleep(200)  # the dialler blows through b's window

    assert seen == ["a"]  # b was never handed out
    assert "run b skipped" in logs.text
    assert [p["json"]["code"] for p in _posts()] == ["abandoned"]
    assert q.summary() == {"completed": 1, "expired": 1}


# --- dial failure reporting -------------------------------------------------


def test_report_failed_payload_and_stable_event_id(monkeypatch, clock):
    _patch(monkeypatch, get_resps=[_Resp(200, [_ready("a")])])
    q = _queue(["a"], organization_slug="acme")
    call = next(q.iter_calls())
    call.report_failed(reason="line busy", code="busy", provider_call_id="CA123")
    call.report_failed(reason="line busy", code="busy", provider_call_id="CA123")

    posts = _posts()
    assert posts[0]["url"] == f"{BASE}/v1/novasynth/runs/a/dial-failed"
    assert posts[0]["params"] == {"organizationSlug": "acme"}
    assert posts[0]["json"]["code"] == "busy"
    assert posts[0]["json"]["reason"] == "line busy"
    assert posts[0]["json"]["providerCallId"] == "CA123"
    # Same key on a resend, so the platform can dedupe it.
    assert posts[0]["json"]["eventId"] == posts[1]["json"]["eventId"]


@pytest.mark.parametrize("status", [400, 404, 500])
def test_report_failed_never_raises_on_a_bad_response(monkeypatch, clock, status):
    _patch(
        monkeypatch,
        get_resps=[_Resp(200, [_ready("a")])],
        post_resps=[_Resp(status)],
    )
    q = _queue(["a"])
    call = next(q.iter_calls())
    call.report_failed(reason="busy", code="busy")  # must not raise


def test_report_failed_never_raises_on_a_network_error(monkeypatch, clock):
    _patch(
        monkeypatch,
        get_resps=[_Resp(200, [_ready("a")])],
        post_exc=RuntimeError("connection reset"),
    )
    q = _queue(["a"])
    call = next(q.iter_calls())
    call.report_failed(reason="busy", code="busy")  # must not raise


def test_409_is_ignored_because_the_dial_actually_connected(monkeypatch, clock, logs):
    _patch(
        monkeypatch,
        get_resps=[_Resp(200, [_ready("a")])],
        post_resps=[_Resp(409)],
    )
    q = _queue(["a"])
    call = next(q.iter_calls())
    call.report_failed(reason="busy", code="busy")
    assert [r for r in logs.records if r.levelno >= logging.ERROR] == []


# --- lifecycle --------------------------------------------------------------


def test_break_in_the_body_releases_the_number_on_exit(monkeypatch, clock, logs):
    _patch(monkeypatch, get_resps=[_Resp(200, [_ready("a")])])
    with _queue(["a"]) as q:
        for _call in q.iter_calls():
            break

    posts = _posts()
    assert len(posts) == 1
    assert posts[0]["url"].endswith("/runs/a/dial-failed")
    assert posts[0]["json"]["code"] == "abandoned"
    assert "never dialled" in logs.text


def test_a_dialled_run_is_not_abandoned_on_exit(monkeypatch, clock):
    _patch(
        monkeypatch,
        get_resps=[_Resp(200, [_ready("a")]), _Resp(200, [_row("a", "completed")])],
    )
    with _queue(["a"]) as q:
        for call in q.iter_calls():
            assert call.wait_until_finished(provider_call_id="CA1") == "completed"
    assert _posts() == []


def test_summary_accounts_for_every_run_in_the_batch(monkeypatch, clock):
    _patch(
        monkeypatch,
        get_resps=[_Resp(200, [_row("a", "completed"), _row("b", "failed")])],
    )
    q = _queue(["a", "b", "c"])  # the platform never reports on c
    q._poll()
    assert q.summary() == {"completed": 1, "failed": 1, "queued": 1}
    assert sum(q.summary().values()) == len(q.run_ids)
