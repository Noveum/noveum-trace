import httpx
import pytest


@pytest.fixture(autouse=True)
def block_real_httpx(monkeypatch):
    """The repo-wide prevent_real_api_calls fixture patches requests/urllib3/
    urllib but not httpx, which is what this package uses. A forgotten patch
    here would arm a real session against a real number, so fail loudly."""

    def _boom(**kwargs):
        raise AssertionError("unmocked httpx call")

    monkeypatch.setattr(httpx, "Client", _boom)
