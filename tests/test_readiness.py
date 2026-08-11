"""readiness 두 분기(성공/실패)를 실제 Postgres 없이 검증한다.

Postgres를 띄운 상태의 통합 검증은 README의 `docker compose up -d db` 절차 참고.
"""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import db_session
from app.core.config import Settings
from app.main import create_app


class _FakeSession:
    """execute()만 흉내내는 최소 스텁."""

    def __init__(self, *, fail: bool) -> None:
        self._fail = fail

    async def execute(self, statement: object) -> None:
        if self._fail:
            raise ConnectionRefusedError("connection refused")


async def _client_with_session(settings: Settings, *, fail: bool) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)

    async def _override() -> AsyncIterator[_FakeSession]:
        yield _FakeSession(fail=fail)

    app.dependency_overrides[db_session] = _override

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac,
    ):
        yield ac


@pytest.fixture
async def ready_client(settings: Settings) -> AsyncIterator[AsyncClient]:
    async for ac in _client_with_session(settings, fail=False):
        yield ac


@pytest.fixture
async def broken_client(settings: Settings) -> AsyncIterator[AsyncClient]:
    async for ac in _client_with_session(settings, fail=True):
        yield ac


async def test_ready_returns_200_when_db_reachable(ready_client: AsyncClient) -> None:
    resp = await ready_client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


async def test_ready_returns_503_when_db_unreachable(broken_client: AsyncClient) -> None:
    resp = await broken_client.get("/api/v1/health/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unavailable"
