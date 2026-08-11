from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(app_env="test", log_level="WARNING")


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    """실제 서버/DB 없이 ASGI 앱에 직접 요청한다.

    lifespan을 태워야 app.state.db_session_factory가 준비된다.
    """
    app = create_app(settings)
    transport = ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test") as ac,
    ):
        yield ac
