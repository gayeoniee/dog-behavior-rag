from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    # embedding_warmup=False: 테스트가 수 GB 모델을 로딩하려 들면 안 된다.
    # 검색 경로 테스트는 dependency_overrides로 가짜를 주입한다.
    return Settings(app_env="test", log_level="WARNING", embedding_warmup=False)


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
