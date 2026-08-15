from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app

# **테스트는 개발자의 .env를 읽지 않는다.**
#
# 이걸 안 하면 테스트가 그 PC의 설정에 묶인다. 실제로 2026-08-15에 임베딩 모델을
# 바꾸며 EMBEDDING_DIM을 512로 내리자 **테스트 25개가 한꺼번에 깨졌다** — 코드는
# 멀쩡한데 .env가 바뀌었기 때문이다. 그전까지 통과한 건 .env 값이 우연히 기본값과
# 같아서였지 격리돼 있어서가 아니었다.
#
# 새 기여자의 .env, CI의 빈 .env, 실험 중인 내 .env가 전부 다른 결과를 내면
# 테스트는 신호가 아니라 잡음이 된다.
Settings.model_config["env_file"] = None


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
