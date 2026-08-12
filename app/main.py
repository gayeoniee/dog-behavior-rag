"""FastAPI 앱 엔트리포인트.

실행: uv run uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.db.session import create_engine, create_session_factory
from app.services.embeddings.base import EmbeddingUnavailableError
from app.services.embeddings.registry import get_embedder

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    engine = create_engine(settings.database_url)
    app.state.db_engine = engine
    app.state.db_session_factory = create_session_factory(engine)
    # 실제 연결은 첫 쿼리 때 이뤄지므로, Postgres가 꺼져 있어도 여기서 죽지 않는다.
    logger.info("앱 기동 — env=%s, llm=%s", settings.app_env, settings.llm_provider)

    # 임베딩 모델은 여기서 1회만 로딩한다. 요청마다 로딩하면 수 GB 모델을 매번
    # 디스크에서 읽는 꼴이 된다.
    embedder = get_embedder(settings)
    app.state.embedder = embedder
    if settings.embedding_warmup:
        try:
            await embedder.warmup()
            logger.info("임베딩 모델 준비 완료 — %s (dim=%d)", embedder.name, embedder.dimension)
        except EmbeddingUnavailableError as exc:
            # 여기서 앱을 죽이지 않는다. /health·/docs·DB 확인은 계속 돼야 하고,
            # 임베딩이 필요한 경로(/chat, POST /documents)만 503으로 실패하면 된다.
            # 이게 `uv sync`(torch 없음)만으로도 앱이 뜨는 성질을 지켜준다.
            logger.warning("임베딩 비활성 — 검색 경로는 503을 반환합니다: %s", exc)
    else:
        logger.info("EMBEDDING_WARMUP=false — 임베딩 모델을 로딩하지 않습니다")

    yield

    await engine.dispose()
    logger.info("앱 종료")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="GY_RAG",
        description="반려동물 훈련/문제행동 상담 RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # 안드로이드 앱 / Streamlit 데모에서 호출할 수 있도록 열어둔다.
    # TODO(배포 전): 실제 오리진으로 좁힐 것.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
