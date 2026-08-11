"""FastAPI 의존성.

엔진/세션팩토리는 lifespan에서 만들어 app.state에 보관하고, 여기서 꺼내 쓴다.
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.services.embeddings.registry import get_embedder
from app.services.llm.registry import get_llm
from app.services.rag_service import RagService
from app.services.vectorstore.pgvector import PgVectorStore


def settings_dep() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dep)]


async def db_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] = request.app.state.db_session_factory
    async with factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(db_session)]


def rag_service(settings: SettingsDep, session: SessionDep) -> RagService:
    # TODO(내일): 임베딩 모델은 무거우므로 lifespan에서 1회 로딩 후 app.state에서 재사용하도록 변경.
    #            지금은 스텁이라 요청마다 만들어도 비용이 없다.
    return RagService(
        embedder=get_embedder(settings),
        store=PgVectorStore(session),
        llm=get_llm(settings),
        default_top_k=settings.top_k,
    )


RagServiceDep = Annotated[RagService, Depends(rag_service)]
