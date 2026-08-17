"""FastAPI 의존성.

엔진/세션팩토리와 임베딩 모델은 lifespan에서 만들어 app.state에 보관하고,
여기서 꺼내 쓴다. 특히 임베더는 **요청마다 만들면 안 된다** — 수 GB 모델이다.
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.services.chunking import ChunkConfig
from app.services.embeddings.base import Embedder
from app.services.evidence_select import EvidenceSelector
from app.services.factcheck_service import FactCheckService
from app.services.ingest_service import IngestService
from app.services.llm.registry import get_llm
from app.services.query_rewrite import QueryRewriter
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


def embedder_dep(request: Request) -> Embedder:
    """lifespan에서 1회 로딩해 둔 임베더. 로딩 실패 시에도 객체는 있고,
    embed() 호출 시점에 EmbeddingUnavailableError가 난다 (엔드포인트가 503으로 변환)."""
    embedder: Embedder = request.app.state.embedder
    return embedder


EmbedderDep = Annotated[Embedder, Depends(embedder_dep)]


def _store(settings: Settings, session: AsyncSession) -> PgVectorStore:
    return PgVectorStore(
        session,
        authority_boost=settings.authority_boost,
        guide_boost=settings.guide_boost,
        max_per_document=settings.max_chunks_per_document,
        candidate_multiplier=settings.candidate_multiplier,
        ef_search=settings.hnsw_ef_search,
                background_weight=settings.language_background_weight,
    )


def _chunk_config(settings: Settings) -> ChunkConfig:
    return ChunkConfig(
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        min_size=settings.chunk_min_size,
    )


def rag_service(settings: SettingsDep, session: SessionDep, embedder: EmbedderDep) -> RagService:
    llm = get_llm(settings)
    return RagService(
        embedder=embedder,
        store=_store(settings, session),
        llm=llm,
        default_top_k=settings.top_k,
        # 재작성에도 같은 LLM을 쓴다. 별도 모델을 둘 이유가 없고, 서버가 꺼져 있으면
        # QueryRewriter가 원문으로 폴백하므로 검색은 계속 동작한다.
        rewriter=QueryRewriter(llm, enabled=settings.query_rewrite_enabled),
        selector=EvidenceSelector(llm, enabled=settings.evidence_select_enabled),
    )


RagServiceDep = Annotated[RagService, Depends(rag_service)]


def ingest_service(
    settings: SettingsDep, session: SessionDep, embedder: EmbedderDep
) -> IngestService:
    return IngestService(
        session=session,
        embedder=embedder,
        store=_store(settings, session),
        chunk_config=_chunk_config(settings),
    )


IngestServiceDep = Annotated[IngestService, Depends(ingest_service)]


def factcheck_service(
    settings: SettingsDep, session: SessionDep, embedder: EmbedderDep
) -> FactCheckService:
    llm = get_llm(settings)
    return FactCheckService(
        embedder=embedder,
        store=_store(settings, session),
        llm=llm,
        # 검증할 주장은 대개 기법에 대한 것이라 재작성이 chat보다 더 중요하다.
        rewriter=QueryRewriter(llm, enabled=settings.query_rewrite_enabled),
        default_top_k=settings.top_k,
    )


FactCheckServiceDep = Annotated[FactCheckService, Depends(factcheck_service)]
