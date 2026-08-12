"""문서 적재 / 조회 엔드포인트.

대량 적재는 이 엔드포인트가 아니라 `scripts.db.load_corpus`를 쓴다 — API 프로세스에
torch를 상주시키지 않고, 10만 자짜리 문서 하나가 요청 타임아웃을 내지 않게 하려는
것이다. 둘 다 같은 `IngestService`를 호출하므로 동작은 동일하다.
"""

import logging

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from app.api.deps import IngestServiceDep, SessionDep
from app.db.models import Chunk, Document
from app.schemas.document import DocumentIn, DocumentOut, IngestResponse
from app.services.embeddings.base import EmbeddingUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


@router.post(
    "/documents",
    response_model=IngestResponse,
    # 202가 아니라 201인 이유: 이제 동기로 처리해 응답 시점에 이미 저장이 끝나 있다.
    status_code=status.HTTP_201_CREATED,
    summary="문서 적재 (청킹 + 임베딩 + 저장)",
)
async def ingest(payload: list[DocumentIn], service: IngestServiceDep) -> IngestResponse:
    try:
        results = await service.ingest_many(payload)
    except EmbeddingUnavailableError as exc:
        logger.warning("적재 불가: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="임베딩 모델이 준비되지 않았습니다 (uv sync --extra hf)",
        ) from exc

    return IngestResponse(
        # 이미 있던 문서(created=False)는 세지 않는다 — 재적재가 멱등하다는 걸
        # 응답만 봐도 알 수 있어야 한다.
        ingested=sum(1 for r in results if r.created),
        documents=[
            DocumentOut(
                id=r.document_id,
                title=r.title,
                source=r.source,
                category=r.category,
                chunk_count=r.chunk_count,
            )
            for r in results
        ],
    )


@router.get(
    "/documents",
    response_model=list[DocumentOut],
    summary="적재된 문서 목록",
)
async def list_documents(session: SessionDep) -> list[DocumentOut]:
    stmt = (
        select(
            Document.id,
            Document.title,
            Document.source,
            Document.category,
            func.count(Chunk.id).label("chunk_count"),
        )
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.id)
    )
    rows = (await session.execute(stmt)).all()
    return [
        DocumentOut(
            id=row.id,
            title=row.title,
            source=row.source,
            category=row.category,
            chunk_count=row.chunk_count,
        )
        for row in rows
    ]
