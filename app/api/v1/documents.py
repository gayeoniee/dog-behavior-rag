"""문서 적재 엔드포인트 — 아직 스텁.

TODO(내일): 청킹 → 임베딩 → pgvector 저장 파이프라인 연결.
"""

from fastapi import APIRouter, status

from app.api.deps import SessionDep
from app.schemas.document import DocumentIn, DocumentOut, IngestResponse

router = APIRouter(tags=["documents"])


@router.post(
    "/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="문서 적재 (청킹 + 임베딩 + 저장)",
)
async def ingest(payload: list[DocumentIn], session: SessionDep) -> IngestResponse:
    # TODO(내일): chunker.split() → embedder.embed() → store.add_chunks()
    return IngestResponse(
        ingested=0,
        documents=[
            DocumentOut(
                id=-1,
                title=doc.title,
                source=doc.source,
                category=doc.category,
                chunk_count=0,
            )
            for doc in payload
        ],
    )


@router.get(
    "/documents",
    response_model=list[DocumentOut],
    summary="적재된 문서 목록",
)
async def list_documents(session: SessionDep) -> list[DocumentOut]:
    # TODO(내일): SELECT * FROM documents
    return []
