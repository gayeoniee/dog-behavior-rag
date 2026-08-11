"""pgvector 기반 벡터 저장소 — 아직 스텁.

TODO(내일):
  - db/models.py의 Chunk 테이블 확정 후 실제 INSERT/SELECT 작성
  - 검색은 코사인 거리 연산자 사용:
        stmt = (
            select(Chunk, Document.title, Document.source,
                   (1 - Chunk.embedding.cosine_distance(embedding)).label("score"))
            .join(Document, Chunk.document_id == Document.id)
            .order_by(Chunk.embedding.cosine_distance(embedding))
            .limit(top_k)
        )
  - 임베딩 정규화 여부와 인덱스 연산자(vector_cosine_ops)를 일치시킬 것
"""

from sqlalchemy.ext.asyncio import AsyncSession

from .base import SearchHit


class PgVectorStore:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_chunks(
        self,
        document_id: int,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        # TODO(내일): 실제 INSERT
        return len(chunks)

    async def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        # TODO(내일): 실제 유사도 검색
        return []
