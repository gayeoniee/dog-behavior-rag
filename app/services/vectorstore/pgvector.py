"""pgvector 기반 벡터 저장소.

임베딩은 `HuggingFaceEmbedder`가 `normalize_embeddings=True`로 정규화해 넣는다.
HNSW 인덱스가 `vector_cosine_ops`이므로 둘이 일치해야 ANN 결과가 정확하다 —
한쪽만 바꾸면 조용히 엉뚱한 순위가 나온다.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document

from .base import SearchHit
from .ranking import Candidate, rank_fused


class PgVectorStore:
    def __init__(
        self,
        session: AsyncSession,
        *,
        authority_boost: float = 0.02,
        guide_boost: float = 0.03,
        max_per_document: int = 2,
        candidate_multiplier: int = 4,
    ) -> None:
        self._session = session
        self._authority_boost = authority_boost
        self._guide_boost = guide_boost
        self._max_per_document = max_per_document
        self._candidate_multiplier = candidate_multiplier

    async def add_chunks(
        self,
        document_id: int,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        """청크를 INSERT한다.

        **commit은 하지 않는다.** 호출자(IngestService)가 문서 행과 청크 행을 한
        트랜잭션으로 묶어야 하기 때문이다 — 중간에 실패했을 때 청크 없는 문서가
        남으면 재적재가 content_hash 때문에 건너뛰어 영구히 빈 문서가 된다.
        """
        rows = [
            Chunk(document_id=document_id, ordinal=i, content=content, embedding=embedding)
            # strict=True: 청크와 임베딩 개수가 어긋나면 조용히 잘리지 않고 여기서 터진다.
            for i, (content, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        self._session.add_all(rows)
        await self._session.flush()
        return len(rows)

    async def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        """언어별로 따로 뽑아 RRF로 합친다.

        **언어를 나누지 않으면 한국어 문서가 상위를 독식한다.** 한→한 코사인이
        한→영보다 구조적으로 높아서, 청크의 1.7%가 상위 5위의 41%를 가져갔다
        (2026-08-17 실측). SQL 하나로 뽑고 파이썬에서 나누는 방법도 있지만,
        그러면 **밀린 언어의 후보를 애초에 못 받아온다** — 한쪽이 독식하는
        상황을 고치려는데 그 독식된 목록으로 나누는 셈이다. 그래서 쿼리를 나눈다.
        검색은 수십 ms라 두 번 도는 비용은 무시할 만하다.
        """
        pools = [
            await self._candidates(embedding, top_k, language)
            for language in ("en", "ko")
        ]
        pools = [p for p in pools if p]  # 한 언어만 있으면 기존 동작과 완전히 같다
        return rank_fused(
            pools,
            top_k,
            authority_boost=self._authority_boost,
            guide_boost=self._guide_boost,
            max_per_document=self._max_per_document,
        )

    async def _candidates(
        self, embedding: list[float], top_k: int, language: str
    ) -> list[Candidate]:
        distance = Chunk.embedding.cosine_distance(embedding)
        stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Document.title,
                Document.source,
                Chunk.content,
                distance.label("distance"),
                Document.authority_tier,
                Document.doc_type,
            )
            .join(Document, Chunk.document_id == Document.id)
            # 아래 두 필터는 코퍼스 품질 불변식이라 호출자가 끌 수 있으면 안 된다.
            #   - aversive: 혐오 기반 훈련법. AVSAB 문서가 정면으로 반박하는 내용이
            #     같은 답변의 근거로 들어가면 답이 자기모순에 빠진다.
            #   - observation: 블로그 등 지배이론이 섞일 수 있는 관찰용 구획.
            .where(Document.methodology != "aversive")
            .where(Document.corpus == "answer")
            .where(Document.language == language)
            .order_by(distance)
            # 부스팅·다양성 재랭킹을 하려면 top_k보다 넉넉히 받아와야 한다.
            .limit(top_k * self._candidate_multiplier)
        )
        rows = (await self._session.execute(stmt)).all()

        # 위치 인자로 풀지 않고 이름을 적는다 — SELECT 컬럼 순서를 나중에 누가
        # 바꿔도 조용히 필드가 뒤바뀌지 않게.
        return [
            Candidate(
                chunk_id=row.id,
                document_id=row.document_id,
                document_title=row.title,
                source=row.source,
                content=row.content,
                distance=row.distance,
                authority_tier=row.authority_tier,
                doc_type=row.doc_type,
            )
            for row in rows
        ]
