"""pgvector 기반 벡터 저장소.

임베딩은 `HuggingFaceEmbedder`가 `normalize_embeddings=True`로 정규화해 넣는다.
HNSW 인덱스가 `vector_cosine_ops`이므로 둘이 일치해야 ANN 결과가 정확하다 —
한쪽만 바꾸면 조용히 엉뚱한 순위가 나온다.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document

from .base import SearchHit
from .ranking import Candidate, rank

_CENTROIDS: dict[str, list[float]] | None = None
"""언어별 청크 임베딩의 평균 벡터. 프로세스당 한 번만 계산한다.

**이걸로 "배경 유사도"를 구한다** — 질문과 그 언어 청크 전체의 평균 유사도다.
정규화된 벡터에서는 **평균 유사도가 평균 벡터와의 내적과 정확히 같아서**
(실측 오차 0.000000) 전체 스캔 139ms가 내적 2ms로 줄어든다. 코퍼스가 커져도
비용이 그대로다.

⚠️ **코사인(`<=>`)이 아니라 내적을 써야 한다.** 코사인은 중심 벡터의 크기로
나누는데, 단위 벡터들을 평균 내면 길이가 1보다 짧아져 값이 부풀려진다.
처음에 코사인으로 짰다가 0.396이어야 할 값이 0.541로 나왔다.

⚠️ 적재 후에도 프로세스가 살아 있으면 값이 낡는다. 스크립트는 매번 새 프로세스라
문제없고, API는 재시작하면 갱신된다. 대량 적재는 API가 아니라 load_corpus로 한다.
"""


def reset_language_background() -> None:
    """캐시를 비운다. 테스트와 적재 직후용."""
    global _CENTROIDS
    _CENTROIDS = None


class PgVectorStore:
    def __init__(
        self,
        session: AsyncSession,
        *,
        authority_boost: float = 0.02,
        guide_boost: float = 0.03,
        max_per_document: int = 2,
        candidate_multiplier: int = 4,
        ef_search: int = 400,
        background_weight: float = 1.0,
    ) -> None:
        self._session = session
        self._ef_search = ef_search
        self._background_weight = background_weight
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
        """언어별로 후보를 뽑고, 각 언어의 **배경 유사도를 뺀 뒤** 한 줄로 세운다.

        **언어를 나눠 조회하는 이유:** 한 번에 뽑으면 한국어가 상위를 독식해서
        (청크의 1.7%가 상위 5위의 41%) 영어 후보가 애초에 안 딸려온다. 독식을
        고치려는데 독식된 목록으로 고치는 셈이 된다. 검색은 수십 ms라 두 번
        도는 비용은 무시할 만하다.

        **점수 보정은 `rank`가 한다** — 후보에 `background`를 붙여 넘기면 된다.
        """
        # **탐색폭을 넓히지 않으면 진짜 상위 후보를 놓친다.** 기본값 40으로는
        # 한국어 청크(전체의 2.9%)에 그래프가 닿지 못해, 유사도 0.62짜리가
        # 빠지고 0.52짜리 영어 논문이 1위로 올라왔다. SET LOCAL이라 이 트랜잭션에만
        # 적용된다 (config.hnsw_ef_search 독스트링에 실측값이 있다).
        await self._session.execute(text(f"SET LOCAL hnsw.ef_search = {int(self._ef_search)}"))
        background = await self._background(embedding)
        candidates: list[Candidate] = []
        for language in ("en", "ko"):
            candidates.extend(
                await self._candidates(embedding, top_k, language, background.get(language, 0.0))
            )
        return rank(
            candidates,
            top_k,
            authority_boost=self._authority_boost,
            guide_boost=self._guide_boost,
            max_per_document=self._max_per_document,
            background_weight=self._background_weight,
        )

    async def _background(self, embedding: list[float]) -> dict[str, float]:
        """언어별 배경 유사도 = 질문 벡터와 그 언어 중심 벡터의 **내적**."""
        global _CENTROIDS
        if _CENTROIDS is None:
            rows = (
                await self._session.execute(
                    # **type_을 줘야 한다.** func.avg는 컬럼 타입을 잃어버려서
                    # 벡터가 문자열로 돌아오고, 내적을 계산하려는 순간 터진다.
                    select(Document.language, func.avg(Chunk.embedding, type_=Vector()))
                    .join(Document, Chunk.document_id == Document.id)
                    .where(Document.methodology != "aversive")
                    .where(Document.corpus == "answer")
                    .group_by(Document.language)
                )
            ).all()
            _CENTROIDS = {row[0]: list(row[1]) for row in rows}
        # 언어가 하나뿐이면 모두 같은 값을 빼므로 순위가 안 바뀐다 — 기존 동작 보존.
        return {
            lang: sum(a * b for a, b in zip(centroid, embedding, strict=True))
            for lang, centroid in _CENTROIDS.items()
        }

    async def _candidates(
        self, embedding: list[float], top_k: int, language: str, background: float
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
                background=background,
            )
            for row in rows
        ]
