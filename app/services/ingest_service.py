"""문서 적재 오케스트레이션: 청킹 → 임베딩 → documents/chunks INSERT.

HTTP 엔드포인트(`POST /documents`)와 오프라인 CLI(`scripts.db.load_corpus`)가
**같은 서비스를 부른다.** 구현이 갈라지지 않게 하려는 것이다 — CLI가 직접
SQL을 쓰면 엔드포인트가 조용히 뒤처진다.
"""

import hashlib
import logging
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.schemas.document import DocumentIn
from app.services.chunking import ChunkConfig, looks_like_paper_boilerplate, split_text
from app.services.embeddings.base import Embedder
from app.services.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    """`scripts/collect/normalize.py`의 content_hash와 **같은 값**을 내야 한다.

    어긋나면 재적재가 멱등하지 않아 같은 문서가 매번 새로 들어간다.
    app이 scripts를 import하지 않으려고 구현을 나눠 뒀고, 대신
    `tests/test_ingest_service.py`가 기댓값을 하드코딩해 고정한다.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class IngestResult:
    document_id: int
    title: str
    source: str | None
    category: str | None
    chunk_count: int
    created: bool
    """False = content_hash가 이미 있어서 건너뛴 것 (에러가 아니다)."""


class IngestService:
    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        store: VectorStore,
        chunk_config: ChunkConfig | None = None,
        paper_boilerplate_filter: bool = True,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._store = store
        self._chunk_config = chunk_config or ChunkConfig()
        self._paper_filter = paper_boilerplate_filter

    async def ingest(self, doc: DocumentIn) -> IngestResult:
        # 원본 content로 해싱한다. 청킹 전처리를 거친 텍스트로 하면 코퍼스가 가진
        # content_hash와 값이 달라져 멱등성이 깨진다.
        digest = doc.content_hash or content_hash(doc.content)

        existing = await self._find_existing(digest)
        if existing is not None:
            return existing

        chunks = split_text(doc.content, self._chunk_config)
        # **논문의 형식 잡음은 답변 근거가 못 되는데 검색에는 걸린다.**
        # 통계·방법론 조각이 "significant"·"dogs were assigned" 같은 어휘로
        # 개 행동 질문에 붙는다. 논문에만 적용한다 — 규칙이 논문 문체를
        # 겨냥해 만들어졌고, 기관 가이드에서는 오탐이 난 적이 있다.
        if self._paper_filter and (doc.source_id or "").startswith("pmc-"):
            kept = [c for c in chunks if not looks_like_paper_boilerplate(c)]
            if kept:
                if len(kept) < len(chunks):
                    logger.info(
                        "논문 형식 잡음 %d/%d청크 제외: %s",
                        len(chunks) - len(kept), len(chunks), doc.title[:40],
                    )
                chunks = kept
        if not chunks:
            logger.warning("청크가 0개라 건너뜀: %s", doc.title)
            return IngestResult(
                document_id=-1,
                title=doc.title,
                source=doc.source,
                category=doc.category,
                chunk_count=0,
                created=False,
            )

        embeddings = await self._embedder.embed(chunks)

        try:
            document = Document(
                title=doc.title,
                content=doc.content,
                content_hash=digest,
                source=doc.source,
                source_id=doc.source_id,
                category=doc.category,
                language=doc.language,
                species=doc.species,
                axis=list(doc.axis),
                methodology=doc.methodology,
                authority_tier=doc.authority_tier,
                published_at=doc.published_at,
                license=doc.license,
                distribution=doc.distribution,
                doc_type=doc.doc_type,
                corpus=doc.corpus,
            )
            self._session.add(document)
            await self._session.flush()  # PK 확보

            count = await self._store.add_chunks(document.id, chunks, embeddings)
            # 문서 단위 commit. 11건 중 9번째가 실패해도 앞 8건은 남는다.
            await self._session.commit()
        except IntegrityError:
            # 동시에 같은 문서를 적재한 경우. 에러가 아니라 "이미 있음"으로 취급한다.
            await self._session.rollback()
            existing = await self._find_existing(digest)
            if existing is not None:
                return existing
            raise

        return IngestResult(
            document_id=document.id,
            title=document.title,
            source=document.source,
            category=document.category,
            chunk_count=count,
            created=True,
        )

    async def ingest_many(self, docs: list[DocumentIn]) -> list[IngestResult]:
        return [await self.ingest(doc) for doc in docs]

    async def _find_existing(self, digest: str) -> IngestResult | None:
        """이미 적재된 문서면 재임베딩 없이 현재 상태를 돌려준다."""
        stmt = (
            select(
                Document.id,
                Document.title,
                Document.source,
                Document.category,
                func.count(Chunk.id).label("chunk_count"),
            )
            .outerjoin(Chunk, Chunk.document_id == Document.id)
            .where(Document.content_hash == digest)
            .group_by(Document.id)
        )
        row = (await self._session.execute(stmt)).first()
        if row is None:
            return None
        return IngestResult(
            document_id=row.id,
            title=row.title,
            source=row.source,
            category=row.category,
            chunk_count=row.chunk_count,
            created=False,
        )
