"""ORM 모델.

스키마 다이어그램과 필드별 근거는 `docs/erd.md`에 있다 — 여기를 고치면 거기도 고칠 것.

Alembic을 쓰지 않는다. 임베딩 차원과 메타데이터 컬럼이 아직 흔들리는 단계라
마이그레이션 이력보다 drop & recreate가 빠르고 정확하며, 코퍼스는 언제든
`scripts.db.load_corpus`로 몇 분 안에 다시 채울 수 있다.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings

_EMBEDDING_DIM = get_settings().embedding_dim
"""import 시점에 고정된다. 모델 introspection이 아니라 설정값을 쓰는 이유는
config.py의 embedding_dim 주석 참조 (요약: torch 없이도 테이블 정의가 돼야 한다)."""


class Base(DeclarativeBase):
    """모든 ORM 모델의 베이스."""


class Document(Base):
    """원문 단위 — `corpus.jsonl` 한 줄에 대응한다."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)

    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    """재적재 멱등성 키. `scripts/collect/normalize.py`가 만든 값(sha256 앞 16자)을
    그대로 받는다. unique라서 같은 문서를 두 번 적재해도 임베딩을 다시 계산하지 않는다.
    String(64)인 건 나중에 전체 sha256으로 바꿔도 스키마를 안 건드리려는 것."""

    source: Mapped[str | None] = mapped_column(String(500), default=None)
    """corpus의 `source_url`. 답변에 붙일 인용 링크라서 SearchHit까지 그대로 흘러간다."""
    source_id: Mapped[str | None] = mapped_column(String(100), default=None)
    category: Mapped[str | None] = mapped_column(String(100), default=None)

    language: Mapped[str] = mapped_column(String(8), default="ko")
    species: Mapped[str] = mapped_column(String(8), default="dog")

    axis: Mapped[list[str]] = mapped_column(ARRAY(String(20)), default=list)
    """problem / cause / training / medical. JSONB가 아니라 ARRAY인 이유:
    4값 고정 어휘라서 `&&` 겹침 연산자와 GIN 인덱스를 바로 쓸 수 있다.
    JSONB는 스키마 없는 페이로드용이고 여기선 containment 캐스팅만 늘어난다."""

    methodology: Mapped[str] = mapped_column(String(20), default="unknown")
    """`aversive`는 검색에서 하드코딩으로 제외된다 (vectorstore/pgvector.py)."""

    authority_tier: Mapped[int] = mapped_column(SmallInteger, default=3)
    """1=기관·학술, 3=일반. **낮을수록 권위가 높다.** 검색 시 부스팅 신호."""

    published_at: Mapped[int | None] = mapped_column(SmallInteger, default=None)
    license: Mapped[str | None] = mapped_column(String(100), default=None)

    distribution: Mapped[str] = mapped_column(String(20), default="personal-only")
    """`open` | `personal-only`. CLAUDE.md가 배포 전 걸러내라고 한 판정을
    license 문자열 매칭이 아니라 한 필드로 만든 것. 기본값이 보수적인 쪽인 이유는
    분류를 빠뜨린 문서가 조용히 배포 대상에 들어가면 안 되기 때문이다."""

    corpus: Mapped[str] = mapped_column(String(20), default="answer")
    """`answer` | `observation`.

    블로그처럼 지배이론이 섞일 수 있는 자료는 `observation`으로 격리한다. 검색이
    `corpus == "answer"`를 하드코딩으로 걸기 때문에 답변 근거로는 절대 나오지 않는다.
    용도는 "사람들이 어떤 말로 묻는가"를 보는 것뿐이다."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # 이 둘은 모든 검색의 WHERE 절에 들어간다.
        Index("ix_documents_methodology", "methodology"),
        Index("ix_documents_corpus", "corpus"),
        # 축별 커버리지 조회용 (배열 겹침 연산자).
        Index("ix_documents_axis", "axis", postgresql_using="gin"),
    )


class Chunk(Base):
    """검색 단위 — 문서를 쪼갠 조각 + 임베딩.

    `methodology`/`authority_tier`를 여기 비정규화하지 않는다. 문서 수백 건 규모에서
    조인은 사실상 공짜다. 청크가 10만 개를 넘고 aversive 문서가 실제로 생기면 그때는
    ANN 후필터가 후보를 잃기 시작하므로, 그 시점에 두 컬럼을 복제하거나
    `WHERE methodology <> 'aversive'` 부분 인덱스를 만들어야 한다.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(SmallInteger)
    """문서 내 순번. 지금 검색에는 안 쓰지만 이웃 청크 확장("적중 청크 다음 것도 같이")과
    디버깅에 필요하고 비용이 0이다."""

    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBEDDING_DIM))

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
    )
