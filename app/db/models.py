"""ORM 모델.

TODO(내일): 실제 테이블 정의.
임베딩 차원(Vector(N))은 임베딩 모델이 확정돼야 정할 수 있어서 아직 비워둔다.
(BAAI/bge-m3 = 1024차원, 다른 모델 쓰면 달라짐)

스케치:

    from pgvector.sqlalchemy import Vector
    from sqlalchemy import ForeignKey, Index, String, Text
    from sqlalchemy.orm import Mapped, mapped_column, relationship

    class Document(Base):
        '''원문 단위 — 훈련 가이드 문서 1건'''
        __tablename__ = "documents"

        id: Mapped[int] = mapped_column(primary_key=True)
        title: Mapped[str] = mapped_column(String(500))
        source: Mapped[str] = mapped_column(String(500))   # 출처 URL/파일명
        category: Mapped[str | None] = mapped_column(String(100))  # 짖음/분리불안/배변 ...
        content: Mapped[str] = mapped_column(Text)

    class Chunk(Base):
        '''검색 단위 — 문서를 쪼갠 조각 + 임베딩'''
        __tablename__ = "chunks"

        id: Mapped[int] = mapped_column(primary_key=True)
        document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
        content: Mapped[str] = mapped_column(Text)
        embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    # 코사인 거리 기준 ANN 인덱스
    Index(
        "ix_chunks_embedding",
        Chunk.embedding,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """모든 ORM 모델의 베이스."""
