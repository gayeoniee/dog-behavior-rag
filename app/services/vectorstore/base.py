"""벡터 저장소 인터페이스."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class SearchHit:
    """검색 결과 1건."""

    chunk_id: int
    document_title: str
    content: str
    score: float
    source: str | None = None


@runtime_checkable
class VectorStore(Protocol):
    async def add_chunks(
        self,
        document_id: int,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        """청크 + 임베딩을 저장하고 저장된 개수를 반환한다."""
        ...

    async def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        """질의 임베딩과 가장 가까운 청크를 top_k개 반환한다."""
        ...
