"""벡터 저장소 인터페이스."""

from collections.abc import Mapping
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

    async def search(
        self, embedding: list[float] | Mapping[str, list[float]], top_k: int
    ) -> list[SearchHit]:
        """질의 임베딩과 가장 가까운 청크를 top_k개 반환한다.

        언어별 질의가 다르면 `{"en": ..., "ko": ...}`를 준다. 벡터 하나를 주면
        모든 언어에 같은 질의를 쓴다 (`query_rewrite.SearchQuery` 참조).
        """
        ...
