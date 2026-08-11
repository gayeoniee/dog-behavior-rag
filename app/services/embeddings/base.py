"""임베딩 인터페이스.

구현체는 이 Protocol만 만족하면 된다. 교체 지점은 registry.py 한 곳.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str:
        """모델 식별자 (로깅/디버깅용)."""
        ...

    @property
    def dimension(self) -> int:
        """임베딩 차원. pgvector 컬럼 정의와 반드시 일치해야 한다."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """문서 적재용 임베딩 (배치)."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """검색 질의용 임베딩.

        모델에 따라 문서용과 질의용 프리픽스가 다를 수 있어서 분리해 둔다
        (예: bge 계열의 "query: " 프리픽스).
        """
        ...
