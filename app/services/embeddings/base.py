"""임베딩 인터페이스.

구현체는 이 Protocol만 만족하면 된다. 교체 지점은 registry.py 한 곳.
"""

from typing import Protocol, runtime_checkable


class EmbeddingUnavailableError(RuntimeError):
    """임베딩 모델을 쓸 수 없다 (미설치 / warmup 실패 / warmup 미실행).

    0 벡터 같은 걸로 조용히 대체하지 않는다 — 그러면 검색이 조용히 무의미해지고
    아무도 눈치채지 못한다. 호출 경로는 이걸 503으로 바꿔 사용자에게 알린다.
    """


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

    async def warmup(self) -> None:
        """무거운 모델을 lifespan에서 1회 로딩한다. 로딩이 필요 없는 구현체는 no-op.

        이 훅이 Protocol에 있어야 하는 이유: 없으면 main.py가
        `getattr(embedder, "load", None)` 같은 걸 해야 하는데, 그건 registry 뒤에
        숨어 있어야 할 provider별 지식이 호출부로 새어나오는 것이다.
        """
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """문서 적재용 임베딩 (배치)."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """검색 질의용 임베딩.

        모델에 따라 질의용 프리픽스가 필요해서 문서용과 분리해 둔다.
        E5 계열은 "query:"/"passage:"를, bge 영어 v1.5는 지시문 프리픽스를 요구한다.
        **현재 쓰는 bge-m3는 instruction-free라 프리픽스가 없다** — 붙이면 오히려
        학습 분포와 어긋난다. 메서드는 다음 모델을 위해 남겨둔다.
        """
        ...
