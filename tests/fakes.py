"""검색 경로 테스트용 가짜 구현체.

Protocol만 만족하면 되므로 torch도 Postgres도 필요 없다.
"""

from app.services.embeddings.base import EmbeddingUnavailableError
from app.services.vectorstore.base import SearchHit


class FakeEmbedder:
    def __init__(self, *, dimension: int = 4, fail: bool = False) -> None:
        self._dimension = dimension
        self._fail = fail

    @property
    def name(self) -> str:
        return "fake:embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def warmup(self) -> None:
        if self._fail:
            raise EmbeddingUnavailableError("테스트용 로딩 실패")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._fail:
            raise EmbeddingUnavailableError("테스트용 로딩 실패")
        return [[0.1] * self._dimension for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


class FakeStore:
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = hits if hits is not None else []
        self.added: list[tuple[int, list[str]]] = []

    async def add_chunks(
        self, document_id: int, chunks: list[str], embeddings: list[list[float]]
    ) -> int:
        self.added.append((document_id, chunks))
        return len(chunks)

    async def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        return self.hits[:top_k]


class FakeLLM:
    @property
    def name(self) -> str:
        return "huggingface:fake"

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        reasoning: bool | None = None,
    ) -> str:
        self.last_prompt = prompt
        return "[stub] 테스트 답변"


def hit(chunk_id: int = 1, *, title: str = "문서 A", score: float = 0.9) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_title=title,
        content=f"근거 본문 {chunk_id}",
        score=score,
        source=f"https://example.test/{chunk_id}",
    )
