"""임베딩 설정(MRL 자르기·접두사) 테스트.

둘 다 **틀려도 에러가 안 나고 조용히 품질만 떨어지는** 종류라 테스트가 필요하다:

  자르기를 안 켠 채 차원만 줄이면  → 예전엔 pgvector INSERT까지 가서 터졌다
  질의에 문서 접두사를 붙이면      → 모델이 질의를 문서로 착각한다 (e5)
"""

import pytest

from app.core.config import Settings
from app.services.embeddings.base import EmbeddingUnavailableError
from app.services.embeddings.huggingface import HuggingFaceEmbedder


def make(**overrides) -> HuggingFaceEmbedder:
    return HuggingFaceEmbedder(Settings(app_env="test", **overrides))


class FakeModel:
    """`_load`가 만드는 SentenceTransformer 대역."""

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim
        self.encoded: list[str] = []
        self.max_seq_length = 0

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, **kwargs):
        self.encoded.extend(texts)
        # 실제 sentence-transformers는 numpy 배열을 돌려주고 호출부가 .tolist()를 쓴다.
        return [_FakeVector(self._dim) for _ in texts]


class _FakeVector:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def tolist(self) -> list[float]:
        return [0.1] * self._dim


# ── 접두사 ───────────────────────────────────────────────────────────


async def test_query_and_passage_prefixes_are_different():
    """**이게 이 파일의 핵심이다.** 질의와 문서에 같은 접두사가 붙으면 e5 계열이
    질의를 문서로 착각해 성능이 조용히 떨어진다."""
    embedder = make(embedding_query_prefix="query: ", embedding_passage_prefix="passage: ")
    embedder._model = FakeModel()

    await embedder.embed(["문서 본문"])
    await embedder.embed_query("질문")

    assert embedder._model.encoded == ["passage: 문서 본문", "query: 질문"]


async def test_no_prefix_by_default():
    """bge-m3는 접두사를 붙이지 않는다. 기본값이 바뀌면 재적재 없이는 검색이 어긋난다."""
    embedder = make()
    embedder._model = FakeModel()

    await embedder.embed(["본문"])
    await embedder.embed_query("질문")

    assert embedder._model.encoded == ["본문", "질문"]


async def test_embed_query_fails_loudly_without_warmup():
    """0 벡터 같은 걸로 조용히 대체하면 검색이 무의미해지고 아무도 모른다."""
    with pytest.raises(EmbeddingUnavailableError):
        await make().embed_query("질문")


# ── MRL 자르기 ───────────────────────────────────────────────────────


def test_dimension_mismatch_raises_when_truncate_is_off():
    """자르기를 안 켰으면 차원이 다를 때 **로딩에서** 죽어야 한다.

    통과시키면 훨씬 나중에 pgvector INSERT에서 불친절하게 터진다.
    """
    embedder = make(embedding_dim=512)
    with pytest.raises(EmbeddingUnavailableError, match="차원 불일치"):
        embedder._check_dimension(1024)


def test_truncate_allows_a_smaller_configured_dim():
    """MRL을 켜면 모델(1024) → 설정(512) 축소가 허용된다."""
    make(embedding_dim=512, embedding_truncate=True)._check_dimension(1024)


def test_truncate_still_rejects_growing_the_dimension():
    """자르기는 줄이는 것이지 늘리는 게 아니다. 768 모델로 1024를 만들 수는 없다."""
    embedder = make(embedding_dim=1024, embedding_truncate=True)
    with pytest.raises(EmbeddingUnavailableError, match="자를 수 없습니다"):
        embedder._check_dimension(768)


def test_truncate_is_off_by_default():
    """기본이 꺼져 있어야 한다 — 켜져 있으면 차원 불일치가 에러 대신
    조용한 성능 저하로 바뀐다."""
    assert Settings(app_env="test").embedding_truncate is False
