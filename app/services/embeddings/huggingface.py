"""HuggingFace 임베딩 구현 — 아직 스텁.

TODO(내일):
  - sentence_transformers.SentenceTransformer(settings.hf_embedding_model) 로딩
  - 모델 로딩은 앱 기동 시 1회 (lifespan), 요청마다 하면 안 됨
  - encode()는 동기 blocking이라 asyncio.to_thread()로 감싸야 이벤트 루프가 안 막힘
  - dimension은 model.get_sentence_embedding_dimension()에서 읽기
"""

from app.core.config import Settings

_STUB_DIMENSION = 1024
"""BAAI/bge-m3 기준. 모델 확정되면 실제 값으로 대체된다."""


class HuggingFaceEmbedder:
    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.hf_embedding_model
        # TODO(내일): self._model = SentenceTransformer(self._model_name)

    @property
    def name(self) -> str:
        return f"huggingface:{self._model_name}"

    @property
    def dimension(self) -> int:
        return _STUB_DIMENSION

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # TODO(내일): await asyncio.to_thread(self._model.encode, texts, normalize_embeddings=True)
        return [[0.0] * self.dimension for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]
