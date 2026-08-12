"""HuggingFace(sentence-transformers) 임베딩 구현.

기본 모델은 BAAI/bge-m3. 코퍼스는 영어, 질문은 한국어인 교차언어 검색이라
다국어 모델이 필수다.

`sentence_transformers` import를 `_load()` 안까지 미루는 게 중요하다 — 모듈 최상단에서
import하면 `uv sync --extra hf` 없이는 앱이 뜨지도, 테스트가 돌지도 않는다.
"""

import asyncio
import logging
from typing import Any

from app.core.config import Settings

from .base import EmbeddingUnavailableError

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder:
    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.hf_embedding_model
        self._dimension = settings.embedding_dim
        self._batch_size = settings.embedding_batch_size
        self._max_seq_length = settings.embedding_max_seq_length
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return f"huggingface:{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    async def warmup(self) -> None:
        if self._model is None:
            await asyncio.to_thread(self._load)

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "sentence-transformers가 설치돼 있지 않습니다 — uv sync --extra hf"
            ) from exc

        logger.info("임베딩 모델 로딩 시작: %s (최초 1회 수 GB 다운로드)", self._model_name)
        model = SentenceTransformer(self._model_name)

        # sentence-transformers 5.x에서 get_sentence_embedding_dimension이
        # get_embedding_dimension으로 개명됐다. 둘 다 지원해 버전에 안 묶이게 한다.
        read_dim = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        actual = read_dim()
        if actual != self._dimension:
            # 여기서 죽어야 한다. 통과시키면 훨씬 나중에 pgvector INSERT에서
            # 훨씬 불친절한 에러로 터지고, 원인을 차원 불일치로 되짚기 어렵다.
            raise EmbeddingUnavailableError(
                f"임베딩 차원 불일치: EMBEDDING_DIM={self._dimension}, "
                f"모델 {self._model_name}={actual}. "
                "설정을 고치고 `scripts.db.init --drop`으로 테이블을 다시 만드세요."
            )

        model.max_seq_length = self._max_seq_length
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise EmbeddingUnavailableError(
                "임베딩 모델이 로딩되지 않았습니다 (warmup 미실행 또는 실패)"
            )
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self._batch_size,
            # 필수. 코사인 거리와 내적을 일치시키고 HNSW의 vector_cosine_ops와 맞춘다.
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    async def embed_query(self, text: str) -> list[float]:
        # bge-m3는 질의 프리픽스를 붙이지 않는다 (base.py의 embed_query 독스트링 참조).
        vectors = await self.embed([text])
        return vectors[0]
