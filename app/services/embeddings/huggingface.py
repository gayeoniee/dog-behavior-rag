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
        self._device = settings.embedding_device
        self._truncate = settings.embedding_truncate
        self._query_prefix = settings.embedding_query_prefix
        self._passage_prefix = settings.embedding_passage_prefix
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

        # "auto"면 None을 넘겨 sentence-transformers가 알아서 고르게 한다
        # (CUDA 있으면 GPU). VRAM을 LLM과 나눠 쓸 때 "cpu"로 강제할 수 있다 —
        # config.py의 embedding_device 주석 참조.
        device = None if self._device == "auto" else self._device
        logger.info(
            "임베딩 모델 로딩 시작: %s (device=%s, 최초 1회 수 GB 다운로드)",
            self._model_name,
            self._device,
        )
        # truncate_dim을 넘기면 sentence-transformers가 자르고 **다시 정규화**까지
        # 해준다. 직접 자르면 길이가 1이 아니게 되어 내적이 코사인 유사도가 아니게 된다.
        model = SentenceTransformer(
            self._model_name,
            device=device,
            truncate_dim=self._dimension if self._truncate else None,
        )

        # sentence-transformers 5.x에서 get_sentence_embedding_dimension이
        # get_embedding_dimension으로 개명됐다. 둘 다 지원해 버전에 안 묶이게 한다.
        read_dim = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        self._check_dimension(read_dim())

        model.max_seq_length = self._max_seq_length
        self._model = model

    def _check_dimension(self, actual: int) -> None:
        """모델이 내놓는 차원이 설정과 맞는가.

        모델 로딩에서 떼어냈다 — 이 판단은 순수 규칙이라 수 GB짜리 모델을 받지 않고도
        검증할 수 있어야 한다.
        """
        if self._truncate:
            # 자르기를 켰으면 모델이 설정보다 **크기만** 하면 된다. 작으면 못 늘린다.
            if actual < self._dimension:
                raise EmbeddingUnavailableError(
                    f"자를 수 없습니다: EMBEDDING_DIM={self._dimension}인데 "
                    f"모델 {self._model_name}는 {actual}차원입니다. "
                    "EMBEDDING_TRUNCATE는 모델 차원보다 작게 줄일 때만 씁니다."
                )
            logger.info("MRL 자르기: %d → %d차원", actual, self._dimension)
            return
        if actual != self._dimension:
            # 여기서 죽어야 한다. 통과시키면 훨씬 나중에 pgvector INSERT에서
            # 훨씬 불친절한 에러로 터지고, 원인을 차원 불일치로 되짚기 어렵다.
            raise EmbeddingUnavailableError(
                f"임베딩 차원 불일치: EMBEDDING_DIM={self._dimension}, "
                f"모델 {self._model_name}={actual}. "
                "설정을 고치고 `scripts.db.init --drop`으로 테이블을 다시 만드세요. "
                "MRL 지원 모델을 잘라 쓰려면 EMBEDDING_TRUNCATE=true."
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise EmbeddingUnavailableError(
                "임베딩 모델이 로딩되지 않았습니다 (warmup 미실행 또는 실패)"
            )
        if not texts:
            return []
        prefixed = [self._passage_prefix + t for t in texts] if self._passage_prefix else texts
        return await asyncio.to_thread(self._encode, prefixed)

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
        """질의 임베딩. **문서와 다른 접두사를 쓴다** (config의 주석 참조).

        `embed`를 재사용하지 않는 이유: `embed`는 문서용 접두사를 붙인다.
        질의에 그걸 붙이면 모델이 질의를 문서로 착각한다 — e5의 query:/passage:
        구분이 정확히 그것 때문에 있다.
        """
        if self._model is None:
            raise EmbeddingUnavailableError(
                "임베딩 모델이 로딩되지 않았습니다 (warmup 미실행 또는 실패)"
            )
        vectors = await asyncio.to_thread(self._encode, [self._query_prefix + text])
        return vectors[0]
