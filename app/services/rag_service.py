"""RAG 오케스트레이션.

파이프라인 전체 흐름을 여기서만 관리한다. 각 단계의 실제 구현은
embeddings / vectorstore / llm 하위 모듈이 담당.
"""

import logging
import time

from app.schemas.chat import ChatResponse, SourceChunk
from app.services.embeddings.base import Embedder
from app.services.llm.base import LLMClient
from app.services.vectorstore.base import SearchHit, VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 반려동물 훈련 전문가입니다.
주어진 참고 자료에 근거해서만 답변하세요.
자료에 없는 내용은 추측하지 말고 모른다고 답하세요.
의학적 처치가 필요해 보이면 수의사 상담을 권하세요."""
# TODO(내일): 프롬프트 튜닝. 지금은 자리만 잡아둔 것.


class RagService:
    """질문 → 검색 → 프롬프트 구성 → 생성."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        default_top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._default_top_k = default_top_k

    async def answer(self, question: str, top_k: int | None = None) -> ChatResponse:
        started = time.perf_counter()
        k = top_k or self._default_top_k

        # 1) 질문 임베딩
        query_vector = await self._embedder.embed_query(question)

        # 2) 유사 청크 검색
        hits = await self._store.search(query_vector, k)

        # 3) 검색 결과로 프롬프트 구성
        prompt = self._build_prompt(question, hits)

        # 4) 답변 생성
        answer = await self._llm.generate(prompt, system=SYSTEM_PROMPT)

        if not hits:
            # 스켈레톤 단계에서는 항상 여기로 온다 (vectorstore가 아직 스텁).
            logger.info("검색 결과 없음 — 아직 적재된 문서가 없거나 vectorstore가 스텁 상태입니다")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ChatResponse(
            answer=answer,
            sources=[self._to_source(h) for h in hits],
            latency_ms=elapsed_ms,
            provider=self._llm.name,
        )

    def _build_prompt(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return f"질문: {question}\n\n(참고 자료 없음)"

        blocks = [
            f"[자료 {i}] {hit.document_title}\n{hit.content}" for i, hit in enumerate(hits, start=1)
        ]
        references = "\n\n".join(blocks)
        return f"다음 자료를 참고해 질문에 답하세요.\n\n{references}\n\n질문: {question}"

    @staticmethod
    def _to_source(hit: SearchHit) -> SourceChunk:
        return SourceChunk(
            chunk_id=hit.chunk_id,
            document_title=hit.document_title,
            content=hit.content,
            score=hit.score,
            source=hit.source,
        )
