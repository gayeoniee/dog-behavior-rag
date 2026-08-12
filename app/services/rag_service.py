"""RAG 오케스트레이션.

파이프라인 전체 흐름을 여기서만 관리한다. 각 단계의 실제 구현은
embeddings / vectorstore / llm 하위 모듈이 담당.
"""

import logging
import time

from app.schemas.chat import ChatResponse, SourceChunk
from app.services.embeddings.base import Embedder
from app.services.llm.base import LLMClient
from app.services.query_rewrite import QueryRewriter
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
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._default_top_k = default_top_k
        # 없으면 재작성 없이 원문으로 검색한다 (테스트·LLM 미연결 상황).
        self._rewriter = rewriter or QueryRewriter(None, enabled=False)

    async def answer(self, question: str, top_k: int | None = None) -> ChatResponse:
        started = time.perf_counter()
        k = top_k or self._default_top_k

        # 1) 검색용 질의 재작성 (한국어 → 영어 기술표현).
        #    bge-m3가 기법 명칭을 교차언어로 못 넘기기 때문 — query_rewrite.py 참조.
        #    실패해도 원문이 그대로 돌아오므로 검색은 계속된다.
        search_query = await self._rewriter.rewrite(question)

        # 2) 질의 임베딩
        query_vector = await self._embedder.embed_query(search_query)

        # 3) 유사 청크 검색
        hits = await self._store.search(query_vector, k)

        # 4) 검색 결과로 프롬프트 구성 — 프롬프트에는 재작성본이 아니라 **원문**을 넣는다.
        #    재작성은 검색용 표현이고, 사용자가 실제로 물은 건 원문이다.
        prompt = self._build_prompt(question, hits)

        # 5) 답변 생성
        answer = await self._llm.generate(prompt, system=SYSTEM_PROMPT)

        if not hits:
            logger.info("검색 결과 없음 — 적재된 문서가 없거나 질의가 코퍼스 범위를 벗어났습니다")

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
