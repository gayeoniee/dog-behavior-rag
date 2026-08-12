"""RAG 오케스트레이션.

파이프라인 전체 흐름을 여기서만 관리한다. 각 단계의 실제 구현은
embeddings / vectorstore / llm 하위 모듈이 담당.
"""

import logging
import time

from app.schemas.chat import ChatResponse, SourceChunk
from app.services.embeddings.base import Embedder
from app.services.evidence_select import EvidenceSelector
from app.services.llm.base import LLMClient
from app.services.query_rewrite import QueryRewriter
from app.services.vectorstore.base import SearchHit, VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 반려동물 훈련 전문가입니다.
주어진 참고 자료에 근거해서만 답변하세요.
자료에 없는 내용은 추측하지 말고 모른다고 답하세요.
의학적 처치가 필요해 보이면 수의사 상담을 권하세요."""

NO_EVIDENCE_SYSTEM_PROMPT = """당신은 반려동물 훈련 전문가입니다.

**참고 자료가 하나도 없는 상태입니다.** 다음을 반드시 지키세요:

1. 첫 문장에서 "이 주제는 참고 자료에 없습니다"라고 명확히 밝히세요.
2. 자료가 있는 것처럼 인용하지 마세요.
3. 반려견 훈련·행동과 무관한 질문(사료 브랜드 추천, 가격, 장소 추천 등)이라면
   답변 범위를 벗어난다고 알리고 끝내세요.
4. 반려견 훈련·행동 질문이라면 일반적인 방향만 2~3문장으로 짧게 안내하고,
   반드시 수의사나 공인 훈련사(CPDT, Dip ACVB) 상담을 권하세요.
5. 길게 쓰지 마세요. 근거가 없을 때 길게 쓰는 것은 지어내는 것입니다."""
"""근거가 0건일 때 쓰는 프롬프트.

같은 프롬프트로 "자료 없음"을 처리하면 모델이 습관적으로 답을 만들어낸다.
아예 다른 계약을 준다 — 짧게, 없다고 먼저 말하고, 전문가에게 넘긴다.
"""


class RagService:
    """질문 → 검색 → 프롬프트 구성 → 생성."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        default_top_k: int = 5,
        rewriter: QueryRewriter | None = None,
        selector: EvidenceSelector | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._default_top_k = default_top_k
        # 없으면 재작성 없이 원문으로 검색한다 (테스트·LLM 미연결 상황).
        self._rewriter = rewriter or QueryRewriter(None, enabled=False)
        # 없으면 선별 없이 검색 결과를 그대로 쓴다 (기존 동작).
        self._selector = selector or EvidenceSelector(None, enabled=False)

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

        # 4) 질문에 실제로 답하는 근거만 남긴다. 검색은 항상 top_k를 돌려주므로
        #    이 단계가 없으면 코퍼스에 없는 주제에도 "가장 덜 무관한" 5건으로
        #    그럴듯한 답을 만들게 된다 (evidence_select.py 참조).
        selection = await self._selector.select(question, hits)

        # 5) 프롬프트 구성 — 재작성본이 아니라 **원문**을 넣는다.
        #    재작성은 검색용 표현이고, 사용자가 실제로 물은 건 원문이다.
        if selection.coverage == "none":
            prompt = f"질문: {question}\n\n(참고 자료 없음)"
            system = NO_EVIDENCE_SYSTEM_PROMPT
        else:
            prompt = self._build_prompt(question, selection.kept)
            system = SYSTEM_PROMPT

        # 6) 답변 생성
        answer = await self._llm.generate(prompt, system=system)

        if selection.coverage == "none":
            logger.info("근거 없음으로 응답 — 질문: %r", question[:50])

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ChatResponse(
            answer=answer,
            # 근거 없음이면 sources를 비운다. 관련 없는 청크를 인용처럼 보여주는 게
            # 지금 신뢰를 깎는 지점이다.
            sources=[self._to_source(h) for h in selection.kept],
            latency_ms=elapsed_ms,
            provider=self._llm.name,
            coverage=selection.coverage,
            coverage_note=selection.note,
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
