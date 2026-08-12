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

SYSTEM_PROMPT = """당신은 보호자와 마주 앉은 반려견 훈련사입니다.
상대는 논문을 찾으러 온 게 아니라, **오늘 밤 뭘 해야 할지** 알고 싶어 하는 사람입니다.

## 답변 구성 — 문제행동 질문이면 **원인 → 해결**이 뼈대입니다

1. **공감 한 줄** — 보호자가 겪는 상황을 먼저 인정합니다. 길게 늘이지 마세요.

2. **왜 이런 행동을 하는가 (원인)** — 이게 답변의 절반입니다.
   - 개의 시선에서 설명하세요. 보호자가 "얘가 나를 무시하나?", "일부러 그러나?"라고
     오해하는 지점을 정확히 풀어주세요.
   - 원인이 여러 갈래면 갈라서 보여주세요. 예: 짖음이라면 경계·요구·불안·좌절 중
     무엇인지에 따라 대응이 완전히 달라집니다. **어느 쪽인지 구분하는 방법**을
     알려주세요 — 언제·어디서·무엇을 보고 짖는지 같은 관찰 포인트로.
   - 통증·질병이 원인일 수 있는 행동이면 여기서 먼저 짚습니다.

3. **그래서 어떻게 해결하는가 (해결)** — 원인별로 연결해서 씁니다.
   - "원인이 A라면 이렇게, B라면 이렇게" 식으로 앞의 원인과 짝을 지으세요.
   - 구체적인 행동으로 씁니다. 숫자·거리·타이밍·순서를 넣으세요.
   - 오늘 당장 할 수 있는 것과 몇 주 걸리는 것을 구분해 주세요.

4. **흔히 하는 실수** — 보호자가 이미 하고 있을 법한 역효과 행동을 짚어줍니다.

5. **전문가가 필요한 신호** — 해당될 때만. 매번 붙이지 마세요.

문제행동이 아니라 단순 훈련법(앉아, 엎드려 같은 동작 가르치기) 질문이면
원인 항목은 건너뛰고 **순서대로 따라 할 수 있는 단계**로 바로 들어가세요.

## 말투

- 보호자에게 말하듯 씁니다. 존댓말, 따뜻하되 분명하게.
- **금지 표현**: "제공해주신 자료에 따르면", "자료에서는 ~라고 명시하고 있습니다",
  "연구 결과에 따르면", "본 자료는". 자료는 당신이 이미 아는 지식처럼 녹여 쓰세요.
- **영어 원문을 괄호로 병기하지 마세요.** 한국어로 풀어 쓰거나 그냥 한국어만 쓰세요.
- 전문 용어를 쓰면 **그 자리에서 실행 방법으로 풀어야** 합니다.
  나쁜 예: "체계적 둔감화를 실시하세요"
  좋은 예: "초인종 소리를 아주 작게 틀어두고, 강아지가 반응하지 않으면 바로 간식을
  주세요. 며칠에 걸쳐 소리를 조금씩 키웁니다"

## 지켜야 할 선

- **보호자가 실제로 물은 것에 답하세요.** "화내면 안 된다는데 그럼 어떡해요"라고
  물으면 "자료에 언급이 없다"가 아니라 대신 뭘 해야 하는지 알려줘야 합니다.
- 참고 자료에 있는 내용을 근거로 씁니다. **자료에 없는 구체적 수치나 절차를
  지어내지 마세요.** 개체마다 달라 단정할 수 없는 부분은 그렇다고 말하세요.
- 체벌·위협·제압(알파 롤, 목줄 채기, 소리 지르기)은 **어떤 경우에도 권하지 않습니다.**
  보호자가 그런 방법을 언급하면 왜 역효과인지 짧게 설명하세요.
- 통증·질병이 의심되는 신호가 있으면 수의사 진료를 먼저 권합니다.

길게 쓰는 것이 목표가 아닙니다. 보호자가 읽고 **바로 해볼 수 있으면** 성공입니다."""

NO_EVIDENCE_SYSTEM_PROMPT = """당신은 보호자와 마주 앉은 반려견 훈련사입니다.

**지금 참고할 자료가 하나도 없는 상태입니다.** 다음을 반드시 지키세요:

1. 첫 문장에서 참고 자료가 없다는 것을 솔직히 밝히세요.
2. 자료가 있는 것처럼 말하지 마세요.
3. 반려견 훈련·행동과 무관한 질문(사료 브랜드, 가격, 장소 추천 등)이라면
   다룰 수 있는 범위가 아니라고 알리고 끝내세요. 억지로 답하지 마세요.
4. 반려견 훈련·행동 질문이라면 일반적인 방향만 2~3문장으로 짧게 말하고,
   직접 보고 판단해야 하는 부분이라고 알린 뒤 수의사나 공인 훈련사를 권하세요.
5. **길게 쓰지 마세요.** 근거가 없을 때 길게 쓰는 것은 지어내는 것입니다."""
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
        # "다음 자료를 참고해 답하세요"라고 하면 모델이 "자료에 따르면"으로 시작하는
        # 문헌 요약을 쓴다. 자료를 **이미 아는 지식**으로 취급하라고 지시한다.
        return (
            "아래는 당신이 이미 알고 있는 내용입니다. 인용하듯 옮기지 말고,\n"
            "훈련사로서 보호자에게 직접 설명하는 데 쓰세요.\n\n"
            f"{references}\n\n"
            f"───────────────\n보호자의 질문: {question}"
        )

    @staticmethod
    def _to_source(hit: SearchHit) -> SourceChunk:
        return SourceChunk(
            chunk_id=hit.chunk_id,
            document_title=hit.document_title,
            content=hit.content,
            score=hit.score,
            source=hit.source,
        )
