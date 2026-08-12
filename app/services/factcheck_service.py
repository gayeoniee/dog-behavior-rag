"""팩트체크: 어디서 본 훈련 조언을 코퍼스와 대조한다.

유튜브·블로그를 코퍼스에 넣지 않기로 한 결정과 짝을 이루는 기능이다. 오염원으로
들이는 대신 검증 대상으로 받는다. 코퍼스에 `avsab-dominance` 같은 반박 근거가
있기 때문에 성립한다 — 지배이론 주장을 넣으면 그게 근거로 올라온다.

파이프라인:

    추출(LLM) → 재작성(LLM) → 검색 → 판정(LLM)

앞의 두 단계는 chat과 공유한다(query_rewrite). 판정만 여기 있다.
"""

import asyncio
import json
import logging
import re
import time

from app.schemas.chat import SourceChunk
from app.schemas.factcheck import ClaimVerdict, FactCheckResponse, Verdict
from app.services.embeddings.base import Embedder
from app.services.llm.base import LLMClient
from app.services.query_rewrite import QueryRewriter
from app.services.vectorstore.base import SearchHit, VectorStore

logger = logging.getLogger(__name__)

MAX_CLAIMS = 5

CORPUS_NOTE = (
    "이 판정은 기관·학술 자료(AVSAB, AAHA, RSPCA, VCA, PMC 오픈액세스)로만 이뤄진 "
    "코퍼스를 기준으로 합니다. 이 코퍼스는 보상 기반 훈련법 문헌으로 구성돼 있어 "
    "혐오·지배 기반 주장은 구조적으로 '배치'로 판정되는 경향이 있습니다. "
    "의도된 설계지만, 중립적인 제3자 판정이 아니라는 점을 감안해 주세요."
)

EXTRACT_SYSTEM = f"""You extract checkable factual claims about dog behaviour and training \
from user-supplied text (a video transcript, blog post, or advice they heard).

Rules:
- Output ONLY a JSON array of strings. No prose, no markdown fences.
- At most {MAX_CLAIMS} claims. Prefer the ones that assert something checkable about \
dog behaviour, causes, or training technique.
- Write each claim in Korean, as a single self-contained sentence.
- Skip greetings, opinions about products, and anything not about dog behaviour.
- If there is no checkable claim, output [].

Example output: ["마운팅은 서열이 높다고 생각해서 하는 행동이다", "복종 자세를 강제해야 한다"]"""

JUDGE_SYSTEM = """You judge whether a claim about dog behaviour is supported by the \
provided veterinary-literature excerpts.

Output ONLY a JSON object, no prose or markdown fences:
{"verdict": "supported" | "contradicted" | "not_covered", "explanation": "<Korean, 2-4 sentences>"}

Rules:
- "supported"    : the excerpts affirm the claim.
- "contradicted" : the excerpts state the opposite, or say the claim's premise is wrong.
- "not_covered"  : the excerpts do not address the claim. USE THIS FREELY. Excerpts that \
are merely on a related topic are NOT evidence. Never guess from your own knowledge.
- Base the explanation ONLY on the excerpts. Quote or paraphrase what they actually say.
- Refer to excerpts as [자료 1], [자료 2] … matching the numbering given."""

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _strip_fences(text: str) -> str:
    text = re.sub(r"<(think|thinking|reasoning)>.*?</\1>", "", text, flags=re.DOTALL | re.I)
    return text.strip()


def parse_claims(raw: str, *, fallback: str) -> list[str]:
    """LLM 출력 → 주장 리스트. 파싱에 실패하면 원문 전체를 하나의 주장으로 본다.

    폴백이 중요하다 — 추출이 깨졌다고 검증을 아예 못 하면 기능이 죽는다.
    원문 하나를 통째로 검증하는 건 덜 정밀할 뿐 틀린 동작은 아니다.
    """
    match = _JSON_ARRAY.search(_strip_fences(raw))
    if match:
        try:
            items = json.loads(match.group())
            claims = [str(c).strip() for c in items if str(c).strip()]
            if claims:
                return claims[:MAX_CLAIMS]
        except json.JSONDecodeError:
            logger.warning("주장 추출 JSON 파싱 실패: %r", raw[:120])
    return [fallback.strip()[:500]]


_VALID_VERDICTS: tuple[Verdict, ...] = ("supported", "contradicted", "not_covered")


def parse_verdict(raw: str) -> tuple[Verdict, str]:
    """LLM 출력 → (verdict, explanation). 파싱 실패는 not_covered로 떨어뜨린다.

    반환 타입이 Verdict라서, 모델이 만들어낸 임의의 라벨이 enum 밖으로 새는 걸
    타입 검사기가 잡아준다.
    """
    match = _JSON_OBJECT.search(_strip_fences(raw))
    if match:
        try:
            data = json.loads(match.group())
            candidate = str(data.get("verdict", "")).strip().lower()
            explanation = str(data.get("explanation", "")).strip()
            for valid in _VALID_VERDICTS:
                if candidate == valid:
                    return valid, explanation
        except json.JSONDecodeError:
            pass
    logger.warning("판정 JSON 파싱 실패, not_covered로 처리: %r", raw[:120])
    return "not_covered", "판정 결과를 해석하지 못했습니다."


class FactCheckService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        rewriter: QueryRewriter,
        default_top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._rewriter = rewriter
        self._default_top_k = default_top_k

    async def check(self, text: str, top_k: int | None = None) -> FactCheckResponse:
        started = time.perf_counter()
        k = top_k or self._default_top_k

        raw_claims = await self._llm.generate(text, system=EXTRACT_SYSTEM, max_tokens=400)
        claims = parse_claims(raw_claims, fallback=text)
        logger.info("주장 %d개 추출", len(claims))

        # 주장 5개를 순차로 돌리면 LLM 왕복이 11회라 1분 가까이 걸린다(실측 57초).
        # LLM 호출은 서로 독립이라 묶어서 보낸다. **DB는 병렬로 쓰지 않는다** —
        # AsyncSession 하나는 커넥션 하나라 동시 쿼리를 못 견딘다. 그래서
        # LLM 구간만 gather하고 검색은 순차로 둔다 (검색은 수십 ms라 병목이 아니다).
        search_queries = await asyncio.gather(*(self._rewriter.rewrite(c) for c in claims))
        hits_per_claim = [await self._search(q, k) for q in search_queries]
        verdicts = await asyncio.gather(
            *(
                self._judge(claim, hits)
                for claim, hits in zip(claims, hits_per_claim, strict=True)
            )
        )

        return FactCheckResponse(
            claims=verdicts,
            corpus_note=CORPUS_NOTE,
            latency_ms=int((time.perf_counter() - started) * 1000),
            provider=self._llm.name,
        )

    async def _search(self, search_query: str, top_k: int) -> list[SearchHit]:
        """재작성된 질의로 근거를 찾는다.

        재작성을 태우는 이유는 chat과 같지만 여기서 더 중요하다 — 검증할 주장이
        대개 기법("복종 자세를 유지한다")에 대한 것이고, 그게 정확히 bge-m3가
        교차언어로 못 넘기는 부분이다.
        """
        return await self._store.search(await self._embedder.embed_query(search_query), top_k)

    async def _judge(self, claim: str, hits: list[SearchHit]) -> ClaimVerdict:
        if not hits:
            return ClaimVerdict(
                claim=claim,
                verdict="not_covered",
                explanation="코퍼스에서 관련 근거를 찾지 못했습니다.",
                sources=[],
            )

        raw = await self._llm.generate(
            self._build_prompt(claim, hits), system=JUDGE_SYSTEM, max_tokens=500
        )
        verdict, explanation = parse_verdict(raw)

        # 인용 없는 단정을 코드에서 막는다. 프롬프트로 부탁하지 않는 이유는
        # 모델이 지킬 거라고 믿을 수 없기 때문이다.
        sources = [_to_source(h) for h in hits]
        if verdict != "not_covered" and not sources:
            verdict = "not_covered"

        return ClaimVerdict(claim=claim, verdict=verdict, explanation=explanation, sources=sources)

    @staticmethod
    def _build_prompt(claim: str, hits: list[SearchHit]) -> str:
        blocks = [
            f"[자료 {i}] {hit.document_title}\n{hit.content}" for i, hit in enumerate(hits, start=1)
        ]
        return (
            "다음 자료를 근거로 주장을 판정하세요.\n\n" + "\n\n".join(blocks) + f"\n\n주장: {claim}"
        )


def _to_source(hit: SearchHit) -> SourceChunk:
    return SourceChunk(
        chunk_id=hit.chunk_id,
        document_title=hit.document_title,
        content=hit.content,
        score=hit.score,
        source=hit.source,
    )
