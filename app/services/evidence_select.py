"""검색된 근거 중 **질문에 실제로 답하는 것만** 골라낸다.

없으면 없다고 말하게 하는 것이 목적이다.

왜 필요한가: 검색은 항상 top_k개를 돌려준다. 코퍼스에 없는 주제를 물어도
"가장 덜 무관한" 5건이 나오고, LLM은 그걸로 그럴듯한 답을 만든다. 결과는
**모델의 일반 지식에 관련 없는 인용을 덧붙인 답**이다. 사용자 표현으로
"그냥 제미나이랑 대화하는 느낌".

왜 점수 임계값이 아닌가 (2026-08-12 측정):

    근거 있음  0.686~0.731
    주제 공백  0.636~0.693   ← 겹친다
    범위 밖    0.539~0.640

근거 있음의 최솟값이 주제 공백의 최댓값보다 낮다. bge-m3의 한→영 코사인은
dynamic range가 좁아 임계값이 성립하지 않는다. 그래서 LLM에게 관련성을 묻는다 —
팩트체크의 `not_covered`와 같은 패턴이고 거기서 잘 동작했다.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from app.services.llm.base import LLMClient, LLMUnavailableError
from app.services.vectorstore.base import SearchHit

logger = logging.getLogger(__name__)

Coverage = Literal["full", "partial", "none"]

NO_EVIDENCE_NOTE = (
    "이 주제는 현재 코퍼스(AVSAB·AAHA·RSPCA·VCA·ASPCA·PMC 오픈액세스)에 "
    "근거 자료가 없습니다. 아래 답변은 근거 문서 없이 작성된 일반적인 안내이므로 "
    "수의사나 공인 훈련사에게 확인하시기 바랍니다."
)

SELECT_SYSTEM = """You decide which of the numbered excerpts actually help answer the \
user's question about dog behaviour or training.

Output ONLY a JSON array of the excerpt numbers that are genuinely useful, e.g. [1,3].
No prose, no markdown fences.

Rules:
- Keep an excerpt ONLY if it contains information that answers this specific question.
- Being on a broadly related topic is NOT enough. An excerpt about separation anxiety \
does not help a question about jumping up. An excerpt about dog behaviour research \
methodology does not help a question asking what to do.
- If NONE of the excerpts answer the question, output [] — this is a normal and \
expected outcome. Do not stretch to find relevance.
- Judge only what the excerpts say, not what you happen to know."""

_JSON_ARRAY = re.compile(r"\[[^\]]*\]", re.DOTALL)
_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_MAX_SELECT_TOKENS = 60
_SNIPPET_CHARS = 700
"""선별용 프롬프트에 넣는 근거 길이. 전문을 넣으면 5×1200자라 판정만으로 느려진다.
앞부분 700자면 주제 판단에는 충분하고, 답변 생성에는 여전히 전문이 들어간다."""


@dataclass(slots=True)
class Selection:
    kept: list[SearchHit] = field(default_factory=list)
    coverage: Coverage = "full"
    note: str | None = None


def parse_indices(raw: str, count: int) -> list[int] | None:
    """LLM 출력 → 0-based 인덱스 목록. 해석 실패면 None (호출자가 폴백).

    빈 배열 `[]`은 실패가 아니라 **"근거 없음"이라는 유효한 판정**이므로
    None과 구분해야 한다.
    """
    text = _THINK.sub("", raw).strip()
    match = _JSON_ARRAY.search(text)
    if not match:
        return None
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    # 1-based로 답하라고 했지만 범위를 벗어난 숫자는 조용히 버린다.
    return sorted({int(i) - 1 for i in items if isinstance(i, int) and 1 <= i <= count})


class EvidenceSelector:
    def __init__(self, llm: LLMClient | None, *, enabled: bool = True) -> None:
        self._llm = llm
        self._enabled = enabled

    async def select(self, question: str, hits: list[SearchHit]) -> Selection:
        if not hits:
            return Selection(kept=[], coverage="none", note=NO_EVIDENCE_NOTE)
        if not self._enabled or self._llm is None:
            return Selection(kept=hits, coverage="full")

        try:
            raw = await self._llm.generate(
                self._build_prompt(question, hits),
                system=SELECT_SYSTEM,
                max_tokens=_MAX_SELECT_TOKENS,
            )
        except LLMUnavailableError as exc:
            # 선별 실패가 답변을 막으면 안 된다. 검색 결과를 그대로 쓴다.
            logger.warning("근거 선별 실패, 검색 결과를 그대로 사용: %s", exc)
            return Selection(kept=hits, coverage="partial")
        except Exception as exc:  # noqa: BLE001 — 어떤 이유로도 답변을 막지 않는다
            logger.warning("근거 선별 중 예외, 검색 결과를 그대로 사용: %s", exc)
            return Selection(kept=hits, coverage="partial")

        indices = parse_indices(raw, len(hits))
        if indices is None:
            logger.warning("근거 선별 결과를 해석하지 못해 그대로 사용: %r", raw[:80])
            return Selection(kept=hits, coverage="partial")

        if not indices:
            logger.info("근거 없음 판정 — 질문: %r", question[:50])
            return Selection(kept=[], coverage="none", note=NO_EVIDENCE_NOTE)

        kept = [hits[i] for i in indices]
        coverage: Coverage = "full" if len(kept) == len(hits) else "partial"
        logger.info("근거 선별: %d건 중 %d건 유지", len(hits), len(kept))
        return Selection(kept=kept, coverage=coverage)

    @staticmethod
    def _build_prompt(question: str, hits: list[SearchHit]) -> str:
        blocks = [
            f"[{i}] {hit.document_title}\n{hit.content[:_SNIPPET_CHARS]}"
            for i, hit in enumerate(hits, start=1)
        ]
        return "\n\n".join(blocks) + f"\n\nQuestion: {question}"
