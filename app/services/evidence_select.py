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

Coverage = Literal["full", "partial", "none", "needs_detail"]
"""근거 충분도.

`none`과 `needs_detail`을 가르는 게 중요하다. 둘 다 근거가 없지만 대응이 다르다:

  none          질문 자체가 범위 밖 (고양이 모래, 사료 가격) → 답하지 않는다
  needs_detail  개 행동 질문은 맞는데 정보가 부족해 원인을 좁힐 수 없다 → **되묻는다**

실제 사례: "강아지가 벽을 자꾸 긁어"는 분리불안·지루함·강박 중 무엇인지에 따라
대응이 완전히 달라서 그대로는 답할 수 없다. 그런데 "혼자 있을 때 벽을 긁어"라고
하면 분리불안 문서가 0.771로 바로 올라온다. 훈련사라면 거절하는 게 아니라
"혼자 있을 때만 그러나요?"라고 물어야 한다.
"""

NO_EVIDENCE_NOTE = (
    "이 주제는 현재 코퍼스(AVSAB·AAHA·RSPCA·VCA·ASPCA·PMC 오픈액세스)에 "
    "근거 자료가 없습니다. 아래 답변은 근거 문서 없이 작성된 일반적인 안내이므로 "
    "수의사나 공인 훈련사에게 확인하시기 바랍니다."
)

NEEDS_DETAIL_NOTE = (
    "질문만으로는 원인을 좁힐 수 없어 근거 자료를 고르지 못했습니다. "
    "아래 되묻는 내용에 답해 주시면 해당하는 자료를 찾아 답변드릴 수 있습니다."
)

SELECT_SYSTEM = """You do two things with the numbered excerpts and the user's question.

Output ONLY a JSON object, no prose or markdown fences:
{"keep": [1,3], "in_domain": true}

**keep** — excerpt numbers that genuinely help answer this specific question.
- Keep an excerpt ONLY if it contains information that answers the question.
- Being on a broadly related topic is NOT enough. An excerpt about separation anxiety \
does not help a question about jumping up. Research methodology does not help a \
question asking what to do.
- If none qualify, output an empty list. This is a normal outcome; do not stretch.

**in_domain** — is the question about dog behaviour, training, or a dog's wellbeing?
- true: any question about what a dog does, why, or how to train it — even if vague, \
even if the excerpts do not cover it. "My dog scratches the wall" is in_domain.
- false: cat care, product or brand recommendations, prices, clinic or cafe locations, \
anything not about dog behaviour or training."""

_JSON_ARRAY = re.compile(r"\[[^\]]*\]", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
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


def parse_selection(raw: str, count: int) -> tuple[list[int], bool] | None:
    """LLM 출력 → (0-based 인덱스 목록, in_domain). 해석 실패면 None (호출자가 폴백).

    빈 목록은 실패가 아니라 **"쓸 근거가 없다"는 유효한 판정**이므로 None과
    구분해야 한다. in_domain이 빠져 있으면 True로 본다 — 개 질문인데 범위 밖으로
    잘못 처리하는 쪽이 그 반대보다 나쁘다.
    """
    text = _THINK.sub("", raw).strip()

    match = _JSON_OBJECT.search(text)
    if match:
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("keep"), list):
            return _clean_indices(data["keep"], count), bool(data.get("in_domain", True))

    # 지시를 무시하고 배열만 뱉는 경우 (작은 모델에서 흔하다).
    array_match = _JSON_ARRAY.search(text)
    if array_match:
        try:
            items = json.loads(array_match.group())
        except json.JSONDecodeError:
            return None
        if isinstance(items, list):
            return _clean_indices(items, count), True
    return None


def _clean_indices(items: list, count: int) -> list[int]:
    """1-based 번호 목록 → 0-based. 범위를 벗어난 숫자는 조용히 버린다."""
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

        parsed = parse_selection(raw, len(hits))
        if parsed is None:
            logger.warning("근거 선별 결과를 해석하지 못해 그대로 사용: %r", raw[:80])
            return Selection(kept=hits, coverage="partial")

        indices, in_domain = parsed
        if not indices:
            # 개 질문인데 근거를 못 고른 것과, 애초에 범위 밖인 것은 다르다.
            # 전자는 되물어야 하고 후자는 답하지 않아야 한다.
            if in_domain:
                logger.info("정보 부족 — 되묻기로 전환: %r", question[:50])
                return Selection(kept=[], coverage="needs_detail", note=NEEDS_DETAIL_NOTE)
            logger.info("범위 밖 판정 — 질문: %r", question[:50])
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
