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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from app.schemas.chat import Turn
from app.services.llm.base import LLMClient, LLMUnavailableError
from app.services.query_rewrite import format_history
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

_CONTEXT_HINT = (
    "Read the question in the context of the conversation above. A follow-up like "
    '"would crate training help?" is about whatever the conversation was about — '
    "judge the excerpts against that, not against the bare sentence.\n"
)
"""맥락이 있을 때만 붙이는 지시.

처음에는 SELECT_SYSTEM에 넣었는데 **단발 질문 두 개가 깨졌다** (18/20 → 16/20).
대화가 없는데 "이전 대화가 주어지면…"을 항상 읽히면 잡음이 된다 — 초인종 질문은
검색 결과가 완전히 같은데 판정만 뒤집혔고, 꼬리 질문은 근거를 하나도 안 남겼다.
쓰이는 자리에만 붙여야 한다.
"""

DOMAIN_SYSTEM = """Is the question asking about a DOG's behaviour, training, or wellbeing?

Answer with exactly one word: DOG or OTHER.

DOG — what a dog does, why it does it, how to train it, whether a behaviour is normal, \
or a dog's health as it affects behaviour. Vague ones still count: "my dog scratches the \
wall", "he barks a lot".
OTHER — everything else, even if a dog is mentioned: cats or other animals, product or \
brand recommendations, prices and costs, surgery fees, where to buy something, clinic or \
cafe locations, adoption paperwork, insurance.

Examples:
"고양이 화장실 모래는 어떤 게 좋아요?" -> OTHER
"강아지 사료 브랜드 추천해주세요" -> OTHER
"강아지 중성화 수술 비용이 얼마예요?" -> OTHER
"강아지가 벽을 자꾸 긁어요" -> DOG
"켄넬에 들어가기 싫어해요" -> DOG"""
"""범위 판정을 근거 선별에서 떼어낸 이유 (2026-08-13 실측).

한 번의 호출로 "쓸 근거 고르기"와 "개 질문인지"를 같이 시키면 gemma-4-e2b(4.6B)가
후자를 놓친다. 선별 프롬프트에 이미 "고양이·가격·브랜드는 false"라고 적혀 있는데도
고양이 모래와 중성화 비용을 in_domain=true로 판정해 out-of-scope가 1/4이었다.

작은 모델이 확실히 하는 형태는 **한 가지만 묻는 단답 분류**다. 큰 모델에는 불필요한
분리지만 손해도 없다 — 근거가 하나도 안 남았을 때만 부르므로 일반 경로의 호출 수는
그대로다. 답할 근거가 있으면 범위 밖인지 물을 이유가 애초에 없다.
"""

_JSON_ARRAY = re.compile(r"\[[^\]]*\]", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_MAX_SELECT_TOKENS = 60
_MAX_DOMAIN_TOKENS = 4
"""한 단어만 받으면 되므로 4토큰. 추론형이면 provider가 사고과정 여유분을 더한다."""
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


def parse_domain(raw: str) -> bool | None:
    """DOG/OTHER 단답 → in_domain. 둘 다 없거나 둘 다 있으면 None (호출자가 폴백).

    부분 문자열로 보지 않고 단어로 본다 — 설명을 덧붙이는 모델이 "not a DOG question,
    it is OTHER"처럼 둘 다 말할 수 있어서, 그때는 판정하지 않는 편이 낫다.
    """
    text = _THINK.sub("", raw).upper()
    words = set(re.findall(r"[A-Z]+", text))
    is_dog, is_other = "DOG" in words, "OTHER" in words
    if is_dog == is_other:
        return None
    return is_dog


def _clean_indices(items: list, count: int) -> list[int]:
    """1-based 번호 목록 → 0-based. 범위를 벗어난 숫자는 조용히 버린다."""
    return sorted({int(i) - 1 for i in items if isinstance(i, int) and 1 <= i <= count})


class EvidenceSelector:
    def __init__(self, llm: LLMClient | None, *, enabled: bool = True) -> None:
        self._llm = llm
        self._enabled = enabled

    async def select(
        self, question: str, hits: list[SearchHit], history: Sequence[Turn] = ()
    ) -> Selection:
        """근거를 고른다. history를 주면 후속 질문을 그 맥락에서 판정한다.

        맥락이 없으면 "켄넬 훈련이 도움이 될까?" 같은 후속 질문이 **무엇에 대한
        질문인지 알 수 없어** 근거를 못 고르고 되묻기로 떨어진다. 검색(재작성)은
        이미 맥락을 쓰고 있었는데 판정만 안 쓰던 비대칭을 없앤다.
        """
        if not hits:
            return Selection(kept=[], coverage="none", note=NO_EVIDENCE_NOTE)
        if not self._enabled or self._llm is None:
            return Selection(kept=hits, coverage="full")

        try:
            raw = await self._llm.generate(
                self._build_prompt(question, hits, history),
                system=SELECT_SYSTEM,
                max_tokens=_MAX_SELECT_TOKENS,
                # 이 프로젝트에서 LLM에게 시키는 일 중 유일하게 숙고가 필요한 판정이다.
                # 추론을 끄면 gemma-4-e2b가 "하나도 안 남김"을 아예 못 내놨다.
                reasoning=True,
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
            # 전자는 되물어야 하고 후자는 답하지 않아야 한다. 갈림길이 여기뿐이라
            # 여기서만 범위를 따로 묻는다 (일반 경로는 호출이 늘지 않는다).
            in_domain = await self._ask_domain(question, fallback=in_domain, history=history)
            if in_domain:
                logger.info("정보 부족 — 되묻기로 전환: %r", question[:50])
                return Selection(kept=[], coverage="needs_detail", note=NEEDS_DETAIL_NOTE)
            logger.info("범위 밖 판정 — 질문: %r", question[:50])
            return Selection(kept=[], coverage="none", note=NO_EVIDENCE_NOTE)

        kept = [hits[i] for i in indices]
        coverage: Coverage = "full" if len(kept) == len(hits) else "partial"
        logger.info("근거 선별: %d건 중 %d건 유지", len(hits), len(kept))
        return Selection(kept=kept, coverage=coverage)

    async def _ask_domain(
        self, question: str, *, fallback: bool, history: Sequence[Turn] = ()
    ) -> bool:
        """질문이 개 행동 영역인지만 따로 묻는다. 실패하면 선별이 준 값을 쓴다.

        폴백이 "범위 안"으로 기우는 게 맞다 — 개 질문을 범위 밖으로 잘못 처리하면
        보호자가 답을 못 받지만, 반대는 되묻기 한 번으로 끝난다.
        """
        if self._llm is None:
            return fallback
        try:
            raw = await self._llm.generate(
                f"{format_history(history)}{question}",
                system=DOMAIN_SYSTEM,
                max_tokens=_MAX_DOMAIN_TOKENS,
                reasoning=True,
            )
        except Exception as exc:  # noqa: BLE001 — 판정 실패가 답변을 막지 않는다
            logger.warning("범위 판정 실패, 선별 결과를 그대로 사용: %s", exc)
            return fallback

        verdict = parse_domain(raw)
        if verdict is None:
            logger.warning("범위 판정을 해석하지 못해 선별 결과를 사용: %r", raw[:60])
            return fallback
        logger.info("범위 판정: %s — %r", "개 행동" if verdict else "범위 밖", question[:40])
        return verdict

    @staticmethod
    def _build_prompt(question: str, hits: list[SearchHit], history: Sequence[Turn] = ()) -> str:
        blocks = [
            f"[{i}] {hit.document_title}\n{hit.content[:_SNIPPET_CHARS]}"
            for i, hit in enumerate(hits, start=1)
        ]
        context = format_history(history)
        hint = _CONTEXT_HINT if context else ""
        return "\n\n".join(blocks) + f"\n\n{context}{hint}Question: {question}"
