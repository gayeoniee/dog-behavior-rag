"""검색 전에 한국어 질문을 영어 기술표현으로 바꾼다.

**왜 필요한지 (실측):** bge-m3는 교차언어 모델이지만 *주제*는 넘나들어도
*기법 명칭*은 못 넘는다. 2026-08-12 코퍼스 282건 기준 측정:

    "복종 자세를 강제로 1~2분 유지시켜 서열을 알려줘야 한다"
      원문      → 0.552  Introduction to Desensitization... (무관)
      영어 재작성 → 0.724  AVSAB Position Statement on Dominance Theory ✔

    "짖을 때 목줄을 잡고 안 돼라고 단호하게 소리쳐야 한다"
      원문      → 0.580  The Influence of Breed, Sex... (무관)
      영어 재작성 → 0.629  AAHA Behavior Management Guidelines ✔

코퍼스에는 반박 근거가 다 있었다(`alpha roll` 6건, `pinning` 22건, `leash jerk`
3건). 찾지 못한 것뿐이다. 커버리지 질문 8개가 전부 통과했던 건 그것들이 전부
주제형("초인종 소리에 짖어요" → barking)이라 생긴 착시였다.

**실패해도 검색을 막지 않는다.** LLM 서버가 꺼져 있으면 원문으로 검색한다 —
재작성은 품질 향상이지 필수 경로가 아니다.
"""

import logging
import re
from collections.abc import Sequence

from app.schemas.chat import Turn
from app.services.llm.base import LLMClient, LLMUnavailableError

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = """You rewrite pet-behaviour questions into English search queries \
for a veterinary-literature database.

Rules:
- Output ONLY the rewritten query. No explanation, no quotes, no preamble.
- Translate Korean to English.
- **If earlier conversation is given, resolve the question against it first.** The user \
may be answering a question you asked. "1번이요" or "네 맞아요" means nothing on its own — \
work out what they are confirming and search for THAT. Example: if you asked "혼자 있을 \
때만 긁나요?" and the user says "1번이요", search for separation-related wall scratching.
- Replace colloquial descriptions with the technical terms used in veterinary \
behaviour literature. Examples:
    "복종 자세를 강제로 유지" -> alpha roll, dominance down, pinning the dog
    "목줄 잡고 혼내기" -> leash correction, leash jerk, positive punishment
    "간식 주며 익숙해지게" -> desensitization and counterconditioning
    "서열/우두머리" -> dominance hierarchy, pack theory
- Keep it under 30 words. Include both the plain description and the technical term.
- If the input is already English technical language, return it unchanged."""

_MAX_REWRITE_TOKENS = 80
_MAX_REWRITE_CHARS = 400

# 지시를 무시하고 설명을 덧붙이는 작은 모델 대비. 따옴표·라벨·코드펜스를 벗긴다.
_STRIP_PREFIX = re.compile(
    r"^\s*(rewritten query|query|answer|output|재작성|검색어)\s*[:：]\s*", re.IGNORECASE
)

# 추론형 모델(Nemotron 3 Nano의 reasoning 모드, Qwen3, DeepSeek-R1 계열)은
# 사고과정을 태그로 감싸 먼저 뱉는다. 안 걷어내면 "첫 줄"이 사고과정 첫 줄이 된다.
_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^.*?</(think|thinking|reasoning)>", re.DOTALL | re.IGNORECASE)


def clean_rewrite(raw: str) -> str:
    """모델 출력에서 실제 질의만 남긴다. 못 쓰겠으면 빈 문자열."""
    text = raw.strip()
    text = _THINK_BLOCK.sub("", text)
    # 여는 태그 없이 닫는 태그만 오는 구현도 있다 (여는 쪽을 프롬프트에 넣는 방식).
    if "</think" in text.lower() or "</reasoning" in text.lower():
        text = _UNCLOSED_THINK.sub("", text)
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text.strip("`")
        text = re.sub(r"^\w+\n", "", text.strip())
    # 여러 줄로 답하면 첫 줄만 쓴다 (설명을 뒤에 붙이는 경우가 많다).
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = _STRIP_PREFIX.sub("", text).strip().strip('"').strip("'").strip()
    return text if 0 < len(text) <= _MAX_REWRITE_CHARS else ""


def looks_like_english(text: str) -> bool:
    """이미 영어면 재작성을 건너뛴다 (LLM 호출 한 번 절약)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_ratio = sum(1 for c in letters if c.isascii()) / len(letters)
    return ascii_ratio > 0.9


def format_history(history: Sequence[Turn]) -> str:
    """대화를 프롬프트에 넣을 형태로. 비어 있으면 빈 문자열."""
    if not history:
        return ""
    lines = [f"{'보호자' if t.role == 'user' else '훈련사'}: {t.content}" for t in history]
    return "이전 대화:\n" + "\n".join(lines) + "\n\n"


class QueryRewriter:
    """검색용 질의 재작성기. LLM이 없거나 실패하면 원문을 그대로 돌려준다."""

    def __init__(self, llm: LLMClient | None, *, enabled: bool = True) -> None:
        self._llm = llm
        self._enabled = enabled

    async def rewrite(self, query: str, history: Sequence[Turn] = ()) -> str:
        # 대화 중이면 영어 질문이라도 재작성해야 한다 — "yes"나 "1" 같은 답은
        # 그 자체로는 검색이 안 되고 맥락에서 풀어야 한다.
        if not self._enabled or self._llm is None:
            return query
        if not history and looks_like_english(query):
            return query

        try:
            raw = await self._llm.generate(
                f"{format_history(history)}보호자: {query}",
                system=REWRITE_SYSTEM,
                max_tokens=_MAX_REWRITE_TOKENS,
                # 숙고가 필요한 일이 아니다 — "줄 당김 → leash pulling" 수준의
                # 용어 치환이다. 추론형 모델에서 켜두면 사고과정 250토큰을
                # 만드느라 6초가 걸린다 (전체 20초 중 3분의 1).
                reasoning=False,
            )
        except LLMUnavailableError as exc:
            # 재작성은 품질 향상이지 필수 경로가 아니다. 검색까지 막지 않는다.
            logger.warning("질의 재작성 실패, 원문으로 검색합니다: %s", exc)
            return query
        except Exception as exc:  # noqa: BLE001 — 어떤 이유로도 검색을 막지 않는다
            logger.warning("질의 재작성 중 예외, 원문으로 검색합니다: %s", exc)
            return query

        rewritten = clean_rewrite(raw)
        if not rewritten:
            logger.warning("재작성 결과를 쓸 수 없어 원문으로 검색합니다: %r", raw[:80])
            return query

        logger.info("질의 재작성: %r → %r", query, rewritten)
        # 원문을 함께 남긴다. 재작성이 핵심어를 빠뜨려도 원문 신호가 살아 있게.
        # 단 "1번이요" 같은 답변은 원문에 신호가 없으므로 재작성만 쓴다.
        return rewritten if len(query.strip()) < 8 else f"{rewritten}\n{query}"
