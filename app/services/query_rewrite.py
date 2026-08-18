"""검색 전에 질문을 **언어마다 하나씩** 검색 질의로 바꾼다.

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

**2026-08-18: 영어 한 줄로는 부족해졌다.** 코퍼스 622건 중 328건이 한국어인데
위 프롬프트는 영어뿐이던 시절 것이다. 영어 재작성이 한 줄로 앞에 붙으면 벡터가
영어 쪽으로 끌려가 **한국어 문서가 경쟁 자체를 못 한다.** 그렇다고 원문만 쓰면
한국어 풀 안에서도 못 찾는다 — 질문은 구어체고 문서는 문어체라서:

    "강아지가 자기 똥을 먹어요…"                   → 식분증 문서 16위
    "반려견이 자신의 배변을 먹는 행동의 원인과 교정 방법"  → 1위 0.734

그래서 `SearchQuery`로 갈라 들고 다니고, 각 언어를 그 언어의 문체로 찾는다.

**실패해도 검색을 막지 않는다.** LLM 서버가 꺼져 있으면 원문으로 검색한다 —
재작성은 품질 향상이지 필수 경로가 아니다. 한 줄만 받아도 그 줄은 살린다.
"""

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.chat import Turn
from app.services.embeddings.base import Embedder
from app.services.llm.base import LLMClient, LLMUnavailableError

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = """You rewrite pet-behaviour questions into English search \
queries for a veterinary-literature database.

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
"""영어 논문·기관 가이드를 찾기 위한 질의. **한 글자도 안 바꿨다.**

한국어 질의를 여기 끼워 넣었다가 영어 쪽이 나빠졌다 (2026-08-18 실측):

    한 호출로 둘 다   "짖는 버릇을 어떻게 고치나요?" 커버리지 통과 1/3
    호출을 나눔       3/3 (대조군과 동일)

모델이 "using behavioral modification techniques" 같은 **범주어**를 쓰기 시작했다 —
검색에 아무것도 안 걸리는 말이다. **한 호출에 한 가지만 시킨다**는 규칙을 어긴
대가였고, `evidence_select`의 범위 판정에서 겪은 것과 같은 고장이다.
"""

REWRITE_SYSTEM_KO = """You restate a Korean pet question in written Korean, so that it \
matches a corpus of dog-training documents written in formal written Korean.

**This is a register change, not a summary and not a reframing.** Keep every content \
word. Keep what the question is actually asking — a cause, a method, whether something \
is true, whether something is a good idea. Do not turn it into a different question.

Change ONLY these:
- 강아지 / 우리 애 / 얘 / 우리 개 -> 반려견
- spoken words -> the written ones the documents use:
    똥 -> 배변,  오줌 -> 소변,  밥 -> 사료,  산책줄 -> 리드줄
- the spoken ending -> a written one (a noun phrase or a plain declarative)
- filler only: 좀, 진짜, 너무너무, ㅠㅠ, 어떡해요

Keep untouched: the specific object, place, time, or situation (엘리베이터, 종이컵, \
산책 중, 6개월), and every fact the owner reported.

Examples:
    "강아지가 자기 똥을 먹어요 왜 그러는 거예요"
      -> 반려견이 자신의 배변을 먹는 이유
    "강아지가 다른 개들과 어울리는 것이 왜 좋지 않나요?"
      -> 반려견이 다른 개들과 어울리는 것이 좋지 않은 이유
    "보상 기반 훈련을 하면 강아지가 제멋대로 행동하게 되나요?"
      -> 보상 기반 훈련이 반려견을 제멋대로 행동하게 만드는지 여부
    "6개월 강아지인데 배변을 자꾸 아무데나 해요 어떻게 가르쳐요"
      -> 생후 6개월 반려견이 아무 곳에나 배변할 때의 교육 방법
    "반려견의 분리불안 교육 방법"
      -> 반려견의 분리불안 교육 방법

That last one is already written Korean. **When the question is already in written \
Korean, return it unchanged.**

Never add words the question did not ask for. Do not append "원인과 교정 방법" or any \
other stock phrase.

If earlier conversation is given, resolve the question against it first — "1번이요" or \
"네 맞아요" means nothing on its own.

Output the restated question only. No label, no explanation, no quotes."""
"""한국어 자막을 찾기 위한 질의. **영어 재작성과 따로 부른다.**

한 호출에 묶으면 영어 쪽이 무너진다 (`REWRITE_SYSTEM` 독스트링 참조). 두 호출은
서로 독립이라 `asyncio.gather`로 동시에 보내므로 **지연은 거의 안 는다.**

**1차는 고정 틀이었다 — `반려견이 <행동>하는 행동의 원인과 교정 방법`.** 32문항은
좋아졌는데 581문항이 hit@5 48.7% → 42.5%로 무너졌다. 틀이 밋밋해서가 아니라
**"문제행동 → 교정"이라는 프레임을 강요**해서였다:

    "다른 개들과 어울리는 것이 왜 좋지 않나요?"
      → "…어울리는 행동의 원인과 교정 방법"     ← 다른 질문이 됐다
    "보상 기반 훈련을 하면 제멋대로 행동하나요?"
      → "…제멋대로 행동하는 행동의 원인과 교정 방법"  ← 완전히 다른 질문

그래서 지금은 **문체만 바꾸고 내용어를 전부 보존한다.** 손으로 확인한 결과
한국어 풀 안 순위는 고정 틀과 같으면서(똥 16위 → 1위) 뜻은 안 비튼다.

**"이미 문어체면 그대로 돌려준다"가 581문항을 지키는 장치다.**
"""

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


def _unwrap(raw: str) -> str:
    """사고과정 태그와 코드펜스를 벗긴다. 라벨 파싱 전에 공통으로 거친다."""
    text = _THINK_BLOCK.sub("", raw.strip())
    # 여는 태그 없이 닫는 태그만 오는 구현도 있다 (여는 쪽을 프롬프트에 넣는 방식).
    if "</think" in text.lower() or "</reasoning" in text.lower():
        text = _UNCLOSED_THINK.sub("", text)
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text.strip("`")
        text = re.sub(r"^\w+\n", "", text.strip())
    return text.strip()


def _tidy(line: str) -> str:
    """한 줄에서 라벨·따옴표를 벗기고 길이를 검사한다. 못 쓰겠으면 빈 문자열."""
    text = _STRIP_PREFIX.sub("", line).strip().strip('"').strip("'").strip()
    return text if 0 < len(text) <= _MAX_REWRITE_CHARS else ""


def clean_rewrite(raw: str) -> str:
    """모델 출력에서 실제 질의 한 줄만 남긴다. 못 쓰겠으면 빈 문자열."""
    text = _unwrap(raw)
    # 여러 줄로 답하면 첫 줄만 쓴다 (설명을 뒤에 붙이는 경우가 많다).
    return _tidy(text.splitlines()[0]) if text else ""


@dataclass(frozen=True)
class SearchQuery:
    """언어별 검색 질의. **한 벡터에 두 언어를 섞지 않기 위해** 갈라서 들고 다닌다.

    섞으면 어느 쪽도 제대로 못 친다 (2026-08-18 실측):

        'Canine coprophagia causes?\\n강아지가 자기 똥을 먹어요 왜 그러는 거예요'
        → 상위 5개 전부 영어, 전부 다른 주제 (파괴적 씹기·분리불안)

    영어 재작성이 한 줄로 앞에 붙으면 벡터가 영어 쪽으로 끌려가 한국어 문서가
    경쟁 자체를 못 한다. 그렇다고 원문만 쓰면 한국어 풀 안에서도 못 찾는다 —
    질문은 구어체, 문서는 문어체라서 식분증 문서가 16위였다. **각 언어에 그
    언어의 문체로 물어야 한다** (문서체로 다시 쓰니 1위 0.734).
    """

    en: str
    ko: str

    def by_language(self) -> dict[str, str]:
        return {"en": self.en, "ko": self.ko}

    @classmethod
    def same(cls, query: str) -> "SearchQuery":
        """양쪽 다 같은 질의. 재작성이 꺼져 있거나 실패했을 때의 폴백."""
        return cls(en=query, ko=query)


async def embed_by_language(embedder: Embedder, query: "SearchQuery") -> dict[str, list[float]]:
    """언어별 질의를 임베딩한다. **같은 문자열이면 한 번만 부른다.**

    재작성이 꺼졌거나 실패하면 EN·KO가 같은 원문이다. 그때 두 번 부르면 요청마다
    임베딩 왕복이 공짜로 하나 늘어난다.
    """
    cache: dict[str, list[float]] = {}
    vectors: dict[str, list[float]] = {}
    for language, text in query.by_language().items():
        if text not in cache:
            cache[text] = await embedder.embed_query(text)
        vectors[language] = cache[text]
    return vectors


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

    def __init__(
        self, llm: LLMClient | None, *, enabled: bool = True, bilingual: bool = True
    ) -> None:
        self._llm = llm
        self._enabled = enabled
        # 끄면 코퍼스가 영어뿐이던 시절 그대로 동작한다 — A/B의 대조군.
        self._bilingual = bilingual

    async def rewrite(self, query: str, history: Sequence[Turn] = ()) -> SearchQuery:
        """언어별 검색 질의를 만든다. 실패하면 그쪽만 원문이다.

        **영어 쪽은 오늘과 한 글자도 다르지 않다** — 같은 프롬프트, 같은 토큰 예산,
        같은 폴백이다. 한국어는 **별도 호출**로 덧붙는다. 그래서 평가가 움직이면
        원인이 한국어 벡터 하나뿐이다.

        처음엔 한 호출로 두 줄을 받으려 했는데 **영어 쪽이 나빠졌다** — 모델이
        "using behavioral modification techniques" 같은 범주어를 쓰기 시작했고
        커버리지 질문 하나가 3/3 → 1/3이 됐다. "한 호출에 한 가지만 시킨다"를
        어긴 대가였다. 나눈 뒤 3/3으로 돌아왔다.

        두 호출은 서로 독립이라 **동시에 보낸다.** LLM만 묶는다 — DB는 커넥션
        하나라 병렬로 쓰면 깨진다 (`factcheck_service`와 같은 이유).
        """
        # 대화 중이면 영어 질문이라도 재작성해야 한다 — "yes"나 "1" 같은 답은
        # 그 자체로는 검색이 안 되고 맥락에서 풀어야 한다.
        if not self._enabled or self._llm is None:
            return SearchQuery.same(query)
        if not history and looks_like_english(query):
            return SearchQuery.same(query)

        prompt = f"{format_history(history)}보호자: {query}"
        jobs = [self._call(prompt, REWRITE_SYSTEM)]
        if self._bilingual:
            jobs.append(self._call(prompt, REWRITE_SYSTEM_KO))
        raws = await asyncio.gather(*jobs)

        english = clean_rewrite(raws[0])
        if not english:
            logger.warning("재작성 결과를 쓸 수 없어 원문으로 검색합니다: %r", raws[0][:80])
            english = query
        else:
            # 원문을 함께 남긴다. 재작성이 핵심어를 빠뜨려도 원문 신호가 살아 있게.
            # 단 "1번이요" 같은 답변은 원문에 신호가 없으므로 재작성만 쓴다.
            english = english if len(query.strip()) < 8 else f"{english}\n{query}"

        if not self._bilingual:
            return SearchQuery.same(english)

        korean = clean_rewrite(raws[1])
        if not korean:
            # 조용히 넘어가면 기능이 켜져 있는데 아무 일도 안 하는 상태가 된다.
            logger.warning("한국어 재작성이 비어 원문으로 검색합니다: %r", raws[1][:80])
            korean = query
        # **한국어 쪽에는 원문을 안 붙인다** — 구어를 섞으면 문체가 도로 흐려져
        # 식분증 문서가 1위에서 밀린다. 그게 이 변경의 요점이다.
        logger.info("질의 재작성: %r → EN=%r KO=%r", query, english.splitlines()[0], korean)
        return SearchQuery(en=english, ko=korean)

    async def _call(self, prompt: str, system: str) -> str:
        """재작성 한 번. **실패해도 예외를 올리지 않는다** — 검색을 막지 않는다."""
        try:
            return await self._llm.generate(  # type: ignore[union-attr]  # 호출 전에 확인함
                prompt,
                system=system,
                max_tokens=_MAX_REWRITE_TOKENS,
                # 숙고가 필요한 일이 아니다 — "줄 당김 → leash pulling" 수준의
                # 용어 치환이다. 추론형 모델에서 켜두면 사고과정 250토큰을
                # 만드느라 6초가 걸린다 (전체 20초 중 3분의 1).
                reasoning=False,
            )
        except LLMUnavailableError as exc:
            # 재작성은 품질 향상이지 필수 경로가 아니다. 검색까지 막지 않는다.
            logger.warning("질의 재작성 실패, 원문으로 검색합니다: %s", exc)
        except Exception as exc:  # noqa: BLE001 — 어떤 이유로도 검색을 막지 않는다
            logger.warning("질의 재작성 중 예외, 원문으로 검색합니다: %s", exc)
        return ""
