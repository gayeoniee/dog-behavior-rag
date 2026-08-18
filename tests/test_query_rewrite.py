"""질의 재작성 단위 테스트. LLM 서버도 torch도 DB도 필요 없다."""

from app.services.llm.base import LLMUnavailableError
from app.services.query_rewrite import (
    REWRITE_SYSTEM,
    REWRITE_SYSTEM_KO,
    QueryRewriter,
    SearchQuery,
    clean_rewrite,
    looks_like_english,
)


class StubLLM:
    """지정한 문자열을 그대로 뱉거나, 지정한 예외를 던지는 가짜 LLM."""

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub"

    async def generate(self, prompt, *, system=None, max_tokens=None, reasoning=None) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply


# ── clean_rewrite: 작은 모델이 지시를 안 지킬 때의 방어 ──────────────


def test_plain_output_passes_through():
    assert clean_rewrite("alpha roll dominance down") == "alpha roll dominance down"


def test_surrounding_quotes_are_stripped():
    assert clean_rewrite('"alpha roll pinning"') == "alpha roll pinning"


def test_label_prefix_is_stripped():
    assert clean_rewrite("Rewritten query: alpha roll") == "alpha roll"
    assert clean_rewrite("검색어: leash correction") == "leash correction"


def test_code_fence_is_unwrapped():
    assert clean_rewrite("```\nalpha roll\n```") == "alpha roll"


def test_trailing_explanation_is_dropped():
    """설명을 뒤에 붙이는 모델이 흔하다. 첫 줄만 쓴다."""
    raw = "alpha roll, dominance down\n\n이 표현이 수의행동학 문헌에서 쓰입니다."
    assert clean_rewrite(raw) == "alpha roll, dominance down"


def test_reasoning_block_is_stripped():
    """추론형 모델(Nemotron reasoning 모드, Qwen3, R1 계열)이 사고과정을 먼저 뱉는다.

    안 걷어내면 "첫 줄만 쓴다" 규칙이 사고과정 첫 줄을 집어간다.
    """
    raw = "<think>\n사용자가 알파 롤을 묻고 있다. 영어 용어는...\n</think>\nalpha roll, pinning"
    assert clean_rewrite(raw) == "alpha roll, pinning"


def test_unclosed_reasoning_block_is_stripped():
    """여는 태그를 프롬프트에 넣어 닫는 태그만 오는 구현도 있다."""
    raw = "사용자 의도를 파악하면...\n</think>\nleash correction, positive punishment"
    assert clean_rewrite(raw) == "leash correction, positive punishment"


def test_empty_or_overlong_output_is_rejected():
    assert clean_rewrite("") == ""
    assert clean_rewrite("   \n  ") == ""
    assert clean_rewrite("x" * 500) == ""


# ── looks_like_english: 불필요한 LLM 호출 방지 ───────────────────────


def test_english_query_is_detected():
    assert looks_like_english("how do I stop my dog barking at the doorbell")


def test_korean_query_is_not_english():
    assert not looks_like_english("강아지가 초인종 소리에 계속 짖어요")


def test_mixed_query_is_not_english():
    assert not looks_like_english("alpha roll이 뭔가요 강아지한테 해도 되나요")


# ── QueryRewriter ────────────────────────────────────────────────────


async def test_english_query_skips_the_llm():
    llm = StubLLM("should not be called")
    query = "alpha roll and dominance down in dog training"
    assert await QueryRewriter(llm).rewrite(query) == SearchQuery.same(query)
    assert llm.calls == 0


async def test_disabled_rewriter_returns_original():
    llm = StubLLM("무시되어야 함")
    query = "강아지가 짖어요"
    assert await QueryRewriter(llm, enabled=False).rewrite(query) == SearchQuery.same(query)
    assert llm.calls == 0


async def test_missing_llm_returns_original():
    query = "강아지가 짖어요"
    assert await QueryRewriter(None).rewrite(query) == SearchQuery.same(query)


async def test_llm_unavailable_falls_back_to_original():
    """LM Studio가 꺼져 있어도 검색은 계속돼야 한다 — 재작성은 필수 경로가 아니다."""
    llm = StubLLM(error=LLMUnavailableError("서버 꺼짐"))
    query = "강아지가 짖어요"
    assert await QueryRewriter(llm).rewrite(query) == SearchQuery.same(query)


async def test_unexpected_exception_falls_back_to_original():
    llm = StubLLM(error=ValueError("예상 못 한 오류"))
    query = "강아지가 짖어요"
    assert await QueryRewriter(llm).rewrite(query) == SearchQuery.same(query)


async def test_unusable_output_falls_back_to_original():
    llm = StubLLM("   ")
    query = "강아지가 짖어요"
    assert await QueryRewriter(llm).rewrite(query) == SearchQuery.same(query)


async def test_single_language_mode_uses_the_old_prompt():
    """대조군은 라벨 없는 한 줄을 받아 양쪽에 같이 쓴다 (코퍼스가 영어뿐이던 시절)."""
    llm = StubLLM("alpha roll, dominance down")
    result = await QueryRewriter(llm, bilingual=False).rewrite("복종 자세를 강제로 유지")
    assert result.en == result.ko
    assert "alpha roll" in result.en



# ── 호출을 나눈 뒤의 동작 ────────────────────────────────────────────


class PerSystemLLM:
    """시스템 프롬프트에 따라 다른 답을 주는 가짜 LLM. 호출이 나뉜 걸 확인한다."""

    def __init__(self, en: str, ko: str) -> None:
        self.en, self.ko, self.systems = en, ko, []

    @property
    def name(self) -> str:
        return "per-system"

    async def generate(self, prompt, *, system=None, max_tokens=None, reasoning=None) -> str:
        self.systems.append(system)
        return self.ko if system is REWRITE_SYSTEM_KO else self.en


async def test_english_and_korean_come_from_separate_calls():
    """한 호출에 묶으면 영어가 나빠진다 — 커버리지 질문이 3/3에서 1/3로 떨어졌다."""
    llm = PerSystemLLM("alpha roll, pinning", "반려견을 제압하는 훈련의 위험성")
    result = await QueryRewriter(llm).rewrite("복종 자세를 강제로 유지시켜야 한다")

    assert llm.systems == [REWRITE_SYSTEM, REWRITE_SYSTEM_KO]
    assert "alpha roll" in result.en
    assert "복종 자세" in result.en  # 영어 쪽은 원문을 함께 남긴다
    assert result.ko == "반려견을 제압하는 훈련의 위험성"
    assert "복종 자세" not in result.ko  # 한국어 쪽은 안 붙인다 — 문체가 흐려진다


async def test_single_language_mode_makes_one_call():
    """대조군은 호출이 하나여야 한다. 두 번 부르면 비교가 공정하지 않다."""
    llm = PerSystemLLM("alpha roll, dominance down", "쓰이면 안 됨")
    result = await QueryRewriter(llm, bilingual=False).rewrite("복종 자세를 강제로 유지")

    assert llm.systems == [REWRITE_SYSTEM]
    assert result.en == result.ko
    assert "alpha roll" in result.en


async def test_korean_call_failing_does_not_break_english():
    """한쪽이 죽어도 다른 쪽은 살아야 한다."""

    class HalfBroken(PerSystemLLM):
        async def generate(self, prompt, *, system=None, max_tokens=None, reasoning=None) -> str:
            self.systems.append(system)
            if system is REWRITE_SYSTEM_KO:
                raise LLMUnavailableError("한국어 호출만 실패")
            return self.en

    llm = HalfBroken("alpha roll, pinning", "")
    query = "복종 자세를 강제로 유지시켜야 한다"
    result = await QueryRewriter(llm).rewrite(query)

    assert "alpha roll" in result.en
    assert result.ko == query  # 그쪽만 원문으로
