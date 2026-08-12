"""질의 재작성 단위 테스트. LLM 서버도 torch도 DB도 필요 없다."""

from app.services.llm.base import LLMUnavailableError
from app.services.query_rewrite import (
    QueryRewriter,
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

    async def generate(self, prompt, *, system=None, max_tokens=None) -> str:
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


async def test_korean_query_is_rewritten_and_keeps_original():
    """재작성이 핵심어를 빠뜨려도 원문 신호가 살아 있어야 한다."""
    llm = StubLLM("alpha roll, dominance down, pinning the dog")
    result = await QueryRewriter(llm).rewrite("복종 자세를 강제로 유지시켜야 한다")
    assert "alpha roll" in result
    assert "복종 자세" in result


async def test_english_query_skips_the_llm():
    llm = StubLLM("should not be called")
    query = "alpha roll and dominance down in dog training"
    assert await QueryRewriter(llm).rewrite(query) == query
    assert llm.calls == 0


async def test_disabled_rewriter_returns_original():
    llm = StubLLM("무시되어야 함")
    query = "강아지가 짖어요"
    assert await QueryRewriter(llm, enabled=False).rewrite(query) == query
    assert llm.calls == 0


async def test_missing_llm_returns_original():
    query = "강아지가 짖어요"
    assert await QueryRewriter(None).rewrite(query) == query


async def test_llm_unavailable_falls_back_to_original():
    """LM Studio가 꺼져 있어도 검색은 계속돼야 한다 — 재작성은 필수 경로가 아니다."""
    llm = StubLLM(error=LLMUnavailableError("서버 꺼짐"))
    query = "강아지가 짖어요"
    assert await QueryRewriter(llm).rewrite(query) == query


async def test_unexpected_exception_falls_back_to_original():
    llm = StubLLM(error=ValueError("예상 못 한 오류"))
    query = "강아지가 짖어요"
    assert await QueryRewriter(llm).rewrite(query) == query


async def test_unusable_output_falls_back_to_original():
    llm = StubLLM("   ")
    query = "강아지가 짖어요"
    assert await QueryRewriter(llm).rewrite(query) == query
