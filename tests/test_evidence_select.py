"""근거 선별 단위 테스트.

핵심은 두 가지다:
  1. 근거가 없으면 **없다고 말하는가** (빈 배열이 유효한 판정인가)
  2. 선별이 실패해도 **답변을 막지 않는가** (폴백)
"""

from app.schemas.chat import Turn
from app.services.evidence_select import (
    EvidenceSelector,
    Selection,
    parse_domain,
    parse_selection,
)
from app.services.llm.base import LLMUnavailableError
from tests.fakes import hit


class StubLLM:
    """reply 하나를 계속 돌려준다. domain_reply를 주면 범위 판정에만 그걸 쓴다."""

    def __init__(
        self,
        reply: str = "",
        error: Exception | None = None,
        domain_reply: str | None = None,
    ) -> None:
        self.reply = reply
        self.error = error
        self.domain_reply = domain_reply
        self.calls = 0
        self.reasoning_flags: list[bool | None] = []
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "stub"

    async def generate(self, prompt, *, system=None, max_tokens=None, reasoning=None) -> str:
        self.calls += 1
        self.reasoning_flags.append(reasoning)
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        if self.domain_reply is not None and system and "DOG or OTHER" in system:
            return self.domain_reply
        return self.reply


# ── parse_selection ──────────────────────────────────────────────────


def test_object_form_is_parsed():
    assert parse_selection('{"keep":[1,3],"in_domain":true}', 5) == ([0, 2], True)


def test_empty_keep_is_a_valid_verdict_not_a_failure():
    """빈 목록은 "쓸 근거가 없다"는 판정이지 파싱 실패가 아니다."""
    assert parse_selection('{"keep":[],"in_domain":true}', 5) == ([], True)


def test_out_of_domain_flag_is_carried():
    assert parse_selection('{"keep":[],"in_domain":false}', 5) == ([], False)


def test_bare_array_falls_back_to_in_domain_true():
    """작은 모델은 지시를 무시하고 배열만 뱉는다. 그때 범위 밖으로 몰면 안 된다."""
    assert parse_selection("[1,3]", 5) == ([0, 2], True)


def test_missing_in_domain_defaults_to_true():
    """개 질문을 범위 밖으로 잘못 처리하는 쪽이 그 반대보다 나쁘다."""
    assert parse_selection('{"keep":[2]}', 5) == ([1], True)


def test_unparseable_output_returns_none():
    assert parse_selection("어떤 자료도 관련이 없습니다", 5) is None
    assert parse_selection("", 5) is None


def test_reasoning_block_is_stripped():
    raw = '<think>1번은 관련있고...</think>\n{"keep":[1],"in_domain":true}'
    assert parse_selection(raw, 3) == ([0], True)


def test_out_of_range_indices_are_dropped():
    assert parse_selection('{"keep":[1,9,42],"in_domain":true}', 3) == ([0], True)


def test_duplicates_are_collapsed():
    assert parse_selection('{"keep":[2,2,1],"in_domain":true}', 3) == ([0, 1], True)


# ── parse_domain ─────────────────────────────────────────────────────


def test_domain_single_word_verdicts():
    assert parse_domain("DOG") is True
    assert parse_domain("OTHER") is False
    assert parse_domain(" other\n") is False


def test_domain_reasoning_block_is_stripped():
    assert parse_domain("<think>고양이 얘기니까 OTHER겠지 DOG는 아님</think>\nOTHER") is False


def test_domain_both_words_is_undecided():
    """설명을 덧붙이면 두 단어가 다 나온다. 그때는 판정하지 않는 편이 낫다."""
    assert parse_domain("This is not a DOG question, it is OTHER") is None


def test_domain_no_verdict_returns_none():
    assert parse_domain("잘 모르겠습니다") is None
    assert parse_domain("") is None


# ── EvidenceSelector ─────────────────────────────────────────────────


async def test_empty_hits_short_circuits_without_calling_llm():
    llm = StubLLM("[1]")
    result = await EvidenceSelector(llm).select("질문", [])
    assert result.coverage == "none"
    assert result.kept == []
    assert llm.calls == 0, "부를 근거가 없는데 LLM을 부르면 안 된다"


async def test_out_of_domain_with_no_evidence_yields_none():
    """범위 밖 질문은 답하지 않는다."""
    llm = StubLLM('{"keep":[],"in_domain":false}', domain_reply="OTHER")
    result = await EvidenceSelector(llm).select("애견카페 추천", [hit(1), hit(2)])
    assert result.coverage == "none"
    assert result.kept == []
    assert result.note and "근거 자료가 없습니다" in result.note


async def test_domain_call_overrides_selector_hint():
    """선별이 범위를 잘못 봐도 전용 판정이 바로잡는다.

    작은 모델은 "근거 고르기"와 "개 질문인지"를 한 번에 시키면 후자를 놓친다.
    실제로 고양이 모래·중성화 비용을 in_domain=true로 판정해 거절이 안 됐다.
    """
    llm = StubLLM('{"keep":[],"in_domain":true}', domain_reply="OTHER")
    result = await EvidenceSelector(llm).select("고양이 모래 뭐가 좋아요", [hit(1)])
    assert result.coverage == "none"


async def test_domain_call_only_happens_when_nothing_kept():
    """답할 근거가 있으면 범위 밖인지 물을 이유가 없다 — 호출을 늘리지 않는다."""
    llm = StubLLM('{"keep":[1],"in_domain":true}')
    await EvidenceSelector(llm).select("질문", [hit(1), hit(2)])
    assert llm.calls == 1


async def test_undecidable_domain_falls_back_to_selector_hint():
    """판정을 못 읽으면 개 질문 쪽으로 기운다 — 되묻기가 거절보다 회복 가능하다."""
    llm = StubLLM('{"keep":[],"in_domain":true}', domain_reply="잘 모르겠습니다")
    result = await EvidenceSelector(llm).select("벽을 긁어요", [hit(1)])
    assert result.coverage == "needs_detail"


async def test_history_reaches_the_selection_prompt():
    """맥락이 없으면 후속 질문("켄넬 훈련이 도움이 될까?")을 판정할 수 없다.

    검색(재작성)은 이미 맥락을 쓰고 있었는데 판정만 안 써서, 같은 취지의 질문이
    한 번은 답이 되고 한 번은 되묻기가 됐다.
    """
    llm = StubLLM('{"keep":[1],"in_domain":true}')
    history = [Turn(role="user", content="밤에 너무 짖는데?")]
    await EvidenceSelector(llm).select("켄넬 훈련이 도움이 될까?", [hit(1)], history)
    assert "밤에 너무 짖는데" in llm.prompts[0]


async def test_selection_asks_for_reasoning():
    """근거 선별은 이 프로젝트에서 유일하게 숙고가 필요한 판정이다."""
    llm = StubLLM('{"keep":[1],"in_domain":true}')
    await EvidenceSelector(llm).select("질문", [hit(1)])
    assert llm.reasoning_flags == [True]


async def test_in_domain_with_no_evidence_asks_for_detail():
    """개 질문인데 근거를 못 고르면 거절이 아니라 되묻기다.

    실제 사례: "강아지가 벽을 자꾸 긁어"는 분리불안·지루함·강박 중 무엇인지에
    따라 대응이 달라 그대로는 답할 수 없다. "혼자 있을 때"를 붙이면 바로 답이 나온다.
    """
    llm = StubLLM('{"keep":[],"in_domain":true}', domain_reply="DOG")
    result = await EvidenceSelector(llm).select("벽을 자꾸 긁어요", [hit(1), hit(2)])
    assert result.coverage == "needs_detail"
    assert result.kept == []
    assert result.note and "되묻는" in result.note


async def test_partial_selection_keeps_only_chosen():
    llm = StubLLM('{"keep":[2],"in_domain":true}')
    hits = [hit(1), hit(2), hit(3)]
    result = await EvidenceSelector(llm).select("질문", hits)
    assert result.coverage == "partial"
    assert [h.chunk_id for h in result.kept] == [2]


async def test_all_kept_is_full_coverage():
    llm = StubLLM('{"keep":[1,2],"in_domain":true}')
    result = await EvidenceSelector(llm).select("질문", [hit(1), hit(2)])
    assert result.coverage == "full"
    assert len(result.kept) == 2


async def test_llm_unavailable_falls_back_to_all_hits():
    """선별 실패가 답변을 막으면 안 된다."""
    llm = StubLLM(error=LLMUnavailableError("서버 꺼짐"))
    hits = [hit(1), hit(2)]
    result = await EvidenceSelector(llm).select("질문", hits)
    assert result.kept == hits
    assert result.coverage == "partial"


async def test_unexpected_exception_falls_back_to_all_hits():
    llm = StubLLM(error=ValueError("예상 못 한 오류"))
    hits = [hit(1)]
    assert (await EvidenceSelector(llm).select("질문", hits)).kept == hits


async def test_unparseable_reply_falls_back_to_all_hits():
    """해석 실패와 "근거 없음"을 혼동하면 멀쩡한 답변이 사라진다."""
    llm = StubLLM("모두 관련 있어 보입니다")
    hits = [hit(1), hit(2)]
    result = await EvidenceSelector(llm).select("질문", hits)
    assert result.kept == hits
    assert result.coverage == "partial"


async def test_disabled_selector_passes_everything_through():
    llm = StubLLM("[]")
    hits = [hit(1), hit(2)]
    result = await EvidenceSelector(llm, enabled=False).select("질문", hits)
    assert result.kept == hits
    assert result.coverage == "full"
    assert llm.calls == 0


async def test_missing_llm_passes_everything_through():
    hits = [hit(1)]
    assert (await EvidenceSelector(None).select("질문", hits)).kept == hits


async def test_default_selection_is_full_coverage():
    assert Selection().coverage == "full"
