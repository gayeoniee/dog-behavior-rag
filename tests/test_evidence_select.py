"""근거 선별 단위 테스트.

핵심은 두 가지다:
  1. 근거가 없으면 **없다고 말하는가** (빈 배열이 유효한 판정인가)
  2. 선별이 실패해도 **답변을 막지 않는가** (폴백)
"""

from app.services.evidence_select import EvidenceSelector, Selection, parse_indices
from app.services.llm.base import LLMUnavailableError
from tests.fakes import hit


class StubLLM:
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


# ── parse_indices ────────────────────────────────────────────────────


def test_plain_array_is_parsed_to_zero_based():
    assert parse_indices("[1,3]", 5) == [0, 2]


def test_empty_array_is_a_valid_verdict_not_a_failure():
    """`[]`은 "근거 없음"이라는 판정이지 파싱 실패가 아니다. None과 구분돼야 한다."""
    assert parse_indices("[]", 5) == []


def test_unparseable_output_returns_none():
    assert parse_indices("어떤 자료도 관련이 없습니다", 5) is None
    assert parse_indices("", 5) is None


def test_reasoning_block_is_stripped():
    assert parse_indices("<think>1번은 관련있고...</think>\n[1]", 3) == [0]


def test_out_of_range_indices_are_dropped():
    assert parse_indices("[1, 9, 42]", 3) == [0]


def test_duplicates_are_collapsed():
    assert parse_indices("[2,2,1]", 3) == [0, 1]


# ── EvidenceSelector ─────────────────────────────────────────────────


async def test_empty_hits_short_circuits_without_calling_llm():
    llm = StubLLM("[1]")
    result = await EvidenceSelector(llm).select("질문", [])
    assert result.coverage == "none"
    assert result.kept == []
    assert llm.calls == 0, "부를 근거가 없는데 LLM을 부르면 안 된다"


async def test_no_relevant_evidence_yields_coverage_none():
    llm = StubLLM("[]")
    result = await EvidenceSelector(llm).select("애견카페 추천", [hit(1), hit(2)])
    assert result.coverage == "none"
    assert result.kept == []
    assert result.note and "근거 자료가 없습니다" in result.note


async def test_partial_selection_keeps_only_chosen():
    llm = StubLLM("[2]")
    hits = [hit(1), hit(2), hit(3)]
    result = await EvidenceSelector(llm).select("질문", hits)
    assert result.coverage == "partial"
    assert [h.chunk_id for h in result.kept] == [2]


async def test_all_kept_is_full_coverage():
    llm = StubLLM("[1,2]")
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
