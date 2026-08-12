"""팩트체크 파서·서비스 단위 테스트. LLM도 DB도 없이 돈다.

집중해서 보는 것: **근거 없는 단정을 막는가.** 팩트체크에서 가장 흔한 실패는
모델이 자기 지식으로 supported/contradicted를 고르는 것이다.
"""

from app.services.factcheck_service import (
    FactCheckService,
    parse_claims,
    parse_verdict,
)
from app.services.query_rewrite import QueryRewriter
from tests.fakes import FakeEmbedder, FakeStore, hit


class ScriptedLLM:
    """호출 순서대로 미리 정한 응답을 뱉는 가짜 LLM."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "scripted"

    async def generate(self, prompt, *, system=None, max_tokens=None) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "{}"


# ── parse_claims ─────────────────────────────────────────────────────


def test_json_array_is_parsed():
    assert parse_claims('["주장 하나", "주장 둘"]', fallback="원문") == ["주장 하나", "주장 둘"]


def test_code_fence_and_prose_are_tolerated():
    raw = '다음과 같습니다:\n```json\n["마운팅은 서열 때문이다"]\n```'
    assert parse_claims(raw, fallback="원문") == ["마운팅은 서열 때문이다"]


def test_reasoning_block_is_stripped():
    raw = '<think>어떤 주장이 있나...</think>\n["복종 자세를 시켜야 한다"]'
    assert parse_claims(raw, fallback="원문") == ["복종 자세를 시켜야 한다"]


def test_broken_json_falls_back_to_whole_text():
    """추출이 깨졌다고 검증을 못 하면 기능이 죽는다. 덜 정밀할 뿐 틀리진 않다."""
    assert parse_claims("주장이 뭐냐면요...", fallback="원문 전체") == ["원문 전체"]


def test_empty_array_falls_back():
    assert parse_claims("[]", fallback="원문 전체") == ["원문 전체"]


def test_claims_are_capped():
    raw = "[" + ",".join(f'"주장 {i}"' for i in range(20)) + "]"
    assert len(parse_claims(raw, fallback="원문")) == 5


# ── parse_verdict ────────────────────────────────────────────────────


def test_verdict_object_is_parsed():
    verdict, explanation = parse_verdict('{"verdict":"contradicted","explanation":"자료와 배치"}')
    assert verdict == "contradicted"
    assert explanation == "자료와 배치"


def test_unknown_verdict_becomes_not_covered():
    """모델이 임의의 라벨을 만들어내도 enum 밖으로 새지 않는다."""
    assert parse_verdict('{"verdict":"maybe","explanation":"애매"}')[0] == "not_covered"


def test_unparseable_verdict_becomes_not_covered():
    assert parse_verdict("잘 모르겠습니다")[0] == "not_covered"


# ── FactCheckService ─────────────────────────────────────────────────


def build_service(llm, hits):
    return FactCheckService(
        embedder=FakeEmbedder(),
        store=FakeStore(hits),
        llm=llm,
        rewriter=QueryRewriter(None, enabled=False),
    )


async def test_contradicted_claim_carries_sources():
    llm = ScriptedLLM(
        '["마운팅은 서열이 높아서 하는 행동이다"]',
        '{"verdict":"contradicted","explanation":"[자료 1]이 반박한다"}',
    )
    result = await build_service(llm, [hit(1, title="AVSAB 지배이론 성명서")]).check("...")

    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.verdict == "contradicted"
    assert claim.sources, "판정에 쓴 근거가 응답에 있어야 한다"
    assert claim.sources[0].document_title == "AVSAB 지배이론 성명서"


async def test_no_hits_forces_not_covered_without_calling_the_judge():
    """검색 결과가 0건이면 판정 LLM을 부르지 않는다 — 부를 근거가 없다."""
    llm = ScriptedLLM('["근거 없는 주장"]')
    result = await build_service(llm, []).check("...")

    assert result.claims[0].verdict == "not_covered"
    assert result.claims[0].sources == []
    # 추출 1회만 호출됐어야 한다.
    assert len(llm.prompts) == 1


async def test_corpus_bias_is_always_disclosed():
    """판정을 중립적인 제3자 검증인 것처럼 보이게 하면 안 된다."""
    llm = ScriptedLLM('["주장"]', '{"verdict":"supported","explanation":"ok"}')
    result = await build_service(llm, [hit(1)]).check("...")
    assert "보상 기반" in result.corpus_note


async def test_each_claim_is_judged_separately():
    llm = ScriptedLLM(
        '["주장 A", "주장 B"]',
        '{"verdict":"supported","explanation":"A"}',
        '{"verdict":"contradicted","explanation":"B"}',
    )
    result = await build_service(llm, [hit(1)]).check("...")
    assert [c.verdict for c in result.claims] == ["supported", "contradicted"]


async def test_evidence_is_numbered_in_the_judge_prompt():
    """설명이 "[자료 1]"로 인용하려면 프롬프트에 번호가 있어야 한다."""
    llm = ScriptedLLM('["주장"]', '{"verdict":"supported","explanation":"ok"}')
    await build_service(llm, [hit(1), hit(2)]).check("...")
    judge_prompt = llm.prompts[-1]
    assert "[자료 1]" in judge_prompt
    assert "[자료 2]" in judge_prompt


async def test_provider_and_latency_are_reported():
    llm = ScriptedLLM('["주장"]', '{"verdict":"supported","explanation":"ok"}')
    result = await build_service(llm, [hit(1)]).check("...")
    assert result.provider == "scripted"
    assert result.latency_ms >= 0
