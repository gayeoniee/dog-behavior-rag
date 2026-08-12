"""doc_type 부스트 테스트.

부스트 크기가 핵심이다. 너무 작으면 논문에 밀리는 문제가 안 고쳐지고,
너무 크면 무관한 가이드가 정확한 논문을 밀어낸다.
"""

from app.services.vectorstore.ranking import Candidate, rank


def cand(chunk_id: int, distance: float, *, doc_type: str = "guide", doc: int | None = None):
    return Candidate(
        chunk_id=chunk_id,
        document_id=doc if doc is not None else chunk_id,
        document_title=f"{'가이드' if doc_type == 'guide' else '논문'} {chunk_id}",
        source=None,
        content=f"내용 {chunk_id}",
        distance=distance,
        authority_tier=1,
        doc_type=doc_type,
    )


def test_guide_wins_a_near_tie_against_a_study():
    """RSPCA 리콜 문서(3청크)가 논문에 밀려 안 나오던 실제 사례."""
    hits = rank(
        [cand(1, 0.30, doc_type="study"), cand(2, 0.32, doc_type="guide")],
        top_k=2,
        guide_boost=0.03,
    )
    assert hits[0].chunk_id == 2


def test_guide_cannot_override_a_clear_gap():
    """근거있음(0.714)과 주제공백(0.673)의 차이가 0.04라, 부스트가 그보다 작아야
    무관한 가이드가 정확한 논문을 밀어내지 않는다."""
    hits = rank(
        [cand(1, 0.25, doc_type="study"), cand(2, 0.32, doc_type="guide")],
        top_k=2,
        guide_boost=0.03,
    )
    assert hits[0].chunk_id == 1


def test_boost_is_off_when_zero():
    hits = rank(
        [cand(1, 0.30, doc_type="study"), cand(2, 0.32, doc_type="guide")],
        top_k=2,
        guide_boost=0.0,
    )
    assert hits[0].chunk_id == 1


def test_exposed_score_excludes_the_guide_boost():
    """부스트된 값은 1.0을 넘을 수 있어 "1.0에 가까울수록 유사" 계약을 깬다."""
    hits = rank([cand(1, 0.0, doc_type="guide")], top_k=1, guide_boost=0.03)
    assert hits[0].score == 1.0


def test_boosts_stack_with_authority():
    """tier1 가이드(+0.02+0.03)가 tier3 논문을 0.04 차이에서 뒤집는다."""
    weak_guide = cand(1, 0.34, doc_type="guide")
    strong_study = cand(2, 0.30, doc_type="study")
    strong_study.authority_tier = 3
    hits = rank([strong_study, weak_guide], top_k=2, authority_boost=0.02, guide_boost=0.03)
    assert hits[0].chunk_id == 1


def test_default_doc_type_is_guide():
    """기존 코드가 doc_type을 안 넘겨도 동작해야 한다."""
    c = Candidate(
        chunk_id=1,
        document_id=1,
        document_title="t",
        source=None,
        content="c",
        distance=0.1,
        authority_tier=1,
    )
    assert c.doc_type == "guide"
