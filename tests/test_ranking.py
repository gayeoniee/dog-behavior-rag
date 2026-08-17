"""검색 재랭킹 단위 테스트. Postgres 없이 돈다.

현재 코퍼스는 전부 authority_tier=1이라 부스팅이 실제로는 no-op다. 그래서 합성
후보로 테스트한다 — 코퍼스가 섞이는 순간 동작해야 하는 로직이다.
"""

import pytest

from app.services.vectorstore.ranking import Candidate, rank


def make(chunk_id: int, distance: float, *, doc: int = 1, tier: int = 1) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id=doc,
        document_title=f"문서 {doc}",
        source=f"https://example.test/{doc}",
        content=f"청크 {chunk_id}",
        distance=distance,
        authority_tier=tier,
    )


def test_orders_by_distance_when_tiers_are_equal():
    hits = rank([make(1, 0.5), make(2, 0.1), make(3, 0.3)], top_k=3, max_per_document=3)
    assert [h.chunk_id for h in hits] == [2, 3, 1]


def test_authority_breaks_a_near_tie():
    """거리 차이 0.005는 부스트 상한 0.02 안이라 tier1이 이겨야 한다."""
    hits = rank(
        [make(1, 0.105, tier=3, doc=1), make(2, 0.110, tier=1, doc=2)],
        top_k=2,
        authority_boost=0.02,
    )
    assert hits[0].chunk_id == 2


def test_authority_cannot_override_a_clear_distance_gap():
    """거리 차이 0.05는 부스트 상한을 넘으므로 tier3이 그대로 1위여야 한다.

    이 비대칭이 설계 의도다 — 권위는 타이브레이커지 검색 신호가 아니다.
    """
    hits = rank(
        [make(1, 0.10, tier=3, doc=1), make(2, 0.15, tier=1, doc=2)],
        top_k=2,
        authority_boost=0.02,
    )
    assert hits[0].chunk_id == 1


def test_per_document_cap_limits_one_document():
    """AAHA 편중 시나리오 — 한 문서가 top_k를 독점하면 안 된다."""
    candidates = [make(i, 0.01 * i, doc=1) for i in range(10)]
    candidates += [make(100 + i, 0.5 + 0.01 * i, doc=2 + i) for i in range(5)]
    hits = rank(candidates, top_k=5, max_per_document=2)
    assert len(hits) == 5
    assert sum(1 for h in hits if h.document_title == "문서 1") == 2


def test_backfills_when_cap_leaves_room():
    """문서가 2개뿐이면 상한 때문에 4건만 나오는 게 아니라 5건을 채워야 한다."""
    candidates = [make(i, 0.01 * i, doc=1) for i in range(5)]
    candidates += [make(100 + i, 0.02 * i, doc=2) for i in range(5)]
    hits = rank(candidates, top_k=5, max_per_document=2)
    assert len(hits) == 5


def test_exposed_score_is_unboosted_cosine_similarity():
    """부스트된 값은 1.0을 넘을 수 있어 안드로이드 API 계약을 깬다."""
    hits = rank([make(1, 0.0, tier=1)], top_k=1, authority_boost=0.02)
    assert hits[0].score == 1.0


def test_returns_fewer_than_top_k_when_candidates_are_scarce():
    assert len(rank([make(1, 0.1)], top_k=5)) == 1


def test_empty_candidates_returns_empty():
    assert rank([], top_k=5) == []


def test_source_and_content_are_carried_through():
    """인용 링크는 SearchHit.source로 전달된다 — 여기서 끊기면 화면에 출처가 안 뜬다."""
    hit = rank([make(7, 0.2, doc=3)], top_k=1)[0]
    assert hit.source == "https://example.test/3"
    assert hit.content == "청크 7"
    assert hit.chunk_id == 7


class TestLanguageBackground:
    """언어별 배경 유사도를 빼서 점수 척도 차이를 없앤다.

    한국어 문서 199건을 답변 코퍼스에 넣었더니 청크의 1.7%가 상위 5위의 41%를
    가져갔다. 자료가 좋아서가 아니라 **한→한 코사인이 한→영보다 구조적으로
    높기** 때문이다.
    """

    def make(self, chunk_id: int, distance: float, background: float = 0.0):
        return Candidate(
            chunk_id=chunk_id,
            document_id=chunk_id,
            document_title=f"문서{chunk_id}",
            source=None,
            content=f"본문{chunk_id}",
            distance=distance,
            authority_tier=2,
            background=background,
        )

    def test_배경이_모두_같으면_순위가_안_바뀐다(self):
        """**가장 중요한 성질.** 코퍼스가 한 언어뿐이면 모두 같은 값을 빼므로
        보정 자체가 아무 일도 하지 않아야 한다 — 아니면 도입이 곧 회귀다."""
        plain = [self.make(i, 0.30 + i * 0.01) for i in range(1, 9)]
        shifted = [self.make(i, 0.30 + i * 0.01, background=0.42) for i in range(1, 9)]

        assert [h.chunk_id for h in rank(plain, top_k=5)] == [
            h.chunk_id for h in rank(shifted, top_k=5)
        ]

    def test_배경이_높은_언어는_그만큼_깎인다(self):
        """한국어 0.70(배경 0.46)과 영어 0.60(배경 0.38)이면 영어가 이겨야 한다.

        원점수만 보면 한국어가 0.10 앞서지만, 배경을 빼면 0.24 대 0.22로
        영어가 앞선다. **"자기 언어 안에서 얼마나 튀는가"가 기준이다.**
        """
        ko = self.make(1, 0.30, background=0.46)   # 0.70 - 0.46 = 0.24
        en = self.make(2, 0.40, background=0.38)   # 0.60 - 0.38 = 0.22

        assert [h.chunk_id for h in rank([ko, en], top_k=2)] == [1, 2]

        # 한국어가 배경만큼만 잘 나온 경우엔 영어가 앞선다
        ko_ordinary = self.make(1, 0.50, background=0.46)   # 0.50 - 0.46 = 0.04
        assert [h.chunk_id for h in rank([ko_ordinary, en], top_k=2)] == [2, 1]

    def test_밖으로_나가는_점수는_보정_전_유사도다(self):
        """보정값은 순위 결정에만 쓴다.

        밖으로 나가는 score는 "1.0에 가까울수록 유사"라는 API 계약을 지켜야 한다.
        배경을 뺀 값은 그 계약을 깬다 — 0.24가 나쁜 근거처럼 보이기 때문이다.
        """
        hits = rank([self.make(1, 0.30, background=0.46)], top_k=1)
        assert hits[0].score == pytest.approx(0.70)

