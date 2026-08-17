"""검색 재랭킹 단위 테스트. Postgres 없이 돈다.

현재 코퍼스는 전부 authority_tier=1이라 부스팅이 실제로는 no-op다. 그래서 합성
후보로 테스트한다 — 코퍼스가 섞이는 순간 동작해야 하는 로직이다.
"""

from app.services.vectorstore.ranking import Candidate, rank, rank_fused


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


class TestLanguageFusion:
    """언어별 목록을 합친다 — 점수 척도가 달라서 그냥 비교할 수 없다.

    한국어 문서 199건을 답변 코퍼스에 넣었더니 청크의 1.7%가 상위 5위의 41%를
    가져갔다. 자료가 좋아서가 아니라 **한→한 코사인이 한→영보다 구조적으로
    높기** 때문이다.
    """

    def make(self, chunk_id: int, distance: float, doc_id: int | None = None):
        return Candidate(
            chunk_id=chunk_id,
            document_id=doc_id if doc_id is not None else chunk_id,
            document_title=f"문서{chunk_id}",
            source=None,
            content=f"본문{chunk_id}",
            distance=distance,
            authority_tier=2,
        )

    def pool(self, start: int, base: float, step: float = 0.02, n: int = 5):
        """풀 하나. **풀 안에서 점수가 흩어져 있어야 실제와 같다** —
        전부 동점이면 표준편차가 0이라 z점수가 무의미해진다."""
        return [self.make(start + i, base + i * step) for i in range(n)]

    def test_풀이_하나면_기존_rank와_결과가_같다(self):
        """**가장 중요한 성질.** 언어가 하나뿐인 코퍼스에서 결과가 바뀌면
        합치기를 도입한 것 자체가 회귀다."""
        cands = self.pool(1, 0.30, n=8)
        assert rank_fused([cands], top_k=5) == rank(cands, top_k=5)

    def test_점수가_낮은_목록도_상위_자리를_얻는다(self):
        """한국어가 전부 0.70, 영어가 전부 0.50이어도 영어가 전멸하면 안 된다.

        점수를 그냥 합치면 영어가 통째로 밀려난다 — 그게 실제로 겪은 일이다.
        """
        ko = self.pool(1, 0.30)    # 유사도 0.70 근처
        en = self.pool(11, 0.50)   # 유사도 0.50 근처

        ids = [h.chunk_id for h in rank_fused([en, ko], top_k=4)]
        assert any(i > 10 for i in ids), "점수 낮은 목록이 전멸했다"
        assert any(i <= 5 for i in ids), "점수 높은 목록이 사라졌다"

    def test_밋밋한_풀도_상위_자리를_가져간다(self):
        """**RRF의 알려진 약점을 못으로 박아둔다 — 고쳤다고 착각하지 않도록.**

        한 풀은 1등이 확 튀고(진짜 관련), 다른 풀은 전부 고만고만하다(무관한 주제).
        RRF는 등수만 보므로 **고만고만한 풀의 1등도 1등 대접**을 받는다.
        고양이·중성화 질문에까지 한국어가 2/5씩 들어간 원인이 이것이다.

        z점수 표준화로 고치려 했으나 실패했다 — 밋밋할수록 표준편차가 작아서
        나누면 오히려 부풀려진다 (rank_fused 독스트링의 계산 참조).
        """
        sharp = [self.make(1, 0.20)] + [self.make(1 + i, 0.60) for i in range(1, 5)]
        flat = [self.make(10 + i, 0.55 + i * 0.001) for i in range(5)]

        ids = [h.chunk_id for h in rank_fused([sharp, flat], top_k=3)]
        assert ids[0] == 1, "확 튀는 후보가 1위여야 한다"
        assert any(i >= 10 for i in ids), (
            "밋밋한 풀도 자리를 얻는다 — 이게 RRF의 대가다. "
            "언젠가 고정 오프셋 보정으로 고치면 이 단언이 깨질 것이고, 그때가 성공이다"
        )

    def test_문서_다양성_상한은_합친_뒤에도_적용된다(self):
        ko = [self.make(i, 0.30 + i * 0.01, doc_id=99) for i in range(1, 6)]
        en = [self.make(10 + i, 0.50 + i * 0.01, doc_id=88) for i in range(1, 6)]

        hits = rank_fused([ko, en], top_k=4, max_per_document=2)
        assert len(hits) == 4
        assert len({h.chunk_id for h in hits}) == 4

