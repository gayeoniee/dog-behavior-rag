"""ANN 후보를 최종 검색 결과로 재랭킹한다.

Postgres 없이 단위 테스트되도록 순수 함수로 분리했다. `pgvector.py`는 SQL로
후보를 넉넉히 뽑아오고, 순위 결정은 전부 여기서 한다.

**부스팅을 SQL ORDER BY에 넣지 않는 이유:** 조인 컬럼(`documents.authority_tier`)이
낀 표현식으로 정렬하면 HNSW 인덱스를 못 쓴다. 코퍼스가 커지는 순간 조용히 full
scan으로 바뀌고, 느려지기 전까지 아무도 모른다. ANN으로 과다 조회한 뒤 애플리케이션에서
재랭킹하는 게 표준 형태다.
"""

from collections import Counter
from dataclasses import dataclass

from .base import SearchHit

__all__ = ["Candidate", "rank"]

def _boosted(
    cand: "Candidate", authority_boost: float, guide_boost: float, background_weight: float = 1.0
) -> float:
    return (
        (1.0 - cand.distance)
        - cand.background * background_weight
        + authority_boost * (3 - cand.authority_tier) / 2
        + (guide_boost if cand.doc_type == "guide" else 0.0)
    )


@dataclass(slots=True)
class Candidate:
    """SQL이 뽑아온 후보 1건. 순위 결정에만 쓰이고 밖으로 나가지 않는다."""

    chunk_id: int
    document_id: int
    document_title: str
    source: str | None
    content: str
    distance: float
    """pgvector 코사인 거리. 0이면 동일, 클수록 멀다."""
    authority_tier: int
    doc_type: str = "guide"
    """`guide` | `study`. 실무 가이드에 소폭 부스트를 준다 (rank 독스트링 참조)."""
    background: float = 0.0
    """이 후보가 속한 **언어의 배경 유사도**. 점수에서 뺀다 (rank 독스트링 참조).

    코퍼스가 한 언어뿐이면 모든 후보가 같은 값을 빼므로 순위가 안 바뀐다.
    기본값 0.0은 "보정 없음"이라 기존 호출부가 그대로 동작한다.
    """


def rank(
    candidates: list[Candidate],
    top_k: int,
    *,
    authority_boost: float = 0.02,
    guide_boost: float = 0.03,
    max_per_document: int = 2,
    background_weight: float = 1.0,
) -> list[SearchHit]:
    """권위·문서종류 부스팅 + 문서 다양성 상한을 적용해 상위 top_k를 고른다.

    ```
    점수 = (1 - distance)
         + authority_boost * (3 - tier) / 2      # tier1 +0.02 … tier3 +0
         + guide_boost * (doc_type == "guide")   # 실무 가이드 +0.03
    ```

    **부스트가 작은 게 핵심이다.** 근거 있는 질문(0.714)과 주제 공백(0.673)의 차이가
    0.04라, 부스트가 그보다 크면 무관한 가이드가 정확한 논문을 밀어낸다. 권위도
    문서 종류도 타이브레이커지 검색 신호가 아니다.

    `guide_boost`가 필요한 이유: 코퍼스 청크의 97%가 논문이라 "어떻게 해요"라는
    질문에 실행 절차 대신 연구 결과가 올라온다. RSPCA 리콜 문서(3청크)가 존재하는데도
    논문 11,000청크에 밀려 상위에 못 오던 것이 실제 사례다.

    곱셈형(`sim * (1 + w)`)은 기각했다 — 유사도가 높을수록 부스트가 커지는데,
    거기가 바로 재정렬이 가장 불필요한 구간이다.

    ## `background` — 언어별 점수 오프셋을 뺀다

    한국어 문서 199건을 넣었더니 **청크의 1.7%가 상위 5위의 41%를 가져갔다**
    (2026-08-17). 한→한 코사인이 한→영보다 구조적으로 높아서다. 점수 척도가
    다른 두 심사위원의 점수를 그냥 비교한 셈이었다.

    보정은 **"자기 언어의 배경보다 얼마나 튀는가"**로 본다. 배경은 그 언어 청크
    전체와의 평균 유사도이고, 대부분의 청크는 어떤 질문과도 무관하므로 이 값이
    "그 언어가 기본으로 받는 점수"다. 실측 31% → 6%로 떨어졌다(코퍼스 비율 1.7%).

    **빼기만 하고 나누지 않는다.** 표준편차로 나누는 z점수도 해봤는데, 밋밋한
    풀일수록 표준편차가 작아 하찮은 차이가 부풀려졌다 — 확 튀는 풀의 1등이
    z=2.00일 때 밋밋한 풀의 1등이 z=1.41이다. 고치려던 문제를 키운다.

    **RRF(등수만 쓰기)도 기각했다.** 독식은 막았지만 청크 하나는 언어가 하나뿐이라
    두 목록이 겹치지 않고, 그래서 "여러 목록이 동의하면 올린다"는 RRF의 강점이
    작동할 자리가 없다. 남는 건 기계적 교대뿐이라 **모든 질문에 2/5씩 자리를
    떼어주는 균일한 세금**이 됐다 (고양이 모래 질문에도 2칸). 실무 평점이
    3.7 → 2.5로 떨어지고 통합 테스트 2개가 깨졌다.
    """
    ordered = sorted(
        candidates,
        key=lambda c: _boosted(c, authority_boost, guide_boost, background_weight),
        reverse=True,
    )

    picked: list[Candidate] = []
    spare: list[Candidate] = []
    used: Counter[int] = Counter()
    for cand in ordered:
        if len(picked) == top_k:
            break
        if used[cand.document_id] < max_per_document:
            picked.append(cand)
            used[cand.document_id] += 1
        else:
            spare.append(cand)

    # 상한 때문에 top_k를 못 채웠으면 밀려난 후보로 메운다.
    # 다양성을 지키자고 근거 개수가 줄어드는 건 그 자체로 손해다.
    if len(picked) < top_k:
        picked.extend(spare[: top_k - len(picked)])

    return [
        SearchHit(
            chunk_id=cand.chunk_id,
            document_title=cand.document_title,
            # 밖으로 나가는 score는 **부스트 전** 순수 코사인 유사도다. 부스트된 값은
            # 1.0을 넘을 수 있어 "1.0에 가까울수록 유사"라는 안드로이드 API 계약을 깬다.
            score=1.0 - cand.distance,
            content=cand.content,
            source=cand.source,
        )
        for cand in picked
    ]
