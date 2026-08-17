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

__all__ = ["Candidate", "rank", "rank_fused", "RRF_K"]

RRF_K = 60
"""RRF 상수. 등수를 점수로 바꿀 때 `1 / (RRF_K + 등수)`를 쓴다.

**이 값이 없으면 1등이 전부를 지배한다.** `1/1`과 `1/2`는 두 배 차이지만,
`1/61`과 `1/62`는 1.6% 차이다. 상수를 키울수록 여러 목록의 의견이 골고루
반영되고, 줄일수록 각 목록의 1등이 강해진다. 60은 원 논문(Cormack 2009)이
쓴 값이고 관행으로 굳었다.
"""


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


def rank(
    candidates: list[Candidate],
    top_k: int,
    *,
    authority_boost: float = 0.02,
    guide_boost: float = 0.03,
    max_per_document: int = 2,
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
    """
    return rank_fused(
        [candidates],
        top_k,
        authority_boost=authority_boost,
        guide_boost=guide_boost,
        max_per_document=max_per_document,
    )


def _boosted(cand: Candidate, authority_boost: float, guide_boost: float) -> float:
    return (
        (1.0 - cand.distance)
        + authority_boost * (3 - cand.authority_tier) / 2
        + (guide_boost if cand.doc_type == "guide" else 0.0)
    )


def rank_fused(
    pools: list[list[Candidate]],
    top_k: int,
    *,
    authority_boost: float = 0.02,
    guide_boost: float = 0.03,
    max_per_document: int = 2,
    rrf_k: int = RRF_K,
) -> list[SearchHit]:
    """여러 후보 목록을 합쳐 상위 top_k를 고른다. 점수 척도가 다를 때 쓴다.

    **점수를 그냥 비교할 수 없을 때 쓴다.** 한국어 문서 199건을 답변 코퍼스에
    넣었더니 청크의 1.7%가 상위 5위의 41%를 차지했다(2026-08-17 실측). 자료가
    좋아서가 아니라 **한→한 코사인이 한→영보다 구조적으로 높기** 때문이다.
    점수 척도가 다른 두 심사위원의 점수를 그냥 더한 셈이었다.

    RRF는 점수를 버리고 등수만 쓴다 — `1 / (rrf_k + 등수)`. 한국어 1등과 영어
    1등이 같은 값을 받으므로 채점 습관이 끼어들 자리가 없다.

    **여기서는 RRF가 "합의 보정"이 아니라 "공평한 교대"로 동작한다.** 청크 하나는
    언어가 하나뿐이라 두 목록에 동시에 나올 수 없고, 따라서 점수가 합산되는 일이
    없다. 결과는 두 목록을 번갈아 뽑는 것에 가깝다. 그게 여기서 원하는 것이다 —
    **영어 문서가 상위권 자리를 잃지 않는 것**이 목적이고, 그중 무엇을 쓸지는
    뒤의 근거 선별 LLM이 정한다.

    부스팅(권위·문서종류)은 **각 풀 안에서** 먼저 적용한다. 그래야 풀이 하나뿐일
    때 기존 `rank()`와 결과가 완전히 같다 — 어느 방식이든 순서를 보존한다.

    ## 왜 RRF가 기본이 아닌가 (2026-08-17 실측)

    **RRF를 먼저 붙였고 독식은 막았지만, 대신 균일한 세금이 됐다.** 상위 5칸 중
    한국어가 차지한 비율이 41% → 42%로 그대로였는데, 분포가 바뀌었다:

        RRF 전   질문마다 0/5 ~ 5/5   (이기는 쪽이 다 가져간다)
        RRF 후   모든 질문이 2/5      (기계적으로 번갈아 뽑는다)

    고양이 모래·중성화 비용처럼 **한국어 자료와 무관한 질문에도 2칸**이 갔다.
    청크 하나는 언어가 하나뿐이라 두 목록이 겹치지 않고, 그래서 RRF의 본래
    강점인 "여러 목록이 동의하면 올려준다"가 작동할 자리가 없다. 남는 건 교대뿐이다.
    평가에서 실무 평점이 3.7 → 2.5로 떨어진 게 그 대가다.

    ## 기각한 대안: 풀 안에서 z점수로 표준화

    "질문마다 적응하게 만들자"는 생각으로 각 풀의 `(점수 - 평균) / 표준편차`를
    써보려 했다. 무관한 주제라 후보가 다 고만고만하면 아무도 안 튀어서 자리를
    못 얻을 것이라 기대했는데, **정반대다.**

        확 튀는 풀   값 0.84, 0.44×4   → 표준편차 0.160 → 1등 z = 2.00
        밋밋한 풀   값 0.490 … 0.486   → 표준편차 0.0014 → 1등 z = 1.41

    **밋밋할수록 표준편차가 작아서, 그걸로 나누면 하찮은 차이가 부풀려진다.**
    "다 고만고만한데 그중 1등"이 오히려 강해진다 — 고치려던 문제를 키운다.
    단위 테스트를 짜서 계산해보고서야 알았다.

    질문마다 적응시키려면 **한 질문 안의 분포**가 아니라 **언어쌍의 고정 오프셋**을
    빼야 한다. 오프셋은 질문의 성질이 아니라 모델의 성질이라, 질문마다 다시
    추정하면 안 되는 값이다. 아직 안 해봤다.
    """
    ordered: list[Candidate] = []
    if len(pools) == 1:
        # 풀이 하나면 어떤 합치기든 순서를 보존하므로 계산을 건너뛴다.
        ordered = sorted(
            pools[0], key=lambda c: _boosted(c, authority_boost, guide_boost), reverse=True
        )
    else:
        fused: dict[int, float] = {}
        seen: dict[int, Candidate] = {}
        for pool in pools:
            ranked = sorted(
                pool, key=lambda c: _boosted(c, authority_boost, guide_boost), reverse=True
            )
            for position, cand in enumerate(ranked, start=1):
                fused[cand.chunk_id] = fused.get(cand.chunk_id, 0.0) + 1.0 / (rrf_k + position)
                seen.setdefault(cand.chunk_id, cand)
        ordered = [seen[cid] for cid in sorted(fused, key=lambda c: fused[c], reverse=True)]
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
