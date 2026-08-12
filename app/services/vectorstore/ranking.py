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
    scored = sorted(
        (
            (
                cand,
                (1.0 - cand.distance)
                + authority_boost * (3 - cand.authority_tier) / 2
                + (guide_boost if cand.doc_type == "guide" else 0.0),
            )
            for cand in candidates
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )

    picked: list[Candidate] = []
    spare: list[Candidate] = []
    used: Counter[int] = Counter()
    for cand, _ in scored:
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
