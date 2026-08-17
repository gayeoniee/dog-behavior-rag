"""논문 청크 중 답변 근거가 될 수 없는 것이 얼마나 되나.

    uv run python -m scripts.eda.paper_chunks
    uv run python -m scripts.eda.paper_chunks --samples 3

**왜 세는가.** 코퍼스 글자 수의 95%가 PMC 논문인데, 05장 실측에서 논문 청크로
만든 보호자 질문이 **0/10** 통과였다(기관 문서는 14/15). 대부분이 방법론·통계·
표 조각이라 "우리 개가 왜 그럴까요"의 근거가 못 된다.

**정제(LLM으로 다시 쓰기)는 위험하다.** 유튜브 자막은 원본이 이미 조언이라
다듬기만 하면 됐지만, 논문은 연구 결과다. "코르티솔이 유의하게 낮았다(p=0.03)"를
보호자용으로 바꾸라고 하면 **논문이 하지 않은 조언을 지어낸다.**

그래서 먼저 **거르기**를 검토한다. 규칙으로 판별되고, LLM을 안 쓰며, 되돌릴 수
있다. 이 스크립트는 "얼마나 걸러지고 얼마나 남는가"만 센다 — **거르는 건
숫자를 보고 정한다.**

각 분류의 표본을 같이 찍는다. 규칙이 멀쩡한 청크를 잡고 있으면 그건 눈으로만
보인다 (`mentions` 오탐으로 이미 한 번 겪었다).
"""

import argparse
import asyncio
import re
import sys
from collections import Counter

from sqlalchemy import select

import scripts.collect  # noqa: F401 — 콘솔 UTF-8
from app.core.config import get_settings
from app.db.models import Chunk, Document
from app.db.session import create_engine, create_session_factory
from app.services.chunking import looks_like_reference_list

_STATS = re.compile(
    r"\bp\s*[=<>]\s*0?\.\d|\bCI\b|\bSD\s*=|\bOR\s*=|\bn\s*=\s*\d|"
    r"\bχ2|\bR2\b|\bp-value|significan(t|ce)\b|\bmean\s*±",
    re.I,
)
"""통계 보고 표현. 하나만 있으면 본문에도 나올 수 있어 개수로 판정한다."""

_METHODS = re.compile(
    r"\b(participants were|dogs were (recruited|enrolled|assigned)|"
    r"data (were|was) (collected|analy[sz]ed)|questionnaire was|"
    r"ethical approval|informed consent|inclusion criteria|"
    r"statistical analys|were performed using|IBM SPSS|R version)\b",
    re.I,
)

_FRONTMATTER = re.compile(
    r"\b(Conflict of Interest|Author Contributions|Funding|Acknowledg(e)?ments?|"
    r"Data Availability|Supplementary Material|Institutional Review Board|"
    r"Informed Consent Statement|Publisher's Note|©|doi\.org/10\.)",
    re.I,
)

_TABLE = re.compile(r"\b(Table|Figure|Fig\.)\s*\d", re.I)


def classify(text: str) -> str:
    """청크를 한 갈래로 분류한다. 순서가 곧 우선순위다."""
    if looks_like_reference_list(text):
        return "참고문헌"
    if _FRONTMATTER.search(text):
        return "저자·기금·부록"
    if len(_STATS.findall(text)) >= 3:
        return "통계 보고"
    if _METHODS.search(text):
        return "연구 방법"
    if len(_TABLE.findall(text)) >= 2:
        return "표·그림 조각"
    return "본문 (쓸 수 있음)"


async def main() -> int:
    parser = argparse.ArgumentParser(description="논문 청크 사용 가능성 분석")
    parser.add_argument("--samples", type=int, default=2, help="분류별로 보여줄 표본 수")
    args = parser.parse_args()

    settings = get_settings()
    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            rows = (
                await session.execute(
                    select(Chunk.content, Document.title, Document.source_id)
                    .join(Document, Chunk.document_id == Document.id)
                    .where(Document.corpus == "answer")
                    .where(Document.source_id.like("pmc-%"))
                )
            ).all()
    finally:
        await engine.dispose()

    if not rows:
        print("✗ PMC 청크가 없습니다 — 적재를 먼저 확인하세요", file=sys.stderr)
        return 1

    kinds = Counter()
    samples: dict[str, list] = {}
    for r in rows:
        kind = classify(r.content)
        kinds[kind] += 1
        samples.setdefault(kind, []).append((r.title, r.content))

    total = len(rows)
    print(f"\nPMC 논문 청크 {total:,}개\n")
    print(f"{'분류':<18} {'청크':>8} {'비율':>8}")
    print("─" * 38)
    drop = 0
    for kind, n in kinds.most_common():
        print(f"{kind:<18} {n:>8,} {n / total:>7.1%}")
        if kind != "본문 (쓸 수 있음)":
            drop += n
    print("─" * 38)
    print(f"{'걸러낼 후보':<18} {drop:>8,} {drop / total:>7.1%}")
    print(f"{'남는 본문':<18} {total - drop:>8,} {(total - drop) / total:>7.1%}")

    print("\n" + "═" * 76)
    print("표본 — **규칙이 멀쩡한 청크를 잡고 있지 않은지 눈으로 볼 것**")
    for kind, items in samples.items():
        print(f"\n▌ {kind}")
        for title, content in items[: args.samples]:
            print(f"   [{title[:52]}]")
            print(f"   {content[:180].replace(chr(10), ' ')}…")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
