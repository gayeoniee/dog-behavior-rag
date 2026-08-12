"""검색 품질 리포트 — `data/eval_questions.yaml` 기준.

    uv run python -m scripts.eval.retrieval_report
    uv run python -m scripts.eval.retrieval_report --save baseline
    uv run python -m scripts.eval.retrieval_report --compare baseline

"성능이 거지같다"를 숫자로 바꾸는 장치다. 이게 없어서 코퍼스를 25배로 늘리고도
품질이 그대로인 걸 한참 뒤에 알았다.

합격 기준이 그룹마다 다르다:
  covered      → 근거를 찾아야 통과
  uncovered    → **거절해야** 통과 (자료가 없다고 말해야 한다)
  out-of-scope → **거절해야** 통과

거절 판정은 근거 선별(evidence_select)이 붙기 전에는 항상 실패한다. 그게 정상이고,
이 리포트는 그 개선을 재기 위해 존재한다.
"""

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.services.embeddings.registry import get_embedder
from app.services.llm.registry import get_llm
from app.services.query_rewrite import QueryRewriter
from app.services.vectorstore.base import SearchHit
from app.services.vectorstore.pgvector import PgVectorStore

EVAL_PATH = Path("data/eval_questions.yaml")
RESULT_DIR = Path("data/eval_results")

PRACTICAL_MARKERS = ("ASPCA", "VCA", "RSPCA", "AVSAB", "AAHA", "Dogs Trust", "PDSA", "Battersea")


@dataclass
class Row:
    question: str
    topic: str
    expect: str
    top_score: float
    practical_in_top: int
    kept: int
    """근거 선별 후 남은 개수. 선별 단계가 없으면 검색 결과 개수와 같다."""
    coverage: str
    passed: bool
    top_title: str


def is_practical(title: str) -> bool:
    return any(m in title for m in PRACTICAL_MARKERS)


def judge(entry: dict, hits: list[SearchHit], kept: list[SearchHit], coverage: str) -> bool:
    """그룹별 합격 판정."""
    if entry["expect"] in ("uncovered", "out-of-scope"):
        # 거절해야 통과. 근거를 붙여 답하면 실패다.
        return coverage == "none"
    blob = " ".join(h.content.lower() for h in kept)
    keywords = [k.lower() for k in entry.get("keywords", [])]
    return coverage != "none" and any(k in blob for k in keywords)


async def run(save: str | None, compare: str | None) -> int:
    if not EVAL_PATH.is_file():
        print(f"✗ {EVAL_PATH} 가 없습니다", file=sys.stderr)
        return 1
    entries = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))

    settings = get_settings()
    embedder = get_embedder(settings)
    await embedder.warmup()
    llm = get_llm(settings)
    rewriter = QueryRewriter(llm, enabled=settings.query_rewrite_enabled)

    # 근거 선별은 B 단계에서 추가된다. 아직 없으면 선별 없이 측정한다.
    try:
        from app.services.evidence_select import EvidenceSelector

        selector = EvidenceSelector(llm, enabled=settings.evidence_select_enabled)
    except (ImportError, AttributeError):
        selector = None

    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    rows: list[Row] = []

    try:
        async with factory() as session:
            store = PgVectorStore(
                session,
                authority_boost=settings.authority_boost,
                max_per_document=settings.max_chunks_per_document,
                candidate_multiplier=settings.candidate_multiplier,
            )
            for i, entry in enumerate(entries, 1):
                q = entry["question"]
                hits = await store.search(
                    await embedder.embed_query(await rewriter.rewrite(q)), settings.top_k
                )
                if selector is not None:
                    sel = await selector.select(q, hits)
                    kept, coverage = sel.kept, sel.coverage
                else:
                    kept, coverage = hits, ("full" if hits else "none")

                row = Row(
                    question=q,
                    topic=entry["topic"],
                    expect=entry["expect"],
                    top_score=round(hits[0].score, 3) if hits else 0.0,
                    practical_in_top=sum(is_practical(h.document_title) for h in hits),
                    kept=len(kept),
                    coverage=coverage,
                    passed=judge(entry, hits, kept, coverage),
                    top_title=hits[0].document_title[:56] if hits else "—",
                )
                rows.append(row)
                print(
                    f"  [{i:>2}/{len(entries)}] {'✓' if row.passed else '✗'} {q[:38]}", flush=True
                )
    finally:
        await engine.dispose()

    _print_report(rows)
    if save:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULT_DIR / f"{save}.json"
        path.write_text(
            json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n✓ 저장: {path}")
    if compare:
        _print_compare(rows, RESULT_DIR / f"{compare}.json")
    return 0


def _print_report(rows: list[Row]) -> None:
    print("\n" + "═" * 96)
    print(f"{'판정':<4} {'그룹':<13} {'점수':>6} {'실무':>4} {'남은':>4} {'coverage':<9} 질문")
    print("─" * 96)
    for r in rows:
        print(
            f"{'✓' if r.passed else '✗':<4} {r.expect:<13} {r.top_score:>6.3f} "
            f"{r.practical_in_top:>4} {r.kept:>4} {r.coverage:<9} {r.question[:34]}"
        )

    print("\n── 그룹별 ──")
    for group in ("covered", "uncovered", "out-of-scope"):
        g = [r for r in rows if r.expect == group]
        if not g:
            continue
        ok = sum(r.passed for r in g)
        scores = [r.top_score for r in g]
        prac = [r.practical_in_top for r in g]
        print(
            f"  {group:<13} {ok}/{len(g)} 통과   "
            f"점수중앙 {statistics.median(scores):.3f}   "
            f"실무평균 {statistics.mean(prac):.1f}/5"
        )
    total_ok = sum(r.passed for r in rows)
    print(f"\n  ═══ 전체 {total_ok}/{len(rows)} 통과 ═══")


def _print_compare(rows: list[Row], path: Path) -> None:
    if not path.is_file():
        print(f"\n⚠️  비교 대상이 없습니다: {path}")
        return
    before = {r["question"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    print(f"\n── {path.stem} 대비 변화 ──")
    changed = 0
    for r in rows:
        b = before.get(r.question)
        if b is None or b["passed"] == r.passed:
            continue
        changed += 1
        arrow = "✗→✓ 개선" if r.passed else "✓→✗ 악화"
        print(f"  {arrow}  [{r.expect}] {r.question[:44]}")
    before_ok = sum(1 for b in before.values() if b["passed"])
    print(f"  전체 {before_ok}/{len(before)} → {sum(r.passed for r in rows)}/{len(rows)}")
    if not changed:
        print("  (판정이 바뀐 질문 없음)")


def main() -> int:
    parser = argparse.ArgumentParser(description="검색 품질 리포트")
    parser.add_argument(
        "--save", metavar="NAME", help="결과를 data/eval_results/NAME.json 으로 저장"
    )
    parser.add_argument("--compare", metavar="NAME", help="저장된 결과와 비교")
    args = parser.parse_args()
    return asyncio.run(run(args.save, args.compare))


if __name__ == "__main__":
    sys.exit(main())
