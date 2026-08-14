"""검색 품질 지표 — hit rate와 MRR.

    uv run python -m scripts.eval.ir_metrics
    uv run python -m scripts.eval.ir_metrics --save baseline
    uv run python -m scripts.eval.ir_metrics --compare baseline
    uv run python -m scripts.eval.ir_metrics --no-rewrite     # 질의 재작성 끄고 비교

`retrieval_report`(21문항)와 무엇이 다른가:

    retrieval_report   사람이 쓴 질문 · 키워드로 통과/실패 · **근거 선별까지** 포함
    ir_metrics         기계가 만든 질문 · 정답 청크로 순위 채점 · **검색만**

여기는 **검색 단계만** 잰다. LLM 근거 선별을 부르지 않으므로 실행마다 흔들리지
않는다(임베딩과 벡터 검색은 결정적이다). 청킹·임베딩 모델·하이브리드 검색처럼
**검색 자체를 바꾸는 실험**은 이 지표로 비교해야 한다.

**두 가지로 채점하는 이유는 05장의 한계 때문이다.** 자동 생성 평가셋은 "정답이
하나"라고 가정하는데, 우리 코퍼스에는 같은 주제 청크가 수십 개다. 그래서:

    청크 단위 (엄격)  질문을 만든 바로 그 청크가 올라왔나
    문서 단위 (관대)  그 청크가 속한 문서의 아무 청크나 올라왔나

**진짜 성능은 이 둘 사이 어딘가에 있다.** 하나만 보면 과소평가하거나 과대평가한다.
"""

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Chunk
from app.db.session import create_engine, create_session_factory
from app.services.embeddings.registry import get_embedder
from app.services.llm.registry import get_llm
from app.services.query_rewrite import QueryRewriter
from app.services.vectorstore.pgvector import PgVectorStore

QA_PATH = Path("data/eval_auto_qa.jsonl")
RESULT_DIR = Path("data/eval_results")

K_VALUES = (1, 3, 5, 10)
"""hit rate를 여러 k에서 본다.

k=1만 보면 "1등을 맞혔나"라 너무 가혹하고, k=10만 보면 실제 답변이 top_k=5만
쓰므로 현실과 안 맞는다. **k=5가 이 시스템의 진짜 기준이다** (settings.top_k).
"""


@dataclass(slots=True)
class Row:
    question: str
    gold_chunk_id: int
    gold_document_id: int
    chunk_rank: int
    """정답 청크의 순위 (1부터). 못 찾았으면 0."""
    doc_rank: int
    """같은 문서 청크가 처음 나온 순위. 못 찾았으면 0."""
    top_title: str


def reciprocal_rank(rank: int) -> float:
    """MRR의 한 항. 1등이면 1.0, 2등이면 0.5, 못 찾으면 0.

    **왜 hit rate만으로 부족한가:** hit rate@5는 1등과 5등을 똑같이 센다. 그런데
    실제로는 1등이 훨씬 낫다 — 프롬프트 앞쪽에 오고, top_k를 줄여도 살아남는다.
    MRR은 그 차이를 점수에 반영한다.
    """
    return 1.0 / rank if rank > 0 else 0.0


def hit_rate(ranks: list[int], k: int) -> float:
    """상위 k 안에 정답이 있던 질문의 비율."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if 0 < r <= k) / len(ranks)


async def run(args: argparse.Namespace) -> int:
    if not args.qa.is_file():
        print(f"✗ {args.qa} 가 없습니다", file=sys.stderr)
        print("  generate_qa 를 먼저 실행하세요", file=sys.stderr)
        return 1
    lines = args.qa.read_text(encoding="utf-8").splitlines()
    pairs = [json.loads(line) for line in lines if line.strip()]
    if not pairs:
        print(f"✗ {args.qa} 가 비어 있습니다", file=sys.stderr)
        return 1

    settings = get_settings()
    embedder = get_embedder(settings)
    await embedder.warmup()
    rewriter = QueryRewriter(
        get_llm(settings) if args.rewrite else None, enabled=args.rewrite
    )

    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    rows: list[Row] = []

    try:
        async with factory() as session:
            # 검색 결과(SearchHit)에는 document_id가 없다. 문서 단위 채점을 하려면
            # 필요하므로 청크→문서 맵을 미리 한 번 만든다 (1만 행이라 수십 ms).
            chunk_to_doc = {
                r.id: r.document_id
                for r in (await session.execute(select(Chunk.id, Chunk.document_id))).all()
            }

            store = PgVectorStore(
                session,
                authority_boost=settings.authority_boost,
                guide_boost=settings.guide_boost,
                max_per_document=settings.max_chunks_per_document,
                candidate_multiplier=settings.candidate_multiplier,
            )
            for i, qa in enumerate(pairs, 1):
                question = qa["question"]
                query = await rewriter.rewrite(question)
                hits = await store.search(await embedder.embed_query(query), args.top_k)

                chunk_rank = doc_rank = 0
                for pos, hit in enumerate(hits, 1):
                    if chunk_rank == 0 and hit.chunk_id == qa["chunk_id"]:
                        chunk_rank = pos
                    if doc_rank == 0 and chunk_to_doc.get(hit.chunk_id) == qa["document_id"]:
                        doc_rank = pos

                rows.append(
                    Row(
                        question=question,
                        gold_chunk_id=qa["chunk_id"],
                        gold_document_id=qa["document_id"],
                        chunk_rank=chunk_rank,
                        doc_rank=doc_rank,
                        top_title=hits[0].document_title[:50] if hits else "—",
                    )
                )
                mark = f"{chunk_rank}위" if chunk_rank else ("문서만" if doc_rank else "✗")
                print(f"  [{i:>3}/{len(pairs)}] {mark:<6} {question[:40]}", flush=True)
    finally:
        await engine.dispose()

    _print_report(rows, args)
    if args.save:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULT_DIR / f"ir-{args.save}.json"
        path.write_text(
            json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n✓ 저장: {path}")
    if args.compare:
        _print_compare(rows, RESULT_DIR / f"ir-{args.compare}.json")
    return 0


def _print_report(rows: list[Row], args: argparse.Namespace) -> None:
    chunk_ranks = [r.chunk_rank for r in rows]
    doc_ranks = [r.doc_rank for r in rows]

    print("\n" + "═" * 72)
    print(f"  {len(rows)}문항 · top_k={args.top_k} · 질의 재작성 {'켬' if args.rewrite else '끔'}")
    print("═" * 72)

    print(f"\n  {'':14} {'청크 단위(엄격)':>16} {'문서 단위(관대)':>16}")
    for k in K_VALUES:
        if k > args.top_k:
            continue
        print(
            f"  hit rate@{k:<5} {hit_rate(chunk_ranks, k):>15.1%} "
            f"{hit_rate(doc_ranks, k):>16.1%}"
        )
    chunk_mrr = statistics.mean(reciprocal_rank(r) for r in chunk_ranks)
    doc_mrr = statistics.mean(reciprocal_rank(r) for r in doc_ranks)
    print(f"  {'MRR':<14} {chunk_mrr:>15.3f} {doc_mrr:>16.3f}")

    missed = [r for r in rows if r.chunk_rank == 0]
    print(f"\n  정답 청크를 못 찾은 질문 {len(missed)}건")
    for r in missed[:5]:
        note = f"(문서는 {r.doc_rank}위)" if r.doc_rank else "(문서도 못 찾음)"
        print(f"    · {r.question[:44]} {note}")
    if len(missed) > 5:
        print(f"    … 외 {len(missed) - 5}건")


def _print_compare(rows: list[Row], path: Path) -> None:
    if not path.is_file():
        print(f"\n⚠️  비교 대상이 없습니다: {path}")
        return
    before = [Row(**r) for r in json.loads(path.read_text(encoding="utf-8"))]
    by_q = {r.question: r for r in before}

    print(f"\n── {path.stem} 대비 변화 ──")
    for label, get in (("청크", lambda r: r.chunk_rank), ("문서", lambda r: r.doc_rank)):
        b = [get(r) for r in before]
        a = [get(r) for r in rows]
        print(
            f"  {label} hit@5  {hit_rate(b, 5):.1%} → {hit_rate(a, 5):.1%}   "
            f"MRR {statistics.mean(reciprocal_rank(x) for x in b):.3f} → "
            f"{statistics.mean(reciprocal_rank(x) for x in a):.3f}"
        )

    moved = [
        (r, by_q[r.question])
        for r in rows
        if r.question in by_q and by_q[r.question].chunk_rank != r.chunk_rank
    ]
    print(f"\n  순위가 바뀐 질문 {len(moved)}건")
    for now, was in moved[:8]:
        arrow = "개선" if (now.chunk_rank or 99) < (was.chunk_rank or 99) else "악화"
        print(f"    {was.chunk_rank or '✗'} → {now.chunk_rank or '✗'} {arrow}  {now.question[:38]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="검색 품질 지표 (hit rate · MRR)")
    parser.add_argument("--qa", type=Path, default=QA_PATH)
    parser.add_argument("--top-k", type=int, default=10, help="이만큼 받아와 순위를 본다")
    parser.add_argument("--save", metavar="NAME")
    parser.add_argument("--compare", metavar="NAME")
    parser.add_argument(
        "--no-rewrite",
        dest="rewrite",
        action="store_false",
        help="질의 재작성을 끄고 잰다 (재작성의 효과를 보려면)",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
