"""리랭킹 실험 — 2차 심사가 실제로 순위를 고치는가.

    # 1단계 — 후보를 뽑아 파일로 (임베딩 모델 + DB 필요)
    uv run python -m scripts.eval.rerank_experiment retrieve --top-n 20
    # 2단계 — 그 파일로 리랭킹 (리랭커만 필요)
    uv run python -m scripts.eval.rerank_experiment rerank --model BAAI/bge-reranker-base

**리랭킹이 무엇인가.** 지금 검색은 질문과 문서를 *각각* 벡터로 바꿔 비교한다
(bi-encoder). 빠르지만 둘을 따로 읽으므로 미묘한 관계를 놓친다. 리랭커는
**질문과 문서를 한 번에 넣고** 관련도를 매긴다(cross-encoder). 정확하지만 느려서,
1차로 넉넉히 뽑은 뒤 그 후보에만 쓴다.

**왜 해볼 만한가 (2026-08-17 실측, 293문항):**

    top-5 안에 정답 청크    48.1%
    top-20 안에 정답 청크   56.0%   ← 6~20위에 숨은 8%p가 리랭킹의 최대치
    top-50 안에 정답 청크   59.4%

못 찾은 사례가 "(문서는 1위)"처럼 **맞는 문서인데 엉뚱한 청크**인 경우가 많다.
문서 안에서 어느 대목이 답인지 가리는 건 cross-encoder가 잘하는 일이다.

**단계를 프로세스로 나눈 이유: 한 프로세스에서 bge-m3와 cross-encoder를 같이
올리면 segfault(139)로 죽는다.** 08장에서 임베딩 모델을 바꿔 끼울 때 겪은 것과
같은 문제라 같은 해법을 쓴다. 부수 효과로 **리랭커를 여러 개 비교할 때 재검색이
필요 없다** — 후보 파일은 한 번만 만들면 된다.

**다양성 상한을 풀고 잰다.** 운영 검색은 문서당 2청크로 제한하는데, 정답이 같은
문서의 세 번째 청크면 **상한이 정답을 막는다.** 리랭커 탓인지 상한 탓인지
섞이면 안 되므로 후보 단계에서는 상한을 없앤다. 상한을 유지한 결과도 같이 찍어서
**둘 중 무엇이 실제로 막고 있었는지** 보이게 한다.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from sqlalchemy import select

import scripts.collect  # noqa: F401 — 콘솔 UTF-8
from app.core.config import get_settings
from app.db.models import Chunk
from app.db.session import create_engine, create_session_factory
from app.services.embeddings.registry import get_embedder
from app.services.vectorstore.pgvector import PgVectorStore

QA_PATH = Path("data/eval_auto_qa_merged.jsonl")
CANDIDATES = Path("data/eval_results/rerank_candidates.json")

DEFAULT_MODEL = "BAAI/bge-reranker-base"
"""⚠️ **이 모델은 여기서 쓸 수 없다 (2026-08-17 실측). 결과가 크게 나빠진다.**

명백한 정답/오답 쌍으로 먼저 채점해보니 원인이 바로 나왔다:

    영어 질문 → 영어 정답      0.9972    ← 영어끼리는 완벽하다
    영어 질문 → 영어 오답      0.0000
    한국어 질문 → 영어 정답    0.0001    ← 교차언어를 아예 못 한다
    한국어 질문 → 영어 오답    0.0000
    한국어 질문 → 한국어 정답  0.3658    ← 한국어끼리도 구분을 못 한다
    한국어 질문 → 한국어 오답  0.3555

**이 프로젝트는 한국어 질문 + 대부분 영어 코퍼스다.** 그러니 리랭킹 후 순위는
관련도가 아니라 **언어로 정렬된 것**이 된다 — 한국어 청크(0.36)가 영어 청크
(0.0001) 위로 전부 올라간다. 150문항에서 hit@5가 59.3% → 38.0%로 떨어졌다.

**교훈: 30초짜리 채점을 12분짜리 실험보다 먼저 했어야 했다.** 판정 LLM을 쓰기 전에
판정기를 먼저 재는 것과 같은 이야기인데(12장), 리랭커도 판정기라는 걸 놓쳤다.

제대로 하려면 다국어 학습된 `BAAI/bge-reranker-v2-m3`(568M·2.3GB)가 필요한데
이 PC에서는 못 올린다 — VRAM은 데스크톱 앱이 3.7GB를 써서 2.2GB만 남고 RAM도
3GB뿐이라 어느 쪽에도 안 들어간다(CUDA OOM / "memory allocation failed").
여유 있는 기기에서 **같은 후보 파일에** `--model BAAI/bge-reranker-v2-m3`로
돌리면 재검색 없이 비교할 수 있다.
"""


def hit_rate(ranks: list[int], k: int) -> float:
    """정답이 상위 k 안에 든 질문의 비율. rank 0은 '못 찾음'이다."""
    return sum(1 for r in ranks if 0 < r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int]) -> float:
    return statistics.mean(1.0 / r if r else 0.0 for r in ranks) if ranks else 0.0


def _ranks(
    order: list[int], gold_chunk: int, doc_of: dict, gold_doc: int
) -> tuple[int, int]:
    """(청크 순위, 문서 순위). 못 찾으면 0."""
    chunk_rank = doc_rank = 0
    for pos, chunk_id in enumerate(order, 1):
        if chunk_rank == 0 and chunk_id == gold_chunk:
            chunk_rank = pos
        if doc_rank == 0 and doc_of.get(chunk_id) == gold_doc:
            doc_rank = pos
    return chunk_rank, doc_rank


def _report(title: str, chunk_ranks: list[int], doc_ranks: list[int] | None) -> None:
    print(f"\n  {title}")
    print(f"    {'':18} {'청크 단위':>10} {'문서 단위':>10}")
    for k in (1, 3, 5):
        doc = f"{hit_rate(doc_ranks, k):>10.1%}" if doc_ranks else f"{'—':>10}"
        print(f"    hit rate@{k:<9} {hit_rate(chunk_ranks, k):>9.1%} {doc}")
    doc_mrr = f"{mrr(doc_ranks):>10.3f}" if doc_ranks else f"{'—':>10}"
    print(f"    {'MRR':<18} {mrr(chunk_ranks):>9.3f} {doc_mrr}")


async def retrieve(args: argparse.Namespace) -> int:
    """1단계 — 질문마다 후보를 뽑아 파일로 남긴다."""
    if not args.qa.is_file():
        print(f"✗ {args.qa} 가 없습니다", file=sys.stderr)
        return 1
    pairs = [json.loads(x) for x in args.qa.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        pairs = pairs[: args.limit]

    settings = get_settings()
    embedder = get_embedder(settings)
    await embedder.warmup()

    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    out: list[dict] = []

    try:
        async with factory() as session:
            chunk_to_doc = {
                r.id: r.document_id
                for r in (await session.execute(select(Chunk.id, Chunk.document_id))).all()
            }
            # 상한 없는 후보 풀 — 리랭커에게 공정한 기회를 준다.
            wide = PgVectorStore(session, max_per_document=args.top_n, candidate_multiplier=4)
            # 운영과 같은 상한. 무엇이 정답을 막고 있었는지 가르기 위해 같이 잰다.
            prod = PgVectorStore(
                session,
                max_per_document=settings.max_chunks_per_document,
                candidate_multiplier=settings.candidate_multiplier,
            )

            for i, qa in enumerate(pairs, 1):
                vec = await embedder.embed_query(qa["question"])
                hits = await wide.search(vec, args.top_n)
                prod_hits = await prod.search(vec, args.top_n)
                out.append(
                    {
                        "question": qa["question"],
                        "gold_chunk": qa["chunk_id"],
                        "gold_doc": qa["document_id"],
                        "candidates": [
                            {
                                "chunk_id": h.chunk_id,
                                "doc_id": chunk_to_doc.get(h.chunk_id),
                                "content": h.content,
                            }
                            for h in hits
                        ],
                        "capped_chunk_ids": [h.chunk_id for h in prod_hits],
                    }
                )
                print(
                    f"  [{i:>3}/{len(pairs)}] 후보 {len(hits):>2}개  {qa['question'][:44]}",
                    flush=True,
                )
    finally:
        await engine.dispose()

    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ 저장: {args.candidates} ({len(out)}문항)")
    print("  다음: rerank — 프로세스를 나눠야 segfault를 피한다")
    return 0


def rerank(args: argparse.Namespace) -> int:
    """2단계 — 저장된 후보를 cross-encoder로 다시 세운다."""
    if not args.candidates.is_file():
        print(f"✗ {args.candidates} 가 없습니다 — retrieve 를 먼저 실행하세요", file=sys.stderr)
        return 1
    rows = json.loads(args.candidates.read_text(encoding="utf-8"))

    from sentence_transformers import CrossEncoder

    print(f"리랭커 로딩: {args.model} ({args.device})", flush=True)
    t0 = time.perf_counter()
    reranker = CrossEncoder(args.model, max_length=512, device=args.device)
    print(f"  {time.perf_counter() - t0:.1f}초\n", flush=True)

    base_c: list[int] = []
    base_d: list[int] = []
    capped_c: list[int] = []
    rerank_c: list[int] = []
    rerank_d: list[int] = []
    elapsed = 0.0

    for i, row in enumerate(rows, 1):
        cands = row["candidates"]
        doc_of = {c["chunk_id"]: c["doc_id"] for c in cands}
        gold_c, gold_d = row["gold_chunk"], row["gold_doc"]

        cc, dd = _ranks([c["chunk_id"] for c in cands], gold_c, doc_of, gold_d)
        base_c.append(cc)
        base_d.append(dd)
        pc, _ = _ranks(row["capped_chunk_ids"], gold_c, doc_of, gold_d)
        capped_c.append(pc)

        t = time.perf_counter()
        scores = reranker.predict([(row["question"], c["content"]) for c in cands])
        elapsed += time.perf_counter() - t
        order = [
            c["chunk_id"]
            for c, _ in sorted(zip(cands, scores, strict=True), key=lambda p: -p[1])
        ]
        rc, rd = _ranks(order, gold_c, doc_of, gold_d)
        rerank_c.append(rc)
        rerank_d.append(rd)

        arrow = "=" if rc == cc else ("↑" if rc and (not cc or rc < cc) else "↓")
        print(
            f"  [{i:>3}/{len(rows)}] {cc or '✗':>3} {arrow} {rc or '✗':<3} "
            f"{row['question'][:44]}",
            flush=True,
        )

    n = len(rows)
    print(f"\n{'=' * 72}\n문항 {n}개 · 후보 {len(rows[0]['candidates'])}개 · {args.model}")
    _report("운영 그대로 (문서당 청크 상한 있음)", capped_c, None)
    _report("상한 해제 (리랭킹 전)", base_c, base_d)
    _report("상한 해제 + 리랭킹", rerank_c, rerank_d)

    print(f"\n  리랭킹 비용: 질문당 {elapsed / n * 1000:.0f}ms")
    print(f"\n  상한 해제만으로   hit@5 {hit_rate(base_c, 5) - hit_rate(capped_c, 5):+.1%}")
    print(f"  리랭킹으로        hit@5 {hit_rate(rerank_c, 5) - hit_rate(base_c, 5):+.1%}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="cross-encoder 리랭킹 효과 측정")
    parser.add_argument("stage", choices=["retrieve", "rerank"])
    parser.add_argument("--qa", type=Path, default=QA_PATH)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-n", type=int, default=20, help="리랭킹할 후보 수")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N문항만")
    parser.add_argument(
        "--device",
        default="cpu",
        help="리랭커를 올릴 곳. 이 PC는 VRAM 2.2GB·RAM 3GB만 남아 큰 모델은 못 올린다",
    )
    args = parser.parse_args()
    return asyncio.run(retrieve(args)) if args.stage == "retrieve" else rerank(args)


if __name__ == "__main__":
    sys.exit(main())
