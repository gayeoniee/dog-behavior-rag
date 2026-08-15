"""하이브리드 검색 실험 — 벡터에 어휘 검색(BM25)을 더하면 좋아지는가.

    uv run python -m scripts.eval.hybrid_experiment --distractors 150
    uv run python -m scripts.eval.hybrid_experiment --rrf-k 10,60,200

**왜 이 실험을 하나.** 06장 baseline에서 단서가 나왔다:

    hit@5   49.5%
    hit@10  52.6%   ← 3.1%p밖에 안 오른다

**못 찾는 건 순서 문제가 아니라 아예 안 걸리는 것이다.** 상위 10위 안에도 없으면
리랭킹(10장)으로는 못 구한다. 후보 자체를 다르게 만들어야 한다.

**벡터 검색이 놓치는 자리가 어휘 검색의 자리다.** 임베딩은 뜻이 비슷하면 잡지만
**드문 고유명사·전문용어의 정확한 일치**에는 약하다. "alpha roll", "counterconditioning"
같은 단어는 그 단어가 있는 문서를 그냥 찾으면 되는데, 임베딩은 주변 의미로 뭉갠다.

**BM25**는 반대다. 단어가 겹치는 문서를 찾고, 흔한 단어(the, dog)는 가중치를 낮추고
드문 단어에 무게를 준다. 둘은 실패하는 자리가 다르므로 합치면 서로 메운다.

**RRF(Reciprocal Rank Fusion)로 합친다.** 점수를 직접 더하지 않는 이유는 코사인
유사도(0~1)와 BM25 점수(범위 없음)가 **단위가 다르기 때문**이다. 정규화해서 더하는
방법도 있지만 질의마다 분포가 달라 불안정하다. RRF는 점수를 버리고 **순위만** 쓴다:

    score(문서) = Σ 1 / (k + 그 검색기에서의 순위)

k는 상위권을 얼마나 우대할지 정한다 (작을수록 1위에 몰아준다).

⚠️ **이 실험의 BM25는 파이썬 구현이다.** 실제로 넣는다면 Postgres 전문검색
(`tsvector` + GIN)을 쓸 텐데 랭킹 함수가 달라 숫자가 그대로 옮겨가지 않는다.
**여기서는 "어휘 신호가 도움이 되는가"만 본다.** 도움이 되면 그때 제대로 만든다.
"""

import argparse
import asyncio
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.services.chunking import ChunkConfig
from app.services.embeddings.registry import get_embedder
from scripts.eval.chunk_experiment import (
    build_passages,
    cached_rewrites,
    covers,
    load_golds,
)

RESULT_PATH = Path("data/eval_results/hybrid-experiment.json")

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """소문자 영숫자 토큰. 코퍼스가 100% 영어라 이걸로 충분하다.

    한국어가 들어오면 공백 토큰화가 의미 단위와 안 맞아 형태소 분석이 필요해진다
    (`chunking.py`의 구분자 주석과 같은 이유).
    """
    return _TOKEN.findall(text.lower())


class BM25:
    """BM25 랭킹. 외부 의존성 없이 40줄이면 되고, 무엇을 하는지 눈에 보인다.

    핵심 아이디어 셋:
      1. 질의 단어가 문서에 많이 나올수록 점수가 높다 (단, 포화한다 — k1)
      2. **드문 단어일수록 무게가 크다** (IDF). "dog"는 코퍼스 전체에 있으니 정보가 없다
      3. 긴 문서는 단어가 많아 유리하므로 길이로 보정한다 (b)
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in docs]
        self.avg_len = sum(self.doc_len) / max(1, len(docs))
        self.freqs: list[Counter[str]] = [Counter(d) for d in docs]
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, freq in enumerate(self.freqs):
            for term in freq:
                self.postings[term].append(i)
        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
            for term, ids in self.postings.items()
        }

    def scores(self, query: list[str]) -> dict[int, float]:
        """질의와 겹치는 문서만 점수를 낸다 (역색인이라 전체를 훑지 않는다)."""
        out: dict[int, float] = defaultdict(float)
        for term in query:
            ids = self.postings.get(term)
            if not ids:
                continue
            idf = self.idf[term]
            for i in ids:
                tf = self.freqs[i][term]
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
                out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out


def rrf(rankings: list[list[int]], k: float) -> list[int]:
    """여러 순위 목록을 RRF로 합쳐 하나의 순위로.

    **점수가 아니라 순위만 쓴다.** 코사인(0~1)과 BM25(범위 없음)는 단위가 달라
    그냥 더할 수 없고, 정규화는 질의마다 분포가 달라 불안정하다.
    """
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] += 1.0 / (k + rank)
    return sorted(fused, key=lambda i: -fused[i])


@dataclass(slots=True)
class Run:
    label: str
    ranks: list[int]

    def hit(self, k: int) -> float:
        return sum(1 for r in self.ranks if 0 < r <= k) / len(self.ranks)

    @property
    def mrr(self) -> float:
        return statistics.mean(1.0 / r if r else 0.0 for r in self.ranks)


def rank_of_gold(order: list[int], gold, passages, threshold: float, top_k: int) -> int:
    for pos, idx in enumerate(order[:top_k], start=1):
        if covers(gold, passages[idx], threshold):
            return pos
    return 0


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    try:
        golds, docs = await load_golds(args.qa, factory)
    finally:
        await engine.dispose()
    if not golds:
        print("✗ 쓸 수 있는 정답 구간이 없습니다", file=sys.stderr)
        return 1

    gold_ids = {g.document_id for g in golds}
    others = [d for d in docs if d["_id"] not in gold_ids]
    random.Random(args.seed).shuffle(others)
    corpus = [d for d in docs if d["_id"] in gold_ids] + others[: args.distractors]
    passages = build_passages(corpus, ChunkConfig().size, ChunkConfig().overlap)
    print(f"\n문항 {len(golds)} · 문서 {len(corpus)}건 · 청크 {len(passages):,}개", flush=True)

    rewrites = await cached_rewrites([g.question for g in golds], settings)
    queries = [rewrites[g.question] for g in golds]

    print("  BM25 색인 만드는 중…", flush=True)
    bm25 = BM25([tokenize(p.text) for p in passages])

    print("  임베딩 중…", flush=True)
    embedder = get_embedder(settings)
    await embedder.warmup()
    qv = await embedder.embed(queries)
    pv = await embedder.embed([p.text for p in passages])

    import numpy as np

    sims = np.asarray(qv, dtype="float32") @ np.asarray(pv, dtype="float32").T
    pool = max(args.top_k, 100)  # RRF에 넣을 후보 깊이

    vector_orders, bm25_orders = [], []
    for i, query in enumerate(queries):
        vector_orders.append(np.argsort(-sims[i])[:pool].tolist())
        scored = bm25.scores(tokenize(query))
        bm25_orders.append(sorted(scored, key=lambda x: -scored[x])[:pool])

    runs = [
        Run("벡터만", [
            rank_of_gold(vector_orders[i], g, passages, args.threshold, args.top_k)
            for i, g in enumerate(golds)
        ]),
        Run("BM25만", [
            rank_of_gold(bm25_orders[i], g, passages, args.threshold, args.top_k)
            for i, g in enumerate(golds)
        ]),
    ]
    for k in [float(x) for x in args.rrf_k.split(",")]:
        runs.append(
            Run(f"RRF k={k:g}", [
                rank_of_gold(
                    rrf([vector_orders[i], bm25_orders[i]], k),
                    g, passages, args.threshold, args.top_k,
                )
                for i, g in enumerate(golds)
            ])
        )

    _report(runs, golds)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "questions": len(golds),
                "distractors": args.distractors,
                "runs": {
                    r.label: {"ranks": r.ranks, "hit@5": r.hit(5), "mrr": r.mrr}
                    for r in runs
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n✓ 저장: {RESULT_PATH}")
    return 0


def _report(runs: list[Run], golds: list) -> None:
    from math import comb

    def p2(a: int, b: int) -> float:
        n = a + b
        if n == 0:
            return 1.0
        return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2**n)

    print("\n" + "═" * 68)
    print(f"  {'방식':<14} {'hit@1':>8} {'hit@5':>8} {'hit@10':>8} {'MRR':>8}")
    print("─" * 68)
    for r in runs:
        print(
            f"  {r.label:<14} {r.hit(1):>8.1%} {r.hit(5):>8.1%} {r.hit(10):>8.1%} {r.mrr:>8.3f}"
        )

    base = runs[0]
    print(f"\n── {base.label} 대비 짝지은 검정 (hit@5) ──")
    for r in runs[1:]:
        pairs = list(zip(base.ranks, r.ranks, strict=True))
        only_base = sum(1 for b, x in pairs if 0 < b <= 5 < (x or 99))
        only_new = sum(1 for b, x in pairs if 0 < x <= 5 < (b or 99))
        p = p2(only_base, only_new)
        print(
            f"  {r.label:<14} {only_base:>3} : {only_new:<3}  p={p:.3f}  "
            f"{'유의미' if p < 0.05 else '구별 불가'}"
        )
    print(f"\n  문항 {len(golds)}개 기준.")


def main() -> int:
    parser = argparse.ArgumentParser(description="하이브리드 검색 실험 (벡터 + BM25)")
    parser.add_argument("--qa", type=Path, default=Path("data/eval_auto_qa_merged.jsonl"))
    parser.add_argument("--distractors", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rrf-k", default="10,60,200", help="RRF의 k 후보 (쉼표 구분)")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
