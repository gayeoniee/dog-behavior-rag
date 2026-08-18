"""한국어 재작성이 질문을 비틀지 않는지 — **본 측정 전에 5분으로 거른다.**

    uv run python -m scripts.eval.ko_rewrite_check
    uv run python -m scripts.eval.ko_rewrite_check --sample 60 --show 12

**왜 필요한가.** 1차 시도에서 KO 재작성이 모든 질문을 한 틀
(`반려견이 <행동>하는 행동의 원인과 교정 방법`)에 밀어넣었고, 581문항 hit@5가
48.7% → 42.5%로 무너졌다. **그걸 30분짜리 A/B를 끝까지 돌리고 나서야 알았다.**

틀이 밋밋해서가 아니라 **프레임을 강요**해서였다:

    "다른 개들과 어울리는 것이 왜 좋지 않나요?"
      → "…어울리는 행동의 원인과 교정 방법"      ← 다른 질문이 됐다

이 스크립트는 **본 측정과 같은 경로를** 40문항으로만 돌린다. 끔/켬 두 번 검색해서
정답 청크 순위를 비교한다.

    비틀림   재작성본과 원문의 임베딩 코사인. 1에 가까울수록 안 비틀었다 (참고용)
    순위     끔 대비 켬에서 정답 청크 순위가 나빠졌나 (판정 기준)

**순위가 나빠진 문항이 20% 이상이면 본 측정으로 넘어가지 않는다.**

⚠️ **처음엔 이걸 틀리게 만들었다.** 한국어 재작성으로 **한국어 풀만** 뒤져서
순위를 봤는데, 정답 청크가 영어인 문항이 절반이다. 실제 시스템에서 그건 영어
질의가 찾는다 — **시스템이 하지 않는 일을 시켜놓고 28% 나빠졌다고 읽었다.**
점검은 본 경로와 같은 모양이어야 한다.

⚠️ **그래도 A/B를 대신하지는 않는다.** 40문항이라 ±2문항이 5%p다. "확실히 나쁜
프롬프트"를 싸게 걸러내는 체일 뿐이다.
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from sqlalchemy import select

import scripts.collect  # noqa: F401 — 콘솔 UTF-8
from app.core.config import get_settings
from app.db.models import Chunk, Document
from app.db.session import create_engine, create_session_factory
from app.services.embeddings.registry import get_embedder
from app.services.llm.registry import get_llm
from app.services.query_rewrite import QueryRewriter, embed_by_language
from app.services.vectorstore.pgvector import PgVectorStore

QA_PATH = Path("data/eval_auto_qa_all.jsonl")

WORSE_LIMIT = 0.20
"""순위가 나빠진 문항 비율의 상한. 넘으면 종료 코드 1.

**경고만 찍으면 안 읽힌다.** 같은 날 `ir_metrics`가 "1문항으로 잰다"를 찍고도
멀쩡한 표를 뱉어서 그대로 넘어갈 뻔했다.
"""


TOP_K = 10
"""본 측정(`ir_metrics`)과 같은 값. 다르게 두면 비교가 안 된다."""


async def main() -> int:
    parser = argparse.ArgumentParser(description="한국어 재작성 사전 점검")
    parser.add_argument("--qa", type=Path, default=QA_PATH)
    parser.add_argument("--sample", type=int, default=40, help="검사할 문항 수")
    parser.add_argument("--show", type=int, default=8, help="눈으로 볼 표본 수")
    parser.add_argument("--seed", type=int, default=7, help="표본 고정 (프롬프트 비교용)")
    args = parser.parse_args()

    if not args.qa.is_file():
        print(f"✗ 평가셋이 없습니다: {args.qa}", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in args.qa.read_text(encoding="utf-8").splitlines()]
    # **표본을 고정한다.** 프롬프트를 고쳐가며 여러 번 돌리는 도구라, 매번 다른
    # 문항을 보면 개선인지 표본 운인지 알 수 없다.
    random.seed(args.seed)
    sample = random.sample(rows, min(args.sample, len(rows)))

    settings = get_settings()
    embedder = get_embedder(settings)
    await embedder.warmup()
    llm = get_llm(settings)
    print(f"  {len(sample)}문항 · LLM={llm.name}\n", flush=True)

    off = QueryRewriter(llm, bilingual=False)
    on = QueryRewriter(llm, bilingual=True)

    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    results: list[dict] = []
    try:
        async with factory() as session:
            store = PgVectorStore(
                session,
                authority_boost=settings.authority_boost,
                guide_boost=settings.guide_boost,
                max_per_document=settings.max_chunks_per_document,
                candidate_multiplier=settings.candidate_multiplier,
                ef_search=settings.hnsw_ef_search,
                background_weight=settings.language_background_weight,
            )
            langs = dict((await session.execute(select(Document.id, Document.language))).all())
            chunk_lang = {
                cid: langs.get(did)
                for cid, did in (
                    await session.execute(select(Chunk.id, Chunk.document_id))
                ).all()
            }

            for i, qa in enumerate(sample, 1):
                question = qa["question"]
                # **본 측정과 같은 경로로 두 번 돌린다** — 끔/켬 각각 재작성부터
                # 병합 검색까지. 한국어 풀만 따로 보면 시스템이 하지 않는 일을
                # 재게 된다 (정답 청크의 절반이 영어다).
                ranks = []
                for rewriter in (off, on):
                    query = await rewriter.rewrite(question)
                    vectors = await embed_by_language(embedder, query)
                    hits = await store.search(vectors, TOP_K)
                    ranks.append(
                        next(
                            (r for r, h in enumerate(hits, 1) if h.chunk_id == qa["chunk_id"]),
                            None,
                        )
                    )
                    if rewriter is on:
                        rewritten = query.ko

                pair = await embedder.embed([question, rewritten])
                skew = sum(a * b for a, b in zip(pair[0], pair[1], strict=True))
                results.append(
                    {
                        "q": question,
                        "rw": rewritten,
                        "skew": skew,
                        "before": ranks[0],
                        "after": ranks[1],
                        "lang": chunk_lang.get(qa["chunk_id"]) or "?",
                    }
                )
                print(
                    f"    [{i}/{len(sample)}] {ranks[0]} → {ranks[1]}  {rewritten[:46]}",
                    flush=True,
                )
    finally:
        await engine.dispose()

    def worse(r: dict) -> bool:
        """못 찾게 됐거나 순위가 밀렸으면 나빠진 것. 둘 다 못 찾았으면 무승부."""
        if r["before"] is None:
            return False
        return r["after"] is None or r["after"] > r["before"]

    def better(r: dict) -> bool:
        if r["after"] is None:
            return False
        return r["before"] is None or r["after"] < r["before"]

    n = len(results)
    n_worse = sum(1 for r in results if worse(r))
    n_better = sum(1 for r in results if better(r))
    skew_avg = sum(r["skew"] for r in results) / max(n, 1)
    unchanged = sum(1 for r in results if r["q"].strip() == r["rw"].strip())

    print(f"\n{'=' * 66}")
    print(f"  {n}문항 · 언어별 질의 끔 → 켬")
    print(f"{'=' * 66}\n")
    print(f"  원문과의 유사도(비틀림)   {skew_avg:.3f}   1에 가까울수록 안 비틀었다")
    print(f"  글자 그대로 유지          {unchanged}/{n}")
    print(f"  정답 순위 좋아짐          {n_better}/{n}")
    print(f"  정답 순위 나빠짐          {n_worse}/{n} = {n_worse / max(n, 1) * 100:.0f}%"
          f"   (상한 {WORSE_LIMIT * 100:.0f}%)")

    shown = sorted(results, key=lambda r: r["skew"])[: args.show]
    print(f"\n  ── 가장 많이 비튼 {len(shown)}건 (눈으로 볼 것) ──")
    for r in shown:
        mark = "✗" if worse(r) else ("○" if better(r) else " ")
        print(f"   {mark} [{r['skew']:.3f}] {r['q'][:52]}")
        print(f"          → {r['rw'][:60]}")
        print(f"          순위 {r['before']} → {r['after']}")

    if n_worse / max(n, 1) >= WORSE_LIMIT:
        print(
            f"\n✗ 나빠진 문항이 {n_worse / max(n, 1) * 100:.0f}%입니다. "
            f"본 측정(ir_metrics)으로 넘어가지 말고 프롬프트를 고치세요.",
            file=sys.stderr,
        )
        return 1
    print("\n✓ 사전 점검 통과 — 본 측정으로 넘어가도 됩니다")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
