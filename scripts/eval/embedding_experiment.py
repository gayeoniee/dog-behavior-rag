"""임베딩 모델 A/B 실험 — 어느 모델이 우리 데이터에서 잘 찾는가.

    uv run python -m scripts.eval.embedding_experiment --models bge-m3,kure-v1
    uv run python -m scripts.eval.embedding_experiment --models all --distractors 150

**왜 리더보드만 보고 고르면 안 되나.** MTEB 점수는 수십 개 과제의 평균이고,
우리 과제는 그중 하나(retrieval)의 특수한 경우다 — **한국어 질문으로 영어 문서를
찾는 교차언어 검색.** 리더보드 1위가 이 조합에서도 1위라는 보장이 없다.

**모델마다 입력 규약이 다르고, 그걸 안 지키면 비교가 부당해진다** (2026-08-15 조사):

    bge-m3        접두사 없음        · 8192토큰
    KURE-v1       접두사 없음        · 8192토큰 (bge-m3에서 한국어로 파인튜닝)
    e5-large      query:/passage: 필수 · **512토큰**  ← 우리 청크 일부가 잘린다
    Qwen3-0.6B    질의에 Instruct 권장 · 32K토큰

e5는 접두사를 빼면 성능이 떨어진다고 모델 카드에 명시돼 있다. 04장에서 "지표가
특정 조건을 원리상 불리하게 만들지 않는지 확인하라"를 배웠는데, 여기서는
**입력 규약이 그 자리다.**

`chunk_experiment`의 채점 장치(구간 겹침, 짝지은 검정)를 그대로 쓴다. 다른 점은
**청킹을 한 번만 하고 모델만 바꾼다**는 것 — 비교 대상이 임베딩이므로 나머지는 고정.
"""

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.services.chunking import ChunkConfig
from app.services.embeddings.huggingface import HuggingFaceEmbedder
from scripts.eval.chunk_experiment import (
    build_passages,
    cached_rewrites,
    load_golds,
    score,
)

RESULT_PATH = Path("data/eval_results/embedding-experiment.json")


@dataclass(slots=True, frozen=True)
class Candidate:
    key: str
    model: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 1024
    note: str = ""


CANDIDATES = {
    c.key: c
    for c in (
        Candidate(
            key="bge-m3",
            model="BAAI/bge-m3",
            note="현재 사용 중. 교차언어 검색을 노리고 골랐지만 비교한 적은 없다",
        ),
        Candidate(
            key="kure-v1",
            model="nlpai-lab/KURE-v1",
            note="bge-m3를 한국어 200만 쌍으로 파인튜닝. 차원·길이가 같아 교체가 공짜다",
        ),
        Candidate(
            key="e5-large",
            model="intfloat/multilingual-e5-large",
            query_prefix="query: ",
            passage_prefix="passage: ",
            max_seq_length=512,
            note="접두사가 필수. 512토큰이라 긴 청크는 잘린다",
        ),
        Candidate(
            key="qwen3-0.6b",
            model="Qwen/Qwen3-Embedding-0.6B",
            query_prefix=(
                "Instruct: Given a Korean question from a dog owner, retrieve English "
                "veterinary-behaviour passages that answer it\nQuery: "
            ),
            max_seq_length=1024,
            note="질의에 지시문을 붙이면 1~5% 오른다고 모델 카드가 말한다",
        ),
    )
}


def make_embedder(candidate: Candidate, settings) -> HuggingFaceEmbedder:
    """후보별 설정으로 임베더를 만든다.

    `Settings`를 복사해서 넘긴다 — 전역 설정을 건드리면 이 프로세스의 다른 부분이
    조용히 영향을 받는다.
    """
    return HuggingFaceEmbedder(
        settings.model_copy(
            update={
                "hf_embedding_model": candidate.model,
                "embedding_max_seq_length": candidate.max_seq_length,
            }
        )
    )


async def run(args: argparse.Namespace) -> int:
    keys = list(CANDIDATES) if args.models == "all" else [
        k.strip() for k in args.models.split(",")
    ]
    for key in keys:
        if key not in CANDIDATES:
            print(f"✗ 모르는 후보: {key} (가능: {', '.join(CANDIDATES)})", file=sys.stderr)
            return 1

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

    gold_doc_ids = {g.document_id for g in golds}
    gold_docs = [d for d in docs if d["_id"] in gold_doc_ids]
    others = [d for d in docs if d["_id"] not in gold_doc_ids]
    random.Random(args.seed).shuffle(others)
    corpus = gold_docs + others[: args.distractors]

    # **청킹은 한 번만 한다.** 비교 대상이 임베딩이므로 청크는 전 모델이 공유한다.
    passages = build_passages(corpus, ChunkConfig().size, ChunkConfig().overlap)
    print(f"\n문항 {len(golds)} · 문서 {len(corpus)}건 · 청크 {len(passages):,}개")
    print(f"후보: {', '.join(keys)}\n", flush=True)

    rewrites = await cached_rewrites([g.question for g in golds], settings)
    queries = [rewrites[g.question] for g in golds]
    texts = [p.text for p in passages]

    results: dict[str, dict] = load_previous(args, len(golds))
    if results:
        print(f"  이전 결과 재사용: {', '.join(results)}\n")
    for key in keys:
        cand = CANDIDATES[key]
        print(f"  [{key}] {cand.model} 로딩·임베딩 중…", flush=True)
        embedder = make_embedder(cand, settings)
        try:
            await embedder.warmup()
            qv = await embedder.embed([cand.query_prefix + q for q in queries])
            pv = await embedder.embed([cand.passage_prefix + t for t in texts])
        except Exception as exc:  # noqa: BLE001 — 한 모델이 죽어도 나머지는 계속
            print(f"    ✗ 실패: {exc}", file=sys.stderr)
            continue

        results[key] = score(golds, passages, qv, pv, args.top_k, args.threshold)
        results[key]["model"] = cand.model
        results[key]["dimension"] = embedder.dimension
        r = results[key]
        print(f"    hit@5 {r['hit@5']:.1%} · MRR {r['mrr']:.3f}", flush=True)
        _save(results, args, len(golds))

    if not results:
        print("✗ 성공한 후보가 없습니다", file=sys.stderr)
        return 1
    _report(results, golds, args.baseline)
    print(f"\n✓ 저장: {RESULT_PATH}")
    return 0


def load_previous(args: argparse.Namespace, questions: int) -> dict[str, dict]:
    """이전 실행 결과를 읽어온다.

    **모델마다 프로세스를 나눠 돌리기 때문에 필요하다.** VRAM 6GB에서 한 프로세스가
    모델을 갈아끼우면 두 번째 로딩에서 세그폴트(exit 139)로 죽는다. 프로세스를
    분리하면 OS가 확실히 정리해준다.

    조건이 다른 실행이 섞이면 비교가 무의미하므로 그때는 버린다.
    """
    if not RESULT_PATH.is_file():
        return {}
    try:
        prev = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    same = (
        prev.get("questions") == questions
        and prev.get("distractors") == args.distractors
        and prev.get("threshold") == args.threshold
    )
    if not same:
        print("  ⚠️ 조건이 다른 이전 결과는 버린다 (문항수·방해문서·임계 중 하나가 다름)")
        return {}
    return prev.get("results", {})


def _save(results: dict[str, dict], args: argparse.Namespace, questions: int) -> None:
    """후보 하나가 끝날 때마다 저장한다 — 모델 하나에 수 분씩 걸리므로."""
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "questions": questions,
                "distractors": args.distractors,
                "threshold": args.threshold,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _report(results: dict[str, dict], golds: list, baseline: str) -> None:
    from math import comb

    def two_sided(a: int, b: int) -> float:
        n = a + b
        if n == 0:
            return 1.0
        lo = min(a, b)
        return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2**n)

    print("\n" + "═" * 74)
    print(f"  {'후보':<12} {'차원':>5} {'hit@1':>8} {'hit@5':>8} {'hit@10':>8} {'MRR':>8}")
    print("─" * 74)
    for key, r in results.items():
        print(
            f"  {key:<12} {r['dimension']:>5} {r['hit@1']:>8.1%} {r['hit@5']:>8.1%} "
            f"{r['hit@10']:>8.1%} {r['mrr']:>8.3f}"
        )

    if baseline not in results or len(results) < 2:
        return
    print(f"\n── {baseline} 대비 짝지은 검정 (hit@5) ──")
    base = results[baseline]["ranks"]
    for key, r in results.items():
        if key == baseline:
            continue
        only_base = sum(1 for b, x in zip(base, r["ranks"], strict=True) if 0 < b <= 5 < (x or 99))
        only_new = sum(1 for b, x in zip(base, r["ranks"], strict=True) if 0 < x <= 5 < (b or 99))
        p = two_sided(only_base, only_new)
        verdict = "유의미" if p < 0.05 else "구별 불가"
        print(f"  {key:<12} {only_base:>3} : {only_new:<3}  p={p:.3f}  {verdict}")
    print(f"\n  문항 {len(golds)}개 기준이다.")


def main() -> int:
    parser = argparse.ArgumentParser(description="임베딩 모델 A/B 실험")
    parser.add_argument("--qa", type=Path, default=Path("data/eval_auto_qa_merged.jsonl"))
    parser.add_argument("--models", default="all", help=f"쉼표 구분 ({', '.join(CANDIDATES)})")
    parser.add_argument("--baseline", default="bge-m3", help="비교 기준 후보")
    parser.add_argument("--distractors", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
