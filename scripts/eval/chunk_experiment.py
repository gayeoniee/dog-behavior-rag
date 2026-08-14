"""청킹 전략 A/B 실험 — 청크 크기를 바꾸면 검색이 좋아지는가.

    uv run python -m scripts.eval.chunk_experiment --sizes 800,1200 --distractors 30
    uv run python -m scripts.eval.chunk_experiment --sizes 400,800,1200,2000 --distractors 150

**이 실험이 어려운 진짜 이유: 정답 라벨이 청킹에 묶여 있다.**

평가셋(`eval_auto_qa*.jsonl`)의 정답은 `chunk_id`, 즉 **1,200자로 자른 그 청크**다.
400자로 다시 자르면 그 청크는 세상에 없다. 라벨이 통째로 무의미해진다.

**해법은 정답을 "청크"가 아니라 "문서 안의 글자 구간"으로 다시 정의하는 것이다.**

    정답 청크가 문서의 12,000~13,200번째 글자였다  ← 이건 청킹과 무관한 사실
    새로 자른 청크가 12,400~12,800을 덮는다        ← 겹치므로 정답으로 인정

정보검색에서 passage-level relevance를 다루는 표준 방식이고, 이게 없으면 청킹
실험 자체가 성립하지 않는다.

**DB를 건드리지 않는다.** 청킹·임베딩·검색을 전부 메모리에서 한다. 실제 적재
(`load_corpus`)는 코퍼스 전체를 다시 임베딩해야 해서 설정 하나당 GPU 15분이 들고,
그동안 기존 인덱스를 날려야 한다. 실험 단계에서 그건 너무 비싸고 위험하다.

**모든 설정이 완전히 같은 조건에서 겨루게 한다:**

  - 같은 문서 집합 (정답 문서 + 방해 문서, seed 고정)
  - 같은 질의 — 질의 재작성을 **한 번만 하고 캐시**해서 전 설정이 같은 검색어를 쓴다.
    설정마다 재작성하면 LLM 출력이 흔들려서 청킹의 효과와 뒤섞인다
  - 같은 지표 (hit rate, MRR — 06장)
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Chunk, Document
from app.db.session import create_engine, create_session_factory
from app.services.chunking import ChunkConfig, clean_for_chunking, split_text
from app.services.embeddings.registry import get_embedder
from app.services.llm.registry import get_llm
from app.services.query_rewrite import QueryRewriter

CORPUS_PATH = Path("data/processed/corpus.jsonl")
DEFAULT_QA = Path("data/eval_auto_qa.jsonl")
REWRITE_CACHE = Path("data/eval_results/rewrites.json")
RESULT_PATH = Path("data/eval_results/chunk-experiment.json")

OVERLAP_THRESHOLD = 0.5
"""검색된 청크와 정답 구간이 이만큼 겹치면 정답으로 친다.

**기준 길이는 둘 중 짧은 쪽이다.** 이게 이 실험의 공정성을 좌우한다.

처음에는 정답 구간 길이로 나눴다가 실험을 통째로 망칠 뻔했다. 정답 청크의 중앙값이
1,061자인데, **400자 청크는 아무리 정확히 맞아도 400/1061 = 38%밖에 못 덮는다.**
즉 임계 50%를 원리상 넘을 수 없어 작은 크기는 검색 성능과 무관하게 0점이 된다.
"작은 청크는 원리상 정답이 될 수 없어 크기 비교가 크기 자랑이 된다"고 이 주석에
적어놓고 정확히 그렇게 만들었다.

짧은 쪽으로 나누면 양쪽이 대칭이 된다:

    400자 청크가 정답 구간 안에 온전히 들어감   → 400/400 = 100%  정답
    2000자 청크가 정답 구간을 통째로 포함        → 1200/1200 = 100% 정답
    400자 청크가 정답 구간을 50자만 스침         → 50/400 = 12.5%   오답

**교훈: 지표가 특정 조건에서 원리상 도달 불가능한 값을 요구하는지 확인할 것.**
실험 결과는 그럴듯한 표로 나오기 때문에 이걸 놓치면 아무도 모른다.
"""


@dataclass(slots=True)
class Gold:
    question: str
    document_id: int
    start: int
    end: int
    """정답 구간 — `clean_for_chunking` 적용 후 문서 본문에서의 글자 위치."""


@dataclass(slots=True)
class Passage:
    document_id: int
    text: str
    start: int
    end: int


def locate(cleaned: str, chunk: str, cursor: int = 0) -> tuple[int, int]:
    """청크가 문서의 몇 번째 글자인지 (근사) 찾는다.

    **청크는 원문의 부분 문자열이 아니다.** `split_text`가 분할 단위를 `"\\n"`으로
    이어 붙이는데 원문에서는 그 자리가 `"\\n\\n"`일 수 있다. 그래서 청크 전체로
    검색하면 못 찾는다 (처음에 이걸로 13건 전부 실패했다).

    대신 **청크 안에서 가장 긴 줄**을 닻으로 쓴다. 그 줄은 분할 단위 하나이므로
    원문에 그대로 있다. 닻의 위치에서 청크 내 오프셋만큼 되돌리면 시작점이 나온다.
    """
    lines = [line for line in chunk.split("\n") if line.strip()]
    anchor = max(lines, key=len) if lines else chunk[:80]

    idx = cleaned.find(anchor, cursor)
    if idx == -1:
        idx = cleaned.find(anchor)
    if idx == -1:
        return -1, -1

    start = max(0, idx - chunk.find(anchor))
    return start, start + len(chunk)


def locate_chunks(cleaned: str, chunks: list[str]) -> list[tuple[int, int]]:
    """각 청크의 구간. 앞에서부터 커서를 옮기며 찾는다.

    겹침(overlap) 때문에 같은 문장이 두 청크에 나타나므로, 매번 처음부터 찾으면
    뒤쪽 청크가 앞쪽 위치로 잘못 잡힌다.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        start, end = locate(cleaned, chunk, cursor)
        spans.append((start, end))
        if start >= 0:
            cursor = max(cursor, start + max(1, len(chunk) // 2))
    return spans


def covers(gold: Gold, passage: Passage, threshold: float) -> bool:
    """이 청크와 정답 구간이 충분히 겹치는가.

    **짧은 쪽 길이로 나눈다** — 이유는 `OVERLAP_THRESHOLD` 주석 참조.
    """
    if passage.document_id != gold.document_id or passage.start < 0:
        return False
    overlap = min(gold.end, passage.end) - max(gold.start, passage.start)
    if overlap <= 0:
        return False
    shorter = max(1, min(gold.end - gold.start, passage.end - passage.start))
    return overlap / shorter >= threshold


def build_passages(docs: list[dict], size: int, overlap: int) -> list[Passage]:
    config = ChunkConfig(size=size, overlap=overlap)
    passages: list[Passage] = []
    for doc in docs:
        cleaned = clean_for_chunking(doc["content"])
        chunks = split_text(doc["content"], config)
        for text, (start, end) in zip(chunks, locate_chunks(cleaned, chunks), strict=True):
            passages.append(Passage(doc["_id"], text, start, end))
    return passages


async def load_golds(qa_path: Path, factory) -> tuple[list[Gold], list[dict]]:
    """평가셋의 (question, chunk_id)를 (question, 문서 내 글자 구간)으로 바꾼다.

    청크 본문은 DB에서 가져온다 — jsonl의 `chunk_excerpt`는 앞 300자뿐이라
    구간의 끝을 알 수 없다.
    """
    lines = qa_path.read_text(encoding="utf-8").splitlines()
    pairs = [json.loads(line) for line in lines if line.strip()]
    if not pairs:
        raise SystemExit(f"✗ {qa_path} 가 비어 있습니다")

    async with factory() as session:
        rows = (
            await session.execute(
                select(Chunk.id, Chunk.content).where(
                    Chunk.id.in_([p["chunk_id"] for p in pairs])
                )
            )
        ).all()
        hashes = (await session.execute(select(Document.id, Document.content_hash))).all()
    chunk_text = {r.id: r.content for r in rows}
    id_by_hash = {r.content_hash: r.id for r in hashes}

    docs = load_corpus(id_by_hash)
    docs_by_id = {d["_id"]: d for d in docs}
    golds: list[Gold] = []
    missing = 0
    for p in pairs:
        text = chunk_text.get(p["chunk_id"])
        doc = docs_by_id.get(p["document_id"])
        if text is None or doc is None:
            missing += 1
            continue
        start, end = locate(clean_for_chunking(doc["content"]), text)
        if start < 0:
            missing += 1
            continue
        golds.append(Gold(p["question"], p["document_id"], start, end))

    if missing:
        print(f"  ⚠️ 정답 구간을 못 찾은 문항 {missing}건 — 제외했다", file=sys.stderr)
    return golds, docs


def load_corpus(id_by_hash: dict[str, int]) -> list[dict]:
    """corpus.jsonl 을 읽고 각 문서에 **DB의 documents.id**를 붙인다.

    **줄 번호를 id로 쓰면 안 된다.** 처음에 "jsonl 1번째 줄 = id 1"로 가정했다가
    정답 구간을 13건 전부 못 찾았다. `load_corpus.py`는 파일 순서대로 넣지 않는다
    (이미 있는 문서는 content_hash로 건너뛰므로 순번이 밀린다).

    그래서 추측하지 않고 **이미 존재하는 고유키(content_hash)로 잇는다.**
    """
    if not CORPUS_PATH.is_file():
        raise SystemExit(f"✗ {CORPUS_PATH} 가 없습니다")
    docs = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            doc_id = id_by_hash.get(d.get("content_hash", ""))
            if doc_id is None:
                continue  # DB에 없는 문서 (적재 이후 수집분 등)
            d["_id"] = doc_id
            docs.append(d)
    return docs


async def cached_rewrites(questions: list[str], settings) -> dict[str, str]:
    """질의 재작성을 **한 번만** 하고 파일에 남긴다.

    설정마다 재작성하면 LLM 출력이 실행마다 달라져서, 청킹을 바꾼 효과인지
    검색어가 바뀐 효과인지 구분할 수 없게 된다. 실험에서 통제해야 할 변수다.
    """
    cache: dict[str, str] = {}
    if REWRITE_CACHE.is_file():
        cache = json.loads(REWRITE_CACHE.read_text(encoding="utf-8"))

    todo = [q for q in questions if q not in cache]
    if todo:
        print(f"  질의 재작성 {len(todo)}건 (캐시 {len(cache)}건 재사용)")
        rewriter = QueryRewriter(get_llm(settings), enabled=True)
        for i, q in enumerate(todo, 1):
            cache[q] = await rewriter.rewrite(q)
            print(f"    [{i}/{len(todo)}] {cache[q][:56]}", flush=True)
        REWRITE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        REWRITE_CACHE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        print(f"  질의 재작성: 캐시 {len(cache)}건 전부 재사용")
    return cache


def score(
    golds: list[Gold],
    passages: list[Passage],
    query_vectors: list[list[float]],
    passage_vectors: list[list[float]],
    top_k: int,
    threshold: float,
) -> dict:
    import numpy as np

    P = np.array(passage_vectors, dtype="float32")
    Q = np.array(query_vectors, dtype="float32")
    # 임베더가 normalize_embeddings=True 라서 내적이 곧 코사인 유사도다.
    sims = Q @ P.T

    ranks: list[int] = []
    for i, gold in enumerate(golds):
        order = np.argsort(-sims[i])[:top_k]
        rank = 0
        for pos, idx in enumerate(order, 1):
            if covers(gold, passages[idx], threshold):
                rank = pos
                break
        ranks.append(rank)

    return {
        "hit@1": sum(1 for r in ranks if r == 1) / len(ranks),
        "hit@5": sum(1 for r in ranks if 0 < r <= 5) / len(ranks),
        "hit@10": sum(1 for r in ranks if 0 < r <= 10) / len(ranks),
        "mrr": statistics.mean(1.0 / r if r else 0.0 for r in ranks),
        "ranks": ranks,
    }


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

    gold_doc_ids = {g.document_id for g in golds}
    gold_docs = [d for d in docs if d["_id"] in gold_doc_ids]
    others = [d for d in docs if d["_id"] not in gold_doc_ids]
    random.Random(args.seed).shuffle(others)
    corpus = gold_docs + others[: args.distractors]

    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"\n문항 {len(golds)} · 문서 {len(corpus)}건")
    print(f"  (정답 문서 {len(gold_docs)} + 방해 문서 {args.distractors})")
    print(f"청크 크기 후보: {sizes} · 겹침 {args.overlap} · 정답 인정 겹침 {args.threshold:.0%}\n")

    # 비용을 먼저 보여준다. 임베딩이 이 실험의 전부다.
    for size in sizes:
        n = len(build_passages(corpus, size, args.overlap))
        print(f"  size={size:>5} → 청크 {n:,}개")
    print()

    embedder = get_embedder(settings)
    await embedder.warmup()
    rewrites = await cached_rewrites([g.question for g in golds], settings)
    query_vectors = await embedder.embed([rewrites[g.question] for g in golds])

    results: dict[int, dict] = {}
    for size in sizes:
        passages = build_passages(corpus, size, args.overlap)
        print(f"  size={size} — 청크 {len(passages):,}개 임베딩 중…", flush=True)
        vectors = await embedder.embed([p.text for p in passages])
        results[size] = score(
            golds, passages, query_vectors, vectors, args.top_k, args.threshold
        )
        results[size]["chunks"] = len(passages)
        r = results[size]
        print(f"    hit@5 {r['hit@5']:.1%} · MRR {r['mrr']:.3f}", flush=True)
        # **크기 하나가 끝날 때마다 저장한다.** 끝에서 한 번만 쓰면 중간에 죽을 때
        # 몇십 분치 GPU 시간이 통째로 날아간다 — 실제로 3/4까지 끝낸 실행을 그렇게
        # 잃었다. generate_qa에서 겪은 것과 같은 실수를 여기서 반복했다.
        _save(results, args, len(golds))

    _report(results, [s for s in sizes if s in results], golds)
    print(f"\n✓ 저장: {RESULT_PATH}")
    return 0


def _save(results: dict[int, dict], args: argparse.Namespace, questions: int) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "distractors": args.distractors,
                "questions": questions,
                "overlap": args.overlap,
                "results": {str(k): v for k, v in results.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _report(results: dict, sizes: list[int], golds: list[Gold]) -> None:
    print("\n" + "═" * 68)
    print(f"  {'크기':>6} {'청크수':>9} {'hit@1':>8} {'hit@5':>8} {'hit@10':>8} {'MRR':>8}")
    print("─" * 68)
    for size in sizes:
        r = results[size]
        print(
            f"  {size:>6} {r['chunks']:>9,} {r['hit@1']:>8.1%} {r['hit@5']:>8.1%} "
            f"{r['hit@10']:>8.1%} {r['mrr']:>8.3f}"
        )
    best = max(sizes, key=lambda s: results[s]["mrr"])
    print(f"\n  MRR 최고: size={best}")
    print(f"  문항 {len(golds)}개 기준이다 — 1문항이 {1 / len(golds):.1%}라는 걸 기억할 것.")


def main() -> int:
    parser = argparse.ArgumentParser(description="청킹 전략 A/B 실험")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--sizes", default="800,1200", help="쉼표로 구분한 청크 크기")
    parser.add_argument("--overlap", type=int, default=ChunkConfig().overlap)
    parser.add_argument("--distractors", type=int, default=30, help="방해 문서 수")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=OVERLAP_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
