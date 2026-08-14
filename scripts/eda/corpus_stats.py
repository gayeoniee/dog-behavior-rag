"""코퍼스 EDA — 청킹 전략을 **추측이 아니라 통계로** 정하기 위한 스크립트.

    uv run python -m scripts.eda.corpus_stats                 # 기본 리포트
    uv run python -m scripts.eda.corpus_stats --sample 2      # 실제 본문 샘플까지
    uv run python -m scripts.eda.corpus_stats --tokens        # 진짜 토큰 수 (모델 필요)
    uv run python -m scripts.eda.corpus_stats --size 800      # 다른 청크 크기로 시뮬레이션

이 프로젝트의 청크 크기 1,200자에는 **근거가 없다.** 그럴듯해서 골랐다. 이 스크립트는
그 결정을 되돌아볼 재료를 만든다:

  - 문서가 얼마나 긴가 → 자를 필요가 있는가
  - 문단이 얼마나 긴가 → 청크 크기가 문단보다 작으면 문단이 매번 쪼개진다
  - 줄 구조가 어떤가 → PDF(문단형)와 HTML(리스트형)은 자르는 방식이 달라야 한다
  - 임베딩 모델 입력 한계 대비 얼마나 쓰고 있는가

**DB도 torch도 필요 없다.** `data/processed/corpus.jsonl`만 읽는다 (--tokens 제외).
수집만 끝난 상태에서도 돌아간다.
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.services.chunking import ChunkConfig, clean_for_chunking, split_text

CORPUS_PATH = Path("data/processed/corpus.jsonl")

# bge-m3가 받을 수 있는 최대 입력.
BGE_M3_MAX_TOKENS = 8192

# **실제로 적용되는 상한은 이쪽이다.** `Settings.embedding_max_seq_length`가 1024로
# 잘라 쓴다(CPU 메모리 스파이크를 막으려고). 이걸 넘는 청크는 뒤가 **조용히 잘린다** —
# 에러도 경고도 없이 없는 셈이 되므로, 모델 상한이 아니라 이 값과 비교해야 한다.
EFFECTIVE_MAX_TOKENS = 1024

# 문장이 제대로 끝났는지 판정. 청크 경계 품질을 보는 데 쓴다.
_SENTENCE_END = re.compile(r"[.!?。？！][\"')\]]?$")

# 논문 섹션 제목처럼 보이는 줄 (짧고, 마침표로 안 끝나고, 대문자로 시작).
_HEADING_LIKE = re.compile(r"^[A-Z0-9][^.!?]{2,60}$")


@dataclass(slots=True)
class Doc:
    source_id: str
    title: str
    content: str
    doc_type: str
    """`study`(논문) | `guide`(실무 가이드). corpus.jsonl에는 없고 여기서 계산한다 —
    load_corpus.derive_doc_type과 같은 규칙(source_id가 pmc-로 시작하면 study)."""


def load_docs(path: Path) -> list[Doc]:
    if not path.is_file():
        print(f"✗ {path} 가 없습니다", file=sys.stderr)
        print("  scripts.collect.normalize 를 먼저 실행하세요", file=sys.stderr)
        raise SystemExit(1)
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            source_id = d.get("source_id") or ""
            docs.append(
                Doc(
                    source_id=source_id,
                    title=d.get("title", ""),
                    content=d.get("content", ""),
                    doc_type="study" if source_id.startswith("pmc-") else "guide",
                )
            )
    return docs


# ── 통계 도우미 ──────────────────────────────────────────────────────


def pct(values: list[int], p: float) -> int:
    """백분위. 평균만 보면 긴 문서 하나에 끌려가므로 분포를 본다."""
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(int(len(ordered) * p), len(ordered) - 1)
    return ordered[idx]


def describe(values: list[int], unit: str = "자") -> str:
    if not values:
        return "(없음)"
    return (
        f"최소 {min(values):,} · p25 {pct(values, 0.25):,} · "
        f"중앙 {pct(values, 0.5):,} · p75 {pct(values, 0.75):,} · "
        f"p95 {pct(values, 0.95):,} · 최대 {max(values):,}{unit}"
    )


def histogram(values: list[int], bins: list[int], width: int = 40) -> list[str]:
    """텍스트 막대그래프. 숫자만 보면 분포 모양이 안 보인다.

    `bins`는 경계값 목록이고 마지막 구간은 "그 이상"이다.
    """
    counts = [0] * (len(bins) + 1)
    for v in values:
        placed = False
        for i, edge in enumerate(bins):
            if v < edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    top = max(counts) or 1
    labels = [f"< {b:,}" for b in bins] + [f">= {bins[-1]:,}"]
    lines = []
    for label, count in zip(labels, counts, strict=True):
        bar = "█" * round(count / top * width)
        share = count / len(values) * 100 if values else 0
        lines.append(f"  {label:>12}  {bar:<{width}} {count:>6,} ({share:4.1f}%)")
    return lines


# ── 분석 단위 ────────────────────────────────────────────────────────


def paragraphs(text: str) -> list[str]:
    """빈 줄로 나눈 문단. 재귀 분할이 **가장 먼저 쓰는 구분자**라 중요하다."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def lines_of(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def heading_ratio(text: str) -> float:
    """섹션 제목처럼 보이는 줄의 비율.

    높으면 문서에 구조가 살아 있다는 뜻이고, 그러면 **제목 기준으로 자르는
    전략(구조 기반 청킹)**을 쓸 수 있다. 낮으면 그런 정보가 추출 과정에서
    이미 날아간 것이다.
    """
    ls = lines_of(text)
    if not ls:
        return 0.0
    return sum(1 for line in ls if _HEADING_LIKE.match(line)) / len(ls)


def ends_cleanly(chunk: str) -> bool:
    """청크가 문장부호로 끝났는가.

    ⚠️ **이 지표 하나만 보면 과장된다.** 가이드 문서는 HTML 목록에서 뽑혀서
    "1. 간식을 코앞에" 처럼 마침표 없이 끝나는 줄이 많은데, 그건 잘린 게 아니라
    원래 그런 문장이다. 그래서 아래 `ends_mid_sentence`를 같이 본다.
    """
    return bool(_SENTENCE_END.search(chunk.rstrip()))


def ends_mid_sentence(chunk: str) -> bool:
    """**명백히** 문장 중간에서 끊겼는가.

    소문자나 쉼표로 끝나면 뒤에 이어질 말이 있었다는 뜻이다. 목록 항목·제목은
    보통 대문자나 숫자로 끝나므로 여기 걸리지 않는다. `ends_cleanly`의 여집합보다
    좁지만 **애매함이 없다** — 지표는 넓고 흐린 것보다 좁고 확실한 게 낫다.
    """
    stripped = chunk.rstrip()
    return bool(stripped) and (stripped[-1].islower() or stripped[-1] == ",")


# ── 리포트 ───────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'━' * 78}\n▌ {title}\n")


def report_overview(docs: list[Doc]) -> None:
    section("1. 문서 개요 — 무엇이 얼마나 있나")
    lengths = [len(d.content) for d in docs]
    by_type = Counter(d.doc_type for d in docs)
    by_source = Counter(d.source_id.split("-")[0] for d in docs)

    print(f"  문서 {len(docs):,}건 · 총 {sum(lengths):,}자")
    print(f"  종류    논문 {by_type['study']:,} · 가이드 {by_type['guide']:,}")
    print(f"  출처    {', '.join(f'{k} {v}' for k, v in by_source.most_common())}")
    print(f"\n  문서 길이  {describe(lengths)}")
    print()
    for line in histogram(lengths, [2_000, 10_000, 20_000, 40_000, 60_000]):
        print(line)

    print("\n  ── 종류별 길이 ──")
    for doc_type in ("study", "guide"):
        sub = [len(d.content) for d in docs if d.doc_type == doc_type]
        if sub:
            print(f"  {doc_type:<8} {describe(sub)}")


def report_structure(docs: list[Doc]) -> None:
    """청킹 전략을 정하는 데 **가장 중요한 절**이다.

    청크 크기는 문단 길이와 비교해야 의미가 있다. 문단보다 작으면 문단이 매번
    쪼개지고, 문단보다 훨씬 크면 무관한 문단들이 한 청크에 섞인다.
    """
    section("2. 문서 구조 — 무엇을 기준으로 자를 수 있나")

    for doc_type, label in (("study", "논문"), ("guide", "가이드")):
        subset = [d for d in docs if d.doc_type == doc_type]
        if not subset:
            continue
        para_lens = [len(p) for d in subset for p in paragraphs(d.content)]
        line_lens = [len(line) for d in subset for line in lines_of(d.content)]
        paras_per_doc = [len(paragraphs(d.content)) for d in subset]
        headings = [heading_ratio(d.content) for d in subset]

        print(f"  ── {label} {len(subset)}건 ──")
        print(f"  문단 길이      {describe(para_lens)}")
        print(f"  줄 길이        {describe(line_lens)}")
        print(f"  문서당 문단 수  {describe(paras_per_doc, unit='개')}")
        print(f"  제목형 줄 비율  {statistics.mean(headings):.1%}")
        print()
        for line in histogram(para_lens, [200, 500, 1_000, 1_500, 3_000]):
            print(line)
        print()


def report_chunking(docs: list[Doc], config: ChunkConfig) -> None:
    section(f"3. 청킹 시뮬레이션 — size={config.size} overlap={config.overlap}")

    all_chunks: list[str] = []
    per_doc: list[int] = []
    cleaned_removed: list[int] = []
    for d in docs:
        cleaned = clean_for_chunking(d.content)
        cleaned_removed.append(len(d.content) - len(cleaned))
        chunks = split_text(d.content, config)
        all_chunks.extend(chunks)
        per_doc.append(len(chunks))

    lengths = [len(c) for c in all_chunks]
    clean_ends = sum(1 for c in all_chunks if ends_cleanly(c))
    broken = sum(1 for c in all_chunks if ends_mid_sentence(c))

    print(f"  청크 {len(all_chunks):,}개 · 문서당 {describe(per_doc, unit='개')}")
    print(f"  청크 길이   {describe(lengths)}")
    print(f"  전처리 제거 {sum(cleaned_removed):,}자 (머리글·페이지번호·하이픈 줄바꿈)")
    print()
    total = len(all_chunks)
    at_cap = sum(1 for x in lengths if x >= config.size - 5)
    under_min = sum(1 for x in lengths if x < config.min_size)
    print(f"  문장부호로 끝난 청크    {clean_ends:,} / {total:,} ({clean_ends / total:.1%})")
    print(f"  ⚠ 문장 중간에서 끊긴 청크 {broken:,} / {total:,} ({broken / total:.1%})")
    print(f"  상한({config.size}자)에 닿은 청크 {at_cap:,}개")
    print(f"  min_size({config.min_size}자) 미만  {under_min:,}개")
    print()
    for line in histogram(lengths, [200, 500, 800, 1_000, config.size]):
        print(line)


def report_budget(docs: list[Doc], config: ChunkConfig, use_tokenizer: bool) -> None:
    """임베딩 모델은 입력 길이 상한이 있다. 얼마나 쓰고 있는지 봐야 한다."""
    section("4. 임베딩 입력 예산 — 모델 한계 대비 얼마나 쓰고 있나")

    sample = [c for d in docs[:40] for c in split_text(d.content, config)]
    if not sample:
        print("  (청크 없음)")
        return

    if use_tokenizer:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            counts = [len(tok.encode(c)) for c in sample]
            source = "실제 토크나이저 (BAAI/bge-m3)"
        except Exception as exc:  # noqa: BLE001 — 없으면 근사로 떨어진다
            print(f"  ⚠️ 토크나이저를 못 불러와 근사치를 씁니다: {exc}")
            counts = [len(c) // 4 for c in sample]
            source = "근사 (영어 기준 4자 ≈ 1토큰)"
    else:
        counts = [len(c) // 4 for c in sample]
        source = "근사 (영어 기준 4자 ≈ 1토큰). --tokens 로 실제 측정"

    print(f"  표본 {len(sample):,}청크 · 계산 방식: {source}")
    print(f"  토큰 수   {describe(counts, unit='토큰')}")

    p95 = pct(counts, 0.95)
    print(
        f"\n  모델 상한 {BGE_M3_MAX_TOKENS:,}토큰   대비 p95 사용률 "
        f"{p95 / BGE_M3_MAX_TOKENS:5.1%}"
    )
    print(
        f"  설정 상한 {EFFECTIVE_MAX_TOKENS:,}토큰   대비 p95 사용률 "
        f"{p95 / EFFECTIVE_MAX_TOKENS:5.1%}   ← 실제로 적용되는 쪽"
    )

    over = sum(1 for c in counts if c > EFFECTIVE_MAX_TOKENS)
    if over:
        print(f"\n  ⚠️ 설정 상한을 넘는 청크 {over:,}개 — 뒤가 잘려 검색에서 사라진다.")
        print("     EMBEDDING_MAX_SEQ_LENGTH를 올리거나 청크를 줄여야 한다.")
    else:
        print(f"\n  ✓ 잘리는 청크 없음 (최대 {max(counts):,}토큰 < {EFFECTIVE_MAX_TOKENS:,})")
        print("    다만 여유가 있다고 청크를 키우는 건 다른 문제다 — 커지면 한 청크에")
        print("    여러 주제가 섞여 임베딩이 뭉개진다. 맞교환이라 지표로 재야 한다.")


def report_samples(docs: list[Doc], config: ChunkConfig, n: int) -> None:
    """멘토 조언: **데이터 샘플을 직접 눈으로 봐야 한다.** 통계는 모양을 알려주지만
    "이 텍스트가 쓸 만한가"는 읽어봐야 안다."""
    section(f"5. 실제 샘플 — 눈으로 확인 ({n}건)")
    picked = [docs[0]] + [d for d in docs if d.doc_type == "study"][:1]
    for d in picked[:n]:
        chunks = split_text(d.content, config)
        print(f"  ── [{d.doc_type}] {d.title[:60]} ──")
        print(f"     {len(d.content):,}자 → 청크 {len(chunks)}개\n")
        for i in (0, len(chunks) // 2):
            if i < len(chunks):
                body = chunks[i].replace("\n", " ⏎ ")
                print(f"     청크 {i} ({len(chunks[i])}자):")
                print(f"       {body[:300]}…\n")


def report_takeaways(docs: list[Doc], config: ChunkConfig) -> None:
    """숫자를 결정으로 옮기는 절. 출력이 아니라 **판단**을 적는다."""
    section("6. 이 통계가 말하는 것")

    para_lens = [len(p) for d in docs for p in paragraphs(d.content)]
    doc_lens = [len(d.content) for d in docs]
    median_para = pct(para_lens, 0.5)
    over = sum(1 for x in para_lens if x > config.size) / len(para_lens)

    print(f"  · 문서 중앙값이 {pct(doc_lens, 0.5):,}자다. 통째로는 못 넣는다 → 청킹은 필수다.")
    print(f"  · 문단 중앙값은 {median_para:,}자로 청크 크기({config.size}자)보다 작다.")
    print(f"    → 보통은 문단 여러 개가 한 청크에 담긴다. 문단이 쪼개지는 경우는 {over:.1%}뿐이다.")
    print("  · 문단이 훨씬 짧다는 건 **청크를 줄여도 문단은 안 깨진다**는 뜻이기도 하다.")
    print("    작은 청크는 검색이 더 정확해지지만 문맥이 줄어든다. 이 맞교환을 04장에서 잰다.")
    print()
    print("  다음 단계: 이 통계를 근거로 청크 크기 후보를 정하고, 05·06장에서 만든")
    print("  평가셋과 지표로 **실제로 어느 쪽이 더 잘 찾는지** 측정한다.")


def main() -> int:
    parser = argparse.ArgumentParser(description="코퍼스 EDA — 청킹 전략의 근거를 만든다")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--size", type=int, default=ChunkConfig().size, help="청크 크기 시뮬레이션")
    parser.add_argument("--overlap", type=int, default=ChunkConfig().overlap)
    parser.add_argument("--sample", type=int, default=0, help="본문 샘플 N건 출력")
    parser.add_argument(
        "--tokens", action="store_true", help="실제 토크나이저로 토큰 수 측정 (transformers 필요)"
    )
    args = parser.parse_args()

    docs = load_docs(args.corpus)
    config = ChunkConfig(size=args.size, overlap=args.overlap)

    print(f"\n코퍼스 EDA — {args.corpus}")
    report_overview(docs)
    report_structure(docs)
    report_chunking(docs, config)
    report_budget(docs, config, args.tokens)
    if args.sample:
        report_samples(docs, config, args.sample)
    report_takeaways(docs, config)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
