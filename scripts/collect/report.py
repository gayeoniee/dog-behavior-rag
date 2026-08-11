"""코퍼스 품질 + 커버리지 리포트.

    uv run python -m scripts.collect.report

합격 기준:
  - 네 축(problem/cause/training/medical) 각각에 문서 존재
  - coverage_questions.yaml의 각 질문 키워드가 코퍼스 어딘가에 존재
  - volatile 소스가 5년 이상 묵지 않았는지
"""

import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .models import CORPUS_PATH

logger = logging.getLogger(__name__)

COVERAGE_PATH = Path("data/coverage_questions.yaml")
AXES = ("problem", "cause", "training", "medical")
VOLATILE_MAX_AGE_YEARS = 5
CHARS_PER_CHUNK = 1500  # 청크 수 추정용 (실제 청킹 전략은 추후 확정)


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(f"{CORPUS_PATH} 없음 — normalize를 먼저 실행하세요")
    with CORPUS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_coverage(docs: list[dict]) -> list[tuple[str, str, bool]]:
    """(axis, question, passed) — 질문 키워드 중 절반 이상이 해당 축 문서에 있으면 통과."""
    if not COVERAGE_PATH.is_file():
        return []
    questions = yaml.safe_load(COVERAGE_PATH.read_text(encoding="utf-8"))

    results = []
    for q in questions:
        axis_docs = [d for d in docs if q["axis"] in d["axis"]]
        blob = " ".join(d["content"].lower() for d in axis_docs)
        keywords = [k.lower() for k in q["keywords"]]
        hits = sum(1 for k in keywords if k in blob)
        passed = bool(axis_docs) and hits >= (len(keywords) + 1) // 2
        results.append((q["axis"], q["question"], passed))
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    docs = load_corpus()

    total_chars = sum(len(d["content"]) for d in docs)
    print("═══ 코퍼스 리포트 ═══")
    print(f"문서 수        : {len(docs)}")
    print(f"총 글자 수     : {total_chars:,}")
    print(f"예상 청크 수   : ~{total_chars // CHARS_PER_CHUNK}")

    print("\n── 축(axis) 분포 ──")
    axis_counts = Counter(a for d in docs for a in d["axis"])
    missing_axes = []
    for axis in AXES:
        count = axis_counts.get(axis, 0)
        marker = "" if count else "  ⚠️ 비어 있음 — 소스를 더 찾아야 함"
        print(f"  {axis:<10}: {count}{marker}")
        if not count:
            missing_axes.append(axis)

    print("\n── 언어 / 방법론 / 권위 ──")
    for field in ("language", "methodology", "authority_tier"):
        counts = Counter(d[field] for d in docs)
        print(f"  {field}: {dict(counts)}")

    bad_method = [d["id"] for d in docs if d["methodology"] in ("aversive", "unknown")]
    if bad_method:
        print(f"  ⚠️ aversive/unknown 문서 {len(bad_method)}건 — 소스 재검토: {bad_method[:5]}")

    print("\n── 최신성 ──")
    year_counts = Counter(d["published_at"] for d in docs)
    print(f"  발행연도 분포: {dict(sorted(year_counts.items()))}")
    now_year = datetime.now(UTC).year
    stale = sorted(
        {
            d["source_id"]
            for d in docs
            if d["volatility"] == "volatile"
            and now_year - d["published_at"] > VOLATILE_MAX_AGE_YEARS
        }
    )
    if stale:
        print(f"  ⚠️ volatile인데 {VOLATILE_MAX_AGE_YEARS}년 초과: {stale} — 최신판 확인 필요")

    print("\n── 커버리지 질문 ──")
    coverage = check_coverage(docs)
    failed = [c for c in coverage if not c[2]]
    for axis, question, passed in coverage:
        print(f"  [{'PASS' if passed else 'FAIL'}] ({axis}) {question}")

    print("\n═══ 판정 ═══")
    ok = not missing_axes and not failed
    if missing_axes:
        print(f"  ✗ 빈 축: {missing_axes}")
    if failed:
        print(f"  ✗ 커버리지 실패 {len(failed)}건 — 해당 축 소스 보강 필요")
    if ok:
        print("  ✓ 네 축 모두 문서 존재 + 커버리지 질문 전체 통과")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
