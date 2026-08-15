"""평가셋 문항에 **질문의 넓이**를 라벨링한다.

    uv run python -m scripts.eval.label_breadth --qa data/eval_auto_qa_merged.jsonl
    uv run python -m scripts.eval.label_breadth --qa ... --sample 15   # 라벨 눈으로 확인

**왜 필요한가.** 자동 생성 평가셋을 검수하다가 이런 문항들을 "공허하다"고 버릴 뻔했다:

    "우리 강아지가 불안해 보일 때는 어떻게 해야 하나요?"
    "집에서 어떻게 훈련을 시작해야 하나요?"

**그건 잘못된 판단이었다. 실사용에서는 오히려 저런 질문이 더 흔하다.** 보호자는
"분리불안 시 외출 전 인사 단축이 도움이 되나요?"라고 묻지 않는다.

문제는 질문이 아니라 **채점 방식**이다. 평가셋은 "정답은 이 청크 하나"라고 가정하는데,
넓은 질문은 답이 되는 청크가 코퍼스에 수십 개다. 좁은 질문에만 맞는 가정을 넓은
질문에도 적용하니 부당하게 실패로 세어진다.

    넓은 질문(broad)    실사용 현실성 높음 · 정답 청크 여럿 → **문서 단위로 채점해야**
    좁은 질문(specific) 실사용 현실성 낮음 · 정답 청크 하나 → 청크 단위 채점이 맞음

**그래서 버리지 않고 라벨을 붙인다.** 06장의 지표를 이 라벨로 쪼개 보면
"청크 단위 점수가 낮은 게 검색이 나빠서인지, 질문이 넓어서인지"를 가를 수 있다.

이 프로젝트에는 넓은 질문을 위한 경로가 이미 있다 — `coverage=needs_detail`로
되묻는 것. **넓은 질문은 되묻기가 정답**이고, IR 지표는 그걸 볼 수 없다.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from app.core.config import get_settings
from app.services.llm.base import LLMClient, LLMUnavailableError
from app.services.llm.registry import get_llm

BREADTH_SYSTEM = """How specific is this question from a dog owner?

Answer with exactly one word: BROAD or SPECIFIC.

BROAD — many different pieces of advice would answer it. No particular trigger, \
situation, or behaviour is named.
  "우리 강아지가 불안해 보일 때는 어떻게 해야 하나요?"
  "집에서 어떻게 훈련을 시작해야 하나요?"

SPECIFIC — names a behaviour, trigger, or situation, so only advice about that \
would answer it.
  "손님만 오면 짖는데 어떻게 차분하게 만들 수 있을까요?"
  "혼자 있을 때만 벽을 긁는 이유가 뭔가요?\""""


async def label_one(llm: LLMClient, question: str) -> str:
    """`broad` | `specific` | `unknown`."""
    try:
        raw = await llm.generate(
            f"Question: {question}",
            system=BREADTH_SYSTEM,
            max_tokens=8,
            # 판정이지만 예산이 8토큰이라 숙고를 켜면 답이 눌린다
            # (generate_qa.is_owner_useful 독스트링 참조).
            reasoning=False,
        )
    except LLMUnavailableError:
        return "unknown"
    words = raw.upper().split()
    if "BROAD" in words:
        return "broad"
    if "SPECIFIC" in words:
        return "specific"
    return "unknown"


async def run(args: argparse.Namespace) -> int:
    if not args.qa.is_file():
        print(f"✗ {args.qa} 가 없습니다", file=sys.stderr)
        return 1
    lines = args.qa.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    settings = get_settings()
    llm = get_llm(settings)
    print(f"{args.qa} — {len(rows)}문항 · 모델={llm.name}\n")

    todo = [r for r in rows if not r.get("breadth")]
    for i, row in enumerate(todo, 1):
        row["breadth"] = await label_one(llm, row["question"])
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}", flush=True)

    args.qa.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    counts = Counter(r.get("breadth", "unknown") for r in rows)
    total = len(rows)
    print(f"\n  broad    {counts['broad']:>4} ({counts['broad'] / total:.0%})")
    print(f"  specific {counts['specific']:>4} ({counts['specific'] / total:.0%})")
    if counts["unknown"]:
        print(f"  unknown  {counts['unknown']:>4}")
    print(f"\n✓ 라벨을 {args.qa} 에 기록했다")

    if args.sample:
        _print_sample(rows, args.sample)
    return 0


def _print_sample(rows: list[dict], n: int) -> None:
    """**라벨을 믿기 전에 눈으로 본다.** 판정기는 언제나 틀릴 수 있다."""
    print(f"\n── 라벨 확인용 표본 (각 {n}개) ──")
    for label in ("broad", "specific"):
        print(f"\n  [{label}]")
        for row in [r for r in rows if r.get("breadth") == label][:n]:
            print(f"    · {row['question']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="질문 넓이 라벨링")
    parser.add_argument("--qa", type=Path, default=Path("data/eval_auto_qa_merged.jsonl"))
    parser.add_argument("--sample", type=int, default=0, help="라벨별로 N개 출력해 검수")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
