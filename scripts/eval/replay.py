"""멀티턴 대화 재현 — 평가셋이 못 보는 층을 본다.

    uv run python -m scripts.eval.replay <라벨>
    uv run python -m scripts.eval.replay <라벨> --file data/replay_night_barking.yaml

`retrieval_report`는 **단발 질문의 검색·선별까지만** 잰다. 답변 문장도 멀티턴도
보지 않는다. 실제로 그 사각지대에서 세 가지가 무너져 있었고(지시문 누출, 축자 반복,
후속 질문이 되묻기로 흘러감) 전부 사용자가 직접 대화해 보고서야 발견됐다.

이 스크립트는 대화를 순서대로 재생하며 **기계적으로 확인 가능한 것만** 검사한다:

  지시문 누출   프롬프트 폼의 슬롯 설명이 답변에 그대로 찍혔는가
  반복          이전 답변과 얼마나 겹치는가 (부분 반복을 놓치지 않으려고 비율로 본다)
  고정 문구     같은 라벨 줄(주의점 등)이 매 턴 같은 말인가

**완전 일치만 보면 놓친다.** 처음엔 그렇게 만들었는데, 진단·단계·주의점을 재사용하고
단계 하나만 바꾼 답변이 검사를 통과했다. 사람 눈에는 같은 답인데 통과한 것이다.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import yaml

API_URL = "http://127.0.0.1:8000/api/v1/chat"
DEFAULT_CONVERSATION = Path("data/replay_conversations.yaml")
RESULT_DIR = Path("data/replay_results")

# 프롬프트(rag_service.py)의 폼 슬롯 설명. 답변에 그대로 있으면 모델이 지시를 베낀 것이다.
LEAK_MARKERS = (
    "가능성 있는 원인들을 한 문장으로",
    "증상을 한 문장으로 되짚고",
    "원인을 가르는 질문",
    "역효과 행동 하나와 그 이유",
    "원인 한 문장",
    "타이밍이나 흔한 실수",
)

# 폼 라벨 — 이 줄만 뽑아 턴끼리 비교한다.
FORM_LABELS = ("진단", "주의점", "포인트", "가능한 원인", "확인이 필요해요")

REPEAT_WARN = 0.5
"""이전 답변과 이만큼 겹치면 경고. 문장 단위 비율이다.

0.5인 이유: 폼이 고정돼 있어 라벨 줄("이렇게 해보세요")은 항상 같으므로 겹침이
0이 될 수 없다. 실제로 문제가 된 사례는 진단·단계·주의점을 재사용하고 단계 하나만
바꾼 것으로 0.7 언저리였고, 정상적인 후속 답변은 0.3 아래였다.
"""


@dataclass
class TurnResult:
    question: str
    answer: str
    coverage: str
    sources: int
    seconds: float
    repeat_ratio: float
    repeat_of: int | None
    leaks: list[str]


def sentences(text: str) -> list[str]:
    """문장·줄 단위로 쪼갠다. 번호 목록이 폼의 일부라 줄 단위가 자연스럽다."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(line)
    return out


def overlap(a: str, b: str) -> float:
    """b에 대해 a가 얼마나 겹치는가 (0~1). 문장 단위로 보되 표현이 조금 달라도 잡는다."""
    sa, sb = sentences(a), sentences(b)
    if not sa or not sb:
        return 0.0
    matched = 0
    for s in sa:
        if any(SequenceMatcher(None, s, t).ratio() > 0.85 for t in sb):
            matched += 1
    return matched / len(sa)


def label_line(text: str, label: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{label}:") or stripped.startswith(f"{label}："):
            return stripped
    return None


def ask(question: str, history: list[dict], timeout: float) -> tuple[dict, float]:
    body = json.dumps({"question": question, "history": history}).encode()
    request = urllib.request.Request(
        API_URL, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload, time.perf_counter() - started


def run(name: str, turns: list[str], label: str, timeout: float) -> list[TurnResult]:
    history: list[dict] = []
    results: list[TurnResult] = []

    for i, question in enumerate(turns, start=1):
        payload, seconds = ask(question, history, timeout)
        answer = payload.get("answer") or ""

        best_ratio, best_of = 0.0, None
        for j, previous in enumerate(results, start=1):
            ratio = overlap(answer, previous.answer)
            if ratio > best_ratio:
                best_ratio, best_of = ratio, j

        results.append(
            TurnResult(
                question=question,
                answer=answer,
                coverage=payload.get("coverage", "?"),
                sources=len(payload.get("sources", [])),
                seconds=seconds,
                repeat_ratio=best_ratio,
                repeat_of=best_of if best_ratio >= REPEAT_WARN else None,
                leaks=[m for m in LEAK_MARKERS if m in answer],
            )
        )
        print(f"  [{i}/{len(turns)}] {question[:34]:36s} {payload.get('coverage','?'):12s}"
              f" 겹침 {best_ratio:.0%}")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

    return results


def report(name: str, results: list[TurnResult]) -> None:
    print(f"\n{'=' * 78}\n[{name}]  {len(results)}턴\n")
    for i, r in enumerate(results, start=1):
        flags = []
        if r.leaks:
            flags.append(f"지시문 누출({r.leaks[0][:12]}…)")
        if r.repeat_of:
            flags.append(f"{r.repeat_of}번과 {r.repeat_ratio:.0%} 겹침")
        mark = "  ⚠ " + " / ".join(flags) if flags else ""
        print(f"[{i}] {r.question}")
        print(f"    {r.coverage} · 근거 {r.sources}건 · {r.seconds:.1f}s{mark}")

    # 라벨별로 매 턴 같은 말을 쓰는지 — 폼이 강제하는 자리를 관성으로 채우는 신호다.
    print("\n── 고정 문구 검사 ──")
    for label in FORM_LABELS:
        lines = [ln for ln in (label_line(r.answer, label) for r in results) if ln]
        if len(lines) < 2:
            continue
        unique = len(set(lines))
        if unique == 1:
            note = "  ← 매 턴 같은 말"
        elif unique < len(lines):
            note = "  ← 일부 턴이 같음"
        else:
            note = ""
        print(f"  {label:12s} {len(lines)}턴 중 서로 다른 문장 {unique}개{note}")

    leaks = sum(1 for r in results if r.leaks)
    repeats = sum(1 for r in results if r.repeat_of)
    print(f"\n── 요약 ──\n  지시문 누출 {leaks}건 · 반복 경고 {repeats}건 "
          f"(임계 {REPEAT_WARN:.0%})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="결과 파일 이름 (data/replay_results/<label>.json)")
    parser.add_argument("--file", type=Path, default=DEFAULT_CONVERSATION)
    parser.add_argument("--name", help="대화 이름. 생략하면 파일의 첫 대화")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"✗ {args.file} 가 없습니다", file=sys.stderr)
        return 1
    conversations = yaml.safe_load(args.file.read_text(encoding="utf-8"))

    chosen = None
    for c in conversations:
        if args.name is None or c["name"] == args.name:
            chosen = c
            break
    if chosen is None:
        print(f"✗ 대화 {args.name!r} 를 찾을 수 없습니다", file=sys.stderr)
        return 1

    print(f"대화: {chosen['name']} — {chosen.get('note', '')}")
    try:
        results = run(chosen["name"], chosen["turns"], args.label, args.timeout)
    except urllib.error.URLError as exc:
        print(f"✗ API에 연결할 수 없습니다 ({API_URL}): {exc}", file=sys.stderr)
        print("  uv run uvicorn app.main:app 로 서버를 먼저 띄우세요", file=sys.stderr)
        return 1

    report(chosen["name"], results)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"{args.label}.json"
    path.write_text(
        json.dumps(
            {"conversation": chosen["name"], "turns": [vars(r) for r in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n✓ 저장: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
