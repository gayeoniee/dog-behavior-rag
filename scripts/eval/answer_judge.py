"""답변 품질 평가 (LLM as a judge) — 검색이 아니라 **답변 자체**를 잰다.

    uv run python -m scripts.eval.answer_judge --calibrate
    uv run python -m scripts.eval.answer_judge --n 21

**왜 필요한가.** 01~09장에서 검색·청킹·임베딩·하이브리드를 전부 숫자로 쟀는데
**답변 자체는 한 번도 안 쟀다.** 사람이 읽어보는 것뿐이었다.

06장에서 단서가 나왔다 — **문서 단위 hit@1이 59.0%**다. 질문의 6할은 정답 문서를
바로 1위로 올린다. 그런데 체감 품질이 그보다 낮다면 병목은 검색이 아니라
**근거 선별이나 답변 생성**이다. 그걸 확인할 도구가 지금 없다.

## 무엇을 재는가

이 프로젝트에서 가장 중요한 건 **근거 충실성(faithfulness)** 이다. RAG의 존재
이유가 "자료에 없으면 말하지 않는다"인데, 답변이 근거에 없는 내용을 말하면
검색을 아무리 잘해도 의미가 없다.

    충실성   답변의 모든 주장이 제시된 근거에 있는가   ← 환각 탐지
    응답성   질문에 실제로 답하는가                    ← 동문서답 탐지

**한 호출에 하나씩 묻는다.** 05장에서 "작은 모델은 두 가지를 묶으면 하나를
버린다"를 비싸게 배웠다.

## 판정기를 믿기 전에 판정기를 잰다

`--calibrate`가 **정답을 아는 쌍**으로 판정기를 채점한다:

    양성   실제 답변 + 그 답변이 쓴 근거      → 충실(YES)해야 한다
    음성   실제 답변 + **다른 질문의 근거**   → 불충실(NO)이어야 한다

음성을 만드는 방법이 핵심이다. 근거를 바꿔치기하면 **답변 내용은 그대로인데
근거와의 관계만 끊긴다** — 충실성 판정기가 정확히 잡아야 하는 상황이다.
못 잡으면 그 판정기는 근거를 안 보고 "그럴듯한가"만 보고 있는 것이다.

`scripts.eval.calibrate_judge`와 같은 설계다. 그때 gemma가 좋은 쌍의 88%를
버리고 있던 걸 이 방법으로 찾아냈다.
"""

import argparse
import asyncio
import json
import random
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from app.core.config import get_settings
from app.services.llm.base import LLMClient, LLMUnavailableError
from app.services.llm.registry import get_llm

API_URL = "http://127.0.0.1:8000/api/v1/chat"
QUESTIONS_PATH = Path("data/eval_questions.yaml")
RESULT_PATH = Path("data/eval_results/answer-judge.json")

FAITHFUL_SYSTEM = """Is this STATEMENT supported by the EXCERPTS?

Answer with exactly one word: YES or NO.

YES — the excerpts say this, or say something that directly implies it.
NO  — the excerpts do not say this, or say the opposite."""
"""**문장 하나씩 묻는다.**

처음에는 "답변의 모든 주장이 지지되는가?"를 한 번에 물었다. **결과가 전부 NO로
포화했다** (Gemini 0/6, gemma 1/6). 문장 하나만 어긋나도 NO이므로 당연한데,
그러면 **개선을 잴 수 없다** — 고치기 전과 후가 똑같이 0이다.

지표가 포화하면 **이진 판정을 비율로 바꾼다.** 문장 단위로 물어 "지지된 문장 /
전체 문장"을 내면, 6문장 중 2문장이 어긋난 답변과 5문장이 어긋난 답변이 구별된다.

호출 수가 답변 문장 수만큼 늘지만(답변당 5~7회) 답변 평가는 오프라인이라 감수한다.
"""

RELEVANT_SYSTEM = """Does the ANSWER address what the owner asked?

Answer with exactly one word: YES or NO.

YES — it answers the question, or explains why it cannot and asks for what it needs.
NO  — it talks about something else."""


@dataclass(slots=True)
class Judged:
    question: str
    answer: str
    coverage: str
    sources: int
    supported: int
    claims: int
    relevant: bool | None
    unsupported: list[str] = field(default_factory=list)
    excerpts: list[str] = field(default_factory=list)

    @property
    def faithfulness(self) -> float:
        return self.supported / self.claims if self.claims else 0.0


def ask(question: str, timeout: float = 300.0) -> dict:
    body = json.dumps({"question": question}).encode()
    request = urllib.request.Request(
        API_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


async def judge(llm: LLMClient, system: str, prompt: str) -> bool | None:
    """YES면 True, NO면 False, 못 읽으면 None."""
    try:
        raw = await llm.generate(prompt, system=system, max_tokens=8, reasoning=False)
    except LLMUnavailableError as exc:
        print(f"  ⚠️ 판정 호출 실패: {exc}", file=sys.stderr)
        return None
    words = raw.upper().split()
    if "NO" in words:
        return False
    if "YES" in words:
        return True
    return None


def build_prompt(question: str, excerpts: list[str], answer: str) -> str:
    joined = "\n\n".join(f"[{i}] {e[:900]}" for i, e in enumerate(excerpts, 1))
    return f"EXCERPTS:\n{joined or '(none)'}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"


# 폼 라벨만 있는 줄("이렇게 해보세요")은 주장이 아니라 제목이라 채점에서 뺀다.
_LABEL_ONLY = ("이렇게 해보세요", "알려주시면 좋아요", "이렇게 확인해요")


def claim_sentences(answer: str) -> list[str]:
    """답변에서 **사실 주장에 해당하는 줄**만 뽑는다.

    폼 라벨과 너무 짧은 줄은 제외한다 — 그것까지 채점하면 분모가 부풀어
    충실성이 실제보다 좋아 보인다.
    """
    out: list[str] = []
    for line in answer.splitlines():
        text = line.strip()
        if not text or text in _LABEL_ONLY or len(text) < 8:
            continue
        out.append(text)
    return out


async def faithfulness(
    llm: LLMClient, excerpts: list[str], answer: str
) -> tuple[int, int, list[str]]:
    """(지지된 문장 수, 전체 문장 수, 지지 안 된 문장들).

    **문장 하나씩 묻는 이유는 이진 판정이 포화하기 때문이다** — FAITHFUL_SYSTEM
    독스트링 참조.
    """
    sentences = claim_sentences(answer)
    if not sentences:
        return 0, 0, []
    joined = "\n\n".join(f"[{i}] {e[:900]}" for i, e in enumerate(excerpts, 1))
    supported = 0
    unsupported: list[str] = []
    for sentence in sentences:
        verdict = await judge(
            llm,
            FAITHFUL_SYSTEM,
            f"EXCERPTS:\n{joined or '(none)'}\n\nSTATEMENT: {sentence}",
        )
        if verdict:
            supported += 1
        else:
            unsupported.append(sentence)
    return supported, len(sentences), unsupported


def load_questions(limit: int) -> list[str]:
    entries = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = [e["question"] for e in entries]
    return questions[:limit] if limit else questions


async def collect(questions: list[str], llm: LLMClient) -> list[Judged]:
    out: list[Judged] = []
    for i, question in enumerate(questions, 1):
        payload = ask(question)
        excerpts = [s["content"] for s in payload.get("sources", [])]
        answer = payload.get("answer", "")
        prompt = build_prompt(question, excerpts, answer)

        supported, claims, unsupported = await faithfulness(llm, excerpts, answer)
        relevant = await judge(llm, RELEVANT_SYSTEM, prompt)
        row = Judged(
            question=question,
            answer=answer,
            coverage=payload.get("coverage", "?"),
            sources=len(excerpts),
            supported=supported,
            claims=claims,
            relevant=relevant,
            unsupported=unsupported,
            excerpts=excerpts,
        )
        out.append(row)
        print(
            f"  [{i:>2}/{len(questions)}] {row.coverage:12s} "
            f"충실 {supported}/{claims} ({row.faithfulness:.0%}) "
            f"{'응답' if relevant else '동문서답'}  {question[:26]}",
            flush=True,
        )
    return out


async def calibrate(rows: list[Judged], llm: LLMClient) -> None:
    """**근거를 바꿔치기해서** 충실성 판정기가 잡는지 본다.

    답변은 그대로 두고 근거만 다른 질문의 것으로 바꾼다. 그러면 답변 내용은
    멀쩡한데 **근거와의 관계만 끊긴다.** 판정기가 이걸 통과시키면 근거를 안 보고
    "말이 되는가"만 보고 있다는 뜻이다 — 그런 판정기는 환각을 못 잡는다.
    """
    usable = [r for r in rows if r.excerpts and r.answer]
    if len(usable) < 2:
        print("\n(근거가 있는 답변이 2건 미만이라 보정을 건너뛴다)")
        return

    rng = random.Random(42)
    print(f"\n── 판정기 보정: 근거 바꿔치기 {len(usable)}건 ──")
    caught = 0
    for i, row in enumerate(usable):
        other = usable[(i + 1 + rng.randrange(len(usable) - 1)) % len(usable)]
        sup, total, _ = await faithfulness(llm, other.excerpts, row.answer)
        # 근거를 바꿔쳤으니 지지율이 뚝 떨어져야 한다. 절반 미만이면 적발로 본다.
        if total and sup / total < 0.5:
            caught += 1

    neg = caught / len(usable)
    pos = sum(r.supported for r in usable) / max(1, sum(r.claims for r in usable))
    balanced = (pos + neg) / 2

    print(f"  양성(진짜 근거)에서 지지된 문장 비율: {pos:.0%}")
    print(f"  음성(바꿔친 근거)을 불충실로 적발: {caught}/{len(usable)} ({neg:.0%})")
    print(f"  **균형 정확도: {balanced:.0%}**  (찍기 = 50%)")
    print()

    # **음성 적발률만 보면 속는다.** 항상 NO라고 답하는 판정기도 음성은 100%다.
    # 양성까지 같이 봐야 "판정하는 것"과 "한쪽으로 미는 것"이 갈린다.
    if balanced < 0.65:
        print("  ⚠️ **이 판정기를 믿으면 안 된다.** 음성 적발률이 높아도 양성을")
        print("     대부분 버리면 사실상 'NO만 말하는 판정기'다 — 찍기와 다를 바 없다.")
        print("     더 강한 모델로 바꾸거나 프롬프트를 보정해야 한다.")
    elif pos < 0.6:
        print("  ⚠️ 음성은 잘 잡지만 양성을 자주 버린다. 실제 품질이 과소평가된다.")
    else:
        print("  ✓ 양쪽 다 쓸 만하다.")


def report(rows: list[Judged]) -> None:
    n = len(rows)
    total_claims = sum(r.claims for r in rows)
    total_supported = sum(r.supported for r in rows)
    relevant = sum(1 for r in rows if r.relevant is True)
    perfect = sum(1 for r in rows if r.claims and r.supported == r.claims)

    print("\n" + "═" * 66)
    print(f"  {n}문항 · 채점한 주장 {total_claims}개")
    if total_claims:
        print(
            f"  근거 충실성   {total_supported}/{total_claims} "
            f"({total_supported / total_claims:.0%})   ← 문장 단위"
        )
    print(f"  전부 지지된 답변  {perfect}/{n}")
    print(f"  질문 응답성   {relevant}/{n} ({relevant / n:.0%})")

    worst = sorted((r for r in rows if r.claims), key=lambda r: r.faithfulness)[:4]
    if worst:
        print("\n  ── 근거에 없는 말 (충실성 하위 4문항) ──")
        for r in worst:
            print(f"    [{r.coverage}] {r.faithfulness:.0%}  {r.question[:32]}")
            for u in r.unsupported[:2]:
                print(f"        ✗ {u[:62]}")

    print("\n  ── coverage별 충실성 ──")
    for cov in ("full", "partial", "needs_detail", "none"):
        group = [r for r in rows if r.coverage == cov]
        claims = sum(r.claims for r in group)
        if not claims:
            continue
        sup = sum(r.supported for r in group)
        print(f"    {cov:<13} {len(group):>2}건 · {sup}/{claims} ({sup / claims:.0%})")


async def run(args: argparse.Namespace) -> int:
    llm = get_llm(get_settings())
    questions = load_questions(args.n)
    print(f"질문 {len(questions)}개 · 판정 LLM = {llm.name}\n", flush=True)

    try:
        rows = await collect(questions, llm)
    except urllib.error.URLError as exc:
        print(f"✗ API에 연결할 수 없습니다 ({API_URL}): {exc}", file=sys.stderr)
        print("  uv run uvicorn app.main:app 를 먼저 띄우세요", file=sys.stderr)
        return 1

    report(rows)
    if args.calibrate:
        await calibrate(rows, llm)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            [{k: v for k, v in asdict(r).items() if k != "excerpts"} for r in rows],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n✓ 저장: {RESULT_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="답변 품질 평가 (LLM as a judge)")
    parser.add_argument("--n", type=int, default=0, help="앞에서 N문항만")
    parser.add_argument(
        "--calibrate", action="store_true", help="근거 바꿔치기로 판정기를 검증한다"
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
