"""판정기 보정 — LLM 판정을 믿기 전에 판정기를 채점한다.

    uv run python -m scripts.eval.calibrate_judge --n 60
    uv run python -m scripts.eval.calibrate_judge --n 60 --variants current,short

**왜 필요한가.** 평가셋 생성의 마지막 단계는 "이 발췌로 그 질문에 답이 되나"를
LLM에게 묻는 검증이다. 그 판정을 그대로 믿고 써 왔는데, 모델을 바꾸자 성격이
완전히 달라졌다 (같은 프롬프트·같은 층):

    Gemini        225개 중 13개 거절 (5.8%)
    gemma-4-e2b    12개 중  9개 거절 (75%)

거절된 것 중에 "보호자가 떠날 준비를 할 때 강아지가 불안해하는 신호는?" 같은
멀쩡한 질문이 섞여 있었다.

**프롬프트만 고쳐서는 안 된다.** "더 많이 통과시키게" 조정하면 아무거나 통과하는
쪽으로 무너지는데, **통과율만 보면 그걸 알아챌 수 없다.** 그래서 정답을 아는
데이터로 양쪽을 같이 잰다:

    양성 — Gemini가 YES로 통과시킨 (질문, 정답 청크) 쌍     → YES가 정답
    음성 — 같은 질문 + **다른 문서**의 청크                 → NO가 정답

    양성 정답률이 낮다 → 좋은 질문을 버린다 (지금 문제)
    음성 정답률이 낮다 → 엉뚱한 근거를 통과시킨다 (평가셋 라벨이 오염된다)

**둘 다 봐야 하고, 균형 정확도로 고른다.** 한쪽만 보면 반드시 반대쪽이 무너진다.
"""

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.models import Chunk
from app.db.session import create_engine, create_session_factory
from app.services.llm.base import LLMClient, LLMUnavailableError
from app.services.llm.registry import get_llm
from scripts.eval.generate_qa import VERIFY_SYSTEM

QA_PATH = Path("data/eval_auto_qa_owner.jsonl")
RESULT_PATH = Path("data/eval_results/judge-calibration.json")

SHORT = """Does the excerpt answer the question?

Answer with exactly one word: YES or NO."""
"""NO 조건의 긴 설명을 덜어낸 것.

가설: 작은 모델은 부정 조건이 길게 나열되면 그쪽으로 기운다. 원래 프롬프트에는
"merely on a related topic", "so vague that almost any text would answer it",
"answer would have to come from outside" 세 갈래가 NO 쪽에만 붙어 있다.
"""

POSITIVE_FRAME = """Is the information needed to answer this question present in the excerpt?

Answer with exactly one word: YES or NO.

YES — the excerpt contains the answer, even partially.
NO  — the excerpt is about something else entirely."""
"""질문을 뒤집어 긍정 쪽에 무게를 실은 것.

"답이 되나?"는 완결성을 묻는 느낌이라 엄격해지고, "필요한 정보가 들어 있나?"는
포함 여부를 묻는다. 평가셋 용도로는 후자가 맞다 — 정답 청크는 답의 근거이지
완성된 답변이 아니다.
"""

VARIANTS = {
    "current": VERIFY_SYSTEM,
    "short": SHORT,
    "positive-frame": POSITIVE_FRAME,
}


@dataclass(slots=True)
class Pair:
    question: str
    content: str
    expected: bool
    """True면 YES가 정답 (양성), False면 NO가 정답 (음성)."""


async def build_pairs(factory, n: int, seed: int) -> list[Pair]:
    """양성 n건 + 음성 n건.

    **음성은 반드시 다른 문서에서 가져온다.** 같은 문서의 다른 청크는 실제로 답이
    될 수 있어서(06장의 "문서 단위 채점"이 그래서 있다) 정답이 NO라고 단정할 수 없다.
    """
    if not QA_PATH.is_file():
        print(f"✗ {QA_PATH} 가 없습니다", file=sys.stderr)
        raise SystemExit(1)
    lines = QA_PATH.read_text(encoding="utf-8").splitlines()
    qa = [json.loads(line) for line in lines if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(qa)
    picked = qa[:n]

    async with factory() as session:
        rows = (
            await session.execute(
                select(Chunk.id, Chunk.content).where(
                    Chunk.id.in_([p["chunk_id"] for p in picked])
                )
            )
        ).all()
        # 음성용 후보 — 평가셋에 쓰인 문서를 통째로 제외한다.
        gold_docs = {p["document_id"] for p in qa}
        others = (
            await session.execute(
                select(Chunk.content, Chunk.document_id)
                .where(Chunk.document_id.notin_(gold_docs))
                .where(func.length(Chunk.content) >= 400)
                .limit(3000)
            )
        ).all()

    text_by_id = {r.id: r.content for r in rows}
    distractors = [r.content for r in others]
    rng.shuffle(distractors)

    pairs: list[Pair] = []
    for i, p in enumerate(picked):
        content = text_by_id.get(p["chunk_id"])
        if content is None:
            continue
        pairs.append(Pair(p["question"], content, expected=True))
        if i < len(distractors):
            pairs.append(Pair(p["question"], distractors[i], expected=False))
    return pairs


async def judge(llm: LLMClient, system: str, pair: Pair) -> bool | None:
    """YES면 True, NO면 False, 못 읽으면 None."""
    try:
        raw = await llm.generate(
            f"Excerpt:\n{pair.content[:1500]}\n\nQuestion: {pair.question}",
            system=system,
            max_tokens=8,
            reasoning=False,
        )
    except LLMUnavailableError as exc:
        print(f"  ⚠️ 호출 실패: {exc}", file=sys.stderr)
        return None
    words = raw.upper().split()
    if "NO" in words:
        return False
    if "YES" in words:
        return True
    return None


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    llm = get_llm(settings)
    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)

    try:
        pairs = await build_pairs(factory, args.n, args.seed)
    finally:
        await engine.dispose()

    pos = [p for p in pairs if p.expected]
    neg = [p for p in pairs if not p.expected]
    print(f"\n판정기 보정 · 모델={llm.name}")
    print(f"양성 {len(pos)}건 (YES가 정답) · 음성 {len(neg)}건 (NO가 정답)\n")

    names = [v.strip() for v in args.variants.split(",")]
    results: dict[str, dict] = {}
    for name in names:
        if name not in VARIANTS:
            print(f"✗ 모르는 변형: {name} (가능: {', '.join(VARIANTS)})", file=sys.stderr)
            return 1
        system = VARIANTS[name]
        ok_pos = ok_neg = unreadable = 0
        for i, p in enumerate(pairs, 1):
            verdict = await judge(llm, system, p)
            if verdict is None:
                unreadable += 1
            elif verdict == p.expected:
                if p.expected:
                    ok_pos += 1
                else:
                    ok_neg += 1
            if i % 20 == 0:
                print(f"    {name}: {i}/{len(pairs)}", flush=True)

        rate_pos = ok_pos / len(pos) if pos else 0.0
        rate_neg = ok_neg / len(neg) if neg else 0.0
        results[name] = {
            "positive": rate_pos,
            "negative": rate_neg,
            "balanced": (rate_pos + rate_neg) / 2,
            "unreadable": unreadable,
        }
        print(f"  {name}: 양성 {rate_pos:.0%} · 음성 {rate_neg:.0%}", flush=True)

    _report(results)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {"model": llm.name, "n": args.n, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n✓ 저장: {RESULT_PATH}")
    return 0


def _report(results: dict[str, dict]) -> None:
    print("\n" + "═" * 66)
    print(f"  {'변형':<16} {'양성(YES맞춤)':>14} {'음성(NO맞춤)':>14} {'균형':>8}")
    print("─" * 66)
    for name, r in results.items():
        print(
            f"  {name:<16} {r['positive']:>14.0%} {r['negative']:>14.0%} "
            f"{r['balanced']:>8.0%}"
        )
    best = max(results, key=lambda k: results[k]["balanced"])
    print(f"\n  균형 정확도 최고: {best}")
    print("\n  읽는 법:")
    print("    양성이 낮다 → 좋은 질문을 버린다 (평가셋이 안 모인다)")
    print("    음성이 낮다 → 엉뚱한 근거를 통과시킨다 (평가셋 라벨이 오염된다)")
    print("    **한쪽만 높은 변형을 고르면 반대쪽이 무너진다.**")


def main() -> int:
    parser = argparse.ArgumentParser(description="검증 판정기 보정")
    parser.add_argument("--n", type=int, default=60, help="양성 표본 수 (음성도 같은 수)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants", default=",".join(VARIANTS), help="쉼표로 구분한 프롬프트 변형"
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
