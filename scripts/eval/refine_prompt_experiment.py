"""정제 프롬프트 A/B — 잡음을 지키면서 균질화만 풀 수 있나.

    uv run python -m scripts.eval.refine_prompt_experiment --variants v1,v2,v3
    uv run python -m scripts.eval.refine_prompt_experiment --report

**왜 필요한가.** 보듬TV 자막 328편이 서로 너무 닮아 검색이 구별하지 못한다:

    영어 문서끼리       0.544
    한국어 원본 자막     0.62    ← 소재가 이미 균질 (한 채널·한 훈련사)
    한국어 정제본       0.73    ← 정제가 +0.10 더 얹었다

무관한 주제 글끼리는 한국어 0.289 · 영어 0.331이라 **언어 산물이 아니다.** 실제로
문서가 닮은 것이고, 그래서 식분증·합사 질문에서 맞는 문서가 상위에 못 온다.

**2차 시도가 왜 기각됐나 — 이 스크립트가 존재하는 이유다.** 틀 강제를 빼고
"훈련사가 다룬 순서대로"로 바꿨더니 유사도는 0.729 → 0.666으로 내려갔는데,
출력을 읽어보니 **정제가 안 된 것**이었다:

    대화 조각 잔재  0/25 → 6/25    "셨어요? 야."
    촬영·영상 언급  0/25 → 3/25
    이름 잔재       3/25 → 6/25    "빠삐. 모서리를 무서워하네요"

유사도가 내려간 건 원본 잡음이 그대로 들어와서다 — **잡음은 제각각이라 유사도를
떨어뜨린다.** 지표를 하나만 봤으면 채택했을 것이다.

그래서 **잡음 지표를 통과 조건으로 먼저 박아둔다**(`gate`). "유사도만 좋아진
후보"는 자동으로 떨어진다.

## 결과 — 셋 다 기각. 그리고 이 방향 자체가 닫혔다 (2026-08-18)

              유사도   수치   고유사물   대화조각  촬영  이름잔재
    기준선     0.731   8/25   25/25      0     0    8/25
    v1        0.729   8/25   23/25      0     0    7/25
    v2        0.717   8/25   21/25      0     0    5/25
    v3        0.727  11/25   23/25      0     0    5/25

잡음 gate는 셋 다 통과했다 — 규칙을 안 건드렸으니 당연하다. **문제는 유사도가
거의 안 움직였다는 것이다** (최선 -0.014). 목표는 원본 수준 0.62였다.

**왜 안 움직였는지는 프롬프트 밖에 있었다.** 코퍼스를 세어보고 알았다:

    한국어  328건 →   340청크   (문서당 1.0)
    영어    294건 → 9,571청크   (문서당 32.6)

**한국어 문서 하나가 청크 하나다.** 정제가 4,065자를 942자로 줄이므로(23%)
한 편이 통째로 한 단락이 된다. 그 안에서 문장 몇 개를 다르게 쓰라고 지시해봐야
벡터가 바뀔 리가 없다 — **압축이 이미 개별성을 버린 뒤였다.**

영어는 논문 한 편이 32청크라 질문이 그중 딱 맞는 하나를 때린다. 한국어는
한 편이 한 벡터라 어떤 질문에도 고만고만하게 걸린다. **균질화가 아니라
입도(granularity) 비대칭이다.**

교훈: **지표가 안 움직이면 지표를 더 만들지 말고 대상을 세어볼 것.**
프롬프트 3차까지 가는 동안 아무도 청크 수를 안 셌다.
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import scripts.collect  # noqa: F401 — 콘솔 UTF-8
from app.core.config import get_settings
from app.services.chunking import split_text
from app.services.embeddings.registry import get_embedder
from app.services.llm.registry import get_llm
from scripts.collect.refine_transcripts import CHUNK_CHARS, REFINE_SYSTEM, split_for_llm

RAW_PATH = Path("data/raw/bodeum-tv.json")
BASELINE_PATH = Path("data/raw/bodeum-tv.refined.json")
"""기준선은 **이미 Gemini로 정제해 둔 결과**다. 다시 만들지 않는다 — 같은 모델·같은
프롬프트라 재생성할 이유가 없고, 25편치 할당량을 아낀다."""

RESULT_DIR = Path("data/eval_results/refine_prompts")

# ── 후보 프롬프트 ────────────────────────────────────────────
# 운영 프롬프트(REFINE_SYSTEM)에 문장을 **덧붙이기만** 한다. Rules의 잡음 제거·
# 이름 금지·수치 보존 조항은 글자 그대로 유지된다 — 2차 실패가 그걸 건드려서 났다.

_OPENING_FREE = """
- **Do not open every document with the same sentence shape.** Start with
  whatever is most specific to THIS episode — the object, the place, the moment
  it goes wrong. "반려견이 ~하는 행동은 ~때문입니다" is fine when it fits and
  wrong when it is a reflex."""

_KEEP_DISTINCT = """
- **Keep the words that make this episode different from every other episode.**
  The specific object (종이컵, 방석, 코담요, 라면박스), the place (엘리베이터,
  현관, 식탁), the move (제자리에서 한 바퀴 돌기, 줄을 고정하기). Someone
  searching will type those words. Generic phrasing makes this document
  indistinguishable from hundreds of others, and then nobody can find it."""

VARIANTS: dict[str, str] = {
    "v1": REFINE_SYSTEM + _OPENING_FREE,
    "v2": REFINE_SYSTEM + _KEEP_DISTINCT,
    "v3": REFINE_SYSTEM + _OPENING_FREE + _KEEP_DISTINCT,
}
"""**"순서 자유화"는 후보에 없다** — 2차에서 반증됐다. 기각된 걸 다시 넣지 않는다."""

# ── 지표 ─────────────────────────────────────────────────────
# ⚠️ 정규식에 약어나 짧은 토큰을 넣을 때는 단어 경계를 확인할 것. 같은 날
#    `CI`가 "spe*ci*fic"에 걸리고 mentions()가 부분 문자열에 걸린 사고가 있었다.

NUMERIC = re.compile(r"\d+\s*(번|회|분|초|개월|마리|미터|걸음|주|일)")
STEPS = re.compile(r"(먼저|그다음|이후|반복합니다|반복하십시오|순서대로)")
OBJECTS = re.compile(
    r"(종이컵|방석|코담요|라면박스|엘리베이터|현관|식탁|켄넬|하네스|리드줄|"
    r"장난감|담요|가위|바리깡|목줄|간식|계단|자동차)"
)
CHATTER = re.compile(r"(셨어요\?|잖아요|해보세요\.|그쵸|야\.|네\. |어요\?)")
FILMING = re.compile(r"(촬영|영상|편집|출연|녹화|구독)")
NAME_LEAK = re.compile(r"(생후 \d+개월 된|[가-힣]{2,3}(이는|이가))")


def mean_pairwise(vectors: list[list[float]]) -> float:
    """서로 간 코사인 유사도의 평균. 낮을수록 문서가 구별된다.

    임베딩이 정규화돼 있으므로 내적이 곧 코사인이다.
    """
    total = count = 0.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            total += sum(a * b for a, b in zip(vectors[i], vectors[j], strict=True))
            count += 1
    return total / count if count else 0.0


async def refine_one(llm, text: str, system: str) -> str:
    parts: list[str] = []
    for piece in split_for_llm(text, CHUNK_CHARS):
        try:
            raw = await llm.generate(
                f"Transcript excerpt:\n{piece}", system=system, max_tokens=1200, reasoning=False
            )
        except Exception as exc:  # noqa: BLE001 — 어떤 실패든 그 편만 건너뛴다
            print(f"      호출 실패: {str(exc)[:70]}", flush=True)
            continue
        cleaned = raw.strip()
        if cleaned and cleaned != "SKIP":
            parts.append(cleaned)
    return "\n\n".join(parts)


async def generate(variant: str, limit: int) -> None:
    """후보 하나로 정제해 저장한다. **한 편마다 저장하고 이어받는다.**

    할당량이 중간에 끊기는 게 정상 시나리오라, 마지막에 한 번만 저장하면
    수십 분치를 통째로 잃는다. 오늘만 같은 실수를 세 번 겪었다.
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / f"{variant}.json"
    done: list[str] = (
        json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else []
    )
    docs = json.loads(RAW_PATH.read_text(encoding="utf-8"))[:limit]
    if len(done) >= len(docs):
        print(f"  [{variant}] 이미 {len(done)}편 — 건너뜀")
        return

    llm = get_llm(get_settings())
    print(f"  [{variant}] {len(done)}/{len(docs)}편부터 시작 · LLM={llm.name}", flush=True)
    for i in range(len(done), len(docs)):
        done.append(await refine_one(llm, docs[i]["text"], VARIANTS[variant]))
        out_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        print(f"    {i + 1}/{len(docs)} ({len(done[-1])}자)", flush=True)


async def report(variants: list[str], limit: int) -> int:
    sets: dict[str, list[str]] = {}
    if BASELINE_PATH.is_file():
        sets["기준선(V0)"] = [
            d["text"] for d in json.loads(BASELINE_PATH.read_text(encoding="utf-8"))[:limit]
        ]
    for v in variants:
        path = RESULT_DIR / f"{v}.json"
        if path.is_file():
            sets[v] = json.loads(path.read_text(encoding="utf-8"))
    if not sets:
        print("✗ 비교할 결과가 없습니다 — 먼저 생성하세요", file=sys.stderr)
        return 1

    embedder = get_embedder(get_settings())
    await embedder.warmup()

    print(f"\n{'':12} {'유사도':>7} {'수치':>7} {'절차어':>7} {'고유사물':>8}"
          f" {'대화조각':>8} {'촬영':>6} {'이름잔재':>8} {'길이':>7}")
    print("-" * 74)
    rows: dict[str, dict] = {}
    for label, texts in sets.items():
        usable = [t for t in texts if t.strip()]
        firsts = [chunks[0] for t in usable if (chunks := split_text(t))]
        vectors = await embedder.embed(firsts)
        n = max(len(usable), 1)
        row = {
            "sim": mean_pairwise(vectors),
            "numeric": sum(1 for t in usable if NUMERIC.search(t)),
            "steps": sum(1 for t in usable if STEPS.search(t)),
            "objects": sum(1 for t in usable if OBJECTS.search(t)),
            "chatter": sum(1 for t in usable if CHATTER.search(t)),
            "filming": sum(1 for t in usable if FILMING.search(t)),
            "names": sum(1 for t in usable if NAME_LEAK.search(t)),
            "chars": sum(len(t) for t in usable) // n,
            "n": n,
        }
        rows[label] = row
        print(
            f"{label:<12} {row['sim']:>7.3f} {row['numeric']:>5}/{n}"
            f" {row['steps']:>5}/{n} {row['objects']:>6}/{n}"
            f" {row['chatter']:>6}/{n} {row['filming']:>4}/{n}"
            f" {row['names']:>6}/{n} {row['chars']:>6}자"
        )

    base = rows.get("기준선(V0)")
    if base:
        print("\n── 채택 판정 (돌리기 전에 정한 기준) ──")
        print("  1단계 gate: 대화조각·촬영·이름잔재가 기준선보다 나빠지면 탈락")
        print("  2단계: 통과분 중 유사도 최저 + 수치·고유사물이 기준선 이상\n")
        passed = []
        for label, row in rows.items():
            if label.startswith("기준선"):
                continue
            fails = [
                k for k in ("chatter", "filming", "names") if row[k] > base[k]
            ]
            if fails:
                print(f"  ✗ {label:<6} gate 탈락 — {', '.join(fails)}가 기준선보다 나쁘다")
            else:
                keeps = row["numeric"] >= base["numeric"] and row["objects"] >= base["objects"]
                mark = "○" if keeps else "△"
                print(
                    f"  {mark} {label:<6} gate 통과 · 유사도 {row['sim']:.3f}"
                    f" ({row['sim'] - base['sim']:+.3f})"
                    + ("" if keeps else " · 구체성이 기준선 미만")
                )
                if keeps:
                    passed.append((row["sim"], label))
        if passed:
            print(f"\n  → 승자: {min(passed)[1]}")
        else:
            print("\n  → 기준선 유지. 후보 중 조건을 다 만족하는 게 없다")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description="정제 프롬프트 A/B")
    parser.add_argument("--variants", default="v1,v2,v3", help=f"쉼표 구분 {tuple(VARIANTS)}")
    parser.add_argument("--limit", type=int, default=25, help="비교에 쓸 영상 수")
    parser.add_argument("--report", action="store_true", help="생성 없이 표만 출력")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        print(f"✗ 모르는 후보: {unknown} (가능: {tuple(VARIANTS)})", file=sys.stderr)
        return 1

    if not args.report:
        for v in variants:
            await generate(v, args.limit)
    return await report(variants, args.limit)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
