"""평가셋 자동 생성 — 청크에서 질문을 만들어 (질문, 정답 청크) 쌍을 얻는다.

    uv run python -m scripts.eval.generate_qa --n 50
    uv run python -m scripts.eval.generate_qa --n 50 --seed 7 --out data/eval_auto_qa.jsonl
    uv run python -m scripts.eval.generate_qa --inspect data/eval_auto_qa.jsonl

**왜 필요한가.** 손으로 쓴 평가셋은 21문항이 한계였다. 그 정도로는 hit rate 같은
지표가 흔들리고(1문항이 4.8%다), 사람이 고른 질문이라 편향도 있다 — 내가 아는
주제만 묻게 된다.

**핵심 아이디어: 정답 라벨이 공짜로 생긴다.** 청크 하나를 골라 "이걸로 답할 수 있는
질문"을 만들면, 그 질문의 정답은 **정의상 그 청크**다. 사람이 11,354개를 뒤져
정답을 표시할 필요가 없다.

**한계도 분명하다. 06장에서 이걸 감안해 지표를 읽어야 한다:**

  - **정답이 하나라고 가정한다.** 비슷한 내용의 청크가 여럿이면(우리 코퍼스에는
    분리불안 청크가 수십 개다) 다른 청크가 1위여도 실제로는 맞는 답일 수 있는데
    지표는 틀렸다고 센다. **즉 점수가 실제보다 낮게 나온다.**
  - **LLM이 만든 질문은 실사용 질문과 다르다.** 보호자는 "우리 개가 밤에 짖어요"라고
    쓰지, 논문 문단을 요약한 질문을 하지 않는다. 그래서 손으로 쓴 21문항을
    **버리지 않고 같이 쓴다.**

**어휘 누출(leakage)을 한국어 생성으로 줄인다.** 질문이 청크의 표현을 그대로 베끼면
검색이 부당하게 쉬워진다. 여기서는 영어 청크로 **한국어** 질문을 만들게 하므로
표현이 그대로 넘어올 수 없고, 덤으로 실사용(한국어 질문 → 영어 코퍼스)과 같은
조건이 된다.
"""

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.models import Chunk, Document
from app.db.session import create_engine, create_session_factory
from app.services.llm.base import LLMClient, LLMUnavailableError
from app.services.llm.registry import get_llm

DEFAULT_OUT = Path("data/eval_auto_qa.jsonl")

GENERATE_SYSTEM = """You write ONE Korean question that a dog owner would actually ask, \
which the given excerpt answers.

Rules:
- Output ONLY the question. No quotes, no explanation, no numbering.
- Write in Korean, in the voice of a worried pet owner talking to a trainer.
  Good:  "강아지가 혼자 있을 때만 짖는데 왜 그런가요?"
  Bad:   "분리불안의 행동학적 기전은 무엇인가?"   (연구자 말투)
- The excerpt must actually answer it. Do not ask about something it only mentions.
- Never refer to the text itself. No "이 문서에 따르면", "위 내용에서", "본문의".
  The owner has not seen any document.
- One sentence. Under 40 characters if you can."""

USEFUL_SYSTEM = """Would a pet owner ever ask about what this excerpt says?

Answer with exactly one word: OWNER or RESEARCH.

OWNER — the excerpt says something that changes what an owner does with their dog: \
why a behaviour happens, how to train or manage it, what a body signal means, when to \
see a vet, how a health problem shows up as behaviour.
RESEARCH — study methodology, statistics, sample sizes, questionnaire validation, \
molecular biology, genetics, microbiome, oncology, pharmacokinetics, funding \
statements, author or reference lists, table and figure fragments.

RESEARCH is the expected answer for much of this corpus. Most of it is journal \
articles written for scientists."""
"""**판정을 생성에서 떼어낸 이유 (2026-08-14 실측).**

처음에는 생성 프롬프트 안에 "쓸모없는 발췌면 SKIP을 출력하라"를 넣었다.
gemma-4-e2b(4.6B)가 **20개 중 0개를 SKIP했다.** 종양 반응, T세포 증식, 심지어
"강아지 의사가 돈을 잘 벌었나요?"까지 질문을 만들어냈다.

이 프로젝트가 이미 겪은 실패 모드다 — `evidence_select`도 "쓸 근거 고르기 + 개
질문인지 판정"을 한 호출에 묶었다가 후자를 놓쳐 out-of-scope가 1/4이었고,
단답 분류로 분리하니 4/4가 됐다.

**작은 모델은 한 호출에 두 가지를 시키면 하나를 버린다.** 프롬프트에 적혀 있어도
안 지킨다. 생성처럼 "무언가를 만들라"는 지시가 있으면 특히 그쪽이 이긴다.
"""

VERIFY_SYSTEM = """Does the excerpt actually answer the question?

Answer with exactly one word: YES or NO.

YES — an owner reading only this excerpt would get a real answer.
NO  — the excerpt is merely on a related topic, or the question is so vague that \
almost any text would "answer" it ("우리 강아지 건강한가요?"), or the answer would \
have to come from outside the excerpt."""
"""생성한 질문을 되짚어 검증한다.

**왜 생성만으로 부족한가 (실측).** 판정기를 붙여 연구용 청크를 걸러낸 뒤에도
6개 중 3개가 쓸 수 없는 질문이었다 — "우리 강아지들 모두 건강한 건가요?",
"아버지가 개 의사였는데 왜 아들은 다른 일을 했을까요?".

문제가 두 종류다: **너무 막연해서** 아무 문서나 답이 되는 질문, 그리고 청크에
적혀 있긴 하지만 **보호자에게 무의미한** 질문. 둘 다 "이 발췌로 답이 되나"를
다시 물으면 걸러진다. 여기서도 한 호출에 한 가지만 시킨다.
"""

# 본문을 안 보고는 성립하지 않는 질문 — 실사용 질문이 아니므로 버린다.
_META_PHRASES = (
    "이 문서",
    "본문",
    "위 내용",
    "위 문단",
    "제시된",
    "주어진",
    "이 글",
    "이 연구",
    "발췌",
    "지문",
)

_MIN_Q_CHARS = 8
_MAX_Q_CHARS = 120

# 표·참고문헌 조각 거르기: 숫자와 기호가 많은 청크는 질문거리가 안 된다.
_DIGIT_HEAVY = 0.15


STRATA = {
    "all": None,
    "owner-docs": "보호자용 기관 문서 (ASPCA·RSPCA·VCA·AVSAB)",
    "papers": "PMC 논문",
}
"""**층화 표집(stratified sampling).**

무작위로 뽑으면 코퍼스 구성이 그대로 반영되는데, 그게 여기서는 문제다 (2026-08-14 실측):

    무작위 표본 (논문 96%)                OWNER 통과율  0/10
    보호자용 기관 문서 (ASPCA·RSPCA·VCA)  OWNER 통과율  14/15

**병목은 모델이 아니라 코퍼스다.** 더 강한 모델(Gemini)로 바꿔도 무작위 표본은
0%였다. 논문 청크의 대부분이 방법론·통계·결과표라 보호자 질문의 근거가 못 된다.

그래서 층을 나눠 뽑을 수 있게 했다. 다만 **`owner-docs` 층은 전체 청크의 2.8%
(320개)뿐**이라, 그 층으로만 만든 평가셋은 "이 시스템이 ASPCA를 잘 찾는가"를 재지
"코퍼스 전체를 잘 찾는가"를 재지 않는다. 결과를 읽을 때 반드시 감안할 것 —
그래서 생성된 각 문항에 어느 층에서 왔는지 남긴다.
"""


@dataclass(slots=True)
class QAPair:
    question: str
    chunk_id: int
    document_id: int
    document_title: str
    chunk_excerpt: str
    """청크 앞부분. 사람이 눈으로 검수할 때 쓴다 — 라벨이 맞는지 보려면 원문이 필요하다."""
    stratum: str = "all"
    """어느 층에서 뽑았는지. 지표를 층별로 쪼개 보려면 필요하다."""


def digit_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for c in text if c.isdigit()) / len(text)


def is_usable_chunk(content: str, min_chars: int) -> bool:
    """질문을 만들 만한 청크인가. LLM을 부르기 전에 싸게 거른다."""
    return len(content) >= min_chars and digit_ratio(content) < _DIGIT_HEAVY


def clean_question(raw: str) -> str:
    """모델 출력에서 질문만 남긴다. 쓸 수 없으면 빈 문자열."""
    text = raw.strip()
    # 추론형 모델의 사고과정 블록 (query_rewrite와 같은 처리).
    text = re.sub(r"<(think|thinking|reasoning)>.*?</\1>", "", text, flags=re.DOTALL | re.I)
    if "</think" in text.lower():
        text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.I)
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text.strip("`")
    # 여러 줄이면 첫 줄만. 설명을 뒤에 붙이는 모델이 있다.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    text = lines[0].strip().strip('"').strip("'").lstrip("-•").strip()
    text = re.sub(r"^\d+[.)]\s*", "", text)
    return text


def is_valid_question(q: str) -> tuple[bool, str]:
    """(쓸 수 있는가, 버리는 이유). 이유를 남기는 건 통계를 보기 위해서다 —
    무엇이 얼마나 걸러졌는지 모르면 프롬프트를 못 고친다."""
    if not q or q == "SKIP":
        return False, "빈 응답/SKIP"
    if len(q) < _MIN_Q_CHARS:
        return False, "너무 짧음"
    if len(q) > _MAX_Q_CHARS:
        return False, "너무 김"
    if "?" not in q and not q.endswith(("요", "까", "나", "죠", "가")):
        return False, "질문 형태가 아님"
    for phrase in _META_PHRASES:
        if phrase in q:
            return False, f"본문 참조({phrase})"
    return True, ""


async def sample_chunks(
    factory, n: int, seed: int, min_chars: int, max_per_doc: int, stratum: str = "all"
) -> list[tuple[int, int, str, str]]:
    """(chunk_id, document_id, title, content) 표본.

    **문서당 상한을 둔다.** 무작위로 뽑으면 청크가 105개인 논문이 3청크짜리 RSPCA
    가이드보다 35배 자주 뽑힌다. 그러면 평가셋이 긴 논문에 대한 시험이 되어버린다.
    """
    async with factory() as session:
        stmt = (
            select(Chunk.id, Chunk.document_id, Document.title, Chunk.content)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.corpus == "answer")
            .where(func.length(Chunk.content) >= min_chars)
        )
        # doc_type이 아니라 source_id로 가른다. AAHA는 doc_type=guide지만 수의사용
        # 진료 지침이라 보호자 문서와 성격이 다르다 (실측에서도 절반이 RESEARCH였다).
        if stratum == "owner-docs":
            stmt = stmt.where(Document.source_id.notlike("pmc-%")).where(
                Document.source_id.notlike("aaha%")
            )
        elif stratum == "papers":
            stmt = stmt.where(Document.source_id.like("pmc-%"))
        rows = (await session.execute(stmt)).all()

    pool = [
        (r.id, r.document_id, r.title, r.content)
        for r in rows
        if is_usable_chunk(r.content, min_chars)
    ]
    rng = random.Random(seed)
    rng.shuffle(pool)

    picked: list[tuple[int, int, str, str]] = []
    used: Counter[int] = Counter()
    for row in pool:
        if len(picked) >= n:
            break
        if used[row[1]] < max_per_doc:
            picked.append(row)
            used[row[1]] += 1
    return picked


async def is_owner_useful(llm: LLMClient, content: str) -> bool:
    """보호자가 물을 만한 내용인가. **생성과 분리된 단답 분류다** (위 독스트링 참조).

    **`reasoning=False`인 이유 (실측).** "판정에는 숙고를 켠다"가 이 프로젝트의 기본
    규칙인데 여기서는 반대였다. 같은 청크(천둥 공포 관련)를 두 설정으로 재보면:

        reasoning=True  max_tokens=8  -> RESEARCH   ← 전부 이렇게 나온다
        reasoning=False max_tokens=8  -> OWNER      ← 맞는 판정

    사고과정도 completion 토큰을 먹기 때문이다. 예산이 8토큰뿐이라 숙고를 켜면
    **생각하다 끝나고** 답이 기본값으로 눌린다. 규칙은 "판정이면 켠다"가 아니라
    **"숙고할 여유를 준 경우에만 켠다"**로 읽어야 한다.

    실패하면 통과시킨다. 여기서 막으면 LLM이 불안정할 때 평가셋이 조용히 비어버린다.
    """
    try:
        raw = await llm.generate(
            f"Excerpt:\n{content[:1500]}",
            system=USEFUL_SYSTEM,
            max_tokens=8,
            reasoning=False,
        )
    except LLMUnavailableError:
        return True
    return "RESEARCH" not in raw.upper()


async def answers_question(llm: LLMClient, content: str, question: str) -> bool:
    """이 청크로 그 질문에 답이 되는가. `is_owner_useful`과 같은 단답 분류다."""
    try:
        raw = await llm.generate(
            f"Excerpt:\n{content[:1500]}\n\nQuestion: {question}",
            system=VERIFY_SYSTEM,
            max_tokens=8,
            reasoning=False,
        )
    except LLMUnavailableError:
        return True
    return "NO" not in raw.upper().split()


async def generate_one(llm: LLMClient, content: str) -> str:
    try:
        raw = await llm.generate(
            f"Excerpt:\n{content[:2000]}",
            system=GENERATE_SYSTEM,
            max_tokens=120,
            # 질문 하나 만드는 일에 숙고는 필요 없다. 추론을 켜면 사고과정 토큰이
            # 예산을 먹어 content가 빈 채로 잘린다 (CLAUDE.md의 로컬 LLM 항목).
            reasoning=False,
        )
    except LLMUnavailableError as exc:
        print(f"  ⚠️ LLM 호출 실패: {exc}", file=sys.stderr)
        return ""
    return clean_question(raw)


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    llm = get_llm(settings)
    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    out_file = None

    try:
        chunks = await sample_chunks(
            factory, args.n, args.seed, args.min_chars, args.max_per_doc, args.stratum
        )
        if not chunks:
            print("✗ 표본을 못 뽑았습니다 — DB에 청크가 있는지 확인하세요", file=sys.stderr)
            return 1
        label = STRATA.get(args.stratum) or "코퍼스 전체"
        print(f"청크 {len(chunks)}개 표본 · 층={args.stratum}({label})")
        print(f"seed={args.seed} · LLM={llm.name}\n")

        # **한 건씩 즉시 기록한다.** 끝에서 한 번에 저장하면 긴 실행이 중간에 죽었을 때
        # (Gemini 무료 티어의 429 같은 이유로) 몇십 분치가 통째로 날아간다.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_file = args.out.open("w", encoding="utf-8")

        pairs: list[QAPair] = []
        rejected: Counter[str] = Counter()
        for i, (chunk_id, doc_id, title, content) in enumerate(chunks, 1):
            if not await is_owner_useful(llm, content):
                rejected["연구용 내용(RESEARCH)"] += 1
                print(f"  [{i:>3}/{len(chunks)}] – 연구용 내용이라 건너뜀", flush=True)
                continue
            question = await generate_one(llm, content)
            ok, reason = is_valid_question(question)
            if not ok:
                rejected[reason] += 1
                print(f"  [{i:>3}/{len(chunks)}] ✗ {reason:<16} {question[:36]}", flush=True)
                continue
            if not await answers_question(llm, content, question):
                rejected["검증 실패(답이 안 됨)"] += 1
                print(f"  [{i:>3}/{len(chunks)}] ✗ 검증 실패        {question[:36]}", flush=True)
                continue
            pair = QAPair(
                question=question,
                chunk_id=chunk_id,
                document_id=doc_id,
                document_title=title,
                chunk_excerpt=content[:300],
                stratum=args.stratum,
            )
            pairs.append(pair)
            out_file.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")
            out_file.flush()
            print(f"  [{i:>3}/{len(chunks)}] ✓ {question[:46]}", flush=True)
    finally:
        if out_file is not None:
            out_file.close()
        await engine.dispose()

    print(f"\n{'─' * 60}")
    print(f"  생성 {len(pairs)} / 시도 {len(chunks)} ({len(pairs) / len(chunks):.0%})")
    if rejected:
        print("  버린 이유: " + " · ".join(f"{k} {v}" for k, v in rejected.most_common()))
    print(f"  ✓ 저장: {args.out}")
    print("\n  ⚠️ 자동 생성이라 **직접 몇 개는 읽어보세요.** --inspect 로 통계를 봅니다.")
    return 0


def inspect(path: Path) -> int:
    """생성된 평가셋의 품질을 본다. 만든 다음 그냥 믿으면 안 된다."""
    if not path.is_file():
        print(f"✗ {path} 가 없습니다", file=sys.stderr)
        return 1
    lines = path.read_text(encoding="utf-8").splitlines()
    pairs = [json.loads(line) for line in lines if line.strip()]
    if not pairs:
        print("✗ 비어 있습니다", file=sys.stderr)
        return 1

    lengths = sorted(len(p["question"]) for p in pairs)
    by_doc = Counter(p["document_title"][:44] for p in pairs)
    dupes = [q for q, c in Counter(p["question"] for p in pairs).items() if c > 1]

    median = lengths[len(lengths) // 2]
    print(f"\n{path} — {len(pairs)}문항\n")
    print(f"  질문 길이   최소 {lengths[0]} · 중앙 {median} · 최대 {lengths[-1]}자")
    print(f"  문서 수     {len(by_doc)}개 (문항당 {len(pairs) / len(by_doc):.1f})")
    print(f"  중복 질문   {len(dupes)}건")
    print("\n  ── 문서별 상위 5 ──")
    for title, count in by_doc.most_common(5):
        print(f"    {count:>3}  {title}")
    print("\n  ── 샘플 5개 (라벨이 맞는지 눈으로 확인) ──")
    for p in pairs[:5]:
        print(f"\n    Q: {p['question']}")
        print(f"       정답 청크 {p['chunk_id']} — {p['document_title'][:50]}")
        print(f"       {p['chunk_excerpt'][:150].replace(chr(10), ' ')}…")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="평가셋 자동 생성 (청크 → 질문)")
    parser.add_argument("--n", type=int, default=50, help="표본 청크 수")
    parser.add_argument("--seed", type=int, default=42, help="재현을 위한 난수 시드")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-chars", type=int, default=400, help="이보다 짧은 청크는 제외")
    parser.add_argument("--max-per-doc", type=int, default=2, help="한 문서에서 뽑을 최대 청크 수")
    parser.add_argument(
        "--stratum",
        choices=tuple(STRATA),
        default="all",
        help="어느 층에서 뽑을지 (STRATA 독스트링 참조)",
    )
    parser.add_argument("--inspect", type=Path, help="생성된 파일의 품질 통계만 본다")
    args = parser.parse_args()

    if args.inspect:
        return inspect(args.inspect)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
