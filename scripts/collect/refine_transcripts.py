"""자막 정제 — 구어체 대화를 검색 가능한 텍스트로 다듬는다.

    uv run python -m scripts.collect.refine_transcripts --source bodeum-tv
    uv run python -m scripts.collect.refine_transcripts --source bodeum-tv --limit 2 --dry-run

**왜 필요한가.** 유튜브 자동자막을 그대로 적재하면 이런 게 청크가 된다:

    그 똥을 저한테 한번 줘 보시겠어요, 어머니? 네. 아, 진짜 똥을 똥 봉투를 주셔야
    똥을 주 내가 이렇게 잡을까요? ... 안녕하세요. 훈련사 강우입니다.

문제가 네 가지다:

  1. **오탈자** — "수의사"→"수혜사", "훈련사"→"홀련 선생님" (자동 인식 한계)
  2. **화자 구분 없음** — 보호자 질문과 훈련사 답변이 한 덩어리
  3. **문장 경계 없음** — 자막은 화면 표시 단위로 쪼개져 있다
  4. **낮은 정보 밀도** — 8분에 4,000자인데 인사·리액션이 절반이다
     (비교: ASPCA 문서 한 편이 23,895자)

03장에서 "청크의 42.8%가 문장 중간에서 끊긴다"를 봤는데, 자막은 애초에 문장이
없어서 그 문제가 훨씬 심하다.

**왜 수집(fetcher)과 분리했나.** `data/raw`와 `data/processed`를 나눈 것과 같은
이유다 — **원본은 비싸게 받고 가공은 언제든 다시 한다.** 정제 프롬프트를 고칠
때마다 유튜브를 다시 긁으면 안 된다.

결과는 `data/raw/<source>.refined.json`에 쓰고, `normalize`가 원본 대신 이걸 쓴다.
"""

import argparse
import asyncio
import json
import logging
import sys

from app.core.config import get_settings
from app.services.llm.base import LLMClient, LLMUnavailableError
from app.services.llm.registry import get_llm

from .models import RAW_DIR

logger = logging.getLogger(__name__)

REFINE_SYSTEM = """You clean up a Korean dog-training consultation transcript so it \
can be used as reference material.

Rewrite the excerpt keeping ONLY what a dog owner could learn from:
- the dog's problem or situation
- what the trainer observed and why the dog behaves that way
- what the trainer told the owner to do, and how

Rules:
- Write in Korean, in plain declarative sentences with proper sentence endings.
- Fix obvious speech-recognition errors from context (수혜사→수의사, 홀련→훈련).
- Drop greetings, introductions, sponsor reads, laughter, filler, and chit-chat.
- Do NOT invent advice that is not in the excerpt. If the excerpt has no usable
  content, output exactly: SKIP
- Keep the trainer's actual reasoning. Do not compress it into a slogan.
- No speaker labels, no timestamps, no markdown."""

CHUNK_CHARS = 3000
"""LLM에 한 번에 넘길 자막 길이.

너무 크면 모델이 뒤쪽을 요약해버리고, 너무 작으면 맥락이 끊긴다. 자막 한 편이
2,000~5,000자라 대부분 1~2회 호출로 끝난다.
"""


def split_for_llm(text: str, size: int) -> list[str]:
    """길이로만 자른다. 자막에는 문장 경계가 없어서 다른 기준이 없다."""
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


async def refine_one(llm: LLMClient, text: str) -> str:
    parts: list[str] = []
    for piece in split_for_llm(text, CHUNK_CHARS):
        try:
            raw = await llm.generate(
                f"Transcript excerpt:\n{piece}",
                system=REFINE_SYSTEM,
                max_tokens=1200,
                # 정리 작업이지 판단이 아니다. 추론을 켜면 사고과정이 예산을 먹어
                # 본문이 잘린다 (CLAUDE.md의 로컬 LLM 항목).
                reasoning=False,
            )
        except LLMUnavailableError as exc:
            logger.warning("정제 호출 실패: %s", exc)
            continue
        cleaned = raw.strip()
        if cleaned and cleaned != "SKIP":
            parts.append(cleaned)
    return "\n\n".join(parts)


async def run(args: argparse.Namespace) -> int:
    raw_path = RAW_DIR / f"{args.source}.json"
    if not raw_path.is_file():
        print(f"✗ {raw_path} 가 없습니다 — fetch를 먼저 실행하세요", file=sys.stderr)
        return 1
    docs = json.loads(raw_path.read_text(encoding="utf-8"))

    # **이미 정제한 건 건너뛴다.** Gemini 무료 티어는 일일 할당량이 있어서 수백 편을
    # 돌리면 중간에 막힌다. 그때 처음부터 다시 하면 할당량을 두 번 쓰는 셈이다.
    target = RAW_DIR / f"{args.source}.refined.json"
    done: dict[str, dict] = {}
    if target.is_file() and not args.dry_run:
        done = {d["source_id"]: d for d in json.loads(target.read_text(encoding="utf-8"))}
        if done:
            print(f"  이미 정제된 {len(done)}건은 건너뛴다")
    docs = [d for d in docs if d["source_id"] not in done]
    if args.limit:
        docs = docs[: args.limit]

    llm = get_llm(get_settings())
    print(f"{raw_path.name} — {len(docs)}건 정제 · LLM={llm.name}\n", flush=True)

    out: list[dict] = list(done.values())
    for i, doc in enumerate(docs, 1):
        before = len(doc["text"])
        refined = await refine_one(llm, doc["text"])
        ratio = len(refined) / before if before else 0
        print(
            f"  [{i:>2}/{len(docs)}] {before:>5}자 → {len(refined):>5}자 "
            f"({ratio:.0%})  {doc['title'][:34]}",
            flush=True,
        )
        if len(refined) < args.min_chars:
            print("      ⚠️ 너무 짧아 제외", flush=True)
            continue
        out.append({**doc, "text": refined})
        # 한 편이 끝날 때마다 저장한다 — 할당량이 끊겨도 여기까지는 남는다.
        if not args.dry_run:
            target.write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    if args.dry_run:
        print("\n(--dry-run 이라 저장하지 않음)")
        if out:
            print("\n── 첫 문서 정제 결과 앞 600자 ──")
            print(out[0]["text"][:600])
        return 0

    target = RAW_DIR / f"{args.source}.refined.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 저장: {target} ({len(out)}건)")
    print("  다음: normalize 가 원본 대신 이 파일을 쓴다")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="자막 LLM 정제")
    parser.add_argument("--source", required=True, help="sources.yaml의 소스 id")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N건만")
    parser.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 본다")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=150,
        help="정제 결과가 이보다 짧으면 버린다. 2~3분짜리 Q&A 영상은 정제하면 "
        "200~300자라 기본값이 낮다 (긴 상담 영상은 600~900자)",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
