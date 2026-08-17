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
import hashlib
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

This is REFERENCE MATERIAL, not a summary. Another dog owner with the same
problem must be able to act on it without watching the video.

For each problem discussed, write a paragraph that carries all three of:

1. WHAT the dog does — the observable behavior and when it happens
2. WHY it happens — the trainer's actual explanation of the cause
3. WHAT TO DO — the concrete procedure, in order, with the details that make
   it reproducible (where to stand, what to hold, how long, what to do when
   the dog reacts, what NOT to do)

Rules:
- **Never keep proper names.** Write 강아지 / 반려견 for the dog and 보호자 for
  the owner, even when the transcript uses a name. A document that says
  "시루가 주저앉습니다" cannot be found by someone asking about their own dog.
- **Keep the specifics that make advice usable.** "간식을 코앞에 대고 천천히
  바닥으로 내립니다" is usable; "긍정적으로 유도합니다" is not. Numbers,
  distances, repetitions, and durations MUST survive.
- **Do not compress steps into a result.** If the trainer walks through five
  moves, write five moves. Losing the order makes it unusable.
- Write in Korean, in plain declarative sentences with proper sentence endings.
- Fix obvious speech-recognition errors from context (수혜사→수의사, 홀련→훈련).
- Drop greetings, introductions, sponsor reads, laughter, filler, and chit-chat.
- Drop the specific dog's breed, age, and backstory unless the advice depends
  on it (then write it as a condition: "어린 강아지의 경우", "대형견은").
- Do NOT invent anything. If one of the three parts is missing from the
  excerpt, write the parts that are there and leave the rest out. If nothing
  usable remains, output exactly: SKIP
- No speaker labels, no timestamps, no markdown."""

def prompt_fingerprint() -> str:
    """지금 프롬프트의 지문. 정제 결과에 같이 저장한다.

    **이어받기가 낡은 결과를 지키는 사고가 있었다.** 프롬프트를 고쳐 놓고 다시
    돌렸는데, 이미 정제된 문서는 건너뛰므로 **아무것도 안 바뀌었다.** 로그는
    "이미 정제된 184건은 건너뛴다"만 찍고 정상 종료했다.

    이어받기는 할당량이 끊겼을 때를 위한 것이지 **프롬프트가 바뀐 뒤에도 옛
    결과를 지키라는 뜻이 아니다.** 지문이 다르면 건너뛰지 않는다.
    """
    return hashlib.sha256(REFINE_SYSTEM.encode("utf-8")).hexdigest()[:12]


CHUNK_CHARS = 3000
"""LLM에 한 번에 넘길 자막 길이.

너무 크면 모델이 뒤쪽을 요약해버리고, 너무 작으면 맥락이 끊긴다. 자막 한 편이
2,000~5,000자라 대부분 1~2회 호출로 끝난다.
"""


def split_for_llm(text: str, size: int) -> list[str]:
    """공백 경계에서 자른다. 자막에는 문장 경계가 없어 길이가 기준이지만,
    **단어 중간에서 끊으면 그 단어가 통째로 망가진다.**

    실제로 "긴장성 부동화"가 조각 경계에 걸려 뒷조각이 "성 부동화가 너무 잦게
    일어나"로 시작했다. 조각마다 따로 정제하므로 LLM은 앞뒤를 볼 수 없고,
    잘린 단어를 그대로 살려 쓴다. 공백까지만 물러나면 이 손실이 사라진다.
    """
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            pieces.append(text[start:])
            break
        # 공백을 못 찾으면(공백 없는 긴 덩어리) 그냥 자른다 — 무한 루프를 막는다.
        space = text.rfind(" ", start + size // 2, end)
        cut = space if space != -1 else end
        pieces.append(text[start:cut])
        start = cut + 1 if space != -1 else cut
    return pieces or [""]


async def refine_one(llm: LLMClient, text: str) -> str:
    """한 편을 정제한다. **LLM이 죽으면 예외를 그대로 올린다.**

    예전에는 호출 실패를 잡아서 `continue`했는데, 그러면 빈 문자열이 돌아오고
    호출부가 그걸 "정제 결과가 너무 짧다"로 읽어 **멀쩡한 문서를 버렸다.**
    할당량이 끊긴 순간부터 남은 수십 편이 전부 "내용 없음"으로 찍힌다:

        [119/202]  621자 → 0자 (0%)   ⚠️ 너무 짧아 제외

    **실패가 성공처럼 보이는 게 가장 나쁘다.** 파일에는 안 써지므로 다시 돌리면
    복구되지만, 나중에 "왜 202편 중 118편뿐이지?"에서 원인을 못 찾는다.
    이제는 위로 던져서 호출부가 멈추고 이유를 말한다.
    """
    parts: list[str] = []
    for piece in split_for_llm(text, CHUNK_CHARS):
        raw = await llm.generate(
            f"Transcript excerpt:\n{piece}",
            system=REFINE_SYSTEM,
            max_tokens=1200,
            # 정리 작업이지 판단이 아니다. 추론을 켜면 사고과정이 예산을 먹어
            # 본문이 잘린다 (CLAUDE.md의 로컬 LLM 항목).
            reasoning=False,
        )
        cleaned = raw.strip()
        if cleaned and cleaned != "SKIP":
            parts.append(cleaned)
    return "\n\n".join(parts)


def inspect(source: str, limit: int) -> int:
    """정제 결과를 **사람이 읽으라고** 출력한다.

    자동 정제는 초안이다. 여기서 봐야 하는 건 통계가 아니라 두 가지다:

      1. LLM이 **원본에 없는 조언을 지어냈는가** — 프롬프트로 금지해도 일어난다
      2. 말이 되는 문장이 됐는가 (오탈자·화자 뒤섞임이 정리됐는가)

    1번을 보려면 정제본만 봐서는 안 된다. **그럴듯하게 읽히는 게 지어낸 것의
    특징**이기 때문이다. 그래서 원본 발췌를 위에 같이 찍는다.
    """
    target = RAW_DIR / f"{source}.refined.json"
    if not target.is_file():
        print(f"✗ {target} 가 없습니다 — 먼저 정제를 실행하세요", file=sys.stderr)
        return 1

    docs = json.loads(target.read_text(encoding="utf-8"))
    raw_path = RAW_DIR / f"{source}.json"
    originals: dict[str, str] = {}
    if raw_path.is_file():
        originals = {d["source_id"]: d["text"] for d in json.loads(raw_path.read_text("utf-8"))}

    lengths = sorted(len(d["text"]) for d in docs)
    total = sum(lengths)
    print(f"{target.name} — {len(docs)}건 · 총 {total:,}자")
    if lengths:
        mid = lengths[len(lengths) // 2]
        print(f"  길이: 최소 {lengths[0]} · 중앙값 {mid} · 최대 {lengths[-1]}")
    if originals:
        before = sum(len(originals.get(d["source_id"], "")) for d in docs)
        if before:
            print(f"  원본 {before:,}자 → 정제 {total:,}자 ({total / before:.0%})")
    print()

    for doc in docs[:limit]:
        print("─" * 72)
        print(doc["title"][:68])
        print(f"  {doc['url']}")
        original = originals.get(doc["source_id"], "")
        if original:
            print("\n  [원본 자막 앞 300자]")
            print("  " + original[:300].replace("\n", " "))
        print("\n  [정제 결과]")
        for line in doc["text"].split("\n"):
            print("  " + line if line else "")
        print()

    if len(docs) > limit:
        print(f"… 외 {len(docs) - limit}건 (--limit 로 더 보기)")
    return 0


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
        stored = json.loads(target.read_text(encoding="utf-8"))
        current = prompt_fingerprint()
        stale = [d for d in stored if d.get("meta", {}).get("prompt") != current]
        if stale and not args.keep_stale:
            print(f"  ⚠️ 프롬프트가 바뀌었다 — 옛 프롬프트로 만든 {len(stale)}건을 다시 정제한다")
            print("     (그대로 두려면 --keep-stale)")
            done = {
                d["source_id"]: d for d in stored if d.get("meta", {}).get("prompt") == current
            }
        else:
            done = {d["source_id"]: d for d in stored}
        if done:
            print(f"  이미 정제된 {len(done)}건은 건너뛴다")
    docs = [d for d in docs if d["source_id"] not in done]
    if args.limit:
        docs = docs[: args.limit]

    llm = get_llm(get_settings())
    print(f"{raw_path.name} — {len(docs)}건 정제 · LLM={llm.name}\n", flush=True)

    out: list[dict] = list(done.values())
    blocked = 0
    for i, doc in enumerate(docs, 1):
        before = len(doc["text"])
        try:
            refined = await refine_one(llm, doc["text"])
        except LLMUnavailableError as exc:
            # **한 편이 막힌 것과 LLM 자체를 못 쓰는 것은 다르다.**
            # 콘텐츠 필터는 그 문서 하나만의 문제이므로 건너뛰고 계속한다.
            # 자동자막은 오인식이 많아 멀쩡한 상담이 걸리기도 한다.
            if "content_filter" in str(exc):
                print(f"      ⚠️ 콘텐츠 필터에 막혀 건너뜀: {doc['title'][:40]}", flush=True)
                blocked += 1
                continue
            # 할당량이 끊긴 경우다. 계속 돌아봐야 한 건도 못 만들고, 남은 문서가
            # 전부 "너무 짧아 제외"로 찍혀 진짜 이유를 가린다.
            print(f"\n✗ LLM을 쓸 수 없어 {i - 1}편에서 멈춘다: {str(exc)[:120]}", file=sys.stderr)
            print(f"  여기까지 {len(out)}편은 저장돼 있다 — 다시 실행하면 이어서 받는다")
            print("  Gemini 무료 티어는 하루 단위로 초기화된다. LM Studio로 마저 하려면")
            print("  .env의 LLM_BASE_URL을 로컬로 바꾸면 되지만, 한 코퍼스에 품질이")
            print("  다른 문서가 섞이므로 되도록 같은 모델로 끝내는 게 좋다.")
            return 1
        ratio = len(refined) / before if before else 0
        print(
            f"  [{i:>2}/{len(docs)}] {before:>5}자 → {len(refined):>5}자 "
            f"({ratio:.0%})  {doc['title'][:34]}",
            flush=True,
        )
        if len(refined) < args.min_chars:
            print("      ⚠️ 너무 짧아 제외", flush=True)
            continue
        # 어느 프롬프트로 만든 결과인지 같이 남긴다 — 프롬프트를 고친 뒤 다시
        # 돌릴 때 이어받기가 낡은 결과를 지키지 않도록.
        meta = {**doc.get("meta", {}), "prompt": prompt_fingerprint()}
        out.append({**doc, "text": refined, "meta": meta})
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
        "--keep-stale",
        action="store_true",
        help="프롬프트가 바뀌어도 옛 정제 결과를 그대로 둔다 (기본은 다시 정제)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="정제된 결과를 원본과 나란히 출력한다 (LLM 호출 없음). "
        "지어낸 내용이 없는지 사람이 확인하는 용도다",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=150,
        help="정제 결과가 이보다 짧으면 버린다. 2~3분짜리 Q&A 영상은 정제하면 "
        "200~300자라 기본값이 낮다 (긴 상담 영상은 600~900자)",
    )
    args = parser.parse_args()
    # 읽기 전용이라 LLM도 설정도 필요 없다 — 다른 기기에서 결과만 검수할 수 있게.
    if args.inspect:
        return inspect(args.source, args.limit or 5)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
