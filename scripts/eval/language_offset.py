"""언어별 배경 유사도 차이를 잰다.

    uv run python -m scripts.eval.language_offset
    uv run python -m scripts.eval.language_offset --sample 3000

**무엇을 재는가.** 한국어 질문을 코퍼스의 **모든** 청크와 비교해 언어별 평균
유사도를 구한다. 대부분의 청크는 어떤 질문과도 무관하므로, 이 평균은 "그 언어의
문서가 기본적으로 받는 점수"에 가깝다. 두 언어의 값이 다르면 그 차이가
**내용과 무관한 순수 언어 오프셋**이다.

**왜 필요한가.** 한국어 문서 199건을 답변 코퍼스에 넣었더니 청크의 1.7%가 상위
5위의 41%를 가져갔다. 한→한 코사인이 한→영보다 구조적으로 높기 때문이다.
RRF로 등수만 쓰는 방법은 독식을 막았지만 **모든 질문에 2/5씩 자리를 떼어주는
균일한 세금**이 됐다.

**왜 상위 후보가 아니라 전체인가 — 여기서 한 번 틀렸다.** 처음엔 각 풀의 상위
후보들로 평균을 내서 z점수를 만들었는데, 상위 후보의 평균은 *그 질문이 그 풀에
얼마나 잘 맞는지*에 따라 움직인다. 잘 맞는 풀은 평균도 높으니 빼면
**살리려던 신호를 스스로 지운다.** 배경은 관련 없는 문서들에서 재야 한다.

읽는 법:
- **`격차`의 표준편차가 작으면** 고정 상수로 빼도 된다 (질문마다 안 변한다는 뜻)
- **크면** 이 접근도 실패다. 그때는 상수가 아니라 다른 장치가 필요하다
"""

import argparse
import asyncio
import statistics as st
import sys

import yaml

import scripts.collect  # noqa: F401 — import 시점에 콘솔을 UTF-8로 바꾼다
from app.core.config import get_settings
from app.services.embeddings.registry import get_embedder

EVAL_PATH = "data/eval_questions.yaml"


async def main() -> int:
    parser = argparse.ArgumentParser(description="언어별 배경 유사도 격차 측정")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="언어당 표본 청크 수. 0이면 전체(권장 — 11,555개는 전부 계산해도 빠르다)",
    )
    args = parser.parse_args()

    from pathlib import Path

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    rows = yaml.safe_load(Path(EVAL_PATH).read_text(encoding="utf-8"))
    questions = [q["question"] for q in (rows["questions"] if isinstance(rows, dict) else rows)]

    settings = get_settings()
    embedder = get_embedder(settings)
    await embedder.warmup()
    engine = create_async_engine(settings.database_url)

    limit = "" if args.sample <= 0 else f"order by random() limit {args.sample}"
    sql = text(
        f"""
        select d.language, avg(1 - (ch.embedding <=> cast(:vec as vector))) as mean_sim,
               count(*) as n
        from (select id, document_id, embedding from chunks {limit}) ch
        join documents d on d.id = ch.document_id
        where d.methodology != 'aversive'
        group by d.language
        """
    )

    gaps: list[float] = []
    print(f"{'질문':<44} {'영어 배경':>9} {'한국어 배경':>11} {'격차':>8}")
    print("─" * 76)
    async with engine.connect() as conn:
        for question in questions:
            vec = await embedder.embed_query(question)
            means = {
                row.language: float(row.mean_sim)
                for row in (await conn.execute(sql, {"vec": str(vec)})).all()
            }
            if "ko" not in means or "en" not in means:
                print("두 언어가 다 있어야 잰다 — 한국어 문서를 적재했는지 확인할 것")
                return 1
            gap = means["ko"] - means["en"]
            gaps.append(gap)
            print(f"{question[:42]:<44} {means['en']:>9.4f} {means['ko']:>11.4f} {gap:>+8.4f}")

    await engine.dispose()

    print("─" * 76)
    mean_gap = st.mean(gaps)
    sd = st.pstdev(gaps)
    print(
        f"격차 평균 {mean_gap:+.4f} · 표준편차 {sd:.4f} · "
        f"최소 {min(gaps):+.4f} 최대 {max(gaps):+.4f}"
    )
    print()
    # 판단 기준: 근거 있는 질문(0.714)과 주제 공백(0.673)의 차이가 0.04다.
    # 오프셋의 흔들림이 그보다 작아야 상수로 빼도 신호를 망치지 않는다.
    if sd < 0.01:
        print(f"✓ 흔들림이 작다 — 고정 상수 {mean_gap:+.4f} 로 빼도 된다")
    elif sd < 0.02:
        print(f"△ 애매하다. 상수({mean_gap:+.4f})로 빼되 평가로 확인할 것")
    else:
        print("✗ 질문마다 너무 달라 상수로 못 뺀다. 다른 방법을 찾을 것")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
