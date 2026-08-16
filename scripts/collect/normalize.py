"""정규화: data/raw/*.json → data/processed/corpus.jsonl

RawDoc에 sources.yaml의 메타데이터(axis, methodology, authority_tier ...)를 합쳐
문서 1건 = JSONL 1줄로 만든다. 이 파일이 이후 청킹·임베딩의 입력이 된다.

    uv run python -m scripts.collect.normalize
"""

import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

from .models import BLOG_CORPUS_PATH, CORPUS_PATH, PROCESSED_DIR, RAW_DIR, RawDoc
from .registry import load_sources

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 200  # 이보다 짧으면 추출 실패로 간주하고 제외


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


REFINED_SUFFIX = ".refined"


def raw_files() -> dict[str, Path]:
    """소스 id → 정규화에 쓸 파일. **정제본이 있으면 그쪽을 쓴다.**

    **정제한 자막이 통째로 버려지고 있었다.** 예전에는 `RAW_DIR.glob("*.json")`을
    돌면서 파일명으로 소스를 찾았는데, `bodeum-tv.refined.json`은 stem이
    `bodeum-tv.refined`라 sources.yaml에서 안 잡혀 "없는 raw 파일"로 건너뛰었다.
    그리고 옆에 있는 `bodeum-tv.json`(오탈자투성이 원본 자막)이 대신 들어갔다.

    **LLM으로 다듬어 놓고 정작 안 쓰고, 안 다듬은 쪽을 쓰고 있었다.** 다른 기기로
    옮겨간 경우엔 더 나쁘다 — 저장소에는 정제본만 실려 있어서(`.gitignore` 예외)
    거기서는 유튜브 문서가 0건이 된다.

    파일명 규칙 하나가 조용히 파이프라인을 갈랐다. `.refined.json`이 `.json`으로
    끝나서 glob에는 걸리는데 stem은 안 맞는, 딱 눈에 안 띄는 형태였다.
    """
    chosen: dict[str, Path] = {}
    for path in sorted(RAW_DIR.glob("*.json")):
        if path.stem.endswith(REFINED_SUFFIX):
            chosen[path.stem[: -len(REFINED_SUFFIX)]] = path  # 정제본이 항상 이긴다
        else:
            chosen.setdefault(path.stem, path)
    return chosen


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    sources = {s.id: s for s in load_sources()}
    if not RAW_DIR.is_dir():
        logger.error("%s 없음 — 먼저 fetch를 실행하세요", RAW_DIR)
        return 1

    records: list[dict] = []
    seen_hashes: dict[str, str] = {}
    skipped_short = 0
    skipped_dupe = 0

    for source_id, path in raw_files().items():
        source = sources.get(source_id)
        if source is None:
            logger.warning("sources.yaml에 없는 raw 파일, 건너뜀: %s", path.name)
            continue

        # **문서 단위로 빼는 장치.** 소스 하나가 수백 편을 내는데 그중 몇 편만
        # 기존 코퍼스와 충돌할 수 있다. 소스를 통째로 격리하면 멀쩡한 나머지를
        # 다 버리게 되고, 통째로 들이면 자기모순이 들어온다.
        # 왜 뺐는지는 sources.yaml에 문장을 인용해 적어둔다 — 근거 없이 빠진
        # 문서가 있으면 나중에 아무도 되돌릴 수 없다.
        excluded = tuple(source.meta.get("exclude_ids") or ())

        for item in json.loads(path.read_text(encoding="utf-8")):
            doc = RawDoc.model_validate(item)
            if any(doc.source_id.endswith(bad) for bad in excluded):
                logger.info("제외 목록에 있어 건너뜀: %s", doc.title[:40])
                continue
            text = clean_text(doc.text)

            if len(text) < MIN_TEXT_CHARS:
                logger.warning("본문 너무 짧음(%d자), 제외: %s", len(text), doc.url)
                skipped_short += 1
                continue

            digest = content_hash(text)
            if digest in seen_hashes:
                logger.warning("중복 본문, 제외: %s (원본: %s)", doc.url, seen_hashes[digest])
                skipped_dupe += 1
                continue
            seen_hashes[digest] = doc.url

            records.append(
                {
                    "id": f"{doc.source_id}:{digest}",
                    "title": doc.title,
                    "content": text,
                    "content_hash": digest,
                    "source_id": doc.source_id,
                    "source_url": doc.url,
                    "language": source.language,
                    "species": source.species,
                    "axis": source.axis,
                    "methodology": source.methodology,
                    "authority_tier": source.authority_tier,
                    "published_at": source.published_at,
                    "volatility": source.volatility,
                    "license": source.license,
                    "fetched_at": doc.fetched_at,
                    "corpus": source.corpus,
                    # 소스 기본값 위에 문서별 값을 덮어쓴다. PMC는 논문마다
                    # license/published_at이 다르므로 이게 없으면 전부 뭉개진다.
                    **doc.meta,
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 답변 근거 코퍼스와 관찰용 코퍼스를 물리적으로 분리한다. 같은 풀에 두면
    # 검색이 둘 다 물어와 답변이 자기모순에 빠질 수 있다(지배이론 등).
    answer = [r for r in records if r["corpus"] == "answer"]
    observation = [r for r in records if r["corpus"] == "observation"]

    _write(CORPUS_PATH, answer)
    logger.info(
        "corpus.jsonl 생성: %d건 (짧아서 제외 %d, 중복 제외 %d)",
        len(answer),
        skipped_short,
        skipped_dupe,
    )
    if observation:
        _write(BLOG_CORPUS_PATH, observation)
        logger.info(
            "corpus_blogs.jsonl 생성: %d건 (관찰용 — 답변 근거로 검색되지 않는다)",
            len(observation),
        )
    elif BLOG_CORPUS_PATH.is_file():
        # **관찰용 소스가 없어졌으면 파일도 지운다.** 안 지우면 낡은 내용이
        # 그대로 남아서, 승격된 문서가 여전히 관찰용인 것처럼 보인다.
        # 출력 파일은 매 실행마다 입력 상태를 그대로 반영해야 한다.
        BLOG_CORPUS_PATH.unlink()
        logger.info("관찰용 소스가 없어 corpus_blogs.jsonl 을 지웠다")
    return 0


def _write(path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
