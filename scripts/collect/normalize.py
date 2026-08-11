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

from .models import CORPUS_PATH, PROCESSED_DIR, RAW_DIR, RawDoc
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

    for path in sorted(RAW_DIR.glob("*.json")):
        source = sources.get(path.stem)
        if source is None:
            logger.warning("sources.yaml에 없는 raw 파일, 건너뜀: %s", path.name)
            continue

        for item in json.loads(path.read_text(encoding="utf-8")):
            doc = RawDoc.model_validate(item)
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
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "corpus.jsonl 생성: %d건 (짧아서 제외 %d, 중복 제외 %d)",
        len(records),
        skipped_short,
        skipped_dupe,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
