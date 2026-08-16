"""수집 실행 CLI.

    uv run python -m scripts.collect.fetch --source avsab-humane-training
    uv run python -m scripts.collect.fetch --all
    uv run python -m scripts.collect.fetch --all --skip-pending

수집 결과는 data/raw/<source_id>.json 에 저장된다.

⚠️ 이 개발 컨테이너는 이그레스 프록시가 대상 도메인을 막을 수 있다.
   네트워크 수집은 로컬 머신에서 실행하는 것을 전제로 한다.
"""

import argparse
import asyncio
import logging
import sys

from .fetchers.base import LicensePendingError, RobotsDisallowedError
from .models import RawDoc, Source, save_raw
from .registry import get_fetcher, get_source, load_sources

logger = logging.getLogger(__name__)


def _save(source: Source, docs: list[RawDoc]) -> None:
    logger.info("저장: %s (%d건)", save_raw(source.id, docs), len(docs))


async def fetch_one(source: Source) -> int:
    fetcher = get_fetcher(source)
    docs = await fetcher.fetch(source)
    _save(source, docs)
    return len(docs)


async def main() -> int:
    parser = argparse.ArgumentParser(description="sources.yaml 기반 문서 수집")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="수집할 소스 id")
    group.add_argument("--all", action="store_true", help="전체 소스 수집")
    parser.add_argument(
        "--skip-pending",
        action="store_true",
        help="license: pending-check 소스를 에러 대신 건너뛴다",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    sources = [get_source(args.source)] if args.source else load_sources()

    total = 0
    failed: list[str] = []
    for source in sources:
        try:
            total += await fetch_one(source)
        except LicensePendingError as exc:
            if args.skip_pending:
                logger.warning("건너뜀(license 미확인): %s", source.id)
                continue
            logger.error("%s", exc)
            failed.append(source.id)
        except RobotsDisallowedError as exc:
            logger.error("%s", exc)
            failed.append(source.id)
        except Exception as exc:  # noqa: BLE001 — 한 소스 실패로 전체를 중단하지 않는다
            logger.error("소스 %s 수집 실패: %s", source.id, exc)
            failed.append(source.id)

    logger.info("완료: 문서 %d건", total)
    if failed:
        logger.error("실패한 소스: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
