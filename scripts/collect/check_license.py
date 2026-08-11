"""라이선스 확인 도우미.

license: pending-check 소스에 대해:
  1. robots.txt 판정을 자동으로 수행하고 (기계적인 부분)
  2. ToS에서 무엇을 찾아야 하는지, 결과에 따라 sources.yaml에 뭘 쓸지 안내한다
     (법적 판단이라 자동화하지 않는 부분)

    uv run python -m scripts.collect.check_license          # pending-check만
    uv run python -m scripts.collect.check_license --all    # 전체 재확인

⚠️ 개발 컨테이너는 프록시가 대상 도메인을 막는다 — 로컬 머신에서 실행할 것.
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from .fetchers.base import RobotsDisallowedError
from .fetchers.http import PoliteClient
from .models import Source
from .registry import load_sources

logger = logging.getLogger(__name__)

# 사이트들이 약관을 두는 흔한 경로 — 안내용이므로 전부 존재하진 않는다
_TOS_PATHS = ("/terms-of-use", "/terms", "/legal", "/terms-and-conditions", "/tos")

_TOS_CHECKLIST = """\
  ToS에서 찾아볼 문구:
    ① "automated means" / "scraping" / "crawling" / "robots" / "spiders" 금지 조항
    ② "personal, non-commercial use" 같은 개인·비상업 한정 조항
  판단:
    ① 발견 → 자동 수집 불가. sources.yaml에서 fetcher: local 로 바꾸고
       브라우저 '다른 이름으로 저장'한 파일을 data/raw/local/<id>/ 에 넣으세요
    ② 발견 → 개인 테스트는 가능. license: personal-use-only 로 기록
       (앱 배포 시 재검토 필요하다는 흔적)
    둘 다 없음 → license: robots-allowed-no-tos-clause 로 기록"""


async def check_source(client: PoliteClient, source: Source) -> None:
    print(f"\n── {source.id} ──")

    if not source.urls:
        print("  URL 없음(local fetcher) — 파일을 받는 시점에 사람이 판단하면 됩니다")
        return

    allowed: list[str] = []
    disallowed: list[str] = []
    unknown: list[str] = []
    for url in source.urls:
        try:
            (allowed if await client.robots_allows(url) else disallowed).append(url)
        except (RobotsDisallowedError, httpx.HTTPError) as exc:
            logger.debug("robots 확인 실패 %s: %s", url, exc)
            unknown.append(url)

    domains = sorted({urlsplit(u).netloc for u in source.urls})
    today = datetime.now(UTC).date().isoformat()

    verdict = f"허용 {len(allowed)} / 거부 {len(disallowed)} / 확인불가 {len(unknown)}"
    print(f"  robots.txt ({', '.join(domains)}):")
    print(f"    {verdict} (총 {len(source.urls)})")

    if unknown:
        print("  ⚠️ robots.txt를 가져오지 못한 URL이 있습니다 — 네트워크(프록시) 문제일 수 있으니")
        print("     로컬 머신에서 다시 실행하세요:")
        for url in unknown:
            print(f"     - {url}")

    if disallowed:
        print("  ✗ robots.txt가 거부한 URL — 자동 수집하면 안 됩니다:")
        for url in disallowed:
            print(f"     - {url}")
        print("  → 해당 URL을 sources.yaml에서 빼거나, fetcher: local 로 전환하세요")
        return

    if allowed and not unknown:
        print("  ✓ robots.txt는 전부 허용. 남은 건 ToS 확인 (사람이 2분):")
        for domain in domains:
            candidates = " 또는 ".join(f"https://{domain}{p}" for p in _TOS_PATHS[:2])
            print(f"    {domain}: {candidates} (없으면 페이지 하단 링크)")
        print(_TOS_CHECKLIST)
        print("  ToS까지 문제없으면 sources.yaml에 붙여넣기:")
        print(f"    license: robots-allowed-tos-checked-{today}")
        print(f"    license_checked_at: {today}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="pending-check 소스의 robots/ToS 확인 도우미")
    parser.add_argument("--all", action="store_true", help="pending 아닌 소스도 재확인")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    sources = load_sources()
    targets = sources if args.all else [s for s in sources if s.license_pending]

    if not targets:
        print("pending-check 소스가 없습니다. 전체 재확인은 --all")
        return 0

    print(f"확인 대상: {len(targets)}개 소스")
    async with PoliteClient(delay_seconds=0.3) as client:
        for source in targets:
            await check_source(client, source)

    print("\n※ robots.txt 판정은 자동이지만 ToS 해석은 최종적으로 사람의 판단입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
