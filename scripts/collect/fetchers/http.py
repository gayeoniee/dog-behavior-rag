"""HTTP 공통 유틸 — robots.txt 확인 + rate limit."""

import asyncio
import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from .base import RobotsDisallowedError

logger = logging.getLogger(__name__)

USER_AGENT = "gy-rag-collector/0.1 (personal research project)"
REQUEST_DELAY_SECONDS = 1.0


class PoliteClient:
    """robots.txt를 준수하고 요청 간 지연을 두는 HTTP 클라이언트."""

    def __init__(
        self,
        *,
        delay_seconds: float = REQUEST_DELAY_SECONDS,
        respect_robots: bool = True,
    ) -> None:
        self._delay = delay_seconds
        self._respect_robots = respect_robots
        self._robots_cache: dict[str, RobotFileParser] = {}
        self._client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=30.0,
        )

    async def __aenter__(self) -> "PoliteClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def get(self, url: str) -> httpx.Response:
        if self._respect_robots:
            await self._check_robots(url)
        await asyncio.sleep(self._delay)
        response = await self._client.get(url)
        response.raise_for_status()
        return response

    async def _check_robots(self, url: str) -> None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"

        parser = self._robots_cache.get(origin)
        if parser is None:
            parser = RobotFileParser()
            robots_url = f"{origin}/robots.txt"
            try:
                resp = await self._client.get(robots_url)
                if resp.status_code >= 400:
                    # robots.txt가 없으면 전체 허용으로 간주 (RFC 9309)
                    parser.parse([])
                else:
                    parser.parse(resp.text.splitlines())
            except httpx.HTTPError as exc:
                # robots.txt 자체를 못 가져오면 보수적으로 막는다
                raise RobotsDisallowedError(
                    f"{robots_url} 를 가져오지 못해 수집을 중단합니다: {exc}"
                ) from exc
            self._robots_cache[origin] = parser

        if not parser.can_fetch(USER_AGENT, url):
            raise RobotsDisallowedError(
                f"robots.txt가 수집을 거부합니다: {url}\n"
                "이 소스는 sources.yaml에서 제외하거나 local fetcher(수동 다운로드)로 전환하세요."
            )
