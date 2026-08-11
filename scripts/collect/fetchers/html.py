"""HTML fetcher — robots.txt 준수 + 본문 추출."""

import logging
from datetime import UTC, datetime

from selectolax.parser import HTMLParser

from ..models import RawDoc, Source
from .base import ensure_license_checked
from .http import PoliteClient

logger = logging.getLogger(__name__)

# 본문이 아닌 요소들 — 추출 전에 제거
_STRIP_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "iframe")

# 본문 후보 셀렉터 — 앞에서부터 먼저 매칭되는 것을 쓴다
_CONTENT_SELECTORS = ("main", "article", "[role='main']", "#content", ".content", "body")


def extract_main_text(html: str) -> tuple[str, str]:
    """(title, body_text)를 뽑는다. 사이트별 커스텀 셀렉터가 필요해지면 확장."""
    tree = HTMLParser(html)

    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""

    for tag in _STRIP_TAGS:
        for node in tree.css(tag):
            node.decompose()

    for selector in _CONTENT_SELECTORS:
        node = tree.css_first(selector)
        if node is None:
            continue
        text = node.text(separator="\n", strip=True)
        if len(text) > 200:  # 너무 짧으면 본문이 아니라 껍데기일 가능성
            return title, text

    return title, ""


class HtmlFetcher:
    async def fetch(self, source: Source) -> list[RawDoc]:
        ensure_license_checked(source)

        docs: list[RawDoc] = []
        async with PoliteClient() as client:
            for url in source.urls:
                logger.info("HTML 수집: %s", url)
                response = await client.get(url)
                title, text = extract_main_text(response.text)
                if not text:
                    logger.warning("본문 추출 실패: %s", url)
                docs.append(
                    RawDoc(
                        source_id=source.id,
                        url=url,
                        title=title or source.name,
                        text=text,
                        fetched_at=datetime.now(UTC).isoformat(),
                    )
                )
        return docs
