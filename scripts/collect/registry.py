"""소스 로딩 + fetcher 선택.

app/services/*/registry.py 와 같은 패턴: 구현체 선택은 여기 한 곳에서만 한다.
"""

import yaml

from .fetchers.base import Fetcher
from .models import SOURCES_PATH, Source


def load_sources() -> list[Source]:
    with SOURCES_PATH.open(encoding="utf-8") as f:
        entries = yaml.safe_load(f)
    sources = [Source.model_validate(e) for e in entries]

    ids = [s.id for s in sources]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"sources.yaml에 중복된 id: {dupes}")
    return sources


def get_source(source_id: str) -> Source:
    for source in load_sources():
        if source.id == source_id:
            return source
    raise KeyError(f"sources.yaml에 없는 소스: {source_id!r}")


def get_fetcher(source: Source) -> Fetcher:
    if source.fetcher == "pdf":
        from .fetchers.pdf import PdfFetcher

        return PdfFetcher()
    if source.fetcher == "html":
        from .fetchers.html import HtmlFetcher

        return HtmlFetcher()
    if source.fetcher == "local":
        from .fetchers.local import LocalFetcher

        return LocalFetcher()
    if source.fetcher == "pmc":
        from .fetchers.pmc import PmcFetcher

        return PmcFetcher()
    raise ValueError(f"지원하지 않는 fetcher: {source.fetcher!r}")
