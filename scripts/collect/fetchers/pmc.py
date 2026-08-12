"""PMC Open Access 서브셋 fetcher — NCBI E-utilities.

이 fetcher가 코퍼스 확대의 핵심이다. 기관 웹사이트는 건건이 ToS를 읽어야 하고
대부분 personal-use-only로 끝나는데, **PMC OA는 라이선스가 논문 메타데이터로
같이 온다.** 읽을 약관이 없고, CC-BY면 배포까지 가능하다.

흐름:

    esearch(db=pmc, term + "open access"[filter]) → PMCID 목록
      → efetch(db=pmc, id=20개씩 배치) → JATS XML
      → <abstract> + <body> 본문, <permissions><license>에서 라이선스

`oa.fcgi`(OA Web Service)도 라이선스를 주지만 ID 하나씩만 받는다. efetch는
20개 배치가 되고 본문까지 같이 오므로 요청 수가 1/20로 준다.

레이트 리밋: API 키 없이 3 req/s가 NCBI 기준이라 지연을 0.4초로 둔다.
ESearch는 최대 10,000건까지만 반환한다(retstart + retmax <= 10000).

JATS 파싱에 stdlib xml.etree를 쓴다 — selectolax는 HTML용이고, 새 의존성을
늘리지 않으려는 것이다.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from urllib.parse import urlencode

from ..models import RawDoc, Source
from .base import ensure_license_checked
from .http import PoliteClient

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH_SIZE = 20
NCBI_DELAY_SECONDS = 0.4  # API 키 없이 3 req/s

# 본문에서 통째로 들어내는 요소. 참고문헌은 애초에 파싱하지 않아
# 청커의 참고문헌 필터가 할 일을 만들지 않는다.
_DROP_TAGS = {"ref-list", "back", "table-wrap", "fig", "graphic", "xref", "table"}

_SPECIES_TERMS = ("dog", "canine", "puppy", "puppies")
_MIN_SPECIES_MENTIONS = 3


def mentions_target_species(text: str) -> bool:
    """개 이야기가 맞는지 최소한으로 확인한다.

    esearch를 [tiab]로 좁혀도 색인 특성상 엉뚱한 논문이 섞인다 — 검증 중에
    "치매 환자 실종 알림 시스템", "이란 디아스포라 건강" 같은 게 들어왔다.
    질의를 고치는 게 1차 방어고, 이건 질의가 나중에 느슨해져도 코퍼스가
    오염되지 않게 하는 2차 방어다.
    """
    lowered = text.lower()
    return sum(lowered.count(term) for term in _SPECIES_TERMS) >= _MIN_SPECIES_MENTIONS


def _text_of(element: ET.Element) -> str:
    """요소의 텍스트를 재귀 수집. _DROP_TAGS는 통째로 건너뛴다."""
    if element.tag in _DROP_TAGS:
        return ""
    parts = [element.text or ""]
    for child in element:
        parts.append(_text_of(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _first_text(root: ET.Element, path: str) -> str:
    node = root.find(path)
    return _text_of(node).strip() if node is not None else ""


def parse_article(article: ET.Element) -> tuple[str, str, dict] | None:
    """JATS <article> → (title, text, meta). 본문이 없으면 None."""
    # JATS는 pub-id-type="pmcid"에 "PMC13454400" 형태로, "pmcaid"에 숫자만 담는다.
    # 둘 다 받아서 숫자만 남긴다.
    pmcid = ""
    for node in article.iter("article-id"):
        if node.get("pub-id-type") in ("pmcid", "pmcaid"):
            pmcid = (node.text or "").strip().removeprefix("PMC")
            if pmcid:
                break
    if not pmcid:
        return None

    title = _first_text(article, ".//title-group/article-title") or f"PMC{pmcid}"

    blocks: list[str] = []
    abstract = article.find(".//abstract")
    if abstract is not None:
        blocks.append(_text_of(abstract).strip())
    body = article.find(".//body")
    if body is not None:
        for section in body.iter("sec"):
            heading = _first_text(section, "title")
            paragraphs = [_text_of(p).strip() for p in section.findall("p") if _text_of(p).strip()]
            if paragraphs:
                blocks.append("\n\n".join([heading, *paragraphs] if heading else paragraphs))
        if not blocks or len(blocks) == 1:
            # 섹션 구조가 없는 논문 — <body> 직하 <p>를 그대로 쓴다.
            blocks.extend(_text_of(p).strip() for p in body.findall(".//p") if _text_of(p).strip())

    text = "\n\n".join(b for b in blocks if b)
    if not text:
        return None

    # 라이선스가 여기 온다. 이게 이 fetcher를 쓰는 이유다 — ToS를 읽지 않아도 된다.
    license_node = article.find(".//permissions/license")
    license_value = "pmc-oa-unspecified"
    if license_node is not None:
        href = ""
        for key, value in license_node.attrib.items():
            if key.endswith("href"):
                href = value
                break
        license_value = _license_label(href, _text_of(license_node))

    year = ""
    for pub_date in article.iter("pub-date"):
        node = pub_date.find("year")
        if node is not None and (node.text or "").strip().isdigit():
            year = node.text.strip()
            break

    meta = {"license": license_value}
    if year:
        meta["published_at"] = int(year)
    return title, text, {**meta, "pmcid": pmcid}


def _license_label(href: str, body: str) -> str:
    """CC 라이선스 URL/문구 → sources.yaml에서 쓰는 짧은 라벨."""
    blob = f"{href} {body}".lower()
    for code, label in (
        ("by-nc-nd", "cc-by-nc-nd"),
        ("by-nc-sa", "cc-by-nc-sa"),
        ("by-nc", "cc-by-nc"),
        ("by-sa", "cc-by-sa"),
        ("by/", "cc-by"),
        ("publicdomain", "cc0"),
        ("zero/", "cc0"),
    ):
        if code in blob:
            return label
    if "creativecommons" in blob:
        return "cc-unspecified"
    return "pmc-oa-unspecified"


class PmcFetcher:
    async def fetch(self, source: Source) -> list[RawDoc]:
        ensure_license_checked(source)
        if not source.query:
            raise ValueError(f"소스 {source.id!r}에 query가 없습니다 (pmc fetcher 필수)")

        # respect_robots=False: eutils는 프로그램 접근을 위해 제공되는 공식 API이고
        # NCBI는 robots.txt가 아니라 레이트 리밋으로 사용량을 규율한다. 대신 그
        # 리밋(3 req/s)을 delay_seconds로 정확히 지킨다.
        async with PoliteClient(delay_seconds=NCBI_DELAY_SECONDS, respect_robots=False) as client:
            ids = await self._search(client, source)
            logger.info("PMC 검색 %r → %d건", source.query, len(ids))

            docs: list[RawDoc] = []
            for start in range(0, len(ids), BATCH_SIZE):
                batch = ids[start : start + BATCH_SIZE]
                docs.extend(await self._fetch_batch(client, source, batch))
                logger.info("  PMC 본문 %d/%d", min(start + BATCH_SIZE, len(ids)), len(ids))
        return docs

    async def _search(self, client: PoliteClient, source: Source) -> list[str]:
        params = urlencode(
            {
                "db": "pmc",
                "term": f'{source.query} AND "open access"[filter]',
                "retmax": source.max_records,
                "retmode": "json",
                # 기본 정렬은 최신순이라 관련 없는 최신 논문이 먼저 온다.
                "sort": "relevance",
            }
        )
        response = await client.get(f"{EUTILS}/esearch.fcgi?{params}")
        return response.json().get("esearchresult", {}).get("idlist", [])

    async def _fetch_batch(
        self, client: PoliteClient, source: Source, ids: list[str]
    ) -> list[RawDoc]:
        params = urlencode({"db": "pmc", "id": ",".join(ids), "retmode": "xml"})
        response = await client.get(f"{EUTILS}/efetch.fcgi?{params}")

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            logger.warning("JATS 파싱 실패, 배치 건너뜀: %s", exc)
            return []

        fetched_at = datetime.now(UTC).isoformat()
        docs: list[RawDoc] = []
        for article in root.iter("article"):
            parsed = parse_article(article)
            if parsed is None:
                continue
            title, text, meta = parsed
            pmcid = meta.pop("pmcid")
            if not mentions_target_species(text):
                logger.info("  대상 종 언급 없음, 제외: PMC%s %s", pmcid, title[:50])
                continue
            docs.append(
                RawDoc(
                    source_id=source.id,
                    url=f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/",
                    title=title,
                    text=text,
                    fetched_at=fetched_at,
                    meta=meta,
                )
            )
        return docs
