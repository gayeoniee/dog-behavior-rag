"""수집 파이프라인 테스트 — 네트워크 없이 동작한다."""

from datetime import UTC, datetime

import pytest

from scripts.collect.fetchers.base import (
    LicensePendingError,
    ensure_license_checked,
)
from scripts.collect.fetchers.html import extract_main_text
from scripts.collect.models import RawDoc, Source
from scripts.collect.normalize import clean_text, content_hash
from scripts.collect.registry import get_fetcher, load_sources


def _source(**overrides) -> Source:
    base = {
        "id": "test-src",
        "name": "테스트 소스",
        "urls": ["https://example.com/a"],
        "fetcher": "html",
        "language": "en",
        "species": "dog",
        "axis": ["problem"],
        "authority_tier": 1,
        "methodology": "reward_based",
        "published_at": 2024,
        "volatility": "stable",
        "license": "public",
    }
    return Source.model_validate(base | overrides)


# ── sources.yaml 자체 검증 ──────────────────────────────────────


def test_sources_yaml_is_valid() -> None:
    """실제 sources.yaml이 스키마에 맞고, 모든 fetcher가 해석 가능해야 한다."""
    sources = load_sources()
    assert sources, "sources.yaml이 비어 있음"
    for source in sources:
        assert get_fetcher(source) is not None


def test_sources_yaml_covers_all_axes() -> None:
    """네 축 각각을 담당하는 소스가 최소 하나는 있어야 한다."""
    sources = load_sources()
    covered = {a for s in sources for a in s.axis}
    assert covered == {"problem", "cause", "training", "medical"}


# ── 라이선스 게이트 ────────────────────────────────────────────


def test_pending_license_blocks_fetch() -> None:
    src = _source(license="pending-check")
    with pytest.raises(LicensePendingError):
        ensure_license_checked(src)


def test_checked_license_passes() -> None:
    ensure_license_checked(_source(license="public-guideline-pdf"))


# ── HTML 본문 추출 ─────────────────────────────────────────────


def test_extract_main_text_prefers_main_tag() -> None:
    body = "개는 몸짓으로 감정을 표현한다. " * 30
    html = f"""
    <html><head><title>바디랭귀지</title></head>
    <body>
      <nav>메뉴 항목들</nav>
      <main><p>{body}</p></main>
      <footer>저작권 고지</footer>
    </body></html>
    """
    title, text = extract_main_text(html)
    assert title == "바디랭귀지"
    assert "몸짓으로 감정을" in text
    assert "메뉴 항목들" not in text
    assert "저작권 고지" not in text


def test_extract_main_text_short_content_returns_empty() -> None:
    _, text = extract_main_text("<html><body><main>짧음</main></body></html>")
    assert text == ""


# ── 정규화 ─────────────────────────────────────────────────────


def test_clean_text_normalizes_whitespace() -> None:
    raw = "첫  줄\t끝\r\n\r\n\r\n\r\n둘째 줄"
    assert clean_text(raw) == "첫 줄 끝\n\n둘째 줄"


def test_content_hash_is_stable_and_distinct() -> None:
    assert content_hash("같은 내용") == content_hash("같은 내용")
    assert content_hash("내용 A") != content_hash("내용 B")


def test_rawdoc_roundtrip() -> None:
    doc = RawDoc(
        source_id="test-src",
        url="https://example.com/a",
        title="제목",
        text="본문",
        fetched_at=datetime.now(UTC).isoformat(),
    )
    assert RawDoc.model_validate(doc.model_dump()) == doc
