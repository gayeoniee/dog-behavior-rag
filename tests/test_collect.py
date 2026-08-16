"""수집 파이프라인 테스트 — 네트워크 없이 동작한다."""

from datetime import UTC, datetime

import pytest

from scripts.collect.fetchers.base import (
    LicensePendingError,
    ensure_license_checked,
)
from scripts.collect.fetchers.html import extract_main_text
from scripts.collect.fetchers.http import USER_AGENT, parse_robots
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


# ── robots.txt 판정 (check_license가 쓰는 로직) ─────────────────


def test_robots_disallow_path_blocks_matching_url() -> None:
    parser = parse_robots("User-agent: *\nDisallow: /private/\n")
    assert not parser.can_fetch(USER_AGENT, "https://example.com/private/page")
    assert parser.can_fetch(USER_AGENT, "https://example.com/pet-care/barking")


def test_robots_empty_allows_everything() -> None:
    parser = parse_robots("")
    assert parser.can_fetch(USER_AGENT, "https://example.com/anything")


def test_robots_specific_agent_rule_does_not_hit_us() -> None:
    """다른 봇만 막는 규칙은 우리 UA에 적용되지 않아야 한다."""
    parser = parse_robots("User-agent: BadBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n")
    assert parser.can_fetch(USER_AGENT, "https://example.com/page")


def test_robots_blanket_disallow_blocks_us() -> None:
    parser = parse_robots("User-agent: *\nDisallow: /\n")
    assert not parser.can_fetch(USER_AGENT, "https://example.com/page")


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


class TestRawFileSelection:
    """정제본이 있으면 그쪽을 쓴다.

    정제한 자막이 통째로 버려지고 있었다. `bodeum-tv.refined.json`은 stem이
    `bodeum-tv.refined`라 sources.yaml에서 안 잡혀 건너뛰고, 옆에 있는
    오탈자투성이 원본이 대신 코퍼스로 들어갔다. **다듬어 놓고 안 다듬은 걸 썼다.**
    """

    def test_정제본이_원본을_이긴다(self, monkeypatch, tmp_path):
        from scripts.collect import normalize

        monkeypatch.setattr(normalize, "RAW_DIR", tmp_path)
        (tmp_path / "bodeum-tv.json").write_text("[]", encoding="utf-8")
        (tmp_path / "bodeum-tv.refined.json").write_text("[]", encoding="utf-8")

        chosen = normalize.raw_files()
        assert set(chosen) == {"bodeum-tv"}, "정제본이 별도 소스로 잡히면 안 된다"
        assert chosen["bodeum-tv"].name == "bodeum-tv.refined.json"

    def test_정제본만_있어도_소스로_잡힌다(self, monkeypatch, tmp_path):
        """다른 기기에는 정제본만 간다 — 저장소에 원본은 안 실려 있다."""
        from scripts.collect import normalize

        monkeypatch.setattr(normalize, "RAW_DIR", tmp_path)
        (tmp_path / "bodeum-tv.refined.json").write_text("[]", encoding="utf-8")

        assert normalize.raw_files()["bodeum-tv"].name == "bodeum-tv.refined.json"

    def test_정제본이_없으면_원본을_쓴다(self, monkeypatch, tmp_path):
        from scripts.collect import normalize

        monkeypatch.setattr(normalize, "RAW_DIR", tmp_path)
        (tmp_path / "rspca-dog-behaviour.json").write_text("[]", encoding="utf-8")

        chosen = normalize.raw_files()
        assert chosen["rspca-dog-behaviour"].name == "rspca-dog-behaviour.json"
