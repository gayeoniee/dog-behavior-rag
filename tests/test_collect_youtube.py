"""유튜브 fetcher의 이어받기 — **데이터를 잃지 않는 것**만 본다.

실제 유튜브를 부르지 않는다. 여기서 검증하려는 건 네트워크가 아니라
**중단됐을 때 그때까지 받은 게 남는가**이기 때문이다.

이 테스트가 있는 이유: 429로 두 번 끊겼는데 두 번 다 0건이 남았다. 저장을
`fetch()`가 끝난 뒤 한 번만 했기 때문인데, 그러면 다시 돌릴 때 1번부터
요청해서 이미 받은 영상에 IP 쿼터를 또 쓴다. 429를 스스로 부르는 구조였다.
"""

import json
import sys
import types

import pytest

from scripts.collect import models
from scripts.collect.fetchers import youtube
from scripts.collect.models import RawDoc, Source


def make_source(**meta) -> Source:
    return Source(
        id="test-yt",
        name="테스트",
        urls=["https://www.youtube.com/playlist?list=PLtest"],
        fetcher="youtube",
        language="ko",
        species="dog",
        axis=["training"],
        authority_tier=2,
        methodology="reward_based",
        published_at=2024,
        volatility="stable",
        license="test",
        license_checked_at="2026-08-16",
        corpus="observation",
        meta=meta,
    )


class FakeYDL:
    """yt-dlp 자리에 끼우는 최소 구현.

    `entries`는 재생목록 조회에, `videos`는 개별 조회에 쓴다.
    """

    entries: list[dict] = []
    videos: dict[str, dict] = {}
    cookiejar = None
    params: dict = {}

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        if "list=" in url:
            return {"entries": self.entries}
        vid = url.rsplit("=", 1)[-1]
        return self.videos[vid]


@pytest.fixture
def fake_yt(monkeypatch, tmp_path):
    """yt_dlp 모듈과 저장 경로를 통째로 갈아끼운다."""
    monkeypatch.setattr(models, "RAW_DIR", tmp_path)
    monkeypatch.setattr(youtube, "SUBTITLE_DELAY_RANGE", (0.0, 0.0))
    monkeypatch.setattr(youtube, "VIDEO_DELAY_RANGE", (0.0, 0.0))
    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = FakeYDL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    return module


def stub_subtitles(monkeypatch, texts: dict[str, str], fail_on: str | None = None):
    """`_extract_text`를 영상별 고정 문자열로 대체한다."""

    def fake(url, http=None):
        vid = url
        if fail_on and vid == fail_on:
            raise RuntimeError("HTTP Error 429: Too Many Requests")
        return texts[vid]

    monkeypatch.setattr(youtube, "_extract_text", fake)


def video(vid: str) -> dict:
    """`_transcript`가 자막 URL로 영상 id를 그대로 쓰게 만든 가짜 정보."""
    return {
        "id": vid,
        "title": f"제목 {vid}",
        "upload_date": "20240101",
        "automatic_captions": {"ko": [{"ext": "json3", "url": vid}]},
    }


@pytest.mark.asyncio
async def test_중간에_429가_나도_그전까지_받은_건_파일에_남는다(fake_yt, monkeypatch, tmp_path):
    FakeYDL.entries = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    FakeYDL.videos = {v: video(v) for v in "abc"}
    stub_subtitles(monkeypatch, {"a": "가" * 500, "b": "나" * 500}, fail_on="c")

    docs = await youtube.YoutubeFetcher().fetch(make_source(limit=10, min_chars=400))

    assert [d.source_id for d in docs] == ["test-yt-a", "test-yt-b"]
    saved = json.loads((tmp_path / "test-yt.json").read_text(encoding="utf-8"))
    assert [r["source_id"] for r in saved] == ["test-yt-a", "test-yt-b"], (
        "429로 끊기기 전에 받은 문서가 파일에 남아 있어야 한다"
    )


@pytest.mark.asyncio
async def test_다시_돌리면_이미_받은_편은_건너뛰고_합쳐진다(fake_yt, monkeypatch, tmp_path):
    """이어받기의 핵심: **덮어쓰기가 이전 결과를 지우면 안 된다.**"""
    models.save_raw(
        "test-yt",
        [
            RawDoc(
                source_id="test-yt-a",
                url="https://www.youtube.com/watch?v=a",
                title="제목 a",
                text="가" * 500,
                fetched_at="2026-08-16T00:00:00+00:00",
            )
        ],
    )
    FakeYDL.entries = [{"id": "a"}, {"id": "b"}]
    FakeYDL.videos = {v: video(v) for v in "ab"}

    requested: list[str] = []

    def fake(url, http=None):
        requested.append(url)
        return "나" * 500

    monkeypatch.setattr(youtube, "_extract_text", fake)

    docs = await youtube.YoutubeFetcher().fetch(make_source(limit=10, min_chars=400))

    assert requested == ["b"], "이미 받은 a에 다시 요청을 쓰면 안 된다"
    assert {d.source_id for d in docs} == {"test-yt-a", "test-yt-b"}


@pytest.mark.asyncio
async def test_min_chars는_소스마다_덮어쓸_수_있다(fake_yt, monkeypatch, tmp_path):
    """2~3분짜리 Q&A는 700~900자가 정상이라 기본값을 낮춰야 했다."""
    FakeYDL.entries = [{"id": "a"}]
    FakeYDL.videos = {"a": video("a")}
    stub_subtitles(monkeypatch, {"a": "가" * 700})

    # **이어받기가 켜져 있으므로 임계값만 바꿔 두 번 부르면 안 된다** —
    # 두 번째 호출이 첫 번째가 저장한 문서를 그대로 합쳐서, 걸러졌는지
    # 알 수 없게 된다. 실제로 이 테스트를 그렇게 짰다가 헛통과할 뻔했다.
    high = make_source(limit=10, min_chars=1500)
    high.id = "test-yt-high"
    assert not await youtube.YoutubeFetcher().fetch(high)
    assert await youtube.YoutubeFetcher().fetch(make_source(limit=10, min_chars=400))


def test_저장은_임시파일을_거친다(monkeypatch, tmp_path):
    """매 편마다 덮어쓰므로, 쓰는 도중 죽어도 이전 내용이 살아 있어야 한다."""
    monkeypatch.setattr(models, "RAW_DIR", tmp_path)
    doc = RawDoc(
        source_id="x",
        url="u",
        title="t",
        text="본문",
        fetched_at="2026-08-16T00:00:00+00:00",
    )
    out = models.save_raw("test-yt", [doc])
    assert out.read_text(encoding="utf-8").strip().startswith("[")
    assert not (tmp_path / "test-yt.json.tmp").exists(), "임시 파일은 남지 않아야 한다"
