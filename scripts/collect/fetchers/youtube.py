"""유튜브 fetcher — 자막을 우선 쓰고, 없으면 음성을 STT로 옮긴다.

**왜 자막이 먼저인가.** 한국어 채널은 대부분 자동 생성 자막이 있고, 그러면 STT가
아예 필요 없다. 실측(강형욱의 보듬TV): 확인한 영상 전부 `ko` 자동자막이 있었다.

    자막   영상당 몇 초 · yt-dlp만 있으면 된다
    STT    영상당 수 분 · ffmpeg/Whisper 모델까지 필요하다

**이 fetcher는 원문을 그대로 담는다.** 자동자막은 구어체에 오탈자가 많고
("수의사"→"수혜사", "훈련사"→"홀련 선생님") 인사·리액션이 절반이라 그대로는
검색 자료로 쓰기 나쁘다. 다만 **정제는 여기서 하지 않는다** —
`scripts.collect.refine_transcripts`가 LLM으로 따로 다듬는다.

이유는 `data/raw`와 `data/processed`를 나눈 것과 같다: **원본은 비싸게 받아오고
가공은 언제든 다시 한다.** 정제 프롬프트를 고칠 때마다 유튜브를 다시 긁으면 안 된다.
"""

import asyncio
import json
import logging
import time
import urllib.request
from datetime import UTC, datetime

from ..models import RawDoc, Source
from .base import ensure_license_checked
from .http import REQUEST_DELAY_SECONDS

logger = logging.getLogger(__name__)

# 이 문구가 제목에 있으면 훈련·상담 콘텐츠로 본다. 같은 채널에 견종백과·여행
# 브이로그·연예인 게스트가 섞여 있어서 그냥 다 받으면 잡음이 들어온다.
#
# **영어와 한국어를 둘 다 넣는다.** 처음에 한국어 마커만 넣었다가 120개 중 1개만
# 걸렀다 — 이 채널은 제목을 영어로 번역해 올린다("[Puppy Training]" 22개).
# 마커는 상상이 아니라 **실제 제목 분포를 보고** 정해야 한다.
TRAINING_MARKERS = (
    "Puppy Training", "Puppy training", "Training for puppy",
    "Junior Training", "Dog Talk", "Training",
    "퍼피교육", "문제행동", "훈련", "교육", "상담", "행동교정",
)

VIDEO_DELAY_SECONDS = REQUEST_DELAY_SECONDS
"""영상 하나를 처리할 때마다 쉬는 시간.

**214편을 연속으로 요청했다가 429(Too Many Requests)로 차단당했다.** 영상마다
메타데이터 1회 + 자막 1회, 총 400회 넘는 요청이 순식간에 나간다.
`http.PoliteClient`가 HTML 수집에 쓰는 것과 같은 값을 쓴다 — 수집 대상 서버에
부담을 주지 않는다는 원칙은 유튜브에도 똑같이 적용된다.
"""

MIN_CHARS = 1500
"""이보다 짧은 자막은 버린다. 8분 상담 영상이 약 4,000자였으니 1,500자 미만은
내용이 거의 없는 것이다 (쇼츠·예고편 등)."""


def _extract_text(url: str) -> str:
    """json3 자막을 이어붙인다.

    자막은 화면 표시 단위로 쪼개져 있어서 문장 경계가 없다. 여기서는 그대로
    이어 붙이기만 하고, 문장 복원은 정제 단계에 맡긴다.
    """
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    parts: list[str] = []
    for event in payload.get("events") or []:
        text = "".join(seg.get("utf8", "") for seg in (event.get("segs") or [])).strip()
        if text and text != "\n":
            parts.append(text)
    return " ".join(parts)


def looks_like_training(title: str, markers: tuple[str, ...]) -> bool:
    return any(m in title for m in markers)


class YoutubeFetcher:
    """채널에서 훈련·상담 영상을 골라 자막을 가져온다.

    `Source.urls`에 채널 URL을, `meta`에 `limit`·`markers`를 넣는다.
    """

    async def fetch(self, source: Source) -> list[RawDoc]:
        ensure_license_checked(source)
        return await asyncio.to_thread(self._fetch, source)

    def _fetch(self, source: Source) -> list[RawDoc]:
        import yt_dlp

        meta = source.meta if isinstance(getattr(source, "meta", None), dict) else {}
        limit = int(meta.get("limit", 30))
        markers = tuple(meta.get("markers") or TRAINING_MARKERS)

        # 재생목록은 통째로 받고, 채널은 제목 필터로 걸러야 하므로 넉넉히 훑는다.
        curated_urls = any("list=" in u for u in source.urls)
        flat = {
            "quiet": True,
            "extract_flat": True,
            "skip_download": True,
            "playlistend": limit if curated_urls else limit * 4,
        }
        entries: list[dict] = []
        with yt_dlp.YoutubeDL(flat) as ydl:
            for url in source.urls:
                info = ydl.extract_info(url, download=False)
                entries.extend(info.get("entries") or [])

        # **재생목록은 제목으로 거르지 않는다.** 사람이 이미 골라놓은 묶음이라
        # 필터가 오히려 방해가 된다 — 이 채널은 제목이 영어로 번역돼 있어서
        # 한국어 마커로 거르면 대부분 날아간다(120개 중 1개만 남았던 적이 있다).
        curated = any("list=" in u for u in source.urls)
        picked = (
            entries
            if curated
            else [e for e in entries if looks_like_training(e.get("title") or "", markers)]
        )
        logger.info(
            "영상 %d개 중 대상 %d개 (%s)",
            len(entries),
            len(picked),
            "재생목록이라 전부" if curated else "제목 필터",
        )

        # 이미 받아둔 영상은 건너뛴다 — 429로 중간에 끊겨도 이어서 받을 수 있게.
        previous = self._existing_docs(source)
        already = {d.source_id for d in previous}
        if already:
            logger.info("이미 받은 %d편은 건너뛰고 결과에 합친다", len(already))

        docs: list[RawDoc] = list(previous)
        detail = {"quiet": True, "skip_download": True}
        with yt_dlp.YoutubeDL(detail) as ydl:
            for entry in picked[:limit]:
                video_id = entry.get("id")
                if f"{source.id}-{video_id}" in already:
                    continue
                try:
                    info = ydl.extract_info(
                        f"https://www.youtube.com/watch?v={video_id}", download=False
                    )
                except Exception as exc:  # noqa: BLE001 — 한 편이 막혀도 나머지는 계속
                    logger.warning("영상 %s 정보 실패: %s", video_id, exc)
                    continue

                try:
                    text = self._transcript(info)
                except Exception as exc:  # noqa: BLE001 — 429 하나가 전체를 죽이면 안 된다
                    logger.warning("영상 %s 자막 실패: %s", video_id, exc)
                    if "429" in str(exc):
                        # 차단당하면 더 두드려봐야 소용없다. 여기까지 모은 걸 돌려준다.
                        logger.warning(
                            "429로 중단 — 누적 %d건. 잠시 뒤 다시 실행하면 이어서 받는다",
                            len(docs),
                        )
                        break
                    continue

                if len(text) < MIN_CHARS:
                    logger.info("영상 %s 자막이 짧아 제외 (%d자)", video_id, len(text))
                    continue

                docs.append(
                    RawDoc(
                        source_id=f"{source.id}-{video_id}",
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        title=str(info.get("title") or video_id),
                        text=text,
                        fetched_at=datetime.now(UTC).isoformat(),
                        meta={"published_at": (info.get("upload_date") or "")[:4]},
                    )
                )
                logger.info("✓ %s (%d자) %s", video_id, len(text), info.get("title", "")[:40])
                time.sleep(VIDEO_DELAY_SECONDS)
        return docs

    def _existing_docs(self, source: Source) -> list[RawDoc]:
        """이전 실행에서 받아둔 문서들.

        **429로 끊기는 게 정상 시나리오라** 이어받기가 필요하다. 그런데 `fetch`가
        결과 파일을 **덮어쓰므로**, 새로 받은 것만 돌려주면 이전 것이 지워진다.
        이어받기가 오히려 데이터를 파괴하는 셈이라 **기존 문서를 함께 돌려준다.**
        """
        from ..models import RAW_DIR

        path = RAW_DIR / f"{source.id}.json"
        if not path.is_file():
            return []
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            return [RawDoc.model_validate(r) for r in rows]
        except Exception:  # noqa: BLE001 — 깨진 파일이면 새로 받는다
            logger.warning("기존 %s 를 읽지 못해 처음부터 받는다", path.name)
            return []

    def _transcript(self, info: dict) -> str:
        """자막 → (없으면) STT. **자막이 있으면 STT를 부르지 않는다.**"""
        captions = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}
        for pool in (captions, auto):
            for lang in ("ko", "ko-orig"):
                formats = pool.get(lang) or []
                chosen = next((f for f in formats if f.get("ext") == "json3"), None)
                if chosen:
                    return _extract_text(chosen["url"])
        logger.info("자막이 없어 STT로 넘어간다: %s", info.get("id"))
        return self._speech_to_text(info)

    def _speech_to_text(self, info: dict) -> str:
        """faster-whisper로 음성을 옮긴다.

        **PyAV로 디코딩하므로 시스템 ffmpeg이 없어도 된다** — 이 PC는 관리자 권한이
        없어 ffmpeg 설치가 번거롭다.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.warning("faster-whisper가 없어 STT를 건너뛴다 (uv pip install faster-whisper)")
            return ""

        import tempfile
        from pathlib import Path

        import yt_dlp

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "audio.m4a"
            # ffmpeg 없이 받으려면 이미 단일 스트림인 m4a를 고른다 (후처리 불필요).
            opts = {"quiet": True, "format": "m4a/bestaudio[ext=m4a]", "outtmpl": str(target)}
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={info.get('id')}"])
            if not target.exists():
                logger.warning("오디오 내려받기 실패: %s", info.get("id"))
                return ""

            model = WhisperModel("medium", device="auto", compute_type="int8")
            segments, _ = model.transcribe(str(target), language="ko", vad_filter=True)
            return " ".join(s.text.strip() for s in segments)
