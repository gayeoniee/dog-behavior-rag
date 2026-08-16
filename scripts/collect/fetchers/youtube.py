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
import random
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import NamedTuple

from ..models import RawDoc, Source, save_raw
from .base import ensure_license_checked

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

SUBTITLE_DELAY_RANGE = (3.0, 7.0)
"""자막을 받기 전 대기 시간의 범위 (초).

**yt-dlp의 `sleep_interval_subtitles`가 우리 경로에서는 발동하지 않으므로**
직접 한다. 차단이 자막(timedtext) 엔드포인트에만 걸렸으니 여기가 가장 중요하다.

**범위로 두는 이유:** 고정 간격은 봇의 일정한 심박으로 읽힌다. 사람이 브라우저로
볼 때 요청 간격이 일정할 리가 없다.
"""

VIDEO_DELAY_RANGE = (1.0, 3.0)
"""영상 하나를 처리한 뒤 **추가로** 쉬는 시간의 범위.

yt-dlp의 `sleep_interval_requests`가 요청 사이를 벌려주므로 이건 보조다.
여기도 고정값이 아니라 범위다 — 같은 이유.

**214편을 연속으로 요청했다가 429(Too Many Requests)로 차단당했다.** 영상마다
메타데이터 1회 + 자막 1회, 총 400회 넘는 요청이 순식간에 나간다.
`http.PoliteClient`가 HTML 수집에 쓰는 것과 같은 값을 쓴다 — 수집 대상 서버에
부담을 주지 않는다는 원칙은 유튜브에도 똑같이 적용된다.

**차단당한 뒤 확인한 것 (2026-08-15) — 다시 헤매지 않도록 적어둔다:**

    메타데이터 요청        정상 200   ← 차단은 자막 엔드포인트에만 걸린다
    백오프 재시도 65초     여전히 429
    yt-dlp 자체 다운로드   여전히 429  ← **구현 방식 탓이 아니다**
    Node JS 런타임 켜기    여전히 429
    하루 가까이 경과       여전히 429

마지막 줄이 중요하다. 처음엔 "urllib로 직접 받아서 yt-dlp의 위장·쿠키를 우회한
탓"이라고 의심했는데, yt-dlp에게 시켜도 똑같이 막혔다. **IP 단위 차단이고 시간이
지나야 풀린다.** 코드를 고쳐서 뚫을 수 있는 게 아니므로 시도하지 말 것.

그래서 진짜 대책은 **처음부터 천천히 받는 것**이다. 이 지연값을 줄이지 말 것.

**우회로도 다 막혀 있다 (2026-08-16 실측). 아래는 시도하지 말 것:**

    자막 android 클라이언트   429   ← 쿼터는 클라이언트가 아니라 IP 단위다
    자막 ios / tv 클라이언트  포맷 자체를 못 받음
    오디오 다운로드(STT 우회) 403   ← n challenge / SABR / PO Token이 또 필요
    오디오 + EJS 해석기       403   여전히

세 번째 줄이 뼈아프다. **자막이 막히면 STT로 우회하면 된다고 설계했는데 그 길도
막혀 있다.** 오디오 스트림은 timedtext와 다른 호스트(googlevideo)인데도 그렇다.

정리하면 **막힌 것은 엔드포인트가 아니라 "로그인하지 않은 우리 IP"다.** 그래서
남은 지렛대는 지연값이 아니라 **신원**뿐이다 — `Http`와 `meta.cookies_from_browser`
참조. 지연을 더 늘리는 여덟 번째 시도는 하지 말 것.
"""

THROTTLE: dict[str, object] = {
    # **이 경로에서 실제로 발동하는 유일한 sleep 옵션이다.**
    # extractor/common.py `_request_webpage`가 매 요청 전에 잠든다 —
    # 메타데이터 조회를 포함한 yt-dlp의 모든 웹 요청에 걸린다.
    "sleep_interval_requests": 2,
    "retries": 3,
    "extractor_retries": 2,
}
"""요청 속도 제한. **이게 없어서 429를 맞았다.**

처음엔 영상 사이에 `time.sleep(1)`만 뒀는데, **영상 하나를 처리할 때 yt-dlp가
내부적으로 여러 번 요청한다.** 214편이면 순식간에 400회가 넘어간다.

**⚠️ 넣어봐야 소용없는 옵션들 (2026-08-16 소스 확인):**

    sleep_interval / max_sleep_interval   다운로드 **직전**에만 발동
                                          → skip_download라 안 걸린다
    sleep_interval_subtitles              yt-dlp가 자막을 받을 때만 발동
                                          → 우린 urllib로 직접 받으므로 안 걸린다

처음엔 이 셋을 다 넣고 "3~7초 무작위 대기가 걸린다"고 적었는데 **거짓이었다.**
옵션 이름만 보고 넣었지 **언제 발동하는지 확인하지 않았다.** 대신 자막 요청 전
대기는 `_sleep_before_subtitle`이 직접 한다.
"""

JS_RUNTIMES: dict[str, dict] = {"node": {}}
"""yt-dlp가 쓸 자바스크립트 런타임.

yt-dlp는 기본적으로 deno만 켠다. 이 PC에는 Next.js 화면 때문에 **Node가 이미
깔려 있으므로** 그걸 쓰게 한다. 안 켜면 실행마다 이런 경고가 뜬다:

    No supported JavaScript runtime could be found.
    YouTube extraction without a JS runtime has been deprecated.

⚠️ **이걸 켜도 429는 안 풀린다** (2026-08-16 실측). 차단과는 무관하고,
앞으로 yt-dlp가 JS 런타임을 필수로 요구할 때를 대비하는 것이다.
"""

MIN_CHARS = 400
"""이보다 짧은 자막은 버린다. `meta.min_chars`로 소스마다 덮어쓸 수 있다.

**처음에 1,500으로 잡았다가 원하는 콘텐츠를 거의 다 걸렀다.** 8분짜리 상담 영상
(약 4,000자)을 기준으로 정한 값인데, "소소한 Q&A" 재생목록은 **편당 2~3분**이라
자막이 700~900자가 정상이다. 실행 로그가 "자막이 짧아 제외"로 도배됐다.

**임계값은 콘텐츠 형식마다 다르다.** 한 소스에서 정한 값을 다른 소스에 그대로
쓰면 안 된다 — 03장에서 논문 문단(506자)과 가이드 문단(87자)이 5.8배 달랐던 것과
같은 이야기다.
"""


# 자막 요청이 429를 맞았을 때 기다릴 시간. 지수적으로 늘린다.
RETRY_DELAYS = (5.0, 15.0, 45.0)
"""**차단은 자막 엔드포인트에만 걸린다** (2026-08-15 실측 — 메타데이터는 계속
200이 오는데 자막만 429였다). 몇 초 기다리면 풀리는 경우가 있어 재시도한다.

세 번 다 실패하면 포기한다. 그 이상 두드리는 건 차단을 길게 만들 뿐이다.
"""


class Http(NamedTuple):
    """자막을 받을 때 쓸 신원.

    **자막 요청이 메타데이터 요청과 다른 신원으로 나가고 있었다.** 메타데이터는
    yt-dlp가 브라우저 헤더·쿠키를 달아 보내는데, 자막은 우리가 맨 urllib으로
    받으면서 `User-Agent: Python-urllib/3.12`를 스스로 광고하고 쿠키도 없었다.

    같은 세션에서 **메타데이터는 200, 자막만 429**가 나오는 상태를 오래 봤는데,
    그 둘이 서버 눈에 다른 클라이언트였던 것이다. yt-dlp가 쓰는 것을 그대로
    쓰게 한다.
    """

    opener: urllib.request.OpenerDirector
    headers: dict[str, str]


def _delay_range(value: object, fallback: tuple[float, float]) -> tuple[float, float]:
    """`meta`의 지연 설정을 (최소, 최대)로 읽는다. 숫자 하나면 고정 간격이 아니라
    그 값을 중심으로 ±25% 흔든다 — 일정한 간격은 봇의 심박으로 읽힌다."""
    if isinstance(value, list | tuple) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    if isinstance(value, int | float):
        return (float(value) * 0.75, float(value) * 1.25)
    return fallback


def _http_from(ydl: object) -> Http:
    jar = getattr(ydl, "cookiejar", None)
    opener = (
        urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        if jar is not None
        else urllib.request.build_opener()
    )
    params = getattr(ydl, "params", {}) or {}
    headers = dict(params.get("http_headers") or {})
    headers.setdefault("Accept-Language", "ko-KR,ko;q=0.9,en-US;q=0.8")
    return Http(opener, headers)


def _sleep_before_subtitle(delay: tuple[float, float] = SUBTITLE_DELAY_RANGE) -> None:
    """자막 요청 전 무작위 대기.

    yt-dlp의 `sleep_interval_subtitles`는 yt-dlp가 자막을 내려받을 때만 발동하는데
    우리는 URL만 받아 urllib로 직접 가져오므로 안 걸린다. 그래서 직접 한다.
    """
    time.sleep(random.uniform(*delay))


def _extract_text(
    url: str,
    http: Http | None = None,
    delay: tuple[float, float] = SUBTITLE_DELAY_RANGE,
) -> str:
    """json3 자막을 이어붙인다.

    자막은 화면 표시 단위로 쪼개져 있어서 문장 경계가 없다. 여기서는 그대로
    이어 붙이기만 하고, 문장 복원은 정제 단계에 맡긴다.

    **yt-dlp를 거치지 않고 직접 받는다.** 그래서 yt-dlp가 해주는 재시도·백오프가
    없어 여기서 직접 해야 한다.
    """
    _sleep_before_subtitle(delay)
    opener = http.opener if http else urllib.request.build_opener()
    request = urllib.request.Request(url, headers=dict(http.headers) if http else {})
    last: Exception | None = None
    for attempt, backoff in enumerate((0.0, *RETRY_DELAYS)):
        if backoff:
            logger.info("자막 429 — %.0f초 후 재시도 (%d/%d)", backoff, attempt, len(RETRY_DELAYS))
            time.sleep(backoff)
        try:
            with opener.open(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code != 429:
                raise
    else:
        raise last if last else RuntimeError("자막 요청 실패")

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
        min_chars = int(meta.get("min_chars", MIN_CHARS))

        # 로그인 상태로 받게 한다 — 익명 요청의 쿼터는 IP 단위로 공유되지만
        # 로그인 요청은 계정 단위로 잡히고 한도가 훨씬 넉넉하다.
        # `meta.cookies_from_browser: chrome` 처럼 지정한다. **아직 검증 안 됨.**
        browser = meta.get("cookies_from_browser")
        cookies = {"cookiesfrombrowser": (str(browser),)} if browser else {}

        # **속도는 소스마다 다르게 잡아야 한다.** 26편을 편당 1.7초로 받고
        # 차단당했다. 214편짜리 재생목록은 훨씬 느리게 가야 한다 —
        # `subtitle_delay: [40, 60]`이면 3시간에 걸쳐 받는다.
        subtitle_delay = _delay_range(meta.get("subtitle_delay"), SUBTITLE_DELAY_RANGE)
        video_delay = _delay_range(meta.get("video_delay"), VIDEO_DELAY_RANGE)
        request_delay = meta.get("request_delay")
        throttle: dict[str, object] = dict(THROTTLE)
        if request_delay:
            throttle["sleep_interval_requests"] = request_delay

        # 재생목록은 통째로 받고, 채널은 제목 필터로 걸러야 하므로 넉넉히 훑는다.
        curated_urls = any("list=" in u for u in source.urls)
        flat = {
            "quiet": True,
            "extract_flat": True,
            "skip_download": True,
            "playlistend": limit if curated_urls else limit * 4,
            "js_runtimes": JS_RUNTIMES,
            **throttle,
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
        detail = {
            "quiet": True,
            "skip_download": True,
            "js_runtimes": JS_RUNTIMES,
            **throttle,
            **cookies,
        }
        with yt_dlp.YoutubeDL(detail) as ydl:
            http = _http_from(ydl)
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
                    text = self._transcript(info, http, subtitle_delay)
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

                if len(text) < min_chars:
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
                # **한 편 받을 때마다 저장한다.** 마지막에 한 번만 저장하면
                # Ctrl+C나 터미널 종료로 그때까지 받은 게 전부 사라진다.
                # 그러면 다시 돌릴 때 1번부터 요청해서, 이미 받았던 영상에
                # IP 쿼터를 또 쓴다 — 429를 스스로 부르는 셈이다.
                save_raw(source.id, docs)
                logger.info("✓ %s (%d자) %s", video_id, len(text), info.get("title", "")[:40])
                time.sleep(random.uniform(*video_delay))
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

    def _transcript(
        self,
        info: dict,
        http: Http | None = None,
        delay: tuple[float, float] = SUBTITLE_DELAY_RANGE,
    ) -> str:
        """자막 → (없으면) STT. **자막이 있으면 STT를 부르지 않는다.**"""
        captions = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}
        for pool in (captions, auto):
            for lang in ("ko", "ko-orig"):
                formats = pool.get(lang) or []
                chosen = next((f for f in formats if f.get("ext") == "json3"), None)
                if chosen:
                    return _extract_text(chosen["url"], http, delay)
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
