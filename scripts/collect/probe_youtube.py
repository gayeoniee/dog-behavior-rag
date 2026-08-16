"""차단이 풀렸는지 **요청 한 번으로** 확인한다.

    uv run python -m scripts.collect.probe_youtube

429를 맞은 뒤 우리가 반복한 실수는 **확인을 수집으로 했다는 것**이다. 풀렸나
보려고 214편짜리 수집을 돌리고, 아직 막혀 있으면 그 시도가 차단을 연장했다.
확인은 싸야 한다 — 자막 1건이면 충분하다.

무엇을 보는가:

    메타데이터   막히면 여기부터 실패한다 (지금까지는 계속 200이었다)
    자막         실제로 막힌 곳. 여기가 200이면 수집을 시작해도 된다

두 줄이 따로 나오는 게 중요하다. "유튜브가 안 된다"와 "자막만 안 된다"는
완전히 다른 상황인데, 수집 로그만 봐서는 구분이 안 됐다.
"""

import argparse
import sys
import urllib.error

# 이 import가 패키지 __init__을 태워 콘솔을 UTF-8로 돌려놓는다 (cp949 크래시 방지).
from .fetchers.youtube import JS_RUNTIMES, _extract_text, _http_from

DEFAULT_VIDEO = "8s3ybEgTpQY"
"""보듬TV Q&A 재생목록의 한 편. 짧아서 자막도 가볍다.

특정 영상이 지워질 수 있으므로 `--video`로 바꿀 수 있게 뒀다.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="유튜브 자막 차단 여부 확인 (요청 1~2회)")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="확인에 쓸 영상 id")
    args = parser.parse_args()

    import yt_dlp

    url = f"https://www.youtube.com/watch?v={args.video}"
    opts = {"quiet": True, "skip_download": True, "js_runtimes": JS_RUNTIMES}

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as exc:  # noqa: BLE001 — 원인을 그대로 보여주는 게 목적이다
            print(f"메타데이터  실패 — {exc}")
            return 1
        print(f"메타데이터  OK — {str(info.get('title') or '')[:40]}")

        auto = info.get("automatic_captions") or {}
        formats = auto.get("ko") or auto.get("ko-orig") or []
        chosen = next((f for f in formats if f.get("ext") == "json3"), None)
        if not chosen:
            print("자막        ko 자동자막이 없다 (다른 영상으로 --video 지정)")
            return 1

        try:
            # 지연 0 — 한 건만 받으므로 기다릴 이유가 없다.
            text = _extract_text(chosen["url"], _http_from(ydl), (0.0, 0.0))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                print("자막        429 — 아직 막혀 있다. 수집을 시작하지 말 것")
            else:
                print(f"자막        HTTP {exc.code} {exc.reason}")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"자막        실패 — {exc}")
            return 1

    print(f"자막        OK — {len(text)}자")
    print()
    print("차단이 풀렸다. 수집을 시작해도 된다:")
    print("    uv run python -m scripts.collect.fetch --source bodeum-qna")
    return 0


if __name__ == "__main__":
    sys.exit(main())
