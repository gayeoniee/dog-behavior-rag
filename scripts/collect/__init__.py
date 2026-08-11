"""오프라인 데이터 수집 파이프라인.

앱 런타임(app/)과 분리된 배치 스크립트다. 흐름:

    fetch (sources.yaml → data/raw/)
      → normalize (data/raw/ → data/processed/corpus.jsonl)
      → report (corpus.jsonl 품질·커버리지 리포트)
"""

import sys

# 이 패키지의 스크립트들은 진단 출력에 ✓ ✗ ⚠️ 같은 기호를 쓴다. 한글 Windows의
# 기본 콘솔 인코딩(cp949)에는 이 문자들이 없어서 print가 UnicodeEncodeError로
# 죽는다 — 리포트 한 줄 때문에 파이프라인 전체가 멈춘다. import 시점에 UTF-8로
# 돌려둔다. errors="replace"는 재설정이 불가능한 환경에서도 크래시 대신 문자
# 하나가 깨지는 쪽으로 실패하게 만드는 보호막이다.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
