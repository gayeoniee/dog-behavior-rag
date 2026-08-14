"""코퍼스 탐색적 데이터 분석 (EDA).

`scripts.collect`·`scripts.eval`과 같은 UTF-8 처리를 한다 (한글 Windows 콘솔이
cp949라 ✓·█ 같은 문자에서 UnicodeEncodeError로 죽는다).
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
