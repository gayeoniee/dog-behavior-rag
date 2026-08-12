"""검색 품질 평가.

`scripts.collect`·`scripts.db`와 같은 UTF-8 처리를 한다 (한글 Windows 콘솔이
cp949라 ✓·⚠️ 같은 기호에서 UnicodeEncodeError로 죽는다).
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
