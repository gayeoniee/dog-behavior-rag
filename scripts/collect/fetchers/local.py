"""Local fetcher — 수동 다운로드한 파일을 처리한다.

로그인·동적 페이지라 자동 수집이 안 되는 소스(동물사랑배움터 등)용.
data/raw/local/<source_id>/ 아래에 .pdf 또는 .txt 를 넣어두면 읽는다.
"""

import logging
from datetime import UTC, datetime

from ..models import LOCAL_DIR, RawDoc, Source
from .pdf import extract_pdf_text

logger = logging.getLogger(__name__)


class LocalFetcher:
    async def fetch(self, source: Source) -> list[RawDoc]:
        # 수동으로 받은 파일이므로 라이선스는 받는 시점에 사람이 판단했다고 본다.
        # (sources.yaml의 license 필드에 근거를 남길 것)
        directory = LOCAL_DIR / source.id
        if not directory.is_dir():
            logger.warning(
                "%s 없음 — 수동 다운로드 파일을 %s 에 넣어주세요 (.pdf/.txt/.md)",
                directory,
                directory,
            )
            return []

        docs: list[RawDoc] = []
        fetched_at = datetime.now(UTC).isoformat()
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() == ".pdf":
                text = extract_pdf_text(path.read_bytes())
            elif path.suffix.lower() in (".txt", ".md"):
                text = path.read_text(encoding="utf-8")
            else:
                logger.info("지원하지 않는 확장자, 건너뜀: %s", path.name)
                continue

            if not text.strip():
                logger.warning("빈 텍스트(스캔본 PDF일 수 있음): %s", path.name)

            docs.append(
                RawDoc(
                    source_id=source.id,
                    url=f"file://{path}",
                    title=path.stem,
                    text=text,
                    fetched_at=fetched_at,
                )
            )
        return docs
