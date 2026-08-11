"""PDF fetcher — URL에서 PDF를 받아 텍스트를 추출한다."""

import io
import logging
from datetime import UTC, datetime

from pypdf import PdfReader

from ..models import RawDoc, Source
from .base import ensure_license_checked
from .http import PoliteClient

logger = logging.getLogger(__name__)


def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


class PdfFetcher:
    async def fetch(self, source: Source) -> list[RawDoc]:
        ensure_license_checked(source)

        docs: list[RawDoc] = []
        async with PoliteClient() as client:
            for url in source.urls:
                logger.info("PDF 수집: %s", url)
                response = await client.get(url)
                text = extract_pdf_text(response.content)
                if not text:
                    logger.warning("텍스트 추출 실패(스캔본일 수 있음): %s", url)
                docs.append(
                    RawDoc(
                        source_id=source.id,
                        url=url,
                        title=source.name,
                        text=text,
                        fetched_at=datetime.now(UTC).isoformat(),
                    )
                )
        return docs
