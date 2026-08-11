"""수집 파이프라인 공통 모델."""

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
LOCAL_DIR = RAW_DIR / "local"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl"
SOURCES_PATH = DATA_DIR / "sources.yaml"

Axis = Literal["problem", "cause", "training", "medical"]
Methodology = Literal["reward_based", "aversive", "mixed", "unknown"]
Volatility = Literal["stable", "volatile"]


class Source(BaseModel):
    """sources.yaml 한 항목."""

    id: str
    name: str
    urls: list[str] = Field(default_factory=list)
    fetcher: Literal["pdf", "html", "local"]
    language: Literal["ko", "en"]
    species: Literal["dog", "cat", "both"]
    axis: list[Axis]
    authority_tier: int = Field(ge=1, le=3)
    methodology: Methodology
    published_at: int
    volatility: Volatility
    superseded_checked_at: date | None = None
    license: str
    license_checked_at: date | None = None

    @property
    def license_pending(self) -> bool:
        return self.license == "pending-check"


class RawDoc(BaseModel):
    """fetcher가 뽑아낸 문서 1건 (정규화 전)."""

    source_id: str
    url: str
    title: str
    text: str
    fetched_at: str  # ISO 8601
