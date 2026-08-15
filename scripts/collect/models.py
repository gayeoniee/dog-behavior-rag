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
BLOG_CORPUS_PATH = PROCESSED_DIR / "corpus_blogs.jsonl"
SOURCES_PATH = DATA_DIR / "sources.yaml"

Axis = Literal["problem", "cause", "training", "medical"]
Methodology = Literal["reward_based", "aversive", "mixed", "unknown"]
Volatility = Literal["stable", "volatile"]
CorpusPartition = Literal["answer", "observation"]


class Source(BaseModel):
    """sources.yaml 한 항목."""

    id: str
    name: str
    urls: list[str] = Field(default_factory=list)
    fetcher: Literal["pdf", "html", "local", "pmc", "youtube"]
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

    corpus: CorpusPartition = "answer"
    """observation = 답변 근거로 쓰지 않는 관찰용 구획.

    블로그처럼 지배이론이 섞일 수 있는 자료를 여기 둔다. normalize가 별도
    파일(corpus_blogs.jsonl)로 뽑고, 검색은 corpus == "answer"만 본다.
    """

    # ── pmc fetcher 전용 ──
    query: str | None = None
    """PMC E-utilities esearch term. urls 대신 이걸로 문서를 찾는다."""
    max_records: int = 50

    # ── youtube fetcher 전용 ──
    meta: dict = Field(default_factory=dict)
    """fetcher별 추가 설정. youtube는 `limit`(가져올 영상 수)과
    `markers`(제목에 이게 있으면 훈련 콘텐츠로 본다)를 읽는다.

    **채널 하나에 훈련 영상과 견종백과·여행 브이로그가 섞여 있어서 필요하다.**
    다 받으면 잡음이 코퍼스에 들어온다."""

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

    meta: dict = Field(default_factory=dict)
    """소스 기본 메타데이터를 문서 단위로 덮어쓸 값.

    PMC 소스 하나가 논문 수백 건을 내는데 published_at·license가 논문마다
    다르기 때문에 필요하다. normalize가 source 값 위에 이걸 덮어쓴다.
    pdf/html/local fetcher는 채우지 않으므로 기존 동작은 그대로다.
    """
