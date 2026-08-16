"""수집 파이프라인 공통 모델."""

import json
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


def save_raw(source_id: str, docs: list[RawDoc]) -> Path:
    """수집 결과를 `data/raw/<source_id>.json`에 쓴다.

    **`fetch._save`와 fetcher가 같은 함수를 쓰게 하려고 여기에 뒀다.** fetcher가
    중간 저장을 하려면 파일 형식이 최종 저장과 한 글자도 달라선 안 된다 —
    다르면 이어받기가 자기가 쓴 파일을 못 읽는다.

    임시 파일에 쓰고 바꿔치기한다. 214편을 받는 도중 매 편마다 덮어쓰는데,
    쓰는 중에 죽으면 파일이 깨져서 **그때까지 받은 걸 전부 잃기 때문**이다.
    이어받기를 만들어놓고 그 이어받기가 데이터를 파괴하면 의미가 없다.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{source_id}.json"
    tmp = out.with_suffix(".json.tmp")
    payload = [doc.model_dump() for doc in docs]
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    return out
