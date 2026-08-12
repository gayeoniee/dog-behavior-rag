"""문서 적재 스키마.

메타데이터 필드는 corpus.jsonl(scripts/collect/normalize.py 산출물)과 맞춘다.
axis / methodology / authority_tier는 장식이 아니라 검색 품질 장치다:
- methodology=aversive 문서는 검색에서 제외
- authority_tier 낮은(권위 높은) 문서를 부스팅
- axis는 코퍼스 커버리지 측정용
"""

from typing import Literal

from pydantic import BaseModel, Field

Axis = Literal["problem", "cause", "training", "medical"]
Methodology = Literal["reward_based", "aversive", "mixed", "unknown"]
Distribution = Literal["open", "personal-only"]
CorpusPartition = Literal["answer", "observation"]


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=500, description="문서 제목")
    content: str = Field(min_length=1, description="문서 본문")
    source: str | None = Field(default=None, max_length=500, description="출처 URL/파일명")
    category: str | None = Field(
        default=None,
        max_length=100,
        description="분류 (예: 짖음, 분리불안, 배변훈련)",
    )

    # ── 코퍼스 메타데이터 (corpus.jsonl과 동일 스키마) ──
    language: Literal["ko", "en"] = Field(default="ko", description="문서 언어")
    species: Literal["dog", "cat", "both"] = Field(default="dog", description="대상 동물")
    axis: list[Axis] = Field(
        default_factory=list,
        description="답변 축: problem(문제행동)/cause(이유)/training(훈련)/medical(의학적 감별)",
    )
    methodology: Methodology = Field(
        default="unknown",
        description="훈련 방법론. aversive는 검색에서 제외된다",
    )
    authority_tier: int = Field(
        default=3,
        ge=1,
        le=3,
        description="1=기관·학술, 2=검증된 전문가, 3=일반",
    )
    published_at: int | None = Field(default=None, description="발행/개정 연도")
    license: str | None = Field(default=None, description="라이선스/이용 근거")

    # ── 아래는 전부 optional + 기본값이라 기존 호출자는 영향받지 않는다 ──
    source_id: str | None = Field(
        default=None,
        max_length=100,
        description="corpus의 source_id. 소스별 진단과 라이선스 일괄 삭제에 쓴다",
    )
    content_hash: str | None = Field(
        default=None,
        max_length=64,
        description="재적재 멱등성 키. 생략하면 content로 계산한다",
    )
    distribution: Distribution = Field(
        default="personal-only",
        description="open=배포 가능. 기본값이 보수적인 쪽인 이유는 분류를 빠뜨린 문서가 "
        "조용히 배포 대상에 들어가면 안 되기 때문",
    )
    corpus: CorpusPartition = Field(
        default="answer",
        description="observation은 답변 근거로 쓰지 않는 관찰용 구획(블로그 등)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "분리불안 초기 대응",
                    "content": "보호자가 외출할 때 ...",
                    "source": "training-guide.pdf",
                    "category": "분리불안",
                }
            ]
        }
    }


class DocumentOut(BaseModel):
    id: int
    title: str
    source: str | None = None
    category: str | None = None
    chunk_count: int = Field(default=0, description="생성된 청크 수")


class IngestResponse(BaseModel):
    ingested: int = Field(description="적재된 문서 수")
    documents: list[DocumentOut] = Field(default_factory=list)
