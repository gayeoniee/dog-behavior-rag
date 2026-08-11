"""문서 적재 스키마."""

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=500, description="문서 제목")
    content: str = Field(min_length=1, description="문서 본문")
    source: str | None = Field(default=None, max_length=500, description="출처 URL/파일명")
    category: str | None = Field(
        default=None,
        max_length=100,
        description="분류 (예: 짖음, 분리불안, 배변훈련)",
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
