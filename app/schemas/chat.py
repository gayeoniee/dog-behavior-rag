"""채팅(질의응답) 요청/응답 스키마.

이 스키마가 안드로이드 앱과의 API 계약이 된다. 바꿀 때 주의.
"""

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """답변 근거로 사용된 문서 조각."""

    chunk_id: int = Field(description="청크 ID")
    document_title: str = Field(description="원문 제목")
    content: str = Field(description="인용된 본문")
    score: float = Field(description="유사도 점수 (1.0에 가까울수록 유사)")
    source: str | None = Field(default=None, description="출처 URL 또는 파일명")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="사용자 질문")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="검색할 청크 수. 생략하면 서버 기본값(TOP_K) 사용",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "강아지가 초인종 소리에 계속 짖어요", "top_k": 5}]
        }
    }


class ChatResponse(BaseModel):
    answer: str = Field(description="생성된 답변")
    sources: list[SourceChunk] = Field(default_factory=list, description="근거 문서 조각")
    latency_ms: int = Field(description="처리 시간(ms)")
    provider: str = Field(description="답변 생성에 사용된 provider")
