"""채팅(질의응답) 요청/응답 스키마.

이 스키마가 안드로이드 앱과의 API 계약이 된다. 바꿀 때 주의.
"""

from typing import Literal

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """답변 근거로 사용된 문서 조각."""

    chunk_id: int = Field(description="청크 ID")
    document_title: str = Field(description="원문 제목")
    content: str = Field(description="인용된 본문")
    score: float = Field(description="유사도 점수 (1.0에 가까울수록 유사)")
    source: str | None = Field(default=None, description="출처 URL 또는 파일명")


class Turn(BaseModel):
    """직전 대화 한 마디."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


MAX_HISTORY_TURNS = 6
"""서버가 실제로 참고하는 최근 대화 수 (3번의 주고받기).

되묻기에 답할 수 있으면 충분하고, 그 이상은 프롬프트만 키운다.
"""


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="사용자 질문")
    history: list[Turn] = Field(
        default_factory=list,
        description="직전 대화 (오래된 것부터). 되묻기에 '1번이요'처럼 답할 수 있게 한다. "
        f"서버는 최근 {MAX_HISTORY_TURNS}개만 사용한다",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="검색할 청크 수. 생략하면 서버 기본값(TOP_K) 사용",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "강아지가 초인종 소리에 계속 짖어요", "top_k": 5},
                {
                    "question": "1번이요",
                    "history": [
                        {"role": "user", "content": "강아지가 벽을 자꾸 긁어"},
                        {
                            "role": "assistant",
                            "content": "알려주시면 좋아요\n1. 혼자 있을 때만 긁나요?\n"
                            "2. 짖거나 안절부절못하나요?",
                        },
                    ],
                },
            ]
        }
    }


Coverage = Literal["full", "partial", "none", "needs_detail"]
"""검색된 근거가 질문을 얼마나 덮는지.

`none`과 `needs_detail`은 둘 다 `sources`가 비어 있지만 의미가 다르다:

  none          질문이 이 서비스 범위 밖 (고양이, 가격, 장소) — 답하지 않는다
  needs_detail  개 행동 질문은 맞는데 정보가 부족해 원인을 좁힐 수 없다 —
                답변이 **되묻는 질문**으로 온다

관련 없는 근거를 인용처럼 보여주는 것이 신뢰를 깎기 때문에 두 경우 모두
`sources`를 비운다.
"""


class ChatResponse(BaseModel):
    answer: str = Field(description="생성된 답변")
    sources: list[SourceChunk] = Field(default_factory=list, description="근거 문서 조각")
    latency_ms: int = Field(description="처리 시간(ms)")
    provider: str = Field(description="답변 생성에 사용된 provider")

    # 아래 둘은 기본값이 있어 기존 클라이언트(안드로이드 앱)를 깨지 않는다.
    coverage: Coverage = Field(
        default="full",
        description="근거 충분도. none이면 코퍼스에 관련 자료가 없고 sources는 비어 있다",
    )
    coverage_note: str | None = Field(
        default=None,
        description="coverage가 none일 때 사용자에게 보여줄 안내",
    )
