"""팩트체크 스키마.

`/chat`과 반대 방향이다. chat은 "질문 → 근거로 답변"이고, factcheck는
"어디서 본 주장 → 근거와 대조"다. 유튜브·블로그 조언을 코퍼스에 넣지 않기로 한
결정과 짝을 이룬다 — 오염원으로 들이지 않고 검증 대상으로 받는다.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import SourceChunk

Verdict = Literal["supported", "contradicted", "not_covered"]
"""`not_covered`가 반드시 있어야 한다.

이 선택지를 빼면 모델이 근거 없이 supported/contradicted 중 하나를 고른다.
팩트체크에서 가장 흔한 실패 모드고, 근거 인용이 존재 이유인 이 프로젝트에서는
특히 나쁘다.
"""


class ClaimVerdict(BaseModel):
    claim: str = Field(description="원문에서 뽑아낸 검증 가능한 주장")
    verdict: Verdict = Field(
        description="supported=자료가 뒷받침 / contradicted=자료와 배치 / "
        "not_covered=자료에 관련 근거 없음"
    )
    explanation: str = Field(description="판정 근거 설명 (한국어)")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="판정에 쓰인 근거 청크. 비어 있으면 verdict는 not_covered로 강등된다",
    )


class FactCheckRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=8000,
        description="검증할 텍스트 (유튜브 자막, 블로그 글, 들은 조언 등)",
    )
    top_k: int | None = Field(default=None, ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "강아지가 마운팅하는 건 서열이 위라고 생각해서예요. "
                    "복종 자세를 시켜서 서열을 알려줘야 합니다."
                }
            ]
        }
    }


class FactCheckResponse(BaseModel):
    claims: list[ClaimVerdict] = Field(default_factory=list)
    corpus_note: str = Field(
        description="코퍼스 편향 고지. 판정을 '중립적'인 것처럼 보이게 하지 않기 위해 항상 붙인다"
    )
    latency_ms: int
    provider: str
