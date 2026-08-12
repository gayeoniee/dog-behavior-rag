"""팩트체크 엔드포인트.

유튜브·블로그에서 본 훈련 조언을 붙여넣으면 코퍼스와 대조해 판정한다.
수집하지 않기로 한 자료를 검증 대상으로 받는 쪽으로 뒤집은 기능이다.
"""

import logging

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import FactCheckServiceDep
from app.schemas.factcheck import FactCheckRequest, FactCheckResponse
from app.services.embeddings.base import EmbeddingUnavailableError
from app.services.llm.base import LLMUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["factcheck"])


@router.post(
    "/factcheck",
    response_model=FactCheckResponse,
    summary="어디서 본 훈련 조언을 코퍼스와 대조",
)
async def factcheck(payload: FactCheckRequest, service: FactCheckServiceDep) -> FactCheckResponse:
    try:
        return await service.check(payload.text, top_k=payload.top_k)
    except (EmbeddingUnavailableError, LLMUnavailableError, SQLAlchemyError) as exc:
        logger.warning("팩트체크 사용 불가: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="검증 서비스가 준비되지 않았습니다",
        ) from exc
