"""질의응답 엔드포인트."""

import logging

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import RagServiceDep
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.embeddings.base import EmbeddingUnavailableError
from app.services.llm.base import LLMUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="질문에 대해 근거 문서와 함께 답변",
)
async def chat(payload: ChatRequest, service: RagServiceDep) -> ChatResponse:
    try:
        return await service.answer(payload.question, top_k=payload.top_k)
    except (EmbeddingUnavailableError, LLMUnavailableError, SQLAlchemyError) as exc:
        # 임베딩 모델이나 DB가 준비 안 된 상태. 500(서버 버그)이 아니라 503(일시적
        # 미준비)이 맞고, 클라이언트는 /health/ready에서 이미 같은 의미를 다룬다.
        logger.warning("검색 경로 사용 불가: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="검색 서비스가 준비되지 않았습니다",
        ) from exc
