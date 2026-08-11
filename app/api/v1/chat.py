"""질의응답 엔드포인트."""

from fastapi import APIRouter

from app.api.deps import RagServiceDep
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="질문에 대해 근거 문서와 함께 답변",
)
async def chat(payload: ChatRequest, service: RagServiceDep) -> ChatResponse:
    return await service.answer(payload.question, top_k=payload.top_k)
