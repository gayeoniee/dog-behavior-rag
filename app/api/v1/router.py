"""v1 라우터 집계."""

from fastapi import APIRouter

from app.api.v1 import chat, documents, factcheck, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(factcheck.router)
api_router.include_router(documents.router)
