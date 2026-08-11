"""헬스체크.

/health       — liveness. 프로세스가 살아있으면 200. DB와 무관.
/health/ready — readiness. DB까지 확인. 실패하면 503.
"""

import logging

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.api.deps import SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness (DB 연결 확인)")
async def ready(session: SessionDep, response: Response) -> dict[str, str]:
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 — 어떤 DB 오류든 503으로 통일
        logger.warning("readiness 실패: %s", exc)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "detail": "database unreachable"}

    return {"status": "ready"}
