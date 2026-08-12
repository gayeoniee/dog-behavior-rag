"""테이블 생성 / 삭제.

    uv run python -m scripts.db.init          # 없으면 만든다
    uv run python -m scripts.db.init --drop   # 전부 지우고 다시 만든다

Alembic을 쓰지 않는 이유는 `app/db/models.py` 모듈 독스트링 참조.
EMBEDDING_DIM을 바꿨다면 반드시 `--drop`으로 다시 만들어야 한다 — 기존
`chunks.embedding`은 옛 차원 그대로라 INSERT가 실패한다.
"""

import argparse
import asyncio
import sys

from sqlalchemy import text

import app.db.models  # noqa: F401 — Base.metadata를 채우려면 import가 필요하다
from app.core.config import get_settings
from app.db.models import Base
from app.db.session import create_engine


async def run(*, drop: bool) -> int:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    try:
        async with engine.begin() as conn:
            # docker/init-pgvector.sql은 볼륨이 비어 있을 때만 실행된다. 기존 볼륨이나
            # 내장 서버(scripts.db.serve)에 붙는 경우를 위해 여기서도 보장한다.
            # CREATE EXTENSION이 먼저여야 chunks DDL에서 vector 타입이 존재한다.
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            if drop:
                await conn.run_sync(Base.metadata.drop_all)
            # MetaData.create_all은 동기 API라 run_sync로 감싼다.
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:  # noqa: BLE001 — 원인을 사람이 읽을 수 있게 바꿔서 보여준다
        print(f"✗ 스키마 생성 실패: {exc}", file=sys.stderr)
        print(
            "  DB가 떠 있는지 확인하세요:\n"
            "    docker compose up -d db\n"
            "    (Docker/관리자 권한이 없으면) uv run python -m scripts.db.serve",
            file=sys.stderr,
        )
        return 1
    finally:
        await engine.dispose()

    print(f"✓ 스키마 준비 완료 (embedding_dim={settings.embedding_dim}, drop={drop})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="documents/chunks 테이블 생성")
    parser.add_argument("--drop", action="store_true", help="기존 테이블을 지우고 다시 만든다")
    return asyncio.run(run(drop=parser.parse_args().drop))


if __name__ == "__main__":
    sys.exit(main())
