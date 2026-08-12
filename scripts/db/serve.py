"""Docker나 관리자 권한이 없는 머신용 내장 Postgres+pgvector.

    uv sync --extra pgdev
    uv run python -m scripts.db.serve          # 기동 + .env의 DATABASE_URL 갱신
    uv run python -m scripts.db.serve --stop

Docker가 되는 머신은 이게 필요 없다 — `docker compose up -d db`가 정상 경로다.
팀 규칙상 DB는 각자 쓰기로 했으므로(공용 DB는 서브PC 적재 시 병목) 서버를 어떻게
띄우든 상관없고, 이건 학원 PC처럼 설치 권한이 없는 환경을 위한 우회로다.

`pgserver`는 PostgreSQL 16 + pgvector 바이너리를 휠에 담아 배포한다. 설치도
관리자 권한도 필요 없는 대신 **기동할 때마다 빈 포트를 새로 잡는다.** 그래서 이
스크립트가 .env의 DATABASE_URL을 직접 갱신한다 — 사람이 매번 포트를 옮겨 적는
것보다 낫고, 갱신 내용을 출력하므로 조용히 바뀌지도 않는다.
"""

import argparse
import re
import sys
from pathlib import Path

PGDATA = Path(".pgdata")
ENV_PATH = Path(".env")
_URI_RE = re.compile(r"^postgresql://(?P<user>[^:@]+):?[^@]*@(?P<hostport>[^/]+)/(?P<db>.+)$")


def _import_pgserver():
    try:
        import pgserver
    except ImportError:
        print(
            "✗ pgserver가 설치돼 있지 않습니다 — uv sync --extra pgdev",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return pgserver


def _to_asyncpg_url(uri: str) -> str:
    """pgserver의 URI를 SQLAlchemy asyncpg 드라이버 형식으로 바꾼다.

    pgserver는 `postgresql://postgres:@127.0.0.1:PORT/postgres`처럼 빈 비밀번호를
    콜론만 남겨 표기하는데, 이걸 그대로 넘기면 드라이버에 따라 파싱이 갈린다.
    """
    m = _URI_RE.match(uri)
    if not m:  # 형식이 바뀌면 조용히 틀린 URL을 쓰느니 드러내는 게 낫다
        raise SystemExit(f"✗ pgserver URI 형식을 해석하지 못했습니다: {uri}")
    return "postgresql+asyncpg://{user}@{hostport}/{db}".format(**m.groupdict())


def _update_env(url: str) -> None:
    if not ENV_PATH.is_file():
        print(f"⚠️  {ENV_PATH} 가 없어 갱신을 건너뜁니다. 아래 값을 직접 넣으세요:")
        print(f"    DATABASE_URL={url}")
        return

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("DATABASE_URL="):
            lines[i] = f"DATABASE_URL={url}"
            replaced = True
            break
    if not replaced:
        lines.append(f"DATABASE_URL={url}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ .env의 DATABASE_URL 갱신: {url}")


def start() -> int:
    pgserver = _import_pgserver()
    PGDATA.mkdir(parents=True, exist_ok=True)

    # cleanup_mode=None: 이 프로세스가 끝나도 postmaster를 살려둔다. 개발 서버는
    # 스크립트보다 오래 살아야 한다 (uvicorn도 pytest도 여기 붙는다).
    server = pgserver.get_server(str(PGDATA.resolve()), cleanup_mode=None)
    server.psql("CREATE EXTENSION IF NOT EXISTS vector;")

    url = _to_asyncpg_url(server.get_uri())
    _update_env(url)
    print("✓ 내장 Postgres 기동 완료. 다음: uv run python -m scripts.db.init")
    return 0


def stop() -> int:
    pgserver = _import_pgserver()
    if not PGDATA.is_dir():
        print("⚠️  .pgdata 가 없습니다 — 기동된 적이 없는 것 같습니다")
        return 0
    pgserver.get_server(str(PGDATA.resolve()), cleanup_mode="stop").cleanup()
    print("✓ 내장 Postgres 정지")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="내장 Postgres+pgvector 기동 (Docker 대체)")
    parser.add_argument("--stop", action="store_true", help="기동된 서버를 정지한다")
    return stop() if parser.parse_args().stop else start()


if __name__ == "__main__":
    sys.exit(main())
