"""적재된 문서의 파생 메타데이터를 다시 계산한다.

    uv run python -m scripts.db.refresh_metadata

`distribution`은 `license`에서 파생되는 값이라, 판정 규칙(OPEN_LICENSES)을 바꾸면
이미 적재된 행은 옛 판정값을 그대로 갖고 있게 된다. 재적재는 content_hash 때문에
건너뛰므로 갱신되지 않는다 — 그래서 이 스크립트가 필요하다.

임베딩을 다시 만들지 않으므로 몇 초면 끝난다. 청킹 전략이나 임베딩 모델이 바뀐
경우에는 이걸로 안 되고 `init --drop` 후 재적재해야 한다.
"""

import asyncio
import sys
from collections import Counter

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Document
from app.db.session import create_engine, create_session_factory
from scripts.db.load_corpus import derive_distribution, derive_doc_type


async def run() -> int:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)

    changed: Counter[str] = Counter()
    try:
        async with factory() as session:
            rows = (await session.execute(select(Document))).scalars().all()
            for doc in rows:
                expected = derive_distribution(doc.license)
                if doc.distribution != expected:
                    changed[f"distribution: {doc.distribution} → {expected}"] += 1
                    doc.distribution = expected

                expected_type = derive_doc_type(doc.source_id)
                if doc.doc_type != expected_type:
                    changed[f"doc_type: {doc.doc_type} → {expected_type}"] += 1
                    doc.doc_type = expected_type
            await session.commit()
    except Exception as exc:  # noqa: BLE001 — 원인을 사람이 읽게 바꿔 보여준다
        print(f"✗ 갱신 실패: {exc}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()

    if not changed:
        print(f"✓ 문서 {len(rows)}건 — 변경 없음 (이미 최신 판정)")
        return 0

    print(f"✓ 문서 {len(rows)}건 검사, {sum(changed.values())}건 갱신")
    for transition, count in changed.most_common():
        print(f"    {count:>4}건  {transition}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
