"""corpus.jsonl → 청킹 → 임베딩 → documents/chunks 적재.

    uv run python -m scripts.db.load_corpus --dry-run   # 청킹만 (torch 불필요)
    uv run python -m scripts.db.load_corpus --limit 1   # 스모크
    uv run python -m scripts.db.load_corpus             # 전체
    uv run python -m scripts.db.load_corpus --corpus data/processed/corpus_blogs.jsonl

**HTTP가 아니라 직접 적재하는 이유:**
  - POST /documents로 하면 API 프로세스에 torch(~2.5GB)가 상주해야 한다.
    직접 적재하면 API는 EMBEDDING_WARMUP=false로 가볍게 띄울 수 있다.
  - AAHA 가이드라인 한 건이 10만 자 → 청크 80개 → CPU 임베딩 수 분이라
    단일 HTTP 요청으로는 타임아웃이 난다.
  - 진행 출력·--limit·중단 후 재개가 공짜로 따라온다.

엔드포인트와 **같은 IngestService**를 부르므로 구현이 갈라지지 않는다.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.schemas.document import DocumentIn
from app.services.chunking import ChunkConfig, split_text
from app.services.embeddings.base import EmbeddingUnavailableError
from app.services.embeddings.registry import get_embedder
from app.services.ingest_service import IngestService
from app.services.vectorstore.pgvector import PgVectorStore

DEFAULT_CORPUS = Path("data/processed/corpus.jsonl")

OPEN_LICENSES = frozenset(
    {
        # 배포·재이용에 제약이 없는 것만 여기 넣는다.
        "cc-by",
        "cc-by-sa",
        "cc0",
        "public-position-statement",
        "public-guideline-pdf",
        "korea-gov-nuri-1",  # 공공누리 제1유형(출처표시)만 해당
    }
)
"""배포 가능한 license 값의 **허용 목록**.

거부 목록이 아니라 허용 목록인 이유: 예전 구현이 "personal-use-only로 시작하지
않으면 open"이었는데, 그러면 처음 보는 license가 조용히 배포 대상이 됐다.
실제로 `korea-gov-publication`으로 넣은 문서가 알고 보니 민간 저작권물
("저작권은 (주)펫앤스토리에게 귀속 ... 무단 복제, 사용 시 법적 제재")이었고
그게 open으로 분류됐다. 정부기관이 배포한다고 공공저작물인 게 아니다.

**NC/ND를 일부러 뺐다.** cc-by-nc는 상업적 이용을, cc-by-nc-nd는 2차적 저작물
작성을 금지한다. RAG 답변이 2차적 저작물인지, 앱이 상업적인지가 아직 정해지지
않았으므로 보수적으로 둔다. 앱의 성격이 확정되면 그때 재분류할 것.
"""


def derive_distribution(license_value: str | None) -> str:
    """license 문자열 → distribution 한 필드.

    CLAUDE.md가 "배포 시 코퍼스에서 제외"라고 한 판정을 매번 문자열 매칭하지 않도록
    적재 시점에 한 번만 정규화한다. **허용 목록에 없으면 전부 personal-only다** —
    분류를 빠뜨린 문서가 조용히 배포 대상이 되면 안 된다.
    """
    if not license_value:
        return "personal-only"
    return "open" if license_value.strip().lower() in OPEN_LICENSES else "personal-only"


def derive_doc_type(source_id: str | None) -> str:
    """PMC 논문이면 study, 나머지(기관 실무 가이드)는 guide.

    소스 id로 판단하는 이유: 실제 구분 기준이 "학술지에 실렸는가"이고, 그게
    `pmc-` 접두사와 정확히 일치한다. 나중에 다른 학술 소스가 생기면 여기만 고친다.
    """
    return "study" if (source_id or "").startswith("pmc-") else "guide"


def to_document_in(record: dict, *, corpus_partition: str) -> DocumentIn:
    """corpus.jsonl(15키) → DocumentIn 리매핑.

    이름이 다른 키: `source_url` → `source`
    버리는 키:
      - `id`      : "{source_id}:{content_hash}"라 두 필드로 복원 가능
      - `volatility` : 수집 단계에서 최신성 경고에만 쓰는 값
      - `fetched_at` : 수집 시각. 문서 내용과 무관
    """
    return DocumentIn(
        title=record["title"],
        content=record["content"],
        content_hash=record.get("content_hash"),
        source=record.get("source_url"),
        source_id=record.get("source_id"),
        category=None,
        language=record.get("language", "en"),
        species=record.get("species", "dog"),
        axis=record.get("axis", []),
        methodology=record.get("methodology", "unknown"),
        authority_tier=record.get("authority_tier", 3),
        published_at=record.get("published_at"),
        license=record.get("license"),
        distribution=derive_distribution(record.get("license")),
        doc_type=derive_doc_type(record.get("source_id")),
        corpus=corpus_partition,
    )


def load_records(path: Path, *, limit: int | None, exclude: str | None) -> list[dict]:
    if not path.is_file():
        raise SystemExit(
            f"✗ {path} 가 없습니다.\n"
            "  먼저 수집을 실행하세요:\n"
            "    uv run python -m scripts.collect.fetch --all\n"
            "    uv run python -m scripts.collect.normalize"
        )
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if exclude:
        before = len(records)
        records = [r for r in records if derive_distribution(r.get("license")) != exclude]
        print(f"  {exclude} 제외: {before - len(records)}건")
    return records[:limit] if limit else records


def dry_run(records: list[dict], config: ChunkConfig) -> int:
    """청킹만 해보고 통계를 낸다. 임베딩도 DB도 torch도 필요 없다."""
    print(f"\n═══ 드라이런 (청킹만, size={config.size} overlap={config.overlap}) ═══")
    total = 0
    rows: list[tuple[int, str]] = []
    for record in records:
        chunks = split_text(record["content"], config)
        total += len(chunks)
        rows.append((len(chunks), record["title"][:60]))

    for count, title in sorted(rows, reverse=True):
        marker = "  ← 편중 주의" if count > total * 0.25 else ""
        print(f"  {count:4d}  {title}{marker}")

    average = total / max(len(records), 1)
    print(f"\n  문서 {len(records)}건 → 청크 {total}개 (문서당 평균 {average:.1f})")
    print("  DB에는 아무것도 쓰지 않았습니다. 실제 적재는 --dry-run 없이 실행하세요.")
    return 0


async def ingest_all(records: list[dict], corpus_partition: str) -> int:
    settings = get_settings()
    embedder = get_embedder(settings)
    try:
        await embedder.warmup()
    except EmbeddingUnavailableError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    print(f"✓ 임베딩 모델 준비 — {embedder.name} (dim={embedder.dimension})")

    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    chunk_config = ChunkConfig(
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        min_size=settings.chunk_min_size,
    )

    created = skipped = chunks_total = 0
    try:
        async with factory() as session:
            service = IngestService(
                session=session,
                embedder=embedder,
                store=PgVectorStore(session),
                chunk_config=chunk_config,
                paper_boilerplate_filter=settings.paper_boilerplate_filter,
            )
            for i, record in enumerate(records, 1):
                doc = to_document_in(record, corpus_partition=corpus_partition)
                result = await service.ingest(doc)
                chunks_total += result.chunk_count
                if result.created:
                    created += 1
                    mark = "✓"
                else:
                    skipped += 1
                    mark = "·"
                # flush=True: 파이프/파일로 리다이렉트하면 stdout이 버퍼링돼서
                # 수십 분짜리 적재의 진행 상황이 끝날 때까지 안 보인다.
                print(
                    f"  {mark} [{i}/{len(records)}] {result.chunk_count:3d}청크  "
                    f"{result.title[:60]}",
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001 — 원인을 사람이 읽게 바꿔 보여준다
        print(f"\n✗ 적재 실패: {exc}", file=sys.stderr)
        print(
            "  DB와 스키마를 확인하세요:\n    uv run python -m scripts.db.init",
            file=sys.stderr,
        )
        return 1
    finally:
        await engine.dispose()

    print(f"\n✓ 신규 {created}건 / 이미 있음 {skipped}건 / 총 청크 {chunks_total}개")
    if created == 0 and skipped:
        print("  (전부 이미 있음 = 재적재가 멱등하게 동작하고 있다는 뜻)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="corpus.jsonl을 DB에 적재")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="corpus.jsonl 경로")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N건만")
    parser.add_argument(
        "--exclude",
        choices=["open", "personal-only"],
        default=None,
        help="해당 distribution 문서를 제외 (배포용 코퍼스 구성 검증)",
    )
    parser.add_argument(
        "--partition",
        choices=["answer", "observation"],
        default="answer",
        help="observation은 답변 근거로 검색되지 않는 관찰용 구획",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="청킹만 하고 통계 출력 (임베딩·DB·torch 불필요)",
    )
    args = parser.parse_args()

    settings = get_settings()
    records = load_records(args.corpus, limit=args.limit, exclude=args.exclude)
    print(f"코퍼스 {args.corpus} — 문서 {len(records)}건")

    if args.dry_run:
        return dry_run(
            records,
            ChunkConfig(
                size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                min_size=settings.chunk_min_size,
            ),
        )
    return asyncio.run(ingest_all(records, args.partition))


if __name__ == "__main__":
    sys.exit(main())
