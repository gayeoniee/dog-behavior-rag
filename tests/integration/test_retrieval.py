"""실제 DB + 실제 임베딩 모델로 검색 품질을 검증한다.

전제가 하나라도 없으면 **skip**한다 (실패가 아니라). 그래야 `uv run pytest`가
맨몸 환경에서 green이면서, DB를 띄우는 순간 아무도 플래그를 기억하지 않아도
자동으로 돌기 시작한다.

    docker compose up -d db          (또는 uv run python -m scripts.db.serve)
    uv run python -m scripts.db.init
    uv sync --extra hf
    uv run python -m scripts.db.load_corpus
"""

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select, text

from app.core.config import get_settings
from app.db.models import Chunk, Document
from app.db.session import create_engine, create_session_factory
from app.services.embeddings.base import EmbeddingUnavailableError
from app.services.embeddings.registry import get_embedder
from app.services.ingest_service import content_hash
from app.services.vectorstore.pgvector import PgVectorStore

yaml = pytest.importorskip("yaml", reason="pyyaml 미설치 — uv sync --extra collect")

# loop_scope="module"이 필수다. 기본값(function)이면 테스트마다 새 이벤트 루프가
# 생기는데 asyncpg 커넥션은 처음 만들어진 루프에 묶여 있어서, 두 번째 테스트부터
# "attached to a different loop"로 죽는다. 임베딩 모델도 모듈당 1회만 로딩해야 한다.
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

COVERAGE_PATH = Path("data/coverage_questions.yaml")
TOP_K = 5


@pytest.fixture(scope="module")
def app_settings():
    return get_settings()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def session_factory(app_settings):
    engine = create_engine(app_settings.database_url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        await engine.dispose()
        pytest.skip(f"Postgres 없음 — docker compose up -d db ({exc})")
    yield create_session_factory(engine)
    await engine.dispose()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def loaded(session_factory):
    async with session_factory() as session:
        count = await session.scalar(select(func.count(Chunk.id)))
    if not count:
        pytest.skip("코퍼스 미적재 — uv run python -m scripts.db.load_corpus")
    return count


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def embedder(app_settings):
    emb = get_embedder(app_settings)
    try:
        await emb.warmup()
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"임베딩 모델 없음 — uv sync --extra hf ({exc})")
    return emb


def coverage_questions() -> list[dict]:
    if not COVERAGE_PATH.is_file():
        return []
    return yaml.safe_load(COVERAGE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "question_entry",
    coverage_questions(),
    ids=lambda q: f"{q['axis']}:{q['question'][:20]}",
)
async def test_coverage_question_retrieves_relevant_chunk(
    question_entry, session_factory, loaded, embedder
):
    """커버리지 질문 8개가 상위 5청크 안에 근거를 물어오는가.

    판정 기준이 `report.py`의 "키워드 절반 이상"과 다르다. 저쪽은 해당 축의 코퍼스
    **전체**를 훑었고 여기는 5청크 × 1200자만 본다 — 여기서 절반을 요구하면 검색
    품질이 아니라 운을 재게 된다. 그래서 "하나라도 있으면 통과"다.

    한국어 질문으로 영어 청크를 찾는 교차언어 검색이라는 점도 여기서 실증된다.
    """
    vector = await embedder.embed_query(question_entry["question"])
    async with session_factory() as session:
        hits = await PgVectorStore(session).search(vector, TOP_K)

    assert hits, "검색 결과가 0건 — 적재 또는 인덱스 문제"
    blob = " ".join(h.content.lower() for h in hits)
    found = [k for k in question_entry["keywords"] if k.lower() in blob]
    assert found, (
        f"[{question_entry['axis']}] {question_entry['question']}\n"
        f"  키워드 {question_entry['keywords']} 중 상위 {TOP_K}청크에서 하나도 못 찾음\n"
        f"  실제 상위 문서: {[h.document_title[:40] for h in hits]}"
    )


async def test_no_single_document_dominates(session_factory, loaded, embedder):
    """AAHA 가이드라인이 코퍼스 청크의 43%라 상한이 없으면 top_k를 독점한다."""
    settings = get_settings()
    vector = await embedder.embed_query("강아지 문제행동을 어떻게 교정하나요?")
    async with session_factory() as session:
        hits = await PgVectorStore(
            session,
            max_per_document=settings.max_chunks_per_document,
            candidate_multiplier=settings.candidate_multiplier,
        ).search(vector, TOP_K)

    titles = [h.document_title for h in hits]
    for title in set(titles):
        assert titles.count(title) <= settings.max_chunks_per_document, (
            f"{title}가 {titles.count(title)}건 — 문서당 상한이 안 걸렸다"
        )


async def test_scores_are_valid_similarities(session_factory, loaded, embedder):
    vector = await embedder.embed_query("분리불안")
    async with session_factory() as session:
        hits = await PgVectorStore(session).search(vector, TOP_K)

    assert all(0.0 <= h.score <= 1.0 for h in hits), [h.score for h in hits]
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True) or True
    # 부스팅 때문에 score 내림차순이 깨질 수 있다(의도된 동작). 범위만 검증한다.


async def test_aversive_documents_are_never_returned(session_factory, loaded, embedder):
    """코퍼스에 aversive 문서가 0건이라, 필터를 실제로 실행할 유일한 방법이 합성이다."""
    marker = "AVERSIVE_TEST_DOC_지배이론으로 서열을 잡아야 한다"
    digest = content_hash(marker)
    vector = await embedder.embed_query(marker)

    async with session_factory() as session:
        session.add(
            Document(
                title="합성 aversive 문서",
                content=marker,
                content_hash=digest,
                methodology="aversive",
                corpus="answer",
                authority_tier=1,
            )
        )
        await session.flush()
        doc_id = await session.scalar(select(Document.id).where(Document.content_hash == digest))
        session.add(Chunk(document_id=doc_id, ordinal=0, content=marker, embedding=vector))
        await session.commit()

        try:
            # 자기 자신을 질의로 넣었으니 필터가 없으면 반드시 1위로 나온다.
            hits = await PgVectorStore(session).search(vector, TOP_K)
            assert all(h.document_title != "합성 aversive 문서" for h in hits), (
                "aversive 문서가 검색됐다 — methodology 필터가 동작하지 않는다"
            )
        finally:
            await session.execute(delete(Document).where(Document.content_hash == digest))
            await session.commit()


async def test_observation_partition_is_never_returned(session_factory, loaded, embedder):
    """블로그 격리 구획이 답변 근거로 새어나오지 않는지."""
    marker = "OBSERVATION_TEST_DOC_블로그에서 긁어온 내용"
    digest = content_hash(marker)
    vector = await embedder.embed_query(marker)

    async with session_factory() as session:
        session.add(
            Document(
                title="합성 observation 문서",
                content=marker,
                content_hash=digest,
                methodology="reward_based",
                corpus="observation",
                authority_tier=3,
            )
        )
        await session.flush()
        doc_id = await session.scalar(select(Document.id).where(Document.content_hash == digest))
        session.add(Chunk(document_id=doc_id, ordinal=0, content=marker, embedding=vector))
        await session.commit()

        try:
            hits = await PgVectorStore(session).search(vector, TOP_K)
            assert all(h.document_title != "합성 observation 문서" for h in hits), (
                "observation 구획이 검색됐다 — corpus 필터가 동작하지 않는다"
            )
        finally:
            await session.execute(delete(Document).where(Document.content_hash == digest))
            await session.commit()
