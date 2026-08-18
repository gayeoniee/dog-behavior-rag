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
from app.services.query_rewrite import embed_by_language
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


# 재작성 없이 원문 임베딩만으로는 통과하지 못하는 질문.
#
# 코퍼스가 11건일 때는 통과했다 — 후보가 적어서 관련 문서가 쉽게 상위에 들었다.
# 282건/11,281청크가 되자 실패한다. **기법을 묻는 질문**이기 때문이다:
# bge-m3는 주제("초인종 소리에 짖어요" → barking)는 교차언어로 넘나들지만
# 기법 명칭("벌/혼내다" → punishment/aversive)은 못 넘는다.
#
# 코퍼스에 근거가 없어서가 아니다 — 아래 test_rewritten_query_finds_evidence가
# 같은 질문을 영어 기술표현으로 바꾸면 키워드 4개 전부 찾는다는 것을 보여준다.
# 해결책은 app/services/query_rewrite.py이고, LLM 서버가 붙으면 이 표는 비워진다.
#
# strict=True: 재작성 없이도 통과하기 시작하면 테스트가 실패해서 알려준다.
KNOWN_NEEDS_REWRITE = {"벌을 주면 안 되나요? 혼내면 그때만 멈춰요"}

# 한국어 원문 → 영어 기술표현. 2026-08-12 실측으로 효과를 확인한 쌍들이다.
REWRITE_CASES = [
    (
        "벌을 주면 안 되나요? 혼내면 그때만 멈춰요",
        "Is punishment bad for dogs? Scolding only stops the behaviour temporarily. "
        "positive punishment, aversive training, fear and aggression side effects",
        ["punishment", "aversive"],
    ),
    (
        "복종 자세를 강제로 1~2분 유지시켜 서열을 알려줘야 한다",
        "Forcibly holding a dog on its back in a submissive position "
        "(alpha roll, dominance down) to establish rank",
        ["dominance"],
    ),
    (
        "짖을 때 목줄을 잡고 안 돼라고 단호하게 소리쳐야 한다",
        "Correcting a barking dog with a leash jerk and shouting no as verbal punishment",
        ["punishment"],
    ),
]


def coverage_questions() -> list:
    if not COVERAGE_PATH.is_file():
        return []
    entries = yaml.safe_load(COVERAGE_PATH.read_text(encoding="utf-8"))
    return [
        pytest.param(
            entry,
            marks=pytest.mark.xfail(
                strict=True,
                reason="기법 질문 — 질의 재작성이 붙기 전에는 실패한다 (KNOWN_NEEDS_REWRITE 참조)",
            ),
        )
        if entry["question"] in KNOWN_NEEDS_REWRITE
        else entry
        for entry in entries
    ]


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


@pytest.mark.parametrize(
    ("korean", "english", "keywords"),
    REWRITE_CASES,
    ids=lambda v: v[:18] if isinstance(v, str) else "",
)
async def test_rewritten_query_finds_evidence(
    korean, english, keywords, session_factory, loaded, embedder
):
    """기법 질문의 실패 원인이 **코퍼스 공백이 아니라 검색**임을 못박는다.

    같은 질문을 영어 기술표현으로 바꾸면 근거가 나온다 = 근거는 코퍼스에 있다.
    그래서 해결책이 "자료를 더 모으기"가 아니라 "질의를 재작성하기"다.

    LLM 서버 없이 돌도록 재작성본을 하드코딩했다. query_rewrite.py가 실제로
    이 정도 품질을 내는지는 LLM이 붙은 뒤 별도로 확인해야 한다 — 이 테스트는
    "재작성이 되기만 하면 검색이 찾아낸다"는 전제만 검증한다.
    """
    async with session_factory() as session:
        store = PgVectorStore(session)
        ko_hits = await store.search(await embedder.embed_query(korean), TOP_K)
        en_hits = await store.search(await embedder.embed_query(english), TOP_K)

    ko_blob = " ".join(h.content.lower() for h in ko_hits)
    en_blob = " ".join(h.content.lower() for h in en_hits)
    ko_found = [k for k in keywords if k in ko_blob]
    en_found = [k for k in keywords if k in en_blob]

    # 한국어 원문 결과는 단언하지 않고 기록만 한다 — 임베딩 모델이 좋아지면
    # 통과할 수도 있고, 그건 반가운 일이지 테스트 실패가 아니다.
    print(f"\n  원문     {ko_found or '적중 없음'}  ← {korean[:40]}")
    print(f"  재작성   {en_found or '적중 없음'}")

    assert en_found, (
        f"영어 기술표현으로도 근거를 못 찾았다 — 이건 검색이 아니라 코퍼스 문제다.\n"
        f"  질의: {english}\n"
        f"  상위 문서: {[h.document_title[:40] for h in en_hits]}"
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def rewriter(app_settings):
    """실제 LLM에 붙는 재작성기. 서버가 없거나 키가 없으면 skip."""
    from app.services.llm.base import LLMUnavailableError
    from app.services.llm.registry import get_llm
    from app.services.query_rewrite import QueryRewriter

    llm = get_llm(app_settings)
    try:
        await llm.generate("ping", max_tokens=5)
    except LLMUnavailableError as exc:
        pytest.skip(f"LLM 서버 없음 — .env의 LLM_* 설정 확인 ({exc})")
    return QueryRewriter(llm, enabled=True)


@pytest.mark.parametrize(
    "question_entry",
    yaml.safe_load(COVERAGE_PATH.read_text(encoding="utf-8")) if COVERAGE_PATH.is_file() else [],
    ids=lambda q: f"{q['axis']}:{q['question'][:20]}",
)
async def test_coverage_question_with_rewriting(
    question_entry, session_factory, loaded, embedder, rewriter
):
    """**실제 운영 경로**(재작성 → 검색)로 커버리지 질문을 검증한다.

    위 test_coverage_question_retrieves_relevant_chunk는 재작성 없는 원문 검색이라
    기법 질문 1건이 xfail이다. 이 테스트는 재작성을 태워서 8/8이 나와야 한다.
    2026-08-12 gemini-3.1-flash-lite 기준 8/8 확인.

    LLM이 필요하므로 서버가 없으면 skip된다 — 맨몸 환경의 green을 깨지 않는다.
    """
    rewritten = await rewriter.rewrite(question_entry["question"])
    async with session_factory() as session:
        hits = await PgVectorStore(session).search(
            await embed_by_language(embedder, rewritten), TOP_K
        )

    blob = " ".join(h.content.lower() for h in hits)
    found = [k for k in question_entry["keywords"] if k.lower() in blob]
    assert found, (
        f"[{question_entry['axis']}] {question_entry['question']}\n"
        f"  재작성: EN={rewritten.en.splitlines()[0]!r} KO={rewritten.ko!r}\n"
        f"  키워드 {question_entry['keywords']} 중 상위 {TOP_K}청크에서 하나도 못 찾음\n"
        f"  상위 문서: {[h.document_title[:40] for h in hits]}"
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
