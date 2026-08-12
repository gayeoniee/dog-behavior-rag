"""chat 엔드포인트 테스트.

검색이 실제로 동작하게 된 뒤로는 DB 없이 이 경로를 그냥 부를 수 없다 —
`test_readiness.py`가 db_session을 오버라이드하듯 rag_service를 오버라이드한다.
"""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import rag_service
from app.core.config import Settings
from app.main import create_app
from app.services.rag_service import RagService
from tests.fakes import FakeEmbedder, FakeLLM, FakeStore, hit


def build_client(settings: Settings, service: RagService) -> tuple:
    app = create_app(settings)
    app.dependency_overrides[rag_service] = lambda: service
    return app, ASGITransport(app=app)


@pytest.fixture
async def chat_client(settings: Settings) -> AsyncIterator[AsyncClient]:
    service = RagService(
        embedder=FakeEmbedder(),
        store=FakeStore([hit(1), hit(2, title="문서 B", score=0.7)]),
        llm=FakeLLM(),
    )
    app, transport = build_client(settings, service)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test") as ac,
    ):
        yield ac


@pytest.fixture
async def broken_chat_client(settings: Settings) -> AsyncIterator[AsyncClient]:
    service = RagService(embedder=FakeEmbedder(fail=True), store=FakeStore(), llm=FakeLLM())
    app, transport = build_client(settings, service)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test") as ac,
    ):
        yield ac


async def test_chat_returns_sources(chat_client: AsyncClient) -> None:
    resp = await chat_client.post("/api/v1/chat", json={"question": "강아지가 짖어요"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["answer"]
    assert len(body["sources"]) == 2
    assert body["sources"][0]["document_title"] == "문서 A"
    # 인용 링크가 API 응답까지 살아 있어야 화면에 출처가 뜬다.
    assert body["sources"][0]["source"].startswith("https://")
    assert body["provider"].startswith("huggingface:")


async def test_chat_scores_never_exceed_one(chat_client: AsyncClient) -> None:
    """SourceChunk.score는 부스트 전 코사인 유사도라는 계약."""
    body = (await chat_client.post("/api/v1/chat", json={"question": "질문"})).json()
    assert all(0.0 <= s["score"] <= 1.0 for s in body["sources"])


async def test_chat_returns_503_when_embedding_unavailable(
    broken_chat_client: AsyncClient,
) -> None:
    """torch가 없거나 모델 로딩이 실패한 상태. 500(버그)이 아니라 503(미준비)."""
    resp = await broken_chat_client.post("/api/v1/chat", json={"question": "강아지가 짖어요"})
    assert resp.status_code == 503


async def test_chat_rejects_empty_question(chat_client: AsyncClient) -> None:
    resp = await chat_client.post("/api/v1/chat", json={"question": ""})
    assert resp.status_code == 422
