"""RagService 오케스트레이션 테스트. 가짜 3종만 쓴다."""

from app.services.rag_service import RagService
from tests.fakes import FakeEmbedder, FakeLLM, FakeStore, hit


async def test_search_hits_map_to_source_chunks() -> None:
    service = RagService(
        embedder=FakeEmbedder(),
        store=FakeStore([hit(7, title="문서 X", score=0.83)]),
        llm=FakeLLM(),
    )
    response = await service.answer("강아지가 짖어요")

    assert len(response.sources) == 1
    source = response.sources[0]
    assert source.chunk_id == 7
    assert source.document_title == "문서 X"
    assert source.score == 0.83
    assert source.source == "https://example.test/7"
    assert source.content == "근거 본문 7"


async def test_provider_comes_from_llm() -> None:
    service = RagService(embedder=FakeEmbedder(), store=FakeStore(), llm=FakeLLM())
    assert (await service.answer("질문")).provider == "huggingface:fake"


async def test_latency_is_recorded() -> None:
    service = RagService(embedder=FakeEmbedder(), store=FakeStore(), llm=FakeLLM())
    assert (await service.answer("질문")).latency_ms >= 0


async def test_no_hits_still_answers() -> None:
    """검색 결과가 없어도 500이 아니라 빈 sources로 응답해야 한다."""
    response = await RagService(embedder=FakeEmbedder(), store=FakeStore([]), llm=FakeLLM()).answer(
        "질문"
    )
    assert response.sources == []
    assert response.answer


async def test_top_k_is_passed_through() -> None:
    hits = [hit(i) for i in range(10)]
    service = RagService(embedder=FakeEmbedder(), store=FakeStore(hits), llm=FakeLLM())
    assert len((await service.answer("질문", top_k=3)).sources) == 3


async def test_retrieved_content_reaches_the_prompt() -> None:
    """근거를 뽑아놓고 프롬프트에 안 넣으면 RAG가 아니다."""
    llm = FakeLLM()
    service = RagService(embedder=FakeEmbedder(), store=FakeStore([hit(3)]), llm=llm)
    await service.answer("강아지가 짖어요")
    assert "근거 본문 3" in llm.last_prompt
    assert "강아지가 짖어요" in llm.last_prompt
