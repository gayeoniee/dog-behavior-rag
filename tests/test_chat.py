from httpx import AsyncClient


async def test_chat_returns_stub_answer(client: AsyncClient) -> None:
    """DB가 없어도 chat 엔드포인트는 스텁 응답을 돌려준다."""
    resp = await client.post("/api/v1/chat", json={"question": "강아지가 짖어요"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["answer"]
    assert body["sources"] == []
    assert body["provider"].startswith("huggingface:")


async def test_chat_rejects_empty_question(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/chat", json={"question": ""})
    assert resp.status_code == 422
