from httpx import AsyncClient


async def test_health_ok(client: AsyncClient) -> None:
    """liveness는 DB 없이도 200이어야 한다."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_openapi_has_endpoints(client: AsyncClient) -> None:
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200

    paths = resp.json()["paths"]
    assert "/api/v1/chat" in paths
    assert "/api/v1/documents" in paths
    assert "/api/v1/health" in paths
