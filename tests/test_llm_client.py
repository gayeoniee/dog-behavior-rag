"""openai-compatible LLM 클라이언트 테스트. 실제 서버 없이 httpx를 목킹한다.

같은 클라이언트가 로컬 서버(키 불필요)와 Gemini(키 필수) 양쪽에 붙으므로,
오류를 원인별로 구분해 안내하는지가 핵심이다.
"""

import httpx
import pytest

from app.core.config import Settings
from app.services.llm.base import LLMUnavailableError
from app.services.llm.openai_compatible import OpenAICompatibleLLM


def make_llm(**overrides) -> OpenAICompatibleLLM:
    base = {
        "llm_base_url": "http://localhost:1234/v1",
        "llm_model": "test-model",
        "llm_api_key": "not-needed",
    }
    return OpenAICompatibleLLM(Settings(**{**base, **overrides}))


def patch_transport(monkeypatch, handler) -> None:
    """httpx.AsyncClient가 지정한 핸들러를 쓰도록 바꾼다."""
    original = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


def json_response(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status, json=payload)


async def test_successful_completion(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return json_response(200, {"choices": [{"message": {"content": "답변입니다"}}]})

    patch_transport(monkeypatch, handler)
    assert await make_llm().generate("질문") == "답변입니다"
    # base_url 끝의 슬래시 유무와 관계없이 경로가 한 번만 붙어야 한다.
    assert captured["url"] == "http://localhost:1234/v1/chat/completions"


async def test_trailing_slash_in_base_url_is_handled(monkeypatch):
    """Gemini 문서의 base_url이 슬래시로 끝난다 — 그대로 붙여넣어도 동작해야 한다."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return json_response(200, {"choices": [{"message": {"content": "ok"}}]})

    patch_transport(monkeypatch, handler)
    llm = make_llm(llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    await llm.generate("질문")
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )


async def test_system_prompt_is_sent_first(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return json_response(200, {"choices": [{"message": {"content": "ok"}}]})

    patch_transport(monkeypatch, handler)
    await make_llm().generate("질문", system="너는 전문가다")
    roles = [m["role"] for m in captured["body"]["messages"]]
    assert roles == ["system", "user"]


async def test_connect_error_points_at_the_server(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    patch_transport(monkeypatch, handler)
    with pytest.raises(LLMUnavailableError, match="연결할 수 없습니다"):
        await make_llm().generate("질문")


async def test_auth_error_points_at_the_key_not_the_server(monkeypatch):
    """401을 "LM Studio를 켜세요"로 안내하면 엉뚱한 곳을 보게 된다."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(401, {"error": {"message": "API key not valid"}})

    patch_transport(monkeypatch, handler)
    llm = make_llm(
        llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        llm_api_key="",
    )
    with pytest.raises(LLMUnavailableError) as exc:
        await llm.generate("질문")
    assert "인증 실패" in str(exc.value)
    assert "aistudio.google.com" in str(exc.value)


async def test_model_not_found_names_the_model(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(404, {"error": {"message": "model not found"}})

    patch_transport(monkeypatch, handler)
    with pytest.raises(LLMUnavailableError) as exc:
        await make_llm(llm_model="없는-모델").generate("질문")
    assert "없는-모델" in str(exc.value)


async def test_rate_limit_is_explained_after_retries_exhausted(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(429, {"error": {"message": "quota exceeded"}})

    patch_transport(monkeypatch, handler)
    with pytest.raises(LLMUnavailableError, match="한도 초과"):
        await make_llm(llm_max_retries=1, llm_retry_base_delay=0).generate("질문")


async def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    """무료 티어 분당 한도는 일시적 상태다. 여기서 포기하면 연속 호출이 몰릴 때
    (평가셋 실행, 팩트체크 병렬 판정) 절반이 조용히 폴백으로 떨어진다."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return json_response(429, {"error": {"message": "quota"}})
        return json_response(200, {"choices": [{"message": {"content": "성공"}}]})

    patch_transport(monkeypatch, handler)
    llm = make_llm(llm_max_retries=4, llm_retry_base_delay=0)
    assert await llm.generate("질문") == "성공"
    assert calls["n"] == 3


async def test_server_error_is_retried(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return json_response(503, {"error": {"message": "overloaded"}})
        return json_response(200, {"choices": [{"message": {"content": "ok"}}]})

    patch_transport(monkeypatch, handler)
    assert await make_llm(llm_retry_base_delay=0).generate("질문") == "ok"
    assert calls["n"] == 2


async def test_auth_error_is_not_retried(monkeypatch):
    """키가 틀린 건 기다려도 안 고쳐진다. 재시도하면 시간만 버린다."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return json_response(401, {"error": {"message": "bad key"}})

    patch_transport(monkeypatch, handler)
    with pytest.raises(LLMUnavailableError, match="인증 실패"):
        await make_llm(llm_retry_base_delay=0).generate("질문")
    assert calls["n"] == 1


async def test_retry_after_header_is_respected(monkeypatch):
    calls = {"n": 0}
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"e": 1}, headers={"Retry-After": "7"})
        return json_response(200, {"choices": [{"message": {"content": "ok"}}]})

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    patch_transport(monkeypatch, handler)
    monkeypatch.setattr("app.services.llm.openai_compatible.asyncio.sleep", fake_sleep)
    await make_llm().generate("질문")
    assert slept == [7.0]


async def test_unexpected_payload_shape_is_reported(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(200, {"unexpected": "shape"})

    patch_transport(monkeypatch, handler)
    with pytest.raises(LLMUnavailableError, match="예상과 다른 응답 형식"):
        await make_llm().generate("질문")


async def test_null_content_becomes_empty_string(monkeypatch):
    """일부 서버가 content를 null로 준다 — 여기서 터지면 안 된다."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(200, {"choices": [{"message": {"content": None}}]})

    patch_transport(monkeypatch, handler)
    assert await make_llm().generate("질문") == ""
