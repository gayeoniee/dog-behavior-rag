"""OpenAI 호환 `/v1/chat/completions` 서버용 LLM 클라이언트.

특정 서비스가 아니라 **프로토콜**에 붙는다. 같은 구현으로 전부 커버된다:

    LM Studio    http://localhost:1234/v1   (GUI, 관리자 권한 불필요)
    Ollama       http://localhost:11434/v1
    llama.cpp    http://localhost:8080/v1
    vLLM / Groq / OpenRouter …

이 프로젝트가 로컬 우선인 이유는 HF Inference API의 무료 크레딧이 월 $0.10라
사실상 쓸 수 없고(2026-08 확인), `hf-inference`는 2025년 7월부터 CPU 추론
위주로 축소돼 최신 instruct 모델을 서빙하지 않기 때문이다. GGUF 양자화 모델을
로컬 GPU에 올리는 쪽이 무료이면서 더 빠르다.

httpx는 이미 기본 의존성이라 새 패키지가 필요 없다 — openai SDK를 안 쓰는 이유다.
"""

import logging

import httpx

from app.core.config import Settings

from .base import LLMUnavailableError

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.llm_base_url.rstrip("/")
        self._model = settings.llm_model
        self._api_key = settings.llm_api_key
        self._timeout = settings.llm_timeout_seconds
        self._default_max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    @property
    def name(self) -> str:
        return f"openai-compatible:{self._model}"

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            # 가장 흔한 실패다. 원인을 바로 알 수 있게 서버 주소를 함께 보여준다.
            raise LLMUnavailableError(
                f"LLM 서버에 연결할 수 없습니다: {self._base_url}\n"
                "  LM Studio를 켜고 Developer 탭에서 Start Server 하셨는지 확인하세요 "
                "(또는 .env의 LLM_BASE_URL)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                f"LLM 응답이 {self._timeout}초 안에 오지 않았습니다. "
                "모델이 GPU에 다 올라갔는지 확인하거나 LLM_TIMEOUT_SECONDS를 늘리세요."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailableError(
                f"LLM 서버 오류 {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMUnavailableError(f"예상과 다른 응답 형식: {str(data)[:200]}") from exc
