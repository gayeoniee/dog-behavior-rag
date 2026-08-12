"""OpenAI 호환 `/v1/chat/completions` 서버용 LLM 클라이언트.

특정 서비스가 아니라 **프로토콜**에 붙는다. 같은 구현으로 전부 커버된다:

    Gemini       https://generativelanguage.googleapis.com/v1beta/openai  (기본값)
    LM Studio    http://localhost:1234/v1   (GUI, 관리자 권한 불필요)
    Ollama       http://localhost:11434/v1
    llama.cpp    http://localhost:8080/v1
    vLLM / Groq / OpenRouter …

HF Inference API는 선택지에서 뺐다 — 무료 크레딧이 월 $0.10라 사실상 쓸 수 없고
(2026-08 확인), `hf-inference`는 2025년 7월부터 CPU 추론 위주로 축소돼 최신
instruct 모델을 서빙하지 않는다.

기본값이 Gemini인 이유는 이 PC의 VRAM이 6GB뿐이기 때문이다. 로컬 7B Q4(4.7GB)와
bge-m3(2.3GB)가 동시에 안 올라가는데, LLM을 밖으로 빼면 GPU를 임베딩이 독점한다
(임베딩이 GPU에서 11배 빠르다). 오프라인이나 데이터를 내보내지 않아야 하는
상황이면 LM Studio로 바꾸면 되고, 코드 변경은 없다.

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
            raise LLMUnavailableError(self._explain_status(exc)) from exc

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMUnavailableError(f"예상과 다른 응답 형식: {str(data)[:200]}") from exc

    def _explain_status(self, exc: httpx.HTTPStatusError) -> str:
        """HTTP 오류를 원인별로 다르게 안내한다.

        같은 클라이언트가 로컬 서버(키 불필요)와 Gemini(키 필수) 양쪽에 붙으므로,
        401을 "LM Studio를 켜세요"로 안내하면 엉뚱한 곳을 보게 된다.
        """
        status = exc.response.status_code
        body = exc.response.text[:200]

        if status in (401, 403):
            hint = (
                "API 키를 확인하세요 (.env의 LLM_API_KEY). "
                "Gemini 키는 https://aistudio.google.com/apikey 에서 발급합니다."
                if not self._api_key or self._api_key == "not-needed"
                else "API 키가 거부됐습니다 (.env의 LLM_API_KEY)."
            )
            return f"LLM 인증 실패 {status}: {hint}\n  응답: {body}"
        if status == 404:
            return (
                f"LLM 모델 또는 경로를 찾을 수 없습니다 {status}.\n"
                f"  LLM_MODEL={self._model!r} 이 서버에 있는 이름인지, "
                f"LLM_BASE_URL={self._base_url!r} 이 맞는지 확인하세요.\n"
                f"  응답: {body}"
            )
        if status == 429:
            return (
                f"LLM 요청 한도 초과 {status}. 무료 티어 분당 한도일 수 있습니다.\n  응답: {body}"
            )
        return f"LLM 서버 오류 {status}: {body}"
