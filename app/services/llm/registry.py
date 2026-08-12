"""LLM provider 선택.

provider를 바꾸는 유일한 지점.

`openai-compatible`이 사실상 만능 분기다 — LM Studio / Ollama / llama.cpp /
vLLM / Groq / OpenRouter가 전부 같은 프로토콜이라 LLM_BASE_URL만 바꾸면 된다.
새 분기가 필요한 건 프로토콜 자체가 다른 경우(예: Anthropic Messages API)뿐이다:

    if provider == "anthropic":
        from .anthropic import AnthropicLLM
        return AnthropicLLM(settings)

(config.py의 Provider Literal에도 이름을 추가해야 함)
"""

from app.core.config import Settings

from .base import LLMClient


def get_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider

    if provider == "openai-compatible":
        from .openai_compatible import OpenAICompatibleLLM

        return OpenAICompatibleLLM(settings)

    if provider == "huggingface":
        from .huggingface import HuggingFaceLLM

        return HuggingFaceLLM(settings)

    raise ValueError(f"지원하지 않는 llm_provider: {provider!r}")
