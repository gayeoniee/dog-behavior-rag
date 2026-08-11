"""LLM provider 선택.

provider를 바꾸는 유일한 지점.

나중에 Claude로 전환할 경우 아래 분기 하나만 추가하면 된다:

    if provider == "anthropic":
        from .anthropic import AnthropicLLM
        return AnthropicLLM(settings)

(config.py의 Provider Literal에도 "anthropic"을 추가해야 함)
"""

from app.core.config import Settings

from .base import LLMClient


def get_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider

    if provider == "huggingface":
        from .huggingface import HuggingFaceLLM

        return HuggingFaceLLM(settings)

    raise ValueError(f"지원하지 않는 llm_provider: {provider!r}")
