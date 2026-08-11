"""LLM 인터페이스.

구현체는 이 Protocol만 만족하면 된다. 교체 지점은 registry.py 한 곳.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    @property
    def name(self) -> str:
        """모델 식별자 (응답의 provider 필드/로깅용)."""
        ...

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """프롬프트를 받아 답변 텍스트를 반환한다."""
        ...
