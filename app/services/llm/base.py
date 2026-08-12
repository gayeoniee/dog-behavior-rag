"""LLM 인터페이스.

구현체는 이 Protocol만 만족하면 된다. 교체 지점은 registry.py 한 곳.
"""

from typing import Protocol, runtime_checkable


class LLMUnavailableError(RuntimeError):
    """LLM 서버에 닿을 수 없거나 응답이 오지 않았다.

    임베딩 쪽 EmbeddingUnavailableError와 같은 취급 — 500(서버 버그)이 아니라
    503(일시적 미준비)으로 변환된다. 로컬 LLM 서버는 사람이 켜고 끄는 것이라
    "안 켜져 있음"이 정상적인 상태 중 하나다.
    """


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
        max_tokens: int | None = None,
    ) -> str:
        """프롬프트를 받아 답변 텍스트를 반환한다.

        max_tokens=None이면 구현체의 기본값(설정값)을 쓴다.
        """
        ...
