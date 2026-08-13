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
        reasoning: bool | None = None,
    ) -> str:
        """프롬프트를 받아 답변 텍스트를 반환한다.

        max_tokens=None이면 구현체의 기본값(설정값)을 쓴다.

        reasoning은 **이 호출이 숙고를 필요로 하는지**를 말한다 (None이면 설정 기본값).
        추론형 모델에만 의미가 있고, 아닌 구현체는 무시하면 된다. 호출부가 "추론형이냐"가
        아니라 "생각이 필요한 일이냐"를 말하게 한 것이다 — 전자는 모델의 사정이고
        후자는 호출부가 아는 사실이다.

        실측 근거(gemma-4-e2b): 판정(근거 선별)은 추론을 켜야 정확해지는데, 답변 생성은
        켜도 평가 점수가 안 바뀌면서 요청당 13초를 더 쓴다. 하나의 설정으로 묶으면
        둘 중 하나를 포기해야 한다.
        """
        ...
