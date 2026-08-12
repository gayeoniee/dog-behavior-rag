"""HuggingFace LLM 구현 — 아직 스텁.

TODO(내일): 두 방식 중 선택
  (a) 로컬 추론: transformers pipeline("text-generation", settings.hf_llm_model)
      → 무료지만 모델 크기만큼 메모리 필요, GPU 없으면 느림
  (b) Inference API: huggingface_hub.AsyncInferenceClient(token=settings.hf_api_token)
      → 서버 자원 안 쓰지만 무료 티어 rate limit 있음

로컬 추론이면 generate()는 blocking이므로 asyncio.to_thread()로 감싸야 한다.
"""

from app.core.config import Settings


class HuggingFaceLLM:
    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.hf_llm_model or "(미설정)"
        self._api_token = settings.hf_api_token
        # TODO(내일): 모델 또는 InferenceClient 초기화

    @property
    def name(self) -> str:
        return f"huggingface:{self._model_name}"

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        # TODO(내일): 실제 추론 호출
        return (
            "[stub] 아직 LLM이 연결되지 않았습니다. "
            "HF_LLM_MODEL을 설정하고 huggingface.py의 generate()를 구현하세요.\n"
            f"(받은 프롬프트 {len(prompt)}자, max_tokens={max_tokens})"
        )
