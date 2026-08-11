"""임베딩 provider 선택.

provider를 바꾸는 유일한 지점. import는 지연시켜야 무거운 의존성(torch 등)이
설치돼 있지 않아도 앱이 뜬다.
"""

from app.core.config import Settings

from .base import Embedder


def get_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider

    if provider == "huggingface":
        from .huggingface import HuggingFaceEmbedder

        return HuggingFaceEmbedder(settings)

    # 나중에 다른 provider를 쓸 경우 여기에 분기를 추가한다.
    raise ValueError(f"지원하지 않는 embedding_provider: {provider!r}")
