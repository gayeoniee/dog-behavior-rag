"""애플리케이션 설정.

모든 환경 변수는 여기 한 곳에서만 읽는다. 다른 모듈은 `get_settings()`를 통해서만 접근.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["huggingface"]
"""지원하는 provider. Claude 등으로 확장할 때 여기와 registry.py 두 곳만 고치면 된다."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 앱 ──
    app_env: str = "local"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_origins: str = "*"

    # ── DB ──
    database_url: str = "postgresql+asyncpg://gyrag:gyrag@localhost:5432/gyrag"

    # ── Provider ──
    llm_provider: Provider = "huggingface"
    embedding_provider: Provider = "huggingface"

    # ── HuggingFace ──
    hf_embedding_model: str = "BAAI/bge-m3"
    hf_llm_model: str = ""
    hf_api_token: str = ""

    # ── Anthropic (provider 전환 시에만 사용. 현재 코드 경로 없음) ──
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"

    # ── 검색 ──
    top_k: int = 5

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """프로세스당 한 번만 .env를 읽는다."""
    return Settings()
