"""애플리케이션 설정.

모든 환경 변수는 여기 한 곳에서만 읽는다. 다른 모듈은 `get_settings()`를 통해서만 접근.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["huggingface", "openai-compatible"]
"""지원하는 provider. 확장할 때 여기와 registry.py 두 곳만 고치면 된다.

`openai-compatible`은 특정 서비스가 아니라 **프로토콜**이다. LM Studio, Ollama,
llama.cpp 서버, vLLM, Groq, OpenRouter가 전부 같은 `/v1/chat/completions`를
쓰므로 구현 하나로 다 커버된다. 바꿀 때는 LLM_BASE_URL / LLM_MODEL만 고치면 된다.
"""


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

    # ── OpenAI 호환 로컬/원격 서버 (LM Studio, Ollama, llama.cpp …) ──
    llm_base_url: str = "http://localhost:1234/v1"
    """LM Studio 기본 포트. Ollama는 http://localhost:11434/v1."""
    llm_model: str = "qwen2.5-7b-instruct"
    llm_api_key: str = "not-needed"
    """로컬 서버는 키를 검사하지 않지만 OpenAI 규격상 헤더가 있어야 하는 구현이 있다."""
    llm_timeout_seconds: float = 120.0
    """CPU 폴백이나 긴 프롬프트를 감안한 값. GPU에 다 올라가면 훨씬 빨리 끝난다."""
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2
    """근거 기반 답변이라 창의성이 필요 없다. 낮을수록 자료에서 덜 벗어난다."""

    # ── 질의 재작성 ──
    query_rewrite_enabled: bool = True
    """한국어 질문을 영어 기술표현으로 바꾼 뒤 임베딩할지.

    측정 근거: "복종 자세를 강제로 1~2분 유지" 원문은 무관 문서를 물어왔고(0.552),
    영어 기술표현으로 바꾸니 AVSAB 지배이론 성명서가 0.724로 1위였다. bge-m3가
    주제는 교차언어로 넘나드는데 기법 명칭(알파 롤 ↔ alpha roll)은 못 넘는다.
    """

    # ── Anthropic (provider 전환 시에만 사용. 현재 코드 경로 없음) ──
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"

    # ── 임베딩 ──
    embedding_dim: int = 1024
    """pgvector `Vector(N)` 컬럼 차원의 유일한 근원. BAAI/bge-m3 = 1024.

    모델 introspection으로 정하지 않는 이유: `db/models.py`가 import 시점에 차원을
    확정해야 하는데, 그러자고 torch를 import하면 `--extra hf` 없이 앱을 띄우거나
    테스트를 돌리는 게 전부 깨진다. 대신 warmup에서 실제 모델 차원과 대조한다.
    이 값을 바꾸면 chunks 테이블을 다시 만들어야 한다 (init --drop).
    """
    embedding_device: Literal["auto", "cuda", "cpu"] = "auto"
    """임베딩을 어디서 돌릴지.

    `auto`면 sentence-transformers가 CUDA가 있을 때 GPU를 쓴다. **VRAM을 LLM과
    나눠 쓴다면 주의할 것** — 이 PC는 VRAM 6GB인데 7B Q4 모델이 4.7GB를 쓰므로
    bge-m3(약 2.3GB)까지 올리면 OOM이다. LM Studio를 같이 돌릴 거면 `cpu`로 두거나,
    적재(load_corpus)를 돌릴 때만 LM Studio를 내리고 `cuda`를 쓴다.
    적재는 오프라인 배치라 후자가 낫다.
    """
    embedding_batch_size: int = 8
    embedding_max_seq_length: int = 1024
    """bge-m3의 기본값은 8192지만 그만큼 필요하지 않다. 1200자 청크가 ~300토큰,
    한국어 2000자 질문이 ~1300토큰이라 1024면 충분하고 CPU 메모리 스파이크를 막는다."""
    embedding_warmup: bool = True
    """앱 기동 시 모델을 선로딩할지. 적재는 오프라인 CLI(scripts.db.load_corpus)가
    하므로, API만 띄워 검색을 안 쓸 거면 꺼서 기동을 가볍게 할 수 있다."""

    # ── 청킹 ──
    chunk_size: int = 1200
    """top_k=5 × 1200자 ≈ 6000자. 다음 라운드에 어떤 LLM을 붙여도 프롬프트에 들어간다.
    모델(bge-m3, 8192토큰)이 아니라 프롬프트 예산이 제약이라 이 값이 나왔다."""
    chunk_overlap: int = 150
    chunk_min_size: int = 200

    # ── 검색 ──
    top_k: int = 5
    authority_boost: float = 0.02
    """authority_tier 부스팅 상한. tier1 +0.02 / tier2 +0.01 / tier3 +0.

    작은 코퍼스에서 1위와 5위의 코사인 격차가 보통 0.02~0.10이라, 이 상한은 근소한
    차이만 뒤집고 의미 없는 tier1을 강한 tier3 위로 올리지는 못한다. 권위는
    타이브레이커지 검색 신호가 아니라는 뜻이고, 이 비대칭이 의도한 설계다.
    """
    max_chunks_per_document: int = 2
    """문서당 반환 청크 상한. AAHA 가이드라인 한 건이 코퍼스 글자 수의 절반이라
    이게 없으면 top_k 5개가 전부 같은 문서에서 나와 근거가 한 출처로 붕괴한다."""
    candidate_multiplier: int = 4
    """부스팅·다양성 재랭킹 전에 몇 배수를 과다 조회할지.

    부스트를 SQL ORDER BY에 넣지 않는 이유이기도 하다 — 조인 컬럼이 낀 표현식으로
    정렬하면 HNSW 인덱스를 못 써서 코퍼스가 커지는 순간 조용히 full scan이 된다.
    """

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """프로세스당 한 번만 .env를 읽는다."""
    return Settings()
