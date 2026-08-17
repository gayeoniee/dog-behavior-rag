"""애플리케이션 설정.

모든 환경 변수는 여기 한 곳에서만 읽는다. 다른 모듈은 `get_settings()`를 통해서만 접근.
"""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
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
    cors_origin_regex: str = ""
    """정규식으로 오리진을 허용한다 (cors_origins와 함께 적용된다).

    LAN에서 접근할 때 필요하다. 개발 서버를 휴대폰이나 다른 PC에서 열면 오리진이
    `http://192.168.0.244:3000`처럼 되는데, IP가 DHCP로 바뀌므로 목록에 박아두면
    계속 깨진다. 사설 대역을 정규식으로 열어두는 편이 낫다.
    """

    # ── DB ──
    database_url: str = "postgresql+asyncpg://gyrag:gyrag@localhost:5432/gyrag"

    # ── Provider ──
    llm_provider: Provider = "huggingface"
    embedding_provider: Provider = "huggingface"

    # ── HuggingFace ──
    hf_embedding_model: str = "BAAI/bge-m3"
    hf_llm_model: str = ""
    hf_api_token: str = ""

    # ── OpenAI 호환 서버 (Gemini, LM Studio, Ollama, llama.cpp …) ──
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    """기본값은 Gemini의 OpenAI 호환 엔드포인트.

    로컬로 돌리려면 LM Studio `http://localhost:1234/v1`,
    Ollama `http://localhost:11434/v1`. 전부 같은 프로토콜이라 URL만 바꾸면 된다.
    """
    llm_model: str = "gemini-3.1-flash-lite"
    """Lite로 충분한 이유: 이 프로젝트가 LLM에 시키는 일이 작다.

    질의 재작성은 입출력 30토큰씩이고, 답변 생성도 검색된 근거를 정리하는
    일이라 추론 부담이 크지 않다. 나중에 팩트체크의 판정(judge) 단계처럼
    분별이 필요한 작업에서 부족하면 gemini-3.6-flash로 올리면 된다.
    """
    llm_api_key: str = ""
    """로컬 서버는 키를 검사하지 않지만 OpenAI 규격상 헤더가 필요한 구현이 있다.
    로컬이면 아무 값이나 넣으면 되고, Gemini면 AI Studio 키를 넣는다."""
    llm_reasoning_effort: str = ""
    """추론형 모델의 사고과정을 얼마나 쓸지. 비우면 요청에 아예 넣지 않는다.

    **비워두면 추론형 로컬 모델에서 짧은 호출이 전부 빈 응답이 된다.** gemma-4-e2b로
    실측: 질의 재작성(max_tokens=80)이 사고과정에 77토큰을 쓰고 `content`는 빈 채로
    `finish_reason=length`로 잘렸다. 재작성·근거 선별은 폴백이 있어서 죽지 않고 **조용히
    비활성화되는데**, 근거 선별이 빠지면 "고양이 모래" 질문에도 개 문서 5건이 붙는다.

    `none`으로 두면 사고과정을 끄고 답만 받는다(같은 호출이 32자 정상 출력). 이 프로젝트가
    LLM에 시키는 일은 재작성 30토큰·근거 선별 JSON 한 줄이라 사고과정이 필요 없다.
    필드를 모르는 서버가 400을 낼 수 있으므로 기본값은 "보내지 않음"이다.
    """

    llm_reasoning_reserve_tokens: int = 2048
    """추론이 켜져 있을 때 `max_tokens`에 더해 주는 여유분.

    호출자는 "답을 몇 토큰까지 받겠다"를 말하지 사고과정 예산을 모른다. 재작성은 80,
    근거 선별은 60을 요구하는데 gemma-4-e2b는 그 작은 작업에도 사고과정을 220~290토큰
    쓴다. 여유분이 없으면 예산을 사고과정이 다 먹고 `content`가 빈 채로 잘린다.

    2048인 이유: 1024로 두고 평가셋을 돌렸더니 20문항 중 1건이 재작성 한 번에
    사고과정 1101토큰을 써서 잘렸다. 평균(220~290)의 4배가 튀는 작업이 있다.
    상한이지 비용이 아니므로 — 모델은 끝나면 멈춘다 — 넉넉히 잡는 편이 맞다.

    프로토콜 층의 문제라 여기서 흡수한다 — 호출부(query_rewrite, evidence_select)가
    모델이 추론형인지 알 필요가 없다. 추론이 꺼져 있으면 더하지 않는다.
    """

    llm_max_retries: int = 4
    """429(요청 한도 초과)를 만났을 때 재시도 횟수.

    무료 티어는 분당 한도가 있어서 연속 호출이 몰리면 바로 걸린다. 평가셋 20문항이
    질문당 LLM을 2~3회 부르는데, 재시도가 없으면 절반이 폴백으로 떨어져 **측정
    자체가 오염된다** (실제로 겪었다). 서버 오류(5xx)에도 같이 적용된다.
    """
    llm_retry_base_delay: float = 2.0
    """재시도 대기의 기준값. 지수 백오프(2, 4, 8, 16초)로 늘어난다.
    응답에 Retry-After 헤더가 있으면 그 값을 우선한다."""

    llm_timeout_seconds: float = 120.0
    """CPU 폴백이나 긴 프롬프트를 감안한 값. GPU에 다 올라가면 훨씬 빨리 끝난다."""
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2
    """근거 기반 답변이라 창의성이 필요 없다. 낮을수록 자료에서 덜 벗어난다."""

    # ── 질의 재작성 ──
    evidence_select_enabled: bool = True
    """검색 결과 중 질문에 실제로 답하는 것만 LLM으로 골라낼지.

    끄면 검색된 top_k를 그대로 쓴다 — 코퍼스에 없는 주제에도 "가장 덜 무관한" 5건에
    근거를 붙여 답하게 되고, 그게 "RAG 느낌이 안 난다"의 직접 원인이었다.
    LLM 왕복이 1회 늘어 응답이 5~8초 느려진다.
    """

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

    embedding_truncate: bool = False
    """모델이 내놓는 벡터를 `embedding_dim`으로 **잘라 쓸지** (MRL).

    **아무 모델에나 켜면 안 된다.** 보통 임베딩은 앞부분만 잘라내면 뜻이 망가진다.
    MRL(Matryoshka Representation Learning)로 학습된 모델만 앞부분이 그 자체로
    쓸 만한 벡터가 되도록 훈련돼 있다 (Qwen3-Embedding 계열).

    기본이 False인 이유: 켜져 있으면 차원 불일치가 **에러 대신 조용한 성능 저하**로
    바뀐다. 명시적으로 켠 사람만 그 대가를 알고 있어야 한다.

    실측 (08장, 293문항 · Qwen3-Embedding-0.6B):

        1024차원  hit@5 51.2%  MRR 0.365
         512차원  hit@5 50.5%  MRR 0.366   ← 저장 절반, 성능 유지
         256차원  hit@5 45.7%  MRR 0.323
         128차원  hit@5 39.2%  MRR 0.278   ← 여기부터 유의미하게 나쁘다
    """
    embedding_query_prefix: str = ""
    """질의에만 붙이는 접두사. 모델마다 규약이 다르다.

        bge-m3      없음
        e5 계열     "query: "
        Qwen3       "Instruct: <과제 설명>\\nQuery: "

    **문서와 질의에 다른 처리를 하는 게 핵심이다.** 그래서 `Embedder` 프로토콜이
    `embed`(문서)와 `embed_query`(질의)를 나눠 갖고 있다. 규약을 안 지키면 모델이
    제 성능을 못 낸다 — e5는 접두사를 빼면 성능이 떨어진다고 모델 카드가 명시한다.
    """
    embedding_passage_prefix: str = ""
    """문서에만 붙이는 접두사 (e5 계열의 `"passage: "`).

    ⚠️ 이 값을 바꾸면 **전체 재적재가 필요하다.** 적재 때와 검색 때의 규약이
    다르면 조용히 엉뚱한 순위가 나온다."""

    @field_validator("embedding_query_prefix", "embedding_passage_prefix")
    @classmethod
    def _unescape(cls, value: str) -> str:
        r"""`.env`에 적은 `\n`을 실제 줄바꿈으로 바꾼다.

        Qwen3의 접두사는 `"Instruct: ...\nQuery: "`처럼 **줄바꿈이 의미를 갖는다.**
        그런데 `.env`는 값을 문자 그대로 읽어서 역슬래시와 n 두 글자가 들어온다.
        따옴표로 감싸는 방식은 dotenv 구현·셸마다 달라 믿을 수 없어서 여기서 푼다.

        **틀려도 에러가 안 난다** — 접두사가 조금 이상한 채로 임베딩이 되고
        검색 품질만 조용히 떨어진다. 그래서 설정 계층에서 확정한다.
        """
        return value.replace("\\n", "\n").replace("\\t", "\t")

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
    guide_boost: float = 0.03
    """실무 가이드(ASPCA/VCA/RSPCA/AAHA) 부스트. 논문(PMC)에는 0.

    코퍼스 청크의 97%가 논문이라 "어떻게 해요" 질문에 실행 절차 대신 연구 결과가
    올라온다. 0.03인 이유: 근거 있는 질문(0.714)과 주제 공백(0.673)의 차이가
    0.04라, 그보다 커지면 무관한 가이드가 정확한 논문을 밀어낸다.
    """
    max_chunks_per_document: int = 3
    """문서당 반환 청크 상한. AAHA 가이드라인 한 건이 코퍼스 글자 수의 절반이라
    이게 없으면 top_k 5개가 전부 같은 문서에서 나와 근거가 한 출처로 붕괴한다.

    **2 → 3으로 올렸다 (2026-08-17).** 리랭킹을 실험하다 발견했다 — 검색 실패
    사례가 "맞는 문서인데 엉뚱한 청크"인 경우가 많았는데, **정답이 그 문서의
    세 번째 청크면 상한이 정답을 막고 있었다.** 293문항 실측(청크 정확도 hit@5):

        상한 1   38.9%      상한 3   53.2%   ← 이득의 대부분
        상한 2   48.1%      상한 5   54.9%   ← 여기서 멈춘다

    **5가 아니라 3인 이유는 공짜가 아니기 때문이다.** 상한을 풀수록 문서 커버리지가
    81.9% → 80.9%로 떨어진다 — 근거가 한 출처로 쏠린다는 뜻이다. 상한 5는 한 문서가
    top_k 5칸을 다 채울 수 있고, 그러면 이 설정을 넣은 이유가 사라진다. 3은 최소
    두 문서를 보장하면서 이득의 75%를 가져간다.
    """
    language_background_weight: float = 1.0
    """언어별 배경 유사도를 얼마나 빼는가 (0=안 뺌, 1=전부 뺌).

    **1.0이 과할 수 있다는 게 실측으로 드러났다.** "집에 들어오면 뛰어올라요"에서
    주제에 맞는 한국어 문서(원점수 0.577)가 주제 밖 영어 논문(0.573)에 밀렸다.
    배경 차이가 0.073인데 실제 관련성 차이는 그보다 작았기 때문이다.

    배경이 언어 오프셋만 담으면 1.0이 맞다. 그런데 **한국어 코퍼스가 주제적으로
    좁아서**(전부 개 행동 상담) 개 질문에서는 배경도 같이 높아진다 — 빼면 안 될
    자리에서 많이 빼는 것이다.

    범위 밖 질문(고양이·가격)은 **보정을 아예 안 해도 한국어가 0%**다. 즉 지금
    보정이 하는 일은 개 질문에서 한국어를 눌러내는 것뿐이다:

        보정 0.00   상위5 중 한국어 46.7%   범위밖 0%
        보정 0.50                  29.5%          0%
        보정 1.00                  16.2%          0%

    **그래도 1.0이 낫다 (21문항 실측).** 뛰어오르기 하나를 살리려다 다른 데서
    더 잃는다:

        1.0   20/21   covered 15/15   실무 4.1   out-of-scope 4/4
        0.5   19/21   covered 14/15   실무 3.9   out-of-scope 4/4
        0.0   17/21   covered 13/15   실무 3.0   out-of-scope 3/4

    이 값을 낮추려면 **먼저 배경 계산을 고쳐야 한다.** 지금 배경은 언어 오프셋과
    "질문이 그 코퍼스의 주제 영역인가"를 같이 담는다. 범위 밖 질문에서 잰 격차
    (평균 +0.016)가 순수 언어 오프셋에 가까운데, 개 질문에서는 +0.073까지 벌어진다.
    """
    hnsw_ef_search: int = 400
    """HNSW 탐색폭. 기본값 40으로는 **진짜 상위 5개를 통째로 놓쳤다.**

    한국어 문서 328편을 넣은 뒤 "줄당김" 질문을 재보니:

        기본(ef_search=40)   영어 논문 0.49~0.52  ← 한국어가 하나도 없다
        정확 검색            한국어 5편 0.57~0.62 ← 진짜 상위 5개
        ef_search=400        정확 검색과 동일

    **작고 의미적으로 떨어진 무리는 근사 검색이 못 찾는다.** 한국어 청크는
    전체의 2.9%뿐이라 HNSW 그래프를 40번 훑는 동안 그 무리에 도달하지
    못한다. 인덱스가 "가장 가까운 것"이 아니라 "가까운 것 중 쉽게 닿는 것"을
    돌려주는 것이다.

    ⚠️ 이 값이 낮으면 **검색이 조용히 틀린다.** 에러도 경고도 없이 2등짜리
    문서가 사라진다. 코퍼스에 성격이 다른 자료를 섞을 때마다 확인할 것 —
    `enable_indexscan=off`로 정확 검색과 대조하면 바로 보인다.

    400은 실측으로 정확 검색과 일치하는 값이다. 코퍼스가 커지면 다시 재야 한다.
    """
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
