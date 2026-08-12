# GY_RAG

반려동물 훈련 / 문제행동 상담 RAG 시스템.

현재 상태: **검색까지 동작한다.** 청킹 → 임베딩(bge-m3) → pgvector 코사인 검색이
실제로 돌고, `/chat`이 근거 문서(`sources[]`)를 반환한다.
**답변 생성(LLM)은 아직 스텁이라 `answer`는 `"[stub] ..."`이다.**

## 요구사항

- [uv](https://docs.astral.sh/uv/) (pip 사용 안 함)
- Postgres + pgvector — Docker 또는 내장 서버(아래 참조)
- Node.js 20+ (Next.js 화면을 볼 경우)

## 빠른 시작

```bash
uv sync                      # 의존성 설치 (가볍다. torch 안 받음)
cp .env.example .env
uv run uvicorn app.main:app --reload
```

- API 문서: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/api/v1/health

검색까지 돌리려면 아래 "전체 파이프라인"을 따라간다.

## 전체 파이프라인 (새 머신 기준)

```bash
# 1) 수집 — 2~3분
uv sync --extra collect
cp .env.example .env
uv run python -m scripts.collect.fetch --all
uv run python -m scripts.collect.normalize
uv run python -m scripts.collect.report      # 네 축 + 커버리지 8개 PASS면 정상

# 2) DB
docker compose up -d db
uv run python -m scripts.db.init

# 3) 청킹 확인 (torch 없이) → 적재
uv run python -m scripts.db.load_corpus --dry-run
uv sync --extra hf --extra collect           # torch 포함, 수 GB
uv run python -m scripts.db.load_corpus

# 4) 실행
uv run uvicorn app.main:app --reload
```

> ⚠️ `uv sync`를 extra 없이 다시 돌리면 이전 extra가 **제거된다.**
> 항상 필요한 extra를 전부 나열할 것: `uv sync --extra hf --extra collect`

동작 확인:

```bash
curl -X POST localhost:8000/api/v1/chat \
  -H 'content-type: application/json' \
  -d '{"question":"강아지가 초인종 소리에 계속 짖어요"}'
# answer는 "[stub] ...", sources는 실제 근거 5건
```

## DB (pgvector) — 각자 로컬

**공용 DB를 쓰지 않는다.** 서브 PC에서 적재할 때 병목이 생길 수 있어서 팀에서
각자 로컬 DB를 쓰기로 했다. 그래서 기기를 옮기면 스키마 생성과 적재를 다시 해야
한다 (코퍼스도 저장소에 없으므로 어차피 재수집이 정상 경로다).

```bash
docker compose up -d db
uv run python -m scripts.db.init          # 테이블 + HNSW 인덱스
uv run python -m scripts.db.init --drop   # 스키마를 바꿨을 때 (재적재 필요)
curl localhost:8000/api/v1/health/ready   # {"status":"ready"}
```

`/health`(liveness)는 DB와 무관하게 200이고, `/health/ready`만 DB를 확인해
연결이 안 되면 503을 준다. 그래서 Postgres 없이도 개발이 막히지 않는다.

스키마와 필드별 근거는 **[docs/erd.md](docs/erd.md)** 에 있다.

### Docker나 관리자 권한이 없는 머신

`pgserver`가 PostgreSQL 16 + pgvector 바이너리를 휠에 담아 배포한다. 설치도
관리자 권한도 필요 없다 (학원 PC 등에서 검증함).

```bash
uv sync --extra pgdev
uv run python -m scripts.db.serve    # 기동 + .env의 DATABASE_URL 자동 갱신
uv run python -m scripts.db.serve --stop
```

기동할 때마다 빈 포트를 새로 잡기 때문에 `serve`가 `.env`를 직접 고친다.

## 의존성 그룹

무거운 ML 패키지는 기본 설치에서 빼뒀다. 필요할 때만 붙인다.

| 명령 | 용도 |
|---|---|
| `uv sync` | API 서버만 (수 초) |
| `uv sync --extra hf` | 실제 임베딩 (sentence-transformers, torch — 수 GB) |
| `uv sync --extra collect` | 데이터 수집 파이프라인 |
| `uv sync --extra pgdev` | 내장 Postgres+pgvector (Docker 없는 머신) |
| `uv sync --extra demo` | Streamlit 데모 |

## 화면

**표준 화면은 `web/`의 Next.js다.** Streamlit 데모는 디버그용으로 남겨뒀다.

```bash
cd web
npm install
cp .env.local.example .env.local
npm run dev        # http://localhost:3000
```

FastAPI 쪽 `.env`의 `CORS_ORIGINS`에 `http://localhost:3000`이 있어야 한다
(`.env.example` 기본값이 그렇게 돼 있다).

`answer`가 스텁인 동안 화면 상단에 "LLM 미연결 — 근거 문서만 표시" 배너가 뜬다.
지금 만든 게 검색이지 답변 생성이 아니므로 화면이 그걸 감추지 않게 한 것이다.

### Streamlit 데모 (선택)

```bash
uv sync --extra demo
uv run uvicorn app.main:app --reload         # 터미널 1
uv run streamlit run demo/streamlit_app.py   # 터미널 2
```

## 데이터 수집

수집 대상은 `data/sources.yaml`에 선언한다 (코드 수정 없이 소스 추가 가능).
소스는 네 답변 축 — **problem**(문제행동) / **cause**(행동의 이유) /
**training**(훈련·교정) / **medical**(의학적 감별) — 을 모두 덮도록 선정한다.

```bash
uv sync --extra collect

# 1) 수집 → data/raw/
uv run python -m scripts.collect.fetch --source avsab-humane-training   # 단일
uv run python -m scripts.collect.fetch --all --skip-pending             # 전체

# 2) 정규화 → data/processed/corpus.jsonl
uv run python -m scripts.collect.normalize

# 3) 품질·커버리지 리포트 (합격 기준 게이트 — 실패 시 exit 1)
uv run python -m scripts.collect.report
```

### 라이선스 현황 (2026-08-12 확인 완료)

`pending-check`는 남아 있지 않다. 네 소스의 ToS를 실제로 읽고 판정한 결과:

| 소스 | 판정 | 결과 |
|---|---|---|
| RSPCA, VCA | 자동 수집 허용 범위 | `personal-use-only` — 그대로 수집 |
| ASPCA | ToS가 로봇 접근 금지, 수동 복사는 허용 | `fetcher: local`로 전환 |
| MSD Vet Manual | ToS가 **수동 저장까지** 금지 | 소스에서 제거 |

판정 근거는 각 소스의 ToS 원문을 `data/sources.yaml`에 주석으로 인용해뒀다.
같은 조사를 반복하지 말 것.

> **배포 전 재검토 필요.** `personal-use-only` / `personal-use-only-manual-copy`
> 소스는 개인·비상업 이용으로 한정된다. 앱으로 배포하려면 코퍼스에서 제외하거나
> 허가를 받아야 한다. `corpus.jsonl`의 `license` 필드로 걸러낼 수 있고, 빠지는 축은
> CC-BY 오픈액세스 문헌(PMC OA 서브셋 등)으로 채우면 된다 — 후보는 `sources.yaml`의
> MSD 제거 주석에 적어뒀다.

#### 수동 저장 대기 소스 (사람이 해야 하는 일)

`fetcher: local` 소스는 **사람이 브라우저로 저장한 파일**만 쓴다. 자동화 도구로
받아오면 ToS가 금지한 "automatic device, process or means"에 그대로 해당하고,
ASPCA가 허용한 건 사람이 뜨는 사본 한 부이기 때문이다. 그래서 이 단계는
스크립트로 대신할 수 없다.

| 소스 | 할 일 |
|---|---|
| `aspca-behavior-issues` | `sources.yaml`의 `urls:` 8개를 브라우저로 열어 본문을 복사 → `data/raw/local/aspca-behavior-issues/<주제>.md` (또는 Ctrl+P → PDF 저장) |
| `korea-gov-materials` | 동물사랑배움터·animal.go.kr 자료실 PDF를 `data/raw/local/korea-gov-materials/`에 저장 |

넣은 뒤:

```bash
uv run python -m scripts.collect.fetch --source aspca-behavior-issues
uv run python -m scripts.collect.normalize
uv run python -m scripts.db.load_corpus   # content_hash 덕분에 기존 문서는 건너뛴다
```

디렉터리가 없으면 `fetch`가 0건으로 넘어갈 뿐 실패하지 않으므로, 나중에 채워도 된다.

#### 새 소스를 추가할 때

`license: pending-check`로 넣으면 fetch가 거부한다. 확인 작업의 기계적인 부분은
스크립트가 대신한다 — **로컬에서** 실행:

```bash
uv run python -m scripts.collect.check_license
```

소스별로 ① robots.txt 판정(자동)과 ② ToS 확인 안내(수동, 소스당 2분)가 나온다.
출력이 시키는 대로 약관 페이지에서 아래 문구를 찾고, 제안된 값을 `sources.yaml`에 붙여넣으면 끝.

| ToS에서 발견한 것 | 판단 | `license`에 쓸 값 |
|---|---|---|
| "automated means/scraping/crawling" 금지 | 자동 수집 불가 → `fetcher: local`로 전환 (브라우저 저장 파일 사용) | — |
| "personal, non-commercial use" 한정 | 개인 테스트 가능. 앱 배포 시 재검토 | `personal-use-only` |
| 관련 조항 없음 + robots 허용 | 정중한 수집(1req/s, UA 명시)은 통상 수용 범위 | `robots-allowed-no-tos-clause` |

규칙:

- **`license: pending-check`인 소스는 fetch가 거부한다.** 위 절차로 확인하고
  `sources.yaml`의 `license`를 갱신해야 수집된다. `--skip-pending`으로 건너뛸 수는 있다.
- HTML 수집은 robots.txt를 준수하고(거부 시 스킵이 아니라 **에러**), 요청 간 1초 지연을 둔다.
  단 **robots.txt 허용이 곧 ToS 허용은 아니다** — ASPCA·MSD는 robots.txt가 전부 허용인데
  ToS가 자동 수집을 금지한 사례다. 둘 다 확인해야 한다.
- 자동 수집이 막힌 사이트(ToS 금지)나 로그인이 필요한 사이트(동물사랑배움터 등)는 파일을
  직접 받아 `data/raw/local/<source_id>/`에 넣으면 `local` fetcher가 처리한다(.pdf/.txt/.md).
  현재 수동 저장 대기: `aspca-behavior-issues`(8페이지), `korea-gov-materials`.
  저장할 URL 목록은 `sources.yaml`의 해당 항목 `urls:`에 있다.
- 유튜브·블로그는 수집하지 않는다 — ToS/저작권 문제에 더해, 지배이론 등
  폐기된 훈련법이 섞여 답변 품질을 떨어뜨린다. 기관·학술 자료만 쓴다.
- 커버리지 질문(`data/coverage_questions.yaml`)은 나중에 검색이 붙으면
  그대로 검색 품질 회귀 테스트가 된다.

> ⚠️ 개발 컨테이너에서는 이그레스 프록시가 대상 도메인을 막을 수 있다.
> 실제 네트워크 수집은 로컬 머신에서 실행할 것.

## 개발

```bash
uv run pytest          # 테스트 (DB·네트워크 없이 동작)
uv run ruff check .    # 린트
uv run ruff format .   # 포맷
uv run mypy app        # 타입 체크
```

## 구조

```
app/
├── main.py                 앱 팩토리 + lifespan
├── core/config.py          모든 환경변수는 여기서만 읽는다
├── api/v1/                 health / chat / documents 엔드포인트
├── schemas/                요청·응답 스키마 (= 안드로이드 앱과의 API 계약)
├── services/
│   ├── rag_service.py      임베딩 → 검색 → 프롬프트 → 생성 오케스트레이션
│   ├── embeddings/         Embedder Protocol + HuggingFace 구현
│   ├── llm/                LLMClient Protocol + HuggingFace 구현
│   └── vectorstore/        VectorStore Protocol + pgvector 구현
└── db/                     엔진/세션, ORM 모델
demo/streamlit_app.py       테스트용 UI
```

## LLM (로컬)

**HF Inference API는 쓰지 않는다.** 2026-08 기준 무료 크레딧이 월 $0.10이고,
`hf-inference`는 2025년 7월부터 CPU 추론 위주로 축소돼 최신 instruct 모델을
서빙하지 않는다. GGUF 양자화 모델을 로컬 GPU에 올리는 쪽이 무료이면서 더 빠르다.

`openai-compatible` provider는 특정 서비스가 아니라 **프로토콜**에 붙는다.
LM Studio · Ollama · llama.cpp 서버 · vLLM · Groq · OpenRouter가 전부 같은
`/v1/chat/completions`를 쓰므로 `LLM_BASE_URL`만 바꾸면 된다.

LM Studio 기준:

1. 모델 받기 — `Qwen2.5-7B-Instruct` GGUF **Q4_K_M** (약 4.7GB)
2. Developer 탭 → **Start Server** (기본 포트 1234)
3. `.env`의 `LLM_MODEL`을 LM Studio가 표시하는 모델 id와 맞춘다

VRAM 6GB면 7B Q4가 거의 전부 올라가 20~30 tok/s 나온다. VRAM이 부족하면
`Qwen2.5-3B-Instruct` Q4_K_M(약 2GB)로 낮춘다.

서버가 꺼져 있어도 앱은 뜬다 — `/chat`만 503을 준다.

### 질의 재작성

한국어 질문을 영어 기술표현으로 바꾼 뒤 임베딩한다. bge-m3가 *주제*는 교차언어로
넘나들지만 *기법 명칭*은 못 넘기 때문이다. 코퍼스 282건에서 실측:

| 질의 | 원문으로 검색 | 영어 재작성 후 |
|---|---|---|
| 복종 자세를 강제로 1~2분 유지 (알파 롤) | 0.552 — 무관 문서 | **0.724 — AVSAB 지배이론 성명서** |
| 목줄 잡고 "안 돼" 소리치기 | 0.580 — 무관 문서 | **0.629 — AAHA 가이드라인** |

코퍼스에는 반박 근거가 다 있었다(`alpha roll` 6건, `pinning` 22건). 찾지 못했을
뿐이다. 커버리지 질문 8개가 전부 통과했던 건 그것들이 주제형 질문이라 생긴 착시다.

LLM 서버가 꺼져 있으면 원문으로 폴백하므로 검색이 막히지는 않는다
(`QUERY_REWRITE_ENABLED=false`로 끌 수도 있다).

### Provider 교체

LLM/임베딩 구현체는 Protocol 뒤에 숨어 있고, 선택 지점은 각 `registry.py`
한 곳뿐이다. Claude 등으로 바꾸려면:

1. `app/services/llm/anthropic.py`에 `LLMClient` Protocol을 만족하는 클래스 추가
2. `app/services/llm/registry.py`에 분기 한 줄 추가
3. `app/core/config.py`의 `Provider` Literal에 이름 추가
4. `.env`의 `LLM_PROVIDER` 변경

호출부(`rag_service.py`, 엔드포인트)는 건드릴 필요 없다.

## 다음 작업 (TODO)

코드에 `TODO(내일)` 주석으로 표시돼 있다.

- [x] **로컬에서 실제 수집 실행** — ToS 확인 완료, fetch → normalize → report 통과
      (문서 11건 / 네 축 전부 / 커버리지 질문 8개 전부 PASS)
- [ ] ASPCA 8페이지 수동 저장 → `data/raw/local/aspca-behavior-issues/` (개인 이용 한정)
- [ ] 임베딩 모델 확정 → 차원 확정 → `chunks.embedding Vector(N)` 컬럼 정의
- [ ] `db/models.py` 실제 테이블 (`documents`, `chunks`) + HNSW 인덱스
- [ ] Alembic 마이그레이션 도입
- [ ] 문서 청킹 전략 + 적재 파이프라인 (`POST /documents` — corpus.jsonl 입력)
- [ ] pgvector 코사인 검색 + `methodology` 필터 / `authority_tier` 부스팅
- [ ] LLM 연결 (로컬 추론 vs Inference API) + 프롬프트 튜닝
- [ ] 임베딩 모델을 lifespan에서 1회 로딩하도록 변경 (요청마다 로딩 금지)
