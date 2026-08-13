# GY_RAG

반려동물 훈련 / 문제행동 상담 RAG 시스템.

현재 상태: **끝에서 끝까지 동작한다.** 질의 재작성 → 임베딩(bge-m3) →
pgvector 코사인 검색 → LLM 답변 생성. `/chat`이 근거 문서(`sources[]`)와 함께
그 자료에 기반한 한국어 답변을 반환한다.

코퍼스 282건 / 청크 11,281개, 커버리지 질문 8/8 통과.

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
# 근거 5건 + 자료에 기반한 한국어 답변
```

> ⚠️ **PowerShell 5.1에서 `Invoke-RestMethod`로 확인하지 말 것.** 응답
> `Content-Type: application/json`에 charset이 없으면 PS 5.1이 ISO-8859-1로
> 디코딩해서 한글이 깨진 것처럼 보인다. **서버는 정상 UTF-8이고 브라우저는
> 제대로 나온다.** 터미널에서 확인하려면 `curl` 또는:
>
> ```powershell
> uv run python -c "import httpx;print(httpx.post('http://localhost:8000/api/v1/chat',json={'question':'강아지가 짖어요'},timeout=120).json()['answer'])"
> ```

## 팩트체크 (`POST /api/v1/factcheck`)

유튜브·블로그는 코퍼스에 넣지 않기로 했다(ToS + 지배이론 오염). 그 결정과 짝을
이루는 기능이다 — **오염원으로 들이는 대신 검증 대상으로 받는다.** 코퍼스에
AVSAB 지배이론 성명서 같은 반박 근거가 있어서 성립한다.

```
추출(LLM) → 재작성(LLM) → 검색 → 판정(LLM)
```

실제 결과 (삭제한 한국어 안내서의 지배이론 조언을 그대로 넣음, 24.8초):

| 주장 | 판정 |
|---|---|
| 마운팅은 서열이 위라서 한다 | **자료와 배치** — AVSAB 지배이론 성명서 |
| "안 돼" 소리치고 분리해야 한다 | **자료와 배치** — 주의 돌리기·대체행동 보상 권장 |
| 복종 자세 1-2분 유지 | **자료와 배치** — "'앉아'는 있어도 1-2분 유지는 자료에 없다" |
| 마킹 시 목줄 잡기 | **자료에 근거 없음** |
| 산책 하루 두 번 | **자료에 근거 없음** |

**`not_covered`가 설계의 핵심이다.** 이 선택지가 없으면 모델이 근거 없이
supported/contradicted 중 하나를 고른다 — 팩트체크의 가장 흔한 실패 모드고,
근거 인용이 존재 이유인 이 프로젝트에서는 특히 나쁘다. 위 5건 중 2건이
`not_covered`로 나온 게 이게 동작한다는 증거다.

방어 장치:

- 근거가 0건이면 판정 LLM을 **부르지 않고** `not_covered`로 확정한다
- 인용 없는 `supported`/`contradicted`는 코드에서 `not_covered`로 강등한다
  (프롬프트로 부탁하지 않는다 — 모델이 지킬 거라고 믿을 수 없다)
- 응답에 **코퍼스 편향 고지**를 항상 붙인다. 우리 코퍼스는 보상 기반 문헌으로만
  구성돼 있어 혐오·지배 기반 주장은 구조적으로 '배치'로 기운다. 의도된 설계지만
  중립적 제3자 판정인 척하면 안 된다

주장 5개면 LLM 왕복이 11회다. 주장끼리 독립이라 LLM 구간을 `asyncio.gather`로
묶어 57초 → 25초로 줄였다. **DB는 병렬로 쓰지 않는다** — `AsyncSession` 하나는
커넥션 하나라 동시 쿼리를 못 견딘다. 검색은 수십 ms라 병목이 아니다.

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

화면은 두 개다:

| 경로 | 용도 |
|---|---|
| `/` | 질문 → 근거 기반 답변. 답변이 `[자료 N]`으로 인용하고 근거 카드에 같은 번호가 붙는다 |
| `/factcheck` | **어디서 본 조언을 붙여넣으면 코퍼스와 대조** |

`answer`가 스텁이면 화면 상단에 "LLM 미연결" 배너가 뜬다 (지금은 안 뜬다).

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

## LLM

`openai-compatible` provider는 특정 서비스가 아니라 **프로토콜**에 붙는다.
Gemini · LM Studio · Ollama · llama.cpp · vLLM · Groq · OpenRouter가 전부 같은
`/v1/chat/completions`를 쓰므로 **`LLM_BASE_URL`만 바꾸면 된다** (코드 변경 없음).

**HF Inference API는 쓰지 않는다.** 2026-08 기준 무료 크레딧이 월 $0.10이고,
`hf-inference`는 2025년 7월부터 CPU 추론 위주로 축소돼 최신 instruct 모델을
서빙하지 않는다.

### (A) Gemini 무료 티어 — 기본값

```bash
# https://aistudio.google.com/apikey 에서 무료 발급 (카드 등록 불필요)
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=gemini-3.1-flash-lite
LLM_API_KEY=<발급받은 키>
```

`gemini-3.1-flash`는 없는 id다 — 3.1은 `gemini-3.1-flash-lite`만 있다.
이 프로젝트가 LLM에 시키는 일이 작아서(질의 재작성은 입출력 30토큰) Lite로
충분하다. 분별이 필요한 작업에서 부족하면 `gemini-3.6-flash`로 올린다.

**이 PC에서는 이쪽이 유리하다.** VRAM이 6GB뿐이라 로컬 LLM(7B Q4 = 4.7GB)과
임베딩(bge-m3 = 2.3GB)이 동시에 안 올라가는데, LLM을 밖으로 빼면 GPU를 임베딩이
독점한다. 대신 질의가 외부로 나가고, 무료 티어는 통상 제품 개선에 사용된다.

### (B) LM Studio — 완전 무료·오프라인

`google/gemma-4-e2b`(4.6B, 4.41GB)로 실제로 돌려본 설정이다 (2026-08-13).

1. Developer 탭 → **Start Server** (기본 포트 1234). CLI로는 `lms server start`
2. 모델을 **컨텍스트 8192 이상으로** 로드한다. 답변 프롬프트가 근거 5청크 ×
   1200자라 4096으로 로드하면 근거가 잘려 조용히 나빠진다 (`lms ps`로 확인)
3. `.env`:

```bash
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=google/gemma-4-e2b   # GET /v1/models 가 돌려주는 id 그대로. 틀리면 404
LLM_API_KEY=not-needed         # 검사하진 않지만 헤더는 항상 나간다
LLM_REASONING_EFFORT=medium    # 추론형 모델일 때만. 아래 참조
EMBEDDING_DEVICE=cpu           # 모델이 4.41GB라 bge-m3(2.3GB)까지 안 올라간다
```

**임베딩을 CPU로 내리는 게 실질적인 대가다.** GPU가 11배 빠르다는 실측은
적재(11,281청크) 이야기이고 서빙은 요청당 짧은 텍스트 1건이라 체감이 작지만,
`load_corpus`를 돌릴 때는 LM Studio를 내리고 `auto`로 되돌려야 한다.

#### 추론형(thinking) 모델이면 반드시 확인할 것

gemma-4-e2b는 사고과정을 먼저 뱉는다. 사고과정도 completion 토큰을 쓰므로
**질의 재작성(80토큰)·근거 선별(60토큰)의 예산을 사고과정이 다 먹고 `content`가
빈 채로 잘린다.** 두 단계는 실패해도 폴백이 있어 죽지 않고 **조용히 비활성화**되는데,
근거 선별이 빠지면 "고양이 모래" 같은 범위 밖 질문에도 개 문서 5건이 붙는다.

`LLM_REASONING_EFFORT`를 설정하면 `openai_compatible.py`가
`LLM_REASONING_RESERVE_TOKENS`(기본 2048)를 예산에 더해 이 문제를 막는다.
빈 응답이 잘려서 나오면 경고 로그가 원인과 해결책을 함께 알려준다.

호출별로도 갈린다 — 근거 선별·범위 판정만 추론을 켜고(`reasoning=True`), 질의
재작성과 답변 생성은 끈다(`reasoning=False`). 생성에 켜두면 요청당 13초를 더 쓰면서
평가 점수는 그대로였고, 재작성은 "줄 당김 → leash pulling" 수준의 용어 치환이라
사고과정이 필요 없다(6초 → 1초, 평가 점수 동일).

#### 속도의 천장 (2026-08-13 실측)

    생성          25 tok/s
    입력 prefill  1,600토큰에 3초

**응답 시간이 전부 "생성한 토큰 수"에 묶여 있다.** 근거 선별이 8~12초인 것은 입력이
커서가 아니라 사고과정 토큰을 250~300개 만들기 때문이다. 그래서 입력을 줄이는 것
(`top_k`, 발췌 길이)은 효과가 거의 없고, 줄일 수 있는 건 토큰 수뿐이다.

**LM Studio 로드 옵션으로는 더 못 얻는다. 다시 시도하지 말 것:**

| 옵션 | 결과 |
|---|---|
| `--gpu max` | 무효 — 자동 판정이 이미 전체 오프로드였다 (26.4 → 24.8 tok/s, 잡음) |
| `--parallel 1` | 무효 — 단일 요청에는 의미가 없다 (기본값은 4) |
| `--speculative-draft-mtp` | **불가** — 이 GGUF에 MTP 헤드가 없다 |

추측 디코딩을 쓰려면 같은 계열의 작은 드래프트 모델을 따로 받아야 하는데,
VRAM 6GB에 4.41GB 모델과 같이 올릴 자리가 없다.

남은 8~12초는 근거 선별이고, **이건 줄이면 안 되는 비용이다** — 여기 추론을 끄면
거절 판정이 무너져 14/20으로 되돌아간다. "포기할 줄 아는 RAG"의 가격이다.

#### 실측 (평가셋 20문항, `data/eval_results/`)

로컬 모델을 쓸 만하게 만드는 과정. **크레이트 항목 라벨을 고치기 전 기준이라**
아래 최종 표와 총점이 다르다.

| 구성 | covered | uncovered | out-of-scope | 전체 | 응답 |
|---|---|---|---|---|---|
| gemma-4-e2b 추론 끔 | 13/14 | 0/2 | 1/4 | 14/20 | 5.9초 |
| gemma-4-e2b 추론 켬 | 14/14 | 0/2 | 1/4 | 15/20 | 28초 |
| gemma-4-e2b + 범위 판정 분리 | 14/14 | 0/2 | 4/4 | 18/20 | 19~21초 |

**로컬로 좁힌 4점 중 3점은 범위 판정을 떼어낸 것이다.** 근거 선별 한 번의 호출에
"쓸 근거 고르기"와 "개 질문인지"를 같이 시키면 4.6B는 후자를 놓친다
(`evidence_select.py`의 `DOMAIN_SYSTEM` 참조).

단계별로는 재작성 1초 / 근거 선별 8~12초 / 범위 판정 2초(근거 0건일 때만) /
답변 생성 2~3초다.

#### 최종 비교 (크레이트 라벨 수정 후, 같은 설정으로 양쪽 측정)

| 구성 | covered | uncovered | out-of-scope | 전체 | 응답 |
|---|---|---|---|---|---|
| Gemini `gemini-3.1-flash-lite` | 14/15 | 0/1 | 4/4 | 18/20 | 4~5초 |
| gemma-4-e2b (로컬) | 15/15 | 0/1 | 4/4 | 18~19/20 | 13~16초 |

**이 1점 차를 "로컬이 더 낫다"로 읽으면 안 된다.** 떨어진 항목이 서로 다르고
(Gemini는 리콜, gemma는 없음) 둘 다 실행마다 뒤집히는 항목이다.

**±1점은 잡음이다.** 질의 재작성이 LLM이라 실행마다 문구가 달라지고, 그게 검색
순위를 바꾼다. 네 번의 실행을 항목별로 겹쳐 보면:

- **리콜** — 1위 문서가 매번 다르다(전자 목걸이 논문 / AAHA / ASPCA 공격성 /
  ASPCA 짖음). 정작 RSPCA "Train Your Dog To Come When Called" 문서는 한 번도
  안 올라온다. **랭킹 문제**이고 통과·실패가 운으로 갈린다
- **크레이트** — 라벨 수정 후 양쪽 다 통과(ASPCA 분리불안 문서가 1위). 다만 다른
  실행에서는 0.650으로 커트라인 0.671에 밀린 적이 있다. 재작성이
  "desensitization and counterconditioning"을 붙이면 방법론 문서가 상위를 채운다
- **뛰어오르기** — 네 번 다 같은 문서, 네 번 다 실패. **이것만 진짜 코퍼스 공백이다**

총점 한 자리를 비교하기 전에 어느 항목이 움직였는지 보라는 뜻이다.

서버가 꺼져 있거나 키가 틀려도 앱은 뜬다 — `/chat`만 503을 주고, 원인별로
다른 메시지를 낸다(연결 실패 / 인증 실패 / 모델 없음 / 한도 초과).

## 검색 품질 평가 (`scripts/eval/`)

**"성능이 거지같다"를 숫자로 바꾸는 장치.** 이게 없어서 코퍼스를 11 → 282건으로
늘리고도 체감 품질이 그대로인 걸 한참 뒤에 알았다 — 늘어난 266건이 전부 연구
논문이라 **양은 25배가 됐는데 주제 커버리지는 그대로**였다.

```bash
uv run python -m scripts.eval.retrieval_report                       # 현재 점수
uv run python -m scripts.eval.retrieval_report --save v2 --compare baseline
```

`data/eval_questions.yaml`은 보호자 실제 어투 20문항이고, 그룹마다 합격 기준이 다르다:

| 그룹 | 통과 조건 |
|---|---|
| `covered` | 근거를 찾아야 |
| `uncovered` | **거절해야** (자료 없다고 말해야) |
| `out-of-scope` | **거절해야** |

거절 케이스가 절반인 게 핵심이다. 근거가 없을 때 지어내지 않는 것이 "RAG 느낌"의
정체이기 때문이다. 자료를 보강해 커버되면 `expect`를 `covered`로 올린다.

### 개선 이력 (2026-08-12)

| 단계 | 통과 | 비고 |
|---|---|---|
| 기준선 | **10/20** | out-of-scope 0/4 — "중성화 비용"에도 근거 5건을 붙여 답했다 |
| + RSPCA 7페이지 | 10/20 | 배변·리콜 문서가 생김 (판정은 아직 그대로) |
| + 근거 선별 | 16/20 | **out-of-scope 4/4** — 거절할 줄 알게 됐다 |
| + `doc_type` 부스트 | **17/20** | 실무가이드 비중 1.8 → **3.5/5** |

남은 3건은 뛰어오르기·줄당김·자원보호로, **전용 문서가 없는 진짜 공백**이다.

### 이전 측정 (2026-08-12, 문서 282건 / 청크 11,281개)

`uv run pytest -m integration` 기준 — **22 passed, 1 xfailed**:

| 항목 | 결과 |
|---|---|
| 커버리지 질문 8개 (**재작성 경로 = 실제 운영 경로**) | **8 / 8 PASS** |
| 커버리지 질문 8개 (재작성 없는 원문 검색) | 7 PASS / 1 xfail |
| 문서당 청크 상한 | PASS (한 문서가 top_k 독점 안 함) |
| aversive 문서 제외 | PASS (합성 문서로 검증) |
| observation 구획 제외 | PASS |
| 재적재 멱등성 | PASS |

xfail 1건은 **"벌을 주면 안 되나요? 혼내면 그때만 멈춰요"** — 기법 질문이라
재작성 없이는 못 찾는다. 코퍼스가 11건일 땐 통과했는데 282건이 되자 실패했다.
후보가 적을 땐 우연히 맞았던 것이다. **재작성을 태우면 통과한다** (키워드 4개
전부, 점수 0.53 → 0.75). 이 xfail은 "재작성이 없으면 이렇게 된다"는 기록으로
남겨둔다 — strict라서 원문만으로 통과하기 시작하면 테스트가 알려준다.

재작성은 다른 7개 질문의 키워드 적중도 늘렸다 (예: 초인종 질문이
`bark, territorial` → `bark, alarm, territorial`).

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
