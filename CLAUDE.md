# CLAUDE.md

반려동물 훈련·문제행동 상담 RAG 시스템. 최종 목표는 안드로이드 앱이고, 지금은 API와
수집 파이프라인, 웹 채팅 화면까지 동작하는 단계다.

명령어·구조·수집 절차는 `README.md`에 있다. 이 파일은 **README에 없는 판단 규칙과
현재 위치**만 담는다.

## 현재 위치

- ✅ 데이터 수집 파이프라인 동작. **적재 기준 문서 294건 / 청크 11,354개**
  (PMC OA + ASPCA 수동분 + RSPCA 훈련 페이지), 네 축 전부, 커버리지 질문 8개 전부 PASS
- ✅ **검색까지 동작한다.** 청킹 → bge-m3 임베딩(1024차원) → pgvector 코사인 검색.
  `/chat`이 실제 근거 문서를 반환하고, 커버리지 질문 8개가 상위 5청크에서 근거를 찾는다
  (한국어 질문 → 영어 문서 교차언어 검색)
- ✅ Next.js 화면(`web/`) — **상담 채팅 UI**(말풍선 누적, 근거 접기, 멀티턴, 새 상담),
  ERD(`docs/erd.md`)
- ✅ **LLM 연결 완료.** `openai-compatible` provider + Gemini 무료 티어
  (`gemini-3.1-flash-lite`). `/chat`이 근거 기반 한국어 답변을 낸다 (약 17초 —
  LLM 왕복 2회: 질의 재작성 + 답변 생성)
- ✅ **질의 재작성으로 커버리지 8/8.** 재작성 없으면 7/8 (기법 질문 1건 실패)
- ✅ **팩트체크** (`POST /api/v1/factcheck`, 화면 `/factcheck`). 유튜브·블로그를
  코퍼스에 안 넣기로 한 결정과 짝을 이룬다 — 오염원이 아니라 검증 대상으로 받는다
- ✅ **검색 품질 평가셋** (`scripts.eval.retrieval_report`, 20문항). 기준선 10/20 →
  현재 **19/20** (covered 14/14, out-of-scope 4/4). 남은 1건은 뛰어오르기 —
  전용 문서가 없는 진짜 공백이고, 재주 훈련과 함께 라이선스 깨끗한 소스가 없다
  (판단 근거는 `sources.yaml`)
- ✅ **답변 형식 고정과 되묻기.** 훈련사 화법 + 폼 고정으로 1,200자 → 300자,
  마크다운은 서버에서 평문화(`plain_text.py`). 근거가 없어도 개 행동 질문이면
  거절 대신 원인을 가르는 질문을 되묻는다(`coverage=needs_detail`), 그 답을
  질의 재작성이 맥락으로 푼다(`history`)
- 다음 작업: 뛰어오르기 소스 확보 → Alembic → 안드로이드 앱

## 검색 품질에 대해 배운 것 (2026-08-12)

체감 품질이 나빠서("그냥 제미나이랑 대화하는 느낌") 추측 대신 측정한 결과:

- **코퍼스는 양이 아니라 주제 커버리지다.** 11 → 282건으로 늘렸는데 늘어난 266건이
  전부 논문이라 체감이 그대로였다. 보호자가 매일 묻는 배변·뛰어오르기·줄당김·리콜에
  전용 문서가 하나도 없었다. 문서가 있는 주제는 0.686~0.731, 없는 주제는 0.636~0.693
- **점수 임계값으로 "자료 없음"을 판별할 수 없다.** bge-m3의 한→영 코사인은 dynamic
  range가 좁아 분포가 겹친다(근거있음 최솟값 0.686 < 주제공백 최댓값 0.693).
  LLM에게 관련성을 묻는 쪽(`evidence_select.py`)이 실제로 동작했다
- **포기할 줄 아는 게 "RAG 느낌"이다.** 검색은 항상 top_k를 돌려주므로 근거 선별이
  없으면 코퍼스에 없는 주제에도 "가장 덜 무관한" 5건으로 답을 만든다.
  out-of-scope 통과율이 0/4 → 4/4로 바뀐 게 체감 차이의 대부분이다
- **평가셋 없이 코퍼스를 늘리지 말 것.** 무엇이 좋아졌는지 말할 수 없다

## 이 PC의 함정 (2026-08-12 확인)

- **torch는 PyPI 기본 설치가 CPU 빌드다.** 처음에 이걸 모르고 적재를 5시간 돌렸다.
  실측으로 **GPU가 11배 빠르다** (11,281청크: CPU 163분 vs GPU 15분).
  `uv sync`가 torch를 다시 깔면 CPU 빌드로 되돌아가므로 그때마다 다시 설치할 것:
  `uv pip install --index-url https://download.pytorch.org/whl/cu126 --upgrade torch`
- **VRAM 6GB를 임베딩과 LLM이 나눠 쓴다.** 7B Q4(4.7GB) + bge-m3(2.3GB)는 안 들어간다.
  LM Studio를 켠 채 적재하려면 `EMBEDDING_DEVICE=cpu`, 적재만 할 거면 LM Studio를
  내리고 `auto`
- **uv도 Docker도 없었고 관리자 권한이 없다.** uv는 공식 스크립트로 설치했고,
  Docker 대신 `--extra pgdev`(pgserver 내장 Postgres)를 쓴다
- **HF Inference API는 무료로 못 쓴다** — 월 $0.10, `hf-inference`는 CPU 소형 모델만.
  Gemini 무료 티어(OpenAI 호환 엔드포인트)를 쓴다. 로컬이 필요하면 LM Studio로
  `LLM_BASE_URL`만 바꾸면 되고 코드 변경은 없다
- **PowerShell 5.1의 `Invoke-RestMethod`로 API를 확인하지 말 것.** charset 없는
  `application/json`을 ISO-8859-1로 디코딩해서 한글이 깨진 것처럼 보인다.
  서버는 정상 UTF-8이다. `curl`이나 python httpx로 확인할 것
- **API 키는 `.env`에만.** `.env.example`은 `.gitignore` 예외라 **커밋된다**

## 새 머신에서 시작할 때

수집 데이터(`data/raw/`, `data/processed/`), `.env`, DB는 저장소에 없다.
재수집·재적재가 정상 경로다. 전체 절차는 README의 "전체 파이프라인"에 있고,
수집만 확인하려면:

```bash
uv sync --extra collect
cp .env.example .env
uv run python -m scripts.collect.fetch --all
uv run python -m scripts.collect.normalize
uv run python -m scripts.collect.report   # 네 축 + 커버리지 8개 PASS면 정상
```

**`uv sync`를 extra 없이 다시 돌리면 이전 extra가 제거된다.** 검색까지 쓰려면
항상 `uv sync --extra hf --extra collect`처럼 전부 나열할 것. 이걸로 두 번 헤맸다.

Docker나 관리자 권한이 없는 머신(학원 PC 등)은 `uv sync --extra pgdev` 후
`scripts.db.serve`가 내장 pgvector를 띄운다. 포트가 매번 바뀌어서 스크립트가
`.env`의 `DATABASE_URL`을 직접 갱신한다.

## 팀 규칙 (RAG·비서 파트 공통)

- uv / FastAPI / 모델은 자유 / **화면은 Next.js** / **DB는 각자 사용** + ERD 작성
- DB를 공용으로 쓰지 않는 이유는 서브 PC 적재 시 병목이다. 그래서 기기를 옮기면
  `scripts.db.init` + `load_corpus`를 다시 돌려야 한다
- 화면은 `web/`(Next.js)가 표준이다. `demo/streamlit_app.py`는 디버그용으로만 남겼다
- ERD는 `docs/erd.md` — **`app/db/models.py`를 고치면 같이 고칠 것**

## 기기를 옮겨 다닐 때

이 프로젝트는 학원 PC·집 PC·휴대폰을 오가며 작업한다. **로컬 세션은 다른 기기로
옮길 수 없다** — `claude --resume`은 그 머신의 로컬 기록만 본다. 반면 클라우드
세션은 어디서든 당겨올 수 있다.

- 시작: `claude --cloud "<할 일>"` — 클라우드 VM이 **GitHub 원격을 클론**한다
- 이어받기: `claude --teleport` (대화까지 그대로 터미널로)
- 폰에서 확인·지시: claude.ai/code 또는 Claude 모바일 앱

전제: 클라우드 세션은 커밋·푸시된 상태만 본다. 로컬에만 있는 변경은 보이지 않는다.
**작업을 마치면 반드시 커밋하고 푸시할 것.**

로컬 세션을 폰에서 이어 보려면 그 세션에서 `/remote-control`을 켜면 된다.

## 데일리 로그 (취몽 MCP)

세션을 마치거나 기능 하나를 완료하면 `submit_daily_log`로 그날 작업을 등록한다.
MCP 서버는 `chwijung`으로 등록돼 있다 (`claude mcp list`로 확인).

**올리기 전에 반드시 요약안을 보여주고 확인을 받는다.** 외부 서비스로 전송되고
포트폴리오·면접 자료로 쓰이므로 사용자가 내용을 보지 않은 채 나가면 안 된다.

무엇을 쓰는가 — 도구 설명이 요구하는 것이 **구현 디테일이 아니라 판단의 기록**이다:

- ✗ "`rag_service.py`의 `SYSTEM_PROMPT`를 수정하고 `plain_text.py`를 추가했다"
- ○ "답변이 논문 요약처럼 나와 실사용이 어려웠다. 원인을 프롬프트로 보고 화법과
  분량 규칙을 다시 정의했더니 1,200자 → 300자로 줄고 실행 가능한 조언이 됐다"

`problems` / `solution` / `result`는 필수이고 각 200자 제한이며, **잘리지 않고
그대로 포트폴리오에 실린다.** result는 가능하면 수치로 쓴다 — 이 프로젝트는
평가셋(`scripts.eval.retrieval_report`)이 있어서 "10/20 → 19/20" 같은 숫자를
바로 쓸 수 있다.

`content`(6000자)는 팀원이 보는 웹 화면용이고 마크다운이 HTML로 변환된다.
작은 작업이면 비워도 된다.

추측·과장 금지. 실제로 한 일만 쓴다. 안 끝난 것은 `status: in_progress`로 둔다.

## 데이터 수집 판단 규칙

이 프로젝트에서 소스 선정은 답변 품질을 직접 좌우한다. 유튜브·블로그를 배제하는
이유는 ToS 문제뿐 아니라 **지배이론 같은 폐기된 훈련법이 섞이기 때문**이다.
기관·학술 자료만 쓴다.

- **PMC OA 서브셋이 코퍼스 확대의 주력이다** (`fetcher: pmc`). 라이선스가 논문
  메타데이터(JATS `<permissions><license>`)로 같이 와서 읽을 ToS가 없고, 대부분
  CC-BY라 배포도 된다. 이걸로 11건 → 274건, `open` 267건이 됐다
- **PMC 질의는 반드시 `[tiab]`로 좁힐 것.** `canine separation anxiety`를 그냥 넣으면
  3,815건이 잡히고 상위 5건 중 4건이 무관 논문(치매 알림 시스템 등)이다. 같은 주제를
  `[tiab]`로 좁히면 22건이고 전부 관련 문헌이다. fetcher가 dog/canine 언급 3회 미만
  문서를 버리는 2차 방어도 있지만, 질의를 고치는 게 1차다
- **robots.txt 허용이 ToS 허용을 뜻하지 않는다.** 둘 다 확인해야 한다.
  ASPCA·MSD는 robots.txt가 전부 허용인데 ToS가 자동 수집을 금지한 실제 사례다
- **반대 방향도 있다** — robots.txt가 용도를 콕 집어 금지하는 경우. 네이버 블로그는
  `ClaudeBot`/`GPTBot`을 이름으로 막으면서 "BOT ACCESS FOR THE PURPOSES OF AI
  TRAINING AND RAG IS STRICTLY PROHIBITED"라고 적어뒀다. 이 프로젝트가 정확히
  RAG라서 수집하지 않는다. 판정 근거는 `sources.yaml` 맨 아래에 있다
- 새 소스는 `license: pending-check`로 넣는다 — `fetch`가 거부한다.
  `uv run python -m scripts.collect.check_license`로 확인 후 값을 갱신한다
- **판정 근거는 `data/sources.yaml`에 ToS 원문을 인용해 주석으로 남긴다.**
  같은 조사를 반복하지 않기 위해서다. 기존 항목들이 그 형식의 예다
- **배포 가능 여부는 `distribution` 한 필드로 본다** (`open` | `personal-only`).
  license 문자열을 매번 매칭하지 않으려고 적재 시점에 정규화한다 —
  `scripts/db/load_corpus.py:derive_distribution`. 모르는 값은 보수적으로
  `personal-only`다. 현재 274건 중 `open` 267 / `personal-only` 7
- **답변 근거 코퍼스와 관찰용 코퍼스는 물리적으로 분리한다** (`corpus` 필드).
  블로그처럼 지배이론이 섞일 수 있는 자료는 `observation`으로 넣고, 검색이
  `corpus = 'answer'`를 하드코딩으로 걸어 답변 근거에서 격리한다. 빈도 기반으로
  "훈련사들의 공통점"을 뽑으면 알파독·서열잡기가 상위로 오는데, 이는
  `avsab-dominance`가 정면으로 반박하는 내용이라 같은 풀에 있으면 답변이
  자기모순에 빠진다. `normalize`가 `corpus_blogs.jsonl`로 따로 뽑는다

## 코드 규칙

- **콘솔 출력에 `✓`·`⚠️` 같은 기호를 쓸 수 있는 건** `scripts/collect/__init__.py`가
  import 시점에 stdout/stderr을 UTF-8로 재설정하기 때문이다. 한글 Windows의 기본
  콘솔(cp949)에는 이 문자들이 없어 그것 없이는 `UnicodeEncodeError`로 죽는다.
  수집 패키지 밖에 새 CLI를 만든다면 같은 처리가 필요하다
- LLM·임베딩 구현체는 Protocol 뒤에 있고 **교체 지점은 각 `registry.py` 한 곳뿐이다.**
  `rag_service.py`와 엔드포인트는 건드리지 않는다 (절차는 README의 "Provider 교체")
- 모든 환경변수는 `app/core/config.py`에서만 읽는다
- **임베딩 모델은 lifespan에서 1회 로딩한다** (`app.state.embedder`). 요청마다 만들면
  수 GB 모델을 매번 읽는다. 로딩 실패는 앱을 죽이지 않고 검색 경로만 503으로 만든다 —
  torch 없이도 `/health`·`/docs`는 떠야 한다
- **`EMBEDDING_DIM`은 `chunks.embedding`의 `vector(N)`과 같아야 한다.** 모델을 바꾸면
  설정과 함께 `scripts.db.init --drop` + 재적재. 불일치하면 warmup이 즉시 죽는다
  (pgvector INSERT까지 가서 불친절하게 터지는 것보다 낫다)
- **검색의 `aversive` 제외와 `corpus='answer'` 필터는 하드코딩이다.** 코퍼스 품질
  불변식이지 호출자가 끌 수 있는 옵션이 아니다
- **권위 부스팅을 SQL `ORDER BY`에 넣지 말 것.** 조인 컬럼이 낀 표현식으로 정렬하면
  HNSW 인덱스를 못 써서 코퍼스가 커질 때 조용히 full scan이 된다. 과다 조회 후
  파이썬에서 재랭킹한다 (`vectorstore/ranking.py`)
- **대량 적재는 API가 아니라 `scripts.db.load_corpus`로 한다.** API 프로세스에 torch를
  상주시키지 않기 위해서다. 다만 같은 `IngestService`를 부르므로 구현은 하나다
- **LLM 판정에 `not_covered`(또는 동등한 "모름") 선택지를 반드시 넣을 것.** 빼면
  모델이 근거 없이 단정한다. 그리고 인용이 비었으면 **코드에서** 강등한다 —
  프롬프트로 부탁하지 않는다
- **여러 주장을 검증할 때 LLM 구간만 `asyncio.gather`로 묶는다.** `AsyncSession`
  하나는 커넥션 하나라 DB를 병렬로 쓰면 깨진다. 검색은 수십 ms라 순차로 둬도 된다
- 주석과 커밋 메시지는 한국어로 쓴다. 무엇을 했는지보다 **왜 그렇게 했는지**를 남긴다
- 무거운 ML 패키지는 기본 설치에서 빠져 있다 (`uv sync --extra hf|demo|collect`)
