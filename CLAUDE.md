# CLAUDE.md

반려동물 훈련·문제행동 상담 RAG 시스템. 최종 목표는 안드로이드 앱이고, 지금은 API와
데이터 수집 파이프라인만 동작하는 단계다.

명령어·구조·수집 절차는 `README.md`에 있다. 이 파일은 **README에 없는 판단 규칙과
현재 위치**만 담는다.

## 현재 위치

- ✅ 데이터 수집 파이프라인 동작. 문서 11건, 네 축(problem/cause/training/medical)
  전부, 커버리지 질문 8개 전부 PASS
- ✅ 소스 라이선스 확정 (`pending-check` 없음)
- ❌ **임베딩·검색·LLM은 전부 스텁이다.** `/api/v1/chat`은 `"[stub] ..."`를 돌려준다
- 다음 작업: 임베딩 모델 확정 → 차원 확정 → 청킹 → pgvector 코사인 검색.
  전체 목록은 README의 "다음 작업 (TODO)"

## 새 머신에서 시작할 때

수집 데이터(`data/raw/`, `data/processed/`)와 `.env`는 저장소에 없다. 재수집이
정상 경로다 — 2~3분 걸린다.

```bash
uv sync --extra collect
cp .env.example .env
uv run python -m scripts.collect.fetch --all
uv run python -m scripts.collect.normalize
uv run python -m scripts.collect.report   # 네 축 + 커버리지 8개 PASS면 정상
```

`report`가 오늘과 같은 결과를 내면 환경이 제대로 선 것이다.

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

## 데이터 수집 판단 규칙

이 프로젝트에서 소스 선정은 답변 품질을 직접 좌우한다. 유튜브·블로그를 배제하는
이유는 ToS 문제뿐 아니라 **지배이론 같은 폐기된 훈련법이 섞이기 때문**이다.
기관·학술 자료만 쓴다.

- **robots.txt 허용이 ToS 허용을 뜻하지 않는다.** 둘 다 확인해야 한다.
  ASPCA·MSD는 robots.txt가 전부 허용인데 ToS가 자동 수집을 금지한 실제 사례다
- 새 소스는 `license: pending-check`로 넣는다 — `fetch`가 거부한다.
  `uv run python -m scripts.collect.check_license`로 확인 후 값을 갱신한다
- **판정 근거는 `data/sources.yaml`에 ToS 원문을 인용해 주석으로 남긴다.**
  같은 조사를 반복하지 않기 위해서다. 기존 항목들이 그 형식의 예다
- `license: personal-use-only` / `personal-use-only-manual-copy` = **앱 배포 시
  코퍼스에서 제외 대상.** 현재 11건 중 7건이 여기 해당한다.
  `corpus.jsonl`의 `license` 필드로 걸러낼 수 있다
- 배포용 코퍼스를 채울 때는 CC-BY 오픈액세스 문헌을 우선한다. PMC OA 서브셋은
  라이선스가 메타데이터로 오므로 ToS를 읽을 필요가 없다. 후보는 `sources.yaml`의
  MSD 제거 주석에 적어뒀다

## 코드 규칙

- **콘솔 출력에 `✓`·`⚠️` 같은 기호를 쓸 수 있는 건** `scripts/collect/__init__.py`가
  import 시점에 stdout/stderr을 UTF-8로 재설정하기 때문이다. 한글 Windows의 기본
  콘솔(cp949)에는 이 문자들이 없어 그것 없이는 `UnicodeEncodeError`로 죽는다.
  수집 패키지 밖에 새 CLI를 만든다면 같은 처리가 필요하다
- LLM·임베딩 구현체는 Protocol 뒤에 있고 **교체 지점은 각 `registry.py` 한 곳뿐이다.**
  `rag_service.py`와 엔드포인트는 건드리지 않는다 (절차는 README의 "Provider 교체")
- 모든 환경변수는 `app/core/config.py`에서만 읽는다
- 주석과 커밋 메시지는 한국어로 쓴다. 무엇을 했는지보다 **왜 그렇게 했는지**를 남긴다
- 무거운 ML 패키지는 기본 설치에서 빠져 있다 (`uv sync --extra hf|demo|collect`)
