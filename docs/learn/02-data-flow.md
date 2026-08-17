# 02. 데이터는 어디서 와서 어디에 있나

← [01. RAG가 무엇을 하는가](01-what-is-rag.md) · [체크리스트로](00-checklist.md)

이 장의 목표는 하나다. **"내 데이터가 지금 어떤 모양으로 어디에 있는지"를
말할 수 있게 되는 것.** 그게 안 되면 03장 이후의 실험을 할 수 없다.

---

## 1. 큰 그림 — 5단계

```
  ①수집        ②정규화        ③청킹         ④임베딩       ⑤적재
인터넷  →  data/raw/  →  data/processed/  →  조각들  →  숫자  →  Postgres
           *.json         corpus.jsonl                            (2개 테이블)

scripts/collect/  scripts/collect/  app/services/  app/services/  scripts/db/
   fetch.py         normalize.py     chunking.py   embeddings/    load_corpus.py
```

명령으로 보면 이렇다 (README의 "전체 파이프라인"과 같다):

```bash
uv run python -m scripts.collect.fetch --all      # ① 인터넷 → data/raw/
uv run python -m scripts.collect.normalize        # ② → data/processed/corpus.jsonl
uv run python -m scripts.db.load_corpus           # ③④⑤ 청킹+임베딩+DB적재
```

**③④⑤가 한 명령에 묶여 있는 게 포인트다.** 청킹 결과는 파일로 저장되지 않는다.
`load_corpus`가 메모리에서 자르고, 바로 임베딩하고, 바로 DB에 넣는다.

---

## 2. 문서 한 건을 끝까지 따라가기

`aaha-behavior-2015` (미국동물병원협회 행동관리 가이드라인 PDF) 한 건이
어떻게 변해가는지 실제 숫자로 보자.

### ① 수집 — 인터넷에서 원본 받기

**무엇이 정하나:** `data/sources.yaml`에 "어디서 무엇을 어떤 방법으로 가져올지"가
적혀 있다. 설정 파일 하나로 끝난다.

```yaml
- id: aaha-behavior-2015
  name: AAHA Canine and Feline Behavior Management Guidelines (2015)
  urls:
    - https://www.aaha.org/.../2015_aaha_..._guidelines_final.pdf
  fetcher: pdf              # ← 이 값으로 어떤 수집기를 쓸지 결정
  language: en
  species: both
  axis: [medical, problem, training]
  authority_tier: 1
  methodology: reward_based
  published_at: 2015
  license: public-guideline-pdf
```

**수집기(fetcher)는 4종류다** (`scripts/collect/fetchers/`):

| fetcher | 대상 | 텍스트 추출 방법 | 소스 수 |
|---|---|---|---|
| `pmc` | PubMed Central 논문 | NCBI API → JATS XML → `<abstract>` + `<body>` | 8 |
| `pdf` | PDF 파일 | `pypdf`로 페이지별 추출 후 이어붙임 | 5 |
| `html` | 웹페이지 | `selectolax`로 `<script>·<nav>·<footer>` 제거 후 본문만 | 2 |
| `local` | 손으로 받아둔 파일 | 로그인·동적 페이지라 자동 수집이 안 되는 것들 | 2 |

> 💡 **`pmc`가 코퍼스 확대의 주력이다.** 논문 263건이 전부 여기서 왔다.
> 기관 웹사이트는 건건이 이용약관을 읽어야 하는데, PMC는 라이선스가 논문
> 메타데이터로 같이 오기 때문이다.
>
> ⚠️ 지금은 **PDF는 `pypdf`, HTML은 `selectolax`**로 텍스트를 뽑는다. 표·제목 구조가
> 다 날아가고 한 덩어리 텍스트가 된다. 멘토가 말한 **Docling**(문서를 마크다운으로
> 변환)은 이 부분을 개선하는 도구다 — 04장에서 다룬다.

**결과:** `data/raw/aaha-behavior-2015.json`

```json
[{
  "source_id": "aaha-behavior-2015",
  "url": "https://www.aaha.org/...",
  "title": "AAHA Canine and Feline Behavior Management Guidelines (2015)",
  "text": "VETERINARY PRACTICE GUIDELINES 2015 AAHA Canine and Feline...",
  "fetched_at": "2026-08-12T00:44:30..."
}]
```

**본문 81,704자.** 필드가 5개뿐인 게 포인트다 — 이 단계는 "가져오기"만 한다.

### ② 정규화 — 메타데이터 합치기

```bash
uv run python -m scripts.collect.normalize
```

`data/raw/*.json`(본문) + `sources.yaml`(메타데이터)를 합쳐서
**문서 1건 = JSONL 1줄**로 만든다.

**결과:** `data/processed/corpus.jsonl` 한 줄 — 필드가 5개에서 **16개로** 늘어난다.

| 필드 | 값 (AAHA) | 어디에 쓰이나 |
|---|---|---|
| `id` | `aaha-behavior-2015:c98efee0...` | 문서 식별 |
| `title` | AAHA Canine and Feline… | 답변의 출처 표시 |
| `content` | (본문 81,704자) | **청킹의 입력** |
| `content_hash` | `c98efee0a8929949` | **재적재 시 중복 방지 키** |
| `source_url` | https://www.aaha.org/… | 답변에 붙는 인용 링크 |
| `language` | `en` | (지금은 전부 en) |
| `species` | `both` | 개/고양이 구분 |
| `axis` | `[medical, problem, training]` | 커버리지 점검용 4축 |
| `methodology` | `reward_based` | **검색에서 `aversive` 제외 필터** |
| `authority_tier` | `1` | 재랭킹 가산점 |
| `published_at` | `2015` | 최신성 판단 |
| `volatility` | `stable` | 재수집 주기 판단 |
| `license` | `public-guideline-pdf` | 배포 가능 여부의 근거 |
| `corpus` | `answer` | **검색이 `answer`만 본다** |

**DB에는 두 필드가 더 생긴다.** JSONL에 없고 적재 시점에 계산된다
(`scripts/db/load_corpus.py`):

| 필드 | 계산 방법 | 쓰임 |
|---|---|---|
| `distribution` | `license` 문자열 → `open` \| `personal-only` | 배포 가능 여부. 매번 문자열 매칭하지 않으려고 한 필드로 정규화 |
| `doc_type` | `source_id`가 `pmc-`로 시작하면 `study`, 아니면 `guide` | 재랭킹 가산점 |

**정규화가 본문에 하는 일**은 생각보다 가볍다 (`clean_text`):
NFC 유니코드 정규화, `\r\n` → `\n`, 연속 공백 축약, 빈 줄 3개 이상 → 2개.

실제로 294건 중 **163건만 길이가 변했고, 줄어든 양의 중앙값은 3자**다
(가장 많이 줄어든 문서도 568자). AAHA는 81,704자 → 81,704자로 **변화 없음**.

> **본격적인 청소는 다음 단계(청킹)에서 한다.** 헷갈리기 쉬운 지점이라 짚고 간다.

### ③ 청킹 — 검색 단위로 자르기

**왜 자르나:** 이 문서는 81,704자다. LLM 프롬프트에 통째로 넣을 수 없고, 넣더라도
답에 필요한 건 그중 한 문단이다. **찾을 수 있는 크기로 잘라야 한다.**

`app/services/chunking.py`의 `split_text()`가 한다. 두 부분으로 나뉜다:

**(1) `clean_for_chunking()` — PDF 찌꺼기 청소**

| 처리 | 왜 |
|---|---|
| 하이픈 줄바꿈 붙이기 | pypdf가 `counter-\nconditioning`을 그대로 뱉는다. 안 붙이면 임베딩이 무관한 두 토막으로 본다 |
| 반복되는 짧은 줄 제거 | 페이지마다 반복되는 머리글·꼬리말 (3회 이상 등장 + 80자 미만) |
| 숫자만 있는 줄 제거 | 페이지 번호 |

**AAHA 실측: 81,704자 → 80,841자 (863자 제거).**

**(2) 재귀 분할 (recursive chunking)**

목표 크기는 **1,200자**, 겹침 **150자**. 구분자를 순서대로 낮춰가며 자른다:

```
"\n\n" (문단) → "\n" (줄) → ". " (문장) → " " (단어)
```

문단 경계로 자를 수 있으면 그렇게 하고, 그래도 1,200자를 넘으면 줄로, 그래도 넘으면
문장으로… 이런 식이다. **의미 단위를 최대한 안 깨려는 것**이다.

**겹침(overlap) 150자가 왜 있나:** 자른 자리에 걸친 문장이 어느 쪽에서도 온전하지
않으면 검색이 놓친다. 그래서 앞 청크의 끝 150자를 다음 청크 앞에 붙인다.

**직접 확인:**

```bash
uv run python -m scripts.db.load_corpus --dry-run   # 청킹만. torch 불필요
```

**AAHA 실측: 80,841자 → 청크 80개.** 앞쪽 청크 길이는 1,128 · 1,186 · 1,189자.

### ④ 임베딩 — 숫자로 바꾸기

청크 80개를 각각 `bge-m3` 모델에 넣어 **숫자 1,024개**로 바꾼다.

```
청크 0번 (1,128자)  →  [-0.0564, -0.0468, -0.0595, 0.0077, 0.0013, ... ]
                        └───────────── 1,024개 ─────────────┘
```

이 숫자 묶음이 "이 청크의 뜻"이다. 자세한 건 [01장 2절](01-what-is-rag.md).

> 💡 이 단계가 파이프라인에서 **가장 느리다.** 11,354청크 기준 GPU 15분 / CPU 163분.
> 11배 차이다. 그래서 임베딩 모델을 바꾸면 전부 다시 계산해야 하고,
> 08장(모델 선정)에서 이 비용을 계산에 넣어야 한다.

### ⑤ 적재 — Postgres에 넣기

`documents` 한 줄 + `chunks` 80줄이 **한 트랜잭션으로** 들어간다.

**AAHA 실측:** `documents.id = 1`, 청크 80개.

---

## 3. 왜 `raw`와 `processed`를 나눴나

| | 담긴 것 | 다시 만드는 비용 |
|---|---|---|
| `data/raw/` | 인터넷에서 받은 **원본 그대로** | **비싸다** — 네트워크 왕복, 상대 서버 부담, 시간 |
| `data/processed/` | 메타데이터 합치고 정리한 것 | **싸다** — 로컬 파일 변환, 몇 초 |

정규화 규칙을 고치고 싶을 때 **다시 수집하지 않아도 된다.** 원본이 있으니
`normalize`만 다시 돌리면 된다. 남의 서버에 반복 요청하는 건 느리기도 하고 예의도
아니다.

> ⚠️ **둘 다 git에 없다** (`.gitignore`). 저작물 원본이라서다.
> 새 기기에서는 재수집이 정상 경로다.

---

## 4. DB — 왜 테이블이 두 개인가

```
documents (294행)                      chunks (11,354행)
┌────────────────────┐                ┌──────────────────────┐
│ id                 │◄───────────────│ document_id          │
│ title              │      1 : N     │ ordinal   (문서 내 순번)│
│ content  (원문 통째) │                │ content   (조각 본문)  │
│ source, axis, ...  │                │ embedding vector(1024)│
└────────────────────┘                └──────────────────────┘
      원문 보관용                            검색 대상
```

- **`documents`** = 문서 1건. 원문을 통째로 갖고 있고 메타데이터가 여기 붙는다
- **`chunks`** = 잘린 조각. **실제 검색은 여기서만 일어난다**

검색이 청크를 찾으면 → `document_id`로 조인해서 제목·출처·권위를 가져온다.
그래서 답변에 "이 근거는 ASPCA 문서에서 왔습니다"를 붙일 수 있다.

자세한 스키마는 [`docs/erd.md`](../erd.md), 코드는 `app/db/models.py`.

### HNSW 인덱스가 뭔가

청크 11,354개와 질문을 **전부 비교하면** 정확하지만 느리다. 코퍼스가 10만, 100만이
되면 못 쓴다.

**HNSW**는 "대충 가까운 것들"을 아주 빠르게 찾는 자료구조다. 정확도를 조금 포기하고
속도를 크게 얻는다(그래서 ANN = Approximate Nearest Neighbor).

```python
# app/db/models.py
Index("ix_chunks_embedding", "embedding",
      postgresql_using="hnsw",
      postgresql_ops={"embedding": "vector_cosine_ops"})
```

> ⚠️ **`vector_cosine_ops`(코사인)로 만들었으면 검색도 코사인이어야 한다.**
> 한쪽만 바꾸면 에러 없이 **조용히 엉뚱한 순위**가 나온다.

**그리고 이것 때문에 가산점을 SQL에 못 넣는다.** `ORDER BY 유사도 + 가산점`으로
쓰면 인덱스를 못 타고 전체 스캔이 된다. 그래서 SQL은 넉넉히(top_k × 4) 가져오고,
가산점은 파이썬에서 준다 (`vectorstore/ranking.py`).

---

## 5. 지금 내 데이터의 진실

숫자를 직접 확인해보자.

```bash
uv run python -m scripts.db.serve      # DB 기동 (포트가 .env에 자동 갱신됨)
```

```python
# corpus.jsonl 통계
import json
from collections import Counter
docs = [json.loads(l) for l in open('data/processed/corpus.jsonl', encoding='utf-8')]
print(len(docs))
print(Counter(d['source_id'].split('-')[0] for d in docs).most_common())
print(Counter(d['language'] for d in docs))
```

### 실측값 (2026-08-14)

| 항목 | 값 |
|---|---|
| 문서 | 294건 |
| 청크 | 11,354개 |
| 총 글자 | 약 980만 자 |
| 문서 길이 | 중앙값 **32,458자**, 최소 1,236 / 최대 82,083 |
| 청크 길이 | 중앙값 1,061자, 평균 990 (상한 1,200) |
| 언어 | **전부 영어** (294/294) |
| 출처 | PMC 논문 **263** · RSPCA 14 · ASPCA 8 · VCA 5 · AVSAB 3 · AAHA 1 |
| 청크 구성 | 논문 **10,937 (96.3%)** · 실무 가이드 417 (3.7%) |

### 여기서 보이는 세 가지 문제

**(1) 코퍼스가 논문 편중이다 (96%).**
보호자는 "어떻게 해요?"를 묻는데 나오는 건 연구 결과다. `ranking.py`의
`guide_boost`(+0.03)가 이걸 보정하려는 장치인데, **근본 해결은 가이드 문서를
늘리는 것**이다.

**(2) 질문은 한국어, 자료는 100% 영어다.**
그래서 질의 재작성이 필수 경로가 됐다. 그리고 **임베딩 모델 선택이 결정적**이다 —
교차언어 성능이 나쁜 모델을 고르면 아무것도 안 찾아진다. 08장의 주제다.

**(3) 코드에 있는데 지금은 아무 일도 안 하는 장치들이 있다.**

| 장치 | 코드 | 지금 데이터 | 실제 효과 |
|---|---|---|---|
| `authority_boost` | `ranking.py` | 문서 **전부 tier=1** | 모두에게 같은 +0.02 → **순위 변화 0** |
| `methodology != 'aversive'` | `pgvector.py` | 전부 `reward_based` | 거르는 것 **없음** |
| `corpus = 'answer'` | `pgvector.py` | 전부 `answer` | 거르는 것 **없음** |
| 참고문헌 청크 제거 | `chunking.py` | 문서 40건 실측 | 걸러낸 청크 **0개** |
| `guide_boost` | `ranking.py` | 가이드 417 / 논문 10,937 | ✅ **이것만 실제로 동작** |

**버그는 아니다.** 앞으로 블로그(`observation`)나 혐오 기반 자료가 들어올 때를
대비한 장치다. 다만 **"있으니까 동작하겠지"라고 믿으면 안 된다** — 지금은 안 한다.
참고문헌 필터는 04장에서 왜 안 걸리는지 확인해볼 항목이다.

---

## 요점 3줄

1. **데이터는 5단계를 거친다.** 수집(`raw/`) → 정규화(`corpus.jsonl`) →
   청킹 → 임베딩 → DB. 청킹 결과는 **파일로 남지 않고** `load_corpus`가 메모리에서
   처리해 바로 넣는다.
2. **테이블이 둘인 이유는 단위가 다르기 때문이다.** 원문은 `documents`에 통째로,
   검색은 `chunks` 단위로. 검색이 청크를 찾고 조인해서 출처를 가져온다.
3. **내 코퍼스는 논문 96% · 영어 100%다.** 이 두 사실이 지금 시스템의 설계
   대부분(질의 재작성, guide_boost)을 만들었고, 남은 문제도 대부분 여기서 나온다.

## 이해 확인 질문

1. 청킹 규칙을 바꾸고 싶다. 어떤 명령부터 다시 돌려야 하나? `fetch`부터 다시?
2. `documents`와 `chunks`를 왜 한 테이블로 합치지 않았나?
3. `authority_boost` 설정을 0.02에서 0.2로 올리면 지금 내 검색 결과는 어떻게 바뀌나?

<details>
<summary>답 보기</summary>

1. **`load_corpus`만** 다시 돌리면 된다. 청킹은 ③단계이고 `corpus.jsonl`(②)은
   그대로 쓸 수 있다. `fetch`(①)는 인터넷에서 받는 단계라 무관하다.
   (단, 임베딩을 다시 계산하므로 GPU 15분이 든다.)
2. **단위가 다르기 때문이다.** 검색은 조각 단위로 해야 하고(문서 전체는 8만 자라
   프롬프트에 못 넣는다), 출처·권위 같은 메타데이터는 문서 단위로 한 번만 있으면
   된다. 청크마다 복사하면 11,354번 중복된다.
3. **아무것도 안 바뀐다.** 문서 294건이 전부 `tier=1`이라 모든 후보가 똑같은
   가산점을 받고, 상수를 더하면 순위는 그대로다. (`tier` 값이 섞여 있어야 의미가 생긴다.)

</details>
