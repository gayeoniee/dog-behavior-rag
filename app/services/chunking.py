"""문서를 검색 단위(청크)로 쪼갠다.

순수 동기 함수만 있고 Settings·torch·DB를 import하지 않는다. 호출자가 `ChunkConfig`를
만들어 넘기는 구조라 픽스처 없이 단위 테스트된다 — 청킹 품질은 검색 품질을 직접
좌우하는데, 무거운 의존성이 붙으면 아무도 테스트를 안 돌리게 된다.

`scripts/collect/normalize.py`가 이미 NFC 정규화·공백 정리·`\\n{3,}` 축약을 했다.
여기서는 그 뒤에도 실제 코퍼스에 남아 있는 것들만 처리한다 (clean_for_chunking 참조).
"""

import re
from collections import Counter
from dataclasses import dataclass

__all__ = [
    "ChunkConfig",
    "clean_for_chunking",
    "looks_like_paper_boilerplate",
    "looks_like_reference_list",
    "split_text",
]


_SEPARATORS = ("\n\n", "\n", ". ", " ")
"""재귀 분할 우선순위. 문자 기준이지 단어 기준이 아니다 — 한국어 문서가 들어올
예정이고 한국어는 공백 토큰화가 의미 단위와 맞지 않는다.

**줄바꿈이 문장부호보다 먼저다.** PDF·HTML 추출물은 문장 중간에서 줄이 바뀌는
일이 흔해서, 이 순서면 문장을 자르게 된다. 순서를 바꾼 대안이 아래에 있고,
어느 쪽이 나은지는 `ChunkConfig.separators`로 갈아끼워 재면 된다.
"""

_SENTENCE_FIRST = ("\n\n", ". ", ".\n", "다. ", "요. ", "\n", " ")
"""문장 경계를 줄바꿈보다 먼저 보는 순서.

`". "`만으로는 부족하다 — PDF는 마침표 뒤에 바로 줄을 바꾸고(`.\\n`), 한국어는
`"다."`·`"요."`로 끝나는데 그 뒤가 줄바꿈인 경우가 많다. 세 형태를 다 넣는다.

⚠️ 이건 **후보**지 기본값이 아니다. 04장에서 배운 대로 청킹 변경은 지표로
확인하기 전에는 채택하지 않는다.
"""


@dataclass(slots=True, frozen=True)
class ChunkConfig:
    size: int = 1200
    overlap: int = 150
    min_size: int = 200
    separators: tuple[str, ...] = _SEPARATORS
    """재귀 분할 구분자 우선순위. 실험에서 갈아끼운다 (`_SENTENCE_FIRST` 참조)."""

_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")
_PAGE_NUM = re.compile(r"\d{1,4}")
_HEADER_MAX_LEN = 80
_HEADER_MIN_REPEATS = 3

_REF_LINE = re.compile(r"\s*(\[?\d{1,3}[\].)]|\d{1,3}\.)\s")
_CITATION = re.compile(r"et al\.|doi\.org|https?://|\b(19|20)\d{2};\s?\d")


def clean_for_chunking(text: str) -> str:
    """PDF 추출물에 남은 노이즈를 걷어낸다.

    세 가지만 처리한다:

    1. 단어 중간 하이픈 줄바꿈 — pypdf가 `counter-\\nconditioning`을 그대로 뱉는다.
       붙이지 않으면 임베딩이 이걸 무관한 두 토막으로 본다.
    2. 러닝 헤더/푸터 — 문서 전체에서 3회 이상 반복되는 짧은 줄.
    3. 페이지 번호만 있는 줄.

    **일부러 하지 않는 것:** 하드 줄바꿈을 문단으로 합치기. AAHA PDF에는 도움이 되지만
    HTML 문서 7건은 `node.text(separator="\\n")`로 뽑혀서 줄 하나가 리스트 항목
    하나다 — 합치면 무관한 항목들이 한 문장으로 융합된다. 임베딩은 여분의 줄바꿈에
    관대하지만 융합된 항목은 실제 품질 손실이라, 위험이 비대칭이므로 두고 간다.
    """
    text = _HYPHEN_WRAP.sub(r"\1\2", text)

    lines = text.split("\n")
    repeats = Counter(stripped for line in lines if (stripped := line.strip()))
    kept = [
        line
        for line in lines
        if not _PAGE_NUM.fullmatch(stripped := line.strip())
        and not (len(stripped) < _HEADER_MAX_LEN and repeats[stripped] >= _HEADER_MIN_REPEATS)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def looks_like_reference_list(chunk: str) -> bool:
    """참고문헌 블록인가.

    AAHA 가이드라인 하나가 코퍼스 글자 수의 절반인데 끝부분이 긴 번호 매긴 참고문헌
    목록이다. 이 청크들은 "veterinarian", "behavior", "journal" 같은 표면 어휘로
    의학 질문에 걸려들기만 하고 답에는 아무 도움이 안 된다.

    조건이 **두 개인 것이 핵심이다.** 번호 목록이라는 것만으로 판정하면
    "1. 앉아 2. 기다려 3. 보상" 같은 훈련 절차가 정확히 같은 모양이라 함께 날아간다.
    훈련 절차에는 DOI가 세 개씩 박혀 있지 않다.
    """
    lines = [line for line in chunk.splitlines() if line.strip()]
    if len(lines) < 5:
        return False
    numbered = sum(1 for line in lines if _REF_LINE.match(line))
    return numbered >= len(lines) * 0.6 and len(_CITATION.findall(chunk)) >= 3


_STATS = re.compile(
    # 단어 경계(\b)를 빼면 안 된다. `CI`가 "specific"·"social" 안에 걸려
    # 영어 본문 대부분이 통계로 잡힌다. 실제로 한 번 그렇게 만들어봤다.
    r"\bp\s*[=<>]\s*0?\.\d|\bCI\b|\bSD\s*=|\bOR\s*=|\bn\s*=\s*\d"
    r"|\bR2\b|\bp-value|\bsignifican(t|ce)\b|\bmean\s+of\b",
    re.I,
)
_METHODS = re.compile(
    r"\b(participants were|dogs were (recruited|enrolled|assigned)"
    r"|data (were|was) (collected|analy[sz]ed)|questionnaire was"
    r"|ethical approval|informed consent|inclusion criteria"
    r"|statistical analys|were performed using|IBM SPSS|R version)\b",
    re.I,
)
_FRONTMATTER = re.compile(
    r"\b(Conflict of Interest|Author Contributions|Funding|Acknowledg(e)?ments?"
    r"|Data Availability|Supplementary Material|Institutional Review Board"
    r"|Informed Consent Statement|Publisher's Note|doi\.org/10\.)",
    re.I,
)


def looks_like_paper_boilerplate(chunk: str) -> bool:
    """논문의 **형식 잡음**인가 — 통계 보고·연구 방법·저자/기금 문구.

    답변 근거가 될 수 없는데 검색에는 걸린다. "significant", "dogs were
    recruited" 같은 표현이 개 행동 질문의 어휘와 겹치기 때문이다. 실제로
    줄당김 질문에서 논문의 방법론 조각이 RSPCA 실무 가이드를 밀어냈다.

    **논문에만 적용한다** (`IngestService`가 source_id로 가른다). 규칙이
    논문 문체를 겨냥해 만들어졌고, 실측상 기관 가이드 417청크 중 1건만
    잘못 걸렸지만 그 1건도 안 잃는 편이 낫다.

    통계는 **표현 3개 이상**일 때만 잡는다. 본문에도 p값이 한 번쯤 나오는데
    그걸로 버리면 결과를 설명하는 멀쩡한 문단이 날아간다.

    실측 (2026-08-18, PMC 10,958청크): 통계 9.0% · 방법 5.3% · 저자·기금 2.4%
    = 16.8%가 걸러진다. 남는 본문의 73%는 보호자에게 쓸모 있다고 판정됐다.
    """
    return bool(
        _FRONTMATTER.search(chunk)
        or len(_STATS.findall(chunk)) >= 3
        or _METHODS.search(chunk)
    )


def _split_to_units(text: str, size: int, separators: tuple[str, ...] = _SEPARATORS) -> list[str]:
    """모든 조각이 `size` 이하가 될 때까지 구분자를 낮춰가며 재귀 분할한다.

    **자른 자리의 구분자를 앞 조각에 되붙인다.** `str.split`은 구분자를 먹어치우는데,
    `". "`로 자르면 **마침표까지 같이 사라진다:**

        'First one. Second one.'.split('. ')  →  ['First one', 'Second one.']
                                                             ↑ 마침표 없음

    줄바꿈이 사라지는 건 무해하지만 문장부호가 사라지는 건 손실이다. 임베딩 입력이
    훼손되고, "이 청크가 문장 끝에서 끝났나"를 재는 지표까지 거짓말을 한다
    (실제로 구분자 순서 실험이 이것 때문에 무효였다 — 멀쩡히 잘린 문장이
    소문자로 끝나서 '중간에서 끊김'으로 집계됐다).

    되붙인 뒤 `rstrip`하므로 공백류 구분자는 그대로 사라지고 문장부호만 남는다.
    """
    if len(text) <= size:
        return [text] if text else []

    if not separators:
        # 구분자를 다 써도 안 쪼개지는 덩어리 (구분 기호 없는 긴 문자열) — 하드 슬라이스.
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, *rest = separators
    parts = text.split(sep)
    units: list[str] = []
    for index, piece in enumerate(parts):
        # 마지막 조각 뒤에는 원래 구분자가 없었다.
        restored = (piece + sep).rstrip() if index < len(parts) - 1 else piece
        if not restored:
            continue
        if len(restored) <= size:
            units.append(restored)
        else:
            units.extend(_split_to_units(restored, size, tuple(rest)))
    return units


def _overlap_tail(chunk: str, overlap: int) -> str:
    """다음 청크의 앞에 붙일 꼬리. 단어 중간에서 시작하지 않도록 공백 경계로 스냅한다."""
    if overlap <= 0 or len(chunk) <= overlap:
        return chunk if overlap > 0 else ""
    tail = chunk[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def split_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    """문서 본문을 청크 리스트로 자른다. 빈 입력이면 빈 리스트."""
    cfg = config or ChunkConfig()
    cleaned = clean_for_chunking(text)
    if not cleaned:
        return []

    units = _split_to_units(cleaned, cfg.size, cfg.separators)
    if not units:
        return []

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) <= cfg.size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = _overlap_tail(current, cfg.overlap)
            seeded = f"{tail}\n{unit}" if tail else unit
            # unit 자체가 size에 가까우면 overlap을 붙이는 순간 상한을 넘는다.
            # 그럴 땐 이번 청크만 overlap을 포기한다 — size 상한이 우선이다.
            current = seeded if len(seeded) <= cfg.size else unit
        else:
            current = unit
    if current:
        chunks.append(current)

    # 꼬리가 너무 짧으면 앞 청크에 병합한다. 버리면 문서의 결론 문단이 조용히 사라진다.
    # 들어갈 자리가 없으면 짧은 채로 남긴다 — size 상한(프롬프트 예산)은 절대 안 넘긴다.
    # 청크가 하나뿐이면 짧아도 그대로 둔다 (짧은 문서 자체를 버릴 수는 없다).
    if len(chunks) > 1 and len(chunks[-1]) < cfg.min_size:
        merged = f"{chunks[-2]}\n{chunks[-1]}"
        if len(merged) <= cfg.size:
            chunks[-2:] = [merged]

    return [c for c in chunks if not looks_like_reference_list(c)]
