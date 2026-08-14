"""청킹 실험의 구간(span) 계산 테스트.

이 로직이 틀리면 **실험 전체가 조용히 거짓말을 한다.** 정답 구간을 잘못 잡으면
어떤 청크 크기가 좋은지가 뒤바뀌는데, 결과는 그럴듯한 표로 나오므로 알아챌 수 없다.

실제로 만들면서 두 번 틀렸고 둘 다 여기 케이스로 넣었다:

  1. `document_id`를 corpus.jsonl 줄 번호로 가정 → 정답 13건 전부 실패
     (load_corpus.py가 파일 순서대로 넣지 않는다)
  2. 청크가 원문의 부분 문자열일 거라고 가정 → 역시 전부 실패
     (split_text가 분할 단위를 "\\n"으로 잇는데 원문은 "\\n\\n"이다)
"""

from app.services.chunking import ChunkConfig, clean_for_chunking, split_text
from scripts.eval.chunk_experiment import Gold, Passage, covers, locate, locate_chunks

DOC = (
    "First paragraph about barking at the doorbell.\n\n"
    "Second paragraph explains separation anxiety in dogs.\n\n"
    "Third paragraph covers crate training basics for puppies.\n\n"
    "Fourth paragraph is about loose lead walking and pulling."
)


def test_locate_finds_a_literal_substring():
    start, end = locate(DOC, "Second paragraph explains separation anxiety in dogs.")
    assert DOC[start:end].startswith("Second paragraph")


def test_locate_handles_chunks_that_are_not_substrings():
    """`split_text`가 문단을 "\\n"으로 이으면 원문("\\n\\n")과 글자가 어긋난다.

    이걸 처리 못 해서 처음에 정답 구간을 하나도 못 찾았다.

    **결과는 정확값이 아니라 근사값이다.** 청크가 문단을 이을 때마다 원문보다
    글자 하나씩 짧아지므로 시작점이 그만큼 밀린다. 위 예에서도 0이 아니라 1이 나온다.
    구간 겹침을 정답 길이의 50%로 판정하므로 이 정도 오차는 판정을 바꾸지 않는다 —
    **다만 "근사"라는 걸 알고 써야 한다.**
    """
    chunk = (
        "First paragraph about barking at the doorbell.\n"
        "Second paragraph explains separation anxiety in dogs."
    )
    assert chunk not in DOC, "전제: 이 청크는 원문의 부분 문자열이 아니다"
    start, end = locate(DOC, chunk)
    assert abs(start - 0) <= 5, f"시작점이 근사 범위 안이어야 한다: {start}"
    assert end > start


def test_locate_returns_negative_when_absent():
    assert locate(DOC, "완전히 다른 내용입니다") == (-1, -1)


def test_locate_chunks_advances_past_earlier_matches():
    """겹침 때문에 같은 문장이 두 청크에 나온다. 매번 처음부터 찾으면 뒤 청크가
    앞 위치로 잘못 잡혀 순서가 뒤죽박죽이 된다."""
    chunks = split_text(DOC, ChunkConfig(size=110, overlap=20))
    spans = locate_chunks(clean_for_chunking(DOC), chunks)
    starts = [s for s, _ in spans if s >= 0]
    assert starts == sorted(starts), f"구간이 문서 순서대로 나와야 한다: {starts}"


def test_real_chunks_are_all_located():
    """실제 청킹 결과가 전부 위치를 찾아야 한다. 하나라도 -1이면 그 청크는
    어떤 질문에도 정답이 될 수 없어 그 설정만 부당하게 손해를 본다."""
    chunks = split_text(DOC, ChunkConfig(size=120, overlap=20))
    spans = locate_chunks(clean_for_chunking(DOC), chunks)
    assert all(s >= 0 for s, _ in spans), spans


# ── covers ───────────────────────────────────────────────────────────


def gold(start: int, end: int, doc: int = 1) -> Gold:
    return Gold(question="q", document_id=doc, start=start, end=end)


def passage(start: int, end: int, doc: int = 1) -> Passage:
    return Passage(document_id=doc, text="t", start=start, end=end)


def test_covers_requires_the_same_document():
    assert not covers(gold(0, 100), passage(0, 100, doc=2), 0.5)


def test_covers_full_containment():
    assert covers(gold(100, 200), passage(0, 400), 0.5)


def test_covers_half_overlap_is_the_boundary():
    # 정답 100자 중 50자를 덮는다 → 임계 0.5면 통과, 0.6이면 실패.
    assert covers(gold(100, 200), passage(150, 400), 0.5)
    assert not covers(gold(100, 200), passage(150, 400), 0.6)


def test_covers_rejects_a_grazing_overlap():
    """살짝 스친 청크가 정답이 되면 **큰 청크가 부당하게 유리해진다** —
    크기 비교 실험이 크기 자랑이 되어버린다."""
    assert not covers(gold(100, 200), passage(190, 900), 0.5)


def test_covers_rejects_unlocated_passages():
    assert not covers(gold(100, 200), passage(-1, -1), 0.5)
