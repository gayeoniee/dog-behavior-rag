"""청킹 단위 테스트. torch도 DB도 필요 없다 — 맨몸 `uv sync`에서 돌아야 한다."""

from app.services.chunking import (
    ChunkConfig,
    clean_for_chunking,
    looks_like_reference_list,
    split_text,
)

SMALL = ChunkConfig(size=200, overlap=30, min_size=50)


# ── clean_for_chunking ────────────────────────────────────────────────


def test_hyphen_wrap_is_rejoined():
    """pypdf가 단어 중간에서 끊은 하이픈. 안 붙이면 임베딩이 두 토막으로 본다."""
    assert "counterconditioning" in clean_for_chunking("counter-\nconditioning works")


def test_hyphen_between_words_is_kept():
    """줄바꿈이 없는 정상 하이픈까지 붙여버리면 안 된다."""
    assert "reward-based" in clean_for_chunking("reward-based training")


def test_page_number_only_lines_are_dropped():
    cleaned = clean_for_chunking("본문 첫 줄\n42\n본문 둘째 줄")
    assert "42" not in cleaned.split("\n")
    assert "본문 첫 줄" in cleaned


def test_running_header_repeated_three_times_is_dropped():
    text = "\n".join(["AAHA Behavior Guidelines", "내용 A"] * 3)
    assert "AAHA Behavior Guidelines" not in clean_for_chunking(text)


def test_line_repeated_twice_is_kept():
    """반복 2회는 러닝 헤더로 보지 않는다 — 정상 본문이 우연히 겹칠 수 있다."""
    text = "\n".join(["같은 문장이다", "내용 A"] * 2)
    assert "같은 문장이다" in clean_for_chunking(text)


def test_long_repeated_line_is_kept():
    """80자 이상이면 3회 반복돼도 헤더가 아니라 본문으로 본다."""
    long_line = "이 문장은 러닝 헤더로 보기에는 충분히 길어서 본문으로 취급되어야 한다" * 3
    assert len(long_line) >= 80
    assert long_line in clean_for_chunking("\n".join([long_line, "사이 내용"] * 3))


# ── looks_like_reference_list ─────────────────────────────────────────


def test_reference_block_is_detected():
    refs = "\n".join(
        f"{i}. Smith J, et al. Canine behavior study. J Vet Behav. 2019;{i}:1-10. "
        f"https://doi.org/10.1000/{i}"
        for i in range(1, 8)
    )
    assert looks_like_reference_list(refs)


def test_numbered_training_protocol_is_not_a_reference_list():
    """핵심 회귀: 훈련 절차가 참고문헌과 똑같은 번호 목록 모양이다."""
    protocol = "\n".join(
        [
            "1. 앉아를 먼저 가르친다",
            "2. 기다려를 3초부터 시작한다",
            "3. 성공하면 즉시 보상한다",
            "4. 거리를 조금씩 늘린다",
            "5. 실패하면 이전 단계로 돌아간다",
            "6. 매일 5분씩 반복한다",
        ]
    )
    assert not looks_like_reference_list(protocol)


def test_short_numbered_block_is_not_a_reference_list():
    assert not looks_like_reference_list("1. Smith et al. doi.org/x\n2. Jones et al. doi.org/y")


# ── split_text ────────────────────────────────────────────────────────


def test_empty_input_returns_empty_list():
    assert split_text("") == []
    assert split_text("   \n\n  \t ") == []


def test_every_chunk_respects_size():
    text = "\n\n".join(f"문단 {i}. " + "가나다라마바사아자차" * 12 for i in range(20))
    assert all(len(c) <= SMALL.size for c in split_text(text, SMALL))


def test_paragraph_without_separators_is_hard_sliced():
    """구분자가 하나도 없는 긴 덩어리도 반드시 size 이하로 쪼개져야 한다."""
    chunks = split_text("가" * 5000, SMALL)
    assert chunks and all(len(c) <= SMALL.size for c in chunks)


def test_short_document_survives_as_single_chunk():
    """min_size보다 짧아도 문서에 청크가 하나뿐이면 버리지 않는다."""
    assert split_text("짧은 문서다.", SMALL) == ["짧은 문서다."]


def test_short_tail_is_merged_into_previous_chunk():
    """꼬리를 버리면 문서의 결론 문단이 조용히 사라진다."""
    cfg = ChunkConfig(size=200, overlap=0, min_size=80)
    text = "\n\n".join(["가" * 190, "나" * 190, "결론이다"])
    chunks = split_text(text, cfg)
    assert all(len(c) >= cfg.min_size for c in chunks)
    assert "결론이다" in chunks[-1]


def test_overlap_carries_context_forward():
    cfg = ChunkConfig(size=120, overlap=40, min_size=10)
    chunks = split_text("\n\n".join("문단내용" * 25 for _ in range(4)), cfg)
    assert len(chunks) >= 2
    # 뒤 청크의 앞부분이 앞 청크의 끝부분에서 온 텍스트를 담고 있어야 한다.
    assert any(chunks[i][-20:] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_overlap_does_not_start_mid_word():
    cfg = ChunkConfig(size=150, overlap=40, min_size=10)
    text = " ".join(f"word{i:03d}" for i in range(200))
    for chunk in split_text(text, cfg):
        assert not chunk.startswith(("ord", "rd", "d0"))


def test_all_content_survives_chunking():
    """청크를 이어붙이면 원문의 모든 문단이 어딘가에는 남아 있어야 한다."""
    paragraphs = [f"고유표식{i} " + "내용" * 40 for i in range(15)]
    joined = "".join(split_text("\n\n".join(paragraphs), SMALL))
    assert all(f"고유표식{i}" in joined for i in range(15))


def test_reference_chunks_are_filtered_out():
    body = "\n\n".join("훈련 방법에 대한 본문이다. " * 10 for _ in range(3))
    refs = "\n".join(
        f"{i}. Smith J, et al. Study. J Vet Behav. 2019;{i}:1-10. https://doi.org/10.1000/{i}"
        for i in range(1, 10)
    )
    chunks = split_text(f"{body}\n\n{refs}", ChunkConfig(size=600, overlap=0, min_size=50))
    assert chunks
    assert not any("doi.org" in c and c.count("doi.org") >= 3 for c in chunks)
