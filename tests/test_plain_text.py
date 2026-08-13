"""마크다운 제거 테스트.

화면이 answer를 평문으로 렌더링하므로 별표가 남으면 그대로 노출된다.
**내용은 지우지 않는다**는 게 핵심이다 — 기호만 벗긴다.
"""

from app.services.plain_text import strip_markdown, trim_to_form


def test_bold_markers_are_removed_but_text_stays():
    assert strip_markdown("**진단** 좌절감입니다") == "진단 좌절감입니다"
    assert strip_markdown("__진단__ 좌절감") == "진단 좌절감"


def test_italic_markers_are_removed():
    assert strip_markdown("*즉시* 간식을 주세요") == "즉시 간식을 주세요"


def test_snake_case_underscores_survive():
    """코드 식별자를 망가뜨리면 안 된다."""
    assert strip_markdown("load_corpus 를 실행하세요") == "load_corpus 를 실행하세요"


def test_multiplication_and_stray_asterisks_are_safe():
    assert strip_markdown("하루 2 * 3회") == "하루 2 * 3회"


def test_headings_are_removed():
    assert strip_markdown("### 이렇게 해보세요\n1. 멈춥니다") == "이렇게 해보세요\n1. 멈춥니다"


def test_bullets_become_middle_dots():
    assert strip_markdown("- 좌절감\n- 흥분") == "· 좌절감\n· 흥분"


def test_numbered_list_is_preserved():
    """번호 목록은 답변 형식의 일부라 남겨야 한다."""
    assert strip_markdown("1. 멈춥니다\n2. 간식을 줍니다") == "1. 멈춥니다\n2. 간식을 줍니다"


def test_inline_code_is_unwrapped():
    assert strip_markdown("`앉아` 라고 말하세요") == "앉아 라고 말하세요"


def test_links_keep_only_the_text():
    """링크는 sources에 이미 있으므로 본문에서는 텍스트만 남긴다."""
    assert strip_markdown("[RSPCA 자료](https://example.com)를 보세요") == "RSPCA 자료를 보세요"


def test_blockquote_marker_is_removed():
    assert strip_markdown("> 인용문입니다") == "인용문입니다"


def test_excess_blank_lines_collapse():
    assert strip_markdown("가\n\n\n\n나") == "가\n\n나"


def test_empty_and_none_like_input():
    assert strip_markdown("") == ""
    assert strip_markdown("   \n  ") == ""


def test_realistic_answer_is_cleaned():
    raw = (
        "**진단:** 좌절감입니다.\n\n"
        "**이렇게 해보세요**\n"
        "1. 멀리 **멈춥니다**.\n"
        "2. 즉시 간식을 줍니다.\n\n"
        "**주의점:** 줄 잡아당기기"
    )
    cleaned = strip_markdown(raw)
    assert "*" not in cleaned
    assert "진단: 좌절감입니다." in cleaned
    assert "1. 멀리 멈춥니다." in cleaned
    assert "주의점: 줄 잡아당기기" in cleaned


# ── trim_to_form ─────────────────────────────────────────────────────


def test_trailing_restatement_after_form_is_cut():
    """폼이 끝난 뒤 같은 조언을 다시 푸는 사족을 자른다 (gemma-4-e2b 실제 출력)."""
    raw = (
        "진단: 흥분해서 당기는 것입니다.\n\n"
        "이렇게 해보세요\n"
        "1. 멈추고 기다리세요.\n\n"
        "주의점: 힘으로 제압하지 마세요.\n\n"
        "보호자가 줄을 너무 당긴다고 하셨으니, 산책 전에 간식을 주고…"
    )
    trimmed = trim_to_form(raw)
    assert trimmed.endswith("주의점: 힘으로 제압하지 마세요.")
    assert "보호자가 줄을" not in trimmed


def test_old_label_is_still_trimmed():
    """프롬프트를 "주의점"으로 바꿔도 모델이 옛 라벨을 뱉는 일이 있다.

    그때 폼의 끝을 못 찾으면 사족 잘라내기가 조용히 죽는다. 둘 다 받는다.
    """
    raw = "진단: 좌절감입니다.\n\n피하세요: 혼내지 마세요.\n\n덧붙이자면 산책을 늘리세요."
    assert trim_to_form(raw).endswith("피하세요: 혼내지 마세요.")


def test_vet_line_is_part_of_the_form_and_survives():
    """`병원:` 줄은 통증이 의심될 때 붙는 폼의 일부다. 사족이 아니다."""
    raw = (
        "진단: 통증일 수 있습니다.\n\n"
        "주의점: 혼내지 마세요.\n\n"
        "병원: 절뚝이면 바로 가보세요.\n\n"
        "다시 말씀드리면 통증이 의심되니 병원에 가보세요."
    )
    trimmed = trim_to_form(raw)
    assert "병원: 절뚝이면 바로 가보세요." in trimmed
    assert "다시 말씀드리면" not in trimmed


def test_training_form_ends_at_point_line():
    raw = (
        "앉아 가르치기\n1. 간식을 코앞에.\n\n"
        "포인트: 엉덩이가 닿는 순간 보상.\n\n"
        "요약하자면 타이밍이 중요합니다."
    )
    assert trim_to_form(raw).endswith("포인트: 엉덩이가 닿는 순간 보상.")


def test_answers_without_form_labels_are_untouched():
    """되묻기·근거 없음 응답은 폼이 다르다. 손대면 내용이 날아간다."""
    raw = (
        "증상만으로는 원인을 좁힐 수 없습니다.\n\n"
        "알려주시면 좋아요\n1. 혼자 있을 때만 그러나요?\n\n"
        "2. 언제부터였나요?"
    )
    assert trim_to_form(raw) == raw


def test_form_at_end_of_text_is_kept_whole():
    raw = "진단: 좌절감입니다.\n\n주의점: 혼내지 마세요."
    assert trim_to_form(raw) == raw


def test_empty_input_survives():
    assert trim_to_form("") == ""
