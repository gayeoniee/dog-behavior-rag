"""마크다운 제거 테스트.

화면이 answer를 평문으로 렌더링하므로 별표가 남으면 그대로 노출된다.
**내용은 지우지 않는다**는 게 핵심이다 — 기호만 벗긴다.
"""

from app.services.plain_text import strip_markdown


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
        "**피하세요:** 줄 잡아당기기"
    )
    cleaned = strip_markdown(raw)
    assert "*" not in cleaned
    assert "진단: 좌절감입니다." in cleaned
    assert "1. 멀리 멈춥니다." in cleaned
    assert "피하세요: 줄 잡아당기기" in cleaned
