"""LLM 답변에서 마크다운 기호를 걷어낸다.

화면(`web/`)과 안드로이드 앱은 `answer`를 **평문 그대로** 보여준다. 마크다운
렌더러를 붙이지 않은 이유는 답변 형식이 고정돼 있어(진단/이렇게 해보세요/피하세요)
서식이 필요 없기 때문이다. 그런데 LLM은 지시해도 습관적으로 `**굵게**`를 넣고,
그러면 화면에 별표가 그대로 노출된다.

프롬프트에서 금지하는 것만으로는 부족해서 **서버에서 한 번 더 거른다.**
클라이언트마다 따로 처리하면 앱과 웹이 어긋나므로 응답을 만드는 쪽에서 정리한다.
"""

import re

__all__ = ["strip_markdown"]

# **굵게** / __굵게__ → 굵게
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
# *기울임* / _기울임_ → 기울임. 단어 중간의 밑줄(snake_case)은 건드리지 않는다.
_ITALIC = re.compile(
    r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])|(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])"
)
# `코드` → 코드
_CODE = re.compile(r"`{1,3}([^`]*?)`{1,3}", re.DOTALL)
# 줄 앞의 #, >, 표 구분선
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_QUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
# 줄 앞의 - * + 불릿 → 가운뎃점. 번호 목록(1.)은 형식에 쓰이므로 남긴다.
_BULLET = re.compile(r"^(\s*)[-*+]\s+", re.MULTILINE)
# [텍스트](링크) → 텍스트 (링크는 sources에 이미 있다)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BLANKS = re.compile(r"\n{3,}")


def strip_markdown(text: str) -> str:
    """마크다운 기호를 제거하고 평문으로 만든다. 내용은 지우지 않는다."""
    if not text:
        return text

    text = _CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _BOLD.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _ITALIC.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _BULLET.sub(r"\1· ", text)
    # 코드펜스만 있던 줄(``` 단독)이 빈 줄로 남는 경우 정리
    text = _BLANKS.sub("\n\n", text)
    return text.strip()
