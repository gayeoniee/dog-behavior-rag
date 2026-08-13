"""LLM 답변에서 마크다운 기호를 걷어낸다.

화면(`web/`)과 안드로이드 앱은 `answer`를 **평문 그대로** 보여준다. 마크다운
렌더러를 붙이지 않은 이유는 답변 형식이 고정돼 있어(진단/이렇게 해보세요/피하세요)
서식이 필요 없기 때문이다. 그런데 LLM은 지시해도 습관적으로 `**굵게**`를 넣고,
그러면 화면에 별표가 그대로 노출된다.

프롬프트에서 금지하는 것만으로는 부족해서 **서버에서 한 번 더 거른다.**
클라이언트마다 따로 처리하면 앱과 웹이 어긋나므로 응답을 만드는 쪽에서 정리한다.
"""

import re

__all__ = ["strip_markdown", "trim_to_form"]

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


# 답변 폼의 **마지막** 구획. 문제행동은 "피하세요:", 훈련법은 "포인트:"로 끝난다.
_FORM_TAIL = re.compile(r"^[ \t]*(피하세요|포인트)\s*[:：]", re.MULTILINE)
# 통증·질병 신호가 있을 때만 마지막에 한 줄 더 붙는다. 이건 폼의 일부라 살린다.
_VET_LINE = re.compile(r"^[ \t]*병원\s*[:：]", re.MULTILINE)


def trim_to_form(text: str) -> str:
    """정해진 폼이 끝난 뒤에 붙은 사족을 잘라낸다.

    작은 모델(gemma-4-e2b)이 폼을 지켜 답한 다음 같은 조언을 한 문단 더 풀어 쓴다.
    폼을 고정한 것이 답변을 1,200자 → 300자로 줄인 장치인데, 사족이 붙으면 그게
    도로 무너진다. 프롬프트에 "반복하지 마세요"라고 이미 적혀 있지만 지키지 않으므로
    **코드에서 보장한다** — 근거 없는 단정을 코드에서 강등하는 것과 같은 이유다.

    라벨이 없으면(되묻기·근거 없음 응답) 손대지 않는다. 그 답변들은 폼이 다르다.
    """
    if not text:
        return text

    tail = _FORM_TAIL.search(text)
    if not tail:
        return text

    # 마지막 구획은 다음 빈 줄까지다.
    end = text.find("\n\n", tail.end())
    if end == -1:
        return text.strip()

    rest = text[end:]
    vet = _VET_LINE.search(rest)
    if vet:
        # "병원:" 줄은 폼의 일부이므로 그 문단까지 살리고 그 뒤를 자른다.
        vet_end = rest.find("\n\n", vet.end())
        kept = rest if vet_end == -1 else rest[:vet_end]
        return (text[:end] + kept).strip()

    return text[:end].strip()


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
