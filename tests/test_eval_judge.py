"""평가 스크립트의 판정 함수 테스트.

**평가셋이 거짓말을 하면 아무도 모른다.** 실제로 그랬다 — `mentions()`가 부분
문자열에 걸려(`tail`←detail, `aging`←managing) 주제를 안 다루는 근거를 통과로 세고
있었고, 커밋 `0ad667f`에서 고칠 때까지 점수가 부풀어 있었다. 그 재발을 여기서 막는다.

DB도 LLM도 부르지 않는다. 두 모듈 다 import 부작용이 없어서 순수 함수만 꺼내 쓴다.
"""

from scripts.eval.replay import REPEAT_WARN, label_line, overlap
from scripts.eval.retrieval_report import is_practical, judge, mentions
from tests.fakes import hit

# ── mentions ─────────────────────────────────────────────────────────
#
# `mentions()`는 소문자로 정규화된 본문을 받는다 (judge()가 h.content.lower()로 넘긴다).


def test_keyword_matches_inflected_forms():
    """뒤쪽 경계를 열어둔 이유. 이게 깨지면 멀쩡한 근거가 떨어진다."""
    assert mentions("the dog keeps barking at night", "bark")
    assert mentions("destructive chewing when left alone", "chew")
    assert mentions("separation anxiety in dogs", "anxiety")


def test_keyword_does_not_match_inside_another_word():
    """실측된 오탐 목록 그대로. 앞쪽 경계가 이걸 막는다.

    전부 키워드가 **다른 단어의 중간이나 끝**에 박힌 경우다 — 거기엔 단어 경계가
    없으므로 `\\b`가 걸러낸다.
    """
    assert not mentions("see the detailed instructions below", "tail")
    assert not mentions("managing the behaviour over time", "aging")
    assert not mentions("adopted from a rescue centre", "cue")
    assert not mentions("this advice is misleading", "lead")
    assert not mentions("a sudden change in behaviour", "den")
    assert not mentions("there is little evidence for this", "den")


def test_prefix_collisions_are_not_prevented():
    """**앞쪽 경계로는 막을 수 없는 종류가 있다.** 이건 한계이지 버그가 아니다.

    `lead`가 "leading"에 걸린다. 앞만 거는 규칙에서는 당연한데, 바로 그 규칙이
    `bark`로 "barking"을 잡게 해주는 것이기도 하다. 둘을 구분하려면 형태소 분석이
    필요하고 이 리포트에 그건 과하다.

    `retrieval_report.mentions`의 주석은 한때 `lead ← leading`을 막힌 오탐으로
    적어두고 있었다 — 실제로 막히는 건 "misleading"뿐이다.

    **대응은 코드가 아니라 평가셋 쪽이다.** 흔한 접두사인 단어는 키워드로 쓰지 않고
    두 단어로 쓴다. leash-pulling 항목이 그렇게 되어 있다.
    """
    assert mentions("leading the dog by the collar", "lead")
    assert not mentions("leading the dog by the collar", "loose lead")
    assert not mentions("leading the dog by the collar", "on the lead")


def test_multi_word_keyword_is_matched_literally():
    """`house training` 같은 두 단어 키워드도 그대로 쓴다."""
    assert mentions("house training takes weeks", "house training")
    assert not mentions("training the dog indoors", "house training")


# ── judge ────────────────────────────────────────────────────────────
#
# 인자 순서는 judge(entry, hits, kept, coverage)다. `hits`는 실제로 쓰이지 않고
# `kept`·`coverage`만 본다 — 넘기되 판정에 영향이 없다는 것을 알고 있어야 한다.


def entry(expect: str, *keywords: str) -> dict:
    return {"question": "질문", "topic": "topic", "expect": expect, "keywords": list(keywords)}


def test_out_of_scope_passes_only_when_refused():
    """범위 밖 질문은 **답하지 않아야** 통과다. 되묻기도 실패다 — 개 질문이 아니니까."""
    e = entry("out-of-scope")
    assert judge(e, [], [], "none")
    assert not judge(e, [hit()], [hit()], "full")
    assert not judge(e, [hit()], [hit()], "partial")
    assert not judge(e, [], [], "needs_detail")


def test_uncovered_accepts_asking_back_but_not_answering():
    """코퍼스 공백은 거절이나 되묻기가 정답이다.

    `partial`은 실패여야 한다 — 지금 뛰어오르기·억제된 배변이 여기서 떨어지고 있고,
    그게 19/21의 나머지 2다. 이 단언이 무너지면 공백이 통과로 둔갑한다.
    """
    e = entry("uncovered")
    assert judge(e, [], [], "none")
    assert judge(e, [], [], "needs_detail")
    assert not judge(e, [hit()], [hit()], "partial")
    assert not judge(e, [hit()], [hit()], "full")


def test_covered_needs_a_keyword_in_the_kept_evidence():
    e = entry("covered", "bark")
    kept = [hit(content="dogs bark when left alone")]
    assert judge(e, kept, kept, "full")
    assert judge(e, kept, kept, "partial"), "부분 근거도 답변이다"


def test_covered_fails_when_evidence_is_off_topic():
    e = entry("covered", "bark")
    kept = [hit(content="crate training basics")]
    assert not judge(e, kept, kept, "full")


def test_covered_fails_when_nothing_was_kept():
    """선별이 전부 버렸으면 검색이 뭘 물어왔든 실패다."""
    e = entry("covered", "bark")
    hits = [hit(content="dogs bark when left alone")]
    assert not judge(e, hits, [], "none")


def test_covered_fails_when_only_a_substring_matches():
    """오탐이 judge()까지 뚫고 오는지 — 어제 점수를 부풀린 바로 그 경로다."""
    e = entry("covered", "tail")
    kept = [hit(content="see the detailed instructions below")]
    assert not judge(e, kept, kept, "full")


def test_covered_passes_if_any_keyword_matches():
    """키워드는 OR다. 하나만 걸려도 통과."""
    e = entry("covered", "punishment", "aversive")
    kept = [hit(content="aversive methods increase fear")]
    assert judge(e, kept, kept, "full")


# ── is_practical ─────────────────────────────────────────────────────


def test_institutional_documents_are_practical():
    assert is_practical("ASPCA — Separation Anxiety")
    assert is_practical("RSPCA: teaching recall")
    assert not is_practical("Journal of Veterinary Behavior 2021")


# ── overlap ──────────────────────────────────────────────────────────

ANSWER = """진단: 분리불안 가능성이 높습니다.
이렇게 해보세요
1. 외출 전 인사를 짧게 합니다.
2. 켄넬을 편한 공간으로 만듭니다.
3. 혼자 두는 시간을 조금씩 늘립니다.
주의점: 혼내지 마세요."""

# 단계 하나만 바꾼 답변. 사람 눈에는 같은 답이고, 완전 일치만 보던 검사는 이걸 통과시켰다.
ONE_STEP_CHANGED = ANSWER.replace("3. 혼자 두는 시간을 조금씩 늘립니다.", "3. 산책량을 늘립니다.")

# 정상적인 후속 답변. 폼 라벨("이렇게 해보세요")만 겹친다.
FOLLOW_UP = """진단: 켄넬 훈련은 도움이 됩니다.
이렇게 해보세요
1. 문을 열어둔 채 간식을 넣어줍니다.
2. 밥그릇을 켄넬 안에 둡니다.
3. 문 닫는 시간을 5초부터 시작합니다.
주의점: 벌로 가두지 마세요."""


def test_identical_answers_overlap_completely():
    assert overlap(ANSWER, ANSWER) == 1.0


def test_partial_repeat_is_caught():
    """실측 0.83. 임계를 넘겨야 경고가 뜬다."""
    assert overlap(ONE_STEP_CHANGED, ANSWER) > REPEAT_WARN


def test_normal_follow_up_stays_below_the_threshold():
    """실측 0.17. 폼이 고정돼 있어 0은 될 수 없지만 임계보다는 한참 아래다."""
    assert overlap(FOLLOW_UP, ANSWER) < REPEAT_WARN


def test_threshold_sits_between_the_two_measured_cases():
    """0.5를 고른 근거 자체를 고정한다 — 임계를 옮기면 이 테스트가 먼저 말한다."""
    assert overlap(FOLLOW_UP, ANSWER) < REPEAT_WARN < overlap(ONE_STEP_CHANGED, ANSWER)


def test_reworded_line_still_counts_as_repeat():
    """표현이 조금 달라도 잡는다 — 내부 유사도 임계 0.85의 근거."""
    reworded = ANSWER.replace("주의점: 혼내지 마세요.", "주의점: 절대 혼내지 마세요.")
    assert overlap(reworded, ANSWER) == 1.0


def test_empty_answer_does_not_crash():
    """1번 턴은 비교 대상이 없고, 모델이 빈 답을 낼 때도 있다."""
    assert overlap("", ANSWER) == 0.0
    assert overlap(ANSWER, "") == 0.0


# ── label_line ───────────────────────────────────────────────────────


def test_label_line_is_found_with_either_colon():
    """모델이 전각 콜론을 쓰는 일이 있다. 놓치면 고정 문구 검사가 조용히 빈다."""
    assert label_line(ANSWER, "주의점") == "주의점: 혼내지 마세요."
    assert label_line("주의점： 혼내지 마세요.", "주의점") == "주의점： 혼내지 마세요."


def test_label_line_returns_none_when_absent():
    assert label_line(ANSWER, "포인트") is None


def test_label_line_ignores_leading_whitespace():
    assert label_line("  진단: 좌절감입니다.", "진단") == "진단: 좌절감입니다."


def test_label_is_matched_only_at_the_line_start():
    """본문 안에 라벨 단어가 나왔다고 그 줄을 집으면 안 된다."""
    assert label_line("앞서 말한 진단: 좌절감을 참고하세요.", "진단") is None
