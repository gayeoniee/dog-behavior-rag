"""corpus.jsonl → DocumentIn 리매핑 + content_hash 고정.

torch도 DB도 필요 없다. `scripts.db.load_corpus`가 import 시점에 엔진을 만들지
않기 때문에 import만으로 안전하다.
"""

from app.services.ingest_service import content_hash
from scripts.db.load_corpus import derive_distribution, to_document_in

# normalize.py가 실제로 내보내는 15키 레코드
RECORD = {
    "id": "vca-know-your-pet:3f9a2b1c8d4e5f60",
    "title": "Dog Behavior--What is Normal?",
    "content": "Dogs communicate through body language ...",
    "content_hash": "3f9a2b1c8d4e5f60",
    "source_id": "vca-know-your-pet",
    "source_url": "https://vcahospitals.com/know-your-pet/dog-behavior",
    "language": "en",
    "species": "dog",
    "axis": ["problem", "cause", "training", "medical"],
    "methodology": "reward_based",
    "authority_tier": 1,
    "published_at": 2024,
    "volatility": "stable",
    "license": "personal-use-only",
    "fetched_at": "2026-08-12T05:14:22.481293+00:00",
}


def test_source_url_maps_to_source():
    """이름이 다른 유일한 키. 여기가 끊기면 답변에 인용 링크가 안 붙는다."""
    doc = to_document_in(RECORD, corpus_partition="answer")
    assert doc.source == RECORD["source_url"]
    assert doc.source_id == "vca-know-your-pet"


def test_metadata_is_carried_through():
    doc = to_document_in(RECORD, corpus_partition="answer")
    assert doc.axis == ["problem", "cause", "training", "medical"]
    assert doc.methodology == "reward_based"
    assert doc.authority_tier == 1
    assert doc.published_at == 2024
    assert doc.language == "en"
    assert doc.content_hash == "3f9a2b1c8d4e5f60"


def test_collect_only_keys_are_dropped():
    """volatility/fetched_at/id는 수집 단계 전용이라 DocumentIn에 없다."""
    doc = to_document_in(RECORD, corpus_partition="answer")
    dumped = doc.model_dump()
    assert "volatility" not in dumped
    assert "fetched_at" not in dumped
    assert "id" not in dumped


def test_partition_is_applied():
    doc = to_document_in(RECORD, corpus_partition="observation")
    assert doc.corpus == "observation"


def test_missing_optional_keys_are_tolerated():
    minimal = {"title": "제목", "content": "본문"}
    doc = to_document_in(minimal, corpus_partition="answer")
    assert doc.title == "제목"
    assert doc.source is None


# ── distribution 파생 ────────────────────────────────────────────────


def test_personal_use_licenses_become_personal_only():
    assert derive_distribution("personal-use-only") == "personal-only"
    assert derive_distribution("personal-use-only-manual-copy") == "personal-only"


def test_permissive_licenses_become_open():
    assert derive_distribution("cc-by") == "open"
    assert derive_distribution("cc0") == "open"
    assert derive_distribution("public-position-statement") == "open"
    assert derive_distribution("public-guideline-pdf") == "open"
    assert derive_distribution("korea-gov-nuri-1") == "open"


def test_nc_and_nd_licenses_are_not_open():
    """NC는 상업적 이용을, ND는 2차적 저작물 작성을 금지한다.

    앱이 상업적인지, RAG 답변이 2차적 저작물인지가 정해지기 전까지는 배포 대상이
    아니다. 정해지면 OPEN_LICENSES에서 재분류할 것.
    """
    assert derive_distribution("cc-by-nc") == "personal-only"
    assert derive_distribution("cc-by-nc-nd") == "personal-only"
    assert derive_distribution("cc-by-nc-sa") == "personal-only"


def test_unknown_license_is_conservative():
    """허용 목록에 없으면 전부 personal-only.

    회귀 대상: `korea-gov-publication`은 예전 구현에서 open으로 분류됐는데,
    실제로 그 이름으로 넣은 PDF가 민간 저작권물이었다 — 정부기관이 배포한다고
    공공저작물인 게 아니다.
    """
    assert derive_distribution(None) == "personal-only"
    assert derive_distribution("") == "personal-only"
    assert derive_distribution("korea-gov-publication") == "personal-only"
    assert derive_distribution("pmc-oa-unspecified") == "personal-only"
    assert derive_distribution("처음 보는 값") == "personal-only"


# ── content_hash ─────────────────────────────────────────────────────


def test_content_hash_matches_normalize_py():
    """`scripts/collect/normalize.py`의 content_hash와 같은 값이어야 한다.

    구현이 두 곳에 나뉘어 있는 이유는 app이 scripts를 import하지 않기 위해서다.
    값이 갈라지면 재적재가 멱등하지 않아 문서가 매번 중복 적재된다.
    아래 기댓값은 sha256(utf-8) 앞 16자를 직접 계산해 박아둔 것이다.
    """
    assert content_hash("강아지가 짖어요") == "72cfbf2ee93d3b9d"
    assert content_hash("Dog behavior guidelines") == "01018822c29ac4dd"
    assert len(content_hash("아무 텍스트")) == 16
