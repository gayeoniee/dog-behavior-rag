"""PDF의 일부 페이지만 뽑아 local fetcher용 텍스트로 저장한다.

    uv run python -m scripts.collect.extract_pages \
        --input "~/Downloads/반려동물 종합안내서.pdf" \
        --pages 52-60 \
        --source korea-gov-materials \
        --name 행동분석

기관 발간물은 전체가 아니라 한 장(章)만 관련 있는 경우가 많다. 예: 65쪽짜리
안내서에서 훈련·행동은 52~60쪽뿐. 전체를 넣으면 사료·법령·보험 같은 무관한
내용이 검색에 섞인다.

원본 PDF는 `data/raw/local/` **밖**에 두세요. 안에 두면 local fetcher가 추출본과
원본을 둘 다 읽어 같은 내용이 두 번 들어갑니다 (normalize의 중복 제거는 본문이
완전히 같을 때만 걸립니다).
"""

import argparse
import sys
from pathlib import Path

from .models import LOCAL_DIR


def parse_range(value: str) -> tuple[int, int]:
    """ "52-60" 또는 "52" → (시작, 끝). 1-based, 양끝 포함 — 사람이 PDF 뷰어에서
    보는 쪽번호를 그대로 쓸 수 있게."""
    if "-" in value:
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
    else:
        start = end = int(value)
    if start < 1 or end < start:
        raise SystemExit(f"✗ 잘못된 페이지 범위: {value!r}")
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF 일부 페이지 → local fetcher용 .txt")
    parser.add_argument("--input", type=Path, required=True, help="원본 PDF 경로")
    parser.add_argument("--pages", required=True, help="추출할 범위 (예: 52-60)")
    parser.add_argument("--source", required=True, help="sources.yaml의 소스 id")
    parser.add_argument("--name", required=True, help="저장할 파일명(확장자 제외) = 문서 제목")
    args = parser.parse_args()

    try:
        from pypdf import PdfReader
    except ImportError:
        print("✗ pypdf 미설치 — uv sync --extra collect", file=sys.stderr)
        return 1

    if not args.input.is_file():
        print(f"✗ 파일이 없습니다: {args.input}", file=sys.stderr)
        return 1

    reader = PdfReader(str(args.input))
    start, end = parse_range(args.pages)
    if start > len(reader.pages):
        print(f"✗ 시작 쪽({start})이 총 쪽수({len(reader.pages)})를 넘습니다", file=sys.stderr)
        return 1
    end = min(end, len(reader.pages))

    pages = [(reader.pages[i].extract_text() or "").strip() for i in range(start - 1, end)]
    text = "\n\n".join(p for p in pages if p)
    if not text:
        print("✗ 텍스트가 비어 있습니다 (스캔본 PDF일 수 있습니다)", file=sys.stderr)
        return 1

    out_dir = LOCAL_DIR / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.txt"
    out_path.write_text(text, encoding="utf-8")

    print(f"✓ {args.input.name} {start}~{end}쪽 → {out_path} ({len(text):,}자)")
    if args.input.resolve().is_relative_to(LOCAL_DIR.resolve()):
        print(
            f"⚠️  원본 PDF가 {LOCAL_DIR} 안에 있습니다. 그대로 두면 fetch가 전체 PDF도 "
            "같이 읽어 중복 적재됩니다 — 밖으로 옮기세요."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
