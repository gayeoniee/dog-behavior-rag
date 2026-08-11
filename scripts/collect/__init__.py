"""오프라인 데이터 수집 파이프라인.

앱 런타임(app/)과 분리된 배치 스크립트다. 흐름:

    fetch (sources.yaml → data/raw/)
      → normalize (data/raw/ → data/processed/corpus.jsonl)
      → report (corpus.jsonl 품질·커버리지 리포트)
"""
