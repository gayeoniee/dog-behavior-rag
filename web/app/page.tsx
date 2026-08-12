"use client";

import { useState } from "react";
import { askQuestion, isStubAnswer, type ChatResponse } from "@/lib/api";

// data/coverage_questions.yaml의 질문들. 검색 회귀 테스트 입력과 같은 것을 쓴다 —
// 화면에서 눌러본 결과와 통합 테스트 결과가 어긋나면 바로 눈에 띈다.
const EXAMPLES = [
  "강아지가 초인종 소리에 계속 짖어요",
  "혼자 두고 나가면 집안 물건을 다 물어뜯어요",
  "꼬리를 흔드는데 왜 으르렁거려요?",
  "나이 든 강아지가 밤에 서성거리고 벽을 보고 있어요",
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError(null);
    try {
      setResult(await askQuestion(trimmed));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>반려동물 훈련·문제행동 상담</h1>
      <p className="subtitle">
        기관·학술 자료(AVSAB, AAHA, RSPCA, VCA, PMC 오픈액세스)만 근거로 씁니다.
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit(question);
        }}
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="예: 강아지가 초인종 소리에 계속 짖어요"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void submit(question);
            }
          }}
        />
        <div className="row">
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? "검색 중…" : "물어보기"}
          </button>
          <span className="score">Ctrl/⌘ + Enter</span>
        </div>
      </form>

      <div className="examples">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            className="chip"
            disabled={loading}
            onClick={() => {
              setQuestion(example);
              void submit(example);
            }}
          >
            {example}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          {/* 지금 answer는 스텁이다. 화면이 그걸 답변인 척하면 안 된다. */}
          {isStubAnswer(result.answer) && (
            <div className="banner" style={{ marginTop: "1.75rem" }}>
              <strong>LLM 미연결</strong> — 아래 답변은 자리표시자입니다. 지금
              동작하는 건 검색이고, 근거 문서는 실제 결과입니다.
            </div>
          )}

          <div className="meta">
            <span>{result.latency_ms}ms</span>
            <span>{result.provider}</span>
            <span>근거 {result.sources.length}건</span>
          </div>
          <div className="answer">{result.answer}</div>

          <h2>근거 문서</h2>
          {result.sources.length === 0 && (
            <p className="subtitle">
              검색 결과가 없습니다. 코퍼스가 적재됐는지 확인하세요
              (<code>scripts.db.load_corpus</code>).
            </p>
          )}
          {result.sources.map((source) => (
            <article key={source.chunk_id} className="source">
              <div className="source-head">
                <span className="source-title">{source.document_title}</span>
                <span className="score">{source.score.toFixed(3)}</span>
              </div>
              <div className="bar">
                <span
                  style={{
                    width: `${Math.max(0, Math.min(100, source.score * 100))}%`,
                  }}
                />
              </div>
              {source.source && (
                <a href={source.source} target="_blank" rel="noreferrer">
                  {source.source}
                </a>
              )}
              <details>
                <summary>본문 보기</summary>
                <p>{source.content}</p>
              </details>
            </article>
          ))}
        </>
      )}
    </main>
  );
}
