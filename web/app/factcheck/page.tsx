"use client";

import Link from "next/link";
import { useState } from "react";
import {
  factCheck,
  type FactCheckResponse,
  type Verdict,
} from "@/lib/api";

const VERDICT: Record<Verdict, { label: string; cls: string }> = {
  contradicted: { label: "자료와 배치", cls: "v-bad" },
  supported: { label: "자료가 뒷받침", cls: "v-good" },
  // 이 라벨이 "판정 불가"가 아니라 "근거 없음"인 게 중요하다 — 자료에 없다는
  // 사실 자체가 결과이지, 시스템이 실패한 게 아니다.
  not_covered: { label: "자료에 근거 없음", cls: "v-none" },
};

// 삭제한 그 한국어 안내서의 지배이론 조언. 기능을 바로 체감할 수 있는 예시다.
const EXAMPLE =
  "강아지가 마운팅을 하는 것은 사람보다 서열이 위라고 생각하기 때문이다. " +
  "'안 돼!'라고 단호하게 소리친 뒤 복종 자세를 1-2분 유지시켜야 한다.";

export default function FactCheckPage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<FactCheckResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(input: string) {
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await factCheck(trimmed));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <nav className="nav">
        <Link href="/">상담</Link>
        <span className="nav-current">조언 검증</span>
      </nav>

      <h1>이 조언, 근거가 있나요?</h1>
      <p className="subtitle">
        유튜브·블로그·주변에서 들은 훈련 조언을 붙여넣으면 기관·학술 자료와
        대조합니다.
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit(text);
        }}
      >
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="예: 강아지가 마운팅하는 건 서열이 위라고 생각해서예요…"
          style={{ minHeight: "140px" }}
        />
        <div className="row">
          <button type="submit" disabled={loading || !text.trim()}>
            {loading ? "검증 중… (20초 내외)" : "검증하기"}
          </button>
          <button
            type="button"
            className="chip"
            disabled={loading}
            onClick={() => {
              setText(EXAMPLE);
              void submit(EXAMPLE);
            }}
          >
            예시로 해보기
          </button>
        </div>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <div className="meta">
            <span>{result.latency_ms}ms</span>
            <span>{result.provider}</span>
            <span>주장 {result.claims.length}개</span>
          </div>

          {result.claims.map((c, i) => (
            <article key={i} className={`verdict ${VERDICT[c.verdict].cls}`}>
              <div className="verdict-head">
                <span className="verdict-badge">
                  {VERDICT[c.verdict].label}
                </span>
                <span className="verdict-claim">{c.claim}</span>
              </div>
              <p className="verdict-why">{c.explanation}</p>
              {c.sources.length > 0 && (
                <details>
                  <summary>근거 {c.sources.length}건</summary>
                  <div>
                    {c.sources.map((s, j) => (
                      <p key={s.chunk_id}>
                        <span className="source-num">[자료 {j + 1}]</span>{" "}
                        <strong>{s.document_title}</strong>{" "}
                        <span className="score">{s.score.toFixed(3)}</span>
                        {s.source && (
                          <>
                            <br />
                            <a href={s.source} target="_blank" rel="noreferrer">
                              {s.source}
                            </a>
                          </>
                        )}
                        <br />
                        {s.content}
                      </p>
                    ))}
                  </div>
                </details>
              )}
            </article>
          ))}

          {/* 판정을 중립적인 제3자 검증인 것처럼 보이게 하지 않는다. */}
          <p className="corpus-note">{result.corpus_note}</p>
        </>
      )}
    </main>
  );
}
