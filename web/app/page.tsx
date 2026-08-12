"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  askQuestion,
  isStubAnswer,
  type ChatResponse,
  type Turn,
} from "@/lib/api";

/** 화면에 쌓이는 대화 한 줄. assistant는 응답 전체를 들고 있어야 근거를 같이 보여준다. */
type Message =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string; data: ChatResponse };

// data/coverage_questions.yaml의 질문들. 검색 회귀 테스트 입력과 같은 것을 쓴다 —
// 화면에서 눌러본 결과와 통합 테스트 결과가 어긋나면 바로 눈에 띈다.
const EXAMPLES = [
  "강아지가 초인종 소리에 계속 짖어요",
  "혼자 두고 나가면 집안 물건을 다 물어뜯어요",
  "앉아를 어떻게 가르쳐요?",
  "산책할 때 줄을 너무 당겨요",
];

/** coverage별 안내. none과 needs_detail은 사용자에게 전혀 다른 상황이다. */
const NOTICE: Record<string, { label: string; text: string } | undefined> = {
  needs_detail: {
    label: "조금 더 알려주세요",
    text: "증상만으로는 원인이 여러 가지라, 아래 질문에 답해 주시면 해당 자료를 찾아 답변드릴 수 있어요.",
  },
  none: {
    label: "참고 자료 없음",
    text: "이 주제는 코퍼스에 근거 자료가 없습니다. 아래 답변은 근거 문서 없이 작성된 일반 안내입니다.",
  },
};

export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    // 서버가 맥락을 풀 수 있도록 지금까지의 대화를 함께 보낸다 (/chat은 무상태).
    const history: Turn[] = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setInput("");
    setError(null);
    setLoading(true);
    try {
      const data = await askQuestion(trimmed, history);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, data },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      // 실패한 질문은 대화에서 되돌린다 — 남겨두면 다음 요청의 맥락을 오염시킨다.
      setMessages((prev) => prev.slice(0, -1));
      setInput(trimmed);
    } finally {
      setLoading(false);
    }
  }

  const empty = messages.length === 0;

  return (
    <main>
      <nav className="nav">
        <span className="nav-current">상담</span>
        <Link href="/factcheck">조언 검증</Link>
        {!empty && (
          <button
            type="button"
            className="chip nav-reset"
            disabled={loading}
            onClick={() => {
              setMessages([]);
              setError(null);
              setInput("");
            }}
          >
            새 상담
          </button>
        )}
      </nav>

      {empty && (
        <>
          <h1>반려동물 훈련·문제행동 상담</h1>
          <p className="subtitle">
            기관·학술 자료(AVSAB, AAHA, RSPCA, VCA, PMC 오픈액세스)만 근거로
            씁니다. 자료에 없으면 없다고 말합니다.
          </p>
          <div className="examples">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                className="chip"
                disabled={loading}
                onClick={() => void send(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </>
      )}

      <div className="thread">
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="msg-user">
              {m.content}
            </div>
          ) : (
            <AssistantMessage key={i} data={m.data} />
          ),
        )}
        {loading && <div className="msg-typing">답변을 찾는 중…</div>}
        <div ref={bottomRef} />
      </div>

      {error && <div className="error">{error}</div>}

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            empty ? "예: 강아지가 초인종 소리에 계속 짖어요" : "메시지 입력"
          }
          rows={2}
          onKeyDown={(e) => {
            // Enter로 보내고 Shift+Enter로 줄바꿈 — 채팅 관습.
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              void send(input);
            }
          }}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          보내기
        </button>
      </form>
    </main>
  );
}

function AssistantMessage({ data }: { data: ChatResponse }) {
  const notice = NOTICE[data.coverage];
  return (
    <div className="msg-bot">
      {isStubAnswer(data.answer) && (
        <div className="banner">
          <strong>LLM 미연결</strong> — 아래 답변은 자리표시자입니다.
        </div>
      )}
      {notice && (
        <div className="banner">
          <strong>{notice.label}</strong> — {data.coverage_note ?? notice.text}
        </div>
      )}

      <div className="msg-body">{data.answer}</div>

      {data.sources.length > 0 && (
        <details className="msg-sources">
          <summary>
            근거 {data.sources.length}건 · {data.latency_ms}ms
          </summary>
          <div>
            {data.sources.map((s, j) => (
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
    </div>
  );
}
