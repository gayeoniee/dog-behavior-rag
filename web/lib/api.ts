/**
 * FastAPI 응답 타입.
 *
 * app/schemas/chat.py 와 1:1로 맞춘다 — 그쪽 독스트링이 이걸 "안드로이드 앱과의
 * API 계약"이라고 부르고, 이 화면도 같은 계약을 쓴다. 필드를 바꾸려면 양쪽을 같이 고칠 것.
 */

export type SourceChunk = {
  chunk_id: number;
  document_title: string;
  content: string;
  /** 코사인 유사도 0~1. 권위 부스팅은 순위에만 쓰이고 이 값에는 반영되지 않는다. */
  score: number;
  source: string | null;
};

/**
 * 근거 충분도. none / needs_detail 둘 다 sources가 비어 있지만 의미가 다르다.
 *   none          질문이 서비스 범위 밖 (고양이, 가격, 장소)
 *   needs_detail  개 행동 질문은 맞는데 정보가 부족해 되묻는 중
 */
export type Coverage = "full" | "partial" | "none" | "needs_detail";

export type ChatResponse = {
  answer: string;
  sources: SourceChunk[];
  latency_ms: number;
  provider: string;
  coverage: Coverage;
  coverage_note: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

/** 서버로 보내는 직전 대화. 되묻기에 "1번이요"로 답할 수 있게 한다. */
export type Turn = { role: "user" | "assistant"; content: string };

export async function askQuestion(
  question: string,
  history: Turn[] = [],
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // 서버도 최근 6개만 쓰지만 보내는 쪽에서도 잘라 페이로드를 줄인다.
    body: JSON.stringify({ question, history: history.slice(-6) }),
    signal,
  });

  if (!response.ok) {
    // 503은 "아직 준비 안 됨"이라 서버 버그와 구분해서 안내한다.
    // 임베딩 모델 미로딩(uv sync --extra hf)이나 DB 미기동이 대부분이다.
    if (response.status === 503) {
      throw new Error(
        "검색 서비스가 준비되지 않았습니다. 임베딩 모델과 DB가 떠 있는지 확인하세요.",
      );
    }
    throw new Error(`API 오류 ${response.status}`);
  }
  return response.json();
}

/** answer가 아직 LLM이 아니라 스텁인지. 화면이 거짓말하지 않게 하려고 확인한다. */
export function isStubAnswer(answer: string): boolean {
  return answer.trimStart().startsWith("[stub]");
}

// ── 팩트체크 ─────────────────────────────────────────────────────────

/** app/schemas/factcheck.py의 Verdict와 1:1. not_covered가 반드시 있어야 한다. */
export type Verdict = "supported" | "contradicted" | "not_covered";

export type ClaimVerdict = {
  claim: string;
  verdict: Verdict;
  explanation: string;
  sources: SourceChunk[];
};

export type FactCheckResponse = {
  claims: ClaimVerdict[];
  corpus_note: string;
  latency_ms: number;
  provider: string;
};

export async function factCheck(
  text: string,
  signal?: AbortSignal,
): Promise<FactCheckResponse> {
  const response = await fetch(`${API_BASE}/api/v1/factcheck`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal,
  });

  if (!response.ok) {
    if (response.status === 503) {
      throw new Error(
        "검증 서비스가 준비되지 않았습니다. LLM과 DB가 떠 있는지 확인하세요.",
      );
    }
    throw new Error(`API 오류 ${response.status}`);
  }
  return response.json();
}
