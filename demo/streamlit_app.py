"""테스트용 Streamlit 데모.

FastAPI를 HTTP로 호출하는 얇은 클라이언트일 뿐이다.
나중에 안드로이드 네이티브 앱을 붙일 때도 같은 API를 그대로 쓰면 된다.

실행:
    uv run uvicorn app.main:app --reload          # 터미널 1
    uv run streamlit run demo/streamlit_app.py    # 터미널 2
"""

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="GY_RAG — 반려동물 훈련 상담", page_icon="🐶")
st.title("🐶 반려동물 훈련 상담")
st.caption(f"API: {API_BASE_URL}")

with st.sidebar:
    top_k = st.slider("검색할 문서 수 (top_k)", min_value=1, max_value=20, value=5)
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for src in msg.get("sources", []):
            with st.expander(f"📄 {src['document_title']} (유사도 {src['score']:.3f})"):
                st.write(src["content"])

question = st.chat_input("궁금한 점을 물어보세요 (예: 강아지가 초인종에 짖어요)")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/chat",
                json={"question": question, "top_k": top_k},
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            st.error(f"API 호출 실패: {exc}\n\nFastAPI 서버가 떠 있는지 확인하세요.")
        else:
            data = resp.json()
            st.markdown(data["answer"])
            st.caption(f"{data['provider']} · {data['latency_ms']}ms")

            for src in data["sources"]:
                with st.expander(f"📄 {src['document_title']} (유사도 {src['score']:.3f})"):
                    st.write(src["content"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"],
                }
            )
