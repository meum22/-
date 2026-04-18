"""
멀티세션 RAG 챗봇 — Supabase 세션/벡터 저장, 스트리밍 답변, 세션 관리 UI.
실행: streamlit run multi-session-ref.py (7.MultiService/code 디렉터리에서)
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# 경로 · 환경
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]  # .../AI-Education
_ENV_PATH = _PROJECT_ROOT / ".env"
_LOG_DIR = _PROJECT_ROOT / "logs"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
VECTOR_BATCH = 10
RETRIEVAL_K = 10


def _setup_logging() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"
    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(fh)


def load_env() -> None:
    load_dotenv(dotenv_path=_ENV_PATH, override=False)


def get_env_status() -> dict[str, bool]:
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "supabase_url": bool(os.getenv("SUPABASE_URL")),
        "supabase_key": bool(os.getenv("SUPABASE_ANON_KEY")),
    }


def get_supabase() -> Client | None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def _chunk_text(chunk: Any) -> str:
    c = getattr(chunk, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for x in c:
            if isinstance(x, dict) and "text" in x:
                parts.append(str(x["text"]))
            else:
                parts.append(str(x))
        return "".join(parts)
    return ""


def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~[^~]+~~", "", text)
    text = re.sub(r"^[\s]*[-_=]{3,}[\s]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def inject_css() -> None:
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div[data-testid="stChatMessage"] { padding: 0.75rem 0; }
.stButton > button {
    background-color: #ff69b4 !important;
    color: #fff !important;
    border: none !important;
}
.stButton > button:hover { filter: brightness(0.95); }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    logo_paths = [
        _PROJECT_ROOT / "logo.png",
        _PROJECT_ROOT / "5.DatabaseSQL" / "code" / "logo.png",
    ]
    logo = next((p for p in logo_paths if p.is_file()), None)
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if logo:
            st.image(str(logo), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<div style="text-align:center;">
  <span style="font-size:4rem !important; font-weight:700;">
    <span style="color:#1f77b4 !important;">멀티세션</span>
    <span style="color:#ffd700 !important;"> RAG 챗봇</span>
  </span>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def get_llm(model_name: str = CHAT_MODEL, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        streaming=True,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=os.getenv("OPENAI_API_KEY"))


def list_sessions(supabase: Client) -> list[dict[str, Any]]:
    r = (
        supabase.table("sessions")
        .select("id,title,created_at,updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return r.data or []


def insert_session(supabase: Client, title: str) -> str:
    r = supabase.table("sessions").insert({"title": title}).execute()
    if not r.data:
        raise RuntimeError("세션 생성에 실패했습니다.")
    return r.data[0]["id"]


def replace_messages(supabase: Client, session_id: str, chat_history: list[dict]) -> None:
    supabase.table("chat_messages").delete().eq("session_id", session_id).execute()
    rows = [
        {"session_id": session_id, "role": m["role"], "content": m["content"]}
        for m in chat_history
    ]
    if rows:
        supabase.table("chat_messages").insert(rows).execute()


def load_messages(supabase: Client, session_id: str) -> list[dict]:
    r = (
        supabase.table("chat_messages")
        .select("role,content,created_at")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return [{"role": x["role"], "content": x["content"]} for x in (r.data or [])]


def delete_session_db(supabase: Client, session_id: str) -> None:
    supabase.table("sessions").delete().eq("id", session_id).execute()


def insert_vector_rows(
    supabase: Client,
    session_id: str,
    file_name: str,
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
) -> None:
    texts = [d.page_content for d in docs]
    for i in range(0, len(texts), VECTOR_BATCH):
        batch_docs = docs[i : i + VECTOR_BATCH]
        batch_texts = texts[i : i + VECTOR_BATCH]
        vectors = embeddings.embed_documents(batch_texts)
        rows = []
        for doc, vec in zip(batch_docs, vectors):
            meta = dict(doc.metadata or {})
            meta.setdefault("source", file_name)
            rows.append(
                {
                    "session_id": session_id,
                    "file_name": file_name,
                    "content": doc.page_content,
                    "metadata": meta,
                    "embedding": vec,
                }
            )
        supabase.table("vector_documents").insert(rows).execute()


def duplicate_vectors_to_session(
    supabase: Client, from_session_id: str, to_session_id: str
) -> None:
    r = (
        supabase.table("vector_documents")
        .select("file_name,content,metadata,embedding")
        .eq("session_id", from_session_id)
        .execute()
    )
    rows_in = r.data or []
    if not rows_in:
        return
    out: list[dict] = []
    for row in rows_in:
        emb = row["embedding"]
        if isinstance(emb, str):
            import json

            emb = json.loads(emb)
        out.append(
            {
                "session_id": to_session_id,
                "file_name": row["file_name"],
                "content": row["content"],
                "metadata": row.get("metadata") or {},
                "embedding": emb,
            }
        )
    for i in range(0, len(out), VECTOR_BATCH):
        supabase.table("vector_documents").insert(out[i : i + VECTOR_BATCH]).execute()


def retrieve_by_rpc(
    supabase: Client,
    session_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int = RETRIEVAL_K,
) -> list[Document]:
    q_emb = embeddings.embed_query(query)
    try:
        r = supabase.rpc(
            "match_vector_documents",
            {
                "query_embedding": q_emb,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        data = r.data or []
    except Exception as e:
        logging.warning("RPC match_vector_documents 실패, 폴백 사용: %s", e)
        return retrieve_fallback(supabase, session_id, query, embeddings, k)

    return [
        Document(
            page_content=row["content"],
            metadata={
                **(row.get("metadata") or {}),
                "file_name": row.get("file_name"),
                "similarity": row.get("similarity"),
            },
        )
        for row in data
    ]


def retrieve_fallback(
    supabase: Client,
    session_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int,
) -> list[Document]:
    r = (
        supabase.table("vector_documents")
        .select("content,metadata,file_name,embedding")
        .eq("session_id", session_id)
        .execute()
    )
    rows = r.data or []
    if not rows:
        return []
    q = np.array(embeddings.embed_query(query), dtype=np.float64)
    scored: list[tuple[float, dict]] = []
    for row in rows:
        emb = row["embedding"]
        if isinstance(emb, str):
            import json

            emb = json.loads(emb)
        v = np.array(emb, dtype=np.float64)
        sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
        scored.append((sim, row))
    scored.sort(key=lambda x: -x[0])
    out: list[Document] = []
    for sim, row in scored[:k]:
        out.append(
            Document(
                page_content=row["content"],
                metadata={
                    **(row.get("metadata") or {}),
                    "file_name": row.get("file_name"),
                    "similarity": sim,
                },
            )
        )
    return out


def list_vector_file_names(supabase: Client, session_id: str) -> list[str]:
    r = (
        supabase.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names = sorted({row["file_name"] for row in (r.data or []) if row.get("file_name")})
    return names


def generate_session_title(openai_client: OpenAI, first_q: str, first_a: str) -> str:
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "첫 사용자 질문과 첫 답변을 바탕으로 짧은 세션 제목을 한국어로 한 줄만 작성하세요. 40자 이내, 따옴표나 접두어 없이.",
            },
            {"role": "user", "content": f"질문:\n{first_q}\n\n답변:\n{first_a}"},
        ],
        temperature=0.4,
        max_tokens=80,
    )
    title = (resp.choices[0].message.content or "").strip()
    return title[:120] if title else "새 세션"


def generate_followup_questions(openai_client: OpenAI, question: str, answer: str) -> str:
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "사용자 질문과 답변을 읽고, 이어서 물어볼 만한 질문을 정확히 3개만 번호 목록으로 한국어 존댓말로 작성하세요.",
            },
            {"role": "user", "content": f"질문:\n{question}\n\n답변:\n{answer[:8000]}"},
        ],
        temperature=0.6,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()


def build_system_prompt_with_rag(context_docs: list[Document]) -> str:
    base = (
        "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
        "내용은 생략하지 말고, 모르면 모른다고 말해줘. 말투는 존대말 한글로 해줘.\n"
        "답변은 반드시 # ## ### 헤딩으로 구조화하고, 구분선(---)과 취소선(~~)은 쓰지 마."
    )
    if not context_docs:
        return base
    blocks = []
    for i, d in enumerate(context_docs, 1):
        fn = d.metadata.get("file_name", "문서")
        blocks.append(f"[{i}] ({fn})\n{d.page_content}")
    ctx = "\n\n".join(blocks)
    return base + "\n\n다음은 참고 문서 발췌입니다:\n" + ctx


def init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "working_session_id" not in st.session_state:
        st.session_state.working_session_id = None
    if "session_options" not in st.session_state:
        st.session_state.session_options = []
    if "selected_session_id" not in st.session_state:
        st.session_state.selected_session_id = None
    if "last_loaded_session_id" not in st.session_state:
        st.session_state.last_loaded_session_id = None


def reset_screen() -> None:
    st.session_state.chat_history = []
    st.session_state.working_session_id = None
    st.session_state.selected_session_id = None
    st.session_state.last_loaded_session_id = None


def main() -> None:
    _setup_logging()
    load_env()
    init_session_state()

    st.set_page_config(
        page_title="멀티세션 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    inject_css()
    render_header()

    env = get_env_status()
    if not all(env.values()):
        st.error(
            "환경 변수가 부족합니다. AI-Education/.env 에 OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY 를 설정하세요."
        )
        st.text(f"OPENAI_API_KEY: {'설정됨' if env['openai'] else '누락'}")
        st.text(f"SUPABASE_URL: {'설정됨' if env['supabase_url'] else '누락'}")
        st.text(f"SUPABASE_ANON_KEY: {'설정됨' if env['supabase_key'] else '누락'}")
        return

    supabase = get_supabase()
    if supabase is None:
        st.error("Supabase 클라이언트를 만들 수 없습니다.")
        return

    embeddings = get_embeddings()
    llm = get_llm(CHAT_MODEL)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 사이드바
    with st.sidebar:
        st.markdown("### LLM 모델")
        model_choice = st.radio(
            "모델",
            options=["gpt-4o-mini"],
            index=0,
            horizontal=True,
        )
        _ = model_choice  # 고정 gpt-4o-mini

        st.markdown("### 세션 관리")
        try:
            st.session_state.session_options = list_sessions(supabase)
        except Exception as e:
            st.warning(f"세션 목록을 불러오지 못했습니다: {e}")
            st.session_state.session_options = []

        labels = ["(선택 없음)"] + [
            f"{s.get('title', '제목 없음')} — {str(s['id'])[:8]}…"
            for s in st.session_state.session_options
        ]
        ids = [None] + [s["id"] for s in st.session_state.session_options]

        def on_session_change() -> None:
            idx = st.session_state.get("_sess_sel_idx", 0)
            if idx > 0 and idx < len(ids):
                sid = ids[idx]
                st.session_state.selected_session_id = sid
                try:
                    st.session_state.chat_history = load_messages(supabase, sid)
                    st.session_state.working_session_id = sid
                    st.session_state.last_loaded_session_id = sid
                except Exception as e:
                    st.session_state["_sess_load_err"] = str(e)

        sel_idx = st.selectbox(
            "저장된 세션",
            range(len(labels)),
            format_func=lambda i: labels[i],
            key="_sess_sel_idx",
            on_change=on_session_change,
        )
        if "_sess_load_err" in st.session_state:
            st.warning(st.session_state.pop("_sess_load_err"))

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("세션로드", width="stretch"):
                if sel_idx > 0:
                    sid = ids[sel_idx]
                    try:
                        st.session_state.chat_history = load_messages(supabase, sid)
                        st.session_state.working_session_id = sid
                        st.session_state.last_loaded_session_id = sid
                        st.success("세션을 불러왔습니다.")
                    except Exception as e:
                        st.error(str(e))
        with bcol2:
            if st.button("세션저장", width="stretch"):
                hist = st.session_state.chat_history
                if len(hist) < 2:
                    st.warning("저장할 대화(질문·답변)가 충분하지 않습니다.")
                else:
                    first_u = next((m["content"] for m in hist if m["role"] == "user"), "")
                    first_a = next(
                        (m["content"] for m in hist if m["role"] == "assistant"),
                        "",
                    )
                    if not first_u or not first_a:
                        st.warning("첫 질문과 첫 답변이 필요합니다.")
                    else:
                        try:
                            title = generate_session_title(openai_client, first_u, first_a)
                            new_id = insert_session(supabase, title)
                            replace_messages(supabase, new_id, hist)
                            wid = st.session_state.working_session_id
                            if wid:
                                duplicate_vectors_to_session(supabase, wid, new_id)
                            st.session_state.working_session_id = new_id
                            st.session_state.session_options = list_sessions(supabase)
                            st.success(f"새 세션이 저장되었습니다: {title}")
                        except Exception as e:
                            st.error(str(e))

        if st.button("세션삭제", width="stretch"):
            if sel_idx <= 0:
                st.warning("삭제할 세션을 선택하세요.")
            else:
                sid = ids[sel_idx]
                try:
                    delete_session_db(supabase, sid)
                    if st.session_state.working_session_id == sid:
                        reset_screen()
                    st.session_state.session_options = list_sessions(supabase)
                    st.success("세션이 삭제되었습니다.")
                except Exception as e:
                    st.error(str(e))

        if st.button("화면초기화", width="stretch"):
            reset_screen()
            st.success("화면을 초기화했습니다.")

        if st.button("vectordb", width="stretch"):
            sid = st.session_state.working_session_id
            if not sid:
                st.info("현재 연결된 세션이 없습니다. PDF 처리 또는 세션을 선택하세요.")
            else:
                try:
                    names = list_vector_file_names(supabase, sid)
                    st.text("vectordb 파일명:\n" + ("\n".join(names) if names else "(없음)"))
                except Exception as e:
                    st.error(str(e))

        st.markdown("### PDF (RAG)")
        uploads = st.file_uploader(
            "PDF 업로드 (다중)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기", width="stretch"):
            if not uploads:
                st.warning("PDF 파일을 선택하세요.")
            else:
                try:
                    if st.session_state.working_session_id is None:
                        sid = insert_session(
                            supabase,
                            f"PDF-{datetime.now().strftime('%Y%m%d %H:%M')}",
                        )
                        st.session_state.working_session_id = sid
                    sid = st.session_state.working_session_id
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                    for up in uploads:
                        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
                        try:
                            os.write(fd, up.getvalue())
                        finally:
                            os.close(fd)
                        try:
                            pages = PyPDFLoader(tmp_path).load()
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                        for d in pages:
                            d.metadata["file_name"] = up.name
                        parts = splitter.split_documents(pages)
                        insert_vector_rows(supabase, sid, up.name, parts, embeddings)
                    st.session_state.session_options = list_sessions(supabase)
                    st.success("파일 처리 및 벡터 저장이 완료되었습니다.")
                except Exception as e:
                    logging.exception("PDF 처리 오류")
                    st.error(str(e))

        st.markdown("### 현재 설정")
        st.text(
            f"모델: {CHAT_MODEL}\n"
            f"임베딩: {EMBEDDING_MODEL}\n"
            f"작업 세션 ID: {st.session_state.working_session_id or '(없음)'}\n"
            f"대화 메시지 수: {len(st.session_state.chat_history)}"
        )

    # 메인 채팅
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(remove_separators(m["content"]), unsafe_allow_html=True)

    user_q = st.chat_input("질문을 입력하세요")
    if user_q:
        if st.session_state.working_session_id is None:
            try:
                st.session_state.working_session_id = insert_session(
                    supabase,
                    f"채팅-{datetime.now().strftime('%Y%m%d %H:%M')}",
                )
                st.session_state.session_options = list_sessions(supabase)
            except Exception as e:
                st.error(f"세션을 시작하지 못했습니다: {e}")
                return

        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        sid = st.session_state.working_session_id
        docs: list[Document] = []
        if sid:
            try:
                docs = retrieve_by_rpc(supabase, sid, user_q, embeddings)
            except Exception as e:
                logging.warning("검색 오류: %s", e)

        system_text = build_system_prompt_with_rag(docs)
        memory = st.session_state.chat_history[:-1][-50:]
        lc_messages: list[Any] = [SystemMessage(content=system_text)]
        for turn in memory:
            if turn["role"] == "user":
                lc_messages.append(HumanMessage(content=turn["content"]))
            else:
                lc_messages.append(AIMessage(content=turn["content"]))
        lc_messages.append(HumanMessage(content=user_q))

        answer_buf: list[str] = []
        with st.chat_message("assistant"):
            place = st.empty()
            try:
                for chunk in llm.stream(lc_messages):
                    piece = _chunk_text(chunk)
                    if piece:
                        answer_buf.append(piece)
                        place.markdown(
                            remove_separators("".join(answer_buf)) + "▌",
                            unsafe_allow_html=True,
                        )
                main_answer = remove_separators("".join(answer_buf))
                follow = generate_followup_questions(openai_client, user_q, main_answer)
                tail = "\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n" + follow
                full = main_answer + tail
                place.markdown(full, unsafe_allow_html=True)
            except Exception as e:
                logging.exception("스트리밍 오류")
                err = f"오류가 발생했습니다: {e}"
                place.markdown(err)
                full = err

        st.session_state.chat_history.append({"role": "assistant", "content": full})

        if sid:
            try:
                replace_messages(supabase, sid, st.session_state.chat_history)
                supabase.table("sessions").update(
                    {"updated_at": datetime.now(timezone.utc).isoformat()}
                ).eq("id", sid).execute()
            except Exception as e:
                logging.warning("자동 저장 실패: %s", e)


if __name__ == "__main__":
    main()
