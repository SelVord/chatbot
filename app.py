"""
app.py — RAG Chatbot (Streamlit)
Run: streamlit run app.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Make data/core/ importable as the `core` package
_data_dir = Path(__file__).parent / "data"
if str(_data_dir) not in sys.path:
    sys.path.insert(0, str(_data_dir))

import streamlit as st

import config
import core.database as db
import core.document_processor as dp
import core.vector_store as vs
from core.rag_chain import ask_stream

# ══════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ── */
body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] * {
    color: #c9cdd8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e8eaf0 !important;
}

/* ── Inputs ── */
input, textarea, select,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #1e2433 !important;
    color: #e8eaf0 !important;
    border: 1px solid #3a3f52 !important;
    border-radius: 6px !important;
}
input::placeholder, textarea::placeholder {
    color: #6b7280 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: #1a1f2e;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
    margin-bottom: 8px;
    padding: 4px 8px;
    color: #e8eaf0 !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #e8eaf0 !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #1e2433 !important;
    color: #e8eaf0 !important;
    border: 1px solid #3a3f52 !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #2a3a5c;
    color: #e8eaf0 !important;
    border: 1px solid #3a4f7a;
    border-radius: 6px;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: #3a4f7a;
    border-color: #5b7abf;
}
.stButton > button[kind="primary"] {
    background-color: #2563eb;
    border-color: #3b82f6;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1d4ed8;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #1e2433 !important;
    color: #e8eaf0 !important;
    border: 1px solid #3a3f52 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background-color: #1a1f2e;
    border: 1px solid #2a2f3e;
    border-radius: 8px;
}
[data-testid="stExpander"] summary {
    color: #a0b0cc !important;
}

/* ── Divider ── */
hr {
    border-color: #2a2f3e !important;
}

/* ── Source cards ── */
.source-card {
    background: #1e2a3e;
    border-left: 3px solid #3b82f6;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 4px 0;
    font-size: 0.82em;
    color: #c0cce0;
}
.source-card b {
    color: #90b4e8;
}
.score-badge {
    display: inline-block;
    background: #2563eb;
    color: #ffffff !important;
    border-radius: 10px;
    padding: 1px 8px;
    font-size: 0.73em;
    margin-left: 6px;
}
.snippet-text {
    color: #8899aa;
    font-size: 0.88em;
}

/* ── Doc chips ── */
.doc-chip {
    display: inline-block;
    background: #1a2e1e;
    border: 1px solid #2d5a35;
    color: #7ec897 !important;
    border-radius: 12px;
    padding: 2px 10px;
    margin: 2px;
    font-size: 0.78em;
}

/* ── Info / warning / success ── */
[data-testid="stAlert"] {
    background-color: #1e2433 !important;
    border-radius: 8px;
}

/* ── Session header ── */
.session-header {
    font-size: 1.15em;
    font-weight: 700;
    color: #e8eaf0;
    padding: 4px 0;
}
.session-sub {
    color: #6b7280;
    font-size: 0.85em;
    font-weight: 400;
}

/* ── Status dot ── */
.status-ready   { color: #4ade80; font-size: 1.1em; }
.status-noindex { color: #f87171; font-size: 1.1em; }

/* ── Tab ── */
[data-testid="stTab"] button {
    color: #a0b0cc !important;
}
[data-testid="stTab"] button[aria-selected="true"] {
    color: #e8eaf0 !important;
    border-bottom-color: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# State init
# ══════════════════════════════════════════════════════════════════════
db.init_db()

for key, default in [
    ("session_id", None),
    ("vector_store", None),
    ("messages", []),
    ("relevance_threshold", config.RELEVANCE_THRESHOLD),
    ("editing_doc", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def load_session(session_id: int) -> None:
    st.session_state.session_id = session_id
    st.session_state.messages = db.get_messages(session_id)
    session = db.get_session(session_id)
    emb = session.get("embedding_provider", config.EMBEDDING_PROVIDER)
    if vs.has_vector_store(session_id):
        try:
            st.session_state.vector_store = vs.load_vector_store(session_id, embedding_provider=emb)
        except Exception as e:
            st.warning(f"Could not load index: {e}")
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None


def current_session() -> dict | None:
    if st.session_state.session_id is None:
        return None
    return db.get_session(st.session_state.session_id)


def index_chunks(chunks, session_id, session):
    """Add chunks to FAISS and update session state."""
    store = vs.add_to_vector_store(
        session_id, chunks, embedding_provider=session["embedding_provider"]
    )
    st.session_state.vector_store = store


def get_ollama_models() -> list[str]:
    """Fetch available Ollama models from the local server, without :latest suffix."""
    try:
        import requests
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [
                m["name"].removesuffix(":latest")
                for m in resp.json().get("models", [])
            ]
            if models:
                return models
    except Exception:
        pass
    return [config.OLLAMA_MODEL]


def rebuild_session_index(session_id: int, session: dict) -> None:
    """Rebuild the FAISS index from all saved files for a session."""
    docs = db.get_documents(session_id)
    all_chunks = []
    for doc in docs:
        file_path = config.UPLOADS_DIR / doc["filename"]
        if file_path.exists():
            try:
                chunks, _ = dp.load_and_split(str(file_path))
                all_chunks.extend(chunks)
            except Exception:
                pass
    vs.delete_vector_store(session_id)
    if all_chunks:
        store = vs.build_vector_store(
            session_id, all_chunks,
            embedding_provider=session["embedding_provider"],
        )
        st.session_state.vector_store = store
    else:
        st.session_state.vector_store = None


# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🤖 RAG Chatbot")
    st.divider()

    # ── Sessions ───────────────────────────────────────────────────────
    st.markdown("### 📁 Sessions")

    sessions = db.list_sessions()
    sid_to_name = {s["id"]: s["name"] for s in sessions}

    col_name, col_add = st.columns([3, 1])
    with col_name:
        new_name = st.text_input("New session name", placeholder="My Bot", label_visibility="collapsed")
    with col_add:
        if st.button("➕", help="Create session", use_container_width=True):
            if new_name.strip():
                _default_model = (
                    config.OPENAI_MODEL
                    if config.LLM_PROVIDER == "openai"
                    else config.OLLAMA_MODEL
                )
                sid = db.create_session(
                    name=new_name.strip(),
                    llm_provider=config.LLM_PROVIDER,
                    llm_model=_default_model,
                    embedding_provider=config.EMBEDDING_PROVIDER,
                )
                load_session(sid)
                st.rerun()
            else:
                st.error("Enter a session name")

    if sessions:
        # None option keeps the app in "no session" state on first load
        _options = [None] + [s["id"] for s in sessions]
        if st.session_state.session_id is None:
            _cur = 0
        else:
            _cur = next(
                (i + 1 for i, s in enumerate(sessions) if s["id"] == st.session_state.session_id),
                0,
            )
        selected_id = st.selectbox(
            "Session",
            options=_options,
            format_func=lambda i: "— select a session —" if i is None else sid_to_name[i],
            index=_cur,
            label_visibility="collapsed",
        )
        if selected_id is not None and selected_id != st.session_state.session_id:
            load_session(selected_id)
            st.rerun()

    # ── Bot config ─────────────────────────────────────────────────────
    if st.session_state.session_id:
        session = current_session()
        st.divider()
        st.markdown("### ⚙️ Bot Settings")

        bot_name = st.text_input("Bot name", value=session["bot_name"])
        system_prompt = st.text_area(
            "System prompt",
            value=session["system_prompt"],
            height=110,
            help="Define the bot's role, tone, and knowledge domain.",
        )

        with st.expander("🔧 Model settings"):
            llm_provider = st.selectbox(
                "LLM provider",
                ["openai", "ollama"],
                index=0 if session["llm_provider"] == "openai" else 1,
            )
            # provider just changed → reset the saved model so we never
            # carry an OpenAI model name into Ollama or vice-versa
            _provider_changed = llm_provider != session["llm_provider"]
            if llm_provider == "openai":
                _openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
                _saved = session["llm_model"] if not _provider_changed else "gpt-4o-mini"
                _model_idx = _openai_models.index(_saved) if _saved in _openai_models else 0
                llm_model = st.selectbox("Model", _openai_models, index=_model_idx)
            else:
                _ollama_models = get_ollama_models()
                _saved = session["llm_model"] if not _provider_changed else _ollama_models[0]
                _model_idx = _ollama_models.index(_saved) if _saved in _ollama_models else 0
                llm_model = st.selectbox("Model", _ollama_models, index=_model_idx)
            emb_provider = st.selectbox(
                "Embeddings",
                ["openai", "local"],
                index=0 if session["embedding_provider"] == "openai" else 1,
                help="local = free, runs on CPU via sentence-transformers",
            )
            st.session_state.relevance_threshold = st.slider(
                "Relevance threshold",
                min_value=0.0, max_value=1.0,
                value=st.session_state.relevance_threshold,
                step=0.05,
                help="Below this score the bot says 'I don't know'.",
            )

        if st.button("💾 Save settings", use_container_width=True, type="primary"):
            db.update_session(
                st.session_state.session_id,
                bot_name=bot_name,
                system_prompt=system_prompt,
                llm_provider=llm_provider,
                llm_model=llm_model,
                embedding_provider=emb_provider,
            )
            from core import llm as _llm, embeddings as _emb
            _llm.get_llm.cache_clear()
            _emb.get_embeddings.cache_clear()
            st.success("Settings saved!")
            st.rerun()

        # ── Documents ──────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📄 Knowledge Base")

        docs = db.get_documents(st.session_state.session_id)
        if docs:
            for doc in docs:
                _fp = config.UPLOADS_DIR / doc["filename"]
                _editable = doc["filename"].endswith(".txt") and _fp.exists()
                if _editable:
                    _c1, _c2 = st.columns([5, 1])
                    _c1.markdown(
                        f'<span class="doc-chip">📎 {doc["filename"]} '
                        f'({doc["page_count"]}p · {doc["chunk_count"]} chunks)</span>',
                        unsafe_allow_html=True,
                    )
                    if _c2.button("✏️", key=f"edit_{doc['id']}", help="View / Edit"):
                        st.session_state.editing_doc = (
                            None if st.session_state.editing_doc == doc["id"] else doc["id"]
                        )
                        st.rerun()
                else:
                    st.markdown(
                        f'<span class="doc-chip">📎 {doc["filename"]} '
                        f'({doc["page_count"]}p · {doc["chunk_count"]} chunks)</span>',
                        unsafe_allow_html=True,
                    )
                # Inline editor (shown when this doc is selected)
                if st.session_state.editing_doc == doc["id"] and _editable:
                    _current = _fp.read_text(encoding="utf-8")
                    _new = st.text_area(
                        f"Editing: {doc['filename']}",
                        value=_current,
                        height=220,
                        key=f"ta_{doc['id']}",
                    )
                    _s1, _s2 = st.columns(2)
                    if _s1.button("💾 Save & Re-index", key=f"save_{doc['id']}", type="primary", use_container_width=True):
                        _fp.write_text(_new, encoding="utf-8")
                        rebuild_session_index(st.session_state.session_id, current_session())
                        st.session_state.editing_doc = None
                        st.success("Saved and re-indexed!")
                        st.rerun()
                    if _s2.button("✖ Cancel", key=f"cancel_{doc['id']}", use_container_width=True):
                        st.session_state.editing_doc = None
                        st.rerun()
        else:
            st.caption("No documents yet.")

        upload_mode = st.radio(
            "Add knowledge",
            ["📁 Upload file", "✏️ Paste text"],
            horizontal=True,
            label_visibility="collapsed",
            key="upload_mode",
        )

        if upload_mode == "📁 Upload file":
            uploaded_files = st.file_uploader(
                "Upload files",
                type=["pdf", "txt", "docx"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if uploaded_files and st.button("📥 Index files", use_container_width=True, type="primary"):
                session = current_session()
                progress = st.progress(0, text="Indexing…")
                total = len(uploaded_files)
                ok_count = 0
                errors = []
                for i, uf in enumerate(uploaded_files):
                    try:
                        file_path = dp.save_uploaded_file(uf, uf.name)
                        chunks, page_count = dp.load_and_split(file_path)
                        index_chunks(chunks, st.session_state.session_id, session)
                        db.save_document(
                            st.session_state.session_id,
                            uf.name, page_count, len(chunks),
                        )
                        ok_count += 1
                    except Exception as e:
                        errors.append(f"**{uf.name}**: {e}")
                    progress.progress((i + 1) / total, text=f"Indexed {i+1}/{total}")
                progress.empty()
                for err in errors:
                    st.error(err)
                if ok_count:
                    st.success(f"✅ Indexed {ok_count} file(s)!")
                    st.rerun()

        else:
            paste_label = st.text_input(
                "Source label (optional)",
                placeholder="e.g. Company FAQ",
                key="paste_label",
            )
            paste_text = st.text_area(
                "Paste your text here",
                height=150,
                placeholder="Paste any text you want the bot to learn from…",
                key="paste_text",
            )
            if st.button("📥 Index text", use_container_width=True, type="primary"):
                if paste_text.strip():
                    session = current_session()
                    label = paste_label.strip() or "pasted_text"
                    filename = label if label.endswith(".txt") else f"{label}.txt"
                    try:
                        with st.spinner("Indexing text…"):
                            # Save to disk so the text is editable later
                            _text_path = config.UPLOADS_DIR / filename
                            _text_path.write_text(paste_text.strip(), encoding="utf-8")
                            chunks, pages = dp.load_and_split_text(paste_text.strip(), source_name=label)
                            index_chunks(chunks, st.session_state.session_id, session)
                            db.save_document(
                                st.session_state.session_id,
                                filename, pages, len(chunks),
                            )
                        st.success(f"✅ Indexed {len(chunks)} chunks!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please paste some text first.")

        # ── Export bot ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📦 Export Bot")
        if vs.has_vector_store(st.session_state.session_id):
            st.caption(
                "Downloads a ready-to-deploy package: FastAPI server, "
                "chat widget, FAISS index, and setup instructions."
            )
            _exp_session = current_session()
            _exp_name = "".join(
                c if c.isalnum() else "_"
                for c in _exp_session["name"]
            ).lower()
            from core.export import create_export_zip
            st.download_button(
                "⬇️ Download export package (ZIP)",
                data=create_export_zip(st.session_state.session_id, _exp_session),
                file_name=f"{_exp_name}_bot.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary",
            )
        else:
            st.caption("Index at least one document to enable export.")

        # ── Danger zone ────────────────────────────────────────────────
        st.divider()
        with st.expander("🗑️ Danger zone"):
            if st.button("Clear chat history", use_container_width=True):
                db.clear_messages(st.session_state.session_id)
                st.session_state.messages = []
                st.rerun()

            if st.button("Delete this session", use_container_width=True):
                vs.delete_vector_store(st.session_state.session_id)
                db.delete_session(st.session_state.session_id)
                st.session_state.session_id = None
                st.session_state.vector_store = None
                st.session_state.messages = []
                st.rerun()

            if st.session_state.messages:
                export_data = db.export_messages_json(st.session_state.session_id)
                st.download_button(
                    "⬇️ Export chat (JSON)",
                    data=export_data,
                    file_name=f"chat_{st.session_state.session_id}_{datetime.now():%Y%m%d}.json",
                    mime="application/json",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════
# Main chat area
# ══════════════════════════════════════════════════════════════════════
if st.session_state.session_id is None:
    st.markdown("""
    <div style="text-align:center; margin-top:100px;">
        <div style="font-size:3em;">🤖</div>
        <h2 style="color:#e8eaf0;">Welcome to RAG Chatbot</h2>
        <p style="color:#6b7280; font-size:1.05em;">
            Create a session in the sidebar &#8594;<br>
            Upload documents, then start chatting.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

session = current_session()
has_index = st.session_state.vector_store is not None

# Header
status_cls = "status-ready" if has_index else "status-noindex"
status_icon = "●" if has_index else "●"
st.markdown(
    f'<div class="session-header">'
    f'<span class="{status_cls}">{status_icon}</span> '
    f'{session["bot_name"]}'
    f'<span class="session-sub"> — {session["name"]}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

if not has_index:
    st.info("📂 No documents indexed yet. Upload files or paste text in the sidebar to get started.")

st.divider()

# ── Chat history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    pct = int(src["score"] * 100)
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <b>{src["file"]}</b>, page {src["page"]}'
                        f'<span class="score-badge">{pct}%</span><br>'
                        f'<span class="snippet-text">{src["snippet"]}…</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ── Input ───────────────────────────────────────────────────────────
if has_index:
    question = st.chat_input(placeholder="Ask a question about your documents…")
else:
    question = None

if question:
    with st.chat_message("user"):
        st.markdown(question)

    db.save_message(st.session_state.session_id, "user", question)
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    with st.chat_message("assistant"):
        session = current_session()
        response_area = st.empty()
        answer = ""
        sources: list = []

        for chunk, src in ask_stream(
            question=question,
            vector_store=st.session_state.vector_store,
            session=session,
            history=st.session_state.messages[:-1],
            threshold=st.session_state.relevance_threshold,
        ):
            if src is not None:
                sources = src
            else:
                answer += chunk
                response_area.markdown(answer + "▌")

        response_area.markdown(answer)

        if sources:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for src in sources:
                    pct = int(src["score"] * 100)
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <b>{src["file"]}</b>, page {src["page"]}'
                        f'<span class="score-badge">{pct}%</span><br>'
                        f'<span class="snippet-text">{src["snippet"]}…</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    db.save_message(st.session_state.session_id, "assistant", answer, sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
