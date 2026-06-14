"""
app.py — interfejs Streamlit dla systemu RAG.

Uruchomienie:
    streamlit run app.py
"""

import streamlit as st
import chromadb
from rag import ask, CHROMA_DIR, COLLECTION, LLM_MODEL, EMBED_MODEL

# ── Konfiguracja strony ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG — Dokumenty ANS",
    page_icon=" ",
    layout="centered",
)

# ── CSS — ciemny gradientowy motyw z glassmorphism ───────────────────────────
st.markdown("""
<style>
/* Tło strony */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    min-height: 100vh;
}

/* Ukryj domyślny header Streamlit */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Tytuł główny */
h1 {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin-bottom: 0 !important;
}

/* Caption pod tytułem */
.stApp p[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}

/* Karty statystyk (glassmorphism) */
.stat-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(167, 139, 250, 0.2);
    border-radius: 16px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.3s;
}
.stat-card:hover {
    border-color: rgba(167, 139, 250, 0.5);
}
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Divider między statystykami a chatem */
.section-divider {
    border: none;
    border-top: 1px solid rgba(167, 139, 250, 0.15);
    margin: 24px 0 16px 0;
}

/* Wiadomości czatu */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(167, 139, 250, 0.12) !important;
    border-radius: 14px !important;
    margin-bottom: 10px !important;
    padding: 12px !important;
}

/* Input czatu */
[data-testid="stChatInput"] textarea {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(167, 139, 250, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.15) !important;
}

/* Tekst ogólny */
.stMarkdown, p, li {
    color: #e2e8f0 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.8) !important;
    border-right: 1px solid rgba(167, 139, 250, 0.15) !important;
}

/* Przycisk */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Success / warning banery */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
    background: rgba(255,255,255,0.04) !important;
}

/* Expander (źródła) */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(167, 139, 250, 0.12) !important;
    border-radius: 10px !important;
}

/* Code block */
code {
    background: rgba(167, 139, 250, 0.15) !important;
    color: #c4b5fd !important;
    border-radius: 6px !important;
    padding: 2px 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Sprawdzenie czy baza istnieje ─────────────────────────────────────────────
@st.cache_resource
def get_db_info():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        col = client.get_collection(COLLECTION)
        count = col.count()
        # Policz unikalne dokumenty źródłowe
        results = col.get(include=["metadatas"])
        sources = set(m.get("source", "?") for m in results["metadatas"])
        return count, len(sources)
    except Exception:
        return 0, 0

chunk_count, doc_count = get_db_info()

# ── Nagłówek ─────────────────────────────────────────────────────────────────
st.title("Asystent dokumentów ANS")
st.caption(f"Model: `{LLM_MODEL}` · Embeddingi: `{EMBED_MODEL}`")

# ── Panel statystyk ───────────────────────────────────────────────────────────
if chunk_count == 0:
    st.warning(
        "⚠️ Baza wektorowa jest pusta lub nie istnieje.\n\n"
        "Uruchom najpierw: `python ingest.py`"
    )
    st.stop()

# Liczba zapytań w sesji
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_history" not in st.session_state:
    st.session_state.sources_history = []

query_count = sum(1 for m in st.session_state.messages if m["role"] == "user")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{doc_count}</div>
        <div class="stat-label">Dokumenty</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{chunk_count}</div>
        <div class="stat-label">Fragmenty (chunki)</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{query_count}</div>
        <div class="stat-label">Zapytania w sesji</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Historia rozmowy ──────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i // 2 < len(st.session_state.sources_history):
            sources = st.session_state.sources_history[i // 2]
            with st.expander("📄 Źródła", expanded=False):
                for c in sources:
                    sim = round((1 - c["distance"]) * 100, 1)
                    st.markdown(f"**{c['source']}** — podobieństwo: `{sim}%`")
                    st.caption(c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"])

# ── Input użytkownika ─────────────────────────────────────────────────────────
if prompt := st.chat_input("Zadaj pytanie o dokumenty..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Szukam w dokumentach i generuję odpowiedź..."):
            history = st.session_state.messages[:-1]
            result = ask(prompt, history if history else None)

        st.markdown(result["answer"])

        with st.expander("📄 Źródła", expanded=False):
            for c in result["sources"]:
                sim = round((1 - c["distance"]) * 100, 1)
                st.markdown(f"**{c['source']}** — podobieństwo: `{sim}%`")
                st.caption(c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.sources_history.append(result["sources"])
    st.rerun()  # odśwież licznik zapytań

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Informacje")
    st.markdown(f"""
    - **Model LLM:** `{LLM_MODEL}`
    - **Embeddingi:** `{EMBED_MODEL}`
    - **Baza:** ChromaDB (lokalna)
    """)

    st.divider()

    if st.button("🗑️ Wyczyść historię"):
        st.session_state.messages = []
        st.session_state.sources_history = []
        st.rerun()

    st.divider()
    st.markdown("**Dodaj dokumenty:**")
    st.code("python ingest.py", language="bash")
    st.caption("Wrzuć PDF/DOCX/TXT do `data/` i uruchom powyższe.")
