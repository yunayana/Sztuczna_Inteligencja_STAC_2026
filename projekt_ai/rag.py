"""
rag.py — retrieval + generacja odpowiedzi przez Ollama.

Użycie (z wiersza poleceń):
    python rag.py "Jakie są zasady zaliczenia praktyk?"

Importowane też przez app.py (Streamlit).
"""

import sys
import chromadb
import ollama

# ── Konfiguracja ──────────────────────────────────────────────────────────────
CHROMA_DIR  = "chroma_db"
COLLECTION  = "ans_docs"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.2:3b"   # zmień na phi3:mini jeśli wolisz
TOP_K       = 4               # ile chunków pobieramy z bazy


# ── Prompt systemowy ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Jesteś pomocnym asystentem akademickim dla studentów ANS (Akademia Nauk Stosowanych).
Odpowiadaj wyłącznie na podstawie podanego kontekstu z dokumentów.
Jeśli kontekst nie zawiera odpowiedzi, napisz wprost: "Nie znalazłem tej informacji w dostępnych dokumentach."
Odpowiadaj po polsku, zwięźle i precyzyjnie."""


# ── Inicjalizacja ChromaDB ────────────────────────────────────────────────────
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION)


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Pobiera top_k najbardziej podobnych chunków do zapytania."""
    collection = get_collection()

    # Embedding zapytania
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    query_vector = resp["embedding"]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "source":   meta.get("source", "?"),
            "chunk_id": meta.get("chunk_id", 0),
            "distance": round(dist, 4),
        })
    return chunks


# ── Generacja odpowiedzi ──────────────────────────────────────────────────────
def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(f"[{i}] (źródło: {c['source']})\n{c['text']}")
    context = "\n\n".join(context_parts)

    return f"""Kontekst z dokumentów:
{context}

Pytanie: {query}
Odpowiedź:"""


def ask(query: str, history: list[dict] | None = None) -> dict:
    """
    Wysyła zapytanie do RAG i zwraca słownik:
      {
        "answer": str,
        "sources": list[dict],   # chunki użyte do odpowiedzi
      }
    """
    chunks = retrieve(query)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Dodaj historię rozmowy (opcjonalna)
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": build_prompt(query, chunks)})

    response = ollama.chat(model=LLM_MODEL, messages=messages)
    answer = response["message"]["content"]

    return {"answer": answer, "sources": chunks}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Co zawierają dostępne dokumenty?"
    print(f"\n❓ Pytanie: {query}\n")

    result = ask(query)

    print("💬 Odpowiedź:")
    print(result["answer"])

    print("\n📄 Źródła:")
    for c in result["sources"]:
        print(f"  • {c['source']} (chunk {c['chunk_id']}, dist={c['distance']})")
