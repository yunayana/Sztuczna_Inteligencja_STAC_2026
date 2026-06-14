"""
ingest.py — wczytuje dokumenty z folderu data/, dzieli na chunki,
generuje embeddingi przez Ollama i zapisuje do ChromaDB.

Użycie:
    python ingest.py
"""

import os
import re
import chromadb
import ollama
from pypdf import PdfReader
from docx import Document

# ── Konfiguracja ──────────────────────────────────────────────────────────────
DATA_DIR       = "data"          # folder z dokumentami
CHROMA_DIR     = "chroma_db"     # folder gdzie zapisze się baza wektorowa
COLLECTION     = "ans_docs"      # nazwa kolekcji w ChromaDB
EMBED_MODEL    = "nomic-embed-text"  # model do embeddingów (ollama pull nomic-embed-text)
CHUNK_SIZE     = 500             # znaki na chunk
CHUNK_OVERLAP  = 50              # nakładanie się chunków


# ── Odczyt plików ─────────────────────────────────────────────────────────────
def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_documents(data_dir: str) -> list[dict]:
    """Wczytuje wszystkie obsługiwane pliki z folderu data/."""
    docs = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        ext = fname.lower().split(".")[-1]
        if ext == "pdf":
            text = read_pdf(fpath)
        elif ext == "docx":
            text = read_docx(fpath)
        elif ext in ("txt", "md"):
            text = read_txt(fpath)
        else:
            print(f"  [pominięto] {fname} — nieobsługiwany format")
            continue
        if text.strip():
            docs.append({"source": fname, "text": text})
            print(f"  [ok] {fname} ({len(text)} znaków)")
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────
def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Prosty chunking po znakach z nakładaniem się."""
    # Usuń nadmiarowe białe znaki
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Cofnij do ostatniej spacji, żeby nie uciąć w połowie słowa
        if end < len(text) and " " in chunk:
            end = text.rfind(" ", start, end) + 1
            chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 20]  # odrzuć bardzo krótkie


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed(texts: list[str]) -> list[list[float]]:
    """Generuje embeddingi przez lokalny Ollama."""
    vectors = []
    for text in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        vectors.append(resp["embedding"])
    return vectors


# ── Zapis do ChromaDB ─────────────────────────────────────────────────────────
def build_index(docs: list[dict]):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Jeśli kolekcja istnieje — usuń ją (przebuduj od zera)
    try:
        client.delete_collection(COLLECTION)
        print("Usunięto starą kolekcję.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION)

    total_chunks = 0
    for doc in docs:
        chunks = split_text(doc["text"])
        print(f"  → {doc['source']}: {len(chunks)} chunków, generuję embeddingi...")
        vectors = embed(chunks)

        ids       = [f"{doc['source']}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc["source"], "chunk_id": i} for i in range(len(chunks))]

        collection.add(documents=chunks, embeddings=vectors, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)

    print(f"\nGotowe! Zindeksowano {total_chunks} chunków z {len(docs)} dokumentów.")
    print(f"Baza zapisana w: {CHROMA_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Stworzono folder '{DATA_DIR}/' — wrzuć tam swoje pliki PDF/DOCX/TXT i uruchom ponownie.")
    else:
        print(f"Wczytuję dokumenty z '{DATA_DIR}/'...")
        docs = load_documents(DATA_DIR)
        if not docs:
            print("Brak dokumentów do zindeksowania. Wrzuć pliki do folderu data/")
        else:
            print(f"\nBudowanie indeksu dla {len(docs)} dokumentów...")
            build_index(docs)
