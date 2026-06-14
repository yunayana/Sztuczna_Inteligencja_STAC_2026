# Sztuczna_Inteligencja_STAC_2026
Trotsenko Yana 21232

# Projekt 

# RAG dla dokumentów ANS

Lokalny system pytań i odpowiedzi oparty na własnych dokumentach z zajęć.

## Stos technologiczny

| Komponent | Technologia |
|---|---|
| LLM | Ollama + `llama3.2:3b` |
| Embeddingi | Ollama + `nomic-embed-text` |
| Baza wektorowa | ChromaDB (lokalna, plikowa) |
| Interfejs | Streamlit |

## Instalacja

### 1. Ollama

Pobierz ze strony [ollama.com](https://ollama.com) i zainstaluj, następnie:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 2. Zależności Pythona

```bash
pip install -r requirements.txt
```

## Użycie

### Krok 1 — dodaj dokumenty

Wrzuć pliki PDF, DOCX lub TXT do folderu `data/`.

### Krok 2 — zbuduj indeks

```bash
python ingest.py
```

### Krok 3 — uruchom interfejs

```bash
streamlit run app.py
```

Otwórz przeglądarkę pod adresem `http://localhost:8501`.

### Tryb CLI (opcjonalnie)

```bash
python rag.py "Jakie są zasady zaliczenia praktyk?"
```

## Struktura projektu

```
rag_project/
├── data/          # Twoje dokumenty (PDF/DOCX/TXT)
├── chroma_db/     # Baza wektorowa (generowana automatycznie)
├── ingest.py      # Indeksowanie dokumentów
├── rag.py         # Retrieval + generacja
├── app.py         # Interfejs Streamlit
└── requirements.txt
```

## Parametry konfiguracyjne

W `ingest.py` i `rag.py` możesz zmienić:

- `CHUNK_SIZE` — rozmiar chunka (domyślnie 500 znaków)
- `CHUNK_OVERLAP` — nakładanie się chunków (50 znaków)
- `LLM_MODEL` — model językowy (`llama3.2:3b` lub `phi3:mini`)
- `TOP_K` — liczba chunków pobieranych przy wyszukiwaniu (4)
