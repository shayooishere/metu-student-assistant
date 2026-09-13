# METU Student Assistant

A local **RAG chatbot** that answers METU Ankara student questions using official university web pages. Answers are grounded in retrieved documents and include source links.

Built as an internship project.

## Features

- Scrapes curated METU pages (ÖİDB, library, dorms, ISO, catalog, campus info, and more)
- Chunks and embeds content with **BAAI/bge-m3** into a local **Chroma** vector DB
- **Hybrid retrieval**: dense search + **BM25**, fused with **RRF**
- Follow-up handling via optional query rewrite + short chat history
- Answers with **Groq** (`openai/gpt-oss-20b`)
- **Streamlit** chat UI (English & Turkish), New chat, Export `.txt`, source links

> Always verify important information on official METU websites.

## Architecture

```text
urls.txt
  → scrape (Trafilatura)
  → chunk
  → embed (BGE-M3) + BM25 index
  → hybrid retrieve (Chroma + BM25 → RRF)
  → Groq LLM
  → Streamlit UI
```

## Tech stack

| Layer | Choice |
|--------|--------|
| UI | Streamlit |
| LLM | Groq — `openai/gpt-oss-20b` |
| Embeddings | `BAAI/bge-m3` |
| Vector DB | Chroma |
| Keyword search | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Scraper | Requests + Trafilatura |

## Repository layout

```text
app.py                 # Streamlit chat UI
requirements.txt
scraper/
  urls.txt             # language|section|url list
  scrape.py            # scrape → data/processed/pages.jsonl
rag/
  chunk.py             # pages → chunks.jsonl
  embed.py             # chunks → chroma_db + BM25
  hybrid.py            # dense + BM25 + RRF
  engine.py            # RAG ask() pipeline
.streamlit/            # theme / config
```

`data/processed/` and `chroma_db/` are gitignored — build them locally (steps below).

## Setup

### 1. Clone

```bash
git clone https://github.com/shayooishere/metu-student-assistant.git
cd metu-student-assistant
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a key from [https://console.groq.com](https://console.groq.com).  
Never commit `.env`.

### 4. Build the knowledge base

From the project root (with the venv active):

```bash
# 1) Scrape pages listed in scraper/urls.txt
python -m scraper.scrape

# 2) Chunk pages
python -m rag.chunk

# 3) Embed into Chroma + build BM25 index
python -m rag.embed
```

Notes:

- Scraping uses a polite delay between requests; a full run can take a while.
- First embedding run downloads `BAAI/bge-m3` and may use several GB of RAM/disk.
- Output:
  - `data/processed/pages.jsonl`
  - `data/processed/chunks.jsonl`
  - `chroma_db/` (vectors + `bm25_index.pkl`)

### 5. Run the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Usage tips

- Ask concrete student questions (registration, library, dorms, fees, ISO, courses, etc.).
- Use **New chat** to clear history (resets follow-up rewrite context).
- Use **Export .txt** to download the conversation with sources.
- For multi-part questions (e.g. add-drop **and** summer school limits), asking separately often retrieves better.

## Configuration (defaults)

In `rag/engine.py` / `rag/hybrid.py`:

- `TOP_K` — chunks sent to the LLM
- `CANDIDATE_K` — candidates from dense and BM25 before RRF
- `MAX_SOURCES` — unique URLs shown in the UI
- `GROQ_MODEL` — `openai/gpt-oss-20b`

Tune these after measuring retrieval quality on your own question set.

## Limitations

- Answers only from the scraped corpus — missing pages mean missing answers
- Not an official METU system (no SSO, SLA, or campus-scale hosting)
- Table-heavy pages (e.g. academic calendar) and multi-intent questions are harder for retrieval
- Free LLM / hosting quotas may limit latency and availability

## Disclaimer

This project is for educational / internship demonstration purposes. It may be incomplete or outdated. Confirm procedures, dates, and requirements on official METU pages before acting on any answer.

## License

No license file is included yet. Add one if you want others to reuse the code under clear terms.
