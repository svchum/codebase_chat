
## Demo

[![CodebaseChat Demo](https://img.youtube.com/vi/zqC93AGlNYY/maxresdefault.jpg)](https://youtu.be/zqC93AGlNYY)

# CodebaseChat

Talk to any GitHub repo. Ask questions and get answers with highlighted code, file names, line numbers, and GitHub links. Built with Streamlit, Tree-sitter, FAISS, SQLite (FTS5), and a swappable LLM (Gemini Flash by default).


## How it works

Repos are ingested (code, docs, HTML, PDFs, and git commits) and split into chunks with Tree-sitter, then stored in a FAISS vector index and a SQLite full-text index, one per project. When you ask a question, the app runs a vector search, a BM25 keyword search, and a symbol-name lookup, merges the results with Reciprocal Rank Fusion, and expands them with surrounding context. It then builds a prompt with citations, sends it to the LLM, and streams back a typewriter-style answer along with its sources.


## Design notes

- **Hybrid retrieval.** Vector search finds code by meaning, BM25 finds exact terms like error messages and variable names, and symbol-name lookup handles questions that mention a specific function or class. Their scores aren't comparable, so Reciprocal Rank Fusion merges them using only rank, which avoids tuning weights.
- **Tree-sitter chunking.** Splitting by function and class keeps each chunk a complete unit instead of cutting code at arbitrary line counts. Files without a grammar fall back to window chunking, so nothing is silently skipped.
- **Contextual expansion.** After retrieval, results are expanded with surrounding code so the LLM sees a match in context, not an isolated snippet.
- **One index per project.** Each repo gets its own FAISS and SQLite index, so switching projects is a clean swap, and deleting a repo means deleting one index. Incremental sync means re-indexing only reprocesses what changed.
- **Last-3-projects registry.** Capping kept repos keeps disk and memory use predictable on a free host, and it means the demo always shows a small, curated set.
- **Local embeddings.** Embeddings run locally with sentence-transformers, so indexing needs no API key or per-call cost. Only answer generation calls an external LLM.
- **Swappable LLM layer.** Gemini, OpenAI, and Anthropic go through LangChain behind one interface, so changing providers is a config change, not a code change.
- **Citations by construction.** The prompt asks the model to cite the retrieved chunks, and citations are parsed back out so answers link to real files and line numbers.
- **Demo-mode safeguards.** Visitors can't index anything, and per-session, daily, length, and cooldown limits keep a public demo from running up costs or being abused.


## Setup

The commands below are for Windows. On macOS or Linux, use `source .venv/bin/activate` and `cp` instead.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
pytest
```

After copying `.env.example`, open `.env` and set `GOOGLE_API_KEY`. Running `pytest` is optional; it checks the core logic and needs no downloads.

## Usage

First, index a repo (a GitHub URL or a local folder). Run this for up to three repos, and again whenever one changes:

```powershell
python scripts/reindex.py --repo https://github.com/owner/name
```

Then start the app:

```powershell
streamlit run app/streamlit_app.py
```

The first index downloads a small embedding model (about 130 MB). The sidebar's project list switches between the kept repos, and switching starts a fresh chat.

Optional personal mode: set `DEMO_MODE=false` in `.env` and restart to get the repo box, indexing buttons, filters, and retrieval settings in the sidebar. Indexing from there works the same way and keeps the last 3 repos.


## Project layout

- `app/streamlit_app.py`: entry point.
- `app/ui/`: helpers, sidebar, home (cover page), chat, sources, and limits.
- `app/ui/styles/`: `base.css`, `cover.css`, `chat.css`, and `widgets.css`.
- `rag/ingest.py`: repo clone/update, file walking (.gitignore aware), PDF text, and git commits.
- `rag/parse.py`: Tree-sitter chunking plus fallbacks.
- `rag/embed.py`: local embeddings (sentence-transformers).
- `rag/index.py`: FAISS and SQLite/FTS5 for one project, incremental sync, and filters.
- `rag/projects.py`: the last-3-projects registry.
- `rag/usage.py`: shared daily question counter.
- `rag/retrieve.py`: vector, keyword, and name retrieval, RRF, and contextual expansion.
- `rag/prompt.py`: query construction and citation parsing.
- `rag/llm.py`: Gemini, OpenAI, or Anthropic via LangChain, with retries.
- `rag/pipeline.py`: orchestration (the `Engine`).
- `scripts/reindex.py`: CLI indexer (also used by CI).
- `.streamlit/config.toml`: base dark theme.
- `tests/`: unit tests using a fake embedder.

## Tech stack

- **Language:** Python
- **UI:** Streamlit, custom CSS
- **Retrieval:** Tree-sitter, FAISS, SQLite FTS5, sentence-transformers
- **LLM:** Gemini Flash (default), OpenAI, Anthropic, via LangChain
- **Tooling:** pytest, GitHub Actions
